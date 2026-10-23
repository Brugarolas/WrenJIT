//! Small, runtime-neutral event loop core for WrenJIT.
//!
//! The VM owns one of these loops today. Later the N:M scheduler can keep the
//! same API while moving timer wakeups to a shared reactor/timer service.
//!
//! Design rules:
//! - microtasks are FIFO;
//! - timeouts are one-shot;
//! - intervals never overlap: the next tick is armed only after the previous
//!   callback completes;
//! - overdue intervals skip missed ticks instead of running catch-up bursts;
//! - cancellation is lazy for queue/heap storage but eager in the active table;
//! - no polling loop is required: `time_until_next()` tells the scheduler how
//!   long it may park the OS thread.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Microtask,
    Timeout,
    Interval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// The event was pending and is now cancelled.
    Cancelled,
    /// An interval callback is already running. Future ticks are cancelled, but
    /// the in-flight callback is allowed to finish cooperatively.
    InFlightInterval,
    /// The handle was stale, already fired, already cancelled or already done.
    NotFound,
}

#[derive(Clone, Debug)]
pub struct FiredEvent<T> {
    pub id: EventId,
    pub kind: EventKind,
    pub payload: T,
}

#[derive(Clone, Debug)]
struct ActiveEvent<T> {
    kind: EventKind,
    payload: T,
    deadline: Option<Instant>,
    period: Option<Duration>,
    generation: u64,
    in_flight: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeapEntry {
    deadline: Instant,
    id: EventId,
    generation: u64,
}

// BinaryHeap is a max heap. Reverse only the deadline/generation ordering so
// the earliest valid deadline is at the top.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.generation.cmp(&self.generation))
            .then_with(|| other.id.0.cmp(&self.id.0))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct WrenEventLoop<T> {
    next_id: u64,
    microtasks: VecDeque<EventId>,
    timers: BinaryHeap<HeapEntry>,
    active: HashMap<EventId, ActiveEvent<T>>,
}

impl<T> Default for WrenEventLoop<T> {
    fn default() -> Self {
        Self {
            next_id: 1,
            microtasks: VecDeque::new(),
            timers: BinaryHeap::new(),
            active: HashMap::new(),
        }
    }
}

impl<T: Clone> WrenEventLoop<T> {
    #[inline]
    fn fresh_id(&mut self) -> EventId {
        loop {
            let id = EventId(self.next_id);
            self.next_id = self.next_id.wrapping_add(1).max(1);
            if !self.active.contains_key(&id) {
                return id;
            }
        }
    }

    pub fn enqueue_microtask(&mut self, payload: T) -> EventId {
        let id = self.fresh_id();
        self.active.insert(
            id,
            ActiveEvent {
                kind: EventKind::Microtask,
                payload,
                deadline: None,
                period: None,
                generation: 0,
                in_flight: false,
            },
        );
        self.microtasks.push_back(id);
        id
    }

    pub fn set_timeout(&mut self, delay: Duration, payload: T) -> Option<EventId> {
        let now = Instant::now();
        let deadline = now.checked_add(delay)?;
        let id = self.fresh_id();
        self.active.insert(
            id,
            ActiveEvent {
                kind: EventKind::Timeout,
                payload,
                deadline: Some(deadline),
                period: None,
                generation: 0,
                in_flight: false,
            },
        );
        self.timers.push(HeapEntry { deadline, id, generation: 0 });
        Some(id)
    }

    pub fn set_interval(&mut self, period: Duration, payload: T) -> Option<EventId> {
        if period.is_zero() {
            return None;
        }
        let now = Instant::now();
        let deadline = now.checked_add(period)?;
        let id = self.fresh_id();
        self.active.insert(
            id,
            ActiveEvent {
                kind: EventKind::Interval,
                payload,
                deadline: Some(deadline),
                period: Some(period),
                generation: 0,
                in_flight: false,
            },
        );
        self.timers.push(HeapEntry { deadline, id, generation: 0 });
        Some(id)
    }

    pub fn cancel(&mut self, id: EventId) -> CancelOutcome {
        let Some(event) = self.active.remove(&id) else {
            return CancelOutcome::NotFound;
        };
        if event.kind == EventKind::Interval && event.in_flight {
            CancelOutcome::InFlightInterval
        } else {
            CancelOutcome::Cancelled
        }
    }

    /// Re-arm an interval after its callback Promise has fulfilled. Returns false
    /// when cancellation removed the interval while the callback was in flight.
    pub fn complete_interval(&mut self, id: EventId) -> bool {
        let now = Instant::now();
        let Some(event) = self.active.get_mut(&id) else {
            return false;
        };
        if event.kind != EventKind::Interval || !event.in_flight {
            return false;
        }

        let period = event.period.expect("interval has no period");
        let previous = event.deadline.unwrap_or(now);
        let next = next_interval_deadline(previous, period, now);
        event.deadline = Some(next);
        event.in_flight = false;
        event.generation = event.generation.wrapping_add(1);
        self.timers.push(HeapEntry {
            deadline: next,
            id,
            generation: event.generation,
        });
        true
    }

    pub fn pop_microtask(&mut self) -> Option<FiredEvent<T>> {
        while let Some(id) = self.microtasks.pop_front() {
            let Some(event) = self.active.remove(&id) else {
                continue; // lazily cancelled/stale
            };
            if event.kind != EventKind::Microtask {
                continue;
            }
            return Some(FiredEvent {
                id,
                kind: EventKind::Microtask,
                payload: event.payload,
            });
        }
        None
    }

    pub fn pop_due_timer(&mut self) -> Option<FiredEvent<T>> {
        let now = Instant::now();
        loop {
            let top = *self.timers.peek()?;
            if top.deadline > now {
                return None;
            }
            self.timers.pop();

            let Some(event) = self.active.get_mut(&top.id) else {
                continue;
            };
            if event.generation != top.generation || event.deadline != Some(top.deadline) {
                continue;
            }

            match event.kind {
                EventKind::Microtask => continue,
                EventKind::Timeout => {
                    let event = self.active.remove(&top.id).expect("active timeout disappeared");
                    return Some(FiredEvent {
                        id: top.id,
                        kind: EventKind::Timeout,
                        payload: event.payload,
                    });
                }
                EventKind::Interval => {
                    if event.in_flight {
                        continue;
                    }
                    event.in_flight = true;
                    return Some(FiredEvent {
                        id: top.id,
                        kind: EventKind::Interval,
                        payload: event.payload.clone(),
                    });
                }
            }
        }
    }

    /// Duration until the earliest valid timer. Stale heap entries are discarded.
    pub fn time_until_next(&mut self) -> Option<Duration> {
        let now = Instant::now();
        loop {
            let top = *self.timers.peek()?;
            let valid = self.active.get(&top.id).is_some_and(|event| {
                event.generation == top.generation
                    && event.deadline == Some(top.deadline)
                    && !event.in_flight
                    && event.kind != EventKind::Microtask
            });
            if !valid {
                self.timers.pop();
                continue;
            }
            return Some(top.deadline.saturating_duration_since(now));
        }
    }

    #[inline]
    pub fn has_live_events(&self) -> bool {
        !self.active.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    pub fn for_each_payload(&self, mut f: impl FnMut(&T)) {
        for event in self.active.values() {
            f(&event.payload);
        }
    }
}

fn next_interval_deadline(previous: Instant, period: Duration, now: Instant) -> Instant {
    let first = previous.checked_add(period).unwrap_or(now);
    if first > now {
        return first;
    }

    // Preserve fixed-rate scheduling while skipping missed ticks. Duration only
    // offers u32 multiplication, so extremely large gaps fall back to now+period.
    let lag = now.saturating_duration_since(first);
    let period_ns = period.as_nanos();
    if period_ns == 0 {
        return now;
    }
    let skipped = lag.as_nanos() / period_ns + 1;
    if skipped <= u32::MAX as u128 {
        if let Some(delta) = period.checked_mul(skipped as u32) {
            if let Some(deadline) = first.checked_add(delta) {
                return deadline;
            }
        }
    }
    now.checked_add(period).unwrap_or(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_microtask_is_skipped() {
        let mut loop_ = WrenEventLoop::default();
        let id = loop_.enqueue_microtask(7u32);
        assert_eq!(loop_.cancel(id), CancelOutcome::Cancelled);
        assert!(loop_.pop_microtask().is_none());
    }

    #[test]
    fn zero_timeout_fires() {
        let mut loop_ = WrenEventLoop::default();
        let id = loop_.set_timeout(Duration::ZERO, 11u32).unwrap();
        let fired = loop_.pop_due_timer().expect("timeout should be due");
        assert_eq!(fired.id, id);
        assert_eq!(fired.payload, 11);
        assert!(!loop_.has_live_events());
    }

    #[test]
    fn interval_does_not_rearm_until_completed() {
        let mut loop_ = WrenEventLoop::default();
        let id = loop_.set_interval(Duration::from_nanos(1), 3u32).unwrap();
        std::thread::sleep(Duration::from_micros(10));
        let fired = loop_.pop_due_timer().expect("interval should fire");
        assert_eq!(fired.id, id);
        assert!(loop_.pop_due_timer().is_none());
        assert!(loop_.complete_interval(id));
    }

    #[test]
    fn cancelling_in_flight_interval_is_distinct() {
        let mut loop_ = WrenEventLoop::default();
        let id = loop_.set_interval(Duration::from_nanos(1), 1u32).unwrap();
        std::thread::sleep(Duration::from_micros(10));
        let _ = loop_.pop_due_timer().unwrap();
        assert_eq!(loop_.cancel(id), CancelOutcome::InFlightInterval);
        assert!(!loop_.complete_interval(id));
    }
}
