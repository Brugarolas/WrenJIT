//! Microtask + timer event loop.
//!
//! Compared with the original NanoEventLoop this version:
//! - uses monotonic `Instant` rather than truncating epoch milliseconds to `int`;
//! - has a real microtask queue (`next_tick`);
//! - stores timers in a min-heap instead of linearly scanning vectors;
//! - sleeps on a `Condvar` until work/deadline instead of polling every 1 ms;
//! - separates resolution from cancellation;
//! - is thread-safe, so other scheduler workers may enqueue work and wake it.

use crate::function::Function;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const VOW_PENDING: u8 = 0;
const VOW_RESOLVED: u8 = 1;
const VOW_CANCELLED: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VowState {
    Pending,
    Resolved,
    Cancelled,
}

struct VowInner {
    state: AtomicU8,
    callbacks: Mutex<Vec<Function<(), ()>>>,
}

/// Completion/cancellation token returned by the event loop.
///
/// A one-shot timer or microtask resolves when it runs. Cancelling leaves it in
/// `Cancelled`, and `.then(...)` callbacks are not invoked for cancellation.
#[derive(Clone)]
pub struct Vow {
    inner: Arc<VowInner>,
}

impl fmt::Debug for Vow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vow").field("state", &self.state()).finish()
    }
}

impl Default for Vow {
    fn default() -> Self {
        Self::new()
    }
}

impl Vow {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(VowInner {
                state: AtomicU8::new(VOW_PENDING),
                callbacks: Mutex::new(Vec::new()),
            }),
        }
    }

    #[inline(always)]
    pub fn state(&self) -> VowState {
        match self.inner.state.load(Ordering::Acquire) {
            VOW_PENDING => VowState::Pending,
            VOW_RESOLVED => VowState::Resolved,
            VOW_CANCELLED => VowState::Cancelled,
            _ => unreachable!("invalid Vow state"),
        }
    }

    #[inline(always)]
    pub fn is_pending(&self) -> bool {
        self.state() == VowState::Pending
    }

    #[inline(always)]
    pub fn is_resolved(&self) -> bool {
        self.state() == VowState::Resolved
    }

    #[inline(always)]
    pub fn is_cancelled(&self) -> bool {
        self.state() == VowState::Cancelled
    }

    /// Register a callback to execute when this Vow resolves.
    /// If it has already resolved, the callback runs immediately.
    pub fn then<F>(&self, callback: F)
    where
        F: FnOnce() + Send + Unpin + 'static,
    {
        if self.is_resolved() {
            callback();
            return;
        }
        if self.is_cancelled() {
            return;
        }

        let mut callback = Some(callback);
        let mut callbacks = self.inner.callbacks.lock().unwrap_or_else(|e| e.into_inner());

        // Close the race with resolve()/cancel() between the optimistic check
        // above and acquiring the callback list.
        match self.state() {
            VowState::Pending => {
                let mut cb = callback;
                callbacks.push(Function::new(move || {
                    if let Some(f) = cb.take() {
                        f();
                    }
                }));
            }
            VowState::Resolved => {
                drop(callbacks);
                if let Some(f) = callback.take() {
                    f();
                }
            }
            VowState::Cancelled => {}
        }
    }

    /// Resolve exactly once. Returns true for the thread that won settlement.
    pub fn resolve(&self) -> bool {
        if self
            .inner
            .state
            .compare_exchange(
                VOW_PENDING,
                VOW_RESOLVED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }

        let callbacks = {
            let mut locked = self.inner.callbacks.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *locked)
        };

        for mut callback in callbacks {
            callback.call0();
        }
        true
    }

    /// Cancel exactly once. Resolution callbacks are discarded, not run.
    pub fn cancel(&self) -> bool {
        if self
            .inner
            .state
            .compare_exchange(
                VOW_PENDING,
                VOW_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }

        self.inner
            .callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventLoopError {
    ZeroInterval,
    AlreadyRunning,
}

impl fmt::Display for EventLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInterval => f.write_str("interval must be greater than zero"),
            Self::AlreadyRunning => f.write_str("event loop is already running"),
        }
    }
}

impl std::error::Error for EventLoopError {}

struct MicroTask {
    callback: Function<(), ()>,
    vow: Vow,
}

struct TimerTask {
    id: u64,
    deadline: Instant,
    interval: Option<Duration>,
    callback: Function<(), ()>,
    vow: Vow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TimerKey {
    deadline: Instant,
    sequence: u64,
    id: u64,
}

struct State {
    running: bool,
    stop_requested: bool,
    next_id: u64,
    next_sequence: u64,
    ticks: u64,
    microtasks: VecDeque<MicroTask>,
    timer_heap: BinaryHeap<Reverse<TimerKey>>,
    timers: HashMap<u64, TimerTask>,
}

struct Inner {
    started: Instant,
    state: Mutex<State>,
    wake: Condvar,
}

/// Thread-safe single-consumer event loop.
///
/// Any thread may schedule/cancel work. `run()` itself is single-consumer and
/// executes callbacks on the thread that called it.
#[derive(Clone)]
pub struct EventLoop {
    inner: Arc<Inner>,
}

impl Default for EventLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLoop {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                started: Instant::now(),
                state: Mutex::new(State {
                    running: false,
                    stop_requested: false,
                    next_id: 1,
                    next_sequence: 1,
                    ticks: 0,
                    microtasks: VecDeque::new(),
                    timer_heap: BinaryHeap::new(),
                    timers: HashMap::new(),
                }),
                wake: Condvar::new(),
            }),
        }
    }

    #[inline]
    fn lock_state(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Monotonic time since this loop was created.
    pub fn elapsed(&self) -> Duration {
        self.inner.started.elapsed()
    }

    pub fn ticks(&self) -> u64 {
        self.lock_state().ticks
    }

    pub fn is_running(&self) -> bool {
        self.lock_state().running
    }

    /// Queue a true microtask. It runs before the next timer selection.
    pub fn next_tick<F>(&self, callback: F) -> Vow
    where
        F: FnOnce() + Send + Unpin + 'static,
    {
        let vow = Vow::new();
        let mut callback = Some(callback);
        let task = MicroTask {
            callback: Function::new(move || {
                if let Some(f) = callback.take() {
                    f();
                }
            }),
            vow: vow.clone(),
        };

        self.lock_state().microtasks.push_back(task);
        self.inner.wake.notify_one();
        vow
    }

    pub fn set_timeout<F>(&self, delay: Duration, callback: F) -> Vow
    where
        F: FnOnce() + Send + Unpin + 'static,
    {
        let mut callback = Some(callback);
        self.insert_timer(
            delay,
            None,
            Function::new(move || {
                if let Some(f) = callback.take() {
                    f();
                }
            }),
        )
    }

    pub fn set_timeout_ms<F>(&self, delay_ms: u64, callback: F) -> Vow
    where
        F: FnOnce() + Send + Unpin + 'static,
    {
        self.set_timeout(Duration::from_millis(delay_ms), callback)
    }

    pub fn set_interval<F>(&self, period: Duration, callback: F) -> Result<Vow, EventLoopError>
    where
        F: FnMut() + Send + Unpin + 'static,
    {
        if period.is_zero() {
            return Err(EventLoopError::ZeroInterval);
        }
        Ok(self.insert_timer(period, Some(period), Function::new(callback)))
    }

    pub fn set_interval_ms<F>(&self, period_ms: u64, callback: F) -> Result<Vow, EventLoopError>
    where
        F: FnMut() + Send + Unpin + 'static,
    {
        self.set_interval(Duration::from_millis(period_ms), callback)
    }

    fn insert_timer(
        &self,
        delay: Duration,
        interval: Option<Duration>,
        callback: Function<(), ()>,
    ) -> Vow {
        let vow = Vow::new();
        let deadline = Instant::now() + delay;
        let mut state = self.lock_state();
        let id = state.next_id;
        state.next_id = state.next_id.wrapping_add(1).max(1);
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.wrapping_add(1);

        state.timers.insert(
            id,
            TimerTask {
                id,
                deadline,
                interval,
                callback,
                vow: vow.clone(),
            },
        );
        state.timer_heap.push(Reverse(TimerKey { deadline, sequence, id }));
        drop(state);
        self.inner.wake.notify_one();
        vow
    }

    /// Cancel a timeout. Kept as a named compatibility helper.
    pub fn clear_timeout(&self, vow: &Vow) {
        if vow.cancel() {
            self.inner.wake.notify_one();
        }
    }

    /// Cancel an interval. Kept as a named compatibility helper.
    pub fn clear_interval(&self, vow: &Vow) {
        if vow.cancel() {
            self.inner.wake.notify_one();
        }
    }

    /// Ask a running loop to stop. It may be restarted later with `run()`.
    pub fn stop(&self) {
        let mut state = self.lock_state();
        state.stop_requested = true;
        drop(state);
        self.inner.wake.notify_all();
    }

    /// Execute queued microtasks and timers that are already due, without
    /// waiting for future deadlines. Returns the number of callbacks run.
    pub fn run_until_idle(&self) -> usize {
        let mut count = 0;
        loop {
            let action = {
                let mut state = self.lock_state();
                state.ticks = state.ticks.wrapping_add(1);
                Self::take_ready_action(&mut state, Instant::now())
            };

            match action {
                Some(Action::Micro(mut task)) => {
                    if !task.vow.is_cancelled() {
                        task.callback.call0();
                        task.vow.resolve();
                    }
                    count += 1;
                }
                Some(Action::Timer(task)) => {
                    self.execute_timer(task);
                    count += 1;
                }
                None => return count,
            }
        }
    }

    /// Start the loop on the current thread and block until `stop()` is called.
    pub fn run(&self) -> Result<(), EventLoopError> {
        {
            let mut state = self.lock_state();
            if state.running {
                return Err(EventLoopError::AlreadyRunning);
            }
            state.running = true;
            state.stop_requested = false;
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_impl();
        }));

        {
            let mut state = self.lock_state();
            state.running = false;
            state.stop_requested = false;
        }

        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
        Ok(())
    }

    fn run_impl(&self) {
        loop {
            let action = {
                let mut state = self.lock_state();
                loop {
                    if state.stop_requested {
                        return;
                    }

                    let now = Instant::now();
                    state.ticks = state.ticks.wrapping_add(1);
                    if let Some(action) = Self::take_ready_action(&mut state, now) {
                        break action;
                    }

                    match Self::next_live_deadline(&mut state) {
                        Some(deadline) => {
                            let wait = deadline.saturating_duration_since(now);
                            let (new_state, _) = self
                                .inner
                                .wake
                                .wait_timeout(state, wait)
                                .unwrap_or_else(|e| e.into_inner());
                            state = new_state;
                        }
                        None => {
                            state = self.inner.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                        }
                    }
                }
            };

            match action {
                Action::Micro(mut task) => {
                    if !task.vow.is_cancelled() {
                        task.callback.call0();
                        task.vow.resolve();
                    }
                }
                Action::Timer(task) => self.execute_timer(task),
            }
        }
    }

    fn take_ready_action(state: &mut State, now: Instant) -> Option<Action> {
        while let Some(task) = state.microtasks.pop_front() {
            if task.vow.is_cancelled() {
                continue;
            }
            return Some(Action::Micro(task));
        }

        loop {
            let key = match state.timer_heap.peek().copied() {
                Some(Reverse(key)) => key,
                None => return None,
            };

            let stale_or_cancelled = match state.timers.get(&key.id) {
                None => true,
                Some(task) => task.deadline != key.deadline || task.vow.is_cancelled(),
            };

            if stale_or_cancelled {
                state.timer_heap.pop();
                if state
                    .timers
                    .get(&key.id)
                    .is_some_and(|task| task.vow.is_cancelled())
                {
                    state.timers.remove(&key.id);
                }
                continue;
            }

            if key.deadline > now {
                return None;
            }

            state.timer_heap.pop();
            return state.timers.remove(&key.id).map(Action::Timer);
        }
    }

    fn next_live_deadline(state: &mut State) -> Option<Instant> {
        loop {
            let key = match state.timer_heap.peek().copied() {
                Some(Reverse(key)) => key,
                None => return None,
            };
            match state.timers.get(&key.id) {
                Some(task) if !task.vow.is_cancelled() && task.deadline == key.deadline => {
                    return Some(key.deadline);
                }
                _ => {
                    state.timer_heap.pop();
                    if state
                        .timers
                        .get(&key.id)
                        .is_some_and(|task| task.vow.is_cancelled())
                    {
                        state.timers.remove(&key.id);
                    }
                }
            }
        }
    }

    fn execute_timer(&self, mut task: TimerTask) {
        if task.vow.is_cancelled() {
            return;
        }

        task.callback.call0();

        match task.interval {
            None => {
                task.vow.resolve();
            }
            Some(period) => {
                if task.vow.is_cancelled() {
                    return;
                }

                // Fixed-rate scheduling without a catch-up storm: preserve the
                // original phase, but skip missed periods if a callback ran late.
                let now = Instant::now();
                let mut next = task.deadline + period;
                while next <= now {
                    next += period;
                }
                task.deadline = next;

                let mut state = self.lock_state();
                let sequence = state.next_sequence;
                state.next_sequence = state.next_sequence.wrapping_add(1);
                let id = task.id;
                state.timer_heap.push(Reverse(TimerKey {
                    deadline: next,
                    sequence,
                    id,
                }));
                state.timers.insert(id, task);
                drop(state);
                self.inner.wake.notify_one();
            }
        }
    }
}

enum Action {
    Micro(MicroTask),
    Timer(TimerTask),
}

#[cfg(test)]
mod tests {
    use super::{EventLoop, VowState};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::Duration;

    #[test]
    fn next_tick_is_a_microtask() {
        let loop_ = EventLoop::new();
        let value = Arc::new(AtomicUsize::new(0));
        let out = value.clone();
        let vow = loop_.next_tick(move || { out.store(42, Ordering::SeqCst); });
        assert_eq!(loop_.run_until_idle(), 1);
        assert_eq!(value.load(Ordering::SeqCst), 42);
        assert_eq!(vow.state(), VowState::Resolved);
    }

    #[test]
    fn cancelled_timeout_does_not_run() {
        let loop_ = EventLoop::new();
        let value = Arc::new(AtomicUsize::new(0));
        let out = value.clone();
        let vow = loop_.set_timeout(Duration::ZERO, move || {
            out.fetch_add(1, Ordering::SeqCst);
        });
        loop_.clear_timeout(&vow);
        loop_.run_until_idle();
        assert_eq!(value.load(Ordering::SeqCst), 0);
        assert_eq!(vow.state(), VowState::Cancelled);
    }

    #[test]
    fn timeout_resolves_and_then_runs() {
        let loop_ = EventLoop::new();
        let value = Arc::new(AtomicUsize::new(0));
        let out = value.clone();
        let vow = loop_.set_timeout(Duration::ZERO, || {});
        vow.then(move || { out.store(1, Ordering::SeqCst); });
        loop_.run_until_idle();
        assert_eq!(value.load(Ordering::SeqCst), 1);
    }
}
