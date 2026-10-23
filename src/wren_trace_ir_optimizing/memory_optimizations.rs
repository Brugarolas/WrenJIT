//! Architectural-memory value forwarding for Trace IR.
//!
//! Heap accesses are handled by `LoadStorePRE` in `advanced.rs`. This pass covers
//! VM frame slots and lifetime-stable absolute Value slots, whose aliasing rules
//! are much simpler and therefore permit stronger forwarding.

use super::*;
use hashbrown::{HashMap, HashSet};

fn execution_order(trace: &Trace) -> Vec<usize> {
    let preheader: HashSet<_> = trace.preheader.iter().copied().collect();
    trace
        .preheader
        .iter()
        .copied()
        .chain((0..trace.nodes.len()).filter(|id| !preheader.contains(id)))
        .collect()
}

#[inline]
fn replace_result(trace: &mut Trace, node: usize, value: TraceValueId) {
    let Some(&result) = trace.nodes[node].results.first() else {
        return;
    };
    alias_value(trace, result, value);
    trace.nodes[node].kind = TraceInst::Nop;
}

/// Forward synchronized architectural loads within one runtime epoch.
///
/// Rules:
/// - a `CommitFrameSlot` establishes the exact value subsequently visible from
///   that slot for the same representation;
/// - a `StoreAbsoluteValue` establishes the exact value at that stable address;
/// - `Runtime` and `RuntimeCall` invalidate both maps because runtime code may
///   mutate registers, globals, module state, or switch the active frame;
/// - heap stores do not alias either namespace.
#[derive(Default)]
pub struct ArchitecturalLoadForwarding;

impl TraceOptimization for ArchitecturalLoadForwarding {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut frames = HashMap::<(FrameSlot, TraceRep), TraceValueId>::new();
        let mut absolute = HashMap::<u64, TraceValueId>::new();
        let mut changed = false;

        for id in execution_order(trace) {
            let kind = trace.nodes[id].kind.clone();
            match kind {
                TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. } => {
                    frames.clear();
                    absolute.clear();
                }
                TraceInst::CommitFrameSlot { slot, value } => {
                    let value = trace.canonical_value(value);
                    frames.retain(|(known_slot, _), _| *known_slot != slot);
                    if let Some(rep) = trace.value(value).map(|value| value.rep) {
                        frames.insert((slot, rep), value);
                    }
                }
                TraceInst::LoadFrameSlot { slot } => {
                    let Some(&result) = trace.nodes[id].results.first() else {
                        continue;
                    };
                    let Some(rep) = trace.value(result).map(|value| value.rep) else {
                        continue;
                    };
                    if let Some(&known) = frames.get(&(slot, rep)) {
                        replace_result(trace, id, known);
                        trace.stats.forwarded_loads += 1;
                        changed = true;
                    } else {
                        frames.insert((slot, rep), result);
                    }
                }
                TraceInst::StoreAbsoluteValue { address, value } => {
                    absolute.insert(address, trace.canonical_value(value));
                }
                TraceInst::LoadAbsoluteValue { address } => {
                    let Some(&result) = trace.nodes[id].results.first() else {
                        continue;
                    };
                    if let Some(&known) = absolute.get(&address) {
                        if trace.value(known).map(|value| value.rep)
                            == trace.value(result).map(|value| value.rep)
                        {
                            replace_result(trace, id, known);
                            trace.stats.forwarded_loads += 1;
                            changed = true;
                        }
                    } else {
                        absolute.insert(address, result);
                    }
                }
                _ => {}
            }
        }

        changed
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DeadStoreKey {
    Frame(FrameSlot),
    Absolute(u64),
}

#[inline]
fn dead_store_key(kind: &TraceInst) -> Option<DeadStoreKey> {
    match kind {
        TraceInst::CommitFrameSlot { slot, .. } => Some(DeadStoreKey::Frame(*slot)),
        TraceInst::StoreAbsoluteValue { address, .. } => Some(DeadStoreKey::Absolute(*address)),
        _ => None,
    }
}

/// Remove an earlier architectural store when it is overwritten before any
/// operation can observe it.
///
/// This pass intentionally limits classic store-over-store DSE to non-trapping
/// VM frame and absolute-value stores. Heap stores carry validation exits and GC
/// write barriers, so deleting the earlier heap store would also delete an
/// observable side exit/barrier. `LoadStorePRE` already removes the safe case of
/// a later *identical* heap store after the first store performed validation.
#[derive(Default)]
pub struct DeadStoreElimination;

impl TraceOptimization for DeadStoreElimination {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut pending = HashMap::<DeadStoreKey, usize>::new();
        let mut changed = false;

        for id in execution_order(trace) {
            let kind = trace.nodes[id].kind.clone();
            if matches!(kind, TraceInst::Nop) {
                continue;
            }

            // A side exit can expose architectural/global state to the
            // interpreter. Runtime and read-barrier boundaries are even stronger
            // observation points, so never move DSE facts across them.
            if kind.exit().is_some()
                || matches!(
                    kind,
                    TraceInst::Runtime { .. }
                        | TraceInst::RuntimeCall { .. }
                        | TraceInst::ReadBarrier { .. }
                )
            {
                pending.clear();
            }

            match &kind {
                TraceInst::LoadFrameSlot { slot } => {
                    pending.remove(&DeadStoreKey::Frame(*slot));
                }
                TraceInst::LoadAbsoluteValue { address } => {
                    pending.remove(&DeadStoreKey::Absolute(*address));
                }
                _ => {}
            }

            let Some(key) = dead_store_key(&kind) else {
                continue;
            };

            if let Some(previous) = pending.insert(key, id) {
                if !matches!(trace.nodes[previous].kind, TraceInst::Nop) {
                    trace.nodes[previous].kind = TraceInst::Nop;
                    trace.stats.eliminated_stores += 1;
                    changed = true;
                }
            }
        }

        changed
    }
}

/// Complete redundant-load elimination for the two memory namespaces modeled
/// by Trace IR: heap/container loads (`LoadStorePRE`) and architectural VM/global
/// loads (`ArchitecturalLoadForwarding`).
#[derive(Default)]
pub struct RedundantLoadElimination;

impl TraceOptimization for RedundantLoadElimination {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut heap = LoadStorePRE;
        let mut architectural = ArchitecturalLoadForwarding;
        let heap_changed = heap.optimize_trace(trace);
        let architectural_changed = architectural.optimize_trace(trace);
        heap_changed || architectural_changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{BytecodePc, Reg};
    use crate::wren_ir::FuncId;

    fn side_builder() -> TraceBuilder {
        TraceBuilder::new(
            TraceId(0),
            TraceAnchor {
                kind: TraceKind::Side,
                location: TraceLocation::new(FuncId(0), BytecodePc(0)),
            },
        )
    }

    #[test]
    fn forwards_frame_and_absolute_loads_but_not_across_runtime() {
        let mut builder = side_builder();
        let slot = FrameSlot::General(Reg(0));
        let value = builder.constant(TraceConstant::NumBits(3.0f64.to_bits()));
        builder.emit0(TraceInst::CommitFrameSlot { slot, value });
        let frame = builder.emit1(
            TraceInst::LoadFrameSlot { slot },
            TraceType::Num,
            TraceRep::Tagged,
        );

        let address = 0x1000;
        builder.emit0(TraceInst::StoreAbsoluteValue { address, value });
        let absolute = builder.emit1(
            TraceInst::LoadAbsoluteValue { address },
            TraceType::Any,
            TraceRep::Tagged,
        );

        builder.runtime(
            TraceLocation::new(FuncId(0), BytecodePc(1)),
            BytecodePc(2),
            false,
        );
        let after_runtime = builder.emit1(
            TraceInst::LoadAbsoluteValue { address },
            TraceType::Any,
            TraceRep::Tagged,
        );
        let exit = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(3));
        builder.emit0(TraceInst::GuardEq {
            lhs: after_runtime,
            rhs: after_runtime,
            exit,
        });
        builder.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = builder.finish().unwrap();

        Optimize::new()
            .with_pass(ArchitecturalLoadForwarding)
            .optimize_trace(&mut trace)
            .unwrap();

        assert_eq!(trace.canonical_value(frame), value);
        assert_eq!(trace.canonical_value(absolute), value);
        assert_eq!(trace.canonical_value(after_runtime), after_runtime);
        assert_eq!(trace.stats.forwarded_loads, 2);
    }

    #[test]
    fn eliminates_overwritten_architectural_stores_before_observation() {
        let mut builder = side_builder();
        let slot = FrameSlot::General(Reg(0));
        let first = builder.constant(TraceConstant::NumBits(1.0f64.to_bits()));
        let second = builder.constant(TraceConstant::NumBits(2.0f64.to_bits()));

        builder.emit0(TraceInst::CommitFrameSlot { slot, value: first });
        let one = builder.constant(TraceConstant::Int32(1));
        let two = builder.constant(TraceConstant::Int32(2));
        builder.emit1(
            TraceInst::IntBinary {
                op: crate::wren_ir::IntBinOp::Add,
                lhs: one,
                rhs: two,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        builder.emit0(TraceInst::CommitFrameSlot { slot, value: second });

        let address = 0x2000;
        builder.emit0(TraceInst::StoreAbsoluteValue { address, value: first });
        builder.emit0(TraceInst::StoreAbsoluteValue { address, value: second });

        let exit = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(4));
        builder.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = builder.finish().unwrap();

        Optimize::new()
            .with_pass(DeadStoreElimination)
            .optimize_trace(&mut trace)
            .unwrap();

        assert!(matches!(trace.nodes[0].kind, TraceInst::Nop));
        assert!(matches!(trace.nodes[2].kind, TraceInst::CommitFrameSlot { .. }));
        assert!(matches!(trace.nodes[3].kind, TraceInst::Nop));
        assert!(matches!(trace.nodes[4].kind, TraceInst::StoreAbsoluteValue { .. }));
        assert_eq!(trace.stats.eliminated_stores, 2);
    }

    #[test]
    fn dead_store_elimination_stops_at_side_exits_and_reads() {
        let mut builder = side_builder();
        let slot = FrameSlot::General(Reg(0));
        let first = builder.constant(TraceConstant::NumBits(1.0f64.to_bits()));
        let second = builder.constant(TraceConstant::NumBits(2.0f64.to_bits()));

        builder.emit0(TraceInst::CommitFrameSlot { slot, value: first });
        let guard_exit = builder.add_side_exit(ExitReason::GuardTruthy, BytecodePc(1));
        let truth = builder.constant(TraceConstant::Bool(true));
        builder.emit0(TraceInst::GuardTruthy {
            value: truth,
            exit: guard_exit,
        });
        builder.emit0(TraceInst::CommitFrameSlot { slot, value: second });
        builder.emit1(
            TraceInst::LoadFrameSlot { slot },
            TraceType::Any,
            TraceRep::Tagged,
        );
        builder.emit0(TraceInst::CommitFrameSlot { slot, value: first });

        let exit = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(5));
        builder.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = builder.finish().unwrap();

        Optimize::new()
            .with_pass(DeadStoreElimination)
            .optimize_trace(&mut trace)
            .unwrap();

        assert!(matches!(trace.nodes[0].kind, TraceInst::CommitFrameSlot { .. }));
        assert!(matches!(trace.nodes[1].kind, TraceInst::GuardTruthy { .. }));
        assert!(matches!(trace.nodes[2].kind, TraceInst::CommitFrameSlot { .. }));
        assert!(matches!(trace.nodes[3].kind, TraceInst::LoadFrameSlot { .. }));
        assert!(matches!(trace.nodes[4].kind, TraceInst::CommitFrameSlot { .. }));
        assert_eq!(trace.stats.eliminated_stores, 0);
    }

}
