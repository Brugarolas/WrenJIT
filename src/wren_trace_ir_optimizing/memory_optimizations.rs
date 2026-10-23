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
}
