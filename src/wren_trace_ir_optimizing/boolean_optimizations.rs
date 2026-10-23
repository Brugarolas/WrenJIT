//! Eliminate tagged-boolean materialization when the value is used only by guards.
//!
//! The recorder already avoids boxing the common directly-recorded branch case.
//! This pass catches equivalent shapes exposed later by inlining, SCCP, PRE, or
//! other Trace IR rewrites.

use super::*;

fn tagged_bool(trace: &Trace, value: TraceValueId) -> Option<bool> {
    let value = trace.canonical_value(value);
    match &trace.value(value)?.origin {
        TraceValueOrigin::Constant(TraceConstant::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn used_in_snapshot_or_phi(trace: &Trace, value: TraceValueId) -> bool {
    let value = trace.canonical_value(value);
    if trace.exits.iter().any(|exit| {
        exit.frames.iter().any(|frame| {
            frame.writes.iter().any(|write| {
                matches!(
                    write.value,
                    SnapshotValue::Value(v) if trace.canonical_value(v) == value
                )
            })
        })
    }) {
        return true;
    }
    trace.loop_phis.iter().any(|phi| {
        trace.canonical_value(phi.initial) == value
            || phi
                .backedge
                .is_some_and(|backedge| trace.canonical_value(backedge) == value)
    })
}

/// Rewrite tagged-boolean branch guards to operate on the raw `TraceRep::Bool`
/// producer and remove the `Select(cond, true, false)` materialization entirely.
#[derive(Default)]
pub struct BooleanMaterializationElimination;

impl TraceOptimization for BooleanMaterializationElimination {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut changed = false;
        let candidates: Vec<_> = trace
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                let TraceInst::Select {
                    condition,
                    if_true,
                    if_false,
                } = &node.kind
                else {
                    return None;
                };
                let &result = node.results.first()?;
                let condition = trace.canonical_value(*condition);
                if trace.value(condition)?.rep != TraceRep::Bool
                    || trace.value(result)?.rep != TraceRep::Tagged
                {
                    return None;
                }
                let if_true = tagged_bool(trace, *if_true)?;
                let if_false = tagged_bool(trace, *if_false)?;
                if if_true == if_false {
                    return None;
                }
                Some((id, trace.canonical_value(result), condition, if_true))
            })
            .collect();

        for (select_id, result, condition, true_when_condition) in candidates {
            if used_in_snapshot_or_phi(trace, result) {
                continue;
            }

            let mut rewrites = Vec::<(usize, TraceInst)>::new();
            let mut valid = true;
            for (id, node) in trace.nodes.iter().enumerate() {
                if id == select_id || matches!(node.kind, TraceInst::Nop) {
                    continue;
                }

                let mut uses = 0usize;
                let mut kind = node.kind.clone();
                kind.visit_operands(|operand| {
                    if trace.canonical_value(*operand) == result {
                        uses += 1;
                    }
                });
                if uses == 0 {
                    continue;
                }

                let rewritten = match &node.kind {
                    TraceInst::GuardTruthy { value, exit }
                        if trace.canonical_value(*value) == result =>
                    {
                        Some(TraceInst::GuardSwitch {
                            tag: condition,
                            observed: TraceSwitchGuard::Case(if true_when_condition { 1 } else { 0 }),
                            exit: *exit,
                        })
                    }
                    TraceInst::GuardFalsey { value, exit }
                        if trace.canonical_value(*value) == result =>
                    {
                        Some(TraceInst::GuardSwitch {
                            tag: condition,
                            observed: TraceSwitchGuard::Case(if true_when_condition { 0 } else { 1 }),
                            exit: *exit,
                        })
                    }
                    TraceInst::GuardEq { lhs, rhs, exit } => {
                        let lhs = trace.canonical_value(*lhs);
                        let rhs = trace.canonical_value(*rhs);
                        let other = if lhs == result {
                            Some(rhs)
                        } else if rhs == result {
                            Some(lhs)
                        } else {
                            None
                        };
                        other.and_then(|other| tagged_bool(trace, other)).map(|expected| {
                            let raw = if expected == true_when_condition { 1 } else { 0 };
                            TraceInst::GuardSwitch {
                                tag: condition,
                                observed: TraceSwitchGuard::Case(raw),
                                exit: *exit,
                            }
                        })
                    }
                    _ => None,
                };

                if uses != 1 || rewritten.is_none() {
                    valid = false;
                    break;
                }
                rewrites.push((id, rewritten.unwrap()));
            }

            if !valid || rewrites.is_empty() {
                continue;
            }
            for (id, kind) in rewrites {
                trace.nodes[id].kind = kind;
            }
            trace.nodes[select_id].kind = TraceInst::Nop;
            trace.stats.peephole_simplified += 1;
            changed = true;
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{BytecodePc, Reg};
    use crate::wren_ir::{CmpKind, FuncId};

    #[test]
    fn removes_tagged_bool_used_only_by_branch_guard() {
        let mut builder = TraceBuilder::new(
            TraceId(0),
            TraceAnchor {
                kind: TraceKind::Side,
                location: TraceLocation::new(FuncId(0), BytecodePc(0)),
            },
        );
        let lhs = builder.read_slot(
            FrameSlot::General(Reg(0)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let rhs = builder.constant(TraceConstant::Int32(10));
        let condition = builder.emit1(
            TraceInst::IntCompare {
                op: CmpKind::Lt,
                lhs,
                rhs,
            },
            TraceType::Bool,
            TraceRep::Bool,
        );
        let yes = builder.constant(TraceConstant::Bool(true));
        let no = builder.constant(TraceConstant::Bool(false));
        let tagged = builder.emit1(
            TraceInst::Select {
                condition,
                if_true: yes,
                if_false: no,
            },
            TraceType::Bool,
            TraceRep::Tagged,
        );
        let exit = builder.add_side_exit(ExitReason::GuardTruthy, BytecodePc(10));
        builder.emit0(TraceInst::GuardTruthy {
            value: tagged,
            exit,
        });
        builder.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = builder.finish().unwrap();

        Optimize::new()
            .with_pass(BooleanMaterializationElimination)
            .optimize_trace(&mut trace)
            .unwrap();

        assert!(matches!(trace.nodes[1].kind, TraceInst::Nop));
        assert!(trace.nodes.iter().any(|node| {
            matches!(
                node.kind,
                TraceInst::GuardSwitch {
                    tag,
                    observed: TraceSwitchGuard::Case(1),
                    ..
                } if tag == condition
            )
        }));
    }
}
