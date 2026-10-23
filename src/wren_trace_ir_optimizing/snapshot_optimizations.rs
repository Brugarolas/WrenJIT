//! Side-exit snapshot canonicalization for Trace IR.
//!
//! The optimizer may prove a dynamic SSA value constant long after the recorder
//! captured a side-exit snapshot. Converting those snapshot entries to literal
//! constants shrinks the native exit value set and can remove the producer web
//! entirely when no hot-path use remains.

use super::*;


#[inline]
fn can_materialize_constant(constant: TraceConstant, target: TraceRep) -> bool {
    let source = constant.layout().1;
    if source == target {
        return !matches!(constant, TraceConstant::String(_) | TraceConstant::BuiltinClass(_));
    }
    matches!(
        (source, target, constant),
        (TraceRep::Int32, TraceRep::Tagged | TraceRep::F64, TraceConstant::Int32(_))
            | (TraceRep::F64, TraceRep::Tagged, TraceConstant::F64Bits(_))
    )
}

/// Rewrite snapshot SSA references to literal constants whenever their canonical
/// value is constant. `canonicalize_references()` already flattens aliases after
/// every pass; this pass performs the representation-changing step from
/// `SnapshotValue::Value` to `SnapshotValue::Constant`.
#[derive(Default)]
pub struct SnapshotConstantPropagation;

impl TraceOptimization for SnapshotConstantPropagation {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let constants: Vec<Option<TraceConstant>> = (0..trace.values.len())
            .map(|index| {
                let value = trace.canonical_value(TraceValueId(index as u32));
                match trace.value(value).map(|value| &value.origin) {
                    Some(TraceValueOrigin::Constant(constant)) => Some(*constant),
                    _ => None,
                }
            })
            .collect();

        let mut changed = false;
        for exit in &mut trace.exits {
            for frame in &mut exit.frames {
                for write in &mut frame.writes {
                    let SnapshotValue::Value(value) = write.value else {
                        continue;
                    };
                    let Some(Some(constant)) = constants.get(value.0 as usize) else {
                        continue;
                    };
                    if !can_materialize_constant(*constant, write.target_rep) {
                        continue;
                    }
                    write.value = SnapshotValue::Constant(*constant);
                    trace.stats.peephole_simplified += 1;
                    changed = true;
                }
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{BytecodePc, Reg};
    use crate::wren_ir::{FuncId, IntBinOp};

    #[test]
    fn turns_folded_snapshot_values_into_literals() {
        let mut builder = TraceBuilder::new(
            TraceId(0),
            TraceAnchor {
                kind: TraceKind::Side,
                location: TraceLocation::new(FuncId(0), BytecodePc(0)),
            },
        );
        let one = builder.constant(TraceConstant::Int32(1));
        let two = builder.constant(TraceConstant::Int32(2));
        let sum = builder.emit1(
            TraceInst::IntBinary {
                op: IntBinOp::Add,
                lhs: one,
                rhs: two,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        builder.write_slot(FrameSlot::General(Reg(0)), sum);
        let exit = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(3));
        builder.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = builder.finish().unwrap();

        Optimize::new()
            .with_pass(SparseConditionalConstantPropagation)
            .with_pass(SnapshotConstantPropagation)
            .with_pass(DeadCodeElimination)
            .optimize_trace(&mut trace)
            .unwrap();

        assert!(matches!(
            trace.exits[exit.0 as usize].frames[0].writes[0].value,
            SnapshotValue::Constant(TraceConstant::Int32(3))
        ));
        assert!(matches!(trace.nodes[0].kind, TraceInst::Nop));
    }
}
