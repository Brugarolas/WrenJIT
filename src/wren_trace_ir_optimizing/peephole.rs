//! Cheap local rewrites and dominated duplicate-guard elimination.
//!
//! Every rewrite is bit-exact with the native lowering: integer operations
//! wrap (shift amounts are masked to 31), checked arithmetic is never touched,
//! and float rewrites are limited to identities that hold for every input,
//! including NaN payloads, signed zeros, infinities and subnormal results.
use super::advanced::{constant, float, integer, intern, order, replace};
use super::*;
use crate::wren_ir::{CmpKind, F64BinOp, F64UnaryOp};

fn producer(trace: &Trace, value: TraceValueId) -> Option<TraceInst> {
    TraceCanonicalization::producer(trace, value).map(|(_, kind)| kind)
}

fn int_const(trace: &Trace, value: TraceValueId) -> Option<i32> {
    constant(trace, value).and_then(integer)
}

fn commutative(op: IntBinOp) -> bool {
    matches!(op, IntBinOp::Add | IntBinOp::Mul | IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor)
}

const MANTISSA: u64 = (1u64 << 52) - 1;

/// `x / d == x * (1 / d)` for every `x` iff `d` and `1 / d` are both normal
/// powers of two: both sides are a single correctly rounded scaling.
fn exact_reciprocal(divisor: f64) -> Option<f64> {
    if !divisor.is_normal() || divisor.to_bits() & MANTISSA != 0 {
        return None;
    }
    let reciprocal = 1.0 / divisor;
    (reciprocal.is_normal() && reciprocal.to_bits() & MANTISSA == 0).then_some(reciprocal)
}

fn exact_i32(value: f64) -> Option<i32> {
    (value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64).then(|| value as i32)
}

enum Rewrite {
    Value(TraceValueId),
    Const(i32),
    Kind(TraceInst),
}

fn simplify(trace: &mut Trace, id: usize) -> Option<Rewrite> {
    use TraceInst::*;
    let result = *trace.nodes[id].results.first()?;
    let _ = result;
    match trace.nodes[id].kind.clone() {
        IntBinary { op, lhs, rhs } => {
            let (lhs, rhs) = (trace.canonical_value(lhs), trace.canonical_value(rhs));
            let (lc, rc) = (int_const(trace, lhs), int_const(trace, rhs));
            // Constants on the right make the remaining rules (and GVN keys) canonical.
            if commutative(op) && lc.is_some() && rc.is_none() {
                return Some(Rewrite::Kind(IntBinary { op, lhs: rhs, rhs: lhs }));
            }
            if lhs == rhs {
                match op {
                    IntBinOp::Sub | IntBinOp::BitXor => return Some(Rewrite::Const(0)),
                    IntBinOp::BitAnd | IntBinOp::BitOr => return Some(Rewrite::Value(lhs)),
                    _ => {}
                }
            }
            let c = rc?;
            match (op, c) {
                (IntBinOp::Add | IntBinOp::Sub | IntBinOp::BitOr | IntBinOp::BitXor
                    | IntBinOp::Shl | IntBinOp::ShrU | IntBinOp::Sar, 0)
                | (IntBinOp::Mul, 1)
                | (IntBinOp::BitAnd, -1) => return Some(Rewrite::Value(lhs)),
                (IntBinOp::Mul | IntBinOp::BitAnd, 0) => return Some(Rewrite::Const(0)),
                (IntBinOp::BitOr, -1) => return Some(Rewrite::Const(-1)),
                (IntBinOp::Mul, -1) => return Some(Rewrite::Kind(IntUnary { op: IntUnaryOp::Neg, value: lhs })),
                (IntBinOp::Sub, _) => {
                    let layout = trace.value(rhs)?;
                    let (ty, rep) = (layout.ty, layout.rep);
                    let negated = intern(trace, TraceConstant::Int32(c.wrapping_neg()), ty, rep);
                    return Some(Rewrite::Kind(IntBinary { op: IntBinOp::Add, lhs, rhs: negated }));
                }
                _ => {}
            }
            // (x op c1) op c2 -> x op (c1 op c2); all are associative modulo 2^32.
            if !commutative(op) {
                return None;
            }
            let IntBinary { op: inner, lhs: x, rhs: c1 } = producer(trace, lhs)? else { return None };
            if inner != op {
                return None;
            }
            let c1 = int_const(trace, c1)?;
            let folded = match op {
                IntBinOp::Add => c1.wrapping_add(c),
                IntBinOp::Mul => c1.wrapping_mul(c),
                IntBinOp::BitAnd => c1 & c,
                IntBinOp::BitOr => c1 | c,
                _ => c1 ^ c,
            };
            let layout = trace.value(rhs)?;
            let (ty, rep) = (layout.ty, layout.rep);
            let folded = intern(trace, TraceConstant::Int32(folded), ty, rep);
            Some(Rewrite::Kind(IntBinary { op, lhs: x, rhs: folded }))
        }
        IntUnary { op, value } => match producer(trace, value)? {
            IntUnary { op: inner, value: x } if inner == op => Some(Rewrite::Value(x)),
            _ => None,
        },
        F64Unary { op: F64UnaryOp::Neg, value } => match producer(trace, value)? {
            F64Unary { op: F64UnaryOp::Neg, value: x } => Some(Rewrite::Value(x)),
            _ => None,
        },
        F64Binary { op: F64BinOp::Div, lhs, rhs } => {
            let reciprocal = exact_reciprocal(constant(trace, rhs).and_then(float)?)?;
            let layout = trace.value(rhs)?;
            let (ty, rep) = (layout.ty, layout.rep);
            let rhs = intern(trace, TraceConstant::F64Bits(reciprocal.to_bits()), ty, rep);
            Some(Rewrite::Kind(F64Binary { op: F64BinOp::Mul, lhs, rhs }))
        }
        F64Binary { op: F64BinOp::Mul, lhs, rhs } => {
            let two = |v| constant(trace, v).and_then(float) == Some(2.0);
            let x = if two(rhs) { lhs } else if two(lhs) { rhs } else { return None };
            Some(Rewrite::Kind(F64Binary { op: F64BinOp::Add, lhs: x, rhs: x }))
        }
        IntCompare { op, lhs, rhs } if trace.canonical_value(lhs) == trace.canonical_value(rhs) => {
            Some(Rewrite::Const(matches!(op, CmpKind::Eq | CmpKind::Le | CmpKind::Ge) as i32))
        }
        // Int32 -> f64 is exact, so comparing the widened values is comparing
        // the integers. A float constant qualifies only when it is an exact i32.
        F64Compare { op, lhs, rhs } => {
            let widened = |v| match producer(trace, v) {
                Some(Int32ToF64 { value }) => Some(value),
                _ => None,
            };
            let (wl, wr) = (widened(lhs), widened(rhs));
            if wl.is_none() && wr.is_none() {
                return None;
            }
            let narrow = |trace: &mut Trace, widened: Option<TraceValueId>, v| match widened {
                Some(value) => Some(value),
                None => {
                    let n = exact_i32(constant(trace, v).and_then(float)?)?;
                    Some(intern(trace, TraceConstant::Int32(n), TraceType::Int32, TraceRep::Int32))
                }
            };
            let lhs = narrow(trace, wl, lhs)?;
            let rhs = narrow(trace, wr, rhs)?;
            Some(Rewrite::Kind(IntCompare { op, lhs, rhs }))
        }
        _ => None,
    }
}

/// Algebraic identities, canonical constant placement, constant reassociation,
/// exact float scaling (`x / 2^k` -> `x * 2^-k`, `x * 2` -> `x + x`) and
/// integer-domain comparisons of widened Int32 values.
#[derive(Default)]
pub struct AlgebraicSimplification;
impl TraceOptimization for AlgebraicSimplification {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut changed = false;
        for _ in 0..4 {
            let mut round = false;
            for id in order(trace) {
                if trace.nodes[id].results.len() != 1 || matches!(trace.nodes[id].kind, TraceInst::Nop) {
                    continue;
                }
                let Some(rewrite) = simplify(trace, id) else { continue };
                match rewrite {
                    Rewrite::Value(value) => replace(trace, id, value),
                    Rewrite::Const(n) => {
                        let result = trace.value(trace.nodes[id].results[0]).unwrap();
                        let (ty, rep) = (result.ty, result.rep);
                        let value = intern(trace, TraceConstant::Int32(n), ty, rep);
                        replace(trace, id, value);
                    }
                    Rewrite::Kind(kind) => trace.nodes[id].kind = kind,
                }
                trace.stats.peephole_simplified += 1;
                round = true;
            }
            changed |= round;
            if !round {
                break;
            }
        }
        changed
    }
}

/// A guard that is dominated by an identical successful guard on the same SSA
/// operands cannot fail. Value guards are pure, so they survive runtime calls;
/// only the method-table epoch can change while the runtime executes.
#[derive(Default)]
pub struct RedundantGuardElimination;
impl TraceOptimization for RedundantGuardElimination {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut seen: HashMap<TraceInst, Option<TraceValueId>> = HashMap::new();
        let mut changed = false;
        for id in order(trace) {
            let kind = trace.nodes[id].kind.clone();
            if matches!(kind, TraceInst::Runtime { .. } | TraceInst::ReadBarrier { .. }) {
                seen.retain(|key, _| !matches!(key, TraceInst::GuardMethodEpoch { .. }));
                continue;
            }
            // GuardClass facts are owned (and narrowed) by ClassTypePropagation.
            if !kind.is_entry_guard() || matches!(kind, TraceInst::GuardClass { .. }) {
                continue;
            }
            let mut key = kind;
            key.visit_operands(|v| *v = trace.canonical_value(*v));
            key.set_exit(SideExitId(0));
            let result = trace.nodes[id].results.first().copied();
            let same_layout = |a: TraceValueId, b: TraceValueId| {
                let (a, b) = (trace.value(a).unwrap(), trace.value(b).unwrap());
                a.ty == b.ty && a.rep == b.rep
            };
            match (seen.get(&key).copied(), result) {
                (Some(_), None) => trace.nodes[id].kind = TraceInst::Nop,
                (Some(Some(previous)), Some(result)) if same_layout(previous, result) => replace(trace, id, previous),
                _ => {
                    seen.insert(key, result);
                    continue;
                }
            }
            trace.stats.redundant_guards += 1;
            trace.stats.eliminated_guards += 1;
            changed = true;
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{BytecodePc, Reg};
    use crate::wren_ir::FuncId;

    fn builder(kind: TraceKind) -> TraceBuilder {
        TraceBuilder::new(TraceId(0), TraceAnchor { kind, location: TraceLocation::new(FuncId(0), BytecodePc(0)) })
    }
    fn int_input(b: &mut TraceBuilder, reg: u8) -> TraceValueId {
        b.read_slot(FrameSlot::General(Reg(reg)), TraceType::Int32, TraceRep::Int32)
    }
    fn f64_input(b: &mut TraceBuilder, reg: u8) -> TraceValueId {
        b.read_slot(FrameSlot::General(Reg(reg)), TraceType::F64, TraceRep::F64)
    }
    fn int(b: &mut TraceBuilder, n: i32) -> TraceValueId { b.constant(TraceConstant::Int32(n)) }
    fn num(b: &mut TraceBuilder, n: f64) -> TraceValueId { b.constant(TraceConstant::F64Bits(n.to_bits())) }
    fn ib(b: &mut TraceBuilder, op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId) -> TraceValueId {
        b.emit1(TraceInst::IntBinary { op, lhs, rhs }, TraceType::Int32, TraceRep::Int32)
    }
    fn fb(b: &mut TraceBuilder, op: F64BinOp, lhs: TraceValueId, rhs: TraceValueId) -> TraceValueId {
        b.emit1(TraceInst::F64Binary { op, lhs, rhs }, TraceType::F64, TraceRep::F64)
    }
    fn finish(mut b: TraceBuilder) -> Trace {
        if matches!(b.trace().anchor.kind, TraceKind::Loop) {
            b.set_terminator(TraceTerminator::LoopBack);
        } else {
            let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(10));
            b.set_terminator(TraceTerminator::Exit(exit));
        }
        b.finish().unwrap()
    }
    fn run(trace: &mut Trace, pass: impl TraceOptimization + 'static) {
        Optimize::new().with_pass(pass).verify_after_each_pass(true).optimize_trace(trace).unwrap();
    }
    fn kind_of(trace: &Trace, value: TraceValueId) -> TraceInst {
        TraceCanonicalization::producer(trace, value).unwrap().1
    }

    #[test]
    fn integer_identities_fold_to_operands_and_constants() {
        let mut b = builder(TraceKind::Side);
        let x = int_input(&mut b, 0);
        let (zero, one, minus_one) = (int(&mut b, 0), int(&mut b, 1), int(&mut b, -1));
        let identities = [
            ib(&mut b, IntBinOp::Add, x, zero), ib(&mut b, IntBinOp::Add, zero, x),
            ib(&mut b, IntBinOp::Sub, x, zero), ib(&mut b, IntBinOp::Mul, x, one),
            ib(&mut b, IntBinOp::Mul, one, x), ib(&mut b, IntBinOp::BitAnd, x, minus_one),
            ib(&mut b, IntBinOp::BitOr, x, zero), ib(&mut b, IntBinOp::BitXor, zero, x),
            ib(&mut b, IntBinOp::Shl, x, zero), ib(&mut b, IntBinOp::ShrU, x, zero),
            ib(&mut b, IntBinOp::Sar, x, zero), ib(&mut b, IntBinOp::BitAnd, x, x),
            ib(&mut b, IntBinOp::BitOr, x, x),
        ];
        let zeros = [
            ib(&mut b, IntBinOp::Mul, x, zero), ib(&mut b, IntBinOp::BitAnd, zero, x),
            ib(&mut b, IntBinOp::Sub, x, x), ib(&mut b, IntBinOp::BitXor, x, x),
        ];
        let ones = ib(&mut b, IntBinOp::BitOr, x, minus_one);
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        for v in identities { assert_eq!(trace.canonical_value(v), x); }
        for v in zeros { assert_eq!(constant(&trace, v), Some(TraceConstant::Int32(0))); }
        assert_eq!(constant(&trace, ones), Some(TraceConstant::Int32(-1)));
    }

    #[test]
    fn checked_arithmetic_and_nonzero_shifts_are_untouched() {
        let mut b = builder(TraceKind::Side);
        let x = int_input(&mut b, 0);
        let (zero, thirty_two) = (int(&mut b, 0), int(&mut b, 32));
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        let checked = b.emit1(TraceInst::IntBinaryChecked { op: IntBinOp::Add, lhs: x, rhs: zero, exit }, TraceType::Int32, TraceRep::Int32);
        let shift = ib(&mut b, IntBinOp::Shl, x, thirty_two);
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        assert_eq!(trace.canonical_value(checked), checked);
        assert_eq!(trace.canonical_value(shift), shift);
        assert_eq!(trace.stats.peephole_simplified, 0);
    }

    #[test]
    fn subtraction_and_constant_chains_reassociate_with_wrapping() {
        let mut b = builder(TraceKind::Side);
        let x = int_input(&mut b, 0);
        let (five, seven, max) = (int(&mut b, 5), int(&mut b, 7), int(&mut b, i32::MAX));
        let a = ib(&mut b, IntBinOp::Add, x, five);
        let added = ib(&mut b, IntBinOp::Sub, a, seven);
        let m = ib(&mut b, IntBinOp::Mul, x, max);
        let multiplied = ib(&mut b, IntBinOp::Mul, max, m);
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        let TraceInst::IntBinary { op: IntBinOp::Add, lhs, rhs } = kind_of(&trace, added) else { panic!() };
        assert_eq!((lhs, constant(&trace, rhs)), (x, Some(TraceConstant::Int32(-2))));
        let TraceInst::IntBinary { op: IntBinOp::Mul, lhs, rhs } = kind_of(&trace, multiplied) else { panic!() };
        assert_eq!((lhs, constant(&trace, rhs)), (x, Some(TraceConstant::Int32(i32::MAX.wrapping_mul(i32::MAX)))));
    }

    #[test]
    fn double_negations_and_negative_one_multiplication() {
        let mut b = builder(TraceKind::Side);
        let x = int_input(&mut b, 0);
        let f = f64_input(&mut b, 1);
        let minus_one = int(&mut b, -1);
        let n1 = b.emit1(TraceInst::IntUnary { op: IntUnaryOp::BitNot, value: x }, TraceType::Int32, TraceRep::Int32);
        let n2 = b.emit1(TraceInst::IntUnary { op: IntUnaryOp::BitNot, value: n1 }, TraceType::Int32, TraceRep::Int32);
        let f1 = b.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value: f }, TraceType::F64, TraceRep::F64);
        let f2 = b.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value: f1 }, TraceType::F64, TraceRep::F64);
        let negated = ib(&mut b, IntBinOp::Mul, x, minus_one);
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        assert_eq!(trace.canonical_value(n2), x);
        assert_eq!(trace.canonical_value(f2), f);
        assert!(matches!(kind_of(&trace, negated), TraceInst::IntUnary { op: IntUnaryOp::Neg, .. }));
    }

    #[test]
    fn float_division_uses_only_exact_reciprocals() {
        let mut b = builder(TraceKind::Side);
        let x = f64_input(&mut b, 0);
        let rewritten: Vec<_> = [4.0, 0.5, -2.0, 1.0].into_iter().map(|d| { let d = num(&mut b, d); fb(&mut b, F64BinOp::Div, x, d) }).collect();
        let kept: Vec<_> = [3.0, 0.0, f64::INFINITY, f64::NAN, f64::MIN_POSITIVE / 4.0, 2f64.powi(1023)]
            .into_iter().map(|d| { let d = num(&mut b, d); fb(&mut b, F64BinOp::Div, x, d) }).collect();
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        for (v, r) in rewritten.into_iter().zip([0.25, 2.0, -0.5, 1.0]) {
            let TraceInst::F64Binary { op: F64BinOp::Mul, rhs, .. } = kind_of(&trace, v) else { panic!() };
            assert_eq!(constant(&trace, rhs).and_then(float), Some(r));
        }
        for v in kept { assert!(matches!(kind_of(&trace, v), TraceInst::F64Binary { op: F64BinOp::Div, .. })); }
    }

    #[test]
    fn widened_int_comparisons_become_integer_comparisons() {
        let mut b = builder(TraceKind::Side);
        let (i, j) = (int_input(&mut b, 0), int_input(&mut b, 1));
        let fi = b.emit1(TraceInst::Int32ToF64 { value: i }, TraceType::F64, TraceRep::F64);
        let fj = b.emit1(TraceInst::Int32ToF64 { value: j }, TraceType::F64, TraceRep::F64);
        let (ten, half, nan) = (num(&mut b, 10.0), num(&mut b, 0.5), num(&mut b, f64::NAN));
        let cmp = |b: &mut TraceBuilder, lhs, rhs| b.emit1(TraceInst::F64Compare { op: CmpKind::Lt, lhs, rhs }, TraceType::Bool, TraceRep::Bool);
        let both = cmp(&mut b, fi, fj);
        let bound = cmp(&mut b, fi, ten);
        let fractional = cmp(&mut b, fi, half);
        let not_a_number = cmp(&mut b, nan, fj);
        let same = b.emit1(TraceInst::IntCompare { op: CmpKind::Le, lhs: i, rhs: i }, TraceType::Bool, TraceRep::Bool);
        let mut trace = finish(b);
        run(&mut trace, AlgebraicSimplification);
        assert!(matches!(kind_of(&trace, both), TraceInst::IntCompare { op: CmpKind::Lt, lhs, rhs } if lhs == i && rhs == j));
        let TraceInst::IntCompare { rhs, .. } = kind_of(&trace, bound) else { panic!() };
        assert_eq!(constant(&trace, rhs), Some(TraceConstant::Int32(10)));
        assert!(matches!(kind_of(&trace, fractional), TraceInst::F64Compare { .. }));
        assert!(matches!(kind_of(&trace, not_a_number), TraceInst::F64Compare { .. }));
        assert_eq!(constant(&trace, same), Some(TraceConstant::Int32(1)));
    }

    #[test]
    fn duplicate_guards_are_removed_and_results_forwarded() {
        let mut b = builder(TraceKind::Side);
        let tagged = b.read_slot(FrameSlot::General(Reg(0)), TraceType::Any, TraceRep::Tagged);
        let other = b.read_slot(FrameSlot::General(Reg(1)), TraceType::Any, TraceRep::Tagged);
        let e1 = b.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        let e2 = b.add_side_exit(ExitReason::GuardValue, BytecodePc(2));
        let first = b.emit1(TraceInst::GuardF64 { value: tagged, exit: e1 }, TraceType::F64, TraceRep::F64);
        b.emit0(TraceInst::GuardTruthy { value: tagged, exit: e1 });
        b.runtime(TraceLocation::new(FuncId(0), BytecodePc(3)), BytecodePc(4), false);
        let second = b.emit1(TraceInst::GuardF64 { value: tagged, exit: e2 }, TraceType::F64, TraceRep::F64);
        b.emit0(TraceInst::GuardTruthy { value: tagged, exit: e2 });
        let distinct = b.emit1(TraceInst::GuardF64 { value: other, exit: e2 }, TraceType::F64, TraceRep::F64);
        let as_int = b.emit1(TraceInst::GuardInt32 { value: tagged, exit: e2 }, TraceType::Int32, TraceRep::Int32);
        let mut trace = finish(b);
        run(&mut trace, RedundantGuardElimination);
        assert_eq!(trace.canonical_value(second), first);
        assert_eq!(trace.canonical_value(distinct), distinct);
        assert_eq!(trace.canonical_value(as_int), as_int);
        assert_eq!(trace.stats.redundant_guards, 2);
    }

    #[test]
    fn method_epoch_guards_dedupe_only_between_runtime_calls() {
        let mut b = builder(TraceKind::Side);
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        for _ in 0..3 { b.emit0(TraceInst::GuardMethodEpoch { expected: 5, exit }); }
        b.emit0(TraceInst::GuardMethodEpoch { expected: 6, exit });
        b.runtime(TraceLocation::new(FuncId(0), BytecodePc(3)), BytecodePc(4), false);
        b.emit0(TraceInst::GuardMethodEpoch { expected: 5, exit });
        let mut trace = finish(b);
        run(&mut trace, RedundantGuardElimination);
        let epochs = trace.nodes.iter().filter(|n| matches!(n.kind, TraceInst::GuardMethodEpoch { .. })).count();
        assert_eq!(epochs, 3);
        assert_eq!(trace.stats.redundant_guards, 2);
    }

    #[test]
    fn identical_loop_body_guards_dedupe_per_iteration() {
        let mut b = builder(TraceKind::Loop);
        let x = int_input(&mut b, 0);
        let len = int(&mut b, 16);
        let phi = b.add_loop_phi(FrameSlot::General(Reg(1)), x, TraceType::Int32, TraceRep::Int32);
        let exit = b.add_side_exit(ExitReason::GuardBounds, BytecodePc(1));
        b.emit0(TraceInst::GuardBounds { index: x, length: len, exit });
        b.emit0(TraceInst::GuardBounds { index: x, length: len, exit });
        b.emit0(TraceInst::GuardBounds { index: phi, length: len, exit });
        let one = int(&mut b, 1);
        let next = ib(&mut b, IntBinOp::Add, phi, one);
        b.set_loop_backedge(phi, next).unwrap();
        let mut trace = finish(b);
        run(&mut trace, RedundantGuardElimination);
        assert_eq!(trace.nodes.iter().filter(|n| matches!(n.kind, TraceInst::GuardBounds { .. })).count(), 2);
    }

    #[test]
    fn default_pipeline_includes_new_passes() {
        let mut b = builder(TraceKind::Side);
        let x = f64_input(&mut b, 0);
        let four = num(&mut b, 4.0);
        let q = fb(&mut b, F64BinOp::Div, x, four);
        b.write_slot(FrameSlot::General(Reg(1)), q);
        let mut trace = finish(b);
        optimization_passes().verify_after_each_pass(true).optimize_trace(&mut trace).unwrap();
        assert!(trace.stats.peephole_simplified > 0);
        assert!(matches!(kind_of(&trace, q), TraceInst::F64Binary { op: F64BinOp::Mul, .. }));
    }
}
