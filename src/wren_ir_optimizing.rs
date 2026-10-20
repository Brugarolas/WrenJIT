#![allow(dead_code)]

use crate::wren_ir::{
    CmpKind, Constant, EffectBehavior, F64BinOp, F64UnaryOp, Function, Inst, InstId, IntBinOp,
    IntUnaryOp, Type, ValueDef, ValueId,
};
use crate::wren_ir_optimizer::{FunctionOptimization, Optimize};

/// Interpreter-oriented WIR optimization pipeline.
///
/// The ordering is deliberate:
/// 1. Fold constants/control and turn algebraic identities into `Move`s.
/// 2. RAUW those moves and physically detach them.
/// 3. Remove the now-dead pure SSA producer chains.
///
/// These are whole-function passes because they naturally need global SSA-use
/// information. Node-local peepholes can still be added through
/// `OptimizationPass` in `wren_ir_optimizer`.
pub fn optimization_passes() -> Optimize {
    Optimize::new()
        .with_function_pass(ConstantFold::default())
        .with_function_pass(CopyPropagation::default())
        .with_function_pass(DeadCodeElimination::default())
}

/// Constant propagation/folding plus cheap algebraic and control-flow folding.
///
/// This pass is intentionally strict about Wren's representation split:
/// `Constant::Num` is a semantic Wren number, while `Constant::Int32` is the
/// internal integer representation. There is no fake `F64` constant; F64
/// values are only *analysed* through their producers and are folded when the
/// consumer's result can be represented correctly (for example `CmpF64` or
/// `F64ToNum`).
#[derive(Default)]
pub struct ConstantFold;

impl FunctionOptimization for ConstantFold {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed_any = false;

        // A replacement can expose a constant to a later node whose arena ID is
        // not necessarily adjacent (especially around CFG construction). Keep a
        // cheap fixed point. In practice this converges in one or two rounds.
        loop {
            let live = live_instruction_ids(function);
            let mut changed = false;

            for id in live {
                let Some(replacement) = fold_instruction(function, id) else {
                    continue;
                };

                let node = &mut function.nodes[id.0 as usize];
                debug_assert_eq!(
                    node.results.len(),
                    replacement.result_count(),
                    "constant folding must preserve SSA result arity"
                );
                debug_assert_eq!(
                    node.kind.effect_behavior(),
                    replacement.effect_behavior(),
                    "constant folding must not change effect-SSA shape"
                );
                node.kind = replacement;
                changed = true;
            }

            if !changed {
                break;
            }
            changed_any = true;
        }

        changed_any
    }
}

fn fold_instruction(function: &Function, id: InstId) -> Option<Inst> {
    use Inst::*;

    let kind = function.nodes.get(id.0 as usize)?.kind.clone();

    match kind {
        IntUnary { op, value } => {
            let value = resolve_i32(function, value, 0)?;
            fold_int_unary(op, value).map(|value| Const(Constant::Int32(value)))
        }

        IntBin { op, left, right } => {
            if left == right {
                match op {
                    IntBinOp::Sub | IntBinOp::BitXor => {
                        return Some(Const(Constant::Int32(0)));
                    }
                    IntBinOp::BitAnd | IntBinOp::BitOr => {
                        return Some(Move(left));
                    }
                    _ => {}
                }
            }

            let lhs = resolve_i32(function, left, 0);
            let rhs = resolve_i32(function, right, 0);

            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                if let Some(value) = fold_int_bin(op, lhs, rhs) {
                    return Some(Const(Constant::Int32(value)));
                }
            }

            fold_int_identity(op, left, right, lhs, rhs)
        }

        CmpInt { kind, left, right } => {
            if left == right {
                let value = match kind {
                    CmpKind::Eq | CmpKind::Le | CmpKind::Ge => true,
                    CmpKind::Ne | CmpKind::Lt | CmpKind::Gt => false,
                };
                return Some(Const(Constant::Bool(value)));
            }

            let left = resolve_i32(function, left, 0)?;
            let right = resolve_i32(function, right, 0)?;
            Some(Const(Constant::Bool(compare_i32(kind, left, right))))
        }

        CmpF64 { kind, left, right } => {
            let left = resolve_f64(function, left, 0)?;
            let right = resolve_f64(function, right, 0)?;
            Some(Const(Constant::Bool(compare_f64(kind, left, right))))
        }

        NumToInt32 { value } => {
            let value = resolve_num(function, value, 0)?;
            exact_i32(value).map(|value| Const(Constant::Int32(value)))
        }

        Int32ToNum { value } => {
            let value = resolve_i32(function, value, 0)?;
            Some(Const(Constant::Num(value as f64)))
        }

        F64ToNum { value } => {
            let value = resolve_f64(function, value, 0)?;
            Some(Const(Constant::Num(value)))
        }

        RefEq { left, right } if left == right && reflexive_ref_eq_type(function, left) => {
            Some(Const(Constant::Bool(true)))
        }

        RefEq { left, right } => {
            let left = resolve_constant(function, left, 0)?;
            let right = resolve_constant(function, right, 0)?;
            fold_ref_eq_constants(&left, &right).map(|value| Const(Constant::Bool(value)))
        }

        Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            if same_edge_payload(&then_edge, &else_edge) {
                return Some(Jump { edge: then_edge });
            }

            let constant = resolve_constant(function, cond, 0)?;
            let take_then = constant_truthiness(&constant);
            Some(Jump {
                edge: if take_then { then_edge } else { else_edge },
            })
        }

        Switch {
            tag,
            default_edge,
            cases,
        } => {
            if cases.is_empty()
                || cases
                    .iter()
                    .all(|case| same_edge_payload(&case.edge, &default_edge))
            {
                return Some(Jump { edge: default_edge });
            }

            let tag = resolve_i32(function, tag, 0)? as i64;
            let edge = cases
                .into_iter()
                .find(|case| case.value == tag)
                .map(|case| case.edge)
                .unwrap_or(default_edge);
            Some(Jump { edge })
        }

        _ => None,
    }
}

fn fold_int_identity(
    op: IntBinOp,
    left: ValueId,
    right: ValueId,
    lhs: Option<i32>,
    rhs: Option<i32>,
) -> Option<Inst> {
    use IntBinOp::*;

    match op {
        Add => {
            if rhs == Some(0) {
                Some(Inst::Move(left))
            } else if lhs == Some(0) {
                Some(Inst::Move(right))
            } else {
                None
            }
        }
        Sub => (rhs == Some(0)).then_some(Inst::Move(left)),
        Mul => {
            if rhs == Some(1) {
                Some(Inst::Move(left))
            } else if lhs == Some(1) {
                Some(Inst::Move(right))
            } else if rhs == Some(0) || lhs == Some(0) {
                Some(Inst::Const(Constant::Int32(0)))
            } else {
                None
            }
        }
        DivS => (rhs == Some(1)).then_some(Inst::Move(left)),
        RemS => (rhs == Some(1)).then_some(Inst::Const(Constant::Int32(0))),
        BitAnd => {
            if rhs == Some(-1) {
                Some(Inst::Move(left))
            } else if lhs == Some(-1) {
                Some(Inst::Move(right))
            } else if rhs == Some(0) || lhs == Some(0) {
                Some(Inst::Const(Constant::Int32(0)))
            } else {
                None
            }
        }
        BitOr | BitXor => {
            if rhs == Some(0) {
                Some(Inst::Move(left))
            } else if lhs == Some(0) {
                Some(Inst::Move(right))
            } else {
                None
            }
        }
        Shl | ShrU | Sar => (rhs == Some(0)).then_some(Inst::Move(left)),
    }
}

#[inline]
fn fold_int_unary(op: IntUnaryOp, value: i32) -> Option<i32> {
    match op {
        IntUnaryOp::Neg => value.checked_neg(),
        IntUnaryOp::BitNot => Some(!value),
    }
}

#[inline]
fn fold_int_bin(op: IntBinOp, left: i32, right: i32) -> Option<i32> {
    match op {
        IntBinOp::Add => left.checked_add(right),
        IntBinOp::Sub => left.checked_sub(right),
        IntBinOp::Mul => left.checked_mul(right),
        IntBinOp::DivS => left.checked_div(right),
        IntBinOp::RemS => left.checked_rem(right),
        IntBinOp::BitAnd => Some(left & right),
        IntBinOp::BitOr => Some(left | right),
        IntBinOp::BitXor => Some(left ^ right),
        IntBinOp::Shl if (0..32).contains(&right) => Some(left.wrapping_shl(right as u32)),
        IntBinOp::ShrU if (0..32).contains(&right) => {
            Some(((left as u32) >> (right as u32)) as i32)
        }
        IntBinOp::Sar if (0..32).contains(&right) => Some(left >> (right as u32)),
        IntBinOp::Shl | IntBinOp::ShrU | IntBinOp::Sar => None,
    }
}

#[inline]
fn compare_i32(kind: CmpKind, left: i32, right: i32) -> bool {
    match kind {
        CmpKind::Eq => left == right,
        CmpKind::Ne => left != right,
        CmpKind::Lt => left < right,
        CmpKind::Le => left <= right,
        CmpKind::Gt => left > right,
        CmpKind::Ge => left >= right,
    }
}

#[inline]
fn compare_f64(kind: CmpKind, left: f64, right: f64) -> bool {
    match kind {
        CmpKind::Eq => left == right,
        CmpKind::Ne => left != right,
        CmpKind::Lt => left < right,
        CmpKind::Le => left <= right,
        CmpKind::Gt => left > right,
        CmpKind::Ge => left >= right,
    }
}

#[inline]
fn exact_i32(value: f64) -> Option<i32> {
    if value.is_finite()
        && value.trunc() == value
        && value >= i32::MIN as f64
        && value <= i32::MAX as f64
    {
        Some(value as i32)
    } else {
        None
    }
}

fn resolve_constant(function: &Function, value: ValueId, depth: u8) -> Option<Constant> {
    if depth >= 48 {
        return None;
    }

    let value_info = function.value(value)?;
    let ValueDef::InstResult(inst, _) = value_info.def else {
        return None;
    };
    let node = function.node(inst)?;

    match &node.kind {
        Inst::Const(constant) => Some(constant.clone()),
        Inst::Move(source) => resolve_constant(function, *source, depth + 1),
        Inst::IntUnary { op, value } => {
            let value = resolve_i32(function, *value, depth + 1)?;
            fold_int_unary(*op, value).map(Constant::Int32)
        }
        Inst::IntBin { op, left, right } => {
            let left = resolve_i32(function, *left, depth + 1)?;
            let right = resolve_i32(function, *right, depth + 1)?;
            fold_int_bin(*op, left, right).map(Constant::Int32)
        }
        Inst::CmpInt { kind, left, right } => Some(Constant::Bool(compare_i32(
            *kind,
            resolve_i32(function, *left, depth + 1)?,
            resolve_i32(function, *right, depth + 1)?,
        ))),
        Inst::CmpF64 { kind, left, right } => Some(Constant::Bool(compare_f64(
            *kind,
            resolve_f64(function, *left, depth + 1)?,
            resolve_f64(function, *right, depth + 1)?,
        ))),
        Inst::Int32ToNum { value } => {
            Some(Constant::Num(resolve_i32(function, *value, depth + 1)? as f64))
        }
        Inst::F64ToNum { value } => {
            Some(Constant::Num(resolve_f64(function, *value, depth + 1)?))
        }
        Inst::NumToInt32 { value } => {
            exact_i32(resolve_num(function, *value, depth + 1)?).map(Constant::Int32)
        }
        _ => None,
    }
}

fn resolve_i32(function: &Function, value: ValueId, depth: u8) -> Option<i32> {
    if depth >= 48 {
        return None;
    }

    match resolve_constant(function, value, depth + 1)? {
        Constant::Int32(value) => Some(value),
        _ => None,
    }
}

fn resolve_num(function: &Function, value: ValueId, depth: u8) -> Option<f64> {
    if depth >= 48 {
        return None;
    }

    match resolve_constant(function, value, depth + 1)? {
        Constant::Num(value) => Some(value),
        _ => None,
    }
}

fn resolve_f64(function: &Function, value: ValueId, depth: u8) -> Option<f64> {
    if depth >= 48 {
        return None;
    }

    let value_info = function.value(value)?;
    let ValueDef::InstResult(inst, _) = value_info.def else {
        return None;
    };
    let node = function.node(inst)?;

    match &node.kind {
        Inst::Move(source) => resolve_f64(function, *source, depth + 1),
        Inst::NumToF64 { value } => resolve_num(function, *value, depth + 1),
        Inst::GuardF64 { value, .. } => resolve_num(function, *value, depth + 1),
        Inst::F64Unary {
            op: F64UnaryOp::Neg,
            value,
        } => Some(-resolve_f64(function, *value, depth + 1)?),
        Inst::F64Bin { op, left, right } => {
            let left = resolve_f64(function, *left, depth + 1)?;
            let right = resolve_f64(function, *right, depth + 1)?;
            Some(match op {
                F64BinOp::Add => left + right,
                F64BinOp::Sub => left - right,
                F64BinOp::Mul => left * right,
                F64BinOp::Div => left / right,
                F64BinOp::Rem => left % right,
            })
        }
        _ => None,
    }
}

#[inline]
fn constant_truthiness(constant: &Constant) -> bool {
    match constant {
        Constant::Null => false,
        Constant::Bool(value) => *value,
        Constant::Num(_) | Constant::String(_) | Constant::Int32(_) => true,
    }
}

#[inline]
fn same_edge_payload(left: &crate::wren_ir::Edge, right: &crate::wren_ir::Edge) -> bool {
    left.target == right.target && left.args == right.args
}

fn reflexive_ref_eq_type(function: &Function, value: ValueId) -> bool {
    function.value(value).is_some_and(|value| {
        value.ty.is_reference_like()
            || matches!(&value.ty, Type::Null | Type::Bool | Type::RawPtr)
    })
}

fn fold_ref_eq_constants(left: &Constant, right: &Constant) -> Option<bool> {
    match (left, right) {
        (Constant::Null, Constant::Null) => Some(true),
        (Constant::Null, Constant::Bool(_)) | (Constant::Bool(_), Constant::Null) => Some(false),
        (Constant::Bool(left), Constant::Bool(right)) => Some(left == right),
        _ => None,
    }
}

/// Global copy propagation / RAUW for `Move`.
///
/// This is linear in the number of live operands plus values. It canonicalizes
/// complete move chains, rewrites normal operands *and deoptimization snapshots*,
/// then detaches the redundant move instructions.
#[derive(Default)]
pub struct CopyPropagation;

impl FunctionOptimization for CopyPropagation {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let value_count = function.values.len();
        if value_count == 0 {
            return false;
        }

        let mut alias: Vec<ValueId> = (0..value_count)
            .map(|index| ValueId(index as u32))
            .collect();

        let live = live_instruction_ids(function);

        for id in live.iter().copied() {
            let node = &function.nodes[id.0 as usize];
            let Inst::Move(source) = &node.kind else {
                continue;
            };
            let [result] = node.results.as_slice() else {
                continue;
            };

            let Some(source_ty) = function.value(*source).map(|value| &value.ty) else {
                continue;
            };
            let Some(result_ty) = function.value(*result).map(|value| &value.ty) else {
                continue;
            };

            // `Move` is not a representation conversion. If malformed IR ever
            // violates this, do not silently erase the type boundary.
            if source_ty == result_ty {
                alias[result.0 as usize] = *source;
            }
        }

        // Compress all move chains before touching uses.
        for index in 0..alias.len() {
            let value = ValueId(index as u32);
            alias[index] = canonical_alias(&alias, value);
        }

        let mut changed = false;

        for id in live.iter().copied() {
            let node = &mut function.nodes[id.0 as usize];
            node.kind.for_each_operand_mut(|operand| {
                let canonical = canonical_alias(&alias, *operand);
                if canonical != *operand {
                    *operand = canonical;
                    changed = true;
                }
            });
        }

        for deopt in &mut function.deopts {
            for slot in &mut deopt.snapshot.slots {
                let canonical = canonical_alias(&alias, slot.value);
                if canonical != slot.value {
                    slot.value = canonical;
                    changed = true;
                }
            }
            for frame in &mut deopt.snapshot.inlined_frames {
                for slot in &mut frame.slots {
                    let canonical = canonical_alias(&alias, slot.value);
                    if canonical != slot.value {
                        slot.value = canonical;
                        changed = true;
                    }
                }
            }
        }

        let mut removable = vec![false; function.nodes.len()];
        for id in live {
            let node = &function.nodes[id.0 as usize];
            if !matches!(&node.kind, Inst::Move(_)) {
                continue;
            }
            let [result] = node.results.as_slice() else {
                continue;
            };
            if canonical_alias(&alias, *result) != *result {
                removable[id.0 as usize] = true;
            }
        }

        if removable.iter().any(|removed| *removed) {
            for block in &mut function.blocks {
                block.insts.retain(|id| !removable[id.0 as usize]);
            }
            function
                .profiles
                .numeric_sites
                .retain(|id, _| !removable.get(id.0 as usize).copied().unwrap_or(false));
            changed = true;
        }

        changed
    }
}

fn canonical_alias(alias: &[ValueId], start: ValueId) -> ValueId {
    let mut current = start;

    // SSA move chains are acyclic. The bound makes this helper robust against
    // malformed input without allocating a visited set in this hot compile path.
    for _ in 0..=alias.len() {
        let Some(next) = alias.get(current.0 as usize).copied() else {
            return current;
        };
        if next == current {
            return current;
        }
        current = next;
    }

    // A cycle means malformed SSA. Refuse to propagate through it rather than
    // detaching a move whose replacement cannot be proven canonical.
    start
}

/// Cascading dead-code elimination for effect-free SSA producers.
///
/// Uses in deoptimization snapshots are roots. Removing one dead instruction
/// decrements the use counts of its operands and can immediately make its
/// producers dead, so a whole chain disappears in one linear worklist pass.
#[derive(Default)]
pub struct DeadCodeElimination;

impl FunctionOptimization for DeadCodeElimination {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        if function.nodes.is_empty() {
            return false;
        }

        let mut live = vec![false; function.nodes.len()];
        let mut uses = vec![0usize; function.values.len()];

        for block in &function.blocks {
            for id in block.insts.iter().copied() {
                live[id.0 as usize] = true;
                for operand in function.nodes[id.0 as usize].kind.operands() {
                    if let Some(count) = uses.get_mut(operand.0 as usize) {
                        *count += 1;
                    }
                }
            }
        }

        // Deopt state is semantically observable on bailout and must keep its
        // referenced SSA values alive.
        for deopt in &function.deopts {
            for slot in &deopt.snapshot.slots {
                if let Some(count) = uses.get_mut(slot.value.0 as usize) {
                    *count += 1;
                }
            }
            for frame in &deopt.snapshot.inlined_frames {
                for slot in &frame.slots {
                    if let Some(count) = uses.get_mut(slot.value.0 as usize) {
                        *count += 1;
                    }
                }
            }
        }

        let mut worklist = Vec::new();
        for id in live_instruction_ids(function) {
            if dce_candidate(function, id, &uses) {
                worklist.push(id);
            }
        }

        let mut changed = false;

        while let Some(id) = worklist.pop() {
            let index = id.0 as usize;
            if !live.get(index).copied().unwrap_or(false)
                || !dce_candidate(function, id, &uses)
            {
                continue;
            }

            live[index] = false;
            changed = true;

            let operands = function.nodes[index].kind.operands();
            for operand in operands {
                let operand_index = operand.0 as usize;
                let Some(count) = uses.get_mut(operand_index) else {
                    continue;
                };
                debug_assert!(*count > 0, "SSA use count underflow for {operand:?}");
                if *count == 0 {
                    continue;
                }
                *count -= 1;

                if *count != 0 {
                    continue;
                }

                let Some(ValueDef::InstResult(producer, _)) =
                    function.value(operand).map(|value| value.def)
                else {
                    continue;
                };

                if live
                    .get(producer.0 as usize)
                    .copied()
                    .unwrap_or(false)
                    && dce_candidate(function, producer, &uses)
                {
                    worklist.push(producer);
                }
            }
        }

        if !changed {
            return false;
        }

        for block in &mut function.blocks {
            block
                .insts
                .retain(|id| live.get(id.0 as usize).copied().unwrap_or(false));
        }

        function
            .profiles
            .numeric_sites
            .retain(|id, _| live.get(id.0 as usize).copied().unwrap_or(false));

        true
    }
}

fn dce_candidate(function: &Function, id: InstId, uses: &[usize]) -> bool {
    let Some(node) = function.node(id) else {
        return false;
    };

    if node.kind.is_terminator()
        || !matches!(node.kind.effect_behavior(), EffectBehavior::None)
        || node.results.is_empty()
    {
        return false;
    }

    node.results.iter().all(|value| {
        uses.get(value.0 as usize)
            .copied()
            .unwrap_or(usize::MAX)
            == 0
    })
}

fn live_instruction_ids(function: &Function) -> Vec<InstId> {
    let capacity = function
        .blocks
        .iter()
        .map(|block| block.insts.len())
        .sum();
    let mut ids = Vec::with_capacity(capacity);

    for block in &function.blocks {
        ids.extend_from_slice(&block.insts);
    }

    ids
}
