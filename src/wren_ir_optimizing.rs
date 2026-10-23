#![allow(dead_code)]

use hashbrown::{HashMap, HashSet};

use crate::wren_ir::{
    BlockId, ClassSet, CmpKind, Constant, Edge, EffectBehavior, EffectKind, F64BinOp, F64UnaryOp,
    FieldSlot, Function, Inst, InstId, IntBinOp, IntUnaryOp, ModuleId, ModuleSlot, Node, Program, SiteId,
    StaticFieldSlot, StringId, Type, ValueDef, ValueId,
};
use crate::wren_ir_escape::ScalarReplacement;
use crate::wren_ir_optimizer::{repair_effect_chain, FunctionOptimization, Optimize};

/// Interpreter-oriented WIR optimization pipeline.
///
/// The default stage combines bounded scalar folding, local value numbering,
/// memory forwarding/DSE, guard cleanup, copy propagation, DCE and cheap CFG
/// cleanup. The implementation intentionally avoids dominator trees, PRE, LICM
/// and other analyses whose cold-start cost belongs in the later JIT tier.
pub fn optimization_passes() -> Optimize {
    // Keep the default pipeline as a single optimizer stage. In debug builds the
    // framework verifies after every stage, so bundling the cheap transforms also
    // avoids repeatedly paying verifier cost during development.
    Optimize::new().with_function_pass(InterpreterOptimizationBundle::default())
}

/// Optimize independent WIR functions concurrently using one optimizer instance
/// per worker. `worker_count == 0` selects the host's available parallelism.
///
/// The default WIR pipeline is deliberately function-local, so workers never
/// share mutable optimizer state or mutate global `Program` metadata. This keeps
/// the hot optimizer data structures thread-local and preserves deterministic
/// function ordering in `Program::functions`.
pub fn optimize_program_parallel(program: &mut Program, worker_count: usize) {
    let function_count = program.functions.len();
    let total_nodes = program
        .functions
        .iter()
        .map(|function| function.nodes.len())
        .sum::<usize>();
    if function_count <= 1 {
        optimization_passes().optimize_program(program);
        return;
    }

    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    // Auto mode avoids creating a thread per tiny function. Roughly one worker
    // per 1K HIR nodes gives EA/SROA enough work to amortize thread startup while
    // still exploiting a few very large functions. Explicit counts are honored.
    let workers = if worker_count == 0 {
        let work_limited = total_nodes.saturating_add(1023) / 1024;
        available.min(work_limited.max(1))
    } else {
        worker_count
    }
    .max(1)
    .min(function_count);

    if workers == 1 {
        optimization_passes().optimize_program(program);
        return;
    }

    // WIR function costs are uneven (EA/SROA can make one function much more
    // expensive than its neighbours), so use a tiny synchronized work queue
    // instead of fixed contiguous chunks. Each `&mut Function` is produced once
    // by `iter_mut`, preserving exclusive ownership without unsafe code.
    let (work_tx, work_rx) = std::sync::mpsc::channel::<&mut Function>();
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let work_rx = std::sync::Arc::clone(&work_rx);
            scope.spawn(move || {
                let mut optimizer = optimization_passes();
                loop {
                    let function = {
                        let receiver = work_rx.lock().expect("WIR optimizer queue poisoned");
                        receiver.recv()
                    };
                    let Ok(function) = function else {
                        break;
                    };
                    optimizer.optimize_function(function);
                }
            });
        }

        for function in &mut program.functions {
            work_tx
                .send(function)
                .expect("all WIR optimizer workers stopped unexpectedly");
        }
        drop(work_tx);
    });
}

/// Cold-start-conscious optimizer bundle for interpreter-visible WIR.
///
/// The expensive optimizations deliberately live elsewhere (JIT/trace IR). This
/// bundle sticks to bounded local analyses and linear whole-function cleanups.
#[derive(Default)]
pub struct InterpreterOptimizationBundle;

impl FunctionOptimization for InterpreterOptimizationBundle {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed_any = false;
        let mut simplify = FastLocalSimplify::default();
        let mut copies = CopyPropagation::default();

        // Two rounds are enough to expose almost all copy/fold/CSE chains while
        // giving cold start a strict upper bound.
        for _ in 0..2 {
            let simplified = simplify.optimize_function(function);
            let propagated = copies.optimize_function(function);
            changed_any |= simplified | propagated;
            if !simplified && !propagated {
                break;
            }
        }

        let mut dce = DeadCodeElimination::default();
        changed_any |= dce.optimize_function(function);

        let mut cfg = ControlFlowCleanup::default();
        let cfg_changed = cfg.optimize_function(function);
        changed_any |= cfg_changed;

        // CFG threading can make block parameters trivially invariant. One final
        // copy/DCE cleanup captures that win without another full fixed point.
        if cfg_changed {
            let propagated = copies.optimize_function(function);
            changed_any |= propagated;
            if propagated {
                changed_any |= dce.optimize_function(function);
            }
        }

        // Partial escape analysis + scalar replacement runs on the cleaned CFG.
        // Virtualized fields turn into plain SSA values, which exposes new
        // folding/copy/DCE opportunities, so re-run the cheap cleanups once.
        let mut sroa = ScalarReplacement::default();
        if sroa.optimize_function(function) {
            changed_any = true;
            simplify.optimize_function(function);
            copies.optimize_function(function);
            dce.optimize_function(function);
        }

        changed_any
    }
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

        // Keep this public standalone pass bounded as well. The default bundle
        // performs its own bounded revisit after copy propagation.
        for _ in 0..2 {
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
            if op == IntUnaryOp::BitNot {
                if let Some(Inst::IntUnary {
                    op: IntUnaryOp::BitNot,
                    value: inner,
                }) = producer_kind(function, value)
                {
                    return Some(Move(*inner));
                }
            }

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

            let lhs = resolve_i32(function, left, 0);
            let rhs = resolve_i32(function, right, 0);
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                return Some(Const(Constant::Bool(compare_i32(kind, lhs, rhs))));
            }
            fold_cmp_int_bounds(kind, lhs, rhs).map(|value| Const(Constant::Bool(value)))
        }

        CmpF64 { kind, left, right } => {
            let left = resolve_f64(function, left, 0)?;
            let right = resolve_f64(function, right, 0)?;
            Some(Const(Constant::Bool(compare_f64(kind, left, right))))
        }

        F64Unary {
            op: F64UnaryOp::Neg,
            value,
        } => {
            if let Some(Inst::F64Unary {
                op: F64UnaryOp::Neg,
                value: inner,
            }) = producer_kind(function, value)
            {
                return Some(Move(*inner));
            }
            None
        }

        NumToInt32 { value } => {
            if let Some(Inst::Int32ToNum { value: inner }) = producer_kind(function, value) {
                return Some(Move(*inner));
            }
            let value = resolve_num(function, value, 0)?;
            exact_i32(value).map(|value| Const(Constant::Int32(value)))
        }

        NumToF64 { value } => {
            if let Some(Inst::F64ToNum { value: inner }) = producer_kind(function, value) {
                return Some(Move(*inner));
            }
            None
        }

        Int32ToNum { value } => {
            let value = resolve_i32(function, value, 0)?;
            Some(Const(Constant::Num(value as f64)))
        }

        F64ToNum { value } => {
            if let Some(Inst::NumToF64 { value: inner }) = producer_kind(function, value) {
                return Some(Move(*inner));
            }
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
        Sub => {
            if rhs == Some(0) {
                Some(Inst::Move(left))
            } else if lhs == Some(0) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::Neg,
                    value: right,
                })
            } else {
                None
            }
        }
        Mul => {
            if rhs == Some(1) {
                Some(Inst::Move(left))
            } else if lhs == Some(1) {
                Some(Inst::Move(right))
            } else if rhs == Some(0) || lhs == Some(0) {
                Some(Inst::Const(Constant::Int32(0)))
            } else if rhs == Some(-1) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::Neg,
                    value: left,
                })
            } else if lhs == Some(-1) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::Neg,
                    value: right,
                })
            } else {
                None
            }
        }
        DivS => {
            if rhs == Some(1) {
                Some(Inst::Move(left))
            } else if rhs == Some(-1) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::Neg,
                    value: left,
                })
            } else {
                None
            }
        }
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
        BitOr => {
            if rhs == Some(0) {
                Some(Inst::Move(left))
            } else if lhs == Some(0) {
                Some(Inst::Move(right))
            } else if rhs == Some(-1) || lhs == Some(-1) {
                Some(Inst::Const(Constant::Int32(-1)))
            } else {
                None
            }
        }
        BitXor => {
            if rhs == Some(0) {
                Some(Inst::Move(left))
            } else if lhs == Some(0) {
                Some(Inst::Move(right))
            } else if rhs == Some(-1) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::BitNot,
                    value: left,
                })
            } else if lhs == Some(-1) {
                Some(Inst::IntUnary {
                    op: IntUnaryOp::BitNot,
                    value: right,
                })
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

fn fold_cmp_int_bounds(kind: CmpKind, lhs: Option<i32>, rhs: Option<i32>) -> Option<bool> {
    match (kind, lhs, rhs) {
        (CmpKind::Lt, _, Some(i32::MIN)) => Some(false),
        (CmpKind::Ge, _, Some(i32::MIN)) => Some(true),
        (CmpKind::Gt, _, Some(i32::MAX)) => Some(false),
        (CmpKind::Le, _, Some(i32::MAX)) => Some(true),
        (CmpKind::Gt, Some(i32::MIN), _) => Some(false),
        (CmpKind::Le, Some(i32::MIN), _) => Some(true),
        (CmpKind::Lt, Some(i32::MAX), _) => Some(false),
        (CmpKind::Ge, Some(i32::MAX), _) => Some(true),
        _ => None,
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

fn producer_kind(function: &Function, value: ValueId) -> Option<&Inst> {
    let ValueDef::InstResult(inst, _) = function.value(value)?.def else {
        return None;
    };
    function.node(inst).map(|node| &node.kind)
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
        (Constant::Bool(left), Constant::Bool(right)) => Some(left == right),
        (Constant::Null, _) | (_, Constant::Null) => Some(false),
        (Constant::Bool(_), Constant::Num(_) | Constant::String(_) | Constant::Int32(_))
        | (Constant::Num(_) | Constant::String(_) | Constant::Int32(_), Constant::Bool(_)) => {
            Some(false)
        }
        _ => None,
    }
}


/// One bounded block-local sweep that combines scalar folding, guard cleanup,
/// local value numbering, load forwarding and conservative dead-store removal.
#[derive(Default)]
pub struct FastLocalSimplify;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PureKey {
    Null,
    Bool(bool),
    Num(u64),
    String(StringId),
    Int32(i32),
    IntUnary(IntUnaryOp, ValueId),
    IntBin(IntBinOp, ValueId, ValueId),
    F64Unary(F64UnaryOp, ValueId),
    F64Bin(F64BinOp, ValueId, ValueId),
    CmpInt(CmpKind, ValueId, ValueId),
    CmpF64(CmpKind, ValueId, ValueId),
    RefEq(ValueId, ValueId),
    NumToInt32(ValueId),
    NumToF64(ValueId),
    Int32ToNum(ValueId),
    F64ToNum(ValueId),
    IsInstance(ValueId, ValueId),
    StringLen(ValueId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GuardKey {
    Type(ValueId, Type),
    Class(ValueId, ClassSet),
    NotNull(ValueId),
    Int32(ValueId),
    F64(ValueId),
    Bounds(ValueId, ValueId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SeqReadKey {
    ListLen(ValueId),
    ListGet(ValueId, ValueId),
    MapLen(ValueId),
    MapGet(ValueId, ValueId),
    StringCodePointAt(ValueId, ValueId),
}

type FieldKey = (ValueId, FieldSlot);
type StaticFieldKey = (ValueId, StaticFieldSlot);
type ModuleKey = (ModuleId, ModuleSlot);

impl FunctionOptimization for FastLocalSimplify {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        if function.blocks.is_empty() {
            return false;
        }

        let mut changed = false;
        let mut removed = vec![false; function.nodes.len()];

        for block_ix in 0..function.blocks.len() {
            let insts = function.blocks[block_ix].insts.clone();
            if insts.is_empty() {
                continue;
            }

            let mut pure_values: HashMap<PureKey, ValueId> = HashMap::new();
            let mut guards: HashMap<GuardKey, Option<ValueId>> = HashMap::new();

            let mut field_values: HashMap<FieldKey, ValueId> = HashMap::new();
            let mut static_values: HashMap<StaticFieldKey, ValueId> = HashMap::new();
            let mut cell_values: HashMap<ValueId, ValueId> = HashMap::new();
            let mut module_values: HashMap<ModuleKey, ValueId> = HashMap::new();
            let mut list_values: HashMap<SeqReadKey, ValueId> = HashMap::new();
            let mut map_values: HashMap<SeqReadKey, ValueId> = HashMap::new();
            let mut string_values: HashMap<SeqReadKey, ValueId> = HashMap::new();

            let mut last_field_store: HashMap<FieldKey, InstId> = HashMap::new();
            let mut last_static_store: HashMap<StaticFieldKey, InstId> = HashMap::new();
            let mut last_cell_store: HashMap<ValueId, InstId> = HashMap::new();
            let mut last_module_store: HashMap<ModuleKey, InstId> = HashMap::new();

            let mut effect_shape_changed = false;

            for id in insts {
                if removed.get(id.0 as usize).copied().unwrap_or(false) {
                    continue;
                }

                // Scalar/control folding is deliberately done in the same sweep as
                // CSE and memory simplification to minimize front-end passes.
                if let Some(replacement) = fold_instruction(function, id) {
                    let node = &mut function.nodes[id.0 as usize];
                    debug_assert_eq!(node.results.len(), replacement.result_count());
                    debug_assert_eq!(node.kind.effect_behavior(), replacement.effect_behavior());
                    node.kind = replacement;
                    changed = true;
                }

                let mut kind = function.nodes[id.0 as usize].kind.clone();

                // Prove guards from static WIR types or an identical dominating
                // guard earlier in the same block. Replacing a value-producing
                // guard by Move keeps its SSA result ID stable.
                if let Some(replacement) = redundant_guard_replacement(function, id, &kind, &guards) {
                    match replacement {
                        GuardReplacement::Move(value) => {
                            function.nodes[id.0 as usize].kind = Inst::Move(value);
                        }
                        GuardReplacement::Erase => {
                            removed[id.0 as usize] = true;
                        }
                    }
                    changed = true;
                    effect_shape_changed = true;
                    continue;
                }

                if let Some((key, result)) = guard_fact(function, id, &kind) {
                    guards.entry(key).or_insert(result);
                }

                kind = function.nodes[id.0 as usize].kind.clone();

                // Block-local CSE is dominance-free by construction: an earlier
                // instruction in the same basic block dominates the later one.
                if let Some(key) = pure_key(&kind) {
                    if let Some(result) = single_result(function, id) {
                        if let Some(existing) = pure_values.get(&key).copied() {
                            function.nodes[id.0 as usize].kind = Inst::Move(existing);
                            changed = true;
                            continue;
                        }
                        pure_values.insert(key, result);
                    }
                }

                let effect_kind = kind.effect_kind();
                match kind {
                    Inst::LoadField { object, slot } => {
                        // Any field load can observe a prior store through an alias.
                        last_field_store.clear();
                        let key = (object, slot);
                        if let Some(value) = field_values.get(&key).copied() {
                            function.nodes[id.0 as usize].kind = Inst::Move(value);
                            changed = true;
                            effect_shape_changed = true;
                        } else if let Some(result) = single_result(function, id) {
                            field_values.insert(key, result);
                        }
                    }
                    Inst::StoreField { object, slot, value } => {
                        let key = (object, slot);
                        if let Some(previous) = last_field_store.insert(key, id) {
                            removed[previous.0 as usize] = true;
                            changed = true;
                            effect_shape_changed = true;
                        }
                        // Different SSA values may still alias the same object.
                        field_values.clear();
                        field_values.insert(key, value);
                    }
                    Inst::LoadStaticField { class, slot } => {
                        last_static_store.clear();
                        let key = (class, slot);
                        if let Some(value) = static_values.get(&key).copied() {
                            function.nodes[id.0 as usize].kind = Inst::Move(value);
                            changed = true;
                            effect_shape_changed = true;
                        } else if let Some(result) = single_result(function, id) {
                            static_values.insert(key, result);
                        }
                    }
                    Inst::StoreStaticField { class, slot, value } => {
                        let key = (class, slot);
                        if let Some(previous) = last_static_store.insert(key, id) {
                            removed[previous.0 as usize] = true;
                            changed = true;
                            effect_shape_changed = true;
                        }
                        static_values.clear();
                        static_values.insert(key, value);
                    }
                    Inst::LoadCell { cell } => {
                        last_cell_store.clear();
                        if let Some(value) = cell_values.get(&cell).copied() {
                            function.nodes[id.0 as usize].kind = Inst::Move(value);
                            changed = true;
                            effect_shape_changed = true;
                        } else if let Some(result) = single_result(function, id) {
                            cell_values.insert(cell, result);
                        }
                    }
                    Inst::StoreCell { cell, value } => {
                        if let Some(previous) = last_cell_store.insert(cell, id) {
                            removed[previous.0 as usize] = true;
                            changed = true;
                            effect_shape_changed = true;
                        }
                        cell_values.clear();
                        cell_values.insert(cell, value);
                    }
                    Inst::LoadModuleVar { module, slot } => {
                        let key = (module, slot);
                        last_module_store.remove(&key);
                        if let Some(value) = module_values.get(&key).copied() {
                            function.nodes[id.0 as usize].kind = Inst::Move(value);
                            changed = true;
                            effect_shape_changed = true;
                        } else if let Some(result) = single_result(function, id) {
                            module_values.insert(key, result);
                        }
                    }
                    Inst::StoreModuleVar { module, slot, value } => {
                        let key = (module, slot);
                        if let Some(previous) = last_module_store.insert(key, id) {
                            removed[previous.0 as usize] = true;
                            changed = true;
                            effect_shape_changed = true;
                        }
                        module_values.insert(key, value);
                    }
                    Inst::ListLen { list } => {
                        cse_read(
                            function,
                            id,
                            SeqReadKey::ListLen(list),
                            &mut list_values,
                            &mut changed,
                            &mut effect_shape_changed,
                        );
                    }
                    Inst::ListGet { list, index } => {
                        cse_read(
                            function,
                            id,
                            SeqReadKey::ListGet(list, index),
                            &mut list_values,
                            &mut changed,
                            &mut effect_shape_changed,
                        );
                    }
                    Inst::ListSet { list, index, value } => {
                        list_values.clear();
                        list_values.insert(SeqReadKey::ListGet(list, index), value);
                    }
                    Inst::ListAppend { .. }
                    | Inst::ListInsert { .. }
                    | Inst::ListRemoveAt { .. } => {
                        list_values.clear();
                    }
                    Inst::MapLen { map } => {
                        cse_read(
                            function,
                            id,
                            SeqReadKey::MapLen(map),
                            &mut map_values,
                            &mut changed,
                            &mut effect_shape_changed,
                        );
                    }
                    Inst::MapGet { map, key } => {
                        cse_read(
                            function,
                            id,
                            SeqReadKey::MapGet(map, key),
                            &mut map_values,
                            &mut changed,
                            &mut effect_shape_changed,
                        );
                    }
                    Inst::MapSet { map, key, value } => {
                        map_values.clear();
                        map_values.insert(SeqReadKey::MapGet(map, key), value);
                    }
                    Inst::MapRemove { .. } => {
                        map_values.clear();
                    }
                    Inst::StringCodePointAt { string, index } => {
                        cse_read(
                            function,
                            id,
                            SeqReadKey::StringCodePointAt(string, index),
                            &mut string_values,
                            &mut changed,
                            &mut effect_shape_changed,
                        );
                    }
                    _ => match effect_kind {
                        Some(
                            EffectKind::Call
                            | EffectKind::Foreign
                            | EffectKind::Fiber
                            | EffectKind::Class
                            | EffectKind::Module,
                        ) => {
                            invalidate_memory_values(
                                &mut field_values,
                                &mut static_values,
                                &mut cell_values,
                                &mut module_values,
                                &mut list_values,
                                &mut map_values,
                            );
                            clear_last_stores(
                                &mut last_field_store,
                                &mut last_static_store,
                                &mut last_cell_store,
                                &mut last_module_store,
                            );
                        }
                        Some(
                            EffectKind::Alloc
                            | EffectKind::Guard
                            | EffectKind::Safepoint
                            | EffectKind::Barrier,
                        ) => {
                            // Allocation/guard/safepoint can make prior stores
                            // observable through GC or deoptimization state, but do
                            // not mutate ordinary Wren memory by themselves.
                            clear_last_stores(
                                &mut last_field_store,
                                &mut last_static_store,
                                &mut last_cell_store,
                                &mut last_module_store,
                            );
                        }
                        _ => {}
                    },
                }
            }

            if removed.iter().any(|removed| *removed) {
                function.blocks[block_ix]
                    .insts
                    .retain(|id| !removed.get(id.0 as usize).copied().unwrap_or(false));
            }

            if effect_shape_changed {
                repair_effect_chain(function, block_ix);
            }
        }

        if changed {
            function
                .profiles
                .numeric_sites
                .retain(|id, _| !removed.get(id.0 as usize).copied().unwrap_or(false));
        }

        changed
    }
}

#[derive(Clone, Copy)]
enum GuardReplacement {
    Move(ValueId),
    Erase,
}

fn redundant_guard_replacement(
    function: &Function,
    _id: InstId,
    kind: &Inst,
    guards: &HashMap<GuardKey, Option<ValueId>>,
) -> Option<GuardReplacement> {
    let key = match kind {
        Inst::GuardType { value, expected, .. } => {
            if function
                .value(*value)
                .is_some_and(|actual| expected.accepts(&actual.ty))
            {
                return Some(GuardReplacement::Move(*value));
            }
            GuardKey::Type(*value, expected.clone())
        }
        Inst::GuardClass { value, expected, .. } => {
            if function
                .value(*value)
                .is_some_and(|actual| class_guard_known(&actual.ty, expected))
            {
                return Some(GuardReplacement::Move(*value));
            }
            GuardKey::Class(*value, expected.clone())
        }
        Inst::GuardNotNull { value, .. } => {
            if function
                .value(*value)
                .is_some_and(|value| type_definitely_non_null(&value.ty))
            {
                return Some(GuardReplacement::Move(*value));
            }
            GuardKey::NotNull(*value)
        }
        Inst::GuardInt32 { value, .. } => {
            if function
                .value(*value)
                .is_some_and(|value| matches!(&value.ty, Type::Int32))
            {
                return Some(GuardReplacement::Move(*value));
            }
            GuardKey::Int32(*value)
        }
        Inst::GuardF64 { value, .. } => {
            if function
                .value(*value)
                .is_some_and(|value| matches!(&value.ty, Type::F64))
            {
                return Some(GuardReplacement::Move(*value));
            }
            GuardKey::F64(*value)
        }
        Inst::GuardBounds { index, length, .. } => {
            if let (Some(index), Some(length)) = (
                resolve_i32(function, *index, 0),
                resolve_i32(function, *length, 0),
            ) {
                if index >= 0 && index < length {
                    return Some(GuardReplacement::Erase);
                }
            }
            GuardKey::Bounds(*index, *length)
        }
        _ => return None,
    };

    guards.get(&key).and_then(|result| match result {
        Some(value) => Some(GuardReplacement::Move(*value)),
        None => Some(GuardReplacement::Erase),
    })
}

fn guard_fact(function: &Function, id: InstId, kind: &Inst) -> Option<(GuardKey, Option<ValueId>)> {
    let result = single_result(function, id);
    match kind {
        Inst::GuardType { value, expected, .. } => {
            Some((GuardKey::Type(*value, expected.clone()), result))
        }
        Inst::GuardClass { value, expected, .. } => {
            Some((GuardKey::Class(*value, expected.clone()), result))
        }
        Inst::GuardNotNull { value, .. } => Some((GuardKey::NotNull(*value), result)),
        Inst::GuardInt32 { value, .. } => Some((GuardKey::Int32(*value), result)),
        Inst::GuardF64 { value, .. } => Some((GuardKey::F64(*value), result)),
        Inst::GuardBounds { index, length, .. } => {
            Some((GuardKey::Bounds(*index, *length), None))
        }
        _ => None,
    }
}

fn class_guard_known(actual: &Type, expected: &ClassSet) -> bool {
    if expected.is_unknown() {
        return true;
    }
    match actual {
        Type::Class(actual) | Type::Instance(actual) if !actual.is_unknown() => actual
            .as_slice()
            .iter()
            .all(|class| expected.contains(*class)),
        Type::Union(types) => types.iter().all(|ty| class_guard_known(ty, expected)),
        Type::Bottom => true,
        _ => false,
    }
}

fn type_definitely_non_null(ty: &Type) -> bool {
    match ty {
        Type::Bottom => true,
        Type::Top | Type::Null | Type::RawPtr => false,
        Type::Union(types) => !types.is_empty() && types.iter().all(type_definitely_non_null),
        _ => true,
    }
}

fn single_result(function: &Function, id: InstId) -> Option<ValueId> {
    let node = function.node(id)?;
    match node.results.as_slice() {
        [result] => Some(*result),
        _ => None,
    }
}

fn pure_key(kind: &Inst) -> Option<PureKey> {
    match kind {
        Inst::Const(Constant::Null) => Some(PureKey::Null),
        Inst::Const(Constant::Bool(value)) => Some(PureKey::Bool(*value)),
        Inst::Const(Constant::Num(value)) => Some(PureKey::Num(value.to_bits())),
        Inst::Const(Constant::String(value)) => Some(PureKey::String(*value)),
        Inst::Const(Constant::Int32(value)) => Some(PureKey::Int32(*value)),
        Inst::IntUnary { op, value } => Some(PureKey::IntUnary(*op, *value)),
        Inst::IntBin { op, left, right } => {
            let (left, right) = if matches!(
                *op,
                IntBinOp::Add | IntBinOp::Mul | IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor
            ) {
                ordered_pair(*left, *right)
            } else {
                (*left, *right)
            };
            Some(PureKey::IntBin(*op, left, right))
        }
        Inst::F64Unary { op, value } => Some(PureKey::F64Unary(*op, *value)),
        Inst::F64Bin { op, left, right } => Some(PureKey::F64Bin(*op, *left, *right)),
        Inst::CmpInt { kind, left, right } => {
            let (left, right) = if matches!(*kind, CmpKind::Eq | CmpKind::Ne) {
                ordered_pair(*left, *right)
            } else {
                (*left, *right)
            };
            Some(PureKey::CmpInt(*kind, left, right))
        }
        Inst::CmpF64 { kind, left, right } => {
            let (left, right) = if matches!(*kind, CmpKind::Eq | CmpKind::Ne) {
                ordered_pair(*left, *right)
            } else {
                (*left, *right)
            };
            Some(PureKey::CmpF64(*kind, left, right))
        }
        Inst::RefEq { left, right } => {
            let (left, right) = ordered_pair(*left, *right);
            Some(PureKey::RefEq(left, right))
        }
        Inst::NumToInt32 { value } => Some(PureKey::NumToInt32(*value)),
        Inst::NumToF64 { value } => Some(PureKey::NumToF64(*value)),
        Inst::Int32ToNum { value } => Some(PureKey::Int32ToNum(*value)),
        Inst::F64ToNum { value } => Some(PureKey::F64ToNum(*value)),
        Inst::IsInstance { value, class } => Some(PureKey::IsInstance(*value, *class)),
        Inst::StringLen { string } => Some(PureKey::StringLen(*string)),
        _ => None,
    }
}

#[inline]
fn ordered_pair(left: ValueId, right: ValueId) -> (ValueId, ValueId) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn cse_read(
    function: &mut Function,
    id: InstId,
    key: SeqReadKey,
    values: &mut HashMap<SeqReadKey, ValueId>,
    changed: &mut bool,
    effect_shape_changed: &mut bool,
) {
    if let Some(value) = values.get(&key).copied() {
        function.nodes[id.0 as usize].kind = Inst::Move(value);
        *changed = true;
        *effect_shape_changed = true;
    } else if let Some(result) = single_result(function, id) {
        values.insert(key, result);
    }
}

fn invalidate_memory_values(
    fields: &mut HashMap<FieldKey, ValueId>,
    statics: &mut HashMap<StaticFieldKey, ValueId>,
    cells: &mut HashMap<ValueId, ValueId>,
    modules: &mut HashMap<ModuleKey, ValueId>,
    lists: &mut HashMap<SeqReadKey, ValueId>,
    maps: &mut HashMap<SeqReadKey, ValueId>,
) {
    fields.clear();
    statics.clear();
    cells.clear();
    modules.clear();
    lists.clear();
    maps.clear();
}

fn clear_last_stores(
    fields: &mut HashMap<FieldKey, InstId>,
    statics: &mut HashMap<StaticFieldKey, InstId>,
    cells: &mut HashMap<ValueId, InstId>,
    modules: &mut HashMap<ModuleKey, InstId>,
) {
    fields.clear();
    statics.clear();
    cells.clear();
    modules.clear();
}

/// Cheap CFG cleanup: bounded jump threading, switch cleanup and unreachable
/// block compaction. Blocks keep stable IDs; unreachable blocks become a single
/// `Unreachable` terminator instead of forcing global ID renumbering.
#[derive(Default)]
pub struct ControlFlowCleanup;

impl FunctionOptimization for ControlFlowCleanup {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed = false;

        for block_ix in 0..function.blocks.len() {
            let Some(id) = function.blocks[block_ix].insts.last().copied() else {
                continue;
            };
            let old = function.nodes[id.0 as usize].kind.clone();
            if let Some(new_kind) = simplify_terminator(function, old) {
                function.nodes[id.0 as usize].kind = new_kind;
                changed = true;
            }
        }

        let reachable = compute_reachable(function);
        let mut compacted = false;

        for block_ix in 0..function.blocks.len() {
            if reachable.get(block_ix).copied().unwrap_or(false) {
                continue;
            }

            let already_unreachable = function.blocks[block_ix].insts.len() == 1
                && matches!(
                    &function.nodes[function.blocks[block_ix].insts[0].0 as usize].kind,
                    Inst::Unreachable
                );
            if already_unreachable {
                continue;
            }

            let source = function.blocks[block_ix]
                .insts
                .first()
                .and_then(|id| function.node(*id))
                .and_then(|node| node.source);
            let id = InstId(function.nodes.len() as u32);
            function.nodes.push(Node {
                id,
                kind: Inst::Unreachable,
                results: Vec::new(),
                effect_in: Some(function.blocks[block_ix].effect),
                effect_out: None,
                source,
            });
            function.blocks[block_ix].insts.clear();
            function.blocks[block_ix].insts.push(id);
            compacted = true;
        }

        if compacted {
            changed = true;
        }

        if changed {
            let live = live_node_bitmap(function);
            function
                .profiles
                .numeric_sites
                .retain(|id, _| live.get(id.0 as usize).copied().unwrap_or(false));
            function.profiles.block_counters.retain(|block, _| {
                reachable
                    .get(block.0 as usize)
                    .copied()
                    .unwrap_or(false)
            });

            let mut live_sites: HashSet<SiteId> = HashSet::new();
            for block in &function.blocks {
                for id in block.insts.iter().copied() {
                    match &function.nodes[id.0 as usize].kind {
                        Inst::Invoke { site, .. } | Inst::InvokeSuper { site, .. } => {
                            live_sites.insert(*site);
                        }
                        _ => {}
                    }
                }
            }
            function
                .profiles
                .call_sites
                .retain(|site, _| live_sites.contains(site));
        }

        changed
    }
}

fn simplify_terminator(function: &Function, kind: Inst) -> Option<Inst> {
    match kind {
        Inst::Jump { mut edge } => {
            if thread_edge(function, &mut edge) {
                Some(Inst::Jump { edge })
            } else {
                None
            }
        }
        Inst::Branch {
            cond,
            mut then_edge,
            mut else_edge,
        } => {
            let mut changed = thread_edge(function, &mut then_edge);
            changed |= thread_edge(function, &mut else_edge);
            if same_edge_payload(&then_edge, &else_edge) {
                return Some(Inst::Jump { edge: then_edge });
            }
            changed.then_some(Inst::Branch {
                cond,
                then_edge,
                else_edge,
            })
        }
        Inst::Switch {
            tag,
            mut default_edge,
            mut cases,
        } => {
            let mut changed = thread_edge(function, &mut default_edge);
            for case in &mut cases {
                changed |= thread_edge(function, &mut case.edge);
            }

            let old_len = cases.len();
            let mut seen = HashSet::with_capacity(cases.len());
            cases.retain(|case| {
                seen.insert(case.value) && !same_edge_payload(&case.edge, &default_edge)
            });
            changed |= cases.len() != old_len;

            if cases.is_empty() {
                return Some(Inst::Jump { edge: default_edge });
            }

            changed.then_some(Inst::Switch {
                tag,
                default_edge,
                cases,
            })
        }
        _ => None,
    }
}

fn thread_edge(function: &Function, edge: &mut Edge) -> bool {
    let original = edge.clone();
    let mut candidate = edge.clone();
    let incoming_effect = edge.effect;
    let mut seen = Vec::<BlockId>::with_capacity(8);
    let mut changed = false;

    for _ in 0..8 {
        if seen.contains(&candidate.target) {
            return false;
        }
        seen.push(candidate.target);

        let Some(next) = thread_edge_once(function, &candidate, incoming_effect) else {
            break;
        };
        candidate = next;
        changed = true;
    }

    if changed
        && (candidate.target != original.target || candidate.args != original.args)
    {
        *edge = candidate;
        true
    } else {
        false
    }
}

fn thread_edge_once(function: &Function, incoming: &Edge, effect: crate::wren_ir::EffectId) -> Option<Edge> {
    if incoming.target == function.entry {
        return None;
    }
    let block = function.block(incoming.target)?;
    if block.insts.len() != 1 {
        return None;
    }
    let node = function.node(block.insts[0])?;
    let Inst::Jump { edge: outgoing } = &node.kind else {
        return None;
    };

    let mut args = outgoing.args.clone();
    for arg in &mut args {
        let Some(value) = function.value(*arg) else {
            continue;
        };
        let ValueDef::BlockParam(owner, index) = value.def else {
            continue;
        };
        if owner == block.id {
            *arg = *incoming.args.get(index as usize)?;
        }
    }

    Some(Edge {
        target: outgoing.target,
        args,
        effect,
    })
}

fn compute_reachable(function: &Function) -> Vec<bool> {
    let mut reachable = vec![false; function.blocks.len()];
    let mut exceptional_successors = vec![Vec::<BlockId>::new(); function.blocks.len()];
    for region in &function.exception_regions {
        for &protected in &region.protected_blocks {
            if let Some(successors) = exceptional_successors.get_mut(protected.0 as usize) {
                if !successors.contains(&region.handler) {
                    successors.push(region.handler);
                }
            }
        }
    }

    let mut stack = Vec::new();
    if (function.entry.0 as usize) < reachable.len() {
        stack.push(function.entry);
    }

    while let Some(block_id) = stack.pop() {
        let block_ix = block_id.0 as usize;
        if block_ix >= reachable.len() || reachable[block_ix] {
            continue;
        }
        reachable[block_ix] = true;

        let Some(block) = function.block(block_id) else {
            continue;
        };
        let Some(term) = block.insts.last().and_then(|id| function.node(*id)) else {
            continue;
        };
        for edge in term.kind.edges() {
            let target_ix = edge.target.0 as usize;
            if target_ix < reachable.len() && !reachable[target_ix] {
                stack.push(edge.target);
            }
        }
        for &handler in &exceptional_successors[block_ix] {
            let target_ix = handler.0 as usize;
            if target_ix < reachable.len() && !reachable[target_ix] {
                stack.push(handler);
            }
        }
    }

    reachable
}

fn live_node_bitmap(function: &Function) -> Vec<bool> {
    let mut live = vec![false; function.nodes.len()];
    for block in &function.blocks {
        for id in block.insts.iter().copied() {
            if let Some(slot) = live.get_mut(id.0 as usize) {
                *slot = true;
            }
        }
    }
    live
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
            if can_substitute(function, *source, *result) {
                alias[result.0 as usize] = *source;
            }
        }

        compress_aliases(&mut alias);

        // A block parameter fed by the same SSA value on every reachable edge is
        // an identity phi. Self-fed backedges are ignored, which also catches the
        // common loop-invariant parameter pattern.
        let reachable = compute_reachable(function);
        let mut candidate = vec![None; value_count];
        let mut conflict = vec![false; value_count];
        for _ in 0..4 {
            candidate.fill(None);
            conflict.fill(false);

            for (block_ix, block) in function.blocks.iter().enumerate() {
                if !reachable.get(block_ix).copied().unwrap_or(false) {
                    continue;
                }
                for id in block.insts.iter().copied() {
                    for edge in function.nodes[id.0 as usize].kind.edges() {
                        let Some(target) = function.block(edge.target) else {
                            continue;
                        };
                        for (&param, &arg) in target.params.iter().zip(&edge.args) {
                            let p = param.0 as usize;
                            if p >= value_count || conflict[p] {
                                continue;
                            }
                            let arg = canonical_alias(&alias, arg);
                            if arg == param {
                                continue;
                            }
                            match candidate[p] {
                                None => candidate[p] = Some(arg),
                                Some(previous) if previous == arg => {}
                                Some(_) => conflict[p] = true,
                            }
                        }
                    }
                }
            }

            let mut added = false;
            for index in 0..value_count {
                if conflict[index] || alias[index] != ValueId(index as u32) {
                    continue;
                }
                let Some(source) = candidate[index] else {
                    continue;
                };
                let result = ValueId(index as u32);
                if matches!(function.values[index].def, ValueDef::BlockParam(_, _))
                    && can_substitute(function, source, result)
                {
                    alias[index] = source;
                    added = true;
                }
            }
            if !added {
                break;
            }
            compress_aliases(&mut alias);
        }

        compress_aliases(&mut alias);
        let mut changed = false;

        for id in live.iter().copied() {
            let node = &mut function.nodes[id.0 as usize];
            node.kind.for_each_operand_mut(|operand| {
                let index = operand.0 as usize;
                let canonical = alias.get(index).copied().unwrap_or(*operand);
                if canonical != *operand {
                    *operand = canonical;
                    changed = true;
                }
            });
        }

        for deopt in &mut function.deopts {
            for slot in &mut deopt.snapshot.slots {
                let canonical = alias
                    .get(slot.value.0 as usize)
                    .copied()
                    .unwrap_or(slot.value);
                if canonical != slot.value {
                    slot.value = canonical;
                    changed = true;
                }
            }
            for frame in &mut deopt.snapshot.inlined_frames {
                for slot in &mut frame.slots {
                    let canonical = alias
                        .get(slot.value.0 as usize)
                        .copied()
                        .unwrap_or(slot.value);
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
            if alias
                .get(result.0 as usize)
                .copied()
                .unwrap_or(*result)
                != *result
            {
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

fn can_substitute(function: &Function, source: ValueId, result: ValueId) -> bool {
    let Some(source_ty) = function.value(source).map(|value| &value.ty) else {
        return false;
    };
    let Some(result_ty) = function.value(result).map(|value| &value.ty) else {
        return false;
    };

    if source_ty == result_ty {
        return true;
    }
    if source_ty.is_internal_representation() || result_ty.is_internal_representation() {
        return false;
    }
    result_ty.accepts(source_ty)
}

fn compress_aliases(alias: &mut [ValueId]) {
    for index in 0..alias.len() {
        let value = ValueId(index as u32);
        let canonical = canonical_alias(alias, value);
        alias[index] = canonical;
    }
}

fn canonical_alias(alias: &[ValueId], start: ValueId) -> ValueId {
    let mut current = start;
    for _ in 0..=alias.len() {
        let Some(next) = alias.get(current.0 as usize).copied() else {
            return current;
        };
        if next == current {
            return current;
        }
        current = next;
    }
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
