//! Analyses for the trace CFG: preheader -> linear guarded body -> backedge.
//! Side exits are leaves, so execution order is also the dominator-tree order.
use super::*;
use crate::wren_ir::{CmpKind, F64BinOp, F64UnaryOp};
use std::collections::VecDeque;

pub(super) fn order(trace: &Trace) -> Vec<usize> {
    let pre: HashSet<_> = trace.preheader.iter().copied().collect();
    trace.preheader.iter().copied()
        .chain((0..trace.nodes.len()).filter(|id| !pre.contains(id))).collect()
}

pub(super) fn constant(trace: &Trace, value: TraceValueId) -> Option<TraceConstant> {
    match trace.value(trace.canonical_value(value))?.origin {
        TraceValueOrigin::Constant(c) => Some(c),
        _ => None,
    }
}

pub(super) fn intern(trace: &mut Trace, c: TraceConstant, ty: TraceType, rep: TraceRep) -> TraceValueId {
    if let Some(v) = trace.values.iter().find(|v| v.ty == ty && v.rep == rep
        && matches!(v.origin, TraceValueOrigin::Constant(other) if c == other)) {
        return v.id;
    }
    let id = TraceValueId(trace.values.len() as u32);
    trace.values.push(TraceValue { id, ty, rep, origin: TraceValueOrigin::Constant(c) });
    id
}

pub(super) fn replace(trace: &mut Trace, node: usize, value: TraceValueId) {
    let result = trace.nodes[node].results[0];
    alias_value(trace, result, value);
    trace.nodes[node].kind = TraceInst::Nop;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lattice { Unknown, Constant(TraceConstant), Varying }
impl Lattice {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, x) | (x, Self::Unknown) => x,
            (Self::Constant(a), Self::Constant(b)) if a == b => self,
            _ => Self::Varying,
        }
    }
}

pub(super) fn integer(c: TraceConstant) -> Option<i32> {
    if let TraceConstant::Int32(n) = c { Some(n) } else { None }
}
pub(super) fn float(c: TraceConstant) -> Option<f64> {
    if let TraceConstant::F64Bits(n) = c { Some(f64::from_bits(n)) } else { None }
}
fn compare<T: PartialOrd>(op: CmpKind, a: T, b: T) -> bool {
    match op { CmpKind::Eq => a == b, CmpKind::Ne => a != b, CmpKind::Lt => a < b,
        CmpKind::Le => a <= b, CmpKind::Gt => a > b, CmpKind::Ge => a >= b }
}

fn evaluate(kind: &TraceInst, get: impl Fn(TraceValueId) -> Option<TraceConstant>) -> Option<TraceConstant> {
    use TraceInst::*;
    let i = |v| integer(get(v)?);
    let f = |v| float(get(v)?);
    let result = match *kind {
        Copy { value } => get(value)?,
        IntUnary { op, value } => TraceConstant::Int32(match op {
            IntUnaryOp::Neg => i(value)?.wrapping_neg(), IntUnaryOp::BitNot => !i(value)?,
        }),
        IntBinary { op, lhs, rhs } | IntBinaryChecked { op, lhs, rhs, .. } => {
            let (a, b) = (i(lhs)?, i(rhs)?);
            let checked = matches!(kind, IntBinaryChecked { .. });
            TraceConstant::Int32(match op {
                IntBinOp::Add if checked => a.checked_add(b)?,
                IntBinOp::Sub if checked => a.checked_sub(b)?,
                IntBinOp::Mul if checked => a.checked_mul(b)?,
                IntBinOp::Add => a.wrapping_add(b), IntBinOp::Sub => a.wrapping_sub(b),
                IntBinOp::Mul => a.wrapping_mul(b), IntBinOp::DivS => a.checked_div(b)?,
                IntBinOp::RemS => a.checked_rem(b)?, IntBinOp::BitAnd => a & b,
                IntBinOp::BitOr => a | b, IntBinOp::BitXor => a ^ b,
                IntBinOp::Shl => a.wrapping_shl(b as u32),
                IntBinOp::ShrU => (a as u32).wrapping_shr(b as u32) as i32,
                IntBinOp::Sar => a.wrapping_shr(b as u32),
            })
        }
        F64Unary { op: F64UnaryOp::Neg, value } =>
            TraceConstant::F64Bits(f(value)?.to_bits() ^ (1u64 << 63)),
        F64Binary { op, lhs, rhs } => {
            let (a, b) = (f(lhs)?, f(rhs)?);
            let n = match op { F64BinOp::Add => a + b, F64BinOp::Sub => a - b,
                F64BinOp::Mul => a * b, F64BinOp::Div => a / b, F64BinOp::Rem => a % b };
            // Preserve the backend's NaN payload/quieting behavior.
            if n.is_nan() { return None; }
            TraceConstant::F64Bits(n.to_bits())
        }
        IntCompare { op, lhs, rhs } => TraceConstant::Int32(compare(op, i(lhs)?, i(rhs)?) as i32),
        F64Compare { op, lhs, rhs } => TraceConstant::Int32(compare(op, f(lhs)?, f(rhs)?) as i32),
        IsNull { value } => TraceConstant::Int32((get(value)? == TraceConstant::Null) as i32),
        Select { condition, if_true, if_false } => get(if i(condition)? != 0 { if_true } else { if_false })?,
        Int32ToF64 { value } => TraceConstant::F64Bits((i(value)? as f64).to_bits()),
        UInt32ToF64 { value } => TraceConstant::F64Bits((i(value)? as u32 as f64).to_bits()),
        BoxInt32 { value } => TraceConstant::NumBits((i(value)? as f64).to_bits()),
        BoxUInt32 { value } => TraceConstant::NumBits((i(value)? as u32 as f64).to_bits()),
        BoxF64 { value } => {
            let n = f(value)?;
            if n.is_nan() { return None; } // native boxing canonicalizes all NaNs
            TraceConstant::NumBits(n.to_bits())
        }
        _ => return None,
    };
    Some(result)
}

/// Sparse fixed-point propagation through all SSA producers and loop edges.
/// Unknown backedges are distinct from varying values; a changing induction
/// variable therefore cannot be folded to its first-iteration constant.
#[derive(Default)]
pub struct SparseConditionalConstantPropagation;
impl TraceOptimization for SparseConditionalConstantPropagation {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let n = trace.values.len();
        let mut lattice = vec![Lattice::Unknown; n];
        let mut users = vec![Vec::new(); n];
        for node in &trace.nodes {
            for &result in &node.results {
                node.kind.clone().visit_operands(|v| users[v.0 as usize].push(result));
            }
        }
        for phi in &trace.loop_phis {
            users[phi.initial.0 as usize].push(phi.value);
            if let Some(v) = phi.backedge { users[v.0 as usize].push(phi.value); }
        }
        for value in &trace.values {
            if let TraceValueOrigin::Alias(v) = value.origin { users[v.0 as usize].push(value.id); }
        }
        let mut queue: VecDeque<_> = trace.values.iter().map(|v| v.id).collect();
        let mut queued = vec![true; n];
        while let Some(id) = queue.pop_front() {
            queued[id.0 as usize] = false;
            let get = |v: TraceValueId| lattice[v.0 as usize];
            let value = &trace.values[id.0 as usize];
            let proposed = match value.origin {
                TraceValueOrigin::Constant(c) => Lattice::Constant(c),
                TraceValueOrigin::Alias(v) => get(v),
                TraceValueOrigin::LoopPhi { .. } | TraceValueOrigin::LoopTemporary => {
                    let Some(phi) = trace.loop_phis.iter().find(|p| p.value == id) else { continue };
                    if phi.entry_exit.is_some() { Lattice::Varying }
                    else { get(phi.initial).meet(phi.backedge.map(get).unwrap_or(Lattice::Varying)) }
                }
                TraceValueOrigin::InstResult { node } => {
                    let kind = &trace.nodes[node].kind;
                    let c = evaluate(kind, |v| match get(v) { Lattice::Constant(c) => Some(c), _ => None });
                    if let Some(c) = c { Lattice::Constant(c) } else {
                        let mut unknown = false;
                        let mut varying = false;
                        kind.clone().visit_operands(|v| { unknown |= get(*v) == Lattice::Unknown; varying |= get(*v) == Lattice::Varying; });
                        if unknown && !varying { Lattice::Unknown } else { Lattice::Varying }
                    }
                }
                _ => Lattice::Varying,
            };
            let next = lattice[id.0 as usize].meet(proposed);
            if next != lattice[id.0 as usize] {
                lattice[id.0 as usize] = next;
                for &user in &users[id.0 as usize] {
                    if !std::mem::replace(&mut queued[user.0 as usize], true) { queue.push_back(user); }
                }
            }
        }
        let mut changed = false;
        for id in order(trace) {
            let [result] = trace.nodes[id].results.as_slice() else { continue };
            let result = *result;
            if matches!(trace.nodes[id].kind, TraceInst::Nop) { continue; }
            if let Lattice::Constant(c) = lattice[result.0 as usize] {
                let v = &trace.values[result.0 as usize];
                let replacement = intern(trace, c, v.ty, v.rep);
                replace(trace, id, replacement);
                trace.stats.sccp_folded += 1;
                changed = true;
            }
        }
        // Eliminate only proven-successful guards; a failing guard still owns
        // its precise snapshot and remains an executable side exit.
        for id in order(trace) {
            let succeeds = match trace.nodes[id].kind {
                TraceInst::GuardTruthy { value, .. } => constant(trace, value).map(|c| !matches!(c, TraceConstant::Null | TraceConstant::Bool(false))),
                TraceInst::GuardFalsey { value, .. } => constant(trace, value).map(|c| matches!(c, TraceConstant::Null | TraceConstant::Bool(false))),
                TraceInst::GuardBounds { index, length, .. } => constant(trace, index).and_then(integer)
                    .zip(constant(trace, length).and_then(integer)).map(|(i, l)| i >= 0 && i < l),
                _ => None,
            };
            if succeeds == Some(true) {
                trace.nodes[id].kind = TraceInst::Nop;
                trace.stats.eliminated_guards += 1;
                changed = true;
            }
        }
        changed
    }
}

/// Simplify unchanged/duplicate loop state and wrapping integer identities.
/// Checked arithmetic and floating-point identities keep their original rules.
#[derive(Default)]
pub struct InductionVariableSimplification;
impl TraceOptimization for InductionVariableSimplification {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut changed = false;
        for id in order(trace) {
            let replacement = match trace.nodes[id].kind {
                TraceInst::IntBinary { op, lhs, rhs } => {
                    match (op, constant(trace, rhs).and_then(integer)) {
                        (IntBinOp::Add | IntBinOp::Sub | IntBinOp::BitOr | IntBinOp::BitXor, Some(0))
                        | (IntBinOp::Mul, Some(1)) => Some(lhs),
                        _ => None,
                    }
                }
                TraceInst::Select { if_true, if_false, .. } if if_true == if_false => Some(if_true),
                _ => None,
            };
            if let Some(v) = replacement {
                replace(trace, id, v);
                changed = true;
            }
        }
        let mut seen = HashMap::new();
        let mut remove = HashSet::new();
        for phi in trace.loop_phis.clone() {
            if phi.entry_exit.is_some() { continue; }
            let initial = trace.canonical_value(phi.initial);
            let back = phi.backedge.map(|v| trace.canonical_value(v));
            let replacement = if back == Some(phi.value) || back == Some(initial) { Some(initial) }
                else { seen.insert((initial, back, phi.rep), phi.value) };
            if let Some(v) = replacement {
                alias_value(trace, phi.value, v);
                remove.insert(phi.value);
                trace.stats.simplified_inductions += 1;
                changed = true;
            }
        }
        trace.loop_phis.retain(|p| !remove.contains(&p.value));
        changed
    }
}

/// Convert wrapping multiplication by a power of two to a shift. For an affine
/// loop induction, carry the scaled value in a new phi and update it by addition
/// at the original multiplication, preserving all snapshot dominance.
#[derive(Default)]
pub struct StrengthReduction;
impl TraceOptimization for StrengthReduction {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut changed = false;
        let mut budget = 16;
        for id in order(trace) {
            let TraceInst::IntBinary { op: IntBinOp::Mul, lhs, rhs } = trace.nodes[id].kind else { continue };
            let Some(scale) = constant(trace, rhs).and_then(integer) else { continue };
            if native_loop(trace) && budget > 0 && !trace.preheader.contains(&id) {
                let induction = trace.loop_phis.iter().find(|p| p.value == lhs && p.rep == TraceRep::Int32 && p.entry_exit.is_none()).cloned();
                if let Some(phi) = induction {
                    let step = phi.backedge.and_then(|v| TraceCanonicalization::producer(trace, v))
                        .and_then(|(_, k)| match k {
                            TraceInst::IntBinary { op: IntBinOp::Add, lhs: p, rhs: s }
                                if p == phi.value => constant(trace, s).and_then(integer),
                            _ => None,
                        });
                    if let Some(step) = step {
                        let source = trace.nodes[id].source;
                        let start = append_preheader(trace, TraceInst::IntBinary { op: IntBinOp::Mul, lhs: phi.initial, rhs }, TraceType::Int32, TraceRep::Int32, source);
                        let value = TraceValueId(trace.values.len() as u32);
                        trace.values.push(TraceValue { id: value, ty: TraceType::Int32, rep: TraceRep::Int32, origin: TraceValueOrigin::LoopTemporary });
                        let delta = intern(trace, TraceConstant::Int32(step.wrapping_mul(scale)), TraceType::Int32, TraceRep::Int32);
                        let next = TraceValueId(trace.values.len() as u32);
                        let node = trace.nodes.len();
                        trace.values.push(TraceValue { id: next, ty: TraceType::Int32, rep: TraceRep::Int32, origin: TraceValueOrigin::InstResult { node } });
                        trace.nodes.push(TraceNode { id: node, kind: TraceInst::IntBinary { op: IntBinOp::Add, lhs: value, rhs: delta }, results: vec![next], source });
                        trace.loop_phis.push(LoopPhi { value, initial: start, backedge: Some(next), rep: TraceRep::Int32, entry_exit: None });
                        replace(trace, id, value);
                        trace.stats.strength_reduced += 1;
                        changed = true;
                        budget -= 1;
                        continue;
                    }
                }
            }
            if scale > 0 && (scale as u32).is_power_of_two() {
                let shift = intern(trace, TraceConstant::Int32(scale.trailing_zeros() as i32), TraceType::Int32, TraceRep::Int32);
                trace.nodes[id].kind = TraceInst::IntBinary { op: IntBinOp::Shl, lhs, rhs: shift };
                trace.stats.strength_reduced += 1;
                changed = true;
            }
        }
        changed
    }
}

/// Facts are scoped to the dominating successful path, never written into the
/// global type of an entry value (which would also affect a preheader/side exit).
#[derive(Default)]
pub struct ClassTypePropagation;
impl TraceOptimization for ClassTypePropagation {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut types = HashMap::new();
        let mut classes: HashMap<TraceValueId, Vec<u64>> = HashMap::new();
        let mut changed = false;
        for id in order(trace) {
            match trace.nodes[id].kind.clone() {
                TraceInst::Runtime { .. } | TraceInst::ReadBarrier { .. } => { types.clear(); classes.clear(); }
                TraceInst::GuardType { value, expected, .. } => {
                    let known = types.get(&value).copied().or_else(|| constant(trace, value).map(|c| c.layout().0));
                    let redundant = expected == TraceType::Any || known == Some(expected);
                    if redundant {
                        if trace.nodes[id].results.is_empty() { trace.nodes[id].kind = TraceInst::Nop; }
                        else { replace(trace, id, value); }
                        trace.stats.eliminated_guards += 1;
                        changed = true;
                    } else {
                        types.insert(value, expected);
                        for &result in &trace.nodes[id].results { types.insert(result, expected); }
                    }
                }
                TraceInst::GuardClass { value, classes: expected, .. } => {
                    if classes.get(&value).is_some_and(|known| known.iter().all(|c| expected.contains(c))) {
                        replace(trace, id, value);
                        trace.stats.eliminated_guards += 1;
                        changed = true;
                    } else {
                        let narrowed = classes.get(&value).map(|old| expected.iter().copied().filter(|c| old.contains(c)).collect()).unwrap_or(expected);
                        classes.insert(value, narrowed.clone());
                        for &result in &trace.nodes[id].results { classes.insert(result, narrowed.clone()); }
                    }
                }
                TraceInst::BoxInt32 { .. } | TraceInst::BoxUInt32 { .. } | TraceInst::BoxF64 { .. } => {
                    for &result in &trace.nodes[id].results { types.insert(result, TraceType::Num); }
                }
                TraceInst::GuardNotNull { value, .. } => {
                    if classes.contains_key(&value) || matches!(types.get(&value), Some(TraceType::Num | TraceType::Bool)) {
                        if trace.nodes[id].results.is_empty() { trace.nodes[id].kind = TraceInst::Nop; }
                        else { replace(trace, id, value); }
                        trace.stats.eliminated_guards += 1;
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        changed
    }
}

fn anchor_exit(trace: &mut Trace) -> SideExitId {
    let id = SideExitId(trace.exits.len() as u32);
    trace.exits.push(SideExit { id, target: ExitTarget::Interpreter(trace.anchor.location),
        frames: vec![FrameSnapshot { disposition: FrameDisposition::Existing { depth_from_top: 0 }, writes: vec![] }], linked_trace: None });
    id
}

/// In a trace the observed branch is already a loop version. Move its invariant
/// path guard to entry; the other version starts at the original interpreter
/// anchor (and can become a side trace). Never reuse a body snapshot at entry.
#[derive(Default)]
pub struct LoopInvariantGuardHoisting;
impl TraceOptimization for LoopInvariantGuardHoisting {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        hoist_guards(trace, false)
    }
}

#[derive(Default)]
pub struct LoopUnswitching;
impl TraceOptimization for LoopUnswitching {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        hoist_guards(trace, true)
    }
}

fn hoist_guards(trace: &mut Trace, branch_only: bool) -> bool {
        if !native_loop(trace) || trace.nodes.iter().any(|n| matches!(n.kind, TraceInst::ReadBarrier { .. })) { return false; }
        let mut available = preheader_values(trace);
        let mut changed = false;
        for id in 0..trace.nodes.len() {
            if trace.preheader.contains(&id) || !trace.nodes[id].kind.is_entry_guard() { continue; }
            let mut invariant = true;
            trace.nodes[id].kind.clone().visit_operands(|v| invariant &= available.contains(v));
            if !invariant { continue; }
            let branch = matches!(trace.nodes[id].kind, TraceInst::GuardTruthy { .. } | TraceInst::GuardFalsey { .. } | TraceInst::GuardSwitch { .. } | TraceInst::GuardEq { .. });
            if branch != branch_only { continue; }
            let exit = anchor_exit(trace);
            trace.nodes[id].kind.set_exit(exit);
            trace.preheader.push(id);
            available.extend(trace.nodes[id].results.iter().copied());
            trace.stats.hoisted_guards += 1;
            trace.stats.unswitched_guards += branch as usize;
            changed = true;
        }
        changed
}

#[derive(Clone, Copy, Debug)]
struct Range { lo: i64, hi: i64 }
impl Range {
    const FULL: Self = Self { lo: i32::MIN as i64, hi: i32::MAX as i64 };
    fn point(n: i32) -> Self { Self { lo: n as i64, hi: n as i64 } }
    fn intersect(self, other: Self) -> Self { Self { lo: self.lo.max(other.lo), hi: self.hi.min(other.hi) } }
}
fn range(trace: &Trace, ranges: &HashMap<TraceValueId, Range>, value: TraceValueId) -> Range {
    constant(trace, value).and_then(integer).map(Range::point).or_else(|| ranges.get(&value).copied()).unwrap_or(Range::FULL)
}

fn comparison(trace: &Trace, value: TraceValueId, truth: bool) -> Option<(CmpKind, TraceValueId, TraceValueId)> {
    let (_, kind) = TraceCanonicalization::producer(trace, value)?;
    match kind {
        TraceInst::IntCompare { op, lhs, rhs } => Some((if truth { op } else { match op {
            CmpKind::Eq => CmpKind::Ne, CmpKind::Ne => CmpKind::Eq, CmpKind::Lt => CmpKind::Ge,
            CmpKind::Le => CmpKind::Gt, CmpKind::Gt => CmpKind::Le, CmpKind::Ge => CmpKind::Lt,
        } }, lhs, rhs)),
        TraceInst::Select { condition, if_true, if_false }
            if constant(trace, if_true) == Some(TraceConstant::Bool(true))
                && constant(trace, if_false) == Some(TraceConstant::Bool(false)) => comparison(trace, condition, truth),
        _ => None,
    }
}

#[derive(Default)]
pub struct RangeAnalysis;
impl TraceOptimization for RangeAnalysis {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut ranges = HashMap::new();
        let mut less_than = HashSet::new();
        let mut changed = false;
        for id in order(trace) {
            let kind = trace.nodes[id].kind.clone();
            let get = |v| range(trace, &ranges, v);
            let inferred = match kind {
                TraceInst::IntBinary { op, lhs, rhs } | TraceInst::IntBinaryChecked { op, lhs, rhs, .. } => {
                    let (a, b) = (get(lhs), get(rhs));
                    let r = match op {
                        IntBinOp::Add => Some(Range { lo: a.lo + b.lo, hi: a.hi + b.hi }),
                        IntBinOp::Sub => Some(Range { lo: a.lo - b.hi, hi: a.hi - b.lo }),
                        IntBinOp::Mul => { let p = [a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi]; Some(Range { lo: *p.iter().min().unwrap(), hi: *p.iter().max().unwrap() }) }
                        IntBinOp::BitAnd if b.lo == b.hi && b.lo >= 0 => Some(Range { lo: 0, hi: b.hi }),
                        _ => None,
                    };
                    if r.is_some_and(|r| r.lo >= Range::FULL.lo && r.hi <= Range::FULL.hi) {
                        if matches!(kind, TraceInst::IntBinaryChecked { .. }) {
                            trace.nodes[id].kind = TraceInst::IntBinary { op, lhs, rhs };
                            trace.stats.eliminated_overflow_checks += 1;
                            changed = true;
                        }
                        r
                    } else { None }
                }
                TraceInst::IntCompare { .. } | TraceInst::F64Compare { .. } | TraceInst::IsNull { .. } => Some(Range { lo: 0, hi: 1 }),
                _ => None,
            };
            if let (Some(r), Some(&v)) = (inferred, trace.nodes[id].results.first()) { ranges.insert(v, r); }
            match kind {
                TraceInst::Runtime { .. } => { ranges.clear(); less_than.clear(); }
                TraceInst::GuardBounds { index, length, .. } => {
                    let (i, l) = (range(trace, &ranges, index), range(trace, &ranges, length));
                    if i.lo >= 0 && (i.hi < l.lo || less_than.contains(&(index, length))) {
                        trace.nodes[id].kind = TraceInst::Nop;
                        trace.stats.eliminated_bounds_checks += 1;
                        changed = true;
                    }
                    ranges.insert(index, i.intersect(Range { lo: 0, hi: l.hi - 1 }));
                    less_than.insert((index, length));
                }
                TraceInst::GuardTruthy { value, .. } | TraceInst::GuardFalsey { value, .. } => {
                    if let Some((mut op, mut lhs, mut rhs)) = comparison(trace, value, matches!(kind, TraceInst::GuardTruthy { .. })) {
                        if matches!(op, CmpKind::Gt | CmpKind::Ge) {
                            std::mem::swap(&mut lhs, &mut rhs);
                            op = if op == CmpKind::Gt { CmpKind::Lt } else { CmpKind::Le };
                        }
                        let (a, b) = (range(trace, &ranges, lhs), range(trace, &ranges, rhs));
                        if matches!(op, CmpKind::Lt | CmpKind::Le) {
                            let strict = (op == CmpKind::Lt) as i64;
                            ranges.insert(lhs, a.intersect(Range { lo: Range::FULL.lo, hi: b.hi - strict }));
                            ranges.insert(rhs, b.intersect(Range { lo: a.lo + strict, hi: Range::FULL.hi }));
                            if strict != 0 { less_than.insert((lhs, rhs)); }
                        } else if op == CmpKind::Eq {
                            ranges.insert(lhs, a.intersect(b)); ranges.insert(rhs, a.intersect(b));
                        }
                    }
                }
                _ => {}
            }
        }
        changed
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Memory {
    Frame(FrameSlot, TraceRep), Field(TraceValueId, u32), Cell(TraceValueId),
    ListLen(TraceValueId), List(TraceValueId, TraceValueId), MapLen(TraceValueId),
    Map(TraceValueId, TraceValueId, Vec<u8>, u32, u32), StringLen(TraceValueId),
}
fn load(kind: &TraceInst, rep: TraceRep) -> Option<Memory> {
    Some(match kind {
        TraceInst::LoadFrameSlot { slot } => Memory::Frame(*slot, rep),
        TraceInst::HeapLoadField { object, slot, .. } => Memory::Field(*object, *slot),
        TraceInst::HeapLoadCell { cell, .. } => Memory::Cell(*cell),
        TraceInst::HeapListLen { list, .. } => Memory::ListLen(*list),
        TraceInst::HeapListGet { list, index, .. } => Memory::List(*list, *index),
        TraceInst::HeapMapLen { map, .. } => Memory::MapLen(*map),
        TraceInst::HeapMapGetShaped { map, key, expected_key, shape, slot, .. } => Memory::Map(*map, *key, expected_key.clone(), *shape, *slot),
        TraceInst::HeapStringLen { string, .. } => Memory::StringLen(*string),
        _ => return None,
    })
}
fn store(kind: &TraceInst, trace: &Trace) -> Option<(Memory, TraceValueId)> {
    Some(match kind {
        TraceInst::CommitFrameSlot { slot, value } => (Memory::Frame(*slot, trace.value(*value)?.rep), *value),
        TraceInst::HeapStoreField { object, slot, value, .. } => (Memory::Field(*object, *slot), *value),
        TraceInst::HeapStoreCell { cell, value, .. } => (Memory::Cell(*cell), *value),
        TraceInst::HeapListSet { list, index, value, .. } => (Memory::List(*list, *index), *value),
        TraceInst::HeapMapSetShaped { map, key, expected_key, shape, slot, value, .. } => (Memory::Map(*map, *key, expected_key.clone(), *shape, *slot), *value),
        _ => return None,
    })
}

/// Memory value numbering and store-to-load forwarding. Different SSA object
/// references may alias: every store kills heap facts, and every runtime/barrier
/// kills all facts. A successful access supplies both value and validation.
#[derive(Default)]
pub struct LoadStorePRE;
impl TraceOptimization for LoadStorePRE {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut available: HashMap<Memory, TraceValueId> = HashMap::new();
        let mut validated_stores = HashSet::new();
        let mut changed = false;
        for id in order(trace) {
            let kind = trace.nodes[id].kind.clone();
            if matches!(kind, TraceInst::Runtime { .. } | TraceInst::ReadBarrier { .. }) {
                available.clear(); validated_stores.clear(); continue;
            }
            if let Some((key, value)) = store(&kind, trace) {
                // A previous identical store already validated this address and
                // executed the post barrier in the same GC/memory epoch.
                if validated_stores.contains(&key) && available.get(&key) == Some(&value) {
                    trace.nodes[id].kind = TraceInst::Nop;
                    trace.stats.eliminated_stores += 1;
                    changed = true;
                    continue;
                }
                // Keep the first store's validation and write barrier. Frame
                // stores also alias across representations of the same slot.
                match &key {
                    Memory::Frame(slot, _) => {
                        available.retain(|k, _| !matches!(k, Memory::Frame(s, _) if s == slot));
                        validated_stores.retain(|k| !matches!(k, Memory::Frame(s, _) if s == slot));
                    }
                    _ => {
                        available.retain(|k, _| matches!(k, Memory::Frame(..)));
                        validated_stores.retain(|k| matches!(k, Memory::Frame(..)));
                    }
                }
                validated_stores.insert(key.clone());
                available.insert(key, value);
            } else if let Some(&result) = trace.nodes[id].results.first() {
                let rep = trace.value(result).unwrap().rep;
                if let Some(key) = load(&kind, rep) {
                    if let Some(&v) = available.get(&key) {
                        if trace.value(v).unwrap().rep == rep {
                            replace(trace, id, v);
                            trace.stats.forwarded_loads += 1;
                            changed = true;
                        }
                    } else { available.insert(key, result); }
                }
            }
        }
        if native_loop(trace) && !trace.nodes.iter().any(|n| matches!(n.kind, TraceInst::ReadBarrier { .. })) {
            let invariant = preheader_values(trace);
            let mut written = false;
            let body: Vec<_> = (0..trace.nodes.len()).filter(|id| !trace.preheader.contains(id)).collect();
            for id in body {
                let kind = trace.nodes[id].kind.clone();
                if store(&kind, trace).is_some_and(|(m, _)| !matches!(m, Memory::Frame(..))) { written = true; }
                if written || !kind.is_heap_read() { continue; }
                let mut inputs_available = true;
                kind.clone().visit_operands(|v| inputs_available &= invariant.contains(v));
                if !inputs_available { continue; }
                let result = trace.nodes[id].results[0];
                let ty = trace.value(result).unwrap().ty;
                let rep = trace.value(result).unwrap().rep;
                let key = load(&kind, rep).unwrap();
                let Some(&next) = available.get(&key) else { continue };
                if trace.value(next).unwrap().rep != rep { continue; }
                let exit = anchor_exit(trace);
                let mut entry_kind = kind;
                entry_kind.set_exit(exit);
                if next == result {
                    // No intervening aliasing write: the read is invariant.
                    trace.nodes[id].kind = entry_kind;
                    trace.preheader.push(id);
                } else {
                    // The entry edge performs the missing read. Subsequent
                    // iterations receive the validated last store/load value.
                    let start = TraceValueId(trace.values.len() as u32);
                    let node = trace.nodes.len();
                    trace.values.push(TraceValue { id: start, ty, rep, origin: TraceValueOrigin::InstResult { node } });
                    trace.nodes.push(TraceNode { id: node, kind: entry_kind, results: vec![start], source: trace.nodes[id].source });
                    trace.preheader.push(node);
                    let value = TraceValueId(trace.values.len() as u32);
                    trace.values.push(TraceValue { id: value, ty, rep, origin: TraceValueOrigin::LoopTemporary });
                    trace.loop_phis.push(LoopPhi { value, initial: start, backedge: Some(next), rep, entry_exit: None });
                    replace(trace, id, value);
                }
                trace.stats.loop_loads_reused += 1;
                changed = true;
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

    fn builder(kind: TraceKind) -> TraceBuilder {
        TraceBuilder::new(TraceId(0), TraceAnchor { kind, location: TraceLocation::new(FuncId(0), BytecodePc(0)) })
    }
    fn input(b: &mut TraceBuilder, reg: u8, rep: TraceRep) -> TraceValueId {
        b.read_slot(FrameSlot::General(Reg(reg)), if rep == TraceRep::Int32 { TraceType::Int32 } else { TraceType::Any }, rep)
    }
    fn int(b: &mut TraceBuilder, n: i32) -> TraceValueId { b.constant(TraceConstant::Int32(n)) }
    fn binary(b: &mut TraceBuilder, op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId) -> TraceValueId {
        b.emit1(TraceInst::IntBinary { op, lhs, rhs }, TraceType::Int32, TraceRep::Int32)
    }
    fn finish(mut b: TraceBuilder) -> Trace {
        if matches!(b.trace().anchor.kind, TraceKind::Loop) { b.set_terminator(TraceTerminator::LoopBack); }
        else { let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(10)); b.set_terminator(TraceTerminator::Exit(exit)); }
        b.finish().unwrap()
    }

    #[test]
    fn sccp_solves_loop_lattice_without_freezing_inductions_or_traps() {
        let mut b = builder(TraceKind::Loop);
        let zero = int(&mut b, 0);
        let one = int(&mut b, 1);
        let invariant = b.add_loop_phi(FrameSlot::General(Reg(0)), one, TraceType::Int32, TraceRep::Int32);
        let changing = b.add_loop_phi(FrameSlot::General(Reg(1)), zero, TraceType::Int32, TraceRep::Int32);
        let fixed = binary(&mut b, IntBinOp::Add, invariant, one);
        let next = binary(&mut b, IntBinOp::Add, changing, one);
        let trap = binary(&mut b, IntBinOp::DivS, one, zero);
        b.set_loop_backedge(invariant, invariant).unwrap();
        b.set_loop_backedge(changing, next).unwrap();
        let mut trace = finish(b);
        Optimize::new().with_pass(SparseConditionalConstantPropagation).optimize_trace(&mut trace).unwrap();
        assert_eq!(constant(&trace, fixed), Some(TraceConstant::Int32(2)));
        assert_eq!(constant(&trace, next), None);
        assert_eq!(constant(&trace, trap), None);
    }

    #[test]
    fn sccp_preserves_float_zero_nan_and_unsigned_boxing() {
        let mut b = builder(TraceKind::Side);
        let zero = b.constant(TraceConstant::F64Bits(0));
        let neg = b.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value: zero }, TraceType::F64, TraceRep::F64);
        let nan = b.constant(TraceConstant::F64Bits(0x7ff8_0000_0000_0001));
        let add = b.emit1(TraceInst::F64Binary { op: F64BinOp::Add, lhs: nan, rhs: zero }, TraceType::F64, TraceRep::F64);
        let minus_one = int(&mut b, -1);
        let boxed = b.emit1(TraceInst::BoxUInt32 { value: minus_one }, TraceType::Num, TraceRep::Tagged);
        let mut trace = finish(b);
        Optimize::new().with_pass(SparseConditionalConstantPropagation).optimize_trace(&mut trace).unwrap();
        assert_eq!(constant(&trace, neg), Some(TraceConstant::F64Bits((-0.0f64).to_bits())));
        assert_eq!(constant(&trace, add), None);
        assert_eq!(constant(&trace, boxed), Some(TraceConstant::NumBits((u32::MAX as f64).to_bits())));
    }

    #[test]
    fn ranges_remove_dominated_bounds_and_safe_overflow_only() {
        let mut b = builder(TraceKind::Side);
        let x = input(&mut b, 0, TraceRep::Int32);
        let mask = int(&mut b, 15);
        let index = binary(&mut b, IntBinOp::BitAnd, x, mask);
        let length = int(&mut b, 16);
        let exit = b.add_side_exit(ExitReason::GuardBounds, BytecodePc(0));
        b.emit0(TraceInst::GuardBounds { index, length, exit });
        b.emit0(TraceInst::GuardBounds { index: x, length, exit });
        b.emit0(TraceInst::GuardBounds { index: x, length, exit });
        b.emit1(TraceInst::IntBinaryChecked { op: IntBinOp::Add, lhs: index, rhs: length, exit }, TraceType::Int32, TraceRep::Int32);
        let unknown = input(&mut b, 1, TraceRep::Int32);
        b.emit1(TraceInst::IntBinaryChecked { op: IntBinOp::Add, lhs: unknown, rhs: length, exit }, TraceType::Int32, TraceRep::Int32);
        let mut trace = finish(b);
        Optimize::new().with_pass(RangeAnalysis).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.stats.eliminated_bounds_checks, 2);
        assert_eq!(trace.stats.eliminated_overflow_checks, 1);
        assert_eq!(trace.nodes.iter().filter(|n| matches!(n.kind, TraceInst::IntBinaryChecked { .. })).count(), 1);
    }

    #[test]
    fn class_facts_are_dominated_and_killed_by_runtime() {
        let mut b = builder(TraceKind::Side);
        let value = input(&mut b, 0, TraceRep::Tagged);
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        for classes in [vec![7], vec![7, 9]] {
            b.emit1(TraceInst::GuardClass { value, classes, exit }, TraceType::Any, TraceRep::Tagged);
        }
        b.runtime(TraceLocation::new(FuncId(0), BytecodePc(1)), BytecodePc(2), false);
        b.emit1(TraceInst::GuardClass { value, classes: vec![7, 9], exit }, TraceType::Any, TraceRep::Tagged);
        let mut trace = finish(b);
        Optimize::new().with_pass(ClassTypePropagation).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.stats.eliminated_guards, 1);
        assert_eq!(trace.nodes.iter().filter(|n| matches!(n.kind, TraceInst::GuardClass { .. })).count(), 2);
    }

    #[test]
    fn guard_unswitching_uses_anchor_snapshot_instead_of_body_state() {
        let mut b = builder(TraceKind::Loop);
        let condition = input(&mut b, 0, TraceRep::Tagged);
        let x = input(&mut b, 1, TraceRep::Int32);
        let one = int(&mut b, 1);
        let sum = binary(&mut b, IntBinOp::Add, x, one);
        b.write_slot(FrameSlot::General(Reg(1)), sum);
        let body_exit = b.add_side_exit(ExitReason::GuardTruthy, BytecodePc(20));
        b.emit0(TraceInst::GuardTruthy { value: condition, exit: body_exit });
        let mut trace = finish(b);
        Optimize::new().with_pass(LoopUnswitching).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.stats.unswitched_guards, 1);
        let exit = trace.nodes[trace.preheader[0]].kind.exit().unwrap();
        assert_ne!(exit, body_exit);
        assert!(trace.exits[exit.0 as usize].frames.iter().all(|f| f.writes.is_empty()));
        assert!(matches!(trace.exits[exit.0 as usize].target, ExitTarget::Interpreter(location) if location == trace.anchor.location));
    }

    #[test]
    fn memory_pre_forwards_stores_but_respects_possible_aliases() {
        let mut b = builder(TraceKind::Side);
        let a = input(&mut b, 0, TraceRep::Tagged);
        let possible_alias = input(&mut b, 1, TraceRep::Tagged);
        let value = b.constant(TraceConstant::NumBits(7.0f64.to_bits()));
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        b.emit0(TraceInst::HeapStoreField { object: a, slot: 0, value, exit });
        b.emit0(TraceInst::HeapStoreField { object: a, slot: 0, value, exit });
        let forwarded = b.emit1(TraceInst::HeapLoadField { object: a, slot: 0, exit }, TraceType::Any, TraceRep::Tagged);
        b.emit0(TraceInst::HeapStoreField { object: possible_alias, slot: 0, value, exit });
        let retained = b.emit1(TraceInst::HeapLoadField { object: a, slot: 0, exit }, TraceType::Any, TraceRep::Tagged);
        let mut trace = finish(b);
        Optimize::new().with_pass(LoadStorePRE).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.canonical_value(forwarded), value);
        assert_eq!(trace.stats.eliminated_stores, 1);
        assert_eq!(trace.canonical_value(retained), retained);
    }

    #[test]
    fn memory_pre_carries_last_store_over_backedge() {
        let mut b = builder(TraceKind::Loop);
        let object = input(&mut b, 0, TraceRep::Tagged);
        let value = b.constant(TraceConstant::NumBits(7.0f64.to_bits()));
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        let loaded = b.emit1(TraceInst::HeapLoadCell { cell: object, exit }, TraceType::Any, TraceRep::Tagged);
        b.write_slot(FrameSlot::General(Reg(1)), loaded);
        b.emit0(TraceInst::HeapStoreCell { cell: object, value, exit });
        let mut trace = finish(b);
        Optimize::new().with_pass(LoadStorePRE).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.stats.loop_loads_reused, 1);
        assert_eq!(trace.loop_phis.len(), 1);
        assert_eq!(trace.loop_phis[0].backedge, Some(value));
        assert_eq!(trace.canonical_value(loaded), trace.loop_phis[0].value);
        assert!(trace.nodes[trace.preheader[0]].kind.is_heap_read());
    }

    #[test]
    fn affine_strength_reduction_uses_wrapping_recurrence() {
        let mut b = builder(TraceKind::Loop);
        let start = input(&mut b, 0, TraceRep::Int32);
        let phi = b.add_loop_phi(FrameSlot::General(Reg(0)), start, TraceType::Int32, TraceRep::Int32);
        let step = int(&mut b, 3);
        let scale = int(&mut b, 7);
        let product = binary(&mut b, IntBinOp::Mul, phi, scale);
        let next = binary(&mut b, IntBinOp::Add, phi, step);
        b.set_loop_backedge(phi, next).unwrap();
        let mut trace = finish(b);
        Optimize::new().with_pass(StrengthReduction).optimize_trace(&mut trace).unwrap();
        assert_eq!(trace.loop_phis.len(), 2);
        assert_eq!(trace.canonical_value(product), trace.loop_phis[1].value);
        let (_, TraceInst::IntBinary { op: IntBinOp::Add, rhs, .. }) = TraceCanonicalization::producer(&trace, trace.loop_phis[1].backedge.unwrap()).unwrap() else { panic!("missing recurrence") };
        assert_eq!(constant(&trace, rhs), Some(TraceConstant::Int32(21)));
    }
}
