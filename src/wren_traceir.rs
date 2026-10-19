//! Trace SSA, architectural snapshots, and the runtime synchronization boundary.
//! Runtime calls commit registers before invoking the VM. Values read afterwards
//! are fresh loads: no heap or register assumption survives a runtime call.

use std::collections::{HashMap, HashSet};
use crate::wren_bytecode::{ArgReg, BytecodePc, Reg, RegisterKind, ReturnReg, SpecialReg};
use crate::wren_ir::{BuiltinId, CmpKind, F64BinOp, F64UnaryOp, FuncId, IntBinOp, IntUnaryOp, StringId};

macro_rules! id {
    ($($name:ident),*) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub u32);
    )*};
}
id!(TraceId, TraceValueId, SideExitId, TraceFrameId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TraceLocation { pub function: FuncId, pub pc: BytecodePc }
impl TraceLocation {
    pub const fn new(function: FuncId, pc: BytecodePc) -> Self { Self { function, pc } }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameSlot { General(Reg), Argument(ArgReg), Return(ReturnReg), Special(SpecialReg) }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceRep { Tagged, Int32, F64, Bool, HeapHandle, RawPtr }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceType { Any, Null, Bool, Num, Int32, F64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceClass { Builtin(BuiltinId) }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceConstant {
    Null, Bool(bool), NumBits(u64), Int32(i32), F64Bits(u64), String(StringId), BuiltinClass(BuiltinId),
}
impl TraceConstant {
    fn layout(self) -> (TraceType, TraceRep) {
        match self {
            Self::Null => (TraceType::Null, TraceRep::Tagged),
            Self::Bool(_) => (TraceType::Bool, TraceRep::Tagged),
            Self::NumBits(_) => (TraceType::Num, TraceRep::Tagged),
            Self::Int32(_) => (TraceType::Int32, TraceRep::Int32),
            Self::F64Bits(_) => (TraceType::F64, TraceRep::F64),
            _ => (TraceType::Any, TraceRep::Tagged),
        }
    }
}
#[derive(Clone, Debug)]
pub enum TraceValueOrigin {
    EntrySlot { frame: TraceFrameId, slot: FrameSlot },
    /// Value guarded/unboxed once in the native preheader. This is used for
    /// invariant loop inputs that would otherwise repeat GuardF64/GuardInt32.
    GuardedEntry { value: TraceValueId, exit: SideExitId },
    Constant(TraceConstant),
    InstResult { node: usize }, LoopPhi { slot: FrameSlot }, Alias(TraceValueId),
}
#[derive(Clone, Debug)]
pub struct TraceValue { pub id: TraceValueId, pub ty: TraceType, pub rep: TraceRep, pub origin: TraceValueOrigin }
#[derive(Clone, Debug)]
pub enum TraceSwitchGuard { Case(i64), Default { excluded: Vec<i64> } }
#[derive(Clone, Debug)]
pub enum TraceInst {
    Nop,
    Copy { value: TraceValueId },
    Reinterpret { value: TraceValueId },
    LoadFrameSlot { slot: FrameSlot },
    CommitFrameSlot { slot: FrameSlot, value: TraceValueId },
    /// Executes precisely one original opcode, with synchronized VM registers.
    Runtime { location: TraceLocation, next: BytecodePc, terminal: bool },
    GuardTruthy { value: TraceValueId, exit: SideExitId },
    GuardFalsey { value: TraceValueId, exit: SideExitId },
    GuardNotNull { value: TraceValueId, exit: SideExitId },
    GuardF64 { value: TraceValueId, exit: SideExitId },
    GuardInt32 { value: TraceValueId, exit: SideExitId },
    GuardBounds { index: TraceValueId, length: TraceValueId, exit: SideExitId },
    GuardEq { lhs: TraceValueId, rhs: TraceValueId, exit: SideExitId },
    GuardSwitch { tag: TraceValueId, observed: TraceSwitchGuard, exit: SideExitId },
    GuardType { value: TraceValueId, expected: TraceType, exit: SideExitId },
    IntUnary { op: IntUnaryOp, value: TraceValueId },
    IntBinary { op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId },
    IntBinaryChecked { op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId, exit: SideExitId },
    F64Unary { op: F64UnaryOp, value: TraceValueId },
    F64Binary { op: F64BinOp, lhs: TraceValueId, rhs: TraceValueId },
    IntCompare { op: CmpKind, lhs: TraceValueId, rhs: TraceValueId },
    F64Compare { op: CmpKind, lhs: TraceValueId, rhs: TraceValueId },
    RefEq { lhs: TraceValueId, rhs: TraceValueId },
    IsNull { value: TraceValueId },
    Select { condition: TraceValueId, if_true: TraceValueId, if_false: TraceValueId },
    BoxInt32 { value: TraceValueId }, BoxF64 { value: TraceValueId },
    Int32ToF64 { value: TraceValueId }, F64ToInt32 { value: TraceValueId, exit: SideExitId },
    TaggedToHeapHandle { value: TraceValueId, exit: SideExitId },
    ReadBarrier { value: TraceValueId },
}
impl TraceInst {
    fn visit_operands(&mut self, mut f: impl FnMut(&mut TraceValueId)) {
        use TraceInst::*;
        match self {
            Nop | LoadFrameSlot { .. } | Runtime { .. } => {},
            Copy { value } | Reinterpret { value } | CommitFrameSlot { value, .. } | GuardTruthy { value, .. }
            | GuardFalsey { value, .. } | GuardNotNull { value, .. } | GuardF64 { value, .. }
            | GuardInt32 { value, .. } | GuardType { value, .. } | IntUnary { value, .. }
            | F64Unary { value, .. } | IsNull { value } | BoxInt32 { value } | BoxF64 { value }
            | Int32ToF64 { value } | F64ToInt32 { value, .. } | TaggedToHeapHandle { value, .. }
            | ReadBarrier { value } => f(value),
            GuardBounds { index, length, .. } => { f(index); f(length); },
            GuardSwitch { tag, .. } => f(tag),
            GuardEq { lhs, rhs, .. } | IntBinary { lhs, rhs, .. } | IntBinaryChecked { lhs, rhs, .. }
            | F64Binary { lhs, rhs, .. } | IntCompare { lhs, rhs, .. } | F64Compare { lhs, rhs, .. }
            | RefEq { lhs, rhs } => { f(lhs); f(rhs); },
            Select { condition, if_true, if_false } => { f(condition); f(if_true); f(if_false); },
        }
    }
    pub fn replace_operand(&mut self, from: TraceValueId, to: TraceValueId) {
        self.visit_operands(|v| if *v == from { *v = to });
    }
    fn exit(&self) -> Option<SideExitId> {
        use TraceInst::*;
        match self {
            GuardTruthy { exit, .. } | GuardFalsey { exit, .. } | GuardNotNull { exit, .. }
            | GuardF64 { exit, .. } | GuardInt32 { exit, .. } | GuardType { exit, .. }
            | GuardBounds { exit, .. } | GuardSwitch { exit, .. } | GuardEq { exit, .. }
            | IntBinaryChecked { exit, .. } | F64ToInt32 { exit, .. } | TaggedToHeapHandle { exit, .. } => Some(*exit),
            _ => None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct TraceNode { pub id: usize, pub kind: TraceInst, pub results: Vec<TraceValueId>, pub source: TraceLocation }
#[derive(Clone, Copy, Debug)]
pub enum ExitReason { GuardTruthy, GuardFalsey, GuardValue, GuardNotNull, GuardInteger, GuardNumber, GuardBounds }
#[derive(Clone, Copy, Debug)]
pub enum ExitTarget { Interpreter(TraceLocation), Trace(TraceId), Return }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotValue { Constant(TraceConstant), Value(TraceValueId) }
#[derive(Clone, Debug)]
pub struct SnapshotWrite {
    pub slot: FrameSlot,
    pub kind: RegisterKind,
    /// Representation expected by the architectural register at the resume PC.
    ///
    /// The optimized trace value may use a different representation. For example,
    /// a Wren-visible Num can stay as raw F64 around a native loop while the
    /// interpreter expects a tagged Value when a guard exits. Native exit code
    /// materializes `value` into `target_rep` only on that cold path.
    pub target_rep: TraceRep,
    pub value: SnapshotValue,
}
#[derive(Clone, Debug)]
pub enum FrameDisposition { Existing { depth_from_top: u32 } }
#[derive(Clone, Debug)]
pub struct FrameSnapshot { pub disposition: FrameDisposition, pub writes: Vec<SnapshotWrite> }
#[derive(Clone, Debug)]
pub struct SideExit { pub id: SideExitId, pub target: ExitTarget, pub frames: Vec<FrameSnapshot>, pub linked_trace: Option<TraceId> }
#[derive(Clone, Debug)]
pub struct LoopPhi {
    pub value: TraceValueId,
    pub initial: TraceValueId,
    pub backedge: Option<TraceValueId>,
    pub rep: TraceRep,
    /// Side exit used by a representation-promoted phi to guard/unbox its tagged
    /// entry value once, before entering the native loop. `None` means the entry
    /// value already has `rep`.
    pub entry_exit: Option<SideExitId>,
}
#[derive(Clone, Copy, Debug)]
pub enum TraceKind { Loop }
#[derive(Clone, Debug)]
pub struct TraceAnchor { pub kind: TraceKind, pub location: TraceLocation }
#[derive(Clone, Debug)]
pub enum TraceTerminator { LoopBack, Exit(SideExitId), RuntimeReturn, Return, Link, Unreachable }
impl TraceTerminator { pub fn replace_operand(&mut self, _: TraceValueId, _: TraceValueId) {} }
#[derive(Clone, Debug, Default)]
pub struct TraceStats {
    pub recorded_bytecodes: usize,
    pub promoted_f64_loop_phis: usize,
    pub promoted_i32_loop_phis: usize,
    pub eliminated_guards: usize,
    pub eliminated_boxes: usize,
    pub eliminated_copies: usize,
}
#[derive(Clone, Debug)]
pub struct Trace {
    pub id: TraceId, pub anchor: TraceAnchor, pub values: Vec<TraceValue>, pub nodes: Vec<TraceNode>,
    pub exits: Vec<SideExit>, pub loop_phis: Vec<LoopPhi>, pub terminator: TraceTerminator, pub stats: TraceStats,
}
impl Trace {
    pub fn value(&self, v: TraceValueId) -> Option<&TraceValue> { self.values.get(v.0 as usize) }
    pub fn value_mut(&mut self, v: TraceValueId) -> Option<&mut TraceValue> { self.values.get_mut(v.0 as usize) }
    pub fn canonical_value(&self, mut v: TraceValueId) -> TraceValueId {
        for _ in 0..self.values.len() {
            match self.value(v).map(|v| &v.origin) {
                Some(TraceValueOrigin::Alias(next)) => v = *next,
                _ => break,
            }
        }
        v
    }
}

#[derive(Clone)]
pub struct TraceBuilder {
    trace: Trace, source: TraceLocation, slots: HashMap<FrameSlot, TraceValueId>,
    dirty: Vec<FrameSlot>, memory_epoch: bool,
}
impl TraceBuilder {
    pub fn new(id: TraceId, anchor: TraceAnchor) -> Self {
        Self { source: anchor.location, trace: Trace { id, anchor, values: vec![], nodes: vec![], exits: vec![],
            loop_phis: vec![], terminator: TraceTerminator::Unreachable, stats: TraceStats::default() },
            slots: HashMap::new(), dirty: vec![], memory_epoch: false }
    }
    pub fn trace(&self) -> &Trace { &self.trace }
    pub fn trace_mut(&mut self) -> &mut Trace { &mut self.trace }
    pub fn set_source(&mut self, source: TraceLocation) { self.source = source; }
    pub fn slot_value(&self, slot: FrameSlot) -> Option<TraceValueId> { self.slots.get(&slot).copied() }
    fn value(&mut self, ty: TraceType, rep: TraceRep, origin: TraceValueOrigin) -> TraceValueId {
        let id = TraceValueId(self.trace.values.len() as u32);
        self.trace.values.push(TraceValue { id, ty, rep, origin }); id
    }
    pub fn constant(&mut self, c: TraceConstant) -> TraceValueId {
        let (ty, rep) = c.layout(); self.value(ty, rep, TraceValueOrigin::Constant(c))
    }
    pub fn read_slot(&mut self, slot: FrameSlot, ty: TraceType, rep: TraceRep) -> TraceValueId {
        if let Some(v) = self.slot_value(slot) { return v; }
        let v = if self.memory_epoch {
            self.emit1(TraceInst::LoadFrameSlot { slot }, ty, rep)
        } else {
            self.value(ty, rep, TraceValueOrigin::EntrySlot { frame: TraceFrameId(0), slot })
        };
        self.slots.insert(slot, v); v
    }
    pub fn write_slot(&mut self, slot: FrameSlot, v: TraceValueId) {
        self.slots.insert(slot, v);
        if !self.dirty.contains(&slot) { self.dirty.push(slot); }
    }
    pub fn emit0(&mut self, kind: TraceInst) {
        let id = self.trace.nodes.len();
        self.trace.nodes.push(TraceNode { id, kind, results: vec![], source: self.source });
    }
    pub fn emit1(&mut self, kind: TraceInst, ty: TraceType, rep: TraceRep) -> TraceValueId {
        let node = self.trace.nodes.len();
        let v = self.value(ty, rep, TraceValueOrigin::InstResult { node });
        self.trace.nodes.push(TraceNode { id: node, kind, results: vec![v], source: self.source }); v
    }
    pub fn add_side_exit(&mut self, _reason: ExitReason, pc: BytecodePc) -> SideExitId {
        let writes = self.dirty.iter().map(|slot| {
            let value = self.slots[slot];
            let kind = match self.trace.value(value).unwrap().rep {
                TraceRep::Int32 | TraceRep::Bool => RegisterKind::Int32,
                TraceRep::F64 => RegisterKind::F64,
                _ => RegisterKind::Value,
            };
            SnapshotWrite { slot: *slot, kind, target_rep: self.trace.value(value).unwrap().rep, value: SnapshotValue::Value(value) }
        }).collect();
        let id = SideExitId(self.trace.exits.len() as u32);
        self.trace.exits.push(SideExit { id, target: ExitTarget::Interpreter(TraceLocation::new(self.source.function, pc)),
            frames: vec![FrameSnapshot { disposition: FrameDisposition::Existing { depth_from_top: 0 }, writes }], linked_trace: None }); id
    }
    pub fn add_loop_phi(&mut self, slot: FrameSlot, initial: TraceValueId, ty: TraceType, rep: TraceRep) -> TraceValueId {
        let value = self.value(ty, rep, TraceValueOrigin::LoopPhi { slot });
        self.trace.loop_phis.push(LoopPhi { value, initial, backedge: None, rep, entry_exit: None }); value
    }
    pub fn set_loop_backedge(&mut self, value: TraceValueId, backedge: TraceValueId) -> Result<(), String> {
        let phi = self.trace.loop_phis.iter_mut().find(|p| p.value == value).ok_or("unknown phi")?;
        phi.backedge = Some(backedge); Ok(())
    }
    pub fn commit(&mut self) {
        for slot in std::mem::take(&mut self.dirty) {
            self.emit0(TraceInst::CommitFrameSlot { slot, value: self.slots[&slot] });
        }
    }
    pub fn runtime(&mut self, location: TraceLocation, next: BytecodePc, terminal: bool) {
        self.commit();
        self.emit0(TraceInst::Runtime { location, next, terminal });
        self.slots.clear();
        self.memory_epoch = true;
    }
    pub fn has_runtime_calls(&self) -> bool { self.memory_epoch }
    pub fn ensure_loop_entries(&mut self) {
        for slot in self.dirty.clone() {
            if !self.trace.values.iter().any(|v| matches!(&v.origin, TraceValueOrigin::EntrySlot { slot: s, .. } if *s == slot)) {
                // A register written without being read still carries its previous
                // iteration value into exits which precede that write.
                self.value(TraceType::Any, TraceRep::Tagged,
                    TraceValueOrigin::EntrySlot { frame: TraceFrameId(0), slot });
            }
        }
    }
    /// Loops containing runtime calls reload entry registers on every iteration.
    /// This also prevents a value collected or changed by a helper from becoming
    /// a loop invariant in B3.
    pub fn prepare_memory_loop(&mut self) {
        self.commit();
        let mut prefix = vec![];
        for value in &mut self.trace.values {
            let slot = match &value.origin {
                TraceValueOrigin::EntrySlot { slot, .. } => Some(*slot),
                _ => None,
            };
            if let Some(slot) = slot {
                value.origin = TraceValueOrigin::InstResult { node: prefix.len() };
                prefix.push(TraceNode { id: prefix.len(), kind: TraceInst::LoadFrameSlot { slot },
                    results: vec![value.id], source: self.trace.anchor.location });
            }
        }
        prefix.append(&mut self.trace.nodes);
        for (id, node) in prefix.iter_mut().enumerate() {
            node.id = id;
            for value in &node.results { self.trace.values[value.0 as usize].origin = TraceValueOrigin::InstResult { node: id }; }
        }
        self.trace.nodes = prefix;
    }
    pub fn set_terminator(&mut self, term: TraceTerminator) { self.trace.terminator = term; }
    pub fn finish(self) -> Result<Trace, Vec<String>> { verify_trace(&self.trace)?; Ok(self.trace) }
}

/// Validate producers, use order, snapshots and loop representation boundaries
/// before handing a trace to the unsafe native backend.
pub fn verify_trace(t: &Trace) -> Result<(), Vec<String>> {
    let mut errors = vec![];
    let mut available = vec![false; t.values.len()];
    for (index, value) in t.values.iter().enumerate() {
        if value.id.0 as usize != index { errors.push("noncanonical value ID".into()); }
        available[index] = matches!(&value.origin, TraceValueOrigin::EntrySlot { .. } | TraceValueOrigin::Constant(_) | TraceValueOrigin::LoopPhi { .. });
    }
    // Guarded entries execute in the native preheader, after ordinary entry loads
    // and before the loop. Their source therefore must itself be preheader-safe.
    for (index, value) in t.values.iter().enumerate() {
        if let TraceValueOrigin::GuardedEntry { value: source, .. } = &value.origin {
            let source = t.canonical_value(*source);
            if available.get(source.0 as usize).copied().unwrap_or(false)
                && t.value(source).is_some_and(|v| v.rep == TraceRep::Tagged)
                && matches!(value.rep, TraceRep::F64 | TraceRep::Int32)
            {
                available[index] = true;
            } else {
                errors.push("invalid guarded entry value".into());
            }
        }
    }
    let check_exit = |id: SideExitId, available: &[bool], errors: &mut Vec<String>| {
        if let Some(exit) = t.exits.get(id.0 as usize) {
            for frame in &exit.frames { for write in &frame.writes {
                if let SnapshotValue::Value(v) = write.value {
                    if !available.get(v.0 as usize).copied().unwrap_or(false) { errors.push(format!("exit {id:?} uses unavailable value {v:?}")); }
                }
            }}
        } else { errors.push(format!("unknown exit {id:?}")); }
    };
    for value in &t.values {
        if let TraceValueOrigin::GuardedEntry { exit, .. } = &value.origin {
            check_exit(*exit, &available, &mut errors);
        }
    }
    for (index, node) in t.nodes.iter().enumerate() {
        if node.id != index { errors.push("noncanonical node ID".into()); }
        node.kind.clone().visit_operands(|v| {
            if !available.get(v.0 as usize).copied().unwrap_or(false) { errors.push(format!("node {index} uses unavailable value {v:?}")); }
        });
        if let Some(exit) = node.kind.exit() { check_exit(exit, &available, &mut errors); }
        for result in &node.results {
            if let Some(v) = t.value(*result) {
                match &v.origin {
                    TraceValueOrigin::InstResult { node } if *node == index => {
                        available[result.0 as usize] = true;
                    }
                    TraceValueOrigin::Alias(target) => {
                        let target = t.canonical_value(*target);
                        if available.get(target.0 as usize).copied().unwrap_or(false) {
                            available[result.0 as usize] = true;
                        } else {
                            errors.push("aliased result targets unavailable value".into());
                        }
                    }
                    _ => errors.push("result has wrong producer".into()),
                }
            } else { errors.push("unknown result".into()); }
        }
    }
    for phi in &t.loop_phis {
        let initial_ok = t.value(phi.initial).is_some_and(|v| {
            available[v.id.0 as usize]
                && (v.rep == phi.rep
                    || (phi.entry_exit.is_some()
                        && v.rep == TraceRep::Tagged
                        && matches!(phi.rep, TraceRep::F64 | TraceRep::Int32)))
        });
        let backedge_ok = phi.backedge.and_then(|v| t.value(v)).is_some_and(|v| {
            v.rep == phi.rep && available[v.id.0 as usize]
        });
        if !initial_ok || !backedge_ok {
            errors.push("invalid loop phi".into());
        }
        if let Some(exit) = phi.entry_exit {
            check_exit(exit, &available, &mut errors);
        }
    }
    match t.terminator {
        TraceTerminator::Exit(id) => check_exit(id, &available, &mut errors),
        TraceTerminator::Unreachable => errors.push("missing terminator".into()),
        _ => {},
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Wren-specific trace canonicalization performed before the generic B3 pipeline.
///
/// The highest-value transformation is representation promotion for loop-carried
/// Wren numbers:
///
/// ```text
/// Tagged phi -> GuardF64 -> arithmetic -> BoxF64 -> Tagged backedge
/// ```
///
/// becomes:
///
/// ```text
/// entry: GuardF64 once
/// loop:  F64 phi -> arithmetic -> F64 backedge
/// ```
///
/// Side exits retain their original architectural representation and therefore
/// re-box only on cold exit blocks.
#[derive(Default)]
pub struct TraceOptimizer;

impl TraceOptimizer {
    pub fn optimize(&self, trace: &mut Trace) -> Result<(), Vec<String>> {
        verify_trace(trace)?;

        // Runtime calls are synchronization boundaries. Representation promotion
        // is intentionally restricted to pure native loops for now.
        let has_runtime = trace.nodes.iter().any(|node| matches!(&node.kind, TraceInst::Runtime { .. }));

        self.cancel_box_guard_pairs(trace);
        if !has_runtime {
            self.hoist_invariant_numeric_guards(trace);
            self.promote_loop_representations(trace);
            self.cancel_box_guard_pairs(trace);
        }
        self.remove_trivial_copies(trace);
        self.remove_dead_box_nodes(trace);
        self.canonicalize_references(trace);

        verify_trace(trace)
    }

    fn producer(trace: &Trace, value: TraceValueId) -> Option<(usize, TraceInst)> {
        let value = trace.canonical_value(value);
        let node = match &trace.value(value)?.origin {
            TraceValueOrigin::InstResult { node } => *node,
            _ => return None,
        };
        Some((node, trace.nodes.get(node)?.kind.clone()))
    }

    fn alias_value(trace: &mut Trace, from: TraceValueId, to: TraceValueId) {
        let to = trace.canonical_value(to);
        if from == to { return; }
        if let Some(value) = trace.value_mut(from) {
            value.origin = TraceValueOrigin::Alias(to);
        }
        for node in &mut trace.nodes {
            node.kind.replace_operand(from, to);
        }
        for exit in &mut trace.exits {
            for frame in &mut exit.frames {
                for write in &mut frame.writes {
                    if write.value == SnapshotValue::Value(from) {
                        write.value = SnapshotValue::Value(to);
                    }
                }
            }
        }
        for phi in &mut trace.loop_phis {
            if phi.initial == from { phi.initial = to; }
            if phi.backedge == Some(from) { phi.backedge = Some(to); }
        }
    }

    fn cancel_box_guard_pairs(&self, trace: &mut Trace) {
        // Iterate to a small fixed point because eliminating one pair can expose
        // another around a loop backedge.
        for _ in 0..4 {
            let mut changed = false;
            for index in 0..trace.nodes.len() {
                let kind = trace.nodes[index].kind.clone();
                let Some(result) = trace.nodes[index].results.first().copied() else { continue; };
                match kind {
                    TraceInst::GuardF64 { value, .. } => {
                        if let Some((_, TraceInst::BoxF64 { value: raw })) = Self::producer(trace, value) {
                            Self::alias_value(trace, result, raw);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_guards += 1;
                            changed = true;
                        }
                    }
                    TraceInst::GuardInt32 { value, .. } => {
                        if let Some((_, TraceInst::BoxInt32 { value: raw })) = Self::producer(trace, value) {
                            Self::alias_value(trace, result, raw);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_guards += 1;
                            changed = true;
                        }
                    }
                    TraceInst::BoxF64 { value } => {
                        if let Some((_, TraceInst::GuardF64 { value: tagged, .. })) = Self::producer(trace, value) {
                            Self::alias_value(trace, result, tagged);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_boxes += 1;
                            changed = true;
                        }
                    }
                    TraceInst::BoxInt32 { value } => {
                        if let Some((_, TraceInst::GuardInt32 { value: tagged, .. })) = Self::producer(trace, value) {
                            Self::alias_value(trace, result, tagged);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_boxes += 1;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
            if !changed { break; }
        }
    }

    fn hoist_invariant_numeric_guards(&self, trace: &mut Trace) {
        #[derive(Clone, Copy)]
        struct GuardCandidate {
            node: usize,
            result: TraceValueId,
            source: TraceValueId,
            rep: TraceRep,
        }

        let mut candidates = Vec::new();
        for (node_index, node) in trace.nodes.iter().enumerate() {
            let Some(result) = node.results.first().copied() else { continue; };
            let (source, rep) = match &node.kind {
                TraceInst::GuardF64 { value, .. } => (trace.canonical_value(*value), TraceRep::F64),
                TraceInst::GuardInt32 { value, .. } => (trace.canonical_value(*value), TraceRep::Int32),
                _ => continue,
            };
            let Some(source_value) = trace.value(source) else { continue; };
            if !matches!(&source_value.origin, TraceValueOrigin::EntrySlot { frame, .. } if frame.0 == 0) {
                continue;
            }
            if source_value.rep != TraceRep::Tagged { continue; }
            candidates.push(GuardCandidate { node: node_index, result, source, rep });
        }

        let mut hoisted = HashMap::<(TraceValueId, TraceRep), TraceValueId>::new();
        for candidate in candidates {
            let key = (candidate.source, candidate.rep);
            let guarded = if let Some(value) = hoisted.get(&key).copied() {
                value
            } else {
                let exit = SideExitId(trace.exits.len() as u32);
                trace.exits.push(SideExit {
                    id: exit,
                    target: ExitTarget::Interpreter(trace.anchor.location),
                    frames: vec![FrameSnapshot {
                        disposition: FrameDisposition::Existing { depth_from_top: 0 },
                        writes: Vec::new(),
                    }],
                    linked_trace: None,
                });
                let id = TraceValueId(trace.values.len() as u32);
                trace.values.push(TraceValue {
                    id,
                    ty: match candidate.rep {
                        TraceRep::F64 => TraceType::F64,
                        TraceRep::Int32 => TraceType::Int32,
                        _ => unreachable!(),
                    },
                    rep: candidate.rep,
                    origin: TraceValueOrigin::GuardedEntry {
                        value: candidate.source,
                        exit,
                    },
                });
                hoisted.insert(key, id);
                id
            };

            Self::alias_value(trace, candidate.result, guarded);
            trace.nodes[candidate.node].kind = TraceInst::Nop;
            trace.stats.eliminated_guards += 1;
        }
    }

    fn promote_loop_representations(&self, trace: &mut Trace) {
        #[derive(Clone, Copy)]
        struct Candidate {
            phi_index: usize,
            phi_value: TraceValueId,
            boxed_backedge: TraceValueId,
            raw_backedge: TraceValueId,
            rep: TraceRep,
        }

        let mut candidates = Vec::new();
        for (phi_index, phi) in trace.loop_phis.iter().enumerate() {
            if phi.rep != TraceRep::Tagged || phi.entry_exit.is_some() { continue; }
            let Some(backedge) = phi.backedge.map(|v| trace.canonical_value(v)) else { continue; };
            let Some((_, producer)) = Self::producer(trace, backedge) else { continue; };
            let (raw_backedge, rep) = match producer {
                TraceInst::BoxF64 { value } => (trace.canonical_value(value), TraceRep::F64),
                TraceInst::BoxInt32 { value } => (trace.canonical_value(value), TraceRep::Int32),
                _ => continue,
            };
            if trace.value(raw_backedge).map(|v| v.rep) != Some(rep) { continue; }

            // Every hot-path use of the tagged phi must be the matching unbox
            // guard. Snapshot uses are fine: they remember target_rep=Tagged and
            // are re-boxed only when the side exit is actually taken.
            let mut compatible = true;
            for node in &trace.nodes {
                let mut saw = false;
                let mut cloned = node.kind.clone();
                cloned.visit_operands(|operand| {
                    if trace.canonical_value(*operand) == phi.value { saw = true; }
                });
                if !saw { continue; }
                compatible &= match (&node.kind, rep) {
                    (TraceInst::GuardF64 { value, .. }, TraceRep::F64) => trace.canonical_value(*value) == phi.value,
                    (TraceInst::GuardInt32 { value, .. }, TraceRep::Int32) => trace.canonical_value(*value) == phi.value,
                    _ => false,
                };
                if !compatible { break; }
            }
            if !compatible { continue; }

            candidates.push(Candidate {
                phi_index,
                phi_value: phi.value,
                boxed_backedge: backedge,
                raw_backedge,
                rep,
            });
        }

        for candidate in candidates {
            // A guard failure before the first native iteration has not mutated
            // architectural state, so the correct resume point is simply the root
            // trace header and no register reconstruction is required.
            let entry_exit = SideExitId(trace.exits.len() as u32);
            trace.exits.push(SideExit {
                id: entry_exit,
                target: ExitTarget::Interpreter(trace.anchor.location),
                frames: vec![FrameSnapshot {
                    disposition: FrameDisposition::Existing { depth_from_top: 0 },
                    writes: Vec::new(),
                }],
                linked_trace: None,
            });

            {
                let phi = &mut trace.loop_phis[candidate.phi_index];
                phi.rep = candidate.rep;
                phi.backedge = Some(candidate.raw_backedge);
                phi.entry_exit = Some(entry_exit);
            }
            if let Some(value) = trace.value_mut(candidate.phi_value) {
                value.rep = candidate.rep;
                value.ty = match candidate.rep {
                    TraceRep::F64 => TraceType::F64,
                    TraceRep::Int32 => TraceType::Int32,
                    _ => value.ty,
                };
            }

            // Remove the per-iteration guard/unbox. The entry guard installed on
            // the LoopPhi establishes the invariant once.
            for index in 0..trace.nodes.len() {
                let kind = trace.nodes[index].kind.clone();
                let Some(result) = trace.nodes[index].results.first().copied() else { continue; };
                let matches = match kind {
                    TraceInst::GuardF64 { value, .. } if candidate.rep == TraceRep::F64 => {
                        trace.canonical_value(value) == candidate.phi_value
                    }
                    TraceInst::GuardInt32 { value, .. } if candidate.rep == TraceRep::Int32 => {
                        trace.canonical_value(value) == candidate.phi_value
                    }
                    _ => false,
                };
                if matches {
                    Self::alias_value(trace, result, candidate.phi_value);
                    trace.nodes[index].kind = TraceInst::Nop;
                    trace.stats.eliminated_guards += 1;
                }
            }

            // Any exit that used the boxed backedge can materialize the raw value
            // directly into its original Tagged target representation.
            let canonical_values: Vec<_> = (0..trace.values.len())
                .map(|index| trace.canonical_value(TraceValueId(index as u32)))
                .collect();
            for exit in &mut trace.exits {
                for frame in &mut exit.frames {
                    for write in &mut frame.writes {
                        if let SnapshotValue::Value(value) = write.value {
                            if canonical_values[value.0 as usize] == candidate.boxed_backedge {
                                write.value = SnapshotValue::Value(candidate.raw_backedge);
                            }
                        }
                    }
                }
            }

            match candidate.rep {
                TraceRep::F64 => trace.stats.promoted_f64_loop_phis += 1,
                TraceRep::Int32 => trace.stats.promoted_i32_loop_phis += 1,
                _ => {}
            }
        }
    }

    fn remove_trivial_copies(&self, trace: &mut Trace) {
        for index in 0..trace.nodes.len() {
            let kind = trace.nodes[index].kind.clone();
            let Some(result) = trace.nodes[index].results.first().copied() else { continue; };
            match kind {
                TraceInst::Copy { value } => {
                    let value = trace.canonical_value(value);
                    if trace.value(value).map(|v| v.rep) == trace.value(result).map(|v| v.rep) {
                        Self::alias_value(trace, result, value);
                        trace.nodes[index].kind = TraceInst::Nop;
                        trace.stats.eliminated_copies += 1;
                    }
                }
                TraceInst::Reinterpret { value } => {
                    let value = trace.canonical_value(value);
                    if trace.value(value).map(|v| v.rep) == trace.value(result).map(|v| v.rep) {
                        Self::alias_value(trace, result, value);
                        trace.nodes[index].kind = TraceInst::Nop;
                        trace.stats.eliminated_copies += 1;
                    }
                }
                _ => {}
            }
        }
    }

    fn used_values(trace: &Trace) -> HashSet<TraceValueId> {
        let mut used = HashSet::new();
        for node in &trace.nodes {
            let mut kind = node.kind.clone();
            kind.visit_operands(|value| { used.insert(trace.canonical_value(*value)); });
        }
        for exit in &trace.exits {
            for frame in &exit.frames {
                for write in &frame.writes {
                    if let SnapshotValue::Value(value) = write.value {
                        used.insert(trace.canonical_value(value));
                    }
                }
            }
        }
        for phi in &trace.loop_phis {
            used.insert(trace.canonical_value(phi.initial));
            if let Some(backedge) = phi.backedge { used.insert(trace.canonical_value(backedge)); }
        }
        used
    }

    fn remove_dead_box_nodes(&self, trace: &mut Trace) {
        let used = Self::used_values(trace);
        let canonical_values: Vec<_> = (0..trace.values.len())
            .map(|index| trace.canonical_value(TraceValueId(index as u32)))
            .collect();
        let mut eliminated_boxes = 0usize;
        for node in &mut trace.nodes {
            let Some(result) = node.results.first().copied() else { continue; };
            if used.contains(&canonical_values[result.0 as usize]) { continue; }
            if matches!(&node.kind, TraceInst::BoxF64 { .. } | TraceInst::BoxInt32 { .. } | TraceInst::Copy { .. } | TraceInst::Reinterpret { .. }) {
                if matches!(&node.kind, TraceInst::BoxF64 { .. } | TraceInst::BoxInt32 { .. }) {
                    eliminated_boxes += 1;
                }
                node.kind = TraceInst::Nop;
            }
        }
        trace.stats.eliminated_boxes += eliminated_boxes;
    }

    fn canonicalize_references(&self, trace: &mut Trace) {
        // Snapshot references are particularly important because the B3 slow-path
        // generator lowers them out-of-line after the main loop is built.
        let aliases: Vec<_> = (0..trace.values.len())
            .map(|index| {
                let id = TraceValueId(index as u32);
                (id, trace.canonical_value(id))
            })
            .collect();
        for (from, to) in aliases {
            if from == to { continue; }
            for node in &mut trace.nodes { node.kind.replace_operand(from, to); }
            for exit in &mut trace.exits {
                for frame in &mut exit.frames {
                    for write in &mut frame.writes {
                        if write.value == SnapshotValue::Value(from) { write.value = SnapshotValue::Value(to); }
                    }
                }
            }
            for phi in &mut trace.loop_phis {
                if phi.initial == from { phi.initial = to; }
                if phi.backedge == Some(from) { phi.backedge = Some(to); }
            }
        }
    }
}

#[cfg(test)]
mod optimizer_tests {
    use super::*;

    #[test]
    fn promotes_tagged_numeric_loop_phi_to_f64() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(10));
        let mut builder = TraceBuilder::new(
            TraceId(0),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(0));
        let initial = builder.read_slot(slot, TraceType::Num, TraceRep::Tagged);
        let phi = builder.add_loop_phi(slot, initial, TraceType::Num, TraceRep::Tagged);
        let guard_exit = builder.add_side_exit(ExitReason::GuardNumber, anchor.pc);
        let raw = builder.emit1(
            TraceInst::GuardF64 { value: phi, exit: guard_exit },
            TraceType::F64,
            TraceRep::F64,
        );
        let one = builder.constant(TraceConstant::F64Bits(1.0f64.to_bits()));
        let next = builder.emit1(
            TraceInst::F64Binary { op: F64BinOp::Add, lhs: raw, rhs: one },
            TraceType::F64,
            TraceRep::F64,
        );
        let boxed = builder.emit1(
            TraceInst::BoxF64 { value: next },
            TraceType::Num,
            TraceRep::Tagged,
        );
        builder.write_slot(slot, boxed);
        let loop_exit = builder.add_side_exit(ExitReason::GuardValue, anchor.pc);
        builder.set_loop_backedge(phi, boxed).unwrap();
        let always_true = builder.constant(TraceConstant::Bool(true));
        builder.emit0(TraceInst::GuardTruthy { value: always_true, exit: loop_exit });
        builder.set_terminator(TraceTerminator::LoopBack);

        let mut trace = builder.finish().unwrap();
        TraceOptimizer::default().optimize(&mut trace).unwrap();

        let phi_desc = &trace.loop_phis[0];
        assert_eq!(phi_desc.rep, TraceRep::F64);
        assert!(phi_desc.entry_exit.is_some());
        assert_eq!(trace.value(phi).unwrap().rep, TraceRep::F64);
        assert_eq!(trace.canonical_value(phi_desc.backedge.unwrap()), next);
        assert!(trace.stats.promoted_f64_loop_phis >= 1);
        assert!(trace.stats.eliminated_guards >= 1);
        assert!(trace.stats.eliminated_boxes >= 1);

        let reconstructed = trace.exits.iter().flat_map(|exit| &exit.frames).flat_map(|frame| &frame.writes)
            .find(|write| write.slot == slot && write.value == SnapshotValue::Value(next));
        assert!(reconstructed.is_some());
        assert_eq!(reconstructed.unwrap().target_rep, TraceRep::Tagged);
    }

    #[test]
    fn promotes_tagged_integer_loop_phi_to_i32() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(20));
        let mut builder = TraceBuilder::new(
            TraceId(1),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(1));
        let initial = builder.read_slot(slot, TraceType::Num, TraceRep::Tagged);
        let phi = builder.add_loop_phi(slot, initial, TraceType::Num, TraceRep::Tagged);
        let guard_exit = builder.add_side_exit(ExitReason::GuardInteger, anchor.pc);
        let raw = builder.emit1(
            TraceInst::GuardInt32 { value: phi, exit: guard_exit },
            TraceType::Int32,
            TraceRep::Int32,
        );
        let one = builder.constant(TraceConstant::Int32(1));
        let next = builder.emit1(
            TraceInst::IntBinary { op: IntBinOp::Add, lhs: raw, rhs: one },
            TraceType::Int32,
            TraceRep::Int32,
        );
        let boxed = builder.emit1(
            TraceInst::BoxInt32 { value: next },
            TraceType::Num,
            TraceRep::Tagged,
        );
        builder.write_slot(slot, boxed);
        builder.set_loop_backedge(phi, boxed).unwrap();
        builder.set_terminator(TraceTerminator::LoopBack);

        let mut trace = builder.finish().unwrap();
        TraceOptimizer::default().optimize(&mut trace).unwrap();

        assert_eq!(trace.loop_phis[0].rep, TraceRep::Int32);
        assert!(trace.loop_phis[0].entry_exit.is_some());
        assert_eq!(trace.canonical_value(trace.loop_phis[0].backedge.unwrap()), next);
        assert!(trace.stats.promoted_i32_loop_phis >= 1);
    }
}
