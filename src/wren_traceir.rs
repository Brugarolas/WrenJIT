//! Trace SSA, architectural snapshots, and the runtime synchronization boundary.
//! Runtime calls commit registers before invoking the VM. Values read afterwards
//! are fresh loads: no heap or register assumption survives a runtime call.

use std::collections::HashMap;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    EntrySlot { frame: TraceFrameId, slot: FrameSlot }, Constant(TraceConstant),
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
pub struct SnapshotWrite { pub slot: FrameSlot, pub kind: RegisterKind, pub value: SnapshotValue }
#[derive(Clone, Debug)]
pub enum FrameDisposition { Existing { depth_from_top: u32 } }
#[derive(Clone, Debug)]
pub struct FrameSnapshot { pub disposition: FrameDisposition, pub writes: Vec<SnapshotWrite> }
#[derive(Clone, Debug)]
pub struct SideExit { pub id: SideExitId, pub target: ExitTarget, pub frames: Vec<FrameSnapshot>, pub linked_trace: Option<TraceId> }
#[derive(Clone, Debug)]
pub struct LoopPhi { pub value: TraceValueId, pub initial: TraceValueId, pub backedge: Option<TraceValueId>, pub rep: TraceRep }
#[derive(Clone, Copy, Debug)]
pub enum TraceKind { Loop }
#[derive(Clone, Debug)]
pub struct TraceAnchor { pub kind: TraceKind, pub location: TraceLocation }
#[derive(Clone, Debug)]
pub enum TraceTerminator { LoopBack, Exit(SideExitId), RuntimeReturn, Return, Link, Unreachable }
impl TraceTerminator { pub fn replace_operand(&mut self, _: TraceValueId, _: TraceValueId) {} }
#[derive(Clone, Debug, Default)]
pub struct TraceStats { pub recorded_bytecodes: usize }
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
            SnapshotWrite { slot: *slot, kind, value: SnapshotValue::Value(value) }
        }).collect();
        let id = SideExitId(self.trace.exits.len() as u32);
        self.trace.exits.push(SideExit { id, target: ExitTarget::Interpreter(TraceLocation::new(self.source.function, pc)),
            frames: vec![FrameSnapshot { disposition: FrameDisposition::Existing { depth_from_top: 0 }, writes }], linked_trace: None }); id
    }
    pub fn add_loop_phi(&mut self, slot: FrameSlot, initial: TraceValueId, ty: TraceType, rep: TraceRep) -> TraceValueId {
        let value = self.value(ty, rep, TraceValueOrigin::LoopPhi { slot });
        self.trace.loop_phis.push(LoopPhi { value, initial, backedge: None, rep }); value
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
            if !self.trace.values.iter().any(|v| matches!(v.origin, TraceValueOrigin::EntrySlot { slot: s, .. } if s == slot)) {
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
            if let TraceValueOrigin::EntrySlot { slot, .. } = value.origin {
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
        available[index] = matches!(value.origin, TraceValueOrigin::EntrySlot { .. } | TraceValueOrigin::Constant(_) | TraceValueOrigin::LoopPhi { .. });
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
    for (index, node) in t.nodes.iter().enumerate() {
        if node.id != index { errors.push("noncanonical node ID".into()); }
        node.kind.clone().visit_operands(|v| {
            if !available.get(v.0 as usize).copied().unwrap_or(false) { errors.push(format!("node {index} uses unavailable value {v:?}")); }
        });
        if let Some(exit) = node.kind.exit() { check_exit(exit, &available, &mut errors); }
        for result in &node.results {
            if let Some(v) = t.value(*result) {
                if !matches!(v.origin, TraceValueOrigin::InstResult { node } if node == index) { errors.push("result has wrong producer".into()); }
                available[result.0 as usize] = true;
            } else { errors.push("unknown result".into()); }
        }
    }
    for phi in &t.loop_phis {
        for v in [Some(phi.initial), phi.backedge] {
            if !v.and_then(|v| t.value(v)).is_some_and(|v| v.rep == phi.rep && available[v.id.0 as usize]) {
                errors.push("invalid loop phi".into());
            }
        }
    }
    match t.terminator {
        TraceTerminator::Exit(id) => check_exit(id, &available, &mut errors),
        TraceTerminator::Unreachable => errors.push("missing terminator".into()),
        _ => {},
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// The backend performs CSE, constant folding, DCE and loop optimizations.
#[derive(Default)]
pub struct TraceOptimizer;
impl TraceOptimizer {
    pub fn optimize(&self, trace: &mut Trace) -> Result<(), Vec<String>> { verify_trace(trace) }
}
