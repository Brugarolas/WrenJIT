//! Trace SSA, architectural snapshots, and the runtime synchronization boundary.
//! Full runtime calls commit registers before invoking the VM. Values read afterwards
//! are fresh loads: no heap or register assumption survives a full runtime call.
//! Non-GC leaf runtime calls are also supported: they synchronize dirty architectural
//! registers but preserve unaffected SSA values across the call. Runtime-written result
//! slots are explicitly reloaded afterwards.

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
pub enum TraceRep { Tagged, Int32, UInt32, F64, Bool, HeapHandle, RawPtr }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceType { Any, Null, Bool, Num, Int32, UInt32, F64 }
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
    /// invariant loop inputs that would otherwise repeat GuardF64/GuardInt32/GuardUInt32.
    GuardedEntry { value: TraceValueId, exit: SideExitId },
    Constant(TraceConstant),
    InstResult { node: usize }, LoopPhi { slot: FrameSlot },
    /// Optimizer-created loop state without an architectural register slot.
    LoopTemporary,
    Alias(TraceValueId),
}
#[derive(Clone, Debug)]
pub struct TraceValue { pub id: TraceValueId, pub ty: TraceType, pub rep: TraceRep, pub origin: TraceValueOrigin }
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TraceSwitchGuard { Case(i64), Default { excluded: Vec<i64> } }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceMathOp { Abs, Floor, Ceil, Sqrt, Round, Sin, Cos, Tan, Log, Exp }
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    GuardUInt32 { value: TraceValueId, exit: SideExitId },
    GuardBounds { index: TraceValueId, length: TraceValueId, exit: SideExitId },
    GuardEq { lhs: TraceValueId, rhs: TraceValueId, exit: SideExitId },
    GuardSwitch { tag: TraceValueId, observed: TraceSwitchGuard, exit: SideExitId },
    GuardType { value: TraceValueId, expected: TraceType, exit: SideExitId },
    IntUnary { op: IntUnaryOp, value: TraceValueId },
    IntBinary { op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId },
    IntBinaryChecked { op: IntBinOp, lhs: TraceValueId, rhs: TraceValueId, exit: SideExitId },
    F64Unary { op: F64UnaryOp, value: TraceValueId },
    F64Math { op: TraceMathOp, value: TraceValueId },
    F64Binary { op: F64BinOp, lhs: TraceValueId, rhs: TraceValueId },
    IntCompare { op: CmpKind, lhs: TraceValueId, rhs: TraceValueId },
    F64Compare { op: CmpKind, lhs: TraceValueId, rhs: TraceValueId },
    RefEq { lhs: TraceValueId, rhs: TraceValueId },
    IsNull { value: TraceValueId },
    Select { condition: TraceValueId, if_true: TraceValueId, if_false: TraceValueId },
    BoxInt32 { value: TraceValueId }, BoxUInt32 { value: TraceValueId }, BoxF64 { value: TraceValueId },
    Int32ToF64 { value: TraceValueId }, UInt32ToF64 { value: TraceValueId }, F64ToInt32 { value: TraceValueId, exit: SideExitId },
    TaggedToHeapHandle { value: TraceValueId, exit: SideExitId },
    HeapLoadField { object: TraceValueId, slot: u32, exit: SideExitId },
    HeapStoreField { object: TraceValueId, value: TraceValueId, slot: u32, exit: SideExitId },
    HeapLoadCell { cell: TraceValueId, exit: SideExitId },
    HeapStoreCell { cell: TraceValueId, value: TraceValueId, exit: SideExitId },
    HeapListLen { list: TraceValueId, exit: SideExitId },
    HeapListGet { list: TraceValueId, index: TraceValueId, exit: SideExitId },
    HeapListSet { list: TraceValueId, index: TraceValueId, value: TraceValueId, exit: SideExitId },
    HeapMapLen { map: TraceValueId, exit: SideExitId },
    HeapStringLen { string: TraceValueId, exit: SideExitId },
    ReadBarrier { value: TraceValueId },
}
impl TraceInst {
    pub fn visit_operands(&mut self, mut f: impl FnMut(&mut TraceValueId)) {
        use TraceInst::*;
        match self {
            Nop | LoadFrameSlot { .. } | Runtime { .. } => {},
            Copy { value } | Reinterpret { value } | CommitFrameSlot { value, .. } | GuardTruthy { value, .. }
            | GuardFalsey { value, .. } | GuardNotNull { value, .. } | GuardF64 { value, .. }
            | GuardInt32 { value, .. } | GuardUInt32 { value, .. } | GuardType { value, .. } | IntUnary { value, .. }
            | F64Unary { value, .. } | F64Math { value, .. } | IsNull { value } | BoxInt32 { value } | BoxUInt32 { value } | BoxF64 { value }
            | Int32ToF64 { value } | UInt32ToF64 { value } | F64ToInt32 { value, .. } | TaggedToHeapHandle { value, .. }
            | ReadBarrier { value }
            | HeapLoadField { object: value, .. }
            | HeapLoadCell { cell: value, .. }
            | HeapListLen { list: value, .. }
            | HeapMapLen { map: value, .. }
            | HeapStringLen { string: value, .. } => f(value),
            GuardBounds { index, length, .. } => { f(index); f(length); },
            GuardSwitch { tag, .. } => f(tag),
            GuardEq { lhs, rhs, .. } | IntBinary { lhs, rhs, .. } | IntBinaryChecked { lhs, rhs, .. }
            | F64Binary { lhs, rhs, .. } | IntCompare { lhs, rhs, .. } | F64Compare { lhs, rhs, .. }
            | RefEq { lhs, rhs }
            | HeapStoreField { object: lhs, value: rhs, .. }
            | HeapStoreCell { cell: lhs, value: rhs, .. }
            | HeapListGet { list: lhs, index: rhs, .. } => { f(lhs); f(rhs); },
            HeapListSet { list, index, value, .. } => { f(list); f(index); f(value); },
            Select { condition, if_true, if_false } => { f(condition); f(if_true); f(if_false); },
        }
    }
    pub fn replace_operand(&mut self, from: TraceValueId, to: TraceValueId) {
        self.visit_operands(|v| if *v == from { *v = to });
    }
    /// Pure, total operations that can execute before any body guard.
    /// Integer division/remainder may trap; loads, guards and barriers observe
    /// runtime state and must remain at their original execution point.
    pub fn is_speculatable(&self) -> bool {
        use TraceInst::*;
        matches!(self,
            Copy { .. } | Reinterpret { .. } | IntUnary { .. }
            | IntBinary { op: IntBinOp::Add | IntBinOp::Sub | IntBinOp::Mul
                | IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor
                | IntBinOp::Shl | IntBinOp::ShrU | IntBinOp::Sar, .. }
            | F64Unary { .. } | F64Math { .. } | F64Binary { .. } | IntCompare { .. }
            | F64Compare { .. } | RefEq { .. } | IsNull { .. } | Select { .. }
            | BoxInt32 { .. } | BoxUInt32 { .. } | BoxF64 { .. }
            | Int32ToF64 { .. } | UInt32ToF64 { .. })
    }

    pub fn exit(&self) -> Option<SideExitId> {
        use TraceInst::*;
        match self {
            GuardTruthy { exit, .. } | GuardFalsey { exit, .. } | GuardNotNull { exit, .. }
            | GuardF64 { exit, .. } | GuardInt32 { exit, .. } | GuardUInt32 { exit, .. } | GuardType { exit, .. }
            | GuardBounds { exit, .. } | GuardSwitch { exit, .. } | GuardEq { exit, .. }
            | IntBinaryChecked { exit, .. } | F64ToInt32 { exit, .. } | TaggedToHeapHandle { exit, .. }
            | HeapLoadField { exit, .. } | HeapStoreField { exit, .. }
            | HeapLoadCell { exit, .. } | HeapStoreCell { exit, .. }
            | HeapListLen { exit, .. } | HeapListGet { exit, .. } | HeapListSet { exit, .. }
            | HeapMapLen { exit, .. } | HeapStringLen { exit, .. } => Some(*exit),
            _ => None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct TraceNode { pub id: usize, pub kind: TraceInst, pub results: Vec<TraceValueId>, pub source: TraceLocation }
#[derive(Clone, Copy, Debug)]
pub enum ExitReason { GuardTruthy, GuardFalsey, GuardValue, GuardNotNull, GuardInteger, GuardUnsignedInteger, GuardNumber, GuardBounds }
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
pub enum TraceKind { Loop, Side }
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
    pub promoted_u32_loop_phis: usize,
    pub eliminated_guards: usize,
    pub eliminated_boxes: usize,
    pub eliminated_copies: usize,
    pub pruned_snapshot_writes: usize,
    pub gvn_eliminated: usize,
    pub licm_hoisted: usize,
    pub pre_eliminated: usize,
}
#[derive(Clone, Debug)]
pub struct Trace {
    pub id: TraceId, pub anchor: TraceAnchor, pub values: Vec<TraceValue>, pub nodes: Vec<TraceNode>,
    /// Stable node IDs executed once, in this order, before initializing loop phis.
    /// All other nodes execute in arena order in the trace body.
    pub preheader: Vec<usize>,
    pub exits: Vec<SideExit>, pub loop_phis: Vec<LoopPhi>,
    /// Synthetic side exit used by the native-loop execution budget. It resumes
    /// at the trace anchor with the end-of-iteration architectural state.
    pub native_budget_exit: Option<SideExitId>,
    pub terminator: TraceTerminator, pub stats: TraceStats,
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
        Self { source: anchor.location, trace: Trace { id, anchor, values: vec![], nodes: vec![], preheader: vec![], exits: vec![],
            loop_phis: vec![], native_budget_exit: None, terminator: TraceTerminator::Unreachable, stats: TraceStats::default() },
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
                TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => RegisterKind::Int32,
                TraceRep::F64 => RegisterKind::F64,
                _ => RegisterKind::Value,
            };
            SnapshotWrite { slot: *slot, kind, target_rep: self.trace.value(value).unwrap().rep, value: SnapshotValue::Value(value) }
        }).collect();
        let id = SideExitId(self.trace.exits.len() as u32);
        self.trace.exits.push(SideExit { id, target: ExitTarget::Interpreter(TraceLocation::new(self.source.function, pc)),
            frames: vec![FrameSnapshot { disposition: FrameDisposition::Existing { depth_from_top: 0 }, writes }], linked_trace: None }); id
    }
    /// Install a watchdog exit at the logical loop header. The snapshot is taken
    /// after recording the iteration, so it contains exactly the values that a
    /// normal interpreter backedge would expose at the next header execution.
    pub fn add_native_budget_exit(&mut self, pc: BytecodePc) -> SideExitId {
        let writes = self.dirty.iter().map(|slot| {
            let value = self.slots[slot];
            let rep = self.trace.value(value).unwrap().rep;
            let kind = match rep {
                TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => RegisterKind::Int32,
                TraceRep::F64 => RegisterKind::F64,
                _ => RegisterKind::Value,
            };
            SnapshotWrite {
                slot: *slot,
                kind,
                target_rep: rep,
                value: SnapshotValue::Value(value),
            }
        }).collect();

        let id = SideExitId(self.trace.exits.len() as u32);
        self.trace.exits.push(SideExit {
            id,
            target: ExitTarget::Interpreter(TraceLocation::new(self.source.function, pc)),
            frames: vec![FrameSnapshot {
                disposition: FrameDisposition::Existing { depth_from_top: 0 },
                writes,
            }],
            linked_trace: None,
        });
        self.trace.native_budget_exit = Some(id);
        id
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

    /// Synchronize all currently dirty architectural registers without forgetting
    /// their SSA values. This is the boundary used by non-GC, non-frame-changing
    /// runtime leaf operations: an exceptional runtime return sees a coherent VM
    /// frame, while the successful native path can keep unaffected values live in
    /// native registers.
    pub fn commit_preserving_ssa(&mut self) {
        let dirty = self.dirty.clone();
        for slot in dirty {
            self.emit0(TraceInst::CommitFrameSlot { slot, value: self.slots[&slot] });
        }
    }

    /// Emit a synchronized runtime call that is guaranteed by the recorder to be
    /// non-GC and non-frame-changing. Unlike `runtime`, this deliberately keeps
    /// the slot cache and loop-carried SSA state alive across the call.
    pub fn runtime_leaf(&mut self, location: TraceLocation, next: BytecodePc) {
        self.commit_preserving_ssa();
        self.emit0(TraceInst::Runtime { location, next, terminal: false });
    }

    /// Reload a register written by `runtime_leaf`. Runtime-owned results are
    /// architectural writes, so keep them in the logical dirty set for loop-phi
    /// construction and side-exit reconstruction even though memory is already
    /// synchronized at this point.
    pub fn reload_slot_after_runtime(
        &mut self,
        slot: FrameSlot,
        ty: TraceType,
        rep: TraceRep,
    ) -> TraceValueId {
        let value = self.emit1(TraceInst::LoadFrameSlot { slot }, ty, rep);
        self.slots.insert(slot, value);
        if !self.dirty.contains(&slot) {
            self.dirty.push(slot);
        }
        value
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
    /// a native-loop invariant in the JIT backend.
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
        available[index] = matches!(value.origin,
            TraceValueOrigin::EntrySlot { .. } | TraceValueOrigin::Constant(_));
        if let TraceValueOrigin::Alias(_) = value.origin {
            if t.value(t.canonical_value(value.id)).is_none_or(|v| matches!(v.origin, TraceValueOrigin::Alias(_))) {
                errors.push("cyclic or invalid value alias".into());
            }
        }
    }
    for (index, node) in t.nodes.iter().enumerate() {
        if node.id != index { errors.push("noncanonical node ID".into()); }
    }
    let is_available = |v: TraceValueId, available: &[bool]| {
        available.get(t.canonical_value(v).0 as usize).copied().unwrap_or(false)
    };
    let check_exit = |id: SideExitId, available: &[bool], errors: &mut Vec<String>| {
        if let Some(exit) = t.exits.get(id.0 as usize) {
            for frame in &exit.frames { for write in &frame.writes {
                if let SnapshotValue::Value(v) = write.value {
                    if !is_available(v, available) { errors.push(format!("exit {id:?} uses unavailable value {v:?}")); }
                }
            }}
        } else { errors.push(format!("unknown exit {id:?}")); }
    };
    // Entry guards and snapshots cannot depend on loop-body values or phis.
    for (index, value) in t.values.iter().enumerate() {
        if let TraceValueOrigin::GuardedEntry { value: source, exit } = value.origin {
            if is_available(source, &available)
                && t.value(t.canonical_value(source)).is_some_and(|v| v.rep == TraceRep::Tagged)
                && matches!(value.rep, TraceRep::F64 | TraceRep::Int32 | TraceRep::UInt32)
            {
                check_exit(exit, &available, &mut errors);
                available[index] = true;
            } else { errors.push("invalid guarded entry value".into()); }
        }
    }
    let check_node = |index: usize, available: &mut [bool], errors: &mut Vec<String>| {
        let node = &t.nodes[index];
        node.kind.clone().visit_operands(|v| {
            if !is_available(*v, available) { errors.push(format!("node {index} uses unavailable value {v:?}")); }
        });
        if let Some(exit) = node.kind.exit() { check_exit(exit, available, errors); }
        for result in &node.results {
            if let Some(v) = t.value(*result) {
                match &v.origin {
                    TraceValueOrigin::InstResult { node: producer } if *producer == index => {
                        // A dead producer must not make an uncomputed value usable.
                        available[result.0 as usize] = !matches!(node.kind, TraceInst::Nop);
                    }
                    TraceValueOrigin::Alias(target) => {
                        if !matches!(node.kind, TraceInst::Nop) {
                            errors.push("aliased result still has a live producer".into());
                        }
                        if !is_available(*target, available) {
                            errors.push("aliased result targets unavailable value".into());
                        }
                    }
                    _ => errors.push("result has wrong producer".into()),
                }
            } else { errors.push("unknown result".into()); }
        }
    };
    let mut preheader = vec![false; t.nodes.len()];
    for &id in &t.preheader {
        let Some(node) = t.nodes.get(id) else {
            errors.push("unknown preheader node".into());
            continue;
        };
        if std::mem::replace(&mut preheader[id], true) { errors.push("duplicate preheader node".into()); }
        if !node.kind.is_speculatable() && !matches!(node.kind, TraceInst::Nop) {
            errors.push("preheader operation may have effects or trap".into());
        }
        check_node(id, &mut available, &mut errors);
    }
    // Check all initial values before publishing any loop parameters. This
    // rejects body-only initializers even when the backedge could supply them.
    let mut phi_values = HashSet::new();
    for phi in &t.loop_phis {
        let initial_ok = is_available(phi.initial, &available)
            && t.value(t.canonical_value(phi.initial)).is_some_and(|v| {
                v.rep == phi.rep || (phi.entry_exit.is_some() && v.rep == TraceRep::Tagged
                    && matches!(phi.rep, TraceRep::F64 | TraceRep::Int32 | TraceRep::UInt32))
            });
        let value_ok = t.value(phi.value).is_some_and(|v| v.rep == phi.rep
            && matches!(v.origin, TraceValueOrigin::LoopPhi { .. } | TraceValueOrigin::LoopTemporary));
        if !initial_ok || !value_ok || !phi_values.insert(phi.value) {
            errors.push("invalid loop phi entry".into());
        }
        if let Some(exit) = phi.entry_exit { check_exit(exit, &available, &mut errors); }
    }
    for phi in &t.loop_phis {
        if let Some(slot) = available.get_mut(phi.value.0 as usize) { *slot = true; }
    }
    for (index, _) in t.nodes.iter().enumerate() {
        if !preheader[index] { check_node(index, &mut available, &mut errors); }
    }
    for phi in &t.loop_phis {
        if !phi.backedge.is_some_and(|v| is_available(v, &available)
            && t.value(t.canonical_value(v)).is_some_and(|v| v.rep == phi.rep)) {
            errors.push("invalid loop phi backedge".into());
        }
    }
    if let Some(id) = t.native_budget_exit {
        check_exit(id, &available, &mut errors);
        if !matches!(t.terminator, TraceTerminator::LoopBack) {
            errors.push("native budget exit is only valid for loop-back traces".into());
        }
    }
    match t.terminator {
        TraceTerminator::Exit(id) => check_exit(id, &available, &mut errors),
        TraceTerminator::Unreachable => errors.push("missing terminator".into()),
        _ => {},
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

// Compatibility for callers using the original optimizer path.
pub use crate::wren_trace_ir_optimizer::TraceOptimizer;

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

    #[test]
    fn promotes_tagged_unsigned_loop_phi_to_u32() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(25));
        let mut builder = TraceBuilder::new(
            TraceId(2),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(2));
        let initial = builder.read_slot(slot, TraceType::Num, TraceRep::Tagged);
        let phi = builder.add_loop_phi(slot, initial, TraceType::Num, TraceRep::Tagged);
        let guard_exit = builder.add_side_exit(ExitReason::GuardUnsignedInteger, anchor.pc);
        let raw = builder.emit1(
            TraceInst::GuardUInt32 { value: phi, exit: guard_exit },
            TraceType::UInt32,
            TraceRep::UInt32,
        );
        let one = builder.constant(TraceConstant::Int32(1));
        let next = builder.emit1(
            TraceInst::IntBinary { op: IntBinOp::BitXor, lhs: raw, rhs: one },
            TraceType::UInt32,
            TraceRep::UInt32,
        );
        let boxed = builder.emit1(
            TraceInst::BoxUInt32 { value: next },
            TraceType::Num,
            TraceRep::Tagged,
        );
        builder.write_slot(slot, boxed);
        builder.set_loop_backedge(phi, boxed).unwrap();
        builder.set_terminator(TraceTerminator::LoopBack);

        let mut trace = builder.finish().unwrap();
        TraceOptimizer::default().optimize(&mut trace).unwrap();

        assert_eq!(trace.loop_phis[0].rep, TraceRep::UInt32);
        assert!(trace.loop_phis[0].entry_exit.is_some());
        assert_eq!(trace.canonical_value(trace.loop_phis[0].backedge.unwrap()), next);
        assert!(trace.stats.promoted_u32_loop_phis >= 1);
    }

    #[test]
    fn native_budget_exit_snapshots_end_of_iteration_state() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(30));
        let mut builder = TraceBuilder::new(
            TraceId(2),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(2));
        let initial = builder.read_slot(slot, TraceType::Num, TraceRep::Tagged);
        let one = builder.constant(TraceConstant::NumBits(1.0f64.to_bits()));
        builder.write_slot(slot, one);
        let phi = builder.add_loop_phi(slot, initial, TraceType::Num, TraceRep::Tagged);
        builder.set_loop_backedge(phi, one).unwrap();
        let exit = builder.add_native_budget_exit(anchor.pc);
        builder.set_terminator(TraceTerminator::LoopBack);

        let trace = builder.finish().unwrap();
        assert_eq!(trace.native_budget_exit, Some(exit));
        let write = trace.exits[exit.0 as usize].frames[0]
            .writes
            .iter()
            .find(|write| write.slot == slot)
            .expect("watchdog exit must reconstruct the dirty loop slot");
        assert_eq!(write.value, SnapshotValue::Value(one));
        assert_eq!(write.target_rep, TraceRep::Tagged);
    }


    #[test]
    fn numeric_phi_promotion_keeps_watchdog_snapshot_deoptimizable() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(40));
        let mut builder = TraceBuilder::new(
            TraceId(3),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(3));
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
        builder.set_loop_backedge(phi, boxed).unwrap();
        let budget_exit = builder.add_native_budget_exit(anchor.pc);
        builder.set_terminator(TraceTerminator::LoopBack);

        let mut trace = builder.finish().unwrap();
        TraceOptimizer::default().optimize(&mut trace).unwrap();

        let write = trace.exits[budget_exit.0 as usize].frames[0]
            .writes
            .iter()
            .find(|write| write.slot == slot)
            .expect("watchdog snapshot must keep the promoted loop value");
        assert_eq!(write.value, SnapshotValue::Value(next));
        assert_eq!(write.target_rep, TraceRep::Tagged);
    }

    #[test]
    fn conservative_optimizer_keeps_tagged_loop_phi() {
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(50));
        let mut builder = TraceBuilder::new(
            TraceId(4),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(4));
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
        builder.set_loop_backedge(phi, boxed).unwrap();
        builder.set_terminator(TraceTerminator::LoopBack);

        let mut trace = builder.finish().unwrap();
        TraceOptimizer::default().optimize_conservative(&mut trace).unwrap();

        assert_eq!(trace.loop_phis[0].rep, TraceRep::Tagged);
        assert!(trace.loop_phis[0].entry_exit.is_none());
        assert_eq!(trace.value(phi).unwrap().rep, TraceRep::Tagged);
        assert_eq!(trace.stats.promoted_f64_loop_phis, 0);
    }

    #[test]
    fn runtime_leaf_commits_without_destroying_ssa_state() {
        let location = TraceLocation::new(FuncId(0), BytecodePc(7));
        let mut builder = TraceBuilder::new(
            TraceId(99),
            TraceAnchor { kind: TraceKind::Loop, location },
        );
        let input = FrameSlot::General(Reg(0));
        let output = FrameSlot::General(Reg(1));

        let value = builder.constant(TraceConstant::Int32(41));
        builder.write_slot(input, value);
        builder.runtime_leaf(location, BytecodePc(8));

        assert_eq!(builder.slot_value(input), Some(value));
        assert!(builder.dirty.contains(&input));
        assert!(!builder.has_runtime_calls());
        assert!(builder.trace.nodes.iter().any(|node| matches!(
            node.kind,
            TraceInst::CommitFrameSlot { slot, value: committed }
                if slot == input && committed == value
        )));
        assert!(builder.trace.nodes.iter().any(|node| matches!(
            node.kind,
            TraceInst::Runtime { location: at, next, terminal: false }
                if at == location && next == BytecodePc(8)
        )));

        let loaded = builder.reload_slot_after_runtime(
            output,
            TraceType::Any,
            TraceRep::Tagged,
        );
        assert_eq!(builder.slot_value(output), Some(loaded));
        assert!(builder.dirty.contains(&output));
        assert!(matches!(
            builder.trace.value(loaded).map(|value| &value.origin),
            Some(TraceValueOrigin::InstResult { .. })
        ));
    }

}
