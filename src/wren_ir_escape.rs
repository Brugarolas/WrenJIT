//! Escape analysis and scalar replacement of aggregates (SROA) for WIR.
//!
//! This module implements a flow-sensitive *partial* escape analysis (in the
//! style of Graal's PEA) on top of WIR's block-parameter SSA form, plus the
//! transformation that consumes it:
//!
//! * Every allocation the IR can model field-by-field (`AllocInstance`,
//!   `AllocCell`, `AllocList`, `MakeRange`, `MakeClosure`, `AllocMap`) starts
//!   *virtual*: its fields are tracked as SSA variables instead of heap slots.
//! * Accesses through a virtual object (`LoadField`/`StoreField`,
//!   `LoadCell`/`StoreCell`, `ListLen`/`ListGet`/`ListSet`/`ListAppend`,
//!   identity tests and truthiness) are rewritten into plain SSA data flow.
//!   Loads become `Move`s of the current field value, stores disappear.
//! * Any other use is an escape. The object is *materialized* right before the
//!   escaping instruction (allocation + initializing stores of the current
//!   field values), but only on the paths that actually escape. Control-flow
//!   merges that see a virtual object on one side and a materialized one on
//!   the other materialize at the end of the virtual predecessors.
//! * Field values flowing around loops/merges are reconnected with new block
//!   parameters (on-the-fly SSA construction, Braun et al.), so scalarized
//!   fields live only in SSA values/registers.
//!
//! The result:
//! * `NoEscape` allocations are deleted completely.
//! * `PartialEscape` allocations are sunk into the escaping paths only.
//! * `GlobalEscape` allocations still get their reads before the escape
//!   scalar-replaced when profitable, which is general SROA: store-to-load
//!   forwarding through the object, including across blocks and loops.
//!
//! Allocations referenced by deoptimization snapshots are never virtualized
//! because WIR has no rematerialization metadata for deopt yet.
//! Protected throwing points pin only allocations live across those points;
//! allocations outside that lifetime retain ordinary PEA/SROA, including in
//! handlers. Pinned objects keep their original allocation and stores, so a
//! handler observes the exact pre-throw state without virtual-object metadata.

#![allow(dead_code)]

use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

use crate::wren_ir::{
    BlockId, ClassId, Constant, Edge, FieldSlot, FuncId, Function, Inst, InstId, SourceSpan,
    Type, Value, ValueDef, ValueId,
};
use crate::wren_ir_optimizer::{repair_effect_chain, FunctionOptimization, InstFactory};
use crate::wren_ir_optimizing::CopyPropagation;

/// Upper bound on distinct materialization sites before PEA gives up on an
/// allocation that still escapes (avoids code growth from duplicated allocs).
const MAX_MATERIALIZATIONS: usize = 8;
/// Maximum number of scalarized fields/elements per allocation.
const MAX_FIELDS: usize = 64;
/// Rounds of EA/SROA + copy propagation (exposes nested virtual objects).
const MAX_ROUNDS: usize = 4;

const NONE: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Public analysis API
// ---------------------------------------------------------------------------

/// Escape classification of a single allocation site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EscapeState {
    /// The object never escapes; the allocation can be removed entirely.
    NoEscape,
    /// The object escapes on some paths only; the allocation can be sunk into
    /// those paths and is never performed on the others.
    PartialEscape,
    /// The object escapes on every path (or cannot be analysed).
    GlobalEscape,
}

#[derive(Clone, Debug)]
pub struct EscapeInfo {
    pub allocation: InstId,
    pub value: ValueId,
    pub state: EscapeState,
    /// Number of program points where the object must be materialized.
    pub materializations: usize,
    /// Field reads / identity tests that can be answered without the heap.
    pub scalarized_reads: usize,
    /// Whether `ScalarReplacement` will transform this allocation.
    pub scalar_replaceable: bool,
}

/// Flow-sensitive (partial) escape analysis over a WIR function.
pub struct EscapeAnalysis;

impl EscapeAnalysis {
    pub fn analyze(function: &Function) -> Vec<EscapeInfo> {
        let analysis = Analysis::run(function);
        analysis
            .candidates
            .iter()
            .zip(&analysis.plans)
            .map(|(candidate, plan)| EscapeInfo {
                allocation: candidate.inst,
                value: candidate.value,
                state: plan.state(),
                materializations: plan.materializations,
                scalarized_reads: plan.virtual_reads,
                scalar_replaceable: plan.selected,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalarReplacementStats {
    /// Allocations deleted entirely (NoEscape).
    pub removed_allocations: usize,
    /// Allocations sunk into escaping paths (PartialEscape).
    pub sunk_allocations: usize,
    /// Allocations kept on every path but with scalar-replaced accesses.
    pub forwarded_allocations: usize,
    /// Materialization sites emitted.
    pub materializations: usize,
    /// Loads/identity tests replaced with SSA values.
    pub scalarized_reads: usize,
}

/// Escape analysis driven scalar replacement / allocation sinking.
#[derive(Default)]
pub struct ScalarReplacement {
    pub stats: ScalarReplacementStats,
}

impl FunctionOptimization for ScalarReplacement {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed_any = false;
        let mut copies = CopyPropagation::default();
        for _ in 0..MAX_ROUNDS {
            if !run_round(function, &mut self.stats) {
                break;
            }
            changed_any = true;
            // Removes the load-replacement moves so nested objects stored in a
            // scalarized field become directly visible to the next round.
            copies.optimize_function(function);
        }
        changed_any
    }
}

// ---------------------------------------------------------------------------
// CFG + dominators
// ---------------------------------------------------------------------------

struct Cfg {
    reachable: Vec<bool>,
    rpo: Vec<usize>,
    preds: Vec<Vec<usize>>,
    succs: Vec<Vec<usize>>,
    dom_pre: Vec<u32>,
    dom_post: Vec<u32>,
}

impl Cfg {
    fn build(function: &Function) -> Self {
        let n = function.blocks.len();
        let mut succs = vec![Vec::new(); n];
        for (ix, block) in function.blocks.iter().enumerate() {
            let Some(term) = block.insts.last().and_then(|id| function.node(*id)) else {
                continue;
            };
            for edge in term.kind.edges() {
                let target = edge.target.0 as usize;
                if target < n && !succs[ix].contains(&target) {
                    succs[ix].push(target);
                }
            }
        }

        // Keep handlers and their successors reachable. These edges are used for
        // dominance/liveness only: virtual state must never flow through them.
        for region in &function.exception_regions {
            for block in &region.protected_blocks {
                let succ = &mut succs[block.0 as usize];
                if !succ.contains(&(region.handler.0 as usize)) {
                    succ.push(region.handler.0 as usize);
                }
            }
        }

        let entry = function.entry.0 as usize;
        let mut reachable = vec![false; n];
        let mut post = Vec::with_capacity(n);
        if entry < n {
            reachable[entry] = true;
            let mut stack = vec![(entry, 0usize)];
            while let Some(top) = stack.last_mut() {
                let (block, next) = *top;
                if next < succs[block].len() {
                    top.1 += 1;
                    let succ = succs[block][next];
                    if !reachable[succ] {
                        reachable[succ] = true;
                        stack.push((succ, 0));
                    }
                } else {
                    post.push(block);
                    stack.pop();
                }
            }
        }
        let rpo: Vec<usize> = post.into_iter().rev().collect();
        let mut rpo_index = vec![NONE; n];
        for (index, &block) in rpo.iter().enumerate() {
            rpo_index[block] = index;
        }

        let mut preds = vec![Vec::new(); n];
        for &block in &rpo {
            for &succ in &succs[block] {
                preds[succ].push(block);
            }
        }

        // Cooper-Harvey-Kennedy iterative dominators.
        let mut idom = vec![NONE; n];
        if !rpo.is_empty() {
            idom[entry] = entry;
            let mut changed = true;
            while changed {
                changed = false;
                for &block in rpo.iter().skip(1) {
                    let mut new_idom = NONE;
                    for &pred in &preds[block] {
                        if idom[pred] == NONE {
                            continue;
                        }
                        new_idom = if new_idom == NONE {
                            pred
                        } else {
                            intersect(&idom, &rpo_index, pred, new_idom)
                        };
                    }
                    if new_idom != NONE && idom[block] != new_idom {
                        idom[block] = new_idom;
                        changed = true;
                    }
                }
            }
        }

        let mut children = vec![Vec::new(); n];
        for &block in &rpo {
            if block != entry && idom[block] != NONE {
                children[idom[block]].push(block);
            }
        }
        let mut dom_pre = vec![0u32; n];
        let mut dom_post = vec![0u32; n];
        if !rpo.is_empty() {
            let mut counter = 0u32;
            dom_pre[entry] = counter;
            counter += 1;
            let mut stack = vec![(entry, 0usize)];
            while let Some(top) = stack.last_mut() {
                let (block, next) = *top;
                if next < children[block].len() {
                    top.1 += 1;
                    let child = children[block][next];
                    dom_pre[child] = counter;
                    counter += 1;
                    stack.push((child, 0));
                } else {
                    dom_post[block] = counter;
                    counter += 1;
                    stack.pop();
                }
            }
        }

        Self {
            reachable,
            rpo,
            preds,
            succs,
            dom_pre,
            dom_post,
        }
    }

    #[inline]
    fn dominates(&self, a: usize, b: usize) -> bool {
        self.reachable[a]
            && self.reachable[b]
            && self.dom_pre[a] <= self.dom_pre[b]
            && self.dom_post[b] <= self.dom_post[a]
    }

    #[inline]
    fn strictly_dominates(&self, a: usize, b: usize) -> bool {
        a != b && self.dominates(a, b)
    }
}

fn intersect(idom: &[usize], rpo_index: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_index[a] > rpo_index[b] {
            a = idom[a];
        }
        while rpo_index[b] > rpo_index[a] {
            b = idom[b];
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Candidates and use classification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Shape {
    Instance { class: ClassId },
    Cell,
    List { capacity: Option<ValueId> },
    Range { inclusive: bool },
    Closure { function: FuncId },
    Map,
}

#[derive(Clone, Debug)]
struct Candidate {
    value: ValueId,
    inst: InstId,
    block: usize,
    pos: usize,
    shape: Shape,
    ty: Type,
    source: Option<SourceSpan>,
    /// Instance field slots accessed through this object (sorted).
    slots: Vec<FieldSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UseOp {
    LoadField(FieldSlot),
    StoreField(FieldSlot),
    LoadCell,
    StoreCell,
    ListLen,
    ListGet(i32),
    ListSet(i32),
    ListAppend,
    WriteBarrier,
    RefEq,
    BranchCond,
    Escape,
}

impl UseOp {
    fn is_read(self) -> bool {
        matches!(
            self,
            UseOp::LoadField(_)
                | UseOp::LoadCell
                | UseOp::ListLen
                | UseOp::ListGet(_)
                | UseOp::RefEq
                | UseOp::BranchCond
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Use {
    block: usize,
    pos: usize,
    op: UseOp,
}

fn producer(function: &Function, value: ValueId) -> Option<&Inst> {
    let ValueDef::InstResult(inst, _) = function.value(value)?.def else {
        return None;
    };
    function.node(inst).map(|node| &node.kind)
}

fn resolve_constant(function: &Function, mut value: ValueId) -> Option<&Constant> {
    for _ in 0..8 {
        match producer(function, value)? {
            Inst::Const(constant) => return Some(constant),
            Inst::Move(source) => value = *source,
            _ => return None,
        }
    }
    None
}

fn resolve_i32(function: &Function, value: ValueId) -> Option<i32> {
    match resolve_constant(function, value)? {
        Constant::Int32(value) => Some(*value),
        _ => None,
    }
}

fn value_type(function: &Function, value: ValueId) -> Option<&Type> {
    function.value(value).map(|value| &value.ty)
}

fn is_wren_value(function: &Function, value: ValueId) -> bool {
    value_type(function, value).is_some_and(|ty| !ty.is_internal_representation())
}

fn single_result_type(function: &Function, inst: InstId) -> Option<&Type> {
    let node = function.node(inst)?;
    let result = *node.results.first()?;
    value_type(function, result)
}

fn candidate_shape(function: &Function, kind: &Inst) -> Option<Shape> {
    match kind {
        Inst::AllocInstance { class } => Some(Shape::Instance { class: *class }),
        Inst::AllocCell { initial } => is_wren_value(function, *initial).then_some(Shape::Cell),
        Inst::AllocList { capacity } => match capacity {
            None => Some(Shape::List { capacity: None }),
            // A negative capacity is a runtime error that must be preserved.
            Some(capacity) => resolve_i32(function, *capacity)
                .filter(|value| *value >= 0)
                .map(|_| Shape::List {
                    capacity: Some(*capacity),
                }),
        },
        // MakeRange fails for non-number endpoints; only drop it when proven safe.
        Inst::MakeRange {
            from,
            to,
            inclusive,
        } => (value_type(function, *from) == Some(&Type::Num)
            && value_type(function, *to) == Some(&Type::Num))
        .then_some(Shape::Range {
            inclusive: *inclusive,
        }),
        Inst::MakeClosure { function: callee, .. } => Some(Shape::Closure { function: *callee }),
        Inst::AllocMap => Some(Shape::Map),
        _ => None,
    }
}

fn classify_use(
    function: &Function,
    inst: InstId,
    kind: &Inst,
    value: ValueId,
    shape: &Shape,
) -> UseOp {
    let readable_result = || {
        single_result_type(function, inst).is_some_and(|ty| !ty.is_internal_representation())
    };
    match (kind, shape) {
        (Inst::LoadField { object, slot }, Shape::Instance { .. })
            if *object == value && readable_result() =>
        {
            UseOp::LoadField(*slot)
        }
        (
            Inst::StoreField {
                object,
                slot,
                value: stored,
            },
            Shape::Instance { .. },
        ) if *object == value && *stored != value && is_wren_value(function, *stored) => {
            UseOp::StoreField(*slot)
        }
        (Inst::LoadCell { cell }, Shape::Cell) if *cell == value && readable_result() => {
            UseOp::LoadCell
        }
        (Inst::StoreCell { cell, value: stored }, Shape::Cell)
            if *cell == value && *stored != value && is_wren_value(function, *stored) =>
        {
            UseOp::StoreCell
        }
        (Inst::ListLen { list }, Shape::List { .. })
            if *list == value && single_result_type(function, inst) == Some(&Type::Int32) =>
        {
            UseOp::ListLen
        }
        (Inst::ListGet { list, index }, Shape::List { .. })
            if *list == value && *index != value && readable_result() =>
        {
            resolve_i32(function, *index).map_or(UseOp::Escape, UseOp::ListGet)
        }
        (
            Inst::ListSet {
                list,
                index,
                value: stored,
            },
            Shape::List { .. },
        ) if *list == value
            && *index != value
            && *stored != value
            && is_wren_value(function, *stored) =>
        {
            resolve_i32(function, *index).map_or(UseOp::Escape, UseOp::ListSet)
        }
        (Inst::ListAppend { list, value: stored }, Shape::List { .. })
            if *list == value && *stored != value && is_wren_value(function, *stored) =>
        {
            UseOp::ListAppend
        }
        (Inst::WriteBarrier { object, value: stored }, _)
            if *object == value && *stored != value =>
        {
            UseOp::WriteBarrier
        }
        (Inst::RefEq { .. }, _) => UseOp::RefEq,
        (
            Inst::Branch {
                cond,
                then_edge,
                else_edge,
            },
            _,
        ) if *cond == value
            && !then_edge.args.contains(&value)
            && !else_edge.args.contains(&value) =>
        {
            UseOp::BranchCond
        }
        _ => UseOp::Escape,
    }
}

/// Same normalization as the interpreter's list primitives.
fn normalize_index(index: i32, len: u32) -> Option<u32> {
    let index = if index < 0 {
        len as i64 + index as i64
    } else {
        index as i64
    };
    (index >= 0 && index < len as i64).then_some(index as u32)
}

/// Transfer function for one use while the object is virtual with `len`
/// elements. `None` means the use escapes.
fn virtual_step(len: u32, op: UseOp) -> Option<u32> {
    match op {
        UseOp::LoadField(_)
        | UseOp::StoreField(_)
        | UseOp::LoadCell
        | UseOp::StoreCell
        | UseOp::ListLen
        | UseOp::WriteBarrier
        | UseOp::RefEq
        | UseOp::BranchCond => Some(len),
        UseOp::ListGet(index) | UseOp::ListSet(index) => normalize_index(index, len).map(|_| len),
        UseOp::ListAppend => ((len as usize) < MAX_FIELDS).then_some(len + 1),
        UseOp::Escape => None,
    }
}

// ---------------------------------------------------------------------------
// Partial escape dataflow
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VState {
    Unknown,
    Virtual(u32),
    Materialized,
}

fn meet(left: VState, right: VState) -> VState {
    match (left, right) {
        (VState::Unknown, other) | (other, VState::Unknown) => other,
        (VState::Virtual(a), VState::Virtual(b)) if a == b => VState::Virtual(a),
        _ => VState::Materialized,
    }
}

#[derive(Clone, Debug, Default)]
struct Plan {
    valid: bool,
    selected: bool,
    in_state: Vec<VState>,
    forced_end: Vec<bool>,
    materializations: usize,
    virtual_reads: usize,
    dies_virtual: bool,
}

impl Plan {
    fn state(&self) -> EscapeState {
        if !self.valid {
            EscapeState::GlobalEscape
        } else if self.materializations == 0 {
            EscapeState::NoEscape
        } else if self.dies_virtual {
            EscapeState::PartialEscape
        } else {
            EscapeState::GlobalEscape
        }
    }
}

struct Analysis {
    cfg: Cfg,
    candidates: Vec<Candidate>,
    /// Uses per candidate, grouped per block, ordered by position.
    uses: Vec<HashMap<usize, Vec<Use>>>,
    plans: Vec<Plan>,
}

impl Analysis {
    fn run(function: &Function) -> Self {
        let cfg = Cfg::build(function);
        let mut candidates = Vec::new();
        let mut index_of: HashMap<ValueId, usize> = HashMap::new();

        for &block in &cfg.rpo {
            for (pos, &inst) in function.blocks[block].insts.iter().enumerate() {
                let Some(node) = function.node(inst) else {
                    continue;
                };
                let Some(shape) = candidate_shape(function, &node.kind) else {
                    continue;
                };
                let Some(&value) = node.results.first() else {
                    continue;
                };
                let Some(ty) = value_type(function, value).cloned() else {
                    continue;
                };
                index_of.insert(value, candidates.len());
                candidates.push(Candidate {
                    value,
                    inst,
                    block,
                    pos,
                    shape,
                    ty,
                    source: node.source,
                    slots: Vec::new(),
                });
            }
        }

        let mut valid = vec![true; candidates.len()];

        // Deopt snapshots would need rematerialization metadata.
        for deopt in &function.deopts {
            let frames = std::iter::once(&deopt.snapshot.slots)
                .chain(deopt.snapshot.inlined_frames.iter().map(|frame| &frame.slots));
            for slots in frames {
                for slot in slots {
                    if let Some(&index) = index_of.get(&slot.value) {
                        valid[index] = false;
                    }
                }
            }
        }

        let mut uses: Vec<HashMap<usize, Vec<Use>>> = vec![HashMap::new(); candidates.len()];
        let mut seen: SmallVec<[ValueId; 8]> = SmallVec::new();
        for &block in &cfg.rpo {
            for (pos, &inst) in function.blocks[block].insts.iter().enumerate() {
                let Some(node) = function.node(inst) else {
                    continue;
                };
                seen.clear();
                for operand in node.kind.operands() {
                    if seen.contains(&operand) {
                        continue;
                    }
                    seen.push(operand);
                    let Some(&index) = index_of.get(&operand) else {
                        continue;
                    };
                    let candidate = &mut candidates[index];
                    let dominated = (block == candidate.block && pos > candidate.pos)
                        || cfg.strictly_dominates(candidate.block, block);
                    if !dominated {
                        valid[index] = false;
                        continue;
                    }
                    let op = classify_use(function, inst, &node.kind, operand, &candidate.shape);
                    if let UseOp::LoadField(slot) | UseOp::StoreField(slot) = op {
                        if let Err(at) = candidate.slots.binary_search(&slot) {
                            candidate.slots.insert(at, slot);
                        }
                    }
                    uses[index]
                        .entry(block)
                        .or_default()
                        .push(Use { block, pos, op });
                }
            }
        }

        let plans = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                if !valid[index] || candidate.slots.len() > MAX_FIELDS
                    || crosses_exception(function, &cfg, candidate, &uses[index])
                {
                    Plan::default()
                } else {
                    analyze_candidate(&cfg, candidate, &uses[index])
                }
            })
            .collect();

        Self {
            cfg,
            candidates,
            uses,
            plans,
        }
    }
}

/// Until exceptional snapshots can encode virtual objects, keep *this*
/// allocation materialized if its lifetime crosses a protected throwing point.
/// Backward liveness includes handlers and loop backedges and kills at the
/// allocation, so an unrelated try, dead object, or post-throw allocation does
/// not disable scalar replacement. Escaping aliases are already materialized by
/// the ordinary PEA use classification.
fn crosses_exception(
    function: &Function,
    cfg: &Cfg,
    candidate: &Candidate,
    uses: &HashMap<usize, Vec<Use>>,
) -> bool {
    if function.exception_regions.is_empty() {
        return false;
    }
    let mut handlers = vec![Vec::new(); function.blocks.len()];
    for region in &function.exception_regions {
        for block in &region.protected_blocks {
            handlers[block.0 as usize].push(region.handler.0 as usize);
        }
    }
    let mut live_in = vec![false; function.blocks.len()];
    loop {
        let mut changed = false;
        for &block in cfg.rpo.iter().rev() {
            let live = block != candidate.block
                && (uses.get(&block).is_some_and(|u| !u.is_empty())
                    || cfg.succs[block].iter().any(|&s| live_in[s]));
            changed |= live && !live_in[block];
            live_in[block] |= live;
        }
        if !changed { break; }
    }
    for &block in &cfg.rpo {
        if handlers[block].is_empty()
            || !(block == candidate.block || cfg.strictly_dominates(candidate.block, block))
        {
            continue;
        }
        let mut live = cfg.succs[block].iter().any(|&s| live_in[s]);
        for (pos, &inst) in function.blocks[block].insts.iter().enumerate().rev() {
            if block == candidate.block && pos <= candidate.pos { break; }
            let own_use = uses.get(&block).and_then(|u| u.iter().find(|u| u.pos == pos));
            live |= own_use.is_some();
            // Accesses proved valid on the virtual candidate cannot throw. A
            // failing/dynamic collection access is an Escape and is retained.
            let virtual_access = own_use.is_some_and(|u| matches!(u.op,
                UseOp::LoadField(_) | UseOp::StoreField(_) | UseOp::LoadCell
                | UseOp::StoreCell | UseOp::ListLen | UseOp::WriteBarrier
                | UseOp::RefEq | UseOp::BranchCond));
            if live && !virtual_access && function.nodes[inst.0 as usize].kind.may_throw() {
                return true;
            }
        }
    }
    false
}

struct BlockSummary {
    out: VState,
    materializations: usize,
    virtual_reads: usize,
    ok: bool,
}

fn simulate_block(
    candidate: &Candidate,
    block: usize,
    in_state: VState,
    uses: Option<&Vec<Use>>,
) -> BlockSummary {
    let mut state = if block == candidate.block {
        VState::Virtual(0)
    } else {
        in_state
    };
    let mut summary = BlockSummary {
        out: state,
        materializations: 0,
        virtual_reads: 0,
        ok: true,
    };
    for use_ in uses.into_iter().flatten() {
        match state {
            VState::Virtual(len) => match virtual_step(len, use_.op) {
                Some(next) => {
                    if use_.op.is_read() {
                        summary.virtual_reads += 1;
                    }
                    state = VState::Virtual(next);
                }
                None => {
                    summary.materializations += 1;
                    state = VState::Materialized;
                }
            },
            VState::Materialized => {}
            VState::Unknown => summary.ok = false,
        }
    }
    summary.out = state;
    summary
}

fn analyze_candidate(cfg: &Cfg, candidate: &Candidate, uses: &HashMap<usize, Vec<Use>>) -> Plan {
    let n = cfg.reachable.len();
    let alloc_block = candidate.block;
    let in_region = |block: usize| cfg.strictly_dominates(alloc_block, block);

    let mut in_state = vec![VState::Unknown; n];
    let mut out_state = vec![VState::Unknown; n];
    let mut forced_end = vec![false; n];
    let mut queued = vec![false; n];
    let mut worklist = std::collections::VecDeque::new();
    worklist.push_back(alloc_block);
    queued[alloc_block] = true;

    let push_succs = |block: usize,
                      worklist: &mut std::collections::VecDeque<usize>,
                      queued: &mut Vec<bool>| {
        for &succ in &cfg.succs[block] {
            if in_region(succ) && !queued[succ] {
                queued[succ] = true;
                worklist.push_back(succ);
            }
        }
    };

    while let Some(block) = worklist.pop_front() {
        queued[block] = false;
        if block != alloc_block {
            let mut state = VState::Unknown;
            for &pred in &cfg.preds[block] {
                state = meet(state, out_state[pred]);
            }
            if state == VState::Materialized {
                // A merge of virtual and materialized objects: materialize at
                // the end of every predecessor that is still virtual.
                for &pred in &cfg.preds[block] {
                    if matches!(out_state[pred], VState::Virtual(_)) {
                        forced_end[pred] = true;
                        out_state[pred] = VState::Materialized;
                        push_succs(pred, &mut worklist, &mut queued);
                    }
                }
            }
            in_state[block] = state;
            if state == VState::Unknown {
                continue;
            }
        }

        let summary = simulate_block(candidate, block, in_state[block], uses.get(&block));
        if !summary.ok {
            return Plan::default();
        }
        let mut out = summary.out;
        if forced_end[block] && matches!(out, VState::Virtual(_)) {
            out = VState::Materialized;
        }
        if out != out_state[block] {
            out_state[block] = out;
            push_succs(block, &mut worklist, &mut queued);
        }
    }

    let mut plan = Plan {
        valid: true,
        ..Plan::default()
    };
    for &block in &cfg.rpo {
        if block != alloc_block && !in_region(block) {
            continue;
        }
        if block != alloc_block && in_state[block] == VState::Unknown {
            return Plan::default();
        }
        let summary = simulate_block(candidate, block, in_state[block], uses.get(&block));
        if !summary.ok {
            return Plan::default();
        }
        plan.materializations += summary.materializations;
        plan.virtual_reads += summary.virtual_reads;
        if matches!(summary.out, VState::Virtual(_)) {
            if forced_end[block] {
                plan.materializations += 1;
            } else if cfg.succs[block].is_empty()
                || cfg.succs[block].iter().any(|succ| !in_region(*succ))
            {
                plan.dies_virtual = true;
            }
        }
    }

    plan.in_state = in_state;
    plan.forced_end = forced_end;
    plan.selected = plan.materializations == 0
        || ((plan.virtual_reads > 0 || plan.dies_virtual)
            && plan.materializations <= MAX_MATERIALIZATIONS);
    plan
}

// ---------------------------------------------------------------------------
// Transformation
// ---------------------------------------------------------------------------

/// SSA variable identifiers: `base + 0` is the object reference after
/// materialization, `base + 1 + k` is scalarized field/element `k`.
struct VarLayout {
    base: Vec<usize>,
    total: usize,
}

impl VarLayout {
    fn new(candidates: &[Candidate], selected: &[usize]) -> Self {
        let mut base = vec![NONE; candidates.len()];
        let mut total = 0;
        for &index in selected {
            base[index] = total;
            let fields = match candidates[index].shape {
                Shape::Instance { .. } => candidates[index].slots.len(),
                Shape::Cell => 1,
                Shape::List { .. } => MAX_FIELDS,
                Shape::Range { .. } | Shape::Closure { .. } | Shape::Map => 0,
            };
            total += 1 + fields;
        }
        Self { base, total }
    }

    fn reference(&self, candidate: usize) -> usize {
        self.base[candidate]
    }

    fn field(&self, candidate: usize, field: usize) -> usize {
        self.base[candidate] + 1 + field
    }
}

/// On-the-fly SSA construction for scalarized fields (Braun et al.).
struct SsaBuilder {
    defs: Vec<HashMap<usize, ValueId>>,
    entry: Vec<HashMap<usize, ValueId>>,
    processed: Vec<bool>,
    /// (block, var, param) phis whose incoming values still need filling.
    pending: Vec<(usize, usize, ValueId)>,
    /// New block params in creation order.
    new_params: Vec<Vec<(usize, ValueId)>>,
    var_types: Vec<Option<Type>>,
}

impl SsaBuilder {
    fn new(blocks: usize, var_types: Vec<Option<Type>>) -> Self {
        Self {
            defs: vec![HashMap::new(); blocks],
            entry: vec![HashMap::new(); blocks],
            processed: vec![false; blocks],
            pending: Vec::new(),
            new_params: vec![Vec::new(); blocks],
            var_types,
        }
    }

    fn write(&mut self, block: usize, var: usize, value: ValueId) {
        self.defs[block].insert(var, value);
    }

    fn read(&mut self, function: &mut Function, cfg: &Cfg, block: usize, var: usize) -> ValueId {
        if let Some(&value) = self.defs[block].get(&var) {
            return value;
        }
        // Walk single-predecessor chains iteratively, then resolve backwards.
        let mut chain: SmallVec<[usize; 8]> = SmallVec::new();
        let mut current = block;
        let value = loop {
            if let Some(&value) = self.entry[current].get(&var) {
                break value;
            }
            let preds = &cfg.preds[current];
            if preds.len() == 1 && self.processed[preds[0]] && preds[0] != current {
                chain.push(current);
                let pred = preds[0];
                if let Some(&value) = self.defs[pred].get(&var) {
                    break value;
                }
                current = pred;
                continue;
            }
            let param = self.add_phi(function, current, var);
            break param;
        };
        for block in chain {
            self.entry[block].insert(var, value);
        }
        value
    }

    fn add_phi(&mut self, function: &mut Function, block: usize, var: usize) -> ValueId {
        let id = ValueId(function.values.len() as u32);
        let index = function.blocks[block].params.len();
        let ty = self.var_types[var].clone().unwrap_or(Type::Bottom);
        function.values.push(Value {
            id,
            ty,
            def: ValueDef::BlockParam(BlockId(block as u32), index as u16),
        });
        function.blocks[block].params.push(id);
        self.entry[block].insert(var, id);
        self.new_params[block].push((var, id));
        self.pending.push((block, var, id));
        id
    }
}

struct Rewriter<'a> {
    cfg: &'a Cfg,
    candidates: &'a [Candidate],
    plans: &'a [Plan],
    layout: VarLayout,
    ssa: SsaBuilder,
    selected: Vec<bool>,
    alloc_of: HashMap<InstId, usize>,
    /// (block, pos) -> uses of selected candidates.
    events: HashMap<(usize, usize), SmallVec<[(usize, UseOp); 2]>>,
    /// Immutable operands of virtual ranges/closures/lists (after rewriting).
    fixed: Vec<SmallVec<[ValueId; 4]>>,
    state: Vec<VState>,
    rewritten: HashSet<InstId>,
    stats: ScalarReplacementStats,
}

fn run_round(function: &mut Function, stats: &mut ScalarReplacementStats) -> bool {
    let analysis = Analysis::run(function);
    let selected: Vec<usize> = analysis
        .plans
        .iter()
        .enumerate()
        .filter_map(|(index, plan)| (plan.valid && plan.selected).then_some(index))
        .collect();
    if selected.is_empty() {
        return false;
    }

    neutralize_unreachable_blocks(function, &analysis.cfg);

    let layout = VarLayout::new(&analysis.candidates, &selected);
    let mut var_types = vec![None; layout.total];
    let mut selected_mask = vec![false; analysis.candidates.len()];
    let mut alloc_of = HashMap::new();
    let mut events: HashMap<(usize, usize), SmallVec<[(usize, UseOp); 2]>> = HashMap::new();
    for &index in &selected {
        selected_mask[index] = true;
        var_types[layout.reference(index)] = Some(analysis.candidates[index].ty.clone());
        alloc_of.insert(analysis.candidates[index].inst, index);
        for uses in analysis.uses[index].values() {
            for use_ in uses {
                events
                    .entry((use_.block, use_.pos))
                    .or_default()
                    .push((index, use_.op));
            }
        }
        let plan = &analysis.plans[index];
        match plan.state() {
            EscapeState::NoEscape => stats.removed_allocations += 1,
            EscapeState::PartialEscape => stats.sunk_allocations += 1,
            EscapeState::GlobalEscape => stats.forwarded_allocations += 1,
        }
    }

    let mut rewriter = Rewriter {
        cfg: &analysis.cfg,
        candidates: &analysis.candidates,
        plans: &analysis.plans,
        ssa: SsaBuilder::new(function.blocks.len(), var_types),
        layout,
        selected: selected_mask,
        alloc_of,
        events,
        fixed: vec![SmallVec::new(); analysis.candidates.len()],
        state: vec![VState::Unknown; analysis.candidates.len()],
        rewritten: HashSet::new(),
        stats: ScalarReplacementStats::default(),
    };

    for &block in &analysis.cfg.rpo {
        rewriter.rewrite_block(function, block);
    }
    rewriter.finish(function);

    stats.materializations += rewriter.stats.materializations;
    stats.scalarized_reads += rewriter.stats.scalarized_reads;

    let live = live_nodes(function);
    let rewritten = &rewriter.rewritten;
    function
        .profiles
        .numeric_sites
        .retain(|id, _| live.contains(id) && !rewritten.contains(id));

    debug_assert_no_removed_uses(function, &analysis.candidates, &rewriter.selected);
    true
}

impl<'a> Rewriter<'a> {
    fn in_scope(&self, candidate: usize, block: usize) -> bool {
        let alloc_block = self.candidates[candidate].block;
        self.cfg.strictly_dominates(alloc_block, block)
    }

    fn rewrite_block(&mut self, function: &mut Function, block: usize) {
        for index in 0..self.candidates.len() {
            if !self.selected[index] {
                continue;
            }
            self.state[index] = if self.in_scope(index, block) {
                self.plans[index].in_state[block]
            } else {
                VState::Unknown
            };
        }

        let old = function.blocks[block].insts.clone();
        let mut new_insts: Vec<InstId> = Vec::with_capacity(old.len() + 4);
        let last = old.len().saturating_sub(1);

        for (pos, &inst) in old.iter().enumerate() {
            let is_terminator = pos == last && function.nodes[inst.0 as usize].kind.is_terminator();
            if is_terminator {
                for index in 0..self.candidates.len() {
                    if self.selected[index]
                        && self.plans[index].forced_end[block]
                        && matches!(self.state[index], VState::Virtual(_))
                    {
                        self.materialize(function, block, index, &mut new_insts);
                    }
                }
            }

            let events = self.events.get(&(block, pos)).cloned().unwrap_or_default();

            // 1. Escaping uses materialize the virtual object first.
            for &(index, op) in &events {
                if let VState::Virtual(len) = self.state[index] {
                    if virtual_step(len, op).is_none() {
                        self.materialize(function, block, index, &mut new_insts);
                    }
                }
            }

            // 2. Materialized objects are referenced through their SSA ref var.
            for &(index, _) in &events {
                if self.state[index] == VState::Materialized {
                    let reference = self.layout.reference(index);
                    let value = self.ssa.read(function, self.cfg, block, reference);
                    let original = self.candidates[index].value;
                    function.nodes[inst.0 as usize]
                        .kind
                        .for_each_operand_mut(|operand| {
                            if *operand == original {
                                *operand = value;
                            }
                        });
                }
            }

            // 3. The allocation itself becomes virtual.
            if let Some(&index) = self.alloc_of.get(&inst) {
                self.virtualize_allocation(function, block, index, inst, &mut new_insts);
                continue;
            }

            // 4. Accesses through virtual objects become SSA data flow.
            let mut keep = true;
            for &(index, op) in &events {
                let VState::Virtual(len) = self.state[index] else {
                    continue;
                };
                keep &= self.apply_virtual_access(function, block, inst, index, len, op, &mut new_insts);
            }
            if keep {
                new_insts.push(inst);
            }
        }

        function.blocks[block].insts = new_insts;
        self.ssa.processed[block] = true;
    }

    fn virtualize_allocation(
        &mut self,
        function: &mut Function,
        block: usize,
        index: usize,
        inst: InstId,
        new_insts: &mut Vec<InstId>,
    ) {
        let kind = function.nodes[inst.0 as usize].kind.clone();
        let candidate = &self.candidates[index];
        match (&candidate.shape, &kind) {
            (Shape::Instance { .. }, _) => {
                if !candidate.slots.is_empty() {
                    let (null_inst, results) = InstFactory::new(function).create(
                        Inst::Const(Constant::Null),
                        &[Type::Null],
                        candidate.source,
                    );
                    new_insts.push(null_inst);
                    for field in 0..candidate.slots.len() {
                        let var = self.layout.field(index, field);
                        self.ssa.write(block, var, results[0]);
                    }
                }
            }
            (Shape::Cell, Inst::AllocCell { initial }) => {
                let var = self.layout.field(index, 0);
                self.ssa.write(block, var, *initial);
            }
            (Shape::List { .. }, Inst::AllocList { capacity }) => {
                self.fixed[index] = capacity.iter().copied().collect();
            }
            (Shape::Range { .. }, Inst::MakeRange { from, to, .. }) => {
                self.fixed[index] = SmallVec::from_slice(&[*from, *to]);
            }
            (Shape::Closure { .. }, Inst::MakeClosure { captures, .. }) => {
                self.fixed[index] = captures.iter().copied().collect();
            }
            _ => {}
        }
        self.state[index] = VState::Virtual(0);
    }

    /// Returns whether the instruction must be kept in the block.
    #[allow(clippy::too_many_arguments)]
    fn apply_virtual_access(
        &mut self,
        function: &mut Function,
        block: usize,
        inst: InstId,
        index: usize,
        len: u32,
        op: UseOp,
        new_insts: &mut Vec<InstId>,
    ) -> bool {
        let kind = function.nodes[inst.0 as usize].kind.clone();
        let slot_var = |this: &Self, slot: FieldSlot| {
            let field = this.candidates[index]
                .slots
                .binary_search(&slot)
                .expect("accessed slot was collected by the analysis");
            this.layout.field(index, field)
        };
        match (op, &kind) {
            (UseOp::LoadField(slot), _) => {
                let var = slot_var(self, slot);
                let value = self.ssa.read(function, self.cfg, block, var);
                self.replace_kind(function, inst, Inst::Move(value));
                self.stats.scalarized_reads += 1;
                true
            }
            (UseOp::StoreField(slot), Inst::StoreField { value, .. }) => {
                let var = slot_var(self, slot);
                self.ssa.write(block, var, *value);
                false
            }
            (UseOp::LoadCell, _) => {
                let var = self.layout.field(index, 0);
                let value = self.ssa.read(function, self.cfg, block, var);
                self.replace_kind(function, inst, Inst::Move(value));
                self.stats.scalarized_reads += 1;
                true
            }
            (UseOp::StoreCell, Inst::StoreCell { value, .. }) => {
                let var = self.layout.field(index, 0);
                self.ssa.write(block, var, *value);
                false
            }
            (UseOp::ListLen, _) => {
                self.replace_kind(function, inst, Inst::Const(Constant::Int32(len as i32)));
                self.stats.scalarized_reads += 1;
                true
            }
            (UseOp::ListGet(raw), _) => {
                let element = normalize_index(raw, len).expect("checked by virtual_step") as usize;
                let var = self.layout.field(index, element);
                let value = self.ssa.read(function, self.cfg, block, var);
                self.replace_kind(function, inst, Inst::Move(value));
                self.stats.scalarized_reads += 1;
                true
            }
            (UseOp::ListSet(raw), Inst::ListSet { value, .. }) => {
                let element = normalize_index(raw, len).expect("checked by virtual_step") as usize;
                let var = self.layout.field(index, element);
                self.ssa.write(block, var, *value);
                false
            }
            (UseOp::ListAppend, Inst::ListAppend { value, .. }) => {
                let var = self.layout.field(index, len as usize);
                self.ssa.write(block, var, *value);
                self.state[index] = VState::Virtual(len + 1);
                false
            }
            (UseOp::WriteBarrier, _) => false,
            (UseOp::RefEq, Inst::RefEq { left, right }) => {
                // A virtual object is only identical to itself: it has never
                // been exposed, so no other value can alias it.
                let equal = left == right;
                self.replace_kind(function, inst, Inst::Const(Constant::Bool(equal)));
                self.stats.scalarized_reads += 1;
                true
            }
            (UseOp::RefEq, Inst::Const(_)) => true,
            (UseOp::BranchCond, Inst::Branch { .. }) => {
                // Objects are always truthy in Wren.
                let source = function.nodes[inst.0 as usize].source;
                let (const_inst, results) = InstFactory::new(function).create(
                    Inst::Const(Constant::Bool(true)),
                    &[Type::Bool],
                    source,
                );
                new_insts.push(const_inst);
                if let Inst::Branch { cond, .. } = &mut function.nodes[inst.0 as usize].kind {
                    *cond = results[0];
                }
                self.stats.scalarized_reads += 1;
                true
            }
            _ => unreachable!("use {op:?} does not match instruction {kind:?}"),
        }
    }

    fn replace_kind(&mut self, function: &mut Function, inst: InstId, kind: Inst) {
        function.nodes[inst.0 as usize].kind = kind;
        self.rewritten.insert(inst);
    }

    fn materialize(
        &mut self,
        function: &mut Function,
        block: usize,
        index: usize,
        new_insts: &mut Vec<InstId>,
    ) {
        let VState::Virtual(len) = self.state[index] else {
            return;
        };
        let candidate = &self.candidates[index];
        let ty = candidate.ty.clone();
        let source = candidate.source;
        let shape = candidate.shape.clone();
        let slots = candidate.slots.clone();

        let alloc_kind = match &shape {
            Shape::Instance { class } => Inst::AllocInstance { class: *class },
            Shape::Cell => {
                let var = self.layout.field(index, 0);
                Inst::AllocCell {
                    initial: self.ssa.read(function, self.cfg, block, var),
                }
            }
            Shape::List { .. } => Inst::AllocList {
                capacity: self.fixed[index].first().copied(),
            },
            Shape::Range { inclusive } => Inst::MakeRange {
                from: self.fixed[index][0],
                to: self.fixed[index][1],
                inclusive: *inclusive,
            },
            Shape::Closure { function: callee } => Inst::MakeClosure {
                function: *callee,
                captures: self.fixed[index].to_vec(),
            },
            Shape::Map => Inst::AllocMap,
        };
        let (alloc_inst, results) = InstFactory::new(function).create(alloc_kind, &[ty], source);
        let object = results[0];
        new_insts.push(alloc_inst);

        match &shape {
            Shape::Instance { .. } => {
                for (field, slot) in slots.iter().enumerate() {
                    let var = self.layout.field(index, field);
                    let value = self.ssa.read(function, self.cfg, block, var);
                    // Fresh instances already hold null in every field.
                    if matches!(resolve_constant(function, value), Some(Constant::Null)) {
                        continue;
                    }
                    let (store, _) = InstFactory::new(function).create(
                        Inst::StoreField {
                            object,
                            slot: *slot,
                            value,
                        },
                        &[],
                        source,
                    );
                    new_insts.push(store);
                }
            }
            Shape::List { .. } => {
                for element in 0..len as usize {
                    let var = self.layout.field(index, element);
                    let value = self.ssa.read(function, self.cfg, block, var);
                    let (append, _) = InstFactory::new(function).create(
                        Inst::ListAppend {
                            list: object,
                            value,
                        },
                        &[],
                        source,
                    );
                    new_insts.push(append);
                }
            }
            _ => {}
        }

        let reference = self.layout.reference(index);
        self.ssa.write(block, reference, object);
        self.state[index] = VState::Materialized;
        self.stats.materializations += 1;
    }

    fn finish(&mut self, function: &mut Function) {
        // Fill phi operands; reading predecessors may create further phis.
        let mut cursor = 0;
        while cursor < self.ssa.pending.len() {
            let (block, var, _) = self.ssa.pending[cursor];
            cursor += 1;
            for pred in self.cfg.preds[block].clone() {
                self.ssa.read(function, self.cfg, pred, var);
            }
        }

        // Type phis by joining incoming types to a fixed point.
        let mut changed = true;
        while changed {
            changed = false;
            for block in 0..function.blocks.len() {
                for &(var, param) in &self.ssa.new_params[block] {
                    if self.ssa.var_types[var].is_some() {
                        continue;
                    }
                    let mut ty = Type::Bottom;
                    for &pred in &self.cfg.preds[block] {
                        let arg = self.ssa.defs[pred]
                            .get(&var)
                            .or_else(|| self.ssa.entry[pred].get(&var))
                            .copied()
                            .expect("phi operand resolved");
                        ty = join_types(&ty, &function.values[arg.0 as usize].ty);
                    }
                    if function.values[param.0 as usize].ty != ty {
                        function.values[param.0 as usize].ty = ty;
                        changed = true;
                    }
                }
            }
        }
        for block in 0..function.blocks.len() {
            for &(_, param) in &self.ssa.new_params[block] {
                if function.values[param.0 as usize].ty == Type::Bottom {
                    function.values[param.0 as usize].ty = Type::Top;
                }
            }
        }

        // Patch every edge into blocks that received new parameters.
        let mut patches: Vec<(usize, usize, SmallVec<[ValueId; 4]>)> = Vec::new();
        for &pred in &self.cfg.rpo {
            for &succ in &self.cfg.succs[pred] {
                if self.ssa.new_params[succ].is_empty() {
                    continue;
                }
                let args = self.ssa.new_params[succ]
                    .iter()
                    .map(|&(var, _)| {
                        self.ssa.defs[pred]
                            .get(&var)
                            .or_else(|| self.ssa.entry[pred].get(&var))
                            .copied()
                            .expect("phi operand resolved")
                    })
                    .collect();
                patches.push((pred, succ, args));
            }
        }
        for (pred, succ, args) in patches {
            let Some(&term) = function.blocks[pred].insts.last() else {
                continue;
            };
            for_each_edge_mut(&mut function.nodes[term.0 as usize].kind, |edge| {
                if edge.target.0 as usize == succ {
                    edge.args.extend(args.iter().copied());
                }
            });
        }

        for block in 0..function.blocks.len() {
            if !function.blocks[block].insts.is_empty() {
                repair_effect_chain(function, block);
            }
        }
    }
}

fn join_types(left: &Type, right: &Type) -> Type {
    match (left, right) {
        (Type::Bottom, other) | (other, Type::Bottom) => other.clone(),
        _ if left == right => left.clone(),
        _ if left.accepts(right) => left.clone(),
        _ if right.accepts(left) => right.clone(),
        _ => Type::Top,
    }
}

fn for_each_edge_mut(kind: &mut Inst, mut f: impl FnMut(&mut Edge)) {
    match kind {
        Inst::Jump { edge } => f(edge),
        Inst::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            f(then_edge);
            f(else_edge);
        }
        Inst::Switch {
            default_edge,
            cases,
            ..
        } => {
            f(default_edge);
            for case in cases {
                f(&mut case.edge);
            }
        }
        _ => {}
    }
}

/// Unreachable blocks may still hold edges into blocks that receive new
/// parameters (or stale uses of removed allocations). Replace them with a
/// lone `Unreachable`, as CFG cleanup does.
fn neutralize_unreachable_blocks(function: &mut Function, cfg: &Cfg) {
    for block in 0..function.blocks.len() {
        if cfg.reachable[block] {
            continue;
        }
        let already = function.blocks[block].insts.len() == 1
            && matches!(
                function.nodes[function.blocks[block].insts[0].0 as usize].kind,
                Inst::Unreachable
            );
        if already {
            continue;
        }
        let (inst, _) = InstFactory::new(function).create(Inst::Unreachable, &[], None);
        function.blocks[block].insts = vec![inst];
        repair_effect_chain(function, block);
    }
}

fn live_nodes(function: &Function) -> HashSet<InstId> {
    function
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter().copied())
        .collect()
}

fn debug_assert_no_removed_uses(function: &Function, candidates: &[Candidate], selected: &[bool]) {
    if !cfg!(debug_assertions) {
        return;
    }
    let removed: HashSet<ValueId> = candidates
        .iter()
        .zip(selected)
        .filter_map(|(candidate, selected)| selected.then_some(candidate.value))
        .collect();
    for block in &function.blocks {
        for inst in &block.insts {
            for operand in function.nodes[inst.0 as usize].kind.operands() {
                assert!(
                    !removed.contains(&operand),
                    "scalar replacement left a use of removed allocation {operand:?} in {inst:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecodecompiler::BytecodeCompiler;
    use crate::wren_interpreter::{ExecutionResult, Interpreter};
    use crate::wren_ir::{
        verify_function, Builder, DeoptSnapshot, FunctionKind, Program, SnapshotSlot,
    };
    use crate::wren_ir_optimizer::Optimize;
    use crate::wren_ir_optimizing::optimization_passes;

    fn count(function: &Function, pred: impl Fn(&Inst) -> bool) -> usize {
        function
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| pred(&function.nodes[inst.0 as usize].kind))
            .count()
    }

    fn is_alloc(kind: &Inst) -> bool {
        matches!(
            kind,
            Inst::AllocInstance { .. }
                | Inst::AllocCell { .. }
                | Inst::AllocList { .. }
                | Inst::MakeClosure { .. }
                | Inst::MakeRange { .. }
                | Inst::AllocMap
        )
    }

    fn sroa(program: &mut Program, id: FuncId) -> ScalarReplacementStats {
        let function = program.functions.iter_mut().find(|f| f.id == id).unwrap();
        let mut pass = ScalarReplacement::default();
        pass.optimize_function(function);
        verify_function(function).unwrap_or_else(|errors| panic!("{errors:#?}\n{function}"));
        pass.stats
    }

    fn run(program: &Program, entry: FuncId) -> crate::wren_interpreter::Value {
        let bytecode = BytecodeCompiler::default().compile(program).unwrap();
        let mut vm = Interpreter::new(&bytecode).unwrap();
        vm.set_jit_options(None);
        let ExecutionResult::Returned(value) = vm.run_module(entry).unwrap() else {
            panic!("module should return");
        };
        value
    }

    fn new_program() -> (Program, crate::wren_ir::ModuleId) {
        let mut program = Program::new();
        let module = program.add_module("main", 0);
        (program, module)
    }

    #[test]
    fn exceptional_liveness_pins_only_the_object_crossing_the_throw() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "exceptions", FunctionKind::ModuleBody, vec![Type::Num], Type::Top);
        let input = b.param(0);
        let index = b.emit1(Inst::NumToInt32 { value: input }, Type::Int32);
        let one = b.const_num(1.0);
        let crossing = b.alloc_cell(one, Type::Num);
        let body = b.new_block(&[]);
        let handler = b.new_block(&[Type::Top]);
        let error = b.block_param(handler, 0);
        b.jump(body, vec![]);
        b.position_at(body);
        let dead_before_throw = b.alloc_cell(one, Type::Num);
        let before = b.load_cell(dead_before_throw, Type::Num);
        let list = b.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        b.emit0(Inst::ListAppend { list, value: before });
        b.emit1(Inst::ListGet { list, index }, Type::Top);
        let two = b.const_num(2.0);
        b.store_cell(crossing, two);
        let after_throw = b.alloc_cell(two, Type::Num);
        let after = b.load_cell(after_throw, Type::Num);
        b.ret(after);
        b.position_at(handler);
        let caught = b.load_cell(crossing, Type::Num);
        let handler_local = b.alloc_cell(caught, Type::Num);
        let result = b.load_cell(handler_local, Type::Num);
        b.ret(result);
        b.add_exception_region(vec![body], handler, error, 0);
        let entry = b.finish();
        let stats = sroa(&mut program, entry);
        assert_eq!(stats.removed_allocations, 3);
        assert_eq!(count(program.function(entry).unwrap(), |i| matches!(i, Inst::AllocCell { .. })), 1);
        let bytecode = BytecodeCompiler::default().compile(&program).unwrap();
        for (index, expected) in [(0, 2.0), (3, 1.0)] {
            let mut vm = Interpreter::new(&bytecode).unwrap();
            vm.set_jit_options(None);
            let ExecutionResult::Returned(value) = vm.run_function(entry, &[crate::wren_interpreter::Value::from_num(index as f64)]).unwrap() else { panic!("expected return") };
            assert_eq!(value.as_num(), Some(expected));
        }
    }

    /// cell = Cell(1); if (p) cell = 2 else cell = 3; loop 3 times cell = cell + cell
    #[test]
    fn non_escaping_cell_lives_in_ssa_across_branches_and_loops() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let one = b.const_num(1.0);
        let cell = b.alloc_cell(one, Type::Top);
        let cond = b.const_bool(true);
        let then_block = b.new_block(&[]);
        let else_block = b.new_block(&[]);
        let header = b.new_block(&[]);
        let i = b.add_block_param(header, Type::Int32);
        let body = b.new_block(&[]);
        let exit = b.new_block(&[]);
        b.branch(cond, then_block, vec![], else_block, vec![]);

        b.position_at(then_block);
        let two = b.const_num(2.0);
        b.store_cell(cell, two);
        let zero = b.const_i32(0);
        b.jump(header, vec![zero]);

        b.position_at(else_block);
        let three = b.const_num(3.0);
        b.store_cell(cell, three);
        let zero = b.const_i32(0);
        b.jump(header, vec![zero]);

        b.position_at(header);
        let limit = b.const_i32(3);
        let more = b.emit1(Inst::CmpInt { kind: crate::wren_ir::CmpKind::Lt, left: i, right: limit }, Type::Bool);
        b.branch(more, body, vec![], exit, vec![]);

        b.position_at(body);
        let current = b.load_cell(cell, Type::Num);
        let as_f64 = b.emit1(Inst::NumToF64 { value: current }, Type::F64);
        let sum = b.emit1(Inst::F64Bin { op: crate::wren_ir::F64BinOp::Add, left: as_f64, right: as_f64 }, Type::F64);
        let boxed = b.emit1(Inst::F64ToNum { value: sum }, Type::Num);
        b.store_cell(cell, boxed);
        let step = b.const_i32(1);
        let next = b.emit1(Inst::IntBin { op: crate::wren_ir::IntBinOp::Add, left: i, right: step }, Type::Int32);
        b.jump(header, vec![next]);

        b.position_at(exit);
        let result = b.load_cell(cell, Type::Top);
        b.ret(result);
        let entry = b.finish();

        let stats = sroa(&mut program, entry);
        assert_eq!(stats.removed_allocations, 1);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, is_alloc), 0, "{function}");
        assert_eq!(count(function, |k| matches!(k, Inst::LoadCell { .. } | Inst::StoreCell { .. })), 0);
        assert_eq!(run(&program, entry).as_num(), Some(16.0));
    }

    #[test]
    fn non_escaping_list_is_scalarized() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let capacity = b.const_i32(2);
        let list = b.emit1(Inst::AllocList { capacity: Some(capacity) }, Type::List(None));
        let ten = b.const_num(10.0);
        let thirty = b.const_num(30.0);
        b.emit0(Inst::ListAppend { list, value: ten });
        b.emit0(Inst::ListAppend { list, value: thirty });
        let last = b.const_i32(-1);
        let five = b.const_num(5.0);
        b.emit0(Inst::ListSet { list, index: last, value: five });
        let len = b.emit1(Inst::ListLen { list }, Type::Int32);
        let len_num = b.emit1(Inst::Int32ToNum { value: len }, Type::Num);
        let first_index = b.const_i32(0);
        let first = b.emit1(Inst::ListGet { list, index: first_index }, Type::Num);
        let second_index = b.const_i32(1);
        let second = b.emit1(Inst::ListGet { list, index: second_index }, Type::Num);
        let a = b.emit1(Inst::NumToF64 { value: first }, Type::F64);
        let c = b.emit1(Inst::NumToF64 { value: second }, Type::F64);
        let d = b.emit1(Inst::NumToF64 { value: len_num }, Type::F64);
        let ac = b.emit1(Inst::F64Bin { op: crate::wren_ir::F64BinOp::Add, left: a, right: c }, Type::F64);
        let acd = b.emit1(Inst::F64Bin { op: crate::wren_ir::F64BinOp::Add, left: ac, right: d }, Type::F64);
        let result = b.emit1(Inst::F64ToNum { value: acd }, Type::Num);
        b.ret(result);
        let entry = b.finish();

        sroa(&mut program, entry);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, is_alloc), 0, "{function}");
        assert_eq!(count(function, |k| matches!(k, Inst::ListAppend { .. } | Inst::ListGet { .. } | Inst::ListSet { .. } | Inst::ListLen { .. })), 0);
        assert_eq!(run(&program, entry).as_num(), Some(17.0));
    }

    #[test]
    fn out_of_bounds_list_access_keeps_runtime_error() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let list = b.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let index = b.const_i32(3);
        let value = b.emit1(Inst::ListGet { list, index }, Type::Top);
        b.ret(value);
        let entry = b.finish();

        sroa(&mut program, entry);
        let function = program.function(entry).unwrap();
        // The failing access forces materialization: the error is preserved.
        assert_eq!(count(function, |k| matches!(k, Inst::ListGet { .. })), 1);
        assert_eq!(count(function, |k| matches!(k, Inst::AllocList { .. })), 1);
    }

    #[test]
    fn dead_closure_and_captured_cell_chain_disappears() {
        let (mut program, module) = new_program();
        let callee = {
            let mut b = Builder::new(&mut program, module, "callee", FunctionKind::Closure, vec![], Type::Top);
            let null = b.const_null();
            b.ret(null);
            b.finish()
        };
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let seven = b.const_num(7.0);
        let cell = b.alloc_cell(seven, Type::Top);
        let _closure = b.make_closure(callee, vec![cell]);
        let value = b.load_cell(cell, Type::Top);
        b.ret(value);
        let entry = b.finish();

        sroa(&mut program, entry);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, is_alloc), 0, "{function}");
        assert_eq!(run(&program, entry).as_num(), Some(7.0));
    }

    /// The object escapes only on one branch: allocation is sunk there.
    #[test]
    fn partially_escaping_cell_is_materialized_only_on_escaping_path() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![Type::Bool], Type::Top);
        let flag = b.param(0);
        let one = b.const_num(1.0);
        let cell = b.alloc_cell(one, Type::Num);
        let five = b.const_num(5.0);
        b.store_cell(cell, five);
        let escape = b.new_block(&[]);
        let local = b.new_block(&[]);
        b.branch(flag, escape, vec![], local, vec![]);

        b.position_at(escape);
        b.ret(cell);

        b.position_at(local);
        let value = b.load_cell(cell, Type::Num);
        b.ret(value);
        let entry = b.finish();

        let info = EscapeAnalysis::analyze(program.function(entry).unwrap());
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].state, EscapeState::PartialEscape);

        let stats = sroa(&mut program, entry);
        assert_eq!(stats.sunk_allocations, 1);
        let function = program.function(entry).unwrap();
        let entry_block = &function.blocks[0];
        assert!(entry_block.insts.iter().all(|i| !is_alloc(&function.nodes[i.0 as usize].kind)));
        let escape_block = &function.blocks[escape.0 as usize];
        let allocs: Vec<_> = escape_block
            .insts
            .iter()
            .filter_map(|i| match &function.nodes[i.0 as usize].kind {
                Inst::AllocCell { initial } => Some(*initial),
                _ => None,
            })
            .collect();
        assert_eq!(allocs, vec![five], "materialized with current field value\n{function}");
        let local_block = &function.blocks[local.0 as usize];
        assert!(local_block.insts.iter().all(|i| !is_alloc(&function.nodes[i.0 as usize].kind)));
    }

    #[test]
    fn instance_fields_are_scalar_replaced_and_identity_folds() {
        let (mut program, module) = new_program();
        let class = program.add_class(module, "Point", None, 2, 0, false);
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![Type::Bool], Type::Top);
        b.set_field_owner(class);
        let flag = b.param(0);
        let object = b.alloc_instance(class);
        let other = b.alloc_instance(class);
        let x = b.const_num(3.0);
        b.store_field(object, FieldSlot(0), x);
        let then_block = b.new_block(&[]);
        let else_block = b.new_block(&[]);
        let merge = b.new_block(&[]);
        b.branch(flag, then_block, vec![], else_block, vec![]);
        b.position_at(then_block);
        let y = b.const_num(4.0);
        b.store_field(object, FieldSlot(1), y);
        b.jump(merge, vec![]);
        b.position_at(else_block);
        b.jump(merge, vec![]);
        b.position_at(merge);
        let same = b.emit1(Inst::RefEq { left: object, right: object }, Type::Bool);
        let different = b.emit1(Inst::RefEq { left: object, right: other }, Type::Bool);
        let _ = (same, different);
        let loaded = b.load_field(object, FieldSlot(1), Type::Top);
        b.ret(loaded);
        let entry = b.finish();

        let stats = sroa(&mut program, entry);
        assert_eq!(stats.removed_allocations, 2);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, is_alloc), 0, "{function}");
        assert_eq!(count(function, |k| matches!(k, Inst::Const(Constant::Bool(true)))), 1);
        assert_eq!(count(function, |k| matches!(k, Inst::Const(Constant::Bool(false)))), 1);
        // The merged field value flows through a new block parameter.
        assert_eq!(function.blocks[merge.0 as usize].params.len(), 1, "{function}");

        let bytecode = BytecodeCompiler::default().compile(&program).unwrap();
        for (flag, expected) in [(true, Some(4.0)), (false, None)] {
            let mut vm = Interpreter::new(&bytecode).unwrap();
            vm.set_jit_options(None);
            let arg = crate::wren_interpreter::Value::from_bool(flag);
            let ExecutionResult::Returned(value) = vm.run_function(entry, &[arg]).unwrap() else {
                panic!("should return");
            };
            match expected {
                Some(num) => assert_eq!(value.as_num(), Some(num)),
                None => assert!(value == crate::wren_interpreter::Value::NULL),
            }
        }
    }

    #[test]
    fn escaping_object_with_reads_keeps_one_allocation_and_forwards_loads() {
        let (mut program, module) = new_program();
        let class = program.add_class(module, "Box", None, 1, 0, false);
        let mut b = Builder::new(&mut program, module, "ctor", FunctionKind::ModuleBody, vec![], Type::Top);
        b.set_field_owner(class);
        let object = b.alloc_instance(class);
        let x = b.const_num(9.0);
        b.store_field(object, FieldSlot(0), x);
        let loaded = b.load_field(object, FieldSlot(0), Type::Top);
        let _ = loaded;
        b.ret(object);
        let entry = b.finish();

        let stats = sroa(&mut program, entry);
        assert_eq!(stats.forwarded_allocations, 1);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, |k| matches!(k, Inst::AllocInstance { .. })), 1);
        assert_eq!(count(function, |k| matches!(k, Inst::LoadField { .. })), 0, "{function}");
        assert_eq!(count(function, |k| matches!(k, Inst::StoreField { .. })), 1);
    }

    #[test]
    fn globally_escaping_allocation_without_reads_is_untouched() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let one = b.const_num(1.0);
        let cell = b.alloc_cell(one, Type::Top);
        b.ret(cell);
        let entry = b.finish();
        let before = program.function(entry).unwrap().to_string();
        let info = EscapeAnalysis::analyze(program.function(entry).unwrap());
        assert_eq!(info[0].state, EscapeState::GlobalEscape);
        assert!(!info[0].scalar_replaceable);
        sroa(&mut program, entry);
        assert_eq!(program.function(entry).unwrap().to_string(), before);
    }

    #[test]
    fn deopt_referenced_allocation_is_pinned() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let one = b.const_num(1.0);
        let cell = b.alloc_cell(one, Type::Top);
        let resume = crate::wren_ir::ResumePoint { id: crate::wren_ir::ResumeId(0) };
        let mut snapshot = DeoptSnapshot::new(resume);
        snapshot.slots.push(SnapshotSlot { slot: 0, value: cell });
        let _deopt = b.deopt_point(snapshot);
        let value = b.load_cell(cell, Type::Top);
        b.ret(value);
        let entry = b.finish();

        let info = EscapeAnalysis::analyze(program.function(entry).unwrap());
        assert!(!info[0].scalar_replaceable);
        sroa(&mut program, entry);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, |k| matches!(k, Inst::AllocCell { .. })), 1);
    }

    #[test]
    fn loop_carried_virtual_list_with_changing_length_materializes_at_merge() {
        // list = []; while (i < 3) { list.add(i) }; return list.count
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let list = b.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let header = b.new_block(&[]);
        let i = b.add_block_param(header, Type::Int32);
        let body = b.new_block(&[]);
        let exit = b.new_block(&[]);
        let zero = b.const_i32(0);
        b.jump(header, vec![zero]);
        b.position_at(header);
        let limit = b.const_i32(3);
        let more = b.emit1(Inst::CmpInt { kind: crate::wren_ir::CmpKind::Lt, left: i, right: limit }, Type::Bool);
        b.branch(more, body, vec![], exit, vec![]);
        b.position_at(body);
        let boxed = b.emit1(Inst::Int32ToNum { value: i }, Type::Num);
        b.emit0(Inst::ListAppend { list, value: boxed });
        let step = b.const_i32(1);
        let next = b.emit1(Inst::IntBin { op: crate::wren_ir::IntBinOp::Add, left: i, right: step }, Type::Int32);
        b.jump(header, vec![next]);
        b.position_at(exit);
        let len = b.emit1(Inst::ListLen { list }, Type::Int32);
        let result = b.emit1(Inst::Int32ToNum { value: len }, Type::Num);
        b.ret(result);
        let entry = b.finish();

        // Length differs around the loop: not representable as scalars, so the
        // allocation stays (no virtual reads, no virtual death).
        sroa(&mut program, entry);
        assert_eq!(run(&program, entry).as_num(), Some(3.0));
    }

    #[test]
    fn default_pipeline_runs_scalar_replacement() {
        let (mut program, module) = new_program();
        let mut b = Builder::new(&mut program, module, "main", FunctionKind::ModuleBody, vec![], Type::Top);
        let one = b.const_num(1.0);
        let cell = b.alloc_cell(one, Type::Top);
        let two = b.const_num(2.0);
        b.store_cell(cell, two);
        let value = b.load_cell(cell, Type::Top);
        b.ret(value);
        let entry = b.finish();

        let mut passes: Optimize = optimization_passes();
        passes.optimize_program(&mut program);
        let function = program.function(entry).unwrap();
        assert_eq!(count(function, is_alloc), 0, "{function}");
        assert_eq!(function.blocks[0].insts.len(), 2, "{function}");
        assert_eq!(run(&program, entry).as_num(), Some(2.0));
    }

    /// End-to-end over real Wren source: constructor-local field reads are
    /// forwarded from the virtual object state instead of re-loading the heap.
    /// (Captured locals escape through `Fn.new(closure)`, an `Invoke`, so they
    /// are intentionally *not* expected to disappear here.)
    #[test]
    fn lowered_source_forwards_constructor_field_reads() {
        let source = r#"
class Pair {
  construct new(a, b) {
    _a = a
    _b = b
    _s = _a + _b
    _t = _s * _a
  }
  t { _t }
}
var result = Pair.new(3, 4).t
"#;
        let parsed = crate::wren_front::parse("sroa.wren", source);
        assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
        let mut hir = Program::new();
        let _manifest = crate::wren_builtin::install_hir_builtins(&mut hir);
        let lowered = crate::wren_lowerer::lower_module(&mut hir, "sroa.wren", source, "sroa", &parsed.module);
        assert!(!lowered.has_errors(), "{:#?}", lowered.diagnostics);

        let loads = |hir: &Program| -> usize {
            hir.functions.iter().map(|f| count(f, |k| matches!(k, Inst::LoadField { .. }))).sum()
        };
        let before = loads(&hir);
        let mut passes: Optimize = optimization_passes();
        passes.optimize_program(&mut hir);
        for function in &hir.functions {
            verify_function(function).unwrap_or_else(|errors| panic!("{errors:#?}\n{function}"));
        }
        let after = loads(&hir);
        // Only the `t` getter's load must remain; the four constructor reads go.
        assert_eq!(before - after, 4, "field loads {before} -> {after}");
    }
}
