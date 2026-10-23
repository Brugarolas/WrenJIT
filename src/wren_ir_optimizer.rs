//! Wren WIR optimizer core.
//!
//! This is an adaptation of the old Napalm optimizer infrastructure to the
//! current Wren SSA IR. The Wren IR uses two independent SSA namespaces:
//! `ValueId` for data and `EffectId` for memory/control effects. Structural
//! edits therefore repair the effect chain after every block rewrite.
//!
//! Important IR contract:
//! - `Function::nodes`, `values`, and `effects` are append-only arenas.
//! - `Block::insts` is the authoritative list of live instructions.
//! - Detached nodes are dead optimizer arena entries and must be accepted by
//!   the verifier (see the small verifier patch supplied with this file).
//! - An instruction may only be erased after all live uses of its result values
//!   have been rewritten or proven dead.

#![allow(dead_code)]
#![allow(clippy::needless_return)]

use std::any::Any;
use hashbrown::HashMap;
use std::collections::VecDeque;
use std::mem::{discriminant, replace, Discriminant};
use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::wren_ir::{
    verify_function, BlockId, EffectBehavior, EffectDef, EffectId, EffectValue, Function, Inst,
    InstId, Node, Program, SourceSpan, Type, Value, ValueDef, ValueId,
};

/// O(1) cursor into one live instruction in a block.
///
/// WIR verifies that `BlockId(n)` lives at `function.blocks[n]`, so block IDs
/// can be converted directly to indices without a search or side table.
pub struct InstCursor<'f> {
    func: *mut Function,
    block_ix: usize,
    pos: usize,
    _marker: PhantomData<&'f mut Function>,
}

impl<'f> InstCursor<'f> {
    #[inline]
    pub fn new(func: &'f mut Function, block: BlockId, pos: usize) -> Self {
        let block_ix = block.0 as usize;
        assert!(
            func.blocks
                .get(block_ix)
                .is_some_and(|candidate| candidate.id == block),
            "invalid BlockId {block:?}"
        );
        Self::new_ix(func, block_ix, pos)
    }

    #[inline]
    pub fn new_ix(func: &'f mut Function, block_ix: usize, pos: usize) -> Self {
        debug_assert!(block_ix < func.blocks.len());
        debug_assert!(pos < func.blocks[block_ix].insts.len());
        Self {
            func: func as *mut Function,
            block_ix,
            pos,
            _marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn func_ref(&self) -> &Function {
        unsafe { &*self.func }
    }

    #[inline]
    unsafe fn func_mut(&mut self) -> &mut Function {
        unsafe { &mut *self.func }
    }

    #[inline]
    pub fn block_id(&self) -> BlockId {
        unsafe { self.func_ref().blocks[self.block_ix].id }
    }

    #[inline]
    pub fn block_index(&self) -> usize {
        self.block_ix
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe { self.func_ref().blocks[self.block_ix].insts.len() }
    }

    #[inline]
    pub fn cur_id(&self) -> InstId {
        unsafe {
            self.func_ref().blocks[self.block_ix]
                .insts
                .get(self.pos)
                .copied()
                .expect("stale InstCursor position")
        }
    }

    #[inline]
    pub fn cur_tag(&self) -> Discriminant<Inst> {
        let id = self.cur_id();
        unsafe { discriminant(&self.func_ref().nodes[id.0 as usize].kind) }
    }

    #[inline]
    pub fn cur_kind(&self) -> Inst {
        let id = self.cur_id();
        unsafe { self.func_ref().nodes[id.0 as usize].kind.clone() }
    }

    #[inline]
    pub fn peek_cur<R>(&self, f: impl FnOnce(&Inst) -> R) -> R {
        let id = self.cur_id();
        unsafe { f(&self.func_ref().nodes[id.0 as usize].kind) }
    }

    #[inline]
    pub fn peek_node<R>(&self, f: impl FnOnce(&Node) -> R) -> R {
        let id = self.cur_id();
        unsafe { f(&self.func_ref().nodes[id.0 as usize]) }
    }

    #[inline]
    pub fn with_cur<R>(&mut self, f: impl FnOnce(&mut Node) -> R) -> R {
        let id = self.cur_id();
        unsafe { f(&mut self.func_mut().nodes[id.0 as usize]) }
    }

    #[inline]
    pub fn with_prev<R>(&mut self, f: impl FnOnce(&mut Node) -> R) -> Option<R> {
        if self.pos == 0 {
            return None;
        }
        let id = unsafe { self.func_ref().blocks[self.block_ix].insts[self.pos - 1] };
        Some(unsafe { f(&mut self.func_mut().nodes[id.0 as usize]) })
    }

    #[inline]
    pub fn with_next<R>(&mut self, f: impl FnOnce(&mut Node) -> R) -> Option<R> {
        if self.pos + 1 >= self.len() {
            return None;
        }
        let id = unsafe { self.func_ref().blocks[self.block_ix].insts[self.pos + 1] };
        Some(unsafe { f(&mut self.func_mut().nodes[id.0 as usize]) })
    }

    #[inline]
    pub fn with_func<R>(&mut self, f: impl FnOnce(&mut Function) -> R) -> R {
        unsafe { f(self.func_mut()) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub prev2: Option<InstId>,
    pub prev1: Option<InstId>,
    pub cur: InstId,
    pub next1: Option<InstId>,
    pub next2: Option<InstId>,
}

impl InstCursor<'_> {
    #[inline]
    pub fn window5(&self) -> Window {
        let block = unsafe { &self.func_ref().blocks[self.block_ix] };
        let p = self.pos;
        let at = |i: usize| block.insts.get(i).copied();
        Window {
            prev2: p.checked_sub(2).and_then(at),
            prev1: p.checked_sub(1).and_then(at),
            cur: at(p).expect("stale InstCursor position"),
            next1: at(p + 1),
            next2: at(p + 2),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WirEdit {
    InsertBefore {
        at: usize,
        insts: SmallVec<[InstId; 4]>,
    },
    InsertAfter {
        at: usize,
        insts: SmallVec<[InstId; 4]>,
    },
    Erase {
        at: usize,
    },
    SpliceRange {
        start: usize,
        end: usize,
        insts: SmallVec<[InstId; 4]>,
    },
}

/// Deferred edits.
///
/// Structural edits are bucketed per block. Replacements are keyed by InstId;
/// their block is resolved through `FuncState` at application time.
pub struct WirEditQueue {
    by_block: Vec<SmallVec<[WirEdit; 8]>>,
    replacements: SmallVec<[(InstId, Inst); 16]>,
}

impl Default for WirEditQueue {
    fn default() -> Self {
        Self {
            by_block: Vec::new(),
            replacements: SmallVec::new(),
        }
    }
}

impl WirEditQueue {
    pub fn with_block_count(blocks: usize) -> Self {
        Self {
            by_block: vec![SmallVec::new(); blocks],
            replacements: SmallVec::new(),
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn ensure_blocks(&mut self, n: usize) {
        if self.by_block.len() < n {
            self.by_block.resize_with(n, SmallVec::new);
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        for edits in &mut self.by_block {
            edits.clear();
        }
        self.replacements.clear();
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.by_block.iter().all(|edits| edits.is_empty())
    }

    #[inline]
    pub fn insert_before_ix(
        &mut self,
        block_ix: usize,
        at: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.ensure_blocks(block_ix + 1);
        self.by_block[block_ix].push(WirEdit::InsertBefore {
            at,
            insts: insts.into(),
        });
    }

    #[inline]
    pub fn insert_after_ix(
        &mut self,
        block_ix: usize,
        at: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.ensure_blocks(block_ix + 1);
        self.by_block[block_ix].push(WirEdit::InsertAfter {
            at,
            insts: insts.into(),
        });
    }

    #[inline]
    pub fn erase_ix(&mut self, block_ix: usize, at: usize) {
        self.ensure_blocks(block_ix + 1);
        self.by_block[block_ix].push(WirEdit::Erase { at });
    }

    #[inline]
    pub fn splice_range_ix(
        &mut self,
        block_ix: usize,
        start: usize,
        end: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.ensure_blocks(block_ix + 1);
        self.by_block[block_ix].push(WirEdit::SpliceRange {
            start,
            end,
            insts: insts.into(),
        });
    }

    /// Dense WIR block IDs make the compatibility API O(1) as well.
    #[inline]
    pub fn insert_before(
        &mut self,
        block: BlockId,
        at: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.insert_before_ix(block.0 as usize, at, insts);
    }

    #[inline]
    pub fn insert_after(
        &mut self,
        block: BlockId,
        at: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.insert_after_ix(block.0 as usize, at, insts);
    }

    #[inline]
    pub fn erase(&mut self, block: BlockId, at: usize) {
        self.erase_ix(block.0 as usize, at);
    }

    #[inline]
    pub fn splice_range(
        &mut self,
        block: BlockId,
        start: usize,
        end: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.splice_range_ix(block.0 as usize, start, end, insts);
    }

    #[inline]
    pub fn replace_kind(&mut self, id: InstId, new_kind: Inst) {
        self.replacements.push((id, new_kind));
    }

    #[inline]
    fn has_structural_edits(&self, block_ix: usize) -> bool {
        self.by_block
            .get(block_ix)
            .is_some_and(|edits| !edits.is_empty())
    }

    #[inline]
    pub fn has_any_for_block_ix(&self, block_ix: usize, fs: &FuncState) -> bool {
        self.has_structural_edits(block_ix)
            || self
                .replacements
                .iter()
                .any(|(id, _)| fs.pos_of(*id).is_some_and(|(bix, _)| bix == block_ix))
    }

    /// Apply all currently queued edits.
    pub fn apply(&mut self, function: &mut Function) {
        let mut fs = FuncState::new(function);
        self.ensure_blocks(function.blocks.len());
        for block_ix in 0..function.blocks.len() {
            if self.has_any_for_block_ix(block_ix, &fs) {
                self.apply_block(function, &mut fs, block_ix);
            }
        }
    }

    fn apply_replacements_for_block(
        &mut self,
        function: &mut Function,
        fs: &FuncState,
        block_ix: usize,
    ) -> bool {
        let mut changed = false;
        let mut i = 0;
        while i < self.replacements.len() {
            let id = self.replacements[i].0;
            if fs.pos_of(id).is_some_and(|(bix, _)| bix == block_ix) {
                let (_, new_kind) = self.replacements.swap_remove(i);
                let node = &mut function.nodes[id.0 as usize];
                assert_eq!(
                    node.results.len(),
                    new_kind.result_count(),
                    "replace_kind changed result arity for {id:?}; use a value-rewriting transform instead"
                );
                let _old = replace(&mut node.kind, new_kind);
                changed = true;
            } else {
                i += 1;
            }
        }
        changed
    }

    fn apply_block(&mut self, function: &mut Function, fs: &mut FuncState, block_ix: usize) {
        self.ensure_blocks(function.blocks.len());

        let replaced = self.apply_replacements_for_block(function, fs, block_ix);
        let structural = self.has_structural_edits(block_ix);
        if !replaced && !structural {
            return;
        }

        if structural {
            let mut edits: SmallVec<[WirEdit; 16]> = SmallVec::new();
            edits.extend(self.by_block[block_ix].drain(..));

            let old = function.blocks[block_ix].insts.clone();
            let old_len = old.len();

            #[derive(Clone)]
            struct RangeEdit {
                start: usize,
                end: usize,
                replacement: SmallVec<[InstId; 4]>,
                sequence: usize,
            }

            let mut before: Vec<SmallVec<[InstId; 4]>> = vec![SmallVec::new(); old_len + 1];
            let mut after: Vec<SmallVec<[InstId; 4]>> = vec![SmallVec::new(); old_len];
            let mut ranges: SmallVec<[RangeEdit; 16]> = SmallVec::new();
            let mut sequence = 0usize;

            for edit in edits {
                match edit {
                    WirEdit::InsertBefore { at, insts } => {
                        before[at.min(old_len)].extend(insts);
                    }
                    WirEdit::InsertAfter { at, insts } => {
                        if old_len == 0 {
                            before[0].extend(insts);
                        } else {
                            after[at.min(old_len - 1)].extend(insts);
                        }
                    }
                    WirEdit::Erase { at } => {
                        if at < old_len {
                            ranges.push(RangeEdit {
                                start: at,
                                end: at + 1,
                                replacement: SmallVec::new(),
                                sequence,
                            });
                            sequence += 1;
                        }
                    }
                    WirEdit::SpliceRange { start, end, insts } => {
                        let start = start.min(old_len);
                        let end = end.min(old_len).max(start);
                        if start == end {
                            before[start].extend(insts);
                        } else {
                            ranges.push(RangeEdit {
                                start,
                                end,
                                replacement: insts,
                                sequence,
                            });
                            sequence += 1;
                        }
                    }
                }
            }

            ranges.sort_by(|a, b| {
                a.start
                    .cmp(&b.start)
                    .then_with(|| a.sequence.cmp(&b.sequence))
            });

            // Overlapping range rewrites are almost always an optimizer bug.
            for pair in ranges.windows(2) {
                assert!(
                    pair[0].end <= pair[1].start,
                    "overlapping WIR splice/erase edits in block {:?}",
                    function.blocks[block_ix].id
                );
            }

            let inserted: usize = before.iter().map(|insts| insts.len()).sum::<usize>()
                + after.iter().map(|insts| insts.len()).sum::<usize>()
                + ranges
                    .iter()
                    .map(|range| range.replacement.len())
                    .sum::<usize>();

            let mut out = Vec::with_capacity(old_len + inserted);
            let mut old_pos = 0usize;
            let mut range_ix = 0usize;

            while old_pos < old_len {
                out.extend_from_slice(&before[old_pos]);

                if range_ix < ranges.len() && ranges[range_ix].start == old_pos {
                    let range = &ranges[range_ix];
                    out.extend_from_slice(&range.replacement);
                    old_pos = range.end;
                    range_ix += 1;
                    continue;
                }

                out.push(old[old_pos]);
                out.extend_from_slice(&after[old_pos]);
                old_pos += 1;
            }
            out.extend_from_slice(&before[old_len]);

            assert_removed_results_are_dead(function, block_ix, &old, &out);

            // Mark all previous positions dead, then repopulate live positions.
            for id in old {
                fs.set_pos(id, None);
            }
            for (pos, id) in out.iter().copied().enumerate() {
                fs.ensure_inst(id);
                fs.set_pos(id, Some((block_ix, pos)));
            }
            function.blocks[block_ix].insts = out;
        }

        repair_effect_chain(function, block_ix);
    }
}

/// Reject a structural deletion that would leave a live SSA use pointing at a
/// detached instruction. Peephole rewrites that replace a value must rewrite
/// those uses before erasing the producer.
fn assert_removed_results_are_dead(
    function: &Function,
    edited_block_ix: usize,
    old: &[InstId],
    new: &[InstId],
) {
    let mut remains = vec![false; function.nodes.len()];
    for id in new.iter().copied() {
        if let Some(slot) = remains.get_mut(id.0 as usize) {
            *slot = true;
        }
    }

    let mut removed_values = vec![false; function.values.len()];
    let mut removed_any_value = false;
    for id in old.iter().copied() {
        if remains.get(id.0 as usize).copied().unwrap_or(false) {
            continue;
        }
        for value in &function.nodes[id.0 as usize].results {
            removed_values[value.0 as usize] = true;
            removed_any_value = true;
        }
    }

    if !removed_any_value {
        return;
    }

    let check_inst = |user: InstId| {
        for operand in function.nodes[user.0 as usize].kind.operands() {
            assert!(
                !removed_values[operand.0 as usize],
                "cannot detach an instruction whose result {operand:?} is still used by live instruction {user:?}"
            );
        }
    };

    for (block_ix, block) in function.blocks.iter().enumerate() {
        if block_ix == edited_block_ix {
            for user in new.iter().copied() {
                check_inst(user);
            }
        } else {
            for user in block.insts.iter().copied() {
                check_inst(user);
            }
        }
    }

    for deopt in &function.deopts {
        for slot in &deopt.snapshot.slots {
            assert!(
                !removed_values[slot.value.0 as usize],
                "cannot detach an instruction whose result {:?} is still referenced by deopt {:?}",
                slot.value,
                deopt.id
            );
        }
        for frame in &deopt.snapshot.inlined_frames {
            for slot in &frame.slots {
                assert!(
                    !removed_values[slot.value.0 as usize],
                    "cannot detach an instruction whose result {:?} is still referenced by an inlined deopt frame",
                    slot.value
                );
            }
        }
    }
}

/// Function-local instruction position index.
///
/// `None` means the arena node is currently detached/dead.
pub struct FuncState {
    pos_of_inst: Vec<Option<(usize, usize)>>,
}

impl FuncState {
    fn new(function: &Function) -> Self {
        let mut positions = vec![None; function.nodes.len()];
        for (block_ix, block) in function.blocks.iter().enumerate() {
            for (pos, id) in block.insts.iter().copied().enumerate() {
                if id.0 as usize >= positions.len() {
                    positions.resize(id.0 as usize + 1, None);
                }
                positions[id.0 as usize] = Some((block_ix, pos));
            }
        }
        Self {
            pos_of_inst: positions,
        }
    }

    #[inline]
    fn ensure_inst(&mut self, id: InstId) {
        let needed = id.0 as usize + 1;
        if self.pos_of_inst.len() < needed {
            self.pos_of_inst.resize(needed, None);
        }
    }

    #[inline]
    fn set_pos(&mut self, id: InstId, value: Option<(usize, usize)>) {
        self.ensure_inst(id);
        self.pos_of_inst[id.0 as usize] = value;
    }

    #[inline]
    pub fn pos_of(&self, id: InstId) -> Option<(usize, usize)> {
        self.pos_of_inst.get(id.0 as usize).copied().flatten()
    }
}

/// Rebuild the effect SSA chain for one live block after instruction edits.
///
/// Existing effect result IDs are reused whenever possible. New effect IDs are
/// allocated only when a formerly pure/sink instruction becomes a chain node or
/// a newly-created chain node has no output yet. Old unused effect entries stay
/// in the append-only arena.
pub(crate) fn repair_effect_chain(function: &mut Function, block_ix: usize) {
    let block_effect = function.blocks[block_ix].effect;
    let insts = function.blocks[block_ix].insts.clone();
    let mut current_effect = block_effect;

    for id in insts {
        let node_ix = id.0 as usize;
        let behavior = function.nodes[node_ix].kind.effect_behavior();

        match behavior {
            EffectBehavior::None => {
                function.nodes[node_ix].effect_in = None;
                function.nodes[node_ix].effect_out = None;
            }
            EffectBehavior::Chain(kind) => {
                let effect_out = match function.nodes[node_ix].effect_out {
                    Some(id) if (id.0 as usize) < function.effects.len() => id,
                    _ => {
                        let effect = EffectId(function.effects.len() as u32);
                        function.effects.push(EffectValue {
                            id: effect,
                            def: EffectDef::InstResult(id, kind),
                        });
                        effect
                    }
                };

                function.nodes[node_ix].effect_in = Some(current_effect);
                function.nodes[node_ix].effect_out = Some(effect_out);
                function.effects[effect_out.0 as usize].def = EffectDef::InstResult(id, kind);
                current_effect = effect_out;
            }
            EffectBehavior::Sink(_) => {
                function.nodes[node_ix].effect_in = Some(current_effect);
                function.nodes[node_ix].effect_out = None;
            }
        }

        set_edge_effects(&mut function.nodes[node_ix].kind, current_effect);
    }
}

#[inline]
fn set_edge_effects(kind: &mut Inst, effect: EffectId) {
    match kind {
        Inst::Jump { edge } => edge.effect = effect,
        Inst::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            then_edge.effect = effect;
            else_edge.effect = effect;
        }
        Inst::Switch {
            default_edge,
            cases,
            ..
        } => {
            default_edge.effect = effect;
            for case in cases {
                case.edge.effect = effect;
            }
        }
        _ => {}
    }
}

pub trait OptimizationWIR {
    fn set_metadata_any(&mut self, _data: &dyn Any) {}

    /// Return true when no later optimization in this pass should inspect the
    /// current instruction during this visit.
    fn optimize_node(&mut self, cur: &mut InstCursor<'_>, q: &mut WirEditQueue) -> bool;

    /// Cheap opcode prefilter used to build the pass-local dispatch cache.
    fn maybe_interesting(&self, _tag: Discriminant<Inst>) -> bool {
        true
    }
}

pub struct OptimizationPass {
    pub optimizations: SmallVec<[Box<dyn OptimizationWIR>; 8]>,
    pass_meta: Option<Box<dyn Any>>,
    by_tag: HashMap<Discriminant<Inst>, SmallVec<[usize; 8]>>,
}

impl OptimizationPass {
    pub fn new(optimizations: Vec<Box<dyn OptimizationWIR>>) -> Self {
        Self {
            optimizations: SmallVec::from_vec(optimizations),
            pass_meta: None,
            by_tag: HashMap::new(),
        }
    }

    pub fn initialize(&mut self, meta: impl Any) {
        self.pass_meta = Some(Box::new(meta));
        let metadata: &dyn Any = self.pass_meta.as_deref().expect("metadata just installed");
        for optimization in &mut self.optimizations {
            optimization.set_metadata_any(metadata);
        }
        self.by_tag.clear();
    }

    #[inline]
    fn interested_for(&mut self, tag: Discriminant<Inst>) -> SmallVec<[usize; 8]> {
        if let Some(cached) = self.by_tag.get(&tag) {
            return cached.clone();
        }

        let mut indices = SmallVec::new();
        for (index, optimization) in self.optimizations.iter().enumerate() {
            if optimization.maybe_interesting(tag) {
                indices.push(index);
            }
        }
        self.by_tag.insert(tag, indices.clone());
        indices
    }

    /// Worklist optimizer. Only instructions near an edit are revisited.
    pub fn pass_optimizations(&mut self, function: &mut Function, recheck_window: usize) {
        let mut state = FuncState::new(function);
        let mut edits = WirEditQueue::with_block_count(function.blocks.len());

        // Unlike the old ticket bitmap, this tracks membership in the queue,
        // not "ever seen in this block". Popping clears the bit, so a later
        // local rewrite can enqueue the instruction again.
        let mut queued = vec![false; function.nodes.len()];

        for block_ix in 0..function.blocks.len() {
            let seed = function.blocks[block_ix].insts.clone();
            let mut worklist = VecDeque::with_capacity(seed.len());
            for id in seed {
                ensure_bool_slot(&mut queued, id);
                if !queued[id.0 as usize] {
                    queued[id.0 as usize] = true;
                    worklist.push_back(id);
                }
            }

            while let Some(id) = worklist.pop_front() {
                ensure_bool_slot(&mut queued, id);
                queued[id.0 as usize] = false;

                let Some((current_block, mut pos)) = state.pos_of(id) else {
                    continue;
                };
                if current_block != block_ix {
                    continue;
                }
                if pos >= function.blocks[block_ix].insts.len() {
                    continue;
                }

                let mut cursor = InstCursor::new_ix(function, block_ix, pos);
                let tag = cursor.cur_tag();
                let interested = self.interested_for(tag);
                for optimization_ix in interested {
                    if self.optimizations[optimization_ix]
                        .optimize_node(&mut cursor, &mut edits)
                    {
                        break;
                    }
                }

                if edits.has_any_for_block_ix(block_ix, &state) {
                    edits.apply_block(function, &mut state, block_ix);

                    if queued.len() < function.nodes.len() {
                        queued.resize(function.nodes.len(), false);
                    }

                    if let Some((new_block, new_pos)) = state.pos_of(id) {
                        if new_block == block_ix {
                            pos = new_pos;
                        }
                    }

                    let len = function.blocks[block_ix].insts.len();
                    if len == 0 {
                        continue;
                    }
                    let start = pos.saturating_sub(recheck_window);
                    let end = pos
                        .saturating_add(recheck_window)
                        .saturating_add(1)
                        .min(len);

                    for index in start..end {
                        let neighbor = function.blocks[block_ix].insts[index];
                        ensure_bool_slot(&mut queued, neighbor);
                        if !queued[neighbor.0 as usize] {
                            queued[neighbor.0 as usize] = true;
                            worklist.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Normally empty because edits are applied immediately. Keep this for
        // transformations that deliberately target a different block.
        if !edits.is_empty() {
            edits.apply(function);
        }
    }
}

#[inline]
fn ensure_bool_slot(bits: &mut Vec<bool>, id: InstId) {
    let needed = id.0 as usize + 1;
    if bits.len() < needed {
        bits.resize(needed, false);
    }
}

/// Whole-function optimization stage.
///
/// Use this for transformations whose natural unit is the complete SSA graph
/// (copy propagation, DCE, CFG cleanup, global analyses). Node-local peepholes
/// should continue to use `OptimizationWIR` + `OptimizationPass`.
pub trait FunctionOptimization {
    /// Run the transform and return whether the function changed.
    fn optimize_function(&mut self, function: &mut Function) -> bool;
}

enum OptimizationStage {
    Nodes(OptimizationPass),
    Function(Box<dyn FunctionOptimization>),
}

pub struct Optimize {
    stages: SmallVec<[OptimizationStage; 4]>,
    verify_after_each_pass: bool,
}

impl Default for Optimize {
    fn default() -> Self {
        Self {
            stages: SmallVec::new(),
            verify_after_each_pass: cfg!(debug_assertions),
        }
    }
}

impl Optimize {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node-local/worklist pass.
    pub fn with_pass(mut self, pass: OptimizationPass) -> Self {
        self.stages.push(OptimizationStage::Nodes(pass));
        self
    }

    /// Add a whole-function pass.
    pub fn with_function_pass(
        mut self,
        pass: impl FunctionOptimization + 'static,
    ) -> Self {
        self.stages
            .push(OptimizationStage::Function(Box::new(pass)));
        self
    }

    pub fn verify_after_each_pass(mut self, enabled: bool) -> Self {
        self.verify_after_each_pass = enabled;
        self
    }

    pub fn optimize_program(&mut self, program: &mut Program) {
        for function in &mut program.functions {
            self.optimize_function(function);
        }
    }

    pub fn optimize_function(&mut self, function: &mut Function) {
        for stage in &mut self.stages {
            match stage {
                OptimizationStage::Nodes(pass) => {
                    pass.pass_optimizations(function, 1);
                }
                OptimizationStage::Function(pass) => {
                    let _ = pass.optimize_function(function);
                }
            }

            if self.verify_after_each_pass {
                if let Err(errors) = verify_function(function) {
                    panic!(
                        "WIR verification failed after optimization stage for {:?}: {errors:#?}",
                        function.id
                    );
                }
            }
        }
    }
}

/// Allocates detached arena nodes for insertion by `WirEditQueue`.
///
/// Effect inputs are intentionally left unset because their final predecessor
/// depends on the insertion location. A chain instruction receives a stable
/// effect result ID immediately; `repair_effect_chain()` fills its input and
/// rewires successor effects after insertion.
pub struct InstFactory<'f> {
    function: &'f mut Function,
}

impl<'f> InstFactory<'f> {
    pub fn new(function: &'f mut Function) -> Self {
        Self { function }
    }

    pub fn create(
        &mut self,
        kind: Inst,
        result_types: &[Type],
        source: Option<SourceSpan>,
    ) -> (InstId, SmallVec<[ValueId; 2]>) {
        assert_eq!(
            result_types.len(),
            kind.result_count(),
            "wrong result count for optimizer-created instruction {kind:?}"
        );
        assert!(
            result_types.len() <= u8::MAX as usize,
            "instruction has too many SSA results"
        );

        let id = InstId(self.function.nodes.len() as u32);
        let mut results = SmallVec::<[ValueId; 2]>::with_capacity(result_types.len());

        for (result_index, ty) in result_types.iter().cloned().enumerate() {
            let value = ValueId(self.function.values.len() as u32);
            self.function.values.push(Value {
                id: value,
                ty,
                def: ValueDef::InstResult(id, result_index as u8),
            });
            results.push(value);
        }

        let effect_out = match kind.effect_behavior() {
            EffectBehavior::Chain(effect_kind) => {
                let effect = EffectId(self.function.effects.len() as u32);
                self.function.effects.push(EffectValue {
                    id: effect,
                    def: EffectDef::InstResult(id, effect_kind),
                });
                Some(effect)
            }
            EffectBehavior::None | EffectBehavior::Sink(_) => None,
        };

        self.function.nodes.push(Node {
            id,
            kind,
            results: results.to_vec(),
            effect_in: None,
            effect_out,
            source,
        });

        (id, results)
    }
}

pub trait EditQueueExt {
    fn insert_before_cur(
        &mut self,
        cur: &InstCursor<'_>,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    );
    fn insert_after_cur(
        &mut self,
        cur: &InstCursor<'_>,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    );
    fn erase_cur(&mut self, cur: &InstCursor<'_>);
    fn replace_cur_kind(&mut self, cur: &InstCursor<'_>, new_kind: Inst);
    fn splice_cur_range(
        &mut self,
        cur: &InstCursor<'_>,
        start: usize,
        end: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    );
}

impl EditQueueExt for WirEditQueue {
    #[inline]
    fn insert_before_cur(
        &mut self,
        cur: &InstCursor<'_>,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.insert_before_ix(cur.block_index(), cur.pos(), insts);
    }

    #[inline]
    fn insert_after_cur(
        &mut self,
        cur: &InstCursor<'_>,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.insert_after_ix(cur.block_index(), cur.pos(), insts);
    }

    #[inline]
    fn erase_cur(&mut self, cur: &InstCursor<'_>) {
        self.erase_ix(cur.block_index(), cur.pos());
    }

    #[inline]
    fn replace_cur_kind(&mut self, cur: &InstCursor<'_>, new_kind: Inst) {
        self.replace_kind(cur.cur_id(), new_kind);
    }

    #[inline]
    fn splice_cur_range(
        &mut self,
        cur: &InstCursor<'_>,
        start: usize,
        end: usize,
        insts: impl Into<SmallVec<[InstId; 4]>>,
    ) {
        self.splice_range_ix(cur.block_index(), start, end, insts);
    }
}
