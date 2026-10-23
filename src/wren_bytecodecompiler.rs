//! Wren HIR -> register-bytecode compiler.
//!
//! This module lowers the SSA Wren IR from `wren_ir.rs` into the executable
//! register bytecode defined by `wren_bytecode.rs`.
//!
//! The lowering is intentionally optimized for interpreter throughput:
//! - dead pure SSA nodes are removed before allocation;
//! - block parameters are lowered as edge parallel copies, never as phi ops;
//! - a global hole-aware register allocator reuses all 256 GPRs aggressively;
//! - move/guard/block-parameter affinities bias allocation toward coalescing;
//! - no GPR is reserved for compiler scratch state; parallel-copy cycles use
//!   `SpecialReg::Scratch0`;
//! - branches use fallthrough whenever the chosen code layout permits it;
//! - critical-edge copies are emitted in shared out-of-line trampolines;
//! - constants, guard descriptors and IC descriptors are deduplicated;
//! - precise GC maps are emitted from physical liveness at every GC-capable PC;
//! - frame-entry parameter/capture moves are encoded in `FrameLayout`, so no
//!   bytecode prologue is needed.
//!
//! Register spilling is deliberately not implemented here because the current
//! bytecode architecture has no spill-slot opcodes. If more than 256 physical
//! values interfere, compilation fails with `CompileError::RegisterPressure`.
//! In normal Wren code this should be exceptionally rare; adding spill slots
//! later can be done without changing the HIR.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::cmp::Reverse;
use hashbrown::HashMap;
use std::collections::VecDeque;
use std::fmt;

use crate::wren_bytecode::{
    encode_bailout_reason, function_kind_uses_this, opcode_for_f64_bin,
    opcode_for_f64_cmp, opcode_for_f64_unary, opcode_for_int_bin,
    opcode_for_int_cmp, opcode_for_int_unary, ArgReg, Assembler, BytecodeDeoptPoint,
    BytecodeFunction, BytecodeInlinedFrameState, BytecodePc, BytecodeProgram,
    BytecodeSnapshotSlot, BytecodeVerifier, CaptureBinding, ClassGuardId, ConstIndex,
    DeoptLocation, ExceptionHandlerEntry, FrameLayout, IcSlot, IncomingSource, InlineCacheDescriptor, Label,
    LiveRegisterEntry, LiveRegisterMap, Opcode, ParameterBinding, ProfileCounterKind, ReferenceMap, Reg, RegisterKind,
    ResumeEntry, RuntimeMetadata, SafepointKind, SourceMapEntry, SpecialReg,
    TypeGuardId, GENERAL_REGISTER_COUNT, MAX_WREN_ARITY,
};
use crate::wren_ir::{
    verify_function, BlockId, ClassId, ClassSet, Constant, DeoptId, EffectBehavior, FuncId,
    Function, FunctionKind, Inst, InstId, MethodSymbol, Program, ResumeId, SiteId, SnapshotSlot,
    StringId, Type, ValueDef, ValueId,
};

// =============================================================================
// Public API
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeTarget {
    FunctionEntry,
    BlockStart(BlockId),
    BeforeInst(InstId),
    AfterInst(InstId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumeBinding {
    pub function: FuncId,
    pub target: ResumeTarget,
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Verify every HIR function before lowering.
    pub verify_ir: bool,
    /// Verify the complete bytecode program after all cross-function fixups.
    pub verify_bytecode: bool,
    /// Remove pure SSA nodes whose results cannot affect observable execution.
    pub eliminate_dead_code: bool,
    /// Bias physical allocation toward eliminating HIR moves and edge copies.
    pub coalesce_registers: bool,
    /// Order CFG successors using existing block profile counts.
    pub profile_guided_layout: bool,
    /// Omit unconditional jumps when the target block is physically next.
    pub omit_fallthrough_jumps: bool,
    /// Emit source ranges into the bytecode side table.
    pub emit_source_map: bool,
    /// Emit block counter bytecodes. Disabled by default for throughput.
    pub instrument_blocks: bool,
    /// Explicit mapping required for speculative HIR deoptimization snapshots.
    ///
    /// HIR resume IDs intentionally do not encode a bytecode location. Mapping
    /// them here avoids inventing an incorrect resume PC during lowering.
    pub resume_bindings: HashMap<ResumeId, ResumeBinding>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            verify_ir: true,
            verify_bytecode: true,
            eliminate_dead_code: true,
            coalesce_registers: true,
            profile_guided_layout: true,
            omit_fallthrough_jumps: true,
            emit_source_map: true,
            instrument_blocks: false,
            resume_bindings: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCompileStats {
    pub function: FuncId,
    pub hir_nodes: usize,
    pub emitted_hir_nodes: usize,
    pub eliminated_hir_nodes: usize,
    pub allocated_values: usize,
    pub general_registers: u16,
    pub bytecode_words: usize,
    pub constants: usize,
    pub inline_caches: usize,
    pub safepoints: usize,
    pub edge_moves: usize,
    pub edge_trampolines: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CompileStats {
    pub functions: Vec<FunctionCompileStats>,
    pub total_bytecode_words: usize,
    pub max_general_registers: u16,
}

pub struct CompilationOutput {
    pub program: BytecodeProgram,
    pub stats: CompileStats,
}

#[derive(Debug)]
pub enum CompileError {
    IrVerification {
        function: FuncId,
        errors: Vec<String>,
    },
    UnknownFunction(FuncId),
    UnknownBlock {
        function: FuncId,
        block: BlockId,
    },
    UnknownValue {
        function: FuncId,
        value: ValueId,
    },
    InvalidFunctionSignature {
        function: FuncId,
        message: String,
    },
    InvalidEdge {
        function: FuncId,
        from: BlockId,
        to: BlockId,
        message: String,
    },
    RepresentationMismatch {
        function: FuncId,
        from: ValueId,
        to: ValueId,
        from_kind: RegisterKind,
        to_kind: RegisterKind,
    },
    RegisterPressure {
        function: FuncId,
        value: ValueId,
        simultaneously_required: usize,
    },
    MissingRegister {
        function: FuncId,
        value: ValueId,
    },
    MissingResumeBinding {
        function: FuncId,
        resume: ResumeId,
    },
    InvalidResumeBinding {
        function: FuncId,
        resume: ResumeId,
        message: String,
    },
    DeoptValueUnavailable {
        function: FuncId,
        value: ValueId,
    },
    Assemble {
        function: FuncId,
        message: String,
    },
    Metadata(String),
    BytecodeVerification(String),
    Internal {
        function: Option<FuncId>,
        message: String,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IrVerification { function, errors } => {
                write!(f, "HIR verification failed for {function:?}")?;
                for error in errors {
                    write!(f, "\n  - {error}")?;
                }
                Ok(())
            }
            Self::UnknownFunction(function) => write!(f, "unknown function {function:?}"),
            Self::UnknownBlock { function, block } => {
                write!(f, "function {function:?} references unknown block {block:?}")
            }
            Self::UnknownValue { function, value } => {
                write!(f, "function {function:?} references unknown value {value:?}")
            }
            Self::InvalidFunctionSignature { function, message } => {
                write!(f, "invalid signature for {function:?}: {message}")
            }
            Self::InvalidEdge {
                function,
                from,
                to,
                message,
            } => write!(
                f,
                "invalid CFG edge {function:?}:{from:?} -> {to:?}: {message}"
            ),
            Self::RepresentationMismatch {
                function,
                from,
                to,
                from_kind,
                to_kind,
            } => write!(
                f,
                "edge representation mismatch in {function:?}: {from:?} ({from_kind:?}) -> {to:?} ({to_kind:?})"
            ),
            Self::RegisterPressure {
                function,
                value,
                simultaneously_required,
            } => write!(
                f,
                "register pressure in {function:?} while allocating {value:?}: requires at least {simultaneously_required} GPRs but the VM exposes 256"
            ),
            Self::MissingRegister { function, value } => {
                write!(f, "no physical register assigned to {function:?}:{value:?}")
            }
            Self::MissingResumeBinding { function, resume } => write!(
                f,
                "function {function:?} uses deopt resume point {resume:?} without a CompileOptions::resume_bindings entry"
            ),
            Self::InvalidResumeBinding {
                function,
                resume,
                message,
            } => write!(
                f,
                "invalid resume binding for {function:?}/{resume:?}: {message}"
            ),
            Self::DeoptValueUnavailable { function, value } => write!(
                f,
                "deopt snapshot in {function:?} references unmaterialized value {value:?}"
            ),
            Self::Assemble { function, message } => {
                write!(f, "bytecode assembly failed for {function:?}: {message}")
            }
            Self::Metadata(message) => write!(f, "metadata lowering failed: {message}"),
            Self::BytecodeVerification(message) => {
                write!(f, "bytecode verification failed: {message}")
            }
            Self::Internal { function, message } => match function {
                Some(function) => write!(f, "internal compiler error in {function:?}: {message}"),
                None => write!(f, "internal compiler error: {message}"),
            },
        }
    }
}

impl std::error::Error for CompileError {}

pub struct BytecodeCompiler {
    options: CompileOptions,
}

impl Default for BytecodeCompiler {
    fn default() -> Self {
        Self::new(CompileOptions::default())
    }
}

impl BytecodeCompiler {
    pub fn new(options: CompileOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    pub fn compile(&self, hir: &Program) -> Result<BytecodeProgram, CompileError> {
        Ok(self.compile_with_stats(hir)?.program)
    }

    pub fn compile_with_stats(&self, hir: &Program) -> Result<CompilationOutput, CompileError> {
        compile_program_impl(hir, &self.options)
    }
}

pub fn compile_program(hir: &Program) -> Result<BytecodeProgram, CompileError> {
    BytecodeCompiler::default().compile(hir)
}

pub fn compile_program_with_options(
    hir: &Program,
    options: CompileOptions,
) -> Result<BytecodeProgram, CompileError> {
    BytecodeCompiler::new(options).compile(hir)
}

// =============================================================================
// Compact bit set used by dataflow analysis
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
struct BitSet {
    words: Vec<u64>,
    bits: usize,
}

impl BitSet {
    fn new(bits: usize) -> Self {
        Self {
            words: vec![0; (bits + 63) / 64],
            bits,
        }
    }

    #[inline]
    fn contains(&self, bit: usize) -> bool {
        debug_assert!(bit < self.bits);
        (self.words[bit / 64] & (1u64 << (bit % 64))) != 0
    }

    #[inline]
    fn insert(&mut self, bit: usize) -> bool {
        debug_assert!(bit < self.bits);
        let word = &mut self.words[bit / 64];
        let mask = 1u64 << (bit % 64);
        let changed = (*word & mask) == 0;
        *word |= mask;
        changed
    }


    fn union_with(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.words.len(), other.words.len());
        let mut changed = false;
        for (dst, src) in self.words.iter_mut().zip(&other.words) {
            let merged = *dst | *src;
            changed |= merged != *dst;
            *dst = merged;
        }
        changed
    }

    fn subtract(&mut self, other: &Self) {
        debug_assert_eq!(self.words.len(), other.words.len());
        for (dst, src) in self.words.iter_mut().zip(&other.words) {
            *dst &= !*src;
        }
    }


    fn iter_ones(&self) -> BitSetOnes<'_> {
        BitSetOnes {
            set: self,
            word_index: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }
}

struct BitSetOnes<'a> {
    set: &'a BitSet,
    word_index: usize,
    current: u64,
}

impl Iterator for BitSetOnes<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                let value = self.word_index * 64 + bit;
                if value < self.set.bits {
                    return Some(value);
                }
                return None;
            }

            self.word_index += 1;
            self.current = *self.set.words.get(self.word_index)?;
        }
    }
}

// =============================================================================
// CFG, DCE and positions
// =============================================================================

#[derive(Clone, Debug)]
struct IncomingArg {
    predecessor: BlockId,
    value: ValueId,
}

#[derive(Clone, Debug)]
struct CfgInfo {
    successors: Vec<Vec<BlockId>>,
    incoming_args: Vec<Vec<Vec<IncomingArg>>>,
}

fn build_cfg(function: &Function) -> Result<CfgInfo, CompileError> {
    let block_count = function.blocks.len();
    let mut successors = vec![Vec::new(); block_count];
    let mut incoming_args = function
        .blocks
        .iter()
        .map(|block| vec![Vec::new(); block.params.len()])
        .collect::<Vec<_>>();

    for block in &function.blocks {
        let Some(&terminator_id) = block.insts.last() else {
            continue;
        };
        let node = function
            .node(terminator_id)
            .ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("missing terminator node {terminator_id:?}"),
            })?;

        for edge in node.kind.edges() {
            let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
                function: function.id,
                block: edge.target,
            })?;
            if edge.args.len() != target.params.len() {
                return Err(CompileError::InvalidEdge {
                    function: function.id,
                    from: block.id,
                    to: edge.target,
                    message: format!(
                        "edge supplies {} values but target expects {}",
                        edge.args.len(),
                        target.params.len()
                    ),
                });
            }

            successors[block.id.0 as usize].push(edge.target);
            for (index, &value) in edge.args.iter().enumerate() {
                incoming_args[edge.target.0 as usize][index].push(IncomingArg {
                    predecessor: block.id,
                    value,
                });
            }
        }

        successors[block.id.0 as usize].sort_unstable();
        successors[block.id.0 as usize].dedup();
    }

    // Exception regions introduce control-flow edges without ordinary SSA arguments.
    // These edges keep handler state live and make block layout aware of cold handlers.
    for region in &function.exception_regions {
        for &protected in &region.protected_blocks {
            let list = successors
                .get_mut(protected.0 as usize)
                .ok_or(CompileError::UnknownBlock {
                    function: function.id,
                    block: protected,
                })?;
            list.push(region.handler);
        }
    }
    for list in &mut successors {
        list.sort_unstable();
        list.dedup();
    }

    Ok(CfgInfo {
        successors,
        incoming_args,
    })
}

#[derive(Clone, Debug)]
struct DeadCodeInfo {
    live_nodes: Vec<bool>,
    materialized_values: BitSet,
}

fn mark_non_edge_operands(kind: &Inst) -> Vec<ValueId> {
    match kind {
        Inst::Jump { .. } => Vec::new(),
        Inst::Branch { cond, .. } => vec![*cond],
        Inst::Switch { tag, .. } => vec![*tag],
        _ => kind.operands().into_vec(),
    }
}

fn analyze_dead_code(
    function: &Function,
    cfg: &CfgInfo,
    eliminate: bool,
) -> Result<DeadCodeInfo, CompileError> {
    let value_count = function.values.len();
    let node_count = function.nodes.len();

    if !eliminate {
        let mut needed_values = BitSet::new(value_count);
        for value in 0..value_count {
            needed_values.insert(value);
        }
        return Ok(DeadCodeInfo {
            live_nodes: vec![true; node_count],
            materialized_values: needed_values,
        });
    }

    let mut live_nodes = vec![false; node_count];
    let mut needed_values = BitSet::new(value_count);
    let mut node_queue = VecDeque::new();
    let mut value_queue = VecDeque::new();

    let mark_node = |id: InstId,
                     live_nodes: &mut [bool],
                     queue: &mut VecDeque<InstId>| {
        let index = id.0 as usize;
        if !live_nodes[index] {
            live_nodes[index] = true;
            queue.push_back(id);
        }
    };

    let mark_value = |id: ValueId,
                      needed_values: &mut BitSet,
                      queue: &mut VecDeque<ValueId>| {
        let index = id.0 as usize;
        if needed_values.insert(index) {
            queue.push_back(id);
        }
    };

    for node in &function.nodes {
        if node.kind.is_terminator() || node.kind.effect_behavior() != EffectBehavior::None {
            mark_node(node.id, &mut live_nodes, &mut node_queue);
        }
    }

    for point in &function.deopts {
        for slot in &point.snapshot.slots {
            mark_value(slot.value, &mut needed_values, &mut value_queue);
        }
        for frame in &point.snapshot.inlined_frames {
            for slot in &frame.slots {
                mark_value(slot.value, &mut needed_values, &mut value_queue);
            }
        }
    }

    // Handler entry values are defined by the VM rather than a normal CFG edge.
    // Keep them allocated even when the source catch binding is unused.
    for region in &function.exception_regions {
        mark_value(region.error, &mut needed_values, &mut value_queue);
    }

    while !node_queue.is_empty() || !value_queue.is_empty() {
        while let Some(id) = node_queue.pop_front() {
            let node = function.node(id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("DCE worklist references missing node {id:?}"),
            })?;
            for value in mark_non_edge_operands(&node.kind) {
                mark_value(value, &mut needed_values, &mut value_queue);
            }
        }

        while let Some(id) = value_queue.pop_front() {
            let value = function.value(id).ok_or(CompileError::UnknownValue {
                function: function.id,
                value: id,
            })?;
            match value.def {
                ValueDef::InstResult(inst, _) => {
                    mark_node(inst, &mut live_nodes, &mut node_queue);
                }
                ValueDef::BlockParam(block, index) => {
                    let incoming = cfg
                        .incoming_args
                        .get(block.0 as usize)
                        .and_then(|params| params.get(index as usize))
                        .ok_or_else(|| CompileError::Internal {
                            function: Some(function.id),
                            message: format!(
                                "missing incoming-argument table for {block:?} parameter {index}"
                            ),
                        })?;
                    for arg in incoming {
                        mark_value(arg.value, &mut needed_values, &mut value_queue);
                    }
                }
                ValueDef::CaptureParam(_) => {}
            }
        }
    }

    let mut materialized_values = needed_values.clone();
    for node in &function.nodes {
        if live_nodes[node.id.0 as usize] {
            for result in &node.results {
                materialized_values.insert(result.0 as usize);
            }
        }
    }

    Ok(DeadCodeInfo {
        live_nodes,
        materialized_values,
    })
}

fn block_hotness(function: &Function, block: BlockId) -> u64 {
    function
        .profiles
        .block_counters
        .get(&block)
        .copied()
        .unwrap_or(0)
}

fn component_rpo(
    seed: BlockId,
    _cfg: &CfgInfo,
    function: &Function,
    profile_guided: bool,
    visited: &mut [bool],
) -> Vec<BlockId> {
    let mut postorder = Vec::new();
    let mut stack = vec![(seed, false)];

    while let Some((block, expanded)) = stack.pop() {
        let index = block.0 as usize;
        if expanded {
            postorder.push(block);
            continue;
        }
        if visited[index] {
            continue;
        }
        visited[index] = true;
        stack.push((block, true));

        // Layout follows only ordinary CFG edges. Exceptional successors are
        // required by liveness/register allocation, but treating cold handlers
        // as layout successors would pull catch/defer cleanup into the hot code
        // stream and hurt I-cache locality despite zero-cost exception tables.
        let mut successors = function
            .block(block)
            .and_then(|block| block.insts.last())
            .and_then(|id| function.node(*id))
            .map(|node| node.kind.edges().into_iter().map(|edge| edge.target).collect::<Vec<_>>())
            .unwrap_or_default();
        successors.sort_unstable();
        successors.dedup();
        if profile_guided {
            successors.sort_by_key(|successor| Reverse(block_hotness(function, *successor)));
        } else {
            successors.sort_by_key(|successor| Reverse(successor.0));
        }

        // The stack is LIFO. Pushing hot successors first means cold successors
        // are visited first, which places hot successors earlier after reversing
        // postorder.
        for successor in successors {
            if !visited[successor.0 as usize] {
                stack.push((successor, false));
            }
        }
    }

    postorder.reverse();
    postorder
}

fn layout_blocks(function: &Function, cfg: &CfgInfo, profile_guided: bool) -> Vec<BlockId> {
    let mut visited = vec![false; function.blocks.len()];
    let mut layout = component_rpo(
        function.entry,
        cfg,
        function,
        profile_guided,
        &mut visited,
    );

    for block in &function.blocks {
        if !visited[block.id.0 as usize] {
            layout.extend(component_rpo(
                block.id,
                cfg,
                function,
                profile_guided,
                &mut visited,
            ));
        }
    }

    layout
}

#[derive(Clone, Debug)]
struct PositionTable {
    block_start: Vec<u32>,
    block_end: Vec<u32>,
    node_use: Vec<u32>,
    node_def: Vec<u32>,
}

fn assign_positions(function: &Function, layout: &[BlockId]) -> PositionTable {
    let mut block_start = vec![0; function.blocks.len()];
    let mut block_end = vec![0; function.blocks.len()];
    let mut node_use = vec![0; function.nodes.len()];
    let mut node_def = vec![0; function.nodes.len()];
    let mut cursor = 0u32;

    for &block_id in layout {
        let block = &function.blocks[block_id.0 as usize];
        block_start[block_id.0 as usize] = cursor;
        cursor = cursor.saturating_add(2);

        for &inst in &block.insts {
            node_use[inst.0 as usize] = cursor;
            node_def[inst.0 as usize] = cursor.saturating_add(1);
            cursor = cursor.saturating_add(2);
        }

        block_end[block_id.0 as usize] = cursor;
        cursor = cursor.saturating_add(2);
    }

    PositionTable {
        block_start,
        block_end,
        node_use,
        node_def,
    }
}

// =============================================================================
// SSA liveness and physical register allocation
// =============================================================================

fn filtered_edge_operands(
    function: &Function,
    edge: &crate::wren_ir::Edge,
    materialized: &BitSet,
) -> Result<Vec<ValueId>, CompileError> {
    let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
        function: function.id,
        block: edge.target,
    })?;
    let mut out = Vec::new();
    for (index, &arg) in edge.args.iter().enumerate() {
        let param = target.params[index];
        if materialized.contains(param.0 as usize) {
            out.push(arg);
        }
    }
    Ok(out)
}


fn node_deopt_id(kind: &Inst) -> Option<DeoptId> {
    match kind {
        Inst::GuardType { deopt, .. }
        | Inst::GuardClass { deopt, .. }
        | Inst::GuardBounds { deopt, .. }
        | Inst::GuardNotNull { deopt, .. }
        | Inst::GuardInt32 { deopt, .. }
        | Inst::GuardF64 { deopt, .. }
        | Inst::Bailout { deopt, .. } => Some(*deopt),
        _ => None,
    }
}

fn append_deopt_uses(function: &Function, node: &crate::wren_ir::Node, out: &mut Vec<ValueId>) {
    let Some(deopt) = node_deopt_id(&node.kind) else {
        return;
    };
    let Some(point) = function.deopts.iter().find(|point| point.id == deopt) else {
        return;
    };
    out.extend(point.snapshot.slots.iter().map(|slot| slot.value));
    for frame in &point.snapshot.inlined_frames {
        out.extend(frame.slots.iter().map(|slot| slot.value));
    }
}

fn lowering_operands(
    function: &Function,
    node: &crate::wren_ir::Node,
    materialized: &BitSet,
) -> Result<Vec<ValueId>, CompileError> {
    let mut out = Vec::new();
    match &node.kind {
        Inst::Jump { edge } => {
            out.extend(filtered_edge_operands(function, edge, materialized)?);
        }
        Inst::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            out.push(*cond);
            out.extend(filtered_edge_operands(function, then_edge, materialized)?);
            out.extend(filtered_edge_operands(function, else_edge, materialized)?);
        }
        Inst::Switch {
            tag,
            default_edge,
            cases,
        } => {
            out.push(*tag);
            out.extend(filtered_edge_operands(function, default_edge, materialized)?);
            for case in cases {
                out.extend(filtered_edge_operands(function, &case.edge, materialized)?);
            }
        }
        _ => out.extend(node.kind.operands()),
    }
    append_deopt_uses(function, node, &mut out);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[derive(Clone, Debug)]
struct LivenessInfo {
    live_out: Vec<BitSet>,
}

fn compute_liveness(
    function: &Function,
    cfg: &CfgInfo,
    dce: &DeadCodeInfo,
    layout: &[BlockId],
) -> Result<LivenessInfo, CompileError> {
    let value_count = function.values.len();
    let block_count = function.blocks.len();
    let mut uses = (0..block_count)
        .map(|_| BitSet::new(value_count))
        .collect::<Vec<_>>();
    let mut defs = (0..block_count)
        .map(|_| BitSet::new(value_count))
        .collect::<Vec<_>>();

    for block in &function.blocks {
        let block_index = block.id.0 as usize;
        for &param in &block.params {
            if dce.materialized_values.contains(param.0 as usize) {
                defs[block_index].insert(param.0 as usize);
            }
        }

        if block.id == function.entry {
            for capture in &function.captures {
                if dce.materialized_values.contains(capture.value.0 as usize) {
                    defs[block_index].insert(capture.value.0 as usize);
                }
            }
        }

        for &inst_id in &block.insts {
            if !dce.live_nodes[inst_id.0 as usize] {
                continue;
            }
            let node = function.node(inst_id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("liveness references missing node {inst_id:?}"),
            })?;
            for operand in lowering_operands(function, node, &dce.materialized_values)? {
                let index = operand.0 as usize;
                if !defs[block_index].contains(index) {
                    uses[block_index].insert(index);
                }
            }
            for &result in &node.results {
                if dce.materialized_values.contains(result.0 as usize) {
                    defs[block_index].insert(result.0 as usize);
                }
            }
        }
    }

    let mut live_in = (0..block_count)
        .map(|_| BitSet::new(value_count))
        .collect::<Vec<_>>();
    let mut live_out = live_in.clone();

    loop {
        let mut changed = false;
        for &block in layout.iter().rev() {
            let index = block.0 as usize;
            let mut new_out = BitSet::new(value_count);
            for successor in &cfg.successors[index] {
                new_out.union_with(&live_in[successor.0 as usize]);
            }

            let mut new_in = new_out.clone();
            new_in.subtract(&defs[index]);
            new_in.union_with(&uses[index]);

            if new_out != live_out[index] {
                live_out[index] = new_out;
                changed = true;
            }
            if new_in != live_in[index] {
                live_in[index] = new_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    Ok(LivenessInfo { live_out })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveRange {
    start: u32,
    end: u32,
}

impl LiveRange {
    #[inline]
    fn intersects(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    #[inline]
    fn contains(self, position: u32) -> bool {
        self.start <= position && position <= self.end
    }
}

#[derive(Clone, Debug)]
struct LiveInterval {
    value: ValueId,
    ranges: Vec<LiveRange>,
    start: u32,
    end: u32,
    use_weight: u64,
}

impl LiveInterval {
    fn new(value: ValueId) -> Self {
        Self {
            value,
            ranges: Vec::new(),
            start: u32::MAX,
            end: 0,
            use_weight: 0,
        }
    }

    fn add_range(&mut self, range: LiveRange) {
        self.start = self.start.min(range.start);
        self.end = self.end.max(range.end);
        self.ranges.push(range);
    }

    fn normalize(&mut self) {
        if self.ranges.is_empty() {
            self.start = u32::MAX;
            self.end = 0;
            return;
        }
        self.ranges.sort_by_key(|range| range.start);
        let mut merged: Vec<LiveRange> = Vec::with_capacity(self.ranges.len());
        for range in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if range.start <= last.end.saturating_add(1) {
                    last.end = last.end.max(range.end);
                    continue;
                }
            }
            merged.push(range);
        }
        self.start = merged.first().unwrap().start;
        self.end = merged.last().unwrap().end;
        self.ranges = merged;
    }

    fn contains(&self, position: u32) -> bool {
        self.ranges.iter().any(|range| range.contains(position))
    }

    fn intersects(&self, other: &Self) -> bool {
        if self.ranges.is_empty() || other.ranges.is_empty() {
            return false;
        }
        if self.end < other.start || other.end < self.start {
            return false;
        }

        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.ranges.len() && right < other.ranges.len() {
            let a = self.ranges[left];
            let b = other.ranges[right];
            if a.intersects(b) {
                return true;
            }
            if a.end < b.start {
                left += 1;
            } else {
                right += 1;
            }
        }
        false
    }
}

fn build_intervals(
    function: &Function,
    dce: &DeadCodeInfo,
    liveness: &LivenessInfo,
    positions: &PositionTable,
    layout: &[BlockId],
) -> Result<Vec<LiveInterval>, CompileError> {
    let mut intervals = function
        .values
        .iter()
        .map(|value| LiveInterval::new(value.id))
        .collect::<Vec<_>>();

    for &block_id in layout {
        let block = function.block(block_id).ok_or(CompileError::UnknownBlock {
            function: function.id,
            block: block_id,
        })?;
        let block_index = block_id.0 as usize;
        let block_start = positions.block_start[block_index];
        let block_end = positions.block_end[block_index];
        let mut local: HashMap<usize, LiveRange> = HashMap::new();

        for value in liveness.live_out[block_index].iter_ones() {
            if dce.materialized_values.contains(value) {
                local.insert(
                    value,
                    LiveRange {
                        start: block_start,
                        end: block_end,
                    },
                );
            }
        }

        for &inst_id in block.insts.iter().rev() {
            if !dce.live_nodes[inst_id.0 as usize] {
                continue;
            }
            let node = function.node(inst_id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("interval construction references missing node {inst_id:?}"),
            })?;
            let use_position = positions.node_use[inst_id.0 as usize];
            let def_position = positions.node_def[inst_id.0 as usize];

            for &result in &node.results {
                let index = result.0 as usize;
                if !dce.materialized_values.contains(index) {
                    continue;
                }
                local
                    .entry(index)
                    .and_modify(|range| range.start = def_position)
                    .or_insert(LiveRange {
                        start: def_position,
                        end: def_position,
                    });
            }

            for operand in lowering_operands(function, node, &dce.materialized_values)? {
                let index = operand.0 as usize;
                if !dce.materialized_values.contains(index) {
                    continue;
                }
                local
                    .entry(index)
                    .and_modify(|range| range.end = range.end.max(use_position))
                    .or_insert(LiveRange {
                        start: block_start,
                        end: use_position,
                    });
                intervals[index].use_weight = intervals[index]
                    .use_weight
                    .saturating_add(block_hotness(function, block_id).max(1));
            }
        }

        for &param in &block.params {
            let index = param.0 as usize;
            if dce.materialized_values.contains(index) {
                local
                    .entry(index)
                    .and_modify(|range| range.start = block_start)
                    .or_insert(LiveRange {
                        start: block_start,
                        end: block_start,
                    });
            }
        }

        if block_id == function.entry {
            for capture in &function.captures {
                let index = capture.value.0 as usize;
                if dce.materialized_values.contains(index) {
                    local
                        .entry(index)
                        .and_modify(|range| range.start = block_start)
                        .or_insert(LiveRange {
                            start: block_start,
                            end: block_start,
                        });
                }
            }
        }

        for (value, range) in local {
            intervals[value].add_range(range);
        }
    }

    for interval in &mut intervals {
        interval.normalize();
    }
    Ok(intervals)
}

#[derive(Clone, Debug, Default)]
struct AffinityGraph {
    edges: Vec<Vec<(ValueId, u64)>>,
}

impl AffinityGraph {
    fn new(value_count: usize) -> Self {
        Self {
            edges: vec![Vec::new(); value_count],
        }
    }

    fn add(&mut self, left: ValueId, right: ValueId, weight: u64) {
        if left == right {
            return;
        }
        self.edges[left.0 as usize].push((right, weight));
        self.edges[right.0 as usize].push((left, weight));
    }
}

fn result_affinity_source(node: &crate::wren_ir::Node) -> Option<ValueId> {
    match &node.kind {
        Inst::Move(value)
        | Inst::GuardType { value, .. }
        | Inst::GuardClass { value, .. }
        | Inst::GuardNotNull { value, .. }
        | Inst::GuardInt32 { value, .. }
        | Inst::GuardF64 { value, .. }
        | Inst::NumToInt32 { value }
        | Inst::NumToF64 { value }
        | Inst::Int32ToNum { value }
        | Inst::F64ToNum { value }
        | Inst::ReadBarrier { object: value } => Some(*value),
        _ => None,
    }
}

fn build_affinities(
    function: &Function,
    cfg: &CfgInfo,
    dce: &DeadCodeInfo,
    enabled: bool,
) -> AffinityGraph {
    let mut graph = AffinityGraph::new(function.values.len());
    if !enabled {
        return graph;
    }

    const COPY_WEIGHT: u64 = 1_000_000_000;
    const EDGE_WEIGHT_BASE: u64 = 1_000;

    for node in &function.nodes {
        if !dce.live_nodes[node.id.0 as usize] || node.results.len() != 1 {
            continue;
        }
        if let Some(source) = result_affinity_source(node) {
            let result = node.results[0];
            if dce.materialized_values.contains(result.0 as usize)
                && dce.materialized_values.contains(source.0 as usize)
            {
                graph.add(result, source, COPY_WEIGHT);
            }
        }
    }

    for block in &function.blocks {
        for (param_index, &param) in block.params.iter().enumerate() {
            if !dce.materialized_values.contains(param.0 as usize) {
                continue;
            }
            for incoming in &cfg.incoming_args[block.id.0 as usize][param_index] {
                if dce.materialized_values.contains(incoming.value.0 as usize) {
                    let weight = EDGE_WEIGHT_BASE.saturating_add(
                        block_hotness(function, incoming.predecessor).min(u64::MAX - EDGE_WEIGHT_BASE),
                    );
                    graph.add(param, incoming.value, weight);
                }
            }
        }
    }

    graph
}

#[derive(Clone, Debug)]
struct RegisterAllocation {
    value_regs: Vec<Option<Reg>>,
    intervals: Vec<LiveInterval>,
    frame_register_count: u16,
}

fn register_kind(ty: &Type) -> RegisterKind {
    match ty {
        Type::Int32 => RegisterKind::Int32,
        Type::F64 => RegisterKind::F64,
        Type::RawPtr => RegisterKind::RawPtr,
        _ => RegisterKind::Value,
    }
}

fn type_may_reference(ty: &Type) -> bool {
    match ty {
        Type::Top => true,
        Type::String
        | Type::List(_)
        | Type::Map
        | Type::Range
        | Type::Class(_)
        | Type::Instance(_)
        | Type::Closure(_)
        | Type::Fiber
        | Type::Foreign(_)
        | Type::Module
        | Type::Cell(_) => true,
        Type::Union(types) => types.iter().any(type_may_reference),
        Type::Bottom
        | Type::Null
        | Type::Bool
        | Type::Num
        | Type::Int32
        | Type::F64
        | Type::RawPtr => false,
    }
}

fn allocate_registers(
    function: &Function,
    intervals: Vec<LiveInterval>,
    affinities: &AffinityGraph,
) -> Result<RegisterAllocation, CompileError> {
    let mut order = intervals
        .iter()
        .enumerate()
        .filter(|(_, interval)| !interval.ranges.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    order.sort_by_key(|&index| {
        let interval = &intervals[index];
        (interval.start, Reverse(interval.use_weight), interval.end)
    });

    let mut value_regs: Vec<Option<Reg>> = vec![None; function.values.len()];
    let mut reg_intervals = (0..GENERAL_REGISTER_COUNT)
        .map(|_| Vec::<usize>::new())
        .collect::<Vec<_>>();
    let mut max_reg: Option<usize> = None;

    let feasible = |reg: usize, interval_index: usize, reg_intervals: &Vec<Vec<usize>>| {
        reg_intervals[reg]
            .iter()
            .all(|&other| !intervals[interval_index].intersects(&intervals[other]))
    };

    for interval_index in order {
        let interval = &intervals[interval_index];
        let value = interval.value;

        let mut preferred_scores: HashMap<usize, u64> = HashMap::new();
        for &(other, weight) in &affinities.edges[value.0 as usize] {
            if let Some(reg) = value_regs[other.0 as usize] {
                let score = preferred_scores.entry(reg.index()).or_default();
                *score = (*score).saturating_add(weight);
            }
        }

        let mut preferred = preferred_scores.into_iter().collect::<Vec<_>>();
        preferred.sort_by_key(|&(reg, score)| (Reverse(score), reg));

        let mut selected = None;
        for (reg, _) in preferred {
            if feasible(reg, interval_index, &reg_intervals) {
                selected = Some(reg);
                break;
            }
        }

        if selected.is_none() {
            selected = (0..GENERAL_REGISTER_COUNT)
                .find(|&reg| feasible(reg, interval_index, &reg_intervals));
        }

        let Some(reg_index) = selected else {
            let pressure = reg_intervals
                .iter()
                .filter(|occupants| {
                    occupants
                        .iter()
                        .any(|&other| interval.intersects(&intervals[other]))
                })
                .count()
                .saturating_add(1);
            return Err(CompileError::RegisterPressure {
                function: function.id,
                value,
                simultaneously_required: pressure,
            });
        };

        let reg = Reg(reg_index as u8);
        value_regs[value.0 as usize] = Some(reg);
        reg_intervals[reg_index].push(interval_index);
        max_reg = Some(max_reg.map_or(reg_index, |current| current.max(reg_index)));
    }

    let frame_register_count = max_reg.map_or(0, |reg| (reg + 1) as u16);
    Ok(RegisterAllocation {
        value_regs,
        intervals,
        frame_register_count,
    })
}

fn value_reg(
    function: &Function,
    allocation: &RegisterAllocation,
    value: ValueId,
) -> Result<Reg, CompileError> {
    allocation
        .value_regs
        .get(value.0 as usize)
        .and_then(|value| *value)
        .ok_or(CompileError::MissingRegister {
            function: function.id,
            value,
        })
}

fn build_frame_layout(
    function: &Function,
    allocation: &RegisterAllocation,
    dce: &DeadCodeInfo,
) -> Result<(FrameLayout, u8), CompileError> {
    let entry = function.block(function.entry).ok_or(CompileError::UnknownBlock {
        function: function.id,
        block: function.entry,
    })?;
    if entry.params.len() != function.signature.params.len() {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: format!(
                "entry block has {} parameters but signature has {}",
                entry.params.len(),
                function.signature.params.len()
            ),
        });
    }

    let uses_this = function_kind_uses_this(function.kind);
    if uses_this && function.signature.params.is_empty() {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: "method-like function has no receiver parameter".to_owned(),
        });
    }
    let explicit_arity = function
        .signature
        .params
        .len()
        .saturating_sub(if uses_this { 1 } else { 0 });
    if explicit_arity > MAX_WREN_ARITY {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: format!(
                "{} explicit parameters exceed the VM maximum of {}",
                explicit_arity, MAX_WREN_ARITY
            ),
        });
    }

    let mut parameters = Vec::new();
    for (index, (&value, ty)) in entry
        .params
        .iter()
        .zip(&function.signature.params)
        .enumerate()
    {
        if !dce.materialized_values.contains(value.0 as usize) {
            continue;
        }
        let destination = value_reg(function, allocation, value)?;
        let source = if uses_this && index == 0 {
            IncomingSource::This
        } else {
            let argument_index = index - if uses_this { 1 } else { 0 };
            let argument = ArgReg::new(argument_index as u8).ok_or_else(|| {
                CompileError::InvalidFunctionSignature {
                    function: function.id,
                    message: format!("invalid argument index {argument_index}"),
                }
            })?;
            IncomingSource::Argument(argument)
        };
        parameters.push(ParameterBinding {
            source,
            destination,
            kind: register_kind(ty),
        });
    }

    let mut captures = Vec::new();
    for (index, capture) in function.captures.iter().enumerate() {
        if !dce.materialized_values.contains(capture.value.0 as usize) {
            continue;
        }
        captures.push(CaptureBinding {
            capture_index: index as u16,
            destination: value_reg(function, allocation, capture.value)?,
            kind: register_kind(&capture.ty),
        });
    }

    let frame = FrameLayout {
        general_register_count: allocation.frame_register_count,
        parameters,
        captures,
        return_kind: register_kind(&function.signature.result),
    };
    frame
        .validate()
        .map_err(|message| CompileError::InvalidFunctionSignature {
            function: function.id,
            message,
        })?;
    Ok((frame, explicit_arity as u8))
}

// =============================================================================
// Bytecode emission helpers
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ConstantKey {
    Null,
    Bool(bool),
    Num(u64),
    String(u32),
    Int32(i32),
}

impl ConstantKey {
    fn from_constant(constant: &Constant) -> Self {
        match constant {
            Constant::Null => Self::Null,
            Constant::Bool(value) => Self::Bool(*value),
            Constant::Num(value) => Self::Num(value.to_bits()),
            Constant::String(value) => Self::String(value.0),
            Constant::Int32(value) => Self::Int32(*value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EdgeTransferKey {
    target: BlockId,
    moves: Vec<(u8, u8)>,
}

#[derive(Clone, Debug)]
struct DeferredTrampoline {
    label: Label,
    target: BlockId,
    moves: Vec<(Reg, Reg)>,
}

#[derive(Clone, Debug)]
struct PcMap {
    block_start: Vec<Option<BytecodePc>>,
    block_end: Vec<Option<BytecodePc>>,
    before_inst: Vec<Option<BytecodePc>>,
    after_inst: Vec<Option<BytecodePc>>,
}

#[derive(Clone, Debug)]
struct FunctionCompileInfo {
    allocation: RegisterAllocation,
    pc_map: PcMap,
}

#[derive(Clone, Copy, Debug)]
enum CopySource {
    General(Reg),
    Scratch,
}

struct FunctionEmitter<'a> {
    program: &'a Program,
    function: &'a Function,
    options: &'a CompileOptions,
    dce: &'a DeadCodeInfo,
    allocation: &'a RegisterAllocation,
    positions: &'a PositionTable,
    layout: &'a [BlockId],
    assembler: Assembler,
    block_labels: Vec<Label>,
    pc_map: PcMap,
    constant_cache: HashMap<ConstantKey, ConstIndex>,
    type_guard_cache: HashMap<Type, TypeGuardId>,
    class_guard_cache: HashMap<ClassSet, ClassGuardId>,
    ic_cache: HashMap<(SiteId, MethodSymbol, Option<ClassId>), IcSlot>,
    trampoline_cache: HashMap<EdgeTransferKey, Label>,
    trampolines: Vec<DeferredTrampoline>,
    stats: FunctionCompileStats,
}

impl<'a> FunctionEmitter<'a> {
    fn new(
        program: &'a Program,
        function: &'a Function,
        options: &'a CompileOptions,
        dce: &'a DeadCodeInfo,
        allocation: &'a RegisterAllocation,
        positions: &'a PositionTable,
        layout: &'a [BlockId],
    ) -> Self {
        let mut assembler = Assembler::new();
        let block_labels = function
            .blocks
            .iter()
            .map(|_| assembler.label())
            .collect::<Vec<_>>();
        Self {
            program,
            function,
            options,
            dce,
            allocation,
            positions,
            layout,
            assembler,
            block_labels,
            pc_map: PcMap {
                block_start: vec![None; function.blocks.len()],
                block_end: vec![None; function.blocks.len()],
                before_inst: vec![None; function.nodes.len()],
                after_inst: vec![None; function.nodes.len()],
            },
            constant_cache: HashMap::new(),
            type_guard_cache: HashMap::new(),
            class_guard_cache: HashMap::new(),
            ic_cache: HashMap::new(),
            trampoline_cache: HashMap::new(),
            trampolines: Vec::new(),
            stats: FunctionCompileStats {
                function: function.id,
                hir_nodes: function.nodes.len(),
                ..FunctionCompileStats::default()
            },
        }
    }

    fn reg(&self, value: ValueId) -> Result<Reg, CompileError> {
        value_reg(self.function, self.allocation, value)
    }

    fn result_reg(&self, node: &crate::wren_ir::Node) -> Result<Reg, CompileError> {
        let value = *node.results.first().ok_or_else(|| CompileError::Internal {
            function: Some(self.function.id),
            message: format!("instruction {:?} has no result", node.id),
        })?;
        self.reg(value)
    }

    fn assemble_error(&self, error: impl fmt::Display) -> CompileError {
        CompileError::Assemble {
            function: self.function.id,
            message: error.to_string(),
        }
    }

    fn constant(&mut self, value: Constant) -> ConstIndex {
        let key = ConstantKey::from_constant(&value);
        if let Some(index) = self.constant_cache.get(&key) {
            return *index;
        }
        let index = self.assembler.chunk.push_constant(value);
        self.constant_cache.insert(key, index);
        index
    }

    fn type_guard(&mut self, ty: &Type) -> TypeGuardId {
        if let Some(id) = self.type_guard_cache.get(ty) {
            return *id;
        }
        let id = self.assembler.chunk.push_type_guard(ty.clone());
        self.type_guard_cache.insert(ty.clone(), id);
        id
    }

    fn class_guard(&mut self, classes: &ClassSet) -> ClassGuardId {
        if let Some(id) = self.class_guard_cache.get(classes) {
            return *id;
        }
        let id = self.assembler.chunk.push_class_guard(classes.clone());
        self.class_guard_cache.insert(classes.clone(), id);
        id
    }

    fn inline_cache(
        &mut self,
        site: SiteId,
        method: MethodSymbol,
        home_class: Option<ClassId>,
    ) -> IcSlot {
        let key = (site, method, home_class);
        if let Some(slot) = self.ic_cache.get(&key) {
            return *slot;
        }
        let slot = self.assembler.chunk.push_inline_cache(InlineCacheDescriptor {
            site,
            method,
            home_class,
        });
        self.ic_cache.insert(key, slot);
        slot
    }

    fn instruction_position(&self, node: InstId) -> u32 {
        self.positions.node_use[node.0 as usize]
    }

    fn live_register_map(&self, position: u32) -> LiveRegisterMap {
        let mut map = LiveRegisterMap::default();
        for interval in &self.allocation.intervals {
            if interval.ranges.is_empty() || !interval.contains(position) {
                continue;
            }
            if let Some(reg) = self.allocation.value_regs[interval.value.0 as usize] {
                map.set_general(reg, true);
            }
        }
        map
    }

    fn live_reference_map(&self, position: u32) -> ReferenceMap {
        let mut map = ReferenceMap::default();
        for interval in &self.allocation.intervals {
            if interval.ranges.is_empty() || !interval.contains(position) {
                continue;
            }
            let value = &self.function.values[interval.value.0 as usize];
            if !type_may_reference(&value.ty) {
                continue;
            }
            if let Some(reg) = self.allocation.value_regs[interval.value.0 as usize] {
                map.set_general(reg, true);
            }
        }

        // These special values are architectural roots of an active Wren frame.
        // Tagged non-reference values are harmless when scanned as tagged roots.
        if function_kind_uses_this(self.function.kind) {
            map.set_special(SpecialReg::This, true);
        }
        if self.function.kind == FunctionKind::Closure || !self.function.captures.is_empty() {
            map.set_special(SpecialReg::Closure, true);
        }
        map.set_special(SpecialReg::Module, true);
        map.set_special(SpecialReg::Fiber, true);
        map.set_special(SpecialReg::CallerFiber, true);
        map.set_special(SpecialReg::Error, true);
        map.set_special(SpecialReg::ResumeValue, true);
        map
    }

    fn mark_transfer_value(
        &self,
        map: &mut ReferenceMap,
        value: ValueId,
        target: TransferTarget,
    ) -> Result<(), CompileError> {
        let ty = &self
            .function
            .value(value)
            .ok_or(CompileError::UnknownValue {
                function: self.function.id,
                value,
            })?
            .ty;
        if !type_may_reference(ty) {
            return Ok(());
        }
        match target {
            TransferTarget::This => map.set_special(SpecialReg::This, true),
            TransferTarget::Closure => map.set_special(SpecialReg::Closure, true),
            TransferTarget::Argument(index) => {
                if let Some(arg) = ArgReg::new(index as u8) {
                    map.set_argument(arg, true);
                }
            }
        }
        Ok(())
    }

    fn call_reference_map(
        &self,
        node: InstId,
        transfer: CallTransfer<'_>,
    ) -> Result<ReferenceMap, CompileError> {
        let mut map = self.live_reference_map(self.instruction_position(node));
        match transfer {
            CallTransfer::Invoke { receiver, args } => {
                self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                for (index, &arg) in args.iter().enumerate() {
                    self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                }
            }
            CallTransfer::Closure { closure, args } => {
                self.mark_transfer_value(&mut map, closure, TransferTarget::Closure)?;
                for (index, &arg) in args.iter().enumerate() {
                    self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                }
            }
            CallTransfer::Direct { callee, args } => {
                let callee = self
                    .program
                    .function(callee)
                    .ok_or(CompileError::UnknownFunction(callee))?;
                if function_kind_uses_this(callee.kind) {
                    if let Some((&receiver, explicit)) = args.split_first() {
                        self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                        for (index, &arg) in explicit.iter().enumerate() {
                            self.mark_transfer_value(
                                &mut map,
                                arg,
                                TransferTarget::Argument(index),
                            )?;
                        }
                    }
                } else {
                    for (index, &arg) in args.iter().enumerate() {
                        self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                    }
                }
            }
            CallTransfer::RuntimeMethod { args } => {
                if let Some((&receiver, explicit)) = args.split_first() {
                    self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                    for (index, &arg) in explicit.iter().enumerate() {
                        self.mark_transfer_value(
                            &mut map,
                            arg,
                            TransferTarget::Argument(index),
                        )?;
                    }
                }
            }
            CallTransfer::None => {}
        }
        Ok(map)
    }

    fn safepoint_kind(opcode: Opcode) -> SafepointKind {
        use Opcode::*;
        match opcode {
            Safepoint => SafepointKind::ExplicitPoll,
            Invoke | InvokeSuper | CallDirect | CallClosure | CallPrimitive => SafepointKind::Call,
            CallForeign => SafepointKind::Foreign,
            ImportModule | ImportVariable => SafepointKind::Module,
            Await | MakeFiber | FiberCall | FiberTry | FiberTransfer | FiberTransferError | FiberYield
            | FiberSuspend => SafepointKind::FiberSwitch,
            BindMethod | BindForeignMethod | ListAppend | ListInsert | MapSet => {
                SafepointKind::RuntimeMutation
            }
            _ => SafepointKind::Allocation,
        }
    }

    fn record_safepoint(
        &mut self,
        node: InstId,
        pc: BytecodePc,
        transfer: CallTransfer<'_>,
    ) -> Result<(), CompileError> {
        let opcode = self.assembler.chunk.code[pc.word_index()]
            .opcode()
            .ok_or_else(|| CompileError::Internal {
                function: Some(self.function.id),
                message: format!("emitted invalid opcode at PC {}", pc.0),
            })?;
        if !opcode.may_gc() {
            return Ok(());
        }
        let references = self.call_reference_map(node, transfer)?;
        self.assembler
            .add_safepoint(pc, Self::safepoint_kind(opcode), references);
        Ok(())
    }

    fn edge_moves(
        &self,
        from: BlockId,
        edge: &crate::wren_ir::Edge,
    ) -> Result<Vec<(Reg, Reg)>, CompileError> {
        let target = self.function.block(edge.target).ok_or(CompileError::UnknownBlock {
            function: self.function.id,
            block: edge.target,
        })?;
        if target.params.len() != edge.args.len() {
            return Err(CompileError::InvalidEdge {
                function: self.function.id,
                from,
                to: edge.target,
                message: "block parameter/argument count mismatch".to_owned(),
            });
        }

        let mut moves = Vec::new();
        for (&source_value, &target_value) in edge.args.iter().zip(&target.params) {
            if !self
                .dce
                .materialized_values
                .contains(target_value.0 as usize)
            {
                continue;
            }
            let source = self.reg(source_value)?;
            let target_reg = self.reg(target_value)?;
            let source_kind = register_kind(
                &self
                    .function
                    .value(source_value)
                    .ok_or(CompileError::UnknownValue {
                        function: self.function.id,
                        value: source_value,
                    })?
                    .ty,
            );
            let target_kind = register_kind(
                &self
                    .function
                    .value(target_value)
                    .ok_or(CompileError::UnknownValue {
                        function: self.function.id,
                        value: target_value,
                    })?
                    .ty,
            );
            if source_kind != target_kind {
                return Err(CompileError::RepresentationMismatch {
                    function: self.function.id,
                    from: source_value,
                    to: target_value,
                    from_kind: source_kind,
                    to_kind: target_kind,
                });
            }
            if source != target_reg {
                moves.push((target_reg, source));
            }
        }
        Ok(moves)
    }

    fn emit_copy_source(&mut self, dst: Reg, source: CopySource) {
        match source {
            CopySource::General(src) => {
                self.assembler.emit_move(dst, src);
            }
            CopySource::Scratch => {
                self.assembler
                    .emit_ab(Opcode::MoveFromSpecial, dst.0, SpecialReg::Scratch0 as u8);
            }
        }
    }

    fn emit_parallel_moves(&mut self, moves: &[(Reg, Reg)]) {
        #[derive(Clone, Copy)]
        struct PendingMove {
            dst: Reg,
            src: CopySource,
        }

        let mut pending = moves
            .iter()
            .copied()
            .filter(|(dst, src)| dst != src)
            .map(|(dst, src)| PendingMove {
                dst,
                src: CopySource::General(src),
            })
            .collect::<Vec<_>>();

        while !pending.is_empty() {
            if pending.len() == 2 {
                if let (CopySource::General(a_src), CopySource::General(b_src)) =
                    (pending[0].src, pending[1].src)
                {
                    if pending[0].dst == b_src && pending[1].dst == a_src {
                        self.assembler
                            .emit_ab(Opcode::Swap, pending[0].dst.0, pending[1].dst.0);
                        self.stats.edge_moves += 1;
                        pending.clear();
                        break;
                    }
                }
            }

            let safe = pending.iter().position(|candidate| {
                !pending.iter().any(|other| match other.src {
                    CopySource::General(src) => src == candidate.dst,
                    CopySource::Scratch => false,
                })
            });

            if let Some(index) = safe {
                let item = pending.swap_remove(index);
                self.emit_copy_source(item.dst, item.src);
                self.stats.edge_moves += 1;
                continue;
            }

            // A cycle remains. Save one old destination value outside the GPR
            // file, then rewrite all reads of it to the scratch special register.
            let cycle_reg = pending[0].dst;
            self.assembler.emit_ab(
                Opcode::MoveToSpecial,
                SpecialReg::Scratch0 as u8,
                cycle_reg.0,
            );
            for item in &mut pending {
                if matches!(item.src, CopySource::General(src) if src == cycle_reg) {
                    item.src = CopySource::Scratch;
                }
            }
        }
    }

    fn deferred_target(
        &mut self,
        from: BlockId,
        edge: &crate::wren_ir::Edge,
    ) -> Result<Label, CompileError> {
        let moves = self.edge_moves(from, edge)?;
        if moves.is_empty() {
            return Ok(self.block_labels[edge.target.0 as usize]);
        }

        let key = EdgeTransferKey {
            target: edge.target,
            moves: moves.iter().map(|(dst, src)| (dst.0, src.0)).collect(),
        };
        if let Some(label) = self.trampoline_cache.get(&key) {
            return Ok(*label);
        }

        let label = self.assembler.label();
        self.trampoline_cache.insert(key, label);
        self.trampolines.push(DeferredTrampoline {
            label,
            target: edge.target,
            moves,
        });
        self.stats.edge_trampolines += 1;
        Ok(label)
    }

    fn emit_inline_edge(
        &mut self,
        from: BlockId,
        edge: &crate::wren_ir::Edge,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        let moves = self.edge_moves(from, edge)?;
        self.emit_parallel_moves(&moves);
        if !(self.options.omit_fallthrough_jumps && next_block == Some(edge.target)) {
            self.assembler
                .emit_jump(self.block_labels[edge.target.0 as usize]);
        }
        Ok(())
    }

    fn emit_branch(
        &mut self,
        block: BlockId,
        cond: ValueId,
        then_edge: &crate::wren_ir::Edge,
        else_edge: &crate::wren_ir::Edge,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        let cond = self.reg(cond)?;

        let fallthrough_then = next_block == Some(then_edge.target);
        let fallthrough_else = next_block == Some(else_edge.target);
        let choose_then = if fallthrough_then != fallthrough_else {
            fallthrough_then
        } else {
            block_hotness(self.function, then_edge.target)
                >= block_hotness(self.function, else_edge.target)
        };

        if choose_then {
            let else_target = self.deferred_target(block, else_edge)?;
            self.assembler.emit_jump_if_false(cond, else_target);
            self.emit_inline_edge(block, then_edge, next_block)?;
        } else {
            let then_target = self.deferred_target(block, then_edge)?;
            self.assembler.emit_jump_if_true(cond, then_target);
            self.emit_inline_edge(block, else_edge, next_block)?;
        }
        Ok(())
    }

    fn emit_switch(
        &mut self,
        block: BlockId,
        tag: ValueId,
        default_edge: &crate::wren_ir::Edge,
        cases: &[crate::wren_ir::SwitchCase],
    ) -> Result<(), CompileError> {
        let default_target = self.deferred_target(block, default_edge)?;
        let mut targets = Vec::with_capacity(cases.len());
        for case in cases {
            targets.push((case.value, self.deferred_target(block, &case.edge)?));
        }
        self.assembler
            .emit_switch(self.reg(tag)?, default_target, targets);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TransferTarget {
    This,
    Closure,
    Argument(usize),
}

#[derive(Clone, Copy)]
enum CallTransfer<'a> {
    None,
    Invoke {
        receiver: ValueId,
        args: &'a [ValueId],
    },
    Closure {
        closure: ValueId,
        args: &'a [ValueId],
    },
    Direct {
        callee: FuncId,
        args: &'a [ValueId],
    },
    RuntimeMethod {
        args: &'a [ValueId],
    },
}

impl FunctionEmitter<'_> {
    fn emit_non_control_node(
        &mut self,
        node: &crate::wren_ir::Node,
    ) -> Result<(), CompileError> {
        use Inst::*;

        let mut safepoint_transfer = CallTransfer::None;
        let pc = match &node.kind {
            Jump { .. } | Branch { .. } | Switch { .. } => {
                return Err(CompileError::Internal {
                    function: Some(self.function.id),
                    message: format!("control instruction {:?} reached non-control emitter", node.id),
                });
            }
            Return(value) => self.assembler.emit_a(Opcode::Return, self.reg(*value)?.0),
            Unreachable => self.assembler.emit0(Opcode::Trap),

            Safepoint => self.assembler.emit0(Opcode::Safepoint),
            OSRPoint => self.assembler.emit0(Opcode::OsrPoint),

            Const(constant) => {
                let dst = self.result_reg(node)?;
                match constant {
                    Constant::Null => self.assembler.emit_a(Opcode::LoadNull, dst.0),
                    Constant::Bool(false) => self.assembler.emit_a(Opcode::LoadFalse, dst.0),
                    Constant::Bool(true) => self.assembler.emit_a(Opcode::LoadTrue, dst.0),
                    Constant::Int32(value) if i8::try_from(*value).is_ok() => self
                        .assembler
                        .emit_ab(Opcode::LoadI8, dst.0, (*value as i8) as u8),
                    Constant::Int32(value) => self.assembler.emit_x32(
                        Opcode::LoadI32,
                        dst.0,
                        0,
                        0,
                        *value as u32,
                    ),
                    other => {
                        let constant = self.constant(other.clone());
                        self.assembler.emit_load_const(dst, constant)
                    }
                }
            }
            Move(value) => {
                let dst = self.result_reg(node)?;
                let src = self.reg(*value)?;
                if dst == src {
                    return Ok(());
                }
                self.assembler.emit_move(dst, src)
            }

            GuardType {
                value,
                expected,
                deopt,
            } => {
                let guard = self.type_guard(expected);
                self.assembler.emit_x32_y32(
                    Opcode::GuardType,
                    self.result_reg(node)?.0,
                    self.reg(*value)?.0,
                    0,
                    guard.0,
                    deopt.0,
                )
            }
            GuardClass {
                value,
                expected,
                deopt,
            } => {
                let guard = self.class_guard(expected);
                self.assembler.emit_x32_y32(
                    Opcode::GuardClass,
                    self.result_reg(node)?.0,
                    self.reg(*value)?.0,
                    0,
                    guard.0,
                    deopt.0,
                )
            }
            GuardBounds {
                index,
                length,
                deopt,
            } => self.assembler.emit_x32(
                Opcode::GuardBounds,
                self.reg(*index)?.0,
                self.reg(*length)?.0,
                0,
                deopt.0,
            ),
            GuardNotNull { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardNotNull,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            GuardInt32 { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardInt32,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            GuardF64 { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardF64,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            Bailout { reason, deopt } => self.assembler.emit_x32(
                Opcode::Bailout,
                encode_bailout_reason(*reason),
                0,
                0,
                deopt.0,
            ),

            Invoke {
                site,
                receiver,
                method,
                args,
            } => {
                let cache = self.inline_cache(*site, *method, None);
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_invoke(self.result_reg(node)?, self.reg(*receiver)?, &arg_regs, cache)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Invoke {
                    receiver: *receiver,
                    args,
                };
                pc
            }
            InvokeSuper {
                site,
                home_class,
                receiver,
                method,
                args,
            } => {
                let cache = self.inline_cache(*site, *method, Some(*home_class));
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_invoke_super(
                        self.result_reg(node)?,
                        self.reg(*receiver)?,
                        &arg_regs,
                        cache,
                    )
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Invoke {
                    receiver: *receiver,
                    args,
                };
                pc
            }
            CallDirect { conv, callee, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_direct(self.result_reg(node)?, *conv, &arg_regs, *callee)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Direct {
                    callee: *callee,
                    args,
                };
                pc
            }
            CallClosure {
                conv,
                closure,
                args,
            } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_closure(
                        self.result_reg(node)?,
                        self.reg(*closure)?,
                        *conv,
                        &arg_regs,
                    )
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Closure {
                    closure: *closure,
                    args,
                };
                pc
            }
            CallPrimitive { primitive, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_primitive(self.result_reg(node)?, &arg_regs, *primitive)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::RuntimeMethod { args };
                pc
            }
            CallForeign { foreign, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_foreign(self.result_reg(node)?, &arg_regs, *foreign)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::RuntimeMethod { args };
                pc
            }

            CreateClass { class, superclass } => {
                let (super_reg, has_superclass) = match superclass {
                    Some(value) => (self.reg(*value)?.0, 1),
                    None => (0, 0),
                };
                self.assembler.emit_x32(
                    Opcode::CreateClass,
                    self.result_reg(node)?.0,
                    super_reg,
                    has_superclass,
                    class.0,
                )
            },
            BindMethod {
                class,
                method,
                function,
                is_static,
            } => self.assembler.emit_x32_y32(
                Opcode::BindMethod,
                self.reg(*class)?.0,
                *is_static as u8,
                0,
                method.0,
                function.0,
            ),
            BindForeignMethod {
                class,
                method,
                foreign,
                is_static,
            } => self.assembler.emit_x32_y32(
                Opcode::BindForeignMethod,
                self.reg(*class)?.0,
                *is_static as u8,
                0,
                method.0,
                foreign.0,
            ),
            AllocInstance { class } => self.assembler.emit_x32(
                Opcode::AllocInstance,
                self.result_reg(node)?.0,
                0,
                0,
                class.0,
            ),
            AllocForeign { class, size } => self.assembler.emit_x32(
                Opcode::AllocForeign,
                self.result_reg(node)?.0,
                self.reg(*size)?.0,
                0,
                class.0,
            ),
            LoadField { object, slot } => self.assembler.emit_x32(
                Opcode::LoadField,
                self.result_reg(node)?.0,
                self.reg(*object)?.0,
                0,
                slot.0 as u32,
            ),
            StoreField {
                object,
                slot,
                value,
            } => self.assembler.emit_x32(
                Opcode::StoreField,
                self.reg(*object)?.0,
                self.reg(*value)?.0,
                0,
                slot.0 as u32,
            ),
            LoadStaticField { class, slot } => self.assembler.emit_x32(
                Opcode::LoadStaticField,
                self.result_reg(node)?.0,
                self.reg(*class)?.0,
                0,
                slot.0 as u32,
            ),
            StoreStaticField { class, slot, value } => self.assembler.emit_x32(
                Opcode::StoreStaticField,
                self.reg(*class)?.0,
                self.reg(*value)?.0,
                0,
                slot.0 as u32,
            ),
            IsInstance { value, class } => self.assembler.emit_abc(
                Opcode::IsInstance,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                self.reg(*class)?.0,
            ),

            ImportModule { module_name } => self.assembler.emit_x32(
                Opcode::ImportModule,
                self.result_reg(node)?.0,
                0,
                0,
                module_name.0,
            ),
            ImportVariable {
                module,
                variable_name,
            } => self.assembler.emit_x32(
                Opcode::ImportVariable,
                self.result_reg(node)?.0,
                self.reg(*module)?.0,
                0,
                variable_name.0,
            ),
            LoadModuleVar { module, slot } => self.assembler.emit_x32_y32(
                Opcode::LoadModuleVar,
                self.result_reg(node)?.0,
                0,
                0,
                module.0,
                slot.0,
            ),
            StoreModuleVar {
                module,
                slot,
                value,
            } => self.assembler.emit_x32_y32(
                Opcode::StoreModuleVar,
                self.reg(*value)?.0,
                0,
                0,
                module.0,
                slot.0,
            ),

            AllocCell { initial } => self.assembler.emit_ab(
                Opcode::AllocCell,
                self.result_reg(node)?.0,
                self.reg(*initial)?.0,
            ),
            LoadCell { cell } => self.assembler.emit_ab(
                Opcode::LoadCell,
                self.result_reg(node)?.0,
                self.reg(*cell)?.0,
            ),
            StoreCell { cell, value } => self.assembler.emit_ab(
                Opcode::StoreCell,
                self.reg(*cell)?.0,
                self.reg(*value)?.0,
            ),
            MakeClosure { function, captures } => {
                let capture_regs = captures
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.assembler
                    .emit_make_closure(self.result_reg(node)?, *function, &capture_regs)
                    .map_err(|error| self.assemble_error(error))?
            }

            AllocList { capacity: None } => {
                self.assembler.emit_a(Opcode::AllocList, self.result_reg(node)?.0)
            }
            AllocList {
                capacity: Some(capacity),
            } => self.assembler.emit_ab(
                Opcode::AllocListCap,
                self.result_reg(node)?.0,
                self.reg(*capacity)?.0,
            ),
            ListLen { list } => self.assembler.emit_ab(
                Opcode::ListLen,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
            ),
            ListGet { list, index } => self.assembler.emit_abc(
                Opcode::ListGet,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
            ),
            ListSet { list, index, value } => self.assembler.emit_abc(
                Opcode::ListSet,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
                self.reg(*value)?.0,
            ),
            ListAppend { list, value } => self.assembler.emit_ab(
                Opcode::ListAppend,
                self.reg(*list)?.0,
                self.reg(*value)?.0,
            ),
            ListInsert { list, index, value } => self.assembler.emit_abc(
                Opcode::ListInsert,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
                self.reg(*value)?.0,
            ),
            ListRemoveAt { list, index } => self.assembler.emit_abc(
                Opcode::ListRemoveAt,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
            ),

            AllocMap => self.assembler.emit_a(Opcode::AllocMap, self.result_reg(node)?.0),
            MapLen { map } => self.assembler.emit_ab(
                Opcode::MapLen,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
            ),
            MapGet { map, key } => self.assembler.emit_abc(
                Opcode::MapGet,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
            ),
            MapSet { map, key, value } => self.assembler.emit_abc(
                Opcode::MapSet,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
                self.reg(*value)?.0,
            ),
            MapRemove { map, key } => self.assembler.emit_abc(
                Opcode::MapRemove,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
            ),

            StringLen { string } => self.assembler.emit_ab(
                Opcode::StringLen,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
            ),
            StringConcat { left, right } => self.assembler.emit_abc(
                Opcode::StringConcat,
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            StringSlice {
                string,
                start,
                count,
            } => self.assembler.emit_x32(
                Opcode::StringSlice,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
                self.reg(*start)?.0,
                self.reg(*count)?.0 as u32,
            ),
            StringCodePointAt { string, index } => self.assembler.emit_abc(
                Opcode::StringCodePointAt,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
                self.reg(*index)?.0,
            ),
            MakeRange {
                from,
                to,
                inclusive,
            } => self.assembler.emit_x32(
                Opcode::MakeRange,
                self.result_reg(node)?.0,
                self.reg(*from)?.0,
                self.reg(*to)?.0,
                if *inclusive { 1 } else { 0 },
            ),

            IntUnary { op, value } => self.assembler.emit_ab(
                opcode_for_int_unary(*op),
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            IntBin { op, left, right } => self.assembler.emit_abc(
                opcode_for_int_bin(*op),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            F64Unary { op, value } => self.assembler.emit_ab(
                opcode_for_f64_unary(*op),
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            F64Bin { op, left, right } => self.assembler.emit_abc(
                opcode_for_f64_bin(*op),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            CmpInt { kind, left, right } => self.assembler.emit_abc(
                opcode_for_int_cmp(*kind),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            CmpF64 { kind, left, right } => self.assembler.emit_abc(
                opcode_for_f64_cmp(*kind),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            RefEq { left, right } => self.assembler.emit_abc(
                Opcode::RefEq,
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            NumToInt32 { value } => self.assembler.emit_ab(
                Opcode::NumToInt32,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            NumToF64 { value } => self.assembler.emit_ab(
                Opcode::NumToF64,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            Int32ToNum { value } => self.assembler.emit_ab(
                Opcode::Int32ToNum,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            F64ToNum { value } => self.assembler.emit_ab(
                Opcode::F64ToNum,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),

            Await { promise } => self.assembler.emit_ab(
                Opcode::Await,
                self.result_reg(node)?.0,
                self.reg(*promise)?.0,
            ),

            MakeFiber { closure } => self.assembler.emit_ab(
                Opcode::MakeFiber,
                self.result_reg(node)?.0,
                self.reg(*closure)?.0,
            ),
            FiberCall { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberCall,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTry { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberTry,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTransfer { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberTransfer,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTransferError { fiber, error } => self.assembler.emit_abc(
                Opcode::FiberTransferError,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*error)?.0,
            ),
            FiberYield { value } => self.assembler.emit_ab(
                Opcode::FiberYield,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            FiberSuspend => self.assembler.emit_a(Opcode::FiberSuspend, self.result_reg(node)?.0),
            FiberAbort { error } => self.assembler.emit_a(Opcode::FiberAbort, self.reg(*error)?.0),

            WriteBarrier { object, value } => self.assembler.emit_ab(
                Opcode::WriteBarrier,
                self.reg(*object)?.0,
                self.reg(*value)?.0,
            ),
            ReadBarrier { object } => self.assembler.emit_ab(
                Opcode::ReadBarrier,
                self.result_reg(node)?.0,
                self.reg(*object)?.0,
            ),
        };

        self.record_safepoint(node.id, pc, safepoint_transfer)?;
        Ok(())
    }

    fn emit_control_node(
        &mut self,
        block: BlockId,
        node: &crate::wren_ir::Node,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        match &node.kind {
            Inst::Jump { edge } => self.emit_inline_edge(block, edge, next_block),
            Inst::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.emit_branch(block, *cond, then_edge, else_edge, next_block),
            Inst::Switch {
                tag,
                default_edge,
                cases,
            } => self.emit_switch(block, *tag, default_edge, cases),
            _ => self.emit_non_control_node(node),
        }
    }

    fn exception_handler_error(&self, block: BlockId) -> Option<ValueId> {
        self.function
            .exception_regions
            .iter()
            .find(|region| region.handler == block)
            .map(|region| region.error)
    }

    fn emit_catch_entry(&mut self, block: BlockId) -> Result<(), CompileError> {
        let Some(error) = self.exception_handler_error(block) else {
            return Ok(());
        };
        let reg = self.reg(error)?;
        let pc = self.assembler.emit_abc(Opcode::Catch, reg.0, 0, 0);
        let references = self.live_reference_map(self.positions.block_start[block.0 as usize]);
        self.assembler
            .add_safepoint(pc, SafepointKind::Allocation, references);
        Ok(())
    }

    fn emit_block_profile_counter(&mut self, block: BlockId) {
        if !self.options.instrument_blocks {
            return;
        }
        let counter = self
            .assembler
            .chunk
            .push_profile_counter(ProfileCounterKind::Block, None);
        self.assembler
            .emit_x32(Opcode::ProfileBlock, 0, 0, 0, counter.0);
        let _ = block;
    }

    fn emit(mut self) -> Result<(BytecodeFunction, FunctionCompileInfo, FunctionCompileStats), CompileError> {
        let layout = self.layout.to_vec();
        for (layout_index, &block_id) in layout.iter().enumerate() {
            let insts = self
                .function
                .block(block_id)
                .ok_or(CompileError::UnknownBlock {
                    function: self.function.id,
                    block: block_id,
                })?
                .insts
                .clone();
            let label = self.block_labels[block_id.0 as usize];
            self.assembler.bind(label);
            self.pc_map.block_start[block_id.0 as usize] = Some(self.assembler.pc());
            self.emit_catch_entry(block_id)?;
            self.emit_block_profile_counter(block_id);
            let next_block = layout.get(layout_index + 1).copied();

            for inst_id in insts {
                let before = self.assembler.pc();
                self.pc_map.before_inst[inst_id.0 as usize] = Some(before);
                let node = self
                    .function
                    .node(inst_id)
                    .cloned()
                    .ok_or_else(|| CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("emission references missing node {inst_id:?}"),
                    })?;

                let live = if self.dce.live_nodes[inst_id.0 as usize] {
                    self.stats.emitted_hir_nodes += 1;
                    let live = self.live_register_map(self.instruction_position(inst_id));
                    self.emit_control_node(block_id, &node, next_block)?;
                    Some(live)
                } else {
                    self.stats.eliminated_hir_nodes += 1;
                    None
                };

                let after = self.assembler.pc();
                if before != after {
                    // Backedges and native exits may rendezvous here without an
                    // allocating opcode. No extra instruction enters the hot loop.
                    if self.assembler.chunk.safepoints.last().is_none_or(|entry| entry.pc != before) {
                        let references = self.live_reference_map(self.instruction_position(inst_id));
                        self.assembler.add_safepoint(
                            before,
                            SafepointKind::ExplicitPoll,
                            references,
                        );
                    }
                    if let Some(live) = live {
                        // Side exits resume at original bytecode instruction boundaries.
                        // Keep all physical GPR liveness separate from GC reference maps.
                        // If a single HIR node emits several bytecodes, the start PC is
                        // still the semantically valid re-entry point for that node.
                        self.assembler.chunk.live_registers.push(LiveRegisterEntry {
                            pc: before,
                            live,
                        });
                    }
                }
                self.pc_map.after_inst[inst_id.0 as usize] = Some(after);
                if self.options.emit_source_map && before != after {
                    if let Some(source) = node.source {
                        self.assembler.chunk.source_map.push(SourceMapEntry {
                            start_pc: before,
                            end_pc: after,
                            source,
                        });
                    }
                }
            }
            self.pc_map.block_end[block_id.0 as usize] = Some(self.assembler.pc());
        }

        // Materialize the zero-cost exception table after every block PC range is known.
        for region in &self.function.exception_regions {
            let handler_pc = self.pc_map.block_start[region.handler.0 as usize].ok_or_else(|| {
                CompileError::Internal {
                    function: Some(self.function.id),
                    message: format!("missing bytecode PC for exception handler {:?}", region.handler),
                }
            })?;
            let error_reg = self.reg(region.error)?;
            for &protected in &region.protected_blocks {
                let start_pc = self.pc_map.block_start[protected.0 as usize].ok_or_else(|| {
                    CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("missing bytecode start PC for protected block {:?}", protected),
                    }
                })?;
                let end_pc = self.pc_map.block_end[protected.0 as usize].ok_or_else(|| {
                    CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("missing bytecode end PC for protected block {:?}", protected),
                    }
                })?;
                if start_pc != end_pc {
                    self.assembler.chunk.exception_handlers.push(ExceptionHandlerEntry {
                        start_pc,
                        end_pc,
                        handler_pc,
                        error_reg,
                        depth: region.depth,
                    });
                }
            }
        }
        self.assembler.chunk.exception_handlers.sort_unstable_by_key(|entry| {
            (entry.start_pc.0, entry.end_pc.0, std::cmp::Reverse(entry.depth))
        });

        for trampoline in self.trampolines.clone() {
            self.assembler.bind(trampoline.label);
            self.emit_parallel_moves(&trampoline.moves);
            self.assembler
                .emit_jump(self.block_labels[trampoline.target.0 as usize]);
        }

        let (frame, arity) = build_frame_layout(self.function, self.allocation, self.dce)?;
        let capture_count = u16::try_from(self.function.captures.len()).map_err(|_| {
            CompileError::InvalidFunctionSignature {
                function: self.function.id,
                message: format!(
                    "{} captures exceed the bytecode u16 capture count",
                    self.function.captures.len()
                ),
            }
        })?;

        let function_id = self.function.id;
        let chunk = self.assembler.finish().map_err(|error| CompileError::Assemble {
            function: function_id,
            message: error.to_string(),
        })?;
        self.stats.allocated_values = self
            .allocation
            .value_regs
            .iter()
            .filter(|reg| reg.is_some())
            .count();
        self.stats.general_registers = self.allocation.frame_register_count;
        self.stats.bytecode_words = chunk.code.len();
        self.stats.constants = chunk.constants.len();
        self.stats.inline_caches = chunk.inline_caches.len();
        self.stats.safepoints = chunk.safepoints.len();

        let bytecode = BytecodeFunction {
            id: self.function.id,
            module: self.function.module,
            kind: self.function.kind,
            is_async: self.function.is_async,
            field_owner: self.function.field_owner,
            name: self.function.signature.name,
            arity,
            capture_count,
            frame,
            chunk,
        };
        let info = FunctionCompileInfo {
            allocation: self.allocation.clone(),
            pc_map: self.pc_map,
        };
        Ok((bytecode, info, self.stats))
    }
}

// =============================================================================
// Function/program compilation and deoptimization fixups
// =============================================================================

fn compile_function(
    program: &Program,
    function: &Function,
    options: &CompileOptions,
) -> Result<(BytecodeFunction, FunctionCompileInfo, FunctionCompileStats), CompileError> {
    if options.verify_ir {
        if let Err(errors) = verify_function(function) {
            return Err(CompileError::IrVerification {
                function: function.id,
                errors: errors.into_iter().map(|error| error.to_string()).collect(),
            });
        }
    }

    let cfg = build_cfg(function)?;
    let dce = analyze_dead_code(function, &cfg, options.eliminate_dead_code)?;
    let layout = layout_blocks(function, &cfg, options.profile_guided_layout);
    let positions = assign_positions(function, &layout);
    let liveness = compute_liveness(function, &cfg, &dce, &layout)?;
    let intervals = build_intervals(function, &dce, &liveness, &positions, &layout)?;
    let affinities = build_affinities(function, &cfg, &dce, options.coalesce_registers);
    let allocation = allocate_registers(function, intervals, &affinities)?;

    // Validate edge representation classes before emission. Doing this after
    // allocation also catches a malformed HIR before parallel-copy generation.
    for block in &function.blocks {
        let Some(&terminator) = block.insts.last() else {
            continue;
        };
        let node = function.node(terminator).ok_or_else(|| CompileError::Internal {
            function: Some(function.id),
            message: format!("missing node {terminator:?}"),
        })?;
        for edge in node.kind.edges() {
            let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
                function: function.id,
                block: edge.target,
            })?;
            for (&from, &to) in edge.args.iter().zip(&target.params) {
                if !dce.materialized_values.contains(to.0 as usize) {
                    continue;
                }
                let from_ty = &function
                    .value(from)
                    .ok_or(CompileError::UnknownValue {
                        function: function.id,
                        value: from,
                    })?
                    .ty;
                let to_ty = &function
                    .value(to)
                    .ok_or(CompileError::UnknownValue {
                        function: function.id,
                        value: to,
                    })?
                    .ty;
                let from_kind = register_kind(from_ty);
                let to_kind = register_kind(to_ty);
                if from_kind != to_kind {
                    return Err(CompileError::RepresentationMismatch {
                        function: function.id,
                        from,
                        to,
                        from_kind,
                        to_kind,
                    });
                }
            }
        }
    }

    FunctionEmitter::new(
        program,
        function,
        options,
        &dce,
        &allocation,
        &positions,
        &layout,
    )
    .emit()
}

fn lower_runtime_metadata(hir: &Program) -> Result<RuntimeMetadata, CompileError> {
    let mut metadata = RuntimeMetadata::default();

    let mut string_index = 0u32;
    while let Some(value) = hir.string(StringId(string_index)) {
        metadata.strings.push(value.to_owned());
        string_index = string_index
            .checked_add(1)
            .ok_or_else(|| CompileError::Metadata("string ID overflow".to_owned()))?;
    }

    let mut method_index = 0u32;
    while let Some(method) = hir.method(MethodSymbol(method_index)) {
        metadata.methods.push(method);
        method_index = method_index
            .checked_add(1)
            .ok_or_else(|| CompileError::Metadata("method ID overflow".to_owned()))?;
    }

    metadata.modules = hir.modules.clone();
    metadata.classes = hir.classes.clone();
    metadata.foreign_functions = hir.foreign_functions.clone();
    metadata
        .rebuild_indexes()
        .map_err(CompileError::Metadata)?;
    Ok(metadata)
}

fn resolve_resume_pc(
    owner: FuncId,
    resume: ResumeId,
    expected_function: FuncId,
    options: &CompileOptions,
    infos: &HashMap<FuncId, FunctionCompileInfo>,
    hir: &Program,
) -> Result<BytecodePc, CompileError> {
    let binding = options
        .resume_bindings
        .get(&resume)
        .copied()
        .ok_or(CompileError::MissingResumeBinding {
            function: owner,
            resume,
        })?;
    if binding.function != expected_function {
        return Err(CompileError::InvalidResumeBinding {
            function: owner,
            resume,
            message: format!(
                "binding targets {:?}, but snapshot expects {:?}",
                binding.function, expected_function
            ),
        });
    }

    let info = infos
        .get(&expected_function)
        .ok_or(CompileError::UnknownFunction(expected_function))?;
    let function = hir
        .function(expected_function)
        .ok_or(CompileError::UnknownFunction(expected_function))?;

    let pc = match binding.target {
        ResumeTarget::FunctionEntry => info.pc_map.block_start[function.entry.0 as usize],
        ResumeTarget::BlockStart(block) => {
            if function.block(block).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target block {block:?}"),
                });
            }
            info.pc_map.block_start[block.0 as usize]
        }
        ResumeTarget::BeforeInst(inst) => {
            if function.node(inst).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target instruction {inst:?}"),
                });
            }
            info.pc_map.before_inst[inst.0 as usize]
        }
        ResumeTarget::AfterInst(inst) => {
            if function.node(inst).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target instruction {inst:?}"),
                });
            }
            info.pc_map.after_inst[inst.0 as usize]
        }
    };

    pc.ok_or_else(|| CompileError::InvalidResumeBinding {
        function: owner,
        resume,
        message: "target has no emitted bytecode PC".to_owned(),
    })
}

fn deopt_location_for_value(
    function: &Function,
    info: &FunctionCompileInfo,
    value: ValueId,
) -> Result<(DeoptLocation, RegisterKind), CompileError> {
    let value_info = function.value(value).ok_or(CompileError::UnknownValue {
        function: function.id,
        value,
    })?;
    let reg = info
        .allocation
        .value_regs
        .get(value.0 as usize)
        .and_then(|reg| *reg)
        .ok_or(CompileError::DeoptValueUnavailable {
            function: function.id,
            value,
        })?;
    Ok((
        DeoptLocation::General(reg),
        register_kind(&value_info.ty),
    ))
}

fn lower_snapshot_slots(
    function: &Function,
    info: &FunctionCompileInfo,
    slots: &[SnapshotSlot],
) -> Result<Vec<BytecodeSnapshotSlot>, CompileError> {
    slots
        .iter()
        .map(|slot| {
            let (location, kind) = deopt_location_for_value(function, info, slot.value)?;
            Ok(BytecodeSnapshotSlot {
                slot: slot.slot,
                location,
                kind,
            })
        })
        .collect()
}

fn finalize_deopts(
    hir: &Program,
    bytecode: &mut BytecodeProgram,
    infos: &HashMap<FuncId, FunctionCompileInfo>,
    options: &CompileOptions,
) -> Result<(), CompileError> {
    let mut function_indices = HashMap::new();
    for (index, function) in bytecode.functions.iter().enumerate() {
        function_indices.insert(function.id, index);
    }

    let mut resume_entries: HashMap<FuncId, HashMap<ResumeId, BytecodePc>> = HashMap::new();

    for function in &hir.functions {
        let info = infos
            .get(&function.id)
            .ok_or(CompileError::UnknownFunction(function.id))?;
        let output_index = *function_indices
            .get(&function.id)
            .ok_or(CompileError::UnknownFunction(function.id))?;
        let mut lowered_points = Vec::with_capacity(function.deopts.len());

        for point in &function.deopts {
            let resume_id = point.snapshot.resume.id;
            let resume_pc = resolve_resume_pc(
                function.id,
                resume_id,
                function.id,
                options,
                infos,
                hir,
            )?;
            resume_entries
                .entry(function.id)
                .or_default()
                .insert(resume_id, resume_pc);

            let slots = lower_snapshot_slots(function, info, &point.snapshot.slots)?;
            let mut inlined_frames = Vec::with_capacity(point.snapshot.inlined_frames.len());
            for frame in &point.snapshot.inlined_frames {
                let resume_pc = resolve_resume_pc(
                    function.id,
                    frame.resume.id,
                    frame.function,
                    options,
                    infos,
                    hir,
                )?;
                resume_entries
                    .entry(frame.function)
                    .or_default()
                    .insert(frame.resume.id, resume_pc);
                inlined_frames.push(BytecodeInlinedFrameState {
                    function: frame.function,
                    resume_pc,
                    slots: lower_snapshot_slots(function, info, &frame.slots)?,
                });
            }

            lowered_points.push(BytecodeDeoptPoint {
                id: point.id,
                resume_pc,
                slots,
                inlined_frames,
            });
        }

        bytecode.functions[output_index].chunk.deopts = lowered_points;
    }

    for (function, entries) in resume_entries {
        let index = *function_indices
            .get(&function)
            .ok_or(CompileError::UnknownFunction(function))?;
        let chunk = &mut bytecode.functions[index].chunk;
        chunk.resume_points = entries
            .into_iter()
            .map(|(id, pc)| ResumeEntry { id, pc })
            .collect();
        chunk.resume_points.sort_by_key(|entry| entry.pc);
    }

    Ok(())
}

fn compile_program_impl(
    hir: &Program,
    options: &CompileOptions,
) -> Result<CompilationOutput, CompileError> {
    let mut bytecode = BytecodeProgram::new();
    bytecode.metadata = lower_runtime_metadata(hir)?;

    let mut infos = HashMap::new();
    let mut stats = CompileStats::default();

    for function in &hir.functions {
        let (compiled, info, function_stats) = compile_function(hir, function, options)?;
        infos.insert(function.id, info);
        stats.total_bytecode_words = stats
            .total_bytecode_words
            .saturating_add(function_stats.bytecode_words);
        stats.max_general_registers = stats
            .max_general_registers
            .max(function_stats.general_registers);
        stats.functions.push(function_stats);
        bytecode
            .push_function(compiled)
            .map_err(CompileError::Metadata)?;
    }

    finalize_deopts(hir, &mut bytecode, &infos, options)?;
    bytecode.rebuild_indexes().map_err(CompileError::Metadata)?;

    if options.verify_bytecode {
        BytecodeVerifier::verify_program(&bytecode)
            .map_err(|error| CompileError::BytecodeVerification(error.to_string()))?;
    }

    Ok(CompilationOutput {
        program: bytecode,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{disassemble, BytecodeVerifier};
    use crate::wren_ir::{Builder, IntBinOp};

    #[test]
    fn compiles_arithmetic_and_reuses_registers() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "main",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Num,
            );
            let one = builder.const_i32(1);
            let two = builder.const_i32(2);
            let sum = builder.emit1(
                Inst::IntBin {
                    op: IntBinOp::Add,
                    left: one,
                    right: two,
                },
                Type::Int32,
            );
            let result = builder.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
            builder.ret(result);
            builder.finish();
        }

        let output = BytecodeCompiler::default()
            .compile_with_stats(&hir)
            .expect("HIR must compile");
        BytecodeVerifier::verify_program(&output.program).expect("bytecode must verify");
        assert_eq!(output.program.functions.len(), 1);
        assert!(output.stats.max_general_registers <= 3);
        let text = disassemble(&output.program.functions[0]).expect("must disassemble");
        assert!(text.contains("IAdd"));
        assert!(text.contains("Int32ToNum"));
    }

    #[test]
    fn lowers_block_parameters_as_parallel_edge_copies() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "select",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Num,
            );
            let then_block = builder.new_block(&[]);
            let else_block = builder.new_block(&[]);
            let join = builder.new_block(&[]);
            let joined = builder.add_block_param(join, Type::Int32);
            let condition = builder.const_bool(true);
            builder.branch(condition, then_block, vec![], else_block, vec![]);

            builder.position_at(then_block);
            let left = builder.const_i32(10);
            builder.jump(join, vec![left]);

            builder.position_at(else_block);
            let right = builder.const_i32(20);
            builder.jump(join, vec![right]);

            builder.position_at(join);
            let result = builder.emit1(Inst::Int32ToNum { value: joined }, Type::Num);
            builder.ret(result);
            builder.finish();
        }

        let output = compile_program(&hir).expect("HIR must compile");
        BytecodeVerifier::verify_program(&output).expect("bytecode must verify");
    }

    #[test]
    fn dce_removes_dead_pure_nodes_but_keeps_effects() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "main",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Null,
            );
            let _dead_a = builder.const_i32(123);
            let _dead_b = builder.const_i32(456);
            let result = builder.const_null();
            builder.ret(result);
            builder.finish();
        }

        let output = BytecodeCompiler::default()
            .compile_with_stats(&hir)
            .expect("HIR must compile");
        assert!(output.stats.functions[0].eliminated_hir_nodes >= 2);
        BytecodeVerifier::verify_program(&output.program).expect("bytecode must verify");
    }
}
