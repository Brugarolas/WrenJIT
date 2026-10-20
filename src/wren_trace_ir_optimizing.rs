//! Trace optimization passes. The default list is shared by root and side traces.
//!
//! A trace has one linear body, side exits, and (for loops) a backedge. PRE
//! therefore performs phi translation across that backedge; it does not invent
//! an arbitrary branch CFG or move effects across runtime synchronization.

use crate::wren_ir::{IntBinOp, IntUnaryOp};
use crate::wren_trace_ir::*;
use crate::wren_trace_ir_optimizer::{
    Optimize, TraceOptimization, alias_value, append_preheader, canonicalize_references,
};
use std::collections::{HashMap, HashSet};

pub fn optimization_passes() -> Optimize {
    passes(true)
}

pub fn conservative_optimization_passes() -> Optimize {
    passes(false)
}

fn passes(promote_loop_representations: bool) -> Optimize {
    Optimize::new()
        .with_pass(TraceCanonicalization {
            promote_loop_representations,
        })
        .with_pass(GlobalValueNumbering)
        .with_pass(LoopInvariantCodeMotion)
        .with_pass(PartialRedundancyElimination)
        .with_pass(GlobalValueNumbering)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Expression {
    kind: TraceInst,
    ty: TraceType,
    rep: TraceRep,
}

fn expression(trace: &Trace, node: usize) -> Option<Expression> {
    let node = &trace.nodes[node];
    if !node.kind.is_speculatable() {
        return None;
    }
    let [result] = node.results.as_slice() else {
        return None;
    };
    let value = trace.value(*result)?;
    // Raw addresses cannot survive a synchronization boundary. These passes
    // intentionally optimize scalar/tagged expressions only.
    if matches!(value.rep, TraceRep::RawPtr | TraceRep::HeapHandle) {
        return None;
    }
    let mut kind = node.kind.clone();
    let mut scalar = true;
    kind.visit_operands(|v| {
        *v = trace.canonical_value(*v);
        scalar &= trace
            .value(*v)
            .is_some_and(|v| !matches!(v.rep, TraceRep::RawPtr | TraceRep::HeapHandle));
    });
    scalar.then_some(Expression {
        kind,
        ty: value.ty,
        rep: value.rep,
    })
}

fn preheader_values(trace: &Trace) -> HashSet<TraceValueId> {
    let mut available: HashSet<_> = trace
        .values
        .iter()
        .filter(|v| {
            matches!(
                v.origin,
                TraceValueOrigin::EntrySlot { .. }
                    | TraceValueOrigin::Constant(_)
                    | TraceValueOrigin::GuardedEntry { .. }
            )
        })
        .map(|v| v.id)
        .collect();
    for &id in &trace.preheader {
        if !matches!(trace.nodes[id].kind, TraceInst::Nop) {
            available.extend(trace.nodes[id].results.iter().copied());
        }
    }
    available
}

fn native_loop(trace: &Trace) -> bool {
    matches!(trace.anchor.kind, TraceKind::Loop)
        && matches!(trace.terminator, TraceTerminator::LoopBack)
        && !trace
            .nodes
            .iter()
            .any(|n| matches!(n.kind, TraceInst::Runtime { .. }))
}

/// Value numbering across the complete linear trace, including its preheader.
/// Constants are interned by their exact bits; float operands remain ordered.
/// Runtime calls clear expression availability; loads and guards are excluded.
#[derive(Default)]
pub struct GlobalValueNumbering;

impl TraceOptimization for GlobalValueNumbering {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let mut changed = false;
        let mut constants = HashMap::new();
        for index in 0..trace.values.len() {
            let value = &trace.values[index];
            let TraceValueOrigin::Constant(constant) = value.origin else {
                continue;
            };
            let key = (constant, value.ty, value.rep);
            if let Some(&existing) = constants.get(&key) {
                let id = value.id;
                alias_value(trace, id, existing);
                changed = true;
            } else {
                constants.insert(key, value.id);
            }
        }
        let preheader: HashSet<_> = trace.preheader.iter().copied().collect();
        let order: Vec<_> = trace
            .preheader
            .iter()
            .copied()
            .chain((0..trace.nodes.len()).filter(|id| !preheader.contains(id)))
            .collect();
        let mut values = HashMap::<Expression, TraceValueId>::new();
        for id in order {
            if matches!(trace.nodes[id].kind, TraceInst::Runtime { .. }) {
                values.clear();
                continue;
            }
            let Some(key) = expression(trace, id) else {
                continue;
            };
            let result = trace.nodes[id].results[0];
            if let Some(&existing) = values.get(&key) {
                alias_value(trace, result, existing);
                trace.nodes[id].kind = TraceInst::Nop;
                trace.stats.gvn_eliminated += 1;
                changed = true;
            } else {
                values.insert(key, result);
            }
        }
        changed
    }
}

/// Hoist total, effect-free scalar computations with invariant operands.
/// Guards, checked arithmetic, memory operations and runtime loops stay put.
#[derive(Default)]
pub struct LoopInvariantCodeMotion;

impl TraceOptimization for LoopInvariantCodeMotion {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        if !native_loop(trace) {
            return false;
        }
        let mut available = preheader_values(trace);
        let preheader: HashSet<_> = trace.preheader.iter().copied().collect();
        let mut changed = false;
        for id in 0..trace.nodes.len() {
            if preheader.contains(&id) {
                continue;
            }
            let Some(mut key) = expression(trace, id) else {
                continue;
            };
            let mut invariant = true;
            key.kind
                .visit_operands(|v| invariant &= available.contains(v));
            if invariant {
                trace.preheader.push(id);
                available.insert(trace.nodes[id].results[0]);
                trace.stats.licm_hoisted += 1;
                changed = true;
            }
        }
        changed
    }
}

/// Loop-edge PRE: for e(phi), reuse an existing e(backedge) on later iterations
/// and compute only the missing e(initial) on the entry edge. A synthetic phi
/// joins the two. No work is added to the loop body or to a side-exit path.
#[derive(Default)]
pub struct PartialRedundancyElimination;

impl TraceOptimization for PartialRedundancyElimination {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        if !native_loop(trace) {
            return false;
        }
        let mut changed = false;
        let preheader: HashSet<_> = trace.preheader.iter().copied().collect();
        let body: Vec<_> = (0..trace.nodes.len())
            .filter(|id| !preheader.contains(id))
            .collect();
        // Bounded to the original nodes. Newly introduced phis may enable a
        // later pipeline pass, but cannot grow this pass's candidate set.
        for &id in &body {
            let Some(key) = expression(trace, id) else {
                continue;
            };
            let available = preheader_values(trace);
            let mut initial = key.clone();
            let mut backedge = key.clone();
            let mut has_phi = false;
            let mut valid = true;
            let mut guarded_inputs = Vec::new();
            initial.kind.visit_operands(|v| {
                if let Some(phi) = trace.loop_phis.iter().find(|phi| phi.value == *v) {
                    has_phi = true;
                    let source = trace.canonical_value(phi.initial);
                    valid &= available.contains(&source);
                    if let Some(exit) = phi.entry_exit {
                        // The existing entry guard already executes before the
                        // loop. Reuse its exit only when it needs no body state.
                        valid &= trace.exits.get(exit.0 as usize).is_some_and(|exit| {
                            exit.frames.iter().all(|frame| frame.writes.is_empty())
                        });
                        if !guarded_inputs
                            .iter()
                            .any(|p: &LoopPhi| p.value == phi.value)
                        {
                            guarded_inputs.push(phi.clone());
                        }
                    } else {
                        *v = source;
                    }
                } else {
                    valid &= available.contains(v);
                }
            });
            if !valid || !has_phi {
                continue;
            }
            backedge.kind.visit_operands(|v| {
                if let Some(phi) = trace.loop_phis.iter().find(|phi| phi.value == *v) {
                    if let Some(next) = phi.backedge {
                        *v = trace.canonical_value(next);
                    } else {
                        valid = false;
                    }
                }
            });
            if !valid {
                continue;
            }
            let Some(redundant) = body
                .iter()
                .copied()
                .find(|&other| other != id && expression(trace, other).as_ref() == Some(&backedge))
            else {
                continue;
            };
            let next = trace.nodes[redundant].results[0];
            let source = trace.nodes[id].source;
            for phi in guarded_inputs {
                let exit = phi.entry_exit.unwrap();
                let value = TraceValueId(trace.values.len() as u32);
                trace.values.push(TraceValue {
                    id: value,
                    ty: trace.value(phi.value).unwrap().ty,
                    rep: phi.rep,
                    origin: TraceValueOrigin::GuardedEntry {
                        value: phi.initial,
                        exit,
                    },
                });
                initial.kind.replace_operand(phi.value, value);
            }
            let start = append_preheader(trace, initial.kind, key.ty, key.rep, source);
            let value = TraceValueId(trace.values.len() as u32);
            trace.values.push(TraceValue {
                id: value,
                ty: key.ty,
                rep: key.rep,
                origin: TraceValueOrigin::LoopTemporary,
            });
            trace.loop_phis.push(LoopPhi {
                value,
                initial: start,
                backedge: Some(next),
                rep: key.rep,
                entry_exit: None,
            });
            let result = trace.nodes[id].results[0];
            alias_value(trace, result, value);
            trace.nodes[id].kind = TraceInst::Nop;
            trace.stats.pre_eliminated += 1;
            changed = true;
        }
        changed
    }
}

/// Existing box/guard cancellation and loop representation specialization.
pub struct TraceCanonicalization {
    pub promote_loop_representations: bool,
}

impl TraceOptimization for TraceCanonicalization {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
        let before = (
            trace.stats.eliminated_guards,
            trace.stats.eliminated_boxes,
            trace.stats.eliminated_copies,
            trace.stats.promoted_f64_loop_phis,
            trace.stats.promoted_i32_loop_phis,
            trace.stats.promoted_u32_loop_phis,
        );
        let has_runtime = trace
            .nodes
            .iter()
            .any(|node| matches!(node.kind, TraceInst::Runtime { .. }));
        self.cancel_box_guard_pairs(trace);
        if !has_runtime {
            self.hoist_invariant_numeric_guards(trace);
            if self.promote_loop_representations {
                self.promote_loop_representations(trace);
            }
            self.cancel_box_guard_pairs(trace);
        }
        self.remove_trivial_copies(trace);
        self.remove_dead_box_nodes(trace);
        canonicalize_references(trace);
        before
            != (
                trace.stats.eliminated_guards,
                trace.stats.eliminated_boxes,
                trace.stats.eliminated_copies,
                trace.stats.promoted_f64_loop_phis,
                trace.stats.promoted_i32_loop_phis,
                trace.stats.promoted_u32_loop_phis,
            )
    }
}

impl TraceCanonicalization {
    fn producer(trace: &Trace, value: TraceValueId) -> Option<(usize, TraceInst)> {
        let value = trace.canonical_value(value);
        let node = match &trace.value(value)?.origin {
            TraceValueOrigin::InstResult { node } => *node,
            _ => return None,
        };
        Some((node, trace.nodes.get(node)?.kind.clone()))
    }

    fn cancel_box_guard_pairs(&self, trace: &mut Trace) {
        // Iterate to a small fixed point because eliminating one pair can expose
        // another around a loop backedge.
        for _ in 0..4 {
            let mut changed = false;
            for index in 0..trace.nodes.len() {
                let kind = trace.nodes[index].kind.clone();
                let Some(result) = trace.nodes[index].results.first().copied() else {
                    continue;
                };
                match kind {
                    TraceInst::GuardF64 { value, .. } => {
                        if let Some((_, producer)) = Self::producer(trace, value) {
                            match producer {
                                TraceInst::BoxF64 { value: raw } => {
                                    alias_value(trace, result, raw);
                                    trace.nodes[index].kind = TraceInst::Nop;
                                    trace.stats.eliminated_guards += 1;
                                    changed = true;
                                }
                                TraceInst::BoxInt32 { value: raw } => {
                                    trace.nodes[index].kind = TraceInst::Int32ToF64 { value: raw };
                                    trace.stats.eliminated_guards += 1;
                                    changed = true;
                                }
                                TraceInst::BoxUInt32 { value: raw } => {
                                    trace.nodes[index].kind = TraceInst::UInt32ToF64 { value: raw };
                                    trace.stats.eliminated_guards += 1;
                                    changed = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    TraceInst::GuardInt32 { value, .. } => {
                        if let Some((_, TraceInst::BoxInt32 { value: raw })) =
                            Self::producer(trace, value)
                        {
                            alias_value(trace, result, raw);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_guards += 1;
                            changed = true;
                        }
                    }
                    TraceInst::GuardUInt32 { value, .. } => {
                        if let Some((_, TraceInst::BoxUInt32 { value: raw })) =
                            Self::producer(trace, value)
                        {
                            alias_value(trace, result, raw);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_guards += 1;
                            changed = true;
                        }
                    }
                    TraceInst::BoxF64 { value } => {
                        if let Some((_, TraceInst::GuardF64 { value: tagged, .. })) =
                            Self::producer(trace, value)
                        {
                            alias_value(trace, result, tagged);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_boxes += 1;
                            changed = true;
                        }
                    }
                    TraceInst::BoxInt32 { value } => {
                        if let Some((_, TraceInst::GuardInt32 { value: tagged, .. })) =
                            Self::producer(trace, value)
                        {
                            alias_value(trace, result, tagged);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_boxes += 1;
                            changed = true;
                        }
                    }
                    TraceInst::BoxUInt32 { value } => {
                        if let Some((_, TraceInst::GuardUInt32 { value: tagged, .. })) =
                            Self::producer(trace, value)
                        {
                            alias_value(trace, result, tagged);
                            trace.nodes[index].kind = TraceInst::Nop;
                            trace.stats.eliminated_boxes += 1;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
            if !changed {
                break;
            }
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
            let Some(result) = node.results.first().copied() else {
                continue;
            };
            let (source, rep) = match &node.kind {
                TraceInst::GuardF64 { value, .. } => (trace.canonical_value(*value), TraceRep::F64),
                TraceInst::GuardInt32 { value, .. } => {
                    (trace.canonical_value(*value), TraceRep::Int32)
                }
                TraceInst::GuardUInt32 { value, .. } => {
                    (trace.canonical_value(*value), TraceRep::UInt32)
                }
                _ => continue,
            };
            let Some(source_value) = trace.value(source) else {
                continue;
            };
            if !matches!(&source_value.origin, TraceValueOrigin::EntrySlot { frame, .. } if frame.0 == 0)
            {
                continue;
            }
            if source_value.rep != TraceRep::Tagged {
                continue;
            }
            candidates.push(GuardCandidate {
                node: node_index,
                result,
                source,
                rep,
            });
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
                        TraceRep::UInt32 => TraceType::UInt32,
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

            alias_value(trace, candidate.result, guarded);
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
            if phi.rep != TraceRep::Tagged || phi.entry_exit.is_some() {
                continue;
            }
            let Some(backedge) = phi.backedge.map(|v| trace.canonical_value(v)) else {
                continue;
            };
            let Some((_, producer)) = Self::producer(trace, backedge) else {
                continue;
            };
            let (raw_backedge, rep) = match producer {
                TraceInst::BoxF64 { value } => (trace.canonical_value(value), TraceRep::F64),
                TraceInst::BoxInt32 { value } => (trace.canonical_value(value), TraceRep::Int32),
                TraceInst::BoxUInt32 { value } => (trace.canonical_value(value), TraceRep::UInt32),
                _ => continue,
            };
            if trace.value(raw_backedge).map(|v| v.rep) != Some(rep) {
                continue;
            }

            // Every hot-path use of the tagged phi must be the matching unbox
            // guard. Snapshot uses are fine: they remember target_rep=Tagged and
            // are re-boxed only when the side exit is actually taken.
            let mut compatible = true;
            for node in &trace.nodes {
                let mut saw = false;
                let mut cloned = node.kind.clone();
                cloned.visit_operands(|operand| {
                    if trace.canonical_value(*operand) == phi.value {
                        saw = true;
                    }
                });
                if !saw {
                    continue;
                }
                compatible &= match (&node.kind, rep) {
                    (TraceInst::GuardF64 { value, .. }, TraceRep::F64) => {
                        trace.canonical_value(*value) == phi.value
                    }
                    (TraceInst::GuardInt32 { value, .. }, TraceRep::Int32) => {
                        trace.canonical_value(*value) == phi.value
                    }
                    (TraceInst::GuardUInt32 { value, .. }, TraceRep::UInt32) => {
                        trace.canonical_value(*value) == phi.value
                    }
                    _ => false,
                };
                if !compatible {
                    break;
                }
            }
            if !compatible {
                continue;
            }

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
                    TraceRep::UInt32 => TraceType::UInt32,
                    _ => value.ty,
                };
            }

            // Remove the per-iteration guard/unbox. The entry guard installed on
            // the LoopPhi establishes the invariant once.
            for index in 0..trace.nodes.len() {
                let kind = trace.nodes[index].kind.clone();
                let Some(result) = trace.nodes[index].results.first().copied() else {
                    continue;
                };
                let matches = match kind {
                    TraceInst::GuardF64 { value, .. } if candidate.rep == TraceRep::F64 => {
                        trace.canonical_value(value) == candidate.phi_value
                    }
                    TraceInst::GuardInt32 { value, .. } if candidate.rep == TraceRep::Int32 => {
                        trace.canonical_value(value) == candidate.phi_value
                    }
                    TraceInst::GuardUInt32 { value, .. } if candidate.rep == TraceRep::UInt32 => {
                        trace.canonical_value(value) == candidate.phi_value
                    }
                    _ => false,
                };
                if matches {
                    alias_value(trace, result, candidate.phi_value);
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
                TraceRep::UInt32 => trace.stats.promoted_u32_loop_phis += 1,
                _ => {}
            }
        }
    }

    fn remove_trivial_copies(&self, trace: &mut Trace) {
        for index in 0..trace.nodes.len() {
            let kind = trace.nodes[index].kind.clone();
            let Some(result) = trace.nodes[index].results.first().copied() else {
                continue;
            };
            match kind {
                TraceInst::Copy { value } => {
                    let value = trace.canonical_value(value);
                    if trace.value(value).map(|v| v.rep) == trace.value(result).map(|v| v.rep) {
                        alias_value(trace, result, value);
                        trace.nodes[index].kind = TraceInst::Nop;
                        trace.stats.eliminated_copies += 1;
                    }
                }
                TraceInst::Reinterpret { value } => {
                    let value = trace.canonical_value(value);
                    if trace.value(value).map(|v| v.rep) == trace.value(result).map(|v| v.rep) {
                        alias_value(trace, result, value);
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
            kind.visit_operands(|value| {
                used.insert(trace.canonical_value(*value));
            });
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
            if let Some(backedge) = phi.backedge {
                used.insert(trace.canonical_value(backedge));
            }
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
            let Some(result) = node.results.first().copied() else {
                continue;
            };
            if used.contains(&canonical_values[result.0 as usize]) {
                continue;
            }
            if matches!(
                &node.kind,
                TraceInst::BoxF64 { .. }
                    | TraceInst::BoxInt32 { .. }
                    | TraceInst::BoxUInt32 { .. }
                    | TraceInst::Copy { .. }
                    | TraceInst::Reinterpret { .. }
            ) {
                if matches!(
                    &node.kind,
                    TraceInst::BoxF64 { .. }
                        | TraceInst::BoxInt32 { .. }
                        | TraceInst::BoxUInt32 { .. }
                ) {
                    eliminated_boxes += 1;
                }
                node.kind = TraceInst::Nop;
            }
        }
        trace.stats.eliminated_boxes += eliminated_boxes;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::wren_bytecode::{BytecodePc, Reg};
    use crate::wren_ir::{CmpKind, F64BinOp, FuncId};

    fn builder(kind: TraceKind) -> TraceBuilder {
        TraceBuilder::new(
            TraceId(0),
            TraceAnchor {
                kind,
                location: TraceLocation::new(FuncId(0), BytecodePc(10)),
            },
        )
    }

    fn binary(
        b: &mut TraceBuilder,
        op: IntBinOp,
        lhs: TraceValueId,
        rhs: TraceValueId,
    ) -> TraceValueId {
        b.emit1(
            TraceInst::IntBinary { op, lhs, rhs },
            TraceType::Int32,
            TraceRep::Int32,
        )
    }

    /// A loop with separate examples for all three passes and live exit state.
    /// Shared with the native backend differential regression.
    pub(crate) fn loop_trace(tagged: bool) -> Trace {
        let mut b = builder(TraceKind::Loop);
        let slot = FrameSlot::General(Reg(0));
        let (ty, rep) = if tagged {
            (TraceType::Num, TraceRep::Tagged)
        } else {
            (TraceType::Int32, TraceRep::Int32)
        };
        let initial = b.read_slot(slot, ty, rep);
        let phi = b.add_loop_phi(slot, initial, ty, rep);
        let raw = if tagged {
            let exit = b.add_side_exit(ExitReason::GuardInteger, BytecodePc(10));
            b.emit1(
                TraceInst::GuardInt32 { value: phi, exit },
                TraceType::Int32,
                TraceRep::Int32,
            )
        } else {
            phi
        };
        let invariant = b.read_slot(
            FrameSlot::General(Reg(1)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let one = b.constant(TraceConstant::Int32(1));
        let two = b.constant(TraceConstant::Int32(2));
        let six = b.constant(TraceConstant::Int32(6));
        let inv = binary(&mut b, IntBinOp::Add, invariant, one);
        let chained = binary(&mut b, IntBinOp::Mul, inv, two);
        let current = binary(&mut b, IntBinOp::Mul, raw, two);
        let duplicate = binary(&mut b, IntBinOp::Mul, raw, two);
        let next = binary(&mut b, IntBinOp::Add, raw, one);
        let future = binary(&mut b, IntBinOp::Mul, next, two);
        let next_phi = if tagged {
            b.emit1(
                TraceInst::BoxInt32 { value: next },
                TraceType::Num,
                TraceRep::Tagged,
            )
        } else {
            next
        };
        b.write_slot(slot, next_phi);
        for (slot, value) in [(2, current), (3, duplicate), (4, future), (5, chained)] {
            b.write_slot(FrameSlot::General(Reg(slot)), value);
        }
        b.set_loop_backedge(phi, next_phi).unwrap();
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(20));
        let condition = b.emit1(
            TraceInst::IntCompare {
                op: CmpKind::Lt,
                lhs: next,
                rhs: six,
            },
            TraceType::Bool,
            TraceRep::Bool,
        );
        let if_true = b.constant(TraceConstant::Bool(true));
        let if_false = b.constant(TraceConstant::Bool(false));
        let condition = b.emit1(
            TraceInst::Select {
                condition,
                if_true,
                if_false,
            },
            TraceType::Bool,
            TraceRep::Tagged,
        );
        b.emit0(TraceInst::GuardTruthy {
            value: condition,
            exit,
        });
        b.add_native_budget_exit(BytecodePc(10));
        b.set_terminator(TraceTerminator::LoopBack);
        b.finish().unwrap()
    }

    #[test]
    fn default_pipeline_runs_gvn_licm_and_loop_pre() {
        for tagged in [false, true] {
            let mut trace = loop_trace(tagged);
            let phis = trace.loop_phis.len();
            optimization_passes().optimize_trace(&mut trace).unwrap();
            assert!(trace.stats.gvn_eliminated >= 1);
            assert!(trace.stats.licm_hoisted >= 2);
            assert_eq!(trace.stats.pre_eliminated, 1);
            assert_eq!(trace.loop_phis.len(), phis + 1);
            assert!(trace.preheader.len() >= 3);
            let temporary = trace.loop_phis.last().unwrap().value;
            assert!(
                trace
                    .exits
                    .iter()
                    .flat_map(|exit| &exit.frames)
                    .flat_map(|frame| &frame.writes)
                    .any(|write| write.value == SnapshotValue::Value(temporary))
            );
            let before = (trace.stats.pre_eliminated, trace.preheader.len());
            optimization_passes().optimize_trace(&mut trace).unwrap();
            assert_eq!((trace.stats.pre_eliminated, trace.preheader.len()), before);
        }
    }

    #[test]
    fn gvn_keeps_float_bits_and_result_representations_distinct() {
        let mut b = builder(TraceKind::Side);
        let x = b.read_slot(FrameSlot::General(Reg(0)), TraceType::F64, TraceRep::F64);
        let positive = b.constant(TraceConstant::F64Bits(0.0f64.to_bits()));
        let negative = b.constant(TraceConstant::F64Bits((-0.0f64).to_bits()));
        let duplicate = b.constant(TraceConstant::F64Bits(0.0f64.to_bits()));
        let mut results = Vec::new();
        for rhs in [positive, negative, duplicate] {
            results.push(b.emit1(
                TraceInst::F64Binary {
                    op: F64BinOp::Add,
                    lhs: x,
                    rhs,
                },
                TraceType::F64,
                TraceRep::F64,
            ));
        }
        let left_nan = b.constant(TraceConstant::F64Bits(0x7ff8_0000_0000_0001));
        let right_nan = b.constant(TraceConstant::F64Bits(0x7ff8_0000_0000_0002));
        let nans = [left_nan, right_nan].map(|v| {
            b.emit1(
                TraceInst::BoxF64 { value: v },
                TraceType::Num,
                TraceRep::Tagged,
            )
        });
        let raw = b.constant(TraceConstant::Int32(-1));
        let signed = b.emit1(
            TraceInst::Reinterpret { value: raw },
            TraceType::Int32,
            TraceRep::Int32,
        );
        let unsigned = b.emit1(
            TraceInst::Reinterpret { value: raw },
            TraceType::UInt32,
            TraceRep::UInt32,
        );
        for (i, value) in results
            .iter()
            .chain(&nans)
            .chain([&signed, &unsigned])
            .enumerate()
        {
            b.write_slot(FrameSlot::General(Reg(i as u8)), *value);
        }
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(20));
        b.set_terminator(TraceTerminator::Exit(exit));
        let mut trace = b.finish().unwrap();
        Optimize::new()
            .with_pass(GlobalValueNumbering)
            .optimize_trace(&mut trace)
            .unwrap();
        assert_eq!(
            trace.canonical_value(results[0]),
            trace.canonical_value(results[2])
        );
        assert_ne!(
            trace.canonical_value(results[0]),
            trace.canonical_value(results[1])
        );
        assert_ne!(
            trace.canonical_value(nans[0]),
            trace.canonical_value(nans[1])
        );
        assert_ne!(
            trace.canonical_value(signed),
            trace.canonical_value(unsigned)
        );
    }

    #[test]
    fn runtime_is_a_barrier_and_loads_guards_and_traps_stay_in_body() {
        let mut b = builder(TraceKind::Loop);
        let x = b.read_slot(
            FrameSlot::General(Reg(0)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let one = b.constant(TraceConstant::Int32(1));
        let zero = b.constant(TraceConstant::Int32(0));
        let a = binary(&mut b, IntBinOp::Add, x, one);
        let division = binary(&mut b, IntBinOp::DivS, x, zero);
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(20));
        let checked = b.emit1(
            TraceInst::IntBinaryChecked {
                op: IntBinOp::Add,
                lhs: x,
                rhs: one,
                exit,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        b.runtime(
            TraceLocation::new(FuncId(0), BytecodePc(11)),
            BytecodePc(12),
            false,
        );
        let b_value = binary(&mut b, IntBinOp::Add, x, one);
        let loads = [0, 0].map(|r| {
            b.emit1(
                TraceInst::LoadFrameSlot {
                    slot: FrameSlot::General(Reg(r)),
                },
                TraceType::Int32,
                TraceRep::Int32,
            )
        });
        for (i, value) in [a, division, checked, b_value, loads[0], loads[1]]
            .into_iter()
            .enumerate()
        {
            b.write_slot(FrameSlot::General(Reg(i as u8)), value);
        }
        b.set_terminator(TraceTerminator::LoopBack);
        let mut trace = b.finish().unwrap();
        Optimize::new()
            .with_pass(GlobalValueNumbering)
            .with_pass(LoopInvariantCodeMotion)
            .with_pass(PartialRedundancyElimination)
            .optimize_trace(&mut trace)
            .unwrap();
        assert!(trace.preheader.is_empty());
        assert_ne!(trace.canonical_value(a), trace.canonical_value(b_value));
        assert_ne!(
            trace.canonical_value(loads[0]),
            trace.canonical_value(loads[1])
        );
        assert!(trace.nodes.iter().any(|n| matches!(
            n.kind,
            TraceInst::IntBinary {
                op: IntBinOp::DivS,
                ..
            }
        )));
        assert!(
            trace
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TraceInst::IntBinaryChecked { .. }))
        );
    }

    #[test]
    fn licm_does_not_speculate_trapping_or_checked_operations() {
        let mut b = builder(TraceKind::Loop);
        let x = b.constant(TraceConstant::Int32(1));
        let zero = b.constant(TraceConstant::Int32(0));
        binary(&mut b, IntBinOp::DivS, x, zero);
        binary(&mut b, IntBinOp::RemS, x, zero);
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(20));
        b.emit1(
            TraceInst::IntBinaryChecked {
                op: IntBinOp::Add,
                lhs: x,
                rhs: zero,
                exit,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        b.set_terminator(TraceTerminator::LoopBack);
        let mut trace = b.finish().unwrap();
        Optimize::new()
            .with_pass(LoopInvariantCodeMotion)
            .optimize_trace(&mut trace)
            .unwrap();
        assert!(trace.preheader.is_empty());
    }

    #[test]
    fn pre_requires_available_backedge_expression_and_a_loop() {
        let mut trace = loop_trace(false);
        // With no loop edge there is nowhere to reuse a previous computation.
        trace.anchor.kind = TraceKind::Side;
        let before = trace.values.len();
        assert!(!PartialRedundancyElimination.optimize_trace(&mut trace));
        assert_eq!(trace.values.len(), before);
        let mut b = builder(TraceKind::Loop);
        let initial = b.constant(TraceConstant::Int32(0));
        let phi = b.add_loop_phi(
            FrameSlot::General(Reg(0)),
            initial,
            TraceType::Int32,
            TraceRep::Int32,
        );
        let one = b.constant(TraceConstant::Int32(1));
        let next = binary(&mut b, IntBinOp::Add, phi, one);
        b.set_loop_backedge(phi, next).unwrap();
        b.set_terminator(TraceTerminator::LoopBack);
        let mut trace = b.finish().unwrap();
        assert!(!PartialRedundancyElimination.optimize_trace(&mut trace));
        assert!(trace.preheader.is_empty());
    }

    #[test]
    fn verifier_rejects_preheader_phi_uses_effects_and_late_initializers() {
        let trace = loop_trace(false);
        let id = trace
            .nodes
            .iter()
            .position(|n| {
                matches!(n.kind, TraceInst::IntBinary {
            op: IntBinOp::Mul, lhs, .. } if lhs == trace.loop_phis[0].value)
            })
            .unwrap();
        let mut invalid = trace.clone();
        invalid.preheader.push(id);
        assert!(verify_trace(&invalid).is_err());
        let mut invalid = trace.clone();
        invalid.loop_phis[0].initial = invalid.nodes[id].results[0];
        assert!(verify_trace(&invalid).is_err());
        let mut invalid = trace;
        let guard = invalid
            .nodes
            .iter()
            .position(|n| n.kind.exit().is_some())
            .unwrap();
        invalid.preheader.push(guard);
        assert!(verify_trace(&invalid).is_err());
    }
}
