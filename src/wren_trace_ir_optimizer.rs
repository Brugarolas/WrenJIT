//! Extensible, verified optimization pipeline for Trace IR.
//!
//! Node/value IDs remain stable. Replaced producers become Nops and their values
//! become aliases; all uses, including deoptimization snapshots, are repaired.

use crate::wren_trace_ir::*;

pub trait TraceOptimization {
    fn optimize_trace(&mut self, trace: &mut Trace) -> bool;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub struct Optimize {
    passes: Vec<Box<dyn TraceOptimization>>,
    verify_after_each_pass: bool,
}

impl Default for Optimize {
    fn default() -> Self {
        Self {
            passes: Vec::new(),
            verify_after_each_pass: cfg!(debug_assertions),
        }
    }
}

impl Optimize {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pass(mut self, pass: impl TraceOptimization + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    pub fn verify_after_each_pass(mut self, enabled: bool) -> Self {
        self.verify_after_each_pass = enabled;
        self
    }

    pub fn optimize_trace(&mut self, trace: &mut Trace) -> Result<(), Vec<String>> {
        verify_trace(trace)?;
        for pass in &mut self.passes {
            pass.optimize_trace(trace);
            canonicalize_references(trace);
            if self.verify_after_each_pass {
                verify_trace(trace).map_err(|errors| {
                    errors
                        .into_iter()
                        .map(|error| format!("{}: {error}", pass.name()))
                        .collect::<Vec<_>>()
                })?;
            }
        }
        verify_trace(trace)
    }
}

/// Compatibility facade for the previous TraceOptimizer entry points.
#[derive(Default)]
pub struct TraceOptimizer;

impl TraceOptimizer {
    pub fn optimize(&self, trace: &mut Trace) -> Result<(), Vec<String>> {
        crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(trace)
    }

    pub fn optimize_conservative(&self, trace: &mut Trace) -> Result<(), Vec<String>> {
        crate::wren_trace_ir_optimizing::conservative_optimization_passes().optimize_trace(trace)
    }
}

pub(crate) fn alias_value(trace: &mut Trace, from: TraceValueId, to: TraceValueId) {
    let to = trace.canonical_value(to);
    if from == to {
        return;
    }
    trace.values[from.0 as usize].origin = TraceValueOrigin::Alias(to);
    canonicalize_references(trace);
}

pub(crate) fn canonicalize_references(trace: &mut Trace) {
    let aliases: Vec<_> = trace
        .values
        .iter()
        .map(|v| trace.canonical_value(v.id))
        .collect();
    // Preserve invalid references for the verifier instead of panicking when a
    // custom pass produces malformed IR.
    let resolve = |v: &mut TraceValueId| {
        if let Some(alias) = aliases.get(v.0 as usize) {
            *v = *alias;
        }
    };
    for node in &mut trace.nodes {
        node.kind.visit_operands(resolve);
    }
    for value in &mut trace.values {
        match &mut value.origin {
            TraceValueOrigin::Alias(v) | TraceValueOrigin::GuardedEntry { value: v, .. } => {
                resolve(v);
            }
            _ => {}
        }
    }
    for exit in &mut trace.exits {
        for frame in &mut exit.frames {
            for write in &mut frame.writes {
                if let SnapshotValue::Value(v) = &mut write.value {
                    resolve(v);
                }
            }
        }
    }
    for phi in &mut trace.loop_phis {
        resolve(&mut phi.initial);
        if let Some(v) = &mut phi.backedge {
            resolve(v);
        }
    }
}

/// Append a pure producer to the preheader without renumbering the trace body.
pub(crate) fn append_preheader(
    trace: &mut Trace,
    kind: TraceInst,
    ty: TraceType,
    rep: TraceRep,
    source: TraceLocation,
) -> TraceValueId {
    assert!(kind.is_speculatable());
    let id = TraceValueId(trace.values.len() as u32);
    let node = trace.nodes.len();
    trace.values.push(TraceValue {
        id,
        ty,
        rep,
        origin: TraceValueOrigin::InstResult { node },
    });
    trace.nodes.push(TraceNode {
        id: node,
        kind,
        results: vec![id],
        source,
    });
    trace.preheader.push(node);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_pass_reports_invalid_ir_with_stage_name() {
        struct BrokenPass;
        impl TraceOptimization for BrokenPass {
            fn optimize_trace(&mut self, trace: &mut Trace) -> bool {
                trace.nodes[0]
                    .kind
                    .visit_operands(|v| *v = TraceValueId(u32::MAX));
                true
            }
        }
        let mut trace = crate::wren_trace_ir_optimizing::tests::loop_trace(false);
        let errors = Optimize::new()
            .with_pass(BrokenPass)
            .verify_after_each_pass(true)
            .optimize_trace(&mut trace)
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("BrokenPass") && e.contains("unavailable"))
        );
    }
}
