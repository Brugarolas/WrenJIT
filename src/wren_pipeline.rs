//! Shared HIR -> optimizations -> bytecode compilation pipeline.
//!
//! Frontends lower into a `Program`, then enter this pipeline before execution.
//! Keep the default pass list in `wren_ir_optimizing::optimization_passes`; custom
//! pipelines can use the same `Optimize` builder for function and node passes.

use crate::wren_bytecode::BytecodeProgram;
use crate::wren_bytecodecompiler::{
    BytecodeCompiler, CompilationOutput, CompileError, CompileOptions,
};
use crate::wren_ir::Program;
use crate::wren_ir_optimizer::Optimize;
use crate::wren_ir_optimizing::optimization_passes;

pub struct CompilationPipeline {
    optimizations: Optimize,
    bytecode: BytecodeCompiler,
}

impl Default for CompilationPipeline {
    fn default() -> Self {
        Self::new(optimization_passes())
    }
}

impl CompilationPipeline {
    /// Select an ordered set of HIR passes. `Optimize::new()` disables this stage.
    pub fn new(optimizations: Optimize) -> Self {
        Self {
            optimizations,
            bytecode: BytecodeCompiler::default(),
        }
    }

    /// Configure bytecode lowering independently of the HIR optimization passes.
    pub fn with_bytecode_options(mut self, options: CompileOptions) -> Self {
        self.bytecode = BytecodeCompiler::new(options);
        self
    }

    /// Optimize the lowered HIR in place, then generate executable bytecode.
    pub fn compile(&mut self, hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
        Ok(self.compile_with_stats(hir)?.program)
    }

    /// Backend statistics describe the HIR after optimization.
    pub fn compile_with_stats(
        &mut self,
        hir: &mut Program,
    ) -> Result<CompilationOutput, CompileError> {
        self.optimizations.optimize_program(hir);
        self.bytecode.compile_with_stats(hir)
    }
}

/// Compile lowered HIR using the language's default optimization passes.
pub fn compile_program(hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
    CompilationPipeline::default().compile(hir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::disassemble;
    use crate::wren_interpreter::{ExecutionResult, Interpreter};
    use crate::wren_ir::{Builder, Constant, FuncId, FunctionKind, Inst, IntBinOp, Type};
    use crate::wren_ir_optimizer::{InstCursor, OptimizationPass, OptimizationWIR, WirEditQueue};
    use crate::wren_ir_optimizing::ConstantFold;

    fn arithmetic_program() -> (Program, FuncId) {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        let mut builder = Builder::new(
            &mut hir,
            module,
            "main",
            FunctionKind::ModuleBody,
            vec![],
            Type::Num,
        );
        builder.const_i32(99); // Dead value exercises the separate backend DCE option.
        let left = builder.const_i32(20);
        let right = builder.const_i32(22);
        let sum = builder.emit1(
            Inst::IntBin {
                op: IntBinOp::Add,
                left,
                right,
            },
            Type::Int32,
        );
        let result = builder.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
        builder.ret(result);
        let entry = builder.finish();
        (hir, entry)
    }

    fn execute(program: &BytecodeProgram, entry: FuncId) -> f64 {
        let mut vm = Interpreter::new(program).unwrap();
        vm.set_jit_options(None);
        let ExecutionResult::Returned(value) = vm.run_module(entry).unwrap() else {
            panic!("module should return");
        };
        value.as_num().unwrap()
    }

    #[test]
    fn default_pipeline_folds_hir_before_bytecode_generation() {
        let (mut hir, entry) = arithmetic_program();
        let baseline = BytecodeCompiler::default().compile(&hir).unwrap();
        assert!(
            disassemble(&baseline.functions[0])
                .unwrap()
                .contains("IAdd")
        );

        let optimized = compile_program(&mut hir).unwrap();
        let function = hir.function(entry).unwrap();
        assert_eq!(function.blocks[0].insts.len(), 2, "constant and return");
        assert!(
            !disassemble(&optimized.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&baseline, entry), 42.0);
        assert_eq!(execute(&optimized, entry), 42.0);
    }

    #[test]
    fn empty_pipeline_preserves_hir_and_backend_configuration() {
        let (mut hir, entry) = arithmetic_program();
        let before = hir.function(entry).unwrap().to_string();
        let output = CompilationPipeline::new(Optimize::new())
            .with_bytecode_options(CompileOptions {
                eliminate_dead_code: false,
                ..CompileOptions::default()
            })
            .compile_with_stats(&mut hir)
            .unwrap();
        assert_eq!(hir.function(entry).unwrap().to_string(), before);
        assert_eq!(output.stats.functions[0].emitted_hir_nodes, 6);
        assert!(
            disassemble(&output.program.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&output.program, entry), 42.0);
    }

    #[test]
    fn custom_node_and_function_passes_run_in_order_before_codegen() {
        struct ReplaceTwenty;
        impl OptimizationWIR for ReplaceTwenty {
            fn optimize_node(&mut self, cur: &mut InstCursor<'_>, q: &mut WirEditQueue) -> bool {
                if matches!(cur.cur_kind(), Inst::Const(Constant::Int32(20))) {
                    q.replace_kind(cur.cur_id(), Inst::Const(Constant::Int32(21)));
                    return true;
                }
                false
            }
        }

        let (mut hir, entry) = arithmetic_program();
        let passes = Optimize::new()
            .with_pass(OptimizationPass::new(vec![Box::new(ReplaceTwenty)]))
            .with_function_pass(ConstantFold);
        let bytecode = CompilationPipeline::new(passes).compile(&mut hir).unwrap();
        assert!(
            !disassemble(&bytecode.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&bytecode, entry), 43.0);
    }
}
