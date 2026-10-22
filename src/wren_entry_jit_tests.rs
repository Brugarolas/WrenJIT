// Entry-tier regressions, included in the interpreter's source-test module.
use crate::wren_bytecode::{ArgReg, ParameterBinding};

#[test]
fn jit_entry_compiles_loop_free_functions_and_resets_cached_code() {
    let mut asm = Assembler::new();
    asm.emit_ab(Opcode::NumToF64, 1, 0);
    asm.emit_abc(Opcode::FMul, 2, 1, 1);
    asm.emit_ab(Opcode::F64ToNum, 3, 2);
    asm.emit_a(Opcode::Return, 3);
    let mut program = simple_program(asm.finish().unwrap(), 4);
    program.functions[0].arity = 1;
    program.functions[0].frame.parameters.push(ParameterBinding {
        source: IncomingSource::Argument(ArgReg(0)), destination: Reg(0), kind: RegisterKind::Value,
    });
    for optimize_trace_ir in [false, true] {
        let mut vm = Interpreter::new(&program).unwrap();
        for threshold in [0, 2, 2] {
            vm.set_jit_options(Some(TracingJitOptions {
                hot_entry_threshold: threshold, hot_side_exit_threshold: 1,
                optimize_trace_ir, ..Default::default()
            }));
            for x in 1..80 {
                let result = vm.run_function(FuncId(0), &[Value::from_num(x as f64)]).unwrap();
                assert!(matches!(result, ExecutionResult::Returned(value) if value.as_num() == Some((x*x) as f64)));
            }
            let stats = vm.jit_stats().unwrap();
            assert_eq!(stats.loop_header_hits, 0, "{stats:?}");
            assert_eq!(stats.native_iterations, 0, "{stats:?}");
            assert_eq!(stats.trace_cooldowns, 0, "{stats:?}");
            assert_eq!(stats.side_traces_compiled, 0, "{stats:?}");
            if threshold == 0 { assert_eq!(stats.traces_compiled, 0); }
            else {
                assert_eq!(stats.entry_traces_compiled, 1, "{stats:?}");
                assert!(stats.entry_native_entries > 60, "{stats:?}");
            }
        }
        vm.set_jit_options(None);
        assert!(matches!(vm.run_function(FuncId(0), &[Value::from_num(9.0)]).unwrap(),
            ExecutionResult::Returned(value) if value.as_num() == Some(81.0)));
    }
}

#[test]
fn jit_entry_prefixes_and_call_continuations_preserve_effects_and_gc() {
    let source = r#"
        class Counter {
            construct new() { _calls = 0 }
            apply(x) {
                _calls = _calls + 1
                return x * 2
            }
            calls { _calls }
        }
        class Test {
            static calculate(counter, x) {
                var prefix = x * 3 + 1
                var result = counter.apply(prefix)
                return result + prefix * 4
            }
            static run() {
                var counter = Counter.new()
                var sum = 0
                for (i in 1..160) {
                    sum = sum + calculate(counter, i)
                    if (i % 20 == 0) System.gc()
                }
                return sum + counter.calls
            }
        }
        var result = Test.run()
    "#;
    let (program, entry, manifest) = compile_wren_module(source);
    for optimize_trace_ir in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_entry_threshold: 2, hot_loop_threshold: u32::MAX,
            optimize_trace_ir, ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 232960.0);
        let stats = vm.jit_stats().unwrap();
        assert!(stats.entry_native_entries > 200, "{stats:?}");
        assert!(stats.entry_traces_compiled >= 3, "{stats:?}");
        assert_eq!(stats.native_iterations, 0, "{stats:?}");
        let jit = vm.jit.as_ref().unwrap();
        assert!(vm.prepared.functions.iter().any(|prepared| {
            prepared.jit_entry_traces.iter().enumerate().any(|(pc, &id)| pc > 0 && id != NO_INDEX
                && jit.trace(TraceId(id)).unwrap().stats.recorded_bytecodes >= 3)
        }), "missing compiled call continuation");
        for prepared in &vm.prepared.functions {
            for &id in &prepared.jit_entry_traces {
                if id == NO_INDEX { continue; }
                let trace = jit.trace(TraceId(id)).unwrap();
                assert!(trace.loop_phis.is_empty());
                assert!(!trace.nodes.iter().any(|node| matches!(node.kind, crate::wren_trace_ir::TraceInst::Runtime { .. })));
            }
        }
    }
}

#[test]
fn jit_entry_branch_exits_recursion_and_numeric_type_changes() {
    let source = r#"
        class Test {
            static choose(x) {
                var y = x * 2
                if (x < 30) return y + 1
                return y - 3
            }
            static recurse(x) {
                if (x < 1) return 1
                var y = x + 2
                return recurse(x - 1) + y
            }
            static run() {
                var sum = 0
                for (i in 0..60) sum = sum + choose(i)
                for (i in 0..20) sum = sum + choose(i + 0.5)
                return sum + recurse(15)
            }
        }
        var result = Test.run()
    "#;
    let (program, entry, manifest) = compile_wren_module(source);
    let mut baseline = source_vm(&program, &manifest);
    baseline.set_jit_options(None);
    baseline.run_module(entry).unwrap();
    let expected = source_result(&baseline);
    for optimize_trace_ir in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_entry_threshold: 2, hot_loop_threshold: u32::MAX,
            optimize_trace_ir, ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), expected);
        assert!(vm.jit_stats().unwrap().entry_native_entries > 50);
    }
}

#[test]
fn jit_entry_hands_loop_headers_to_the_loop_tier() {
    let (program, entry, manifest) = compile_wren_module(r#"
        class Test {
            static sum(n) {
                var total = n * 2
                var i = 0
                while (i < n) {
                    total = total + i
                    i = i + 1
                }
                return total * 2
            }
            static run() {
                var total = 0
                for (i in 1..20) total = total + sum(200)
                return total
            }
        }
        var result = Test.run()
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2, hot_loop_threshold: 2, ..Default::default()
    }));
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), 812000.0);
    let stats = vm.jit_stats().unwrap();
    assert!(stats.entry_native_entries > 10, "{stats:?}");
    assert!(stats.native_iterations > 3000, "{stats:?}");
}

#[test]
fn jit_entry_resumes_inside_fused_instructions_and_at_step_limit() {
    for max_trace_steps in [1, 2, 4096] {
        for constant_return in [false, true] {
            let mut asm = Assembler::new();
            asm.emit_a(Opcode::LoadTrue, 0);
            if !constant_return {
                asm.emit_ab(Opcode::Move, 1, 0);
                asm.emit_ab(Opcode::Move, 2, 1);
                asm.emit_ab(Opcode::Move, 0, 2);
            }
            asm.emit_a(Opcode::Return, 0);
            let program = simple_program(asm.finish().unwrap(), 3);
            let mut vm = Interpreter::new(&program).unwrap();
            vm.set_jit_options(Some(TracingJitOptions {
                hot_entry_threshold: 1, max_trace_steps, ..Default::default()
            }));
            for _ in 0..10 {
                assert!(matches!(vm.run_function(FuncId(0), &[]).unwrap(),
                    ExecutionResult::Returned(value) if value == Value::TRUE));
            }
            assert_eq!(vm.jit_stats().unwrap().entry_traces_compiled, 1);
            assert_eq!(vm.jit_stats().unwrap().entry_native_entries, 9);
            vm.invalidate_method_traces();
            assert!(matches!(vm.run_function(FuncId(0), &[]).unwrap(),
                ExecutionResult::Returned(value) if value == Value::TRUE));
            assert_eq!(vm.jit_stats().unwrap().entry_native_entries, 0);
        }
    }
}

#[test]
fn jit_entry_leaves_runtime_errors_to_the_interpreter() {
    let (program, entry, manifest) = compile_wren_module(r#"
        class Test {
            static calc(x) {
                var y = x + 1
                return y.missingMethod()
            }
        }
        var result = Test.calc(3)
    "#);
    let mut expected = None;
    for threshold in [0, 1] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_entry_threshold: threshold, ..Default::default()
        }));
        let error = vm.run_module(entry).unwrap_err().to_string();
        if let Some(expected) = &expected { assert_eq!(&error, expected); }
        else { expected = Some(error); }
    }
}
