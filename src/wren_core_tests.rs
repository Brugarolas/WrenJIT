//! Source -> parser -> HIR -> bytecode -> interpreter regression tests.

use std::cell::RefCell;
use std::rc::Rc;

use crate::wren_bytecode::BytecodeProgram;
use crate::wren_interpreter::{
    BuiltinClass, ExecutionResult, Host, HostContext, Interpreter, RuntimeError, Value,
};
use crate::wren_ir::{ForeignFnId, FuncId, MethodKind, PrimitiveId};
use crate::{wren_bytecodecompiler, wren_core, wren_front, wren_lowerer};

#[derive(Clone, Default)]
struct Output(Rc<RefCell<Vec<u8>>>);

impl Output {
    fn text(&self) -> String {
        String::from_utf8(self.0.borrow().clone()).unwrap()
    }
}

impl Host for Output {
    fn write(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(())
    }

    fn call_primitive(
        &mut self,
        _: &mut HostContext<'_>,
        primitive: PrimitiveId,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        assert_eq!(primitive, PrimitiveId(0));
        assert_eq!(args.len(), 2);
        Ok(Value::from_num(42.0))
    }

    fn call_foreign(
        &mut self,
        _: &mut HostContext<'_>,
        _: ForeignFnId,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        assert_eq!(args.len(), 1);
        Ok(Value::from_num(17.0))
    }
}

fn compile(source: &str) -> (BytecodeProgram, FuncId) {
    let parsed = wren_front::parse("test.wren", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let lowered = wren_lowerer::lower_new_program("test.wren", source, "test", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let entry = lowered.lowered.unwrap().entry;
    (
        wren_bytecodecompiler::compile_program(&lowered.program).unwrap(),
        entry,
    )
}

fn execute(source: &str) -> Result<String, RuntimeError> {
    let (program, entry) = compile(source);
    let output = Output::default();
    let mut vm = Interpreter::with_host(&program, Box::new(output.clone()))?;
    assert!(matches!(
        vm.run_module(entry)?,
        ExecutionResult::Returned(_)
    ));
    Ok(output.text())
}

#[test]
fn print_aliases_are_regular_calls_and_return_the_argument() {
    assert_eq!(
        execute(
            r#"
System.Print("Hola")
System . Print("con espacios")
System.write("sin salto")
System.print()
System.Print()
var value = System.print(42)
System.print(value)
var console = System
console.Print(true)
System.print(null)
"#
        )
        .unwrap(),
        "Hola\ncon espacios\nsin salto\n\n42\n42\ntrue\nnull\n"
    );
}

#[test]
fn arithmetic_comparisons_and_interpolation_use_native_dispatch() {
    assert_eq!(
        execute(
            r#"
System.print("resultado: %(6 * 7)")
System.Print("uno" + " dos")
System.print(8 / 2 - 1)
System.print(-5 % 2)
System.print(1 < 2)
System.print(2 <= 2)
System.print(3 > 2)
System.print(2 >= 3)
System.print(!false)
System.print(!null)
System.print(!0)
System.print(("a" + "b") == "ab")
System.print(0 == -0)
var nan = 0 / 0
System.print(nan == nan)
System.print(nan != nan)
"#
        )
        .unwrap(),
        "resultado: 42\nuno dos\n3\n-1\ntrue\ntrue\ntrue\nfalse\ntrue\ntrue\nfalse\ntrue\ntrue\nfalse\ntrue\n"
    );
}

#[test]
fn methods_closures_and_user_to_string_work_with_gc() {
    assert_eq!(
        execute(
            r#"
class Greeting {
  construct new(text) { _text = text }
  toString {
    System.gc()
    return "Hola " + _text
  }
  show() { System.Print(this) }
}
var greeting = Greeting.new("mundo")
greeting.show()
var show = Fn.new { |value| System.Print(value) }
show.call(greeting)
System.print("%(greeting)")
var clock = System.clock
System.gc()
System.print(System.clock >= clock)
"#
        )
        .unwrap(),
        "Hola mundo\nHola mundo\nHola mundo\ntrue\n"
    );
}

#[test]
fn core_names_can_be_shadowed_without_changing_other_classes() {
    assert_eq!(
        execute(
            r#"
var console = System
{
  var System = "local"
  console.Print(System)
}
class Custom {
  static Print(text) { System.print(text) }
}
Custom.Print("custom")
System.print("core")
"#
        )
        .unwrap(),
        "local\ncustom\ncore\n"
    );
    assert_eq!(
        execute(
            r#"
class System {
  static Print(value) { value }
}
System.Print("silencio")
"#
        )
        .unwrap(),
        ""
    );
}

#[test]
fn invalid_signatures_and_types_report_runtime_errors() {
    for (source, expected) in [
        ("System.Print(1, 2)", "2 argument(s)"),
        ("System.PrintMore(1)", "PrintMore"),
        ("System.clock()", "clock"),
        ("System.print(1 + true)", "number argument"),
        ("System.print(\"a\" + 1)", "string argument"),
        ("Fn.new(1)", "expects a function"),
        ("var f = Fn.new { |a| a }\nf.call()", "argument"),
        (
            "class Bad {\n construct new() {}\n toString { 123 }\n}\nSystem.print(Bad.new())",
            "toString must return a string",
        ),
    ] {
        let error = execute(source).expect_err(source);
        assert!(
            error.message.contains(expected),
            "{source}: {}",
            error.message
        );
        assert!(error.function.is_some());
    }
}

#[test]
fn automatic_and_explicit_gc_preserve_native_arguments_and_captures() {
    assert_eq!(
        execute(
            r#"
var f = null
{
  var text = "hello" + " world"
  f = Fn.new { text }
}
var i = 0
while (i < 3000) {
  var temporary = "iteration %(i)"
  if (i % 2000 == 0) System.gc()
  i = i + 1
}
System.print(f.call())
"#
        )
        .unwrap(),
        "hello world\n"
    );
}

#[test]
fn rebinding_a_native_invalidates_existing_call_caches() {
    let (program, entry) = compile("System.print(1 + 2)");
    let method = program
        .metadata
        .methods
        .iter()
        .enumerate()
        .find(|(_, method)| {
            program.metadata.string(method.name) == Some("+")
                && method.kind == MethodKind::Method
                && method.arity == 1
        })
        .map(|(index, _)| crate::wren_ir::MethodSymbol(index as u32))
        .unwrap();
    let output = Output::default();
    let mut vm = Interpreter::with_host(&program, Box::new(output.clone())).unwrap();
    vm.run_module(entry).unwrap();
    vm.bind_builtin_primitive(BuiltinClass::Num, method, PrimitiveId(0));
    vm.run_module(entry).unwrap();
    assert_eq!(output.text(), "3\n42\n");
}

#[test]
fn user_foreign_methods_still_reach_the_host() {
    assert_eq!(
        execute("class Native {\n foreign static answer()\n}\nSystem.print(Native.answer())")
            .unwrap(),
        "17\n"
    );
}

#[test]
fn output_errors_propagate_without_panicking() {
    struct BrokenOutput;
    impl Host for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> Result<(), RuntimeError> {
            Err(RuntimeError::new("output unavailable"))
        }
    }
    let (program, entry) = compile("System.print(42)");
    let mut vm = Interpreter::with_host(&program, Box::new(BrokenOutput)).unwrap();
    assert_eq!(
        vm.run_module(entry).unwrap_err().message,
        "output unavailable"
    );
}

#[test]
fn core_is_compiled_once_for_multiple_modules() {
    let mut hir = crate::wren_ir::Program::new();
    let mut entries = Vec::new();
    for (name, source) in [("one", "System.Print(1)"), ("two", "System.print(2)")] {
        let parsed = wren_front::parse(name, source);
        let lowered = wren_lowerer::lower_module(&mut hir, name, source, name, &parsed.module);
        assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
        entries.push(lowered.lowered.unwrap().entry);
    }
    assert_eq!(
        hir.modules
            .iter()
            .filter(|module| hir.string(module.name) == Some(wren_core::MODULE_NAME))
            .count(),
        1
    );
    let program = wren_bytecodecompiler::compile_program(&hir).unwrap();
    let output = Output::default();
    let mut vm = Interpreter::with_host(&program, Box::new(output.clone())).unwrap();
    for entry in entries {
        vm.run_module(entry).unwrap();
    }
    assert_eq!(output.text(), "1\n2\n");
}

#[test]
fn builtin_globals_are_not_assignable_without_a_declaration() {
    let source = "System = null";
    let parsed = wren_front::parse("test", source);
    let lowered = wren_lowerer::lower_new_program("test", source, "test", &parsed.module);
    assert!(lowered.has_errors());
}

#[test]
fn functions_report_arity_and_ignore_extra_arguments() {
    assert_eq!(
        execute("var f = Fn.new { |a| a }\nSystem.print(f.arity)\nSystem.print(f.call(7, 8))")
            .unwrap(),
        "1\n7\n"
    );
}

#[test]
fn parser_recovers_from_unmatched_module_braces() {
    let parsed = wren_front::parse("test", "}\nSystem.Print(1)\n");
    assert!(parsed.has_errors());
    assert_eq!(parsed.module.statements.len(), 1);
}
