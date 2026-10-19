//! The small default core library. Built-ins use ordinary Wren method dispatch.
//!
//! Add Wren wrappers to SOURCE and native signatures to the tables below. A new
//! global class must also be listed in GLOBALS, in declaration order. Native
//! implementations live in Interpreter::call_native; no parser changes or
//! reserved PrimitiveId values are needed. This is not yet the full Wren core.
//!
//! Currently installed globals:
//! - System.print()/print(value), Print()/Print(value), write(value), clock, gc().
//! - Fn.new(body), with call(...) and arity on function values.
//!
//! Values also support toString, !, == and !=, plus numeric arithmetic and
//! comparisons and string concatenation. Collection protocols, the remaining
//! numeric methods, and the other standard global classes are not implemented.
//!
//! System output goes through Host::write (stdout by default). The core's
//! foreign functions are resolved separately from an embedding host's native
//! registrations, which continue to use Host::call_primitive/call_foreign.

use crate::wren_ir::{MethodKind, ModuleSlot};

pub(crate) const MODULE_NAME: &str = "<wren-core>";
pub(crate) const GLOBALS: &[&str] = &["System", "Fn"];

pub(crate) fn global_slot(name: &str) -> Option<ModuleSlot> {
    GLOBALS
        .iter()
        .position(|&global| global == name)
        .map(|i| ModuleSlot(i as u32))
}

// Wrappers deliberately invoke toString through Wren dispatch, so user classes
// can override it and conversions can allocate, call methods, or report errors.
pub(crate) const SOURCE: &str = r#"
class System {
  foreign static writeString_(text)
  foreign static clock
  foreign static gc()

  static write(value) {
    writeString_(value.toString)
    return value
  }

  static print(value) {
    write(value)
    writeString_("\n")
    return value
  }

  static print() {
    writeString_("\n")
  }

  static Print(value) { print(value) }
  static Print() { print() }
}

class Fn {
  foreign static new(body)
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinClass {
    Null,
    Bool,
    Num,
    String,
    List,
    Map,
    Range,
    Closure,
    Fiber,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeMethod {
    WriteString,
    Clock,
    CollectGarbage,
    FnNew,
    ClosureCall,
    ClosureArity,
    ToString,
    Not,
    Equal,
    NotEqual,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    StringConcat,
}

pub(crate) fn foreign_method(
    class: &str,
    name: &str,
    kind: MethodKind,
    arity: u8,
) -> Option<NativeMethod> {
    use MethodKind::{Getter, Method};
    use NativeMethod::*;
    Some(match (class, name, kind, arity) {
        ("System", "writeString_", Method, 1) => WriteString,
        ("System", "clock", Getter, 0) => Clock,
        ("System", "gc", Method, 0) => CollectGarbage,
        ("Fn", "new", Method, 1) => FnNew,
        _ => return None,
    })
}

/// Default Object behavior, also inherited by user-defined classes.
pub(crate) fn object_method(name: &str, kind: MethodKind, arity: u8) -> Option<NativeMethod> {
    use MethodKind::{Getter, Method};
    use NativeMethod::*;
    Some(match (name, kind, arity) {
        ("toString", Getter, 0) => ToString,
        ("!", Method, 0) => Not,
        ("==", Method, 1) => Equal,
        ("!=", Method, 1) => NotEqual,
        _ => return None,
    })
}

pub(crate) fn instance_method(
    class: BuiltinClass,
    name: &str,
    kind: MethodKind,
    arity: u8,
) -> Option<NativeMethod> {
    use BuiltinClass::*;
    use MethodKind::{Getter, Method};
    use NativeMethod::*;
    Some(match (class, name, kind, arity) {
        (Num, "-", Method, 0) => Negate,
        (Num, "+", Method, 1) => Add,
        (Num, "-", Method, 1) => Subtract,
        (Num, "*", Method, 1) => Multiply,
        (Num, "/", Method, 1) => Divide,
        (Num, "%", Method, 1) => Remainder,
        (Num, "<", Method, 1) => Less,
        (Num, "<=", Method, 1) => LessEqual,
        (Num, ">", Method, 1) => Greater,
        (Num, ">=", Method, 1) => GreaterEqual,
        (String, "+", Method, 1) => StringConcat,
        (Closure, "call", Method, 0..=16) => ClosureCall,
        (Closure, "arity", Getter, 0) => ClosureArity,
        _ => return object_method(name, kind, arity),
    })
}
