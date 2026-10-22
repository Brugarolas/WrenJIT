//! Wren core runtime and engine-provided primitives.
//!
//! The frontend/lowerer reserve the standard core globals in every module. The
//! interpreter materializes those globals as builtin class objects. This module
//! owns the method/primitive manifest and the native implementations.

use crate::wren_interpreter::{
    num_to_u32, BuiltinClass, Host, HostContext, Interpreter, NativeBuiltin, RuntimeError, Value,
};
use crate::wren_ir::{Builder, BuiltinId, CmpKind, FuncId, FunctionKind, Inst, IntBinOp, MethodKind, MethodSymbol, PrimitiveId, Program, Type};

use std::io::{self, Write};
use std::hash::BuildHasher;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug)]
pub enum BuiltinTarget {
    Instance(BuiltinClass),
    Static(BuiltinId),
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveBinding {
    pub target: BuiltinTarget,
    pub method: MethodSymbol,
    pub primitive: PrimitiveId,
}

#[derive(Clone, Debug, Default)]
pub struct BuiltinManifest {
    bindings: Vec<PrimitiveBinding>,
    native_bindings: Vec<NativeBinding>,
    primitives: Vec<Option<BuiltinPrimitive>>,
    list_remove: Option<(MethodSymbol, FuncId)>,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeBinding {
    pub target: BuiltinTarget,
    pub method: MethodSymbol,
    pub native: NativeBuiltin,
}

impl BuiltinManifest {
    pub fn bindings(&self) -> &[PrimitiveBinding] {
        &self.bindings
    }

    fn register(
        &mut self,
        program: &mut Program,
        target: BuiltinTarget,
        name: &str,
        kind: MethodKind,
        arity: u8,
        implementation: BuiltinPrimitive,
    ) -> MethodSymbol {
        let method = program.intern_method(name, kind, arity);
        let primitive = program.fresh_primitive();
        let index = primitive.0 as usize;
        if self.primitives.len() <= index {
            self.primitives.resize(index + 1, None);
        }
        self.primitives[index] = Some(implementation);
        self.bindings.push(PrimitiveBinding {
            target,
            method,
            primitive,
        });
        method
    }

    fn register_native(
        &mut self,
        program: &mut Program,
        target: BuiltinTarget,
        name: &str,
        kind: MethodKind,
        arity: u8,
        native: NativeBuiltin,
    ) -> MethodSymbol {
        let method = program.intern_method(name, kind, arity);
        self.native_bindings.push(NativeBinding { target, method, native });
        method
    }

    fn implementation(&self, primitive: PrimitiveId) -> Option<BuiltinPrimitive> {
        self.primitives
            .get(primitive.0 as usize)
            .copied()
            .flatten()
    }
}

#[derive(Clone, Copy, Debug)]
enum BuiltinPrimitive {
    RandomNew,
    RandomInt,
    RangeCount,
    ListIsEmpty,
    MapIsEmpty,
    StringIsEmpty,
    MapKeysIsEmpty,
    ObjectEq,
    ObjectNe,
    ObjectToString,
    ObjectType,

    BoolNot,
    BoolToString,
    NullNot,
    NullToString,

    NumAdd,
    NumSub,
    NumMul,
    NumDiv,
    NumMod,
    NumInclusiveRange,
    NumExclusiveRange,
    NumLt,
    NumLe,
    NumGt,
    NumGe,
    NumEq,
    NumNe,
    NumNeg,
    NumBitNot,
    NumBitAnd,
    NumBitOr,
    NumBitXor,
    NumShl,
    NumShr,
    NumAbs,
    NumFloor,
    NumCeil,
    NumRound,
    NumSqrt,
    NumSin,
    NumCos,
    NumTan,
    NumLog,
    NumExp,
    NumIsInteger,
    NumIsNan,
    NumIsInfinity,
    NumToString,
    NumFromString,
    NumPi,
    NumTau,
    NumInfinity,
    NumNan,

    StringCount,
    StringAdd,
    StringEq,
    StringNe,
    StringSubscript,
    StringContains,
    StringStartsWith,
    StringEndsWith,
    StringIndexOf,
    StringToString,
    StringIterate,
    StringIteratorValue,
    StringFromCodePoint,

    ListCount,
    ListConcat,
    ListAdd,
    ListInsert,
    ListRemoveAt,
    ListClear,
    ListSubscript,
    ListSubscriptSet,
    ListIterate,
    ListIteratorValue,
    ListToString,
    ListFilled,

    MapCount,
    MapContainsKey,
    MapSubscript,
    MapSubscriptSet,
    MapRemove,
    MapClear,
    MapIterate,
    MapIteratorValue,
    MapKeys,
    MapKeysCount,
    MapKeysIterate,
    MapKeysIteratorValue,
    MapEntryKey,
    MapEntryValue,
    MapToString,

    RangeFrom,
    RangeTo,
    RangeMin,
    RangeMax,
    RangeInclusive,
    RangeIterate,
    RangeIteratorValue,
    RangeToString,

    SystemPrint,
    SystemWrite,
    SystemClock,

    DateNow,
    DateNowMs,
    DateSleep,
}

/// Reserves all method symbols and primitive IDs required by the core runtime.
///
/// Call this before lowering user code into the same `Program`; interning then
/// guarantees that user `Invoke` nodes refer to exactly the same MethodSymbols.
pub fn install_hir_builtins(program: &mut Program) -> BuiltinManifest {
    let mut manifest = BuiltinManifest::default();

    let instance = |class| BuiltinTarget::Instance(class);
    let static_ = |class| BuiltinTarget::Static(class);

    // Object protocol. Bind these to each concrete builtin receiver class below
    // instead of adding inheritance checks to the interpreter hot lookup path.
    for class in [
        BuiltinClass::Null,
        BuiltinClass::Bool,
        BuiltinClass::Num,
        BuiltinClass::String,
        BuiltinClass::List,
        BuiltinClass::Map,
        BuiltinClass::Range,
        BuiltinClass::Closure,
        BuiltinClass::Fiber,
        BuiltinClass::MapEntry,
        BuiltinClass::MapKeys,
        BuiltinClass::Random,
    ] {
        manifest.register(program, instance(class), "==", MethodKind::Method, 1, BuiltinPrimitive::ObjectEq);
        manifest.register(program, instance(class), "!=", MethodKind::Method, 1, BuiltinPrimitive::ObjectNe);
        manifest.register(program, instance(class), "toString", MethodKind::Getter, 0, BuiltinPrimitive::ObjectToString);
    }

    // Null and Bool.
    manifest.register(program, instance(BuiltinClass::Null), "!", MethodKind::Method, 0, BuiltinPrimitive::NullNot);
    manifest.register(program, instance(BuiltinClass::Null), "toString", MethodKind::Getter, 0, BuiltinPrimitive::NullToString);
    manifest.register(program, instance(BuiltinClass::Bool), "!", MethodKind::Method, 0, BuiltinPrimitive::BoolNot);
    manifest.register(program, instance(BuiltinClass::Bool), "toString", MethodKind::Getter, 0, BuiltinPrimitive::BoolToString);

    // Num instance methods and operators.
    let n = instance(BuiltinClass::Num);
    for (name, op) in [
        ("+", BuiltinPrimitive::NumAdd), ("-", BuiltinPrimitive::NumSub),
        ("*", BuiltinPrimitive::NumMul), ("/", BuiltinPrimitive::NumDiv),
        ("%", BuiltinPrimitive::NumMod), ("<", BuiltinPrimitive::NumLt),
        ("<=", BuiltinPrimitive::NumLe), (">", BuiltinPrimitive::NumGt),
        (">=", BuiltinPrimitive::NumGe), ("==", BuiltinPrimitive::NumEq),
        ("!=", BuiltinPrimitive::NumNe), ("&", BuiltinPrimitive::NumBitAnd),
        ("|", BuiltinPrimitive::NumBitOr), ("^", BuiltinPrimitive::NumBitXor),
        ("<<", BuiltinPrimitive::NumShl), (">>", BuiltinPrimitive::NumShr),
        ("..", BuiltinPrimitive::NumInclusiveRange),
        ("...", BuiltinPrimitive::NumExclusiveRange),
    ] {
        manifest.register(program, n, name, MethodKind::Method, 1, op);
    }
    manifest.register(program, n, "-", MethodKind::Method, 0, BuiltinPrimitive::NumNeg);
    manifest.register(program, n, "~", MethodKind::Method, 0, BuiltinPrimitive::NumBitNot);
    for (name, op) in [
        ("abs", BuiltinPrimitive::NumAbs), ("floor", BuiltinPrimitive::NumFloor),
        ("ceil", BuiltinPrimitive::NumCeil), ("round", BuiltinPrimitive::NumRound),
        ("sqrt", BuiltinPrimitive::NumSqrt), ("sin", BuiltinPrimitive::NumSin),
        ("cos", BuiltinPrimitive::NumCos), ("tan", BuiltinPrimitive::NumTan),
        ("log", BuiltinPrimitive::NumLog), ("exp", BuiltinPrimitive::NumExp),
        ("isInteger", BuiltinPrimitive::NumIsInteger), ("isNan", BuiltinPrimitive::NumIsNan),
        ("isInfinity", BuiltinPrimitive::NumIsInfinity), ("toString", BuiltinPrimitive::NumToString),
    ] {
        manifest.register(program, n, name, MethodKind::Getter, 0, op);
    }
    let ns = static_(BuiltinId::Num);
    manifest.register(program, ns, "fromString", MethodKind::Method, 1, BuiltinPrimitive::NumFromString);
    manifest.register(program, ns, "pi", MethodKind::Getter, 0, BuiltinPrimitive::NumPi);
    manifest.register(program, ns, "tau", MethodKind::Getter, 0, BuiltinPrimitive::NumTau);
    manifest.register(program, ns, "infinity", MethodKind::Getter, 0, BuiltinPrimitive::NumInfinity);
    manifest.register(program, ns, "nan", MethodKind::Getter, 0, BuiltinPrimitive::NumNan);

    // String.
    let st = instance(BuiltinClass::String);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::StringCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::StringIsEmpty),
        ("+", MethodKind::Method, 1, BuiltinPrimitive::StringAdd),
        ("==", MethodKind::Method, 1, BuiltinPrimitive::StringEq),
        ("!=", MethodKind::Method, 1, BuiltinPrimitive::StringNe),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::StringSubscript),
        ("contains", MethodKind::Method, 1, BuiltinPrimitive::StringContains),
        ("startsWith", MethodKind::Method, 1, BuiltinPrimitive::StringStartsWith),
        ("endsWith", MethodKind::Method, 1, BuiltinPrimitive::StringEndsWith),
        ("indexOf", MethodKind::Method, 1, BuiltinPrimitive::StringIndexOf),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::StringToString),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::StringIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::StringIteratorValue),
    ] {
        manifest.register(program, st, name, kind, arity, op);
    }
    manifest.register(program, static_(BuiltinId::String), "fromCodePoint", MethodKind::Method, 1, BuiltinPrimitive::StringFromCodePoint);

    // List.
    let list = instance(BuiltinClass::List);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::ListCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::ListIsEmpty),
        ("+", MethodKind::Method, 1, BuiltinPrimitive::ListConcat),
        ("add", MethodKind::Method, 1, BuiltinPrimitive::ListAdd),
        ("insert", MethodKind::Method, 2, BuiltinPrimitive::ListInsert),
        ("removeAt", MethodKind::Method, 1, BuiltinPrimitive::ListRemoveAt),
        ("clear", MethodKind::Method, 0, BuiltinPrimitive::ListClear),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::ListSubscript),
        ("[]=", MethodKind::SubscriptSetter, 2, BuiltinPrimitive::ListSubscriptSet),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::ListIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::ListIteratorValue),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::ListToString),
    ] {
        manifest.register(program, list, name, kind, arity, op);
    }
    manifest.register(program, static_(BuiltinId::List), "filled", MethodKind::Method, 2, BuiltinPrimitive::ListFilled);
    let remove = program.intern_method("remove", MethodKind::Method, 1);
    let equals = program.intern_method("==", MethodKind::Method, 1);
    let module = program.add_module("<core:list>", 0);
    let mut b = Builder::new(program, module, "List.remove", FunctionKind::Method,
        vec![Type::List(None), Type::Top], Type::Top);
    let list = b.param(0);
    let value = b.param(1);
    let header = b.new_block(&[]);
    let index = b.add_block_param(header, Type::Int32);
    let body = b.new_block(&[]);
    let found = b.new_block(&[]);
    let advance = b.new_block(&[]);
    let missing = b.new_block(&[]);
    let zero = b.const_i32(0);
    b.jump(header, vec![zero]);
    b.position_at(header);
    let len = b.emit1(Inst::ListLen { list }, Type::Int32);
    let more = b.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: index, right: len }, Type::Bool);
    b.branch(more, body, vec![], missing, vec![]);
    b.position_at(body);
    let item = b.emit1(Inst::ListGet { list, index }, Type::Top);
    // Dispatch == normally: user-defined equality can execute Wren and trigger GC.
    let equal = b.invoke(item, equals, vec![value]);
    b.branch(equal, found, vec![], advance, vec![]);
    b.position_at(found);
    let removed = b.emit1(Inst::ListRemoveAt { list, index }, Type::Top);
    b.ret(removed);
    b.position_at(advance);
    let one = b.const_i32(1);
    let next = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: index, right: one }, Type::Int32);
    b.jump(header, vec![next]);
    b.position_at(missing);
    let null = b.const_null();
    b.ret(null);
    manifest.list_remove = Some((remove, b.finish()));

    // Map.
    let map = instance(BuiltinClass::Map);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::MapCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::MapIsEmpty),
        ("containsKey", MethodKind::Method, 1, BuiltinPrimitive::MapContainsKey),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::MapSubscript),
        ("[]=", MethodKind::SubscriptSetter, 2, BuiltinPrimitive::MapSubscriptSet),
        ("remove", MethodKind::Method, 1, BuiltinPrimitive::MapRemove),
        ("clear", MethodKind::Method, 0, BuiltinPrimitive::MapClear),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::MapIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::MapIteratorValue),
        ("keys", MethodKind::Getter, 0, BuiltinPrimitive::MapKeys),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::MapToString),
    ] {
        manifest.register(program, map, name, kind, arity, op);
    }

    // Each additional subscript descends into a nested List or Map. Setters
    // reserve one explicit argument for the value within the 16-argument ABI.
    for (target, getter, setter) in [
        (instance(BuiltinClass::List), BuiltinPrimitive::ListSubscript, BuiltinPrimitive::ListSubscriptSet),
        (map, BuiltinPrimitive::MapSubscript, BuiltinPrimitive::MapSubscriptSet),
    ] {
        for arity in 2..=crate::wren_front::MAX_PARAMETERS as u8 {
            manifest.register(program, target, "[]", MethodKind::SubscriptGetter, arity, getter);
            if arity < crate::wren_front::MAX_PARAMETERS as u8 {
                manifest.register(program, target, "[]=", MethodKind::SubscriptSetter, arity + 1, setter);
            }
        }
    }

    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::MapKeysCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::MapKeysIsEmpty),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::MapKeysIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::MapKeysIteratorValue),
    ] {
        manifest.register(program, instance(BuiltinClass::MapKeys), name, kind, arity, op);
    }
    manifest.register(program, instance(BuiltinClass::MapEntry), "key", MethodKind::Getter, 0, BuiltinPrimitive::MapEntryKey);
    manifest.register(program, instance(BuiltinClass::MapEntry), "value", MethodKind::Getter, 0, BuiltinPrimitive::MapEntryValue);

    // Range.
    let range = instance(BuiltinClass::Range);
    for (name, kind, arity, op) in [
        ("from", MethodKind::Getter, 0, BuiltinPrimitive::RangeFrom),
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::RangeCount),
        ("to", MethodKind::Getter, 0, BuiltinPrimitive::RangeTo),
        ("min", MethodKind::Getter, 0, BuiltinPrimitive::RangeMin),
        ("max", MethodKind::Getter, 0, BuiltinPrimitive::RangeMax),
        ("isInclusive", MethodKind::Getter, 0, BuiltinPrimitive::RangeInclusive),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::RangeIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::RangeIteratorValue),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::RangeToString),
    ] {
        manifest.register(program, range, name, kind, arity, op);
    }

    // Fn.new/arity/call(...) are VM-native because closures own bytecode frame metadata.
    manifest.register_native(program, static_(BuiltinId::Fn), "new", MethodKind::Method, 1, NativeBuiltin::FnNew);
    manifest.register_native(program, instance(BuiltinClass::Closure), "arity", MethodKind::Getter, 0, NativeBuiltin::FnArity);
    for arity in 0..=16u8 {
        manifest.register_native(
            program,
            instance(BuiltinClass::Closure),
            "call",
            MethodKind::Method,
            arity,
            NativeBuiltin::FnCall,
        );
    }

    // Fiber operations manipulate the VM scheduler/frame graph and therefore
    // deliberately bypass the Host primitive ABI.
    manifest.register_native(program, static_(BuiltinId::Fiber), "current", MethodKind::Getter, 0, NativeBuiltin::FiberCurrent);
    manifest.register_native(program, static_(BuiltinId::Fiber), "new", MethodKind::Method, 1, NativeBuiltin::FiberNew);
    manifest.register_native(program, instance(BuiltinClass::Fiber), "error", MethodKind::Getter, 0, NativeBuiltin::FiberError);
    manifest.register_native(program, instance(BuiltinClass::Fiber), "isDone", MethodKind::Getter, 0, NativeBuiltin::FiberIsDone);
    for arity in 0..=1u8 {
        manifest.register_native(program, instance(BuiltinClass::Fiber), "call", MethodKind::Method, arity, NativeBuiltin::FiberCall);
        manifest.register_native(program, instance(BuiltinClass::Fiber), "try", MethodKind::Method, arity, NativeBuiltin::FiberTry);
        manifest.register_native(program, instance(BuiltinClass::Fiber), "transfer", MethodKind::Method, arity, NativeBuiltin::FiberTransfer);
    }
    manifest.register_native(program, instance(BuiltinClass::Fiber), "transferError", MethodKind::Method, 1, NativeBuiltin::FiberTransferError);
    manifest.register_native(program, static_(BuiltinId::Fiber), "yield", MethodKind::Method, 0, NativeBuiltin::FiberYield);
    manifest.register_native(program, static_(BuiltinId::Fiber), "yield", MethodKind::Method, 1, NativeBuiltin::FiberYield);
    manifest.register_native(program, static_(BuiltinId::Fiber), "suspend", MethodKind::Method, 0, NativeBuiltin::FiberSuspend);
    manifest.register_native(program, static_(BuiltinId::Fiber), "abort", MethodKind::Method, 1, NativeBuiltin::FiberAbort);

    // System and Date are static-only engine classes.
    let system = static_(BuiltinId::System);
    manifest.register(program, system, "print", MethodKind::Method, 0, BuiltinPrimitive::SystemPrint);
    manifest.register(program, system, "print", MethodKind::Method, 1, BuiltinPrimitive::SystemPrint);
    manifest.register(program, system, "write", MethodKind::Method, 1, BuiltinPrimitive::SystemWrite);
    manifest.register(program, system, "clock", MethodKind::Getter, 0, BuiltinPrimitive::SystemClock);
    manifest.register_native(program, system, "gc", MethodKind::Method, 0, NativeBuiltin::SystemGc);

    let date = static_(BuiltinId::Date);
    manifest.register(program, date, "now", MethodKind::Getter, 0, BuiltinPrimitive::DateNow);
    manifest.register(program, date, "nowMs", MethodKind::Getter, 0, BuiltinPrimitive::DateNowMs);
    manifest.register(program, date, "sleep", MethodKind::Method, 1, BuiltinPrimitive::DateSleep);

    manifest.register(program, static_(BuiltinId::Random), "new", MethodKind::Method, 0, BuiltinPrimitive::RandomNew);
    for arity in [1, 2] {
        manifest.register(program, instance(BuiltinClass::Random), "int", MethodKind::Method, arity, BuiltinPrimitive::RandomInt);
    }

    manifest
}

/// Connects the already-reserved manifest to a concrete interpreter.
pub fn install_vm_builtins(interpreter: &mut Interpreter<'_>, manifest: &BuiltinManifest) {
    if let Some((method, function)) = manifest.list_remove {
        interpreter.bind_builtin_function(BuiltinClass::List, method, function);
    }
    for binding in manifest.bindings() {
        match binding.target {
            BuiltinTarget::Instance(class) => {
                interpreter.bind_builtin_primitive(class, binding.method, binding.primitive);
            }
            BuiltinTarget::Static(class) => {
                interpreter.bind_builtin_static_primitive(class, binding.method, binding.primitive);
            }
        }
    }
    for binding in &manifest.native_bindings {
        match binding.target {
            BuiltinTarget::Instance(class) => {
                interpreter.bind_builtin_native(class, binding.method, binding.native);
            }
            BuiltinTarget::Static(class) => {
                interpreter.bind_builtin_static_native(class, binding.method, binding.native);
            }
        }
    }
}

pub struct BuiltinHost {
    manifest: BuiltinManifest,
    started: Instant,
}

impl BuiltinHost {
    pub fn new(manifest: BuiltinManifest) -> Self {
        Self {
            manifest,
            started: Instant::now(),
        }
    }
}

impl Host for BuiltinHost {
    /// Tell the tracing JIT which primitive bindings have the exact stock Wren
    /// numeric semantics that it is allowed to inline.  The JIT must not
    /// specialize merely by method name because embedders may replace primitive
    /// implementations.
    fn standard_numeric_method(&self, primitive: PrimitiveId, name: &str, arity: u8) -> bool {
        use BuiltinPrimitive::*;
        matches!(
            (self.manifest.implementation(primitive), name, arity),
            (Some(NumNeg), "-", 0)
                | (Some(NumAdd), "+", 1)
                | (Some(NumSub), "-", 1)
                | (Some(NumMul), "*", 1)
                | (Some(NumDiv), "/", 1)
                | (Some(NumMod), "%", 1)
                | (Some(NumLt), "<", 1)
                | (Some(NumLe), "<=", 1)
                | (Some(NumGt), ">", 1)
                | (Some(NumGe), ">=", 1)
                | (Some(NumEq), "==", 1)
                | (Some(NumNe), "!=", 1)
                | (Some(NumBitNot), "~", 0)
                | (Some(NumBitAnd), "&", 1)
                | (Some(NumBitOr), "|", 1)
                | (Some(NumBitXor), "^", 1)
                | (Some(NumShl), "<<", 1)
                | (Some(NumShr), ">>", 1)
                | (Some(NumAbs), "abs", 0)
                | (Some(NumFloor), "floor", 0)
                | (Some(NumCeil), "ceil", 0)
                | (Some(NumSqrt), "sqrt", 0)
                | (Some(NumRound), "round", 0)
                | (Some(NumSin), "sin", 0)
                | (Some(NumCos), "cos", 0)
                | (Some(NumTan), "tan", 0)
                | (Some(NumLog), "log", 0)
                | (Some(NumExp), "exp", 0)
                | (Some(NumIsNan), "isNan", 0)
                | (Some(NumIsInfinity), "isInfinity", 0)
                | (Some(NumIsInteger), "isInteger", 0)
        )
    }

    fn call_primitive(
        &mut self,
        context: &mut HostContext<'_>,
        primitive: PrimitiveId,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        let implementation = self
            .manifest
            .implementation(primitive)
            .ok_or_else(|| RuntimeError::new(format!("unregistered builtin primitive {primitive:?}")))?;
        execute_primitive(implementation, context, args, self.started)
    }
}

#[inline]
fn receiver(args: &[Value]) -> Result<Value, RuntimeError> {
    args.first().copied().ok_or_else(|| RuntimeError::new("builtin call has no receiver"))
}

#[inline]
fn argument(args: &[Value], index: usize) -> Result<Value, RuntimeError> {
    args.get(index + 1).copied().ok_or_else(|| RuntimeError::new("builtin argument missing"))
}

#[inline]
fn num(value: Value, what: &str) -> Result<f64, RuntimeError> {
    value.as_num().ok_or_else(|| RuntimeError::new(format!("{what} must be a Num")))
}

#[inline]
fn integer(value: Value, what: &str) -> Result<i64, RuntimeError> {
    let n = num(value, what)?;
    if !n.is_finite() || n.fract() != 0.0 {
        return Err(RuntimeError::new(format!("{what} must be an integer")));
    }
    Ok(n as i64)
}

#[inline]
fn bitwise_operand(value: Value, what: &str) -> Result<u32, RuntimeError> {
    Ok(num_to_u32(num(value, what)?))
}

fn normalized_index(index: i64, len: usize) -> Result<usize, RuntimeError> {
    let len_i = len as i64;
    let index = if index < 0 { len_i + index } else { index };
    if index < 0 || index >= len_i {
        Err(RuntimeError::new("index out of bounds"))
    } else {
        Ok(index as usize)
    }
}

/// Resolve the containing List or Map, leaving the final subscript for the
/// getter or setter. Intermediate list indices cannot be slices; map keys
/// retain their usual semantics regardless of the depth of the lookup.
fn nested_subscript_target(
    context: &HostContext<'_>,
    mut container: Value,
    indices: &[Value],
) -> Result<(Value, Value), RuntimeError> {
    let (&last, path) = indices.split_last()
        .ok_or_else(|| RuntimeError::new("builtin argument missing"))?;
    for &subscript in path {
        container = if let Some(len) = context.list_len(container) {
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_get(container, index)
                .ok_or_else(|| RuntimeError::new("list index out of bounds"))?
        } else if context.map_len(container).is_some() {
            context.map_get(container, subscript)?
        } else {
            return Err(RuntimeError::new("receiver is not List or Map"));
        };
    }
    Ok((container, last))
}

fn slice_bounds(
    from: f64,
    to: f64,
    inclusive: bool,
    len: usize,
) -> Result<(usize, usize, bool), RuntimeError> {
    let length = len as f64;
    // Wren permits an empty copy at the end, including copies of an empty list.
    if from == length && to == if inclusive { -1.0 } else { length } {
        return Ok((0, 0, false));
    }
    if !from.is_finite() || from.fract() != 0.0 {
        return Err(RuntimeError::new("range start must be an integer"));
    }
    let start = if from < 0.0 { length + from } else { from };
    if start < 0.0 || start >= length {
        return Err(RuntimeError::new("range start out of bounds"));
    }
    if !to.is_finite() || to.fract() != 0.0 {
        return Err(RuntimeError::new("range end must be an integer"));
    }
    let mut end = if to < 0.0 { length + to } else { to };
    if !inclusive {
        if start == end {
            return Ok((start as usize, 0, false));
        }
        end += if end >= start { -1.0 } else { 1.0 };
    }
    if end < 0.0 || end >= length {
        return Err(RuntimeError::new("range end out of bounds"));
    }
    let start = start as usize;
    let end = end as usize;
    Ok((start, start.abs_diff(end) + 1, end < start))
}

fn execute_primitive(
    op: BuiltinPrimitive,
    context: &mut HostContext<'_>,
    args: &[Value],
    started: Instant,
) -> Result<Value, RuntimeError> {
    use BuiltinPrimitive::*;

    let this = receiver(args)?;
    match op {
        RandomNew => {
            let seed = std::collections::hash_map::RandomState::new().hash_one(());
            Ok(context.allocate_random(seed))
        }
        RandomInt => {
            let (from, to) = if args.len() == 2 {
                (0.0, num(argument(args, 0)?, "upper bound")?)
            } else {
                (num(argument(args, 0)?, "lower bound")?, num(argument(args, 1)?, "upper bound")?)
            };
            const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
            for bound in [from, to] {
                if !bound.is_finite() || bound.fract() != 0.0 || bound.abs() > MAX_SAFE_INTEGER {
                    return Err(RuntimeError::new("Random.int bounds must be safe finite integers"));
                }
            }
            if from >= to {
                return Err(RuntimeError::new("Random.int upper bound must be greater than lower bound"));
            }
            let lower = from as i64;
            let width = (to as i64 - lower) as u64;
            // Rejection avoids the bias of reducing every u64 modulo width.
            let threshold = width.wrapping_neg() % width;
            loop {
                let value = context.random_u64(this)?;
                if value >= threshold {
                    return Ok(Value::from_num((lower + (value % width) as i64) as f64));
                }
            }
        }
        RangeCount => {
            let (from, to, inclusive) = context.range_parts(this)
                .ok_or_else(|| RuntimeError::new("receiver is not Range"))?;
            let distance = (to - from).abs();
            Ok(Value::from_num(if inclusive { distance.floor() + 1.0 } else { distance.ceil() }))
        }
        ListIsEmpty => Ok(Value::from_bool(context.list_len(this)
            .ok_or_else(|| RuntimeError::new("receiver is not List"))? == 0)),
        MapIsEmpty => Ok(Value::from_bool(context.map_len(this)
            .ok_or_else(|| RuntimeError::new("receiver is not Map"))? == 0)),
        StringIsEmpty => Ok(Value::from_bool(context.string_bytes(this)
            .ok_or_else(|| RuntimeError::new("receiver is not String"))?.is_empty())),
        MapKeysIsEmpty => {
            let map = context.map_keys_source(this)?;
            Ok(Value::from_bool(context.map_len(map)
                .ok_or_else(|| RuntimeError::new("receiver is not Map"))? == 0))
        }
        ObjectEq => Ok(Value::from_bool(this.raw_bits() == argument(args, 0)?.raw_bits())),
        ObjectNe => Ok(Value::from_bool(this.raw_bits() != argument(args, 0)?.raw_bits())),
        ObjectToString | BoolToString | NullToString | NumToString | ListToString | MapToString | RangeToString => {
            Ok(context.allocate_string(context.format_value(this)))
        }
        ObjectType => Ok(Value::NULL),
        BoolNot => Ok(Value::from_bool(!this.as_bool().ok_or_else(|| RuntimeError::new("receiver is not Bool"))?)),
        NullNot => Ok(Value::TRUE),

        NumAdd => Ok(Value::from_num(num(this, "receiver")? + num(argument(args, 0)?, "argument")?)),
        NumSub => Ok(Value::from_num(num(this, "receiver")? - num(argument(args, 0)?, "argument")?)),
        NumMul => Ok(Value::from_num(num(this, "receiver")? * num(argument(args, 0)?, "argument")?)),
        NumDiv => Ok(Value::from_num(num(this, "receiver")? / num(argument(args, 0)?, "argument")?)),
        NumMod => Ok(Value::from_num(num(this, "receiver")? % num(argument(args, 0)?, "argument")?)),
        NumInclusiveRange | NumExclusiveRange => {
            let from = num(this, "range start")?;
            let to = num(argument(args, 0)?, "range end")?;
            Ok(context.allocate_range(from, to, matches!(op, NumInclusiveRange)))
        }
        NumLt => Ok(Value::from_bool(num(this, "receiver")? < num(argument(args, 0)?, "argument")?)),
        NumLe => Ok(Value::from_bool(num(this, "receiver")? <= num(argument(args, 0)?, "argument")?)),
        NumGt => Ok(Value::from_bool(num(this, "receiver")? > num(argument(args, 0)?, "argument")?)),
        NumGe => Ok(Value::from_bool(num(this, "receiver")? >= num(argument(args, 0)?, "argument")?)),
        NumEq => Ok(Value::from_bool(argument(args, 0)?.as_num().is_some_and(|rhs| num(this, "receiver").unwrap() == rhs))),
        NumNe => Ok(Value::from_bool(!argument(args, 0)?.as_num().is_some_and(|rhs| num(this, "receiver").unwrap() == rhs))),
        NumNeg => Ok(Value::from_num(-num(this, "receiver")?)),
        NumBitNot => Ok(Value::from_num((!bitwise_operand(this, "receiver")?) as f64)),
        NumBitAnd => Ok(Value::from_num((bitwise_operand(this, "receiver")? & bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumBitOr => Ok(Value::from_num((bitwise_operand(this, "receiver")? | bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumBitXor => Ok(Value::from_num((bitwise_operand(this, "receiver")? ^ bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumShl => Ok(Value::from_num(bitwise_operand(this, "receiver")?.wrapping_shl(bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumShr => Ok(Value::from_num(bitwise_operand(this, "receiver")?.wrapping_shr(bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumAbs => Ok(Value::from_num(num(this, "receiver")?.abs())),
        NumFloor => Ok(Value::from_num(num(this, "receiver")?.floor())),
        NumCeil => Ok(Value::from_num(num(this, "receiver")?.ceil())),
        NumRound => Ok(Value::from_num(num(this, "receiver")?.round())),
        NumSqrt => Ok(Value::from_num(num(this, "receiver")?.sqrt())),
        NumSin => Ok(Value::from_num(num(this, "receiver")?.sin())),
        NumCos => Ok(Value::from_num(num(this, "receiver")?.cos())),
        NumTan => Ok(Value::from_num(num(this, "receiver")?.tan())),
        NumLog => Ok(Value::from_num(num(this, "receiver")?.ln())),
        NumExp => Ok(Value::from_num(num(this, "receiver")?.exp())),
        NumIsInteger => { let n = num(this, "receiver")?; Ok(Value::from_bool(n.is_finite() && n.fract() == 0.0)) }
        NumIsNan => Ok(Value::from_bool(num(this, "receiver")?.is_nan())),
        NumIsInfinity => Ok(Value::from_bool(num(this, "receiver")?.is_infinite())),
        NumFromString => {
            let value = argument(args, 0)?;
            let text = context.string_utf8(value).ok_or_else(|| RuntimeError::new("Num.fromString expects String"))?;
            Ok(text.trim().parse::<f64>().map(Value::from_num).unwrap_or(Value::NULL))
        }
        NumPi => Ok(Value::from_num(std::f64::consts::PI)),
        NumTau => Ok(Value::from_num(std::f64::consts::TAU)),
        NumInfinity => Ok(Value::from_num(f64::INFINITY)),
        NumNan => Ok(Value::from_num(f64::NAN)),

        StringCount => Ok(Value::from_num(context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?.len() as f64)),
        StringAdd => {
            let rhs = argument(args, 0)?;
            let mut out = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?.to_vec();
            out.extend_from_slice(context.string_bytes(rhs).ok_or_else(|| RuntimeError::new("String.+ expects String"))?);
            Ok(context.allocate_bytes(out))
        }
        StringEq | StringNe => {
            let rhs = argument(args, 0)?;
            let equal = context.string_bytes(this).zip(context.string_bytes(rhs)).is_some_and(|(a,b)| a == b);
            Ok(Value::from_bool(if matches!(op, StringEq) { equal } else { !equal }))
        }
        StringSubscript => {
            let byte = {
                let bytes = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
                let index = normalized_index(integer(argument(args, 0)?, "index")?, bytes.len())?;
                bytes[index]
            };
            Ok(context.allocate_bytes([byte]))
        }
        StringContains | StringStartsWith | StringEndsWith | StringIndexOf => {
            let haystack = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
            let needle = context.string_bytes(argument(args, 0)?).ok_or_else(|| RuntimeError::new("argument must be String"))?;
            let position = haystack.windows(needle.len()).position(|w| w == needle);
            match op {
                StringContains => Ok(Value::from_bool(position.is_some())),
                StringStartsWith => Ok(Value::from_bool(haystack.starts_with(needle))),
                StringEndsWith => Ok(Value::from_bool(haystack.ends_with(needle))),
                StringIndexOf => Ok(Value::from_num(position.map_or(-1.0, |p| p as f64))),
                _ => unreachable!(),
            }
        }
        StringToString => Ok(this),
        StringIterate => {
            let len = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?.len();
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        StringIteratorValue => {
            let byte = {
                let bytes = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
                let index = normalized_index(integer(argument(args,0)?, "iterator")?, bytes.len())?;
                bytes[index]
            };
            Ok(context.allocate_bytes([byte]))
        }
        StringFromCodePoint => {
            let point = integer(argument(args, 0)?, "code point")?;
            let ch = char::from_u32(point as u32).ok_or_else(|| RuntimeError::new("invalid Unicode code point"))?;
            let mut bytes = [0u8; 4];
            Ok(context.allocate_string(ch.encode_utf8(&mut bytes)))
        }

        ListCount => Ok(Value::from_num(context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))? as f64)),
        ListConcat => {
            let rhs = argument(args, 0)?;
            let left_len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let right_len = context.list_len(rhs).ok_or_else(|| RuntimeError::new("List.+ expects List"))?;
            let capacity = left_len.checked_add(right_len)
                .ok_or_else(|| RuntimeError::new("concatenated list is too large"))?;
            let mut values = Vec::with_capacity(capacity);
            for (list, len) in [(this, left_len), (rhs, right_len)] {
                for index in 0..len {
                    values.push(context.list_get(list, index)
                        .ok_or_else(|| RuntimeError::new("list index out of bounds"))?);
                }
            }
            Ok(context.allocate_list(values))
        }
        ListAdd => context.list_push(this, argument(args,0)?),
        ListInsert => {
            let index = integer(argument(args,0)?, "index")?;
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = if index < 0 { (len as i64 + index + 1).max(0) } else { index } as usize;
            context.list_insert(this, index, argument(args,1)?)
        }
        ListRemoveAt => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = normalized_index(integer(argument(args,0)?, "index")?, len)?;
            context.list_remove_at(this, index)
        }
        ListClear => { context.list_clear(this)?; Ok(Value::NULL) }
        ListSubscript | MapSubscript => {
            let (container, subscript) = nested_subscript_target(context, this, &args[1..])?;
            if context.map_len(container).is_some() {
                return context.map_get(container, subscript);
            }
            let len = context.list_len(container).ok_or_else(|| RuntimeError::new("receiver is not List or Map"))?;
            if let Some((from, to, inclusive)) = context.range_parts(subscript) {
                let (start, count, reverse) = slice_bounds(from, to, inclusive, len)?;
                let mut values = Vec::with_capacity(count);
                for offset in 0..count {
                    let index = if reverse { start - offset } else { start + offset };
                    values.push(context.list_get(container, index)
                        .ok_or_else(|| RuntimeError::new("list index out of bounds"))?);
                }
                return Ok(context.allocate_list(values));
            }
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_get(container,index).ok_or_else(|| RuntimeError::new("list index out of bounds"))
        }
        ListSubscriptSet | MapSubscriptSet => {
            let (&value, indices) = args[1..].split_last()
                .ok_or_else(|| RuntimeError::new("builtin argument missing"))?;
            let (container, subscript) = nested_subscript_target(context, this, indices)?;
            if context.map_len(container).is_some() {
                return context.map_set(container, subscript, value);
            }
            let len = context.list_len(container).ok_or_else(|| RuntimeError::new("receiver is not List or Map"))?;
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_set(container,index,value)
        }
        ListIterate => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        ListIteratorValue => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = normalized_index(integer(argument(args,0)?, "iterator")?,len)?;
            context.list_get(this,index).ok_or_else(|| RuntimeError::new("list iterator out of bounds"))
        }
        ListFilled => {
            let count = integer(argument(args,0)?, "count")?;
            if count < 0 || count > 16_777_216 { return Err(RuntimeError::new("invalid List.filled size")); }
            Ok(context.allocate_list(vec![argument(args,1)?; count as usize]))
        }

        MapCount => Ok(Value::from_num(context.map_len(this).ok_or_else(|| RuntimeError::new("receiver is not Map"))? as f64)),
        MapContainsKey => Ok(Value::from_bool(context.map_contains_key(this, argument(args,0)?)?)),
        MapRemove => context.map_remove(this, argument(args,0)?),
        MapClear => { context.map_clear(this)?; Ok(Value::NULL) }
        MapIterate | MapKeysIterate => {
            let map = if matches!(op, MapKeysIterate) { context.map_keys_source(this)? } else { this };
            let len = context.map_len(map).ok_or_else(|| RuntimeError::new("receiver is not Map"))?;
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        MapIteratorValue => context.map_nth_entry(this, integer(argument(args,0)?, "iterator")? as usize),
        MapKeys => context.map_keys(this),
        MapKeysCount => {
            let map = context.map_keys_source(this)?;
            Ok(Value::from_num(context.map_len(map).ok_or_else(|| RuntimeError::new("receiver is not Map"))? as f64))
        }
        MapKeysIteratorValue => {
            let map = context.map_keys_source(this)?;
            context.map_nth_key(map, integer(argument(args,0)?, "iterator")? as usize)
        }
        MapEntryKey => Ok(context.map_entry_parts(this)?.0),
        MapEntryValue => Ok(context.map_entry_parts(this)?.1),

        RangeFrom => Ok(Value::from_num(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.0)),
        RangeTo => Ok(Value::from_num(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.1)),
        RangeMin => { let (a,b,_) = context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?; Ok(Value::from_num(a.min(b))) }
        RangeMax => { let (a,b,_) = context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?; Ok(Value::from_num(a.max(b))) }
        RangeInclusive => Ok(Value::from_bool(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.2)),
        RangeIterate => {
            let (from,to,inclusive)=context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?;
            let current = if argument(args,0)?.is_null() { from } else { num(argument(args,0)?, "iterator")? + if to >= from {1.0}else{-1.0} };
            let valid = if to >= from { if inclusive { current <= to } else { current < to } } else if inclusive { current >= to } else { current > to };
            Ok(if valid { Value::from_num(current) } else { Value::FALSE })
        }
        RangeIteratorValue => Ok(argument(args,0)?),

        SystemPrint => {
            if args.len() == 1 {
                println!();
                Ok(Value::NULL)
            } else {
                let value = argument(args,0)?;
                println!("{}", context.format_value(value));
                Ok(value)
            }
        }
        SystemWrite => {
            print!("{}", context.format_value(argument(args,0)?));
            io::stdout().flush().map_err(|e| RuntimeError::new(format!("stdout write failed: {e}")))?;
            Ok(argument(args,0)?)
        }
        SystemClock => Ok(Value::from_num(started.elapsed().as_secs_f64())),

        DateNow => {
            let seconds = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
            Ok(Value::from_num(seconds))
        }
        DateNowMs => {
            let millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64() * 1000.0;
            Ok(Value::from_num(millis))
        }
        DateSleep => {
            let ms = num(argument(args,0)?, "milliseconds")?;
            if !ms.is_finite() || ms < 0.0 { return Err(RuntimeError::new("Date.sleep expects non-negative milliseconds")); }
            std::thread::sleep(Duration::from_secs_f64(ms / 1000.0));
            Ok(Value::NULL)
        }
    }
}
