//! Differential tests execute generated machine code for every bytecode opcode.
use super::*;
use crate::wren_bytecode::*;
use crate::wren_ir::{ClassInfo, ModuleInfo, MethodSignature, MethodKind, ForeignFunctionInfo, SiteId, DeoptId};

struct TestHost;
impl Host for TestHost {
    fn call_primitive(&mut self, _: &mut HostContext<'_>, _: PrimitiveId, _: &[Value]) -> Result<Value, RuntimeError> { Ok(Value::TRUE) }
    fn call_foreign(&mut self, _: &mut HostContext<'_>, _: ForeignFnId, _: &[Value]) -> Result<Value, RuntimeError> { Ok(Value::FALSE) }
}

fn program_for(opcode: Opcode) -> BytecodeProgram {
    use Opcode::*;
    let mut asm = Assembler::new();
    let (a, b, c) = (0, 1, 2);
    match opcode {
        Invoke | InvokeSuper => { asm.emit_x32(opcode, a, b, 0, 0); },
        CallDirect | MakeClosure => { asm.emit_x32(opcode, a, 0, 0, 1); },
        CallClosure => { asm.emit_x32(opcode, a, b, 0, 0); },
        CallPrimitive => { asm.emit_x32(opcode, a, 0, 0, 0); },
        CallForeign => { asm.emit_call_foreign(Reg(a), &[Reg(b)], ForeignFnId(0)).unwrap(); },
        GuardType | GuardClass | LoadModuleVar | StoreModuleVar => { asm.emit_x32_y32(opcode, a, b, c, 0, 0); },
        BindMethod => { asm.emit_x32_y32(opcode, a, 0, 0, 0, 1); },
        BindForeignMethod => { asm.emit_x32_y32(opcode, a, 0, 0, 0, 0); },
        CreateClass => { asm.emit_x32(opcode, a, b, 0, 0); },
        StringSlice => { asm.emit_x32(opcode, a, b, c, 3); },
        _ => match opcode.fixed_word_len() {
            Some(1) => { asm.emit_abc(opcode, a, b, c); },
            Some(2) => { asm.emit_x32(opcode, a, b, c, 0); },
            _ => panic!("missing encoding for {opcode:?}"),
        },
    }
    let next = asm.chunk.pc();
    asm.emit_a(Return, 0);
    let mut chunk = asm.finish().unwrap();
    chunk.constants.push(Constant::String(StringId(1)));
    chunk.inline_caches.push(InlineCacheDescriptor { site: SiteId(0), method: MethodSymbol(0), home_class: (opcode == InvokeSuper).then_some(ClassId(1)) });
    chunk.type_guards.push(TypeGuardDescriptor { expected: Type::Num });
    chunk.class_guards.push(ClassGuardDescriptor { expected: ClassSet::singleton(ClassId(0)) });
    chunk.deopts.push(BytecodeDeoptPoint { id: DeoptId(0), resume_pc: next, slots: vec![], inlined_frames: vec![] });
    chunk.switch_tables.push(SwitchTable { default_target: next, cases: vec![SwitchTarget { value: 42, target: next }] });
    chunk.profile_counters.push(ProfileCounterDescriptor { kind: if opcode == ProfileNumeric { ProfileCounterKind::Numeric } else { ProfileCounterKind::Block }, site: None });
    if opcode.may_gc() {
        let mut references = ReferenceMap::default();
        for i in 0..8 { references.set_general(Reg(i), true); }
        chunk.safepoints.push(SafepointEntry { pc: BytecodePc(0), kind: SafepointKind::Call, references });
    }
    let mut program = BytecodeProgram::new();
    program.metadata.strings = vec!["main".into(), "héllo".into(), "method".into()];
    program.metadata.modules.push(ModuleInfo { id: ModuleId(0), name: StringId(0), variable_count: 2, builtin_bindings: vec![] });
    program.metadata.methods.push(MethodSignature { name: StringId(2), kind: MethodKind::Method, arity: 0 });
    for id in 0..2 {
        program.metadata.classes.push(ClassInfo { id: ClassId(id), module: ModuleId(0), name: StringId(0), superclass: (id == 1).then_some(ClassId(0)), instance_field_count: 1, static_field_count: 1, is_foreign: opcode == AllocForeign });
    }
    program.metadata.foreign_functions.push(ForeignFunctionInfo { id: ForeignFnId(0), module: StringId(0), class: StringId(0), signature: MethodSymbol(0), is_static: false });
    let frame = FrameLayout { general_register_count: 8, parameters: vec![], captures: vec![], return_kind: RegisterKind::Value };
    program.push_function(BytecodeFunction { id: FuncId(0), module: ModuleId(0), name: StringId(0), kind: FunctionKind::ModuleBody, arity: 0, capture_count: 0, frame: frame.clone(), chunk }).unwrap();
    let mut child = Assembler::new();
    child.emit_a(LoadTrue, 0);
    child.emit_a(Return, 0);
    program.push_function(BytecodeFunction { id: FuncId(1), module: ModuleId(0), name: StringId(0), kind: FunctionKind::Closure, arity: 0, capture_count: 0, frame, chunk: child.finish().unwrap() }).unwrap();
    program.rebuild_indexes().unwrap();
    program
}

fn setup<'p>(program: &'p BytecodeProgram, opcode: Opcode) -> Interpreter<'p> {
    use Opcode::*;
    let mut vm = Interpreter::with_host(program, Box::new(TestHost)).unwrap();
    vm.set_jit_options(None);
    vm.push_frame_direct(0, &[], None, None).unwrap();
    vm.modules[0].initialized = true;
    vm.modules[0].variables[0] = Value::TRUE;
    vm.modules[0].exports.insert(StringId(0), 0);
    vm.classes[0].instance_methods[0] = Some(CallTarget::Primitive(PrimitiveId(0)));
    let mut regs = [Value::NULL.word(); 8];
    regs[0] = Value::TRUE.word();
    regs[1] = i32_word(7);
    regs[2] = i32_word(2);
    regs[3] = i32_word(2);
    match opcode {
        FNeg | FAdd | FSub | FMul | FDiv | FRem | FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe | F64ToNum => {
            regs[1] = f64_word(-7.5); regs[2] = f64_word(2.0);
        },
        GuardType | GuardInt32 | GuardF64 | NumToInt32 | NumToF64 | MakeRange => {
            regs[1] = Value::from_num(7.0).word(); regs[2] = Value::from_num(2.0).word();
        },
        GuardBounds => { regs[0] = i32_word(0); },
        GuardClass | LoadField | StoreField | IsInstance | Invoke | InvokeSuper => {
            let obj = vm.heap.alloc(HeapObject::Instance { class_slot: 0, fields: vec![Value::TRUE].into_boxed_slice() }).word();
            regs[1] = obj;
            if opcode == StoreField { regs[0] = obj; regs[1] = Value::FALSE.word(); }
            if opcode == IsInstance { regs[2] = vm.classes[0].class_value.word(); }
        },
        BindMethod | BindForeignMethod | LoadStaticField | StoreStaticField => {
            regs[0] = vm.classes[0].class_value.word(); regs[1] = regs[0];
            if opcode == StoreStaticField { regs[1] = Value::TRUE.word(); }
        },
        ImportVariable => { regs[1] = vm.modules[0].module_value.word(); },
        LoadCell | StoreCell => {
            regs[1] = vm.heap.alloc(HeapObject::Cell(Value::TRUE.word())).word();
            if opcode == StoreCell { regs[0] = regs[1]; regs[1] = Value::FALSE.word(); }
        },
        ListLen | ListGet | ListSet | ListAppend | ListInsert | ListRemoveAt => {
            let list = vm.heap.alloc(HeapObject::List(vec![Value::FALSE, Value::TRUE])).word();
            regs[1] = list; regs[2] = i32_word(0);
            if matches!(opcode, ListSet | ListAppend | ListInsert) {
                regs[0] = list; regs[1] = if opcode == ListAppend { Value::TRUE.word() } else { i32_word(0) }; regs[2] = Value::TRUE.word();
            }
        },
        MapLen | MapGet | MapSet | MapRemove => {
            let map = vm.heap.alloc(HeapObject::Map(WrenMap::default())).word();
            regs[1] = map; regs[2] = Value::TRUE.word();
            if opcode == MapSet { regs[0] = map; regs[1] = Value::FALSE.word(); }
        },
        StringLen | StringConcat | StringSlice | StringCodePointAt => {
            regs[1] = vm.strings[1].word();
            regs[2] = if opcode == StringConcat { regs[1] } else { i32_word(0) };
        },
        CallClosure | MakeFiber | FiberCall | FiberTry | FiberTransfer | FiberTransferError => {
            let closure = vm.heap.alloc(HeapObject::Closure { function_index: 1, captures: Box::new([]) });
            regs[1] = if matches!(opcode, CallClosure | MakeFiber) { closure.word() }
                else { vm.allocate_fiber(closure).unwrap().word() };
            regs[2] = Value::NULL.word();
        },
        _ => {},
    }
    for (i, word) in regs.into_iter().enumerate() { vm.trace_write_slot(FrameSlot::General(Reg(i as u8)), word).unwrap(); }
    vm.current_frame_mut().arguments[1] = Value::TRUE.word();
    vm.current_frame_mut().returns[1] = Value::FALSE.word();
    vm
}

/// Heap handles are deterministic in these fixtures. Include contents as well
/// as registers so stores, binding, allocation and profile effects are checked.
fn state(vm: &Interpreter<'_>) -> String {
    let fibers: Vec<_> = vm.fibers.iter().map(|f| f.as_ref().map(|f| (
        format!("{:?}", f.status), f.registers.clone(), f.frames.iter().map(|frame| (
            frame.function, frame.pc, frame.arguments, frame.returns, frame.special,
        )).collect::<Vec<_>>(), f.last_value, f.error,
    ))).collect();
    let heap: Vec<_> = vm.heap.slots.iter().map(|slot| format!("{:?}", slot.as_ref().map(|slot| &slot.object))).collect();
    let modules: Vec<_> = vm.modules.iter().map(|m| (&m.variables, m.initialized, m.initializing)).collect();
    let classes: Vec<_> = vm.classes.iter().map(|c| (&c.static_fields, &c.instance_methods, &c.static_methods)).collect();
    format!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", vm.run_state, vm.result, vm.error, fibers, heap, modules, classes, vm.profile_counters, vm.trace_current_location())
}

#[test]
fn all_122_opcodes_execute_in_native_traces_like_the_interpreter() {
    let mut count = 0;
    for raw in 0..=u8::MAX {
        let Some(opcode) = Opcode::from_u8(raw) else { continue; };
        let program = program_for(opcode);
        let mut baseline = setup(&program, opcode);
        let mut native = setup(&program, opcode);
        baseline.trace_execute_unfused_step().unwrap();
        let mut jit = TracingJit::default();
        jit.execute_one_for_test(&mut native).unwrap_or_else(|e| panic!("{opcode:?}: {e}"));
        assert_eq!(jit.stats().native_entries, 1, "{opcode:?}");
        assert_eq!(state(&native), state(&baseline), "{opcode:?}");
        count += 1;
    }
    assert_eq!(count, 122);
}
