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
        program.metadata.classes.push(ClassInfo { id: ClassId(id), module: ModuleId(0), name: StringId(0), superclass: (id == 1).then_some(ClassId(0)), own_instance_field_count: 1, static_field_count: 1, is_foreign: opcode == AllocForeign });
    }
    program.metadata.foreign_functions.push(ForeignFunctionInfo { id: ForeignFnId(0), module: StringId(0), class: StringId(0), signature: MethodSymbol(0), is_static: false });
    let frame = FrameLayout { general_register_count: 8, parameters: vec![], captures: vec![], return_kind: RegisterKind::Value };
    program.push_function(BytecodeFunction { id: FuncId(0), module: ModuleId(0), name: StringId(0), kind: FunctionKind::ModuleBody, is_async: false, field_owner: matches!(opcode, LoadField | StoreField).then_some(ClassId(0)), arity: 0, capture_count: 0, frame: frame.clone(), chunk }).unwrap();
    let mut child = Assembler::new();
    child.emit_a(LoadTrue, 0);
    child.emit_a(Return, 0);
    program.push_function(BytecodeFunction { id: FuncId(1), module: ModuleId(0), name: StringId(0), kind: FunctionKind::Closure, is_async: false, field_owner: None, arity: 0, capture_count: 0, frame, chunk: child.finish().unwrap() }).unwrap();
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
            regs[1] = vm.heap.alloc(HeapObject::Cell(Box::new(Value::TRUE.word()))).word();
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
            let closure = vm.heap.alloc(HeapObject::Closure { function_index: 1, captures: Box::new([]), reference_indices: Box::new([]) });
            regs[1] = if matches!(opcode, CallClosure | MakeFiber) { closure.word() }
                else { vm.allocate_fiber(closure).unwrap().word() };
            regs[2] = Value::NULL.word();
        },
        _ => {},
    }
    for (i, word) in regs.into_iter().enumerate() { vm.trace_write_slot(FrameSlot::General(Reg(i as u8)), word).unwrap(); }
    vm.current_banks_mut().arguments[1] = Value::TRUE.word();
    vm.current_banks_mut().returns[1] = Value::FALSE.word();
    // These fixtures step frames without run_function's root provider. Keep
    // their comparison heap alive while the other VM initializes or collects.
    for value in vm.heap.allocations.clone() {
        vm.add_root(value);
    }
    vm
}

/// Canonicalize process-wide handles by VM-local allocation order.
fn state(vm: &mut Interpreter<'_>) -> String {
    // Commit the cached PC before comparing the saved frame state.
    vm.flush_hot_pc();
    let word = |word: RegisterWord| {
        let value = Value::from_word(word);
        if value.is_heap() {
            let index = vm.heap.allocations.iter().position(|&v| v == value)
                .expect("fixture reference must belong to this VM");
            Value::from_heap(index as u32).word()
        } else {
            word
        }
    };
    let value = |value: Value| Value::from_word(word(value.word()));
    let values = |items: &[Value]| items.iter().copied().map(value).collect::<Vec<_>>();
    let fibers: Vec<_> = vm.fibers.iter().map(|f| f.as_ref().map(|f| {
        let f = unsafe { f.get() };
        (
        format!("{:?}", f.status), f.registers.iter().copied().map(word).collect::<Vec<_>>(), f.frames.iter().enumerate().map(|(depth, frame)| {
            let banks = &f.banks[depth];
            (frame.function, frame.pc, banks.arguments.map(word), banks.returns.map(word), banks.special.map(word))
        }).collect::<Vec<_>>(), value(f.last_value), value(f.error),
    )})).collect();
    let heap: Vec<_> = vm.heap.allocations.iter().map(|&v| {
        match vm.heap.get(v) {
            Some(HeapObject::List(items)) => format!("List({:?})", values(items)),
            Some(HeapObject::Map(map)) => {
                let mut entries: Vec<_> = (0..map.len()).map(|index| {
                    let key = map.nth_key(&vm.heap.map_shapes, index).unwrap();
                    let v = map.nth_value(index).unwrap();
                    let key = match key {
                        MapKey::Object(handle) => format!("{:?}", value(Value::from_heap(handle))),
                        key => format!("{key:?}"),
                    };
                    (key, format!("{:?}", value(v)))
                }).collect();
                entries.sort();
                format!("Map({entries:?})")
            }
            Some(HeapObject::Cell(w)) => format!("Cell({:?})", word(**w)),
            Some(HeapObject::Closure { function_index, captures, reference_indices }) => {
                let captures: Vec<_> = captures.iter().enumerate().map(|(i, &w)| {
                    if reference_indices.contains(&(i as u16)) { word(w) } else { w }
                }).collect();
                format!("Closure({function_index}, {captures:?}, {reference_indices:?})")
            }
            Some(HeapObject::Instance { class_slot, fields }) => format!("Instance({class_slot}, {:?})", values(fields)),
            Some(HeapObject::FiberObject { fiber, .. }) => format!("Fiber({fiber})"),
            object => format!("{object:?}"),
        }
    }).collect();
    let modules: Vec<_> = vm.modules.iter().map(|m| (values(&m.variables), m.initialized, m.initializing)).collect();
    let classes: Vec<_> = vm.classes.iter().map(|c| (values(&c.static_fields), &c.instance_methods, &c.static_methods)).collect();
    format!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", vm.run_state, value(vm.result), vm.error, fibers, heap, modules, classes, vm.profile_counters, vm.trace_current_location())
}

#[test]
fn floating_remainder_executes_without_interpreter_callbacks() {
    let program = program_for(Opcode::FRem);
    for lhs in [-0.0, 0.0, -7.5, 7.5, f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for rhs in [-0.0, 0.0, -2.0, 2.0, f64::INFINITY, f64::NAN] {
            let mut native = setup(&program, Opcode::FRem);
            native.trace_write_slot(FrameSlot::General(Reg(1)), f64_word(lhs)).unwrap();
            native.trace_write_slot(FrameSlot::General(Reg(2)), f64_word(rhs)).unwrap();
            let mut jit = TracingJit::new(TracingJitOptions::default());
            jit.execute_one_for_test(&mut native).unwrap();
            assert_eq!(jit.stats().native_entries, 1);
            let actual = f64::from_bits(native.trace_read_slot(FrameSlot::General(Reg(0))).unwrap().0);
            let expected = lhs % rhs;
            assert!(if expected.is_nan() { actual.is_nan() }
                else { actual.to_bits() == expected.to_bits() }, "{lhs:?} % {rhs:?}: {actual:?}");
        }
    }
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
        let mut options = TracingJitOptions::default();
        options.no_runtime_calls_in_native_trace = false;
        let mut jit = TracingJit::new(options);
        jit.execute_one_for_test(&mut native).unwrap_or_else(|e| panic!("{opcode:?}: {e}"));
        assert_eq!(jit.stats().native_entries, 1, "{opcode:?}");
        assert_eq!(state(&mut native), state(&mut baseline), "{opcode:?}");
        count += 1;
    }
    assert_eq!(count, 122);
}

#[test]
fn heap_fast_paths_execute_without_runtime_callbacks() {
    use Opcode::*;
    for opcode in [
        LoadField, StoreField, LoadCell, StoreCell,
        ListLen, ListGet, ListSet, MapLen, StringLen, ReadBarrier,
    ] {
        let program = program_for(opcode);
        let mut baseline = setup(&program, opcode);
        let mut native = setup(&program, opcode);
        baseline.trace_execute_unfused_step().unwrap();

        // The default forbids runtime opcode callbacks. Success therefore proves
        // that this opcode was lowered through the direct heap fast path.
        let mut jit = TracingJit::new(TracingJitOptions::default());
        assert!(jit.options().no_runtime_calls_in_native_trace, "{opcode:?}");
        jit.execute_one_for_test(&mut native)
            .unwrap_or_else(|error| panic!("direct {opcode:?} fast path failed: {error}"));
        assert_eq!(state(&mut native), state(&mut baseline), "{opcode:?}");
    }
}

#[test]
fn list_fast_paths_preserve_negative_index_semantics() {
    use Opcode::*;
    for opcode in [ListGet, ListSet] {
        let program = program_for(opcode);
        let mut baseline = setup(&program, opcode);
        let mut native = setup(&program, opcode);

        let index_reg = if opcode == ListGet { Reg(2) } else { Reg(1) };
        baseline.trace_write_slot(FrameSlot::General(index_reg), i32_word(-1)).unwrap();
        native.trace_write_slot(FrameSlot::General(index_reg), i32_word(-1)).unwrap();

        baseline.trace_execute_unfused_step().unwrap();
        let mut jit = TracingJit::new(TracingJitOptions::default());
        jit.execute_one_for_test(&mut native)
            .unwrap_or_else(|error| panic!("negative-index {opcode:?} fast path failed: {error}"));
        assert_eq!(state(&mut native), state(&mut baseline), "{opcode:?}");
    }
}

#[test]
fn heap_reference_stores_use_the_native_barrier_path() {
    use Opcode::*;
    for opcode in [StoreField, StoreCell, ListSet] {
        let program = program_for(opcode);
        let mut baseline = setup(&program, opcode);
        let mut native = setup(&program, opcode);

        let baseline_child = baseline.heap.alloc(HeapObject::String(Arc::from(b"child".as_slice())));
        baseline.add_root(baseline_child);
        let native_child = native.heap.alloc(HeapObject::String(Arc::from(b"child".as_slice())));
        native.add_root(native_child);
        let value_reg = match opcode {
            StoreField | StoreCell => Reg(1),
            ListSet => Reg(2),
            _ => unreachable!(),
        };
        baseline.trace_write_slot(FrameSlot::General(value_reg), baseline_child.word()).unwrap();
        native.trace_write_slot(FrameSlot::General(value_reg), native_child.word()).unwrap();

        baseline.trace_execute_unfused_step().unwrap();
        let mut jit = TracingJit::new(TracingJitOptions::default());
        jit.execute_one_for_test(&mut native)
            .unwrap_or_else(|error| panic!("reference-store {opcode:?} fast path failed: {error}"));
        assert_eq!(state(&mut native), state(&mut baseline), "{opcode:?}");
    }
}

#[test]
fn native_class_guards_cover_class_sets_deopt_and_moving_gc() {
    use crate::wren_trace_ir::TraceInst;
    for mode in 0..3 {
        let mut program = program_for(Opcode::GuardClass);
        // Class IDs deliberately differ from runtime class slots.
        program.metadata.classes[0].id = ClassId(11);
        program.metadata.classes[1].id = ClassId(29);
        program.metadata.classes[1].superclass = Some(ClassId(11));
        program.functions[0].chunk.class_guards[0].expected = if mode == 2 {
            ClassSet::new()
        } else if mode == 0 {
            ClassSet::singleton(ClassId(11))
        } else {
            ClassSet::from_iter([ClassId(11), ClassId(29)])
        };
        program.rebuild_indexes().unwrap();
        let mut vm = setup(&program, Opcode::GuardClass);
        let mut jit = TracingJit::new(TracingJitOptions::default());
        jit.execute_one_for_test(&mut vm).unwrap();
        let trace = jit.trace(TraceId(0)).unwrap();
        assert!(trace.nodes.iter().any(|node| matches!(node.kind, TraceInst::GuardClass { .. })));
        assert!(!trace.nodes.iter().any(|node| matches!(node.kind, TraceInst::Runtime { .. })));

        let mut values = Vec::new();
        for slot in 0..2 {
            values.push((vm.heap.alloc(HeapObject::Instance {
                class_slot: slot, fields: vec![Value::TRUE].into_boxed_slice(),
            }), slot == 0 || mode != 0));
            values.push((vm.classes[slot as usize].class_value, slot == 0 || mode != 0));
            values.push((vm.heap.alloc(HeapObject::Foreign { class_slot: slot, bytes: vec![] }), slot == 0 || mode != 0));
        }
        values.extend([(Value::NULL, false), (Value::FALSE, false), (Value::from_num(3.0), false),
            (vm.strings[1], false), (vm.heap.alloc(HeapObject::List(vec![])), false),
            (vm.builtin_value(crate::wren_ir::BuiltinId::Num), false)]);
        for &(value, _) in &values { vm.add_root(value); }
        for _ in 0..3 {
            vm.collect_garbage();
            for &(value, matches) in &values {
                vm.trace_set_location(TraceLocation::new(FuncId(0), BytecodePc(0))).unwrap();
                vm.trace_write_slot(FrameSlot::General(Reg(1)), value.word()).unwrap();
                vm.trace_write_slot(FrameSlot::General(Reg(0)), Value::FALSE.word()).unwrap();
                jit.enter_trace(&mut vm, TraceId(0)).unwrap();
                if matches {
                    assert_eq!(vm.trace_current_location().pc, BytecodePc(3));
                    assert_eq!(vm.trace_read_slot(FrameSlot::General(Reg(0))).unwrap(), value.word());
                } else {
                    // Failure leaves the input and destination untouched, and
                    // resumes at the original bytecode guard for VM deopt.
                    assert_eq!(vm.trace_current_location().pc, BytecodePc(0));
                    assert_eq!(vm.trace_read_slot(FrameSlot::General(Reg(0))).unwrap(), Value::FALSE.word());
                    vm.trace_execute_unfused_step().unwrap();
                    assert_eq!(vm.trace_current_location().pc, BytecodePc(3));
                }
            }
        }
    }
}
