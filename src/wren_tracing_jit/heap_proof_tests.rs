use super::*;
use crate::wren_trace_ir::{TraceAnchor, TraceBuilder};

fn list_trace(repeats: u8, explicit_bounds: bool, runtime: bool) -> Trace {
    let location = TraceLocation::new(FuncId(0), BytecodePc(0));
    let mut b = TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Side, location });
    let list = b.read_slot(FrameSlot::General(Reg(0)), TraceType::Any, TraceRep::Tagged);
    let index = b.read_slot(FrameSlot::General(Reg(1)), TraceType::Int32, TraceRep::Int32);
    let failure = b.add_side_exit(ExitReason::GuardBounds, location.pc);
    if explicit_bounds {
        let length = b.emit1(TraceInst::HeapListLen { list, exit: failure }, TraceType::Int32, TraceRep::Int32);
        b.emit0(TraceInst::GuardBounds { index, length, exit: failure });
    }
    for i in 0..repeats {
        if runtime && i != 0 {
            b.runtime_leaf(location, BytecodePc(1));
        }
        let value = b.emit1(TraceInst::HeapListGet { list, index, exit: failure }, TraceType::Any, TraceRep::Tagged);
        b.write_slot(FrameSlot::General(Reg(i + 2)), value);
    }
    let success = b.add_side_exit(ExitReason::GuardValue, BytecodePc(2));
    b.set_terminator(TraceTerminator::Exit(success));
    b.finish().unwrap()
}

#[test]
fn repeated_list_accesses_and_explicit_bounds_do_not_add_native_guards() {
    let mut jit = TracingJit::new(TracingJitOptions { cranelift_dump_ir: true, ..Default::default() });
    let count = |compiled: CompiledTrace| compiled.debug_text.unwrap().matches("brif ").count();
    let single = count(jit.compile_trace(&list_trace(1, false, false)).unwrap());
    assert_eq!(count(jit.compile_trace(&list_trace(3, false, false)).unwrap()), single);
    assert_eq!(count(jit.compile_trace(&list_trace(3, true, false)).unwrap()), single);
    assert!(count(jit.compile_trace(&list_trace(2, false, true)).unwrap()) > single);
}

#[test]
fn runtime_invalidates_list_length_and_negative_index_proofs() {
    struct Mutation { entry: *mut TraceHeapEntry, len: usize, kind: TraceHeapKind }
    unsafe extern "C" fn mutate(activation: *mut TraceActivation, _: u32, _: u32, _: u32) -> u32 {
        let mutation = unsafe { &*((*activation).runtime as *const Mutation) };
        let entry = unsafe { &*mutation.entry };
        entry.len.store(mutation.len, Ordering::Relaxed);
        entry.kind.store(mutation.kind as u32, Ordering::Relaxed);
        TRACE_RESULT_CONTINUE
    }
    let mut jit = TracingJit::new(TracingJitOptions::default());
    let compiled = jit.compile_trace(&list_trace(2, false, true)).unwrap();
    let mut entries: Vec<_> = (0..TRACE_HEAP_CHUNK_SIZE).map(|_| TraceHeapEntry::new(0)).collect();
    let chunks = [entries.as_mut_ptr()];
    let mut data = [num_word(10.0), num_word(20.0), num_word(30.0)];
    entries[0].address.store(1, Ordering::Relaxed);
    entries[0].data.store(data.as_mut_ptr() as usize, Ordering::Relaxed);
    for (len, kind, succeeds) in [(2, TraceHeapKind::List, true), (0, TraceHeapKind::List, false), (3, TraceHeapKind::Cell, false)] {
        entries[0].len.store(3, Ordering::Relaxed);
        entries[0].kind.store(TraceHeapKind::List as u32, Ordering::Relaxed);
        let mut mutation = Mutation { entry: entries.as_mut_ptr(), len, kind };
        let mut general = [RegisterWord(TAG_HEAP), int_word(-1), RegisterWord(0), RegisterWord(0)];
        let mut activation = TraceActivation::new(TraceActivationInputs {
            general: general.as_mut_ptr(), arguments: std::ptr::null_mut(), returns: std::ptr::null_mut(),
            special: std::ptr::null_mut(), runtime: &mut mutation as *mut _ as *mut c_void,
            execute: mutate, heap_chunks: chunks.as_ptr(), heap_write_barrier: noop_heap_write_barrier,
        });
        activation.current_links = compiled.native_links.as_ptr();
        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, succeeds as u32);
        assert_eq!(general[2], num_word(30.0));
        if succeeds { assert_eq!(general[3], num_word(20.0)); }
    }
}

#[test]
fn preheader_list_proofs_do_not_survive_runtime_on_the_backedge() {
    unsafe extern "C" fn empty_list(activation: *mut TraceActivation, _: u32, _: u32, _: u32) -> u32 {
        let entry = unsafe { &*((*activation).runtime as *const TraceHeapEntry) };
        entry.len.store(0, Ordering::Relaxed);
        TRACE_RESULT_CONTINUE
    }
    let location = TraceLocation::new(FuncId(0), BytecodePc(0));
    let mut b = TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Loop, location });
    let list = b.read_slot(FrameSlot::General(Reg(0)), TraceType::Any, TraceRep::Tagged);
    let index = b.constant(TraceConstant::Int32(0));
    let failure = b.add_side_exit(ExitReason::GuardBounds, location.pc);
    let length = b.emit1(TraceInst::HeapListLen { list, exit: failure }, TraceType::Int32, TraceRep::Int32);
    b.emit0(TraceInst::GuardBounds { index, length, exit: failure });
    b.trace_mut().preheader = vec![0, 1];
    let value = b.emit1(TraceInst::HeapListGet { list, index, exit: failure }, TraceType::Any, TraceRep::Tagged);
    b.write_slot(FrameSlot::General(Reg(1)), value);
    b.runtime_leaf(location, BytecodePc(1));
    b.add_native_budget_exit(location.pc);
    b.set_terminator(TraceTerminator::LoopBack);
    let trace = b.finish().unwrap();
    let mut jit = TracingJit::new(TracingJitOptions { native_loop_iteration_budget: 3, ..Default::default() });
    let compiled = jit.compile_trace(&trace).unwrap();
    let mut entries: Vec<_> = (0..TRACE_HEAP_CHUNK_SIZE).map(|_| TraceHeapEntry::new(0)).collect();
    let chunks = [entries.as_mut_ptr()];
    let mut data = [num_word(42.0)];
    entries[0].address.store(1, Ordering::Relaxed);
    entries[0].kind.store(TraceHeapKind::List as u32, Ordering::Relaxed);
    entries[0].data.store(data.as_mut_ptr() as usize, Ordering::Relaxed);
    entries[0].len.store(1, Ordering::Relaxed);
    let mut general = [RegisterWord(TAG_HEAP), RegisterWord(0)];
    let mut activation = TraceActivation::new(TraceActivationInputs {
        general: general.as_mut_ptr(), arguments: std::ptr::null_mut(), returns: std::ptr::null_mut(),
        special: std::ptr::null_mut(), runtime: entries.as_mut_ptr() as *mut c_void,
        execute: empty_list, heap_chunks: chunks.as_ptr(), heap_write_barrier: noop_heap_write_barrier,
    });
    activation.current_links = compiled.native_links.as_ptr();
    assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
    assert_eq!(activation.exit_id, failure.0, "the second iteration must recheck bounds");
    assert_eq!(general[1], num_word(42.0));
}
