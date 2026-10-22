//! High-throughput register-based interpreter for `wren_bytecode.rs`.
//!
//! The hot dispatch loop is token-threaded: bytecode is decoded once when the
//! interpreter is created and every instruction is replaced by a direct Rust
//! function pointer plus compact decoded operands. Runtime dispatch therefore
//! performs no giant opcode `match` and no variable-width bytecode decoding.
//!
//! The implementation deliberately keeps slow paths (method lookup, allocation,
//! GC, foreign calls, deoptimization, module loading) out of tiny arithmetic,
//! move and branch handlers. Calls use the bytecode ABI directly and frames are
//! backed by a contiguous register stack to avoid one allocation per call.
//!
//! This file is self-contained apart from `wren_ir` and `wren_bytecode`.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::identity_op)]
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use crate::wren_gc::{
    Collector as GcCollector, Config as GcConfig, Descriptor as GcDescriptor, Mutator as GcMutator,
    Object as GcObject, RootProvider as GcRootProvider, RootRegistration as GcRootRegistration,
    RootVisitor as GcRootVisitor, Slot as GcSlot, TraceLayout as GcTraceLayout,
    RAW_CODEC as GC_RAW_CODEC,
};
use crate::wren_trace_ir::{FrameSlot, TraceConstant, TraceId, TraceLocation};
use crate::wren_tracing_jit::{
    JitAction, TraceActivation, TraceActivationInputs, TraceHeapEntry,
    TraceHeapKind, TraceRuntime, TraceStep, TracingJit, TracingJitOptions, TracingJitStats,
    TRACE_HEAP_CHUNK_BITS, TRACE_HEAP_CHUNK_MASK, TRACE_HEAP_CHUNK_SIZE,
};

use crate::wren_bytecode::{
    decode_bailout_reason, function_kind_uses_this, instruction_word_len,
    BytecodeFunction, BytecodePc, BytecodeProgram, BytecodeVerifier, ClassGuardDescriptor,
    DeoptLocation, FrameLayout, IncomingSource, Opcode, ReferenceMap, Reg, RegisterKind,
    RegisterWord, SpecialReg, SwitchTable, TypeGuardDescriptor,
    ARGUMENT_REGISTER_COUNT, RETURN_REGISTER_COUNT, SPECIAL_REGISTER_COUNT,
};
use crate::wren_ir::{
    BailoutReason, BuiltinId, ClassId, ClassSet, Constant, ForeignFnId, FuncId, FunctionKind,
    MethodSymbol, ModuleId, PrimitiveId, StringId, Type,
};

// =============================================================================
// Value representation
// =============================================================================

/// NaN-boxed Wren value.
///
/// Raw `Int32`, `F64` and `RawPtr` bytecode registers are *not* instances of
/// `Value`; their handlers use `RegisterWord` directly. Wren-visible numbers are
/// stored as ordinary IEEE-754 doubles. All NaNs are canonicalized so the tag
/// space can never be confused with a number.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Value(u64);

/// Wren bitwise operands are truncated to unsigned 32 bits, not signed i64.
#[inline]
pub(crate) fn num_to_u32(number: f64) -> u32 {
    if number >= 0.0 && number < 4294967296.0 {
        number as u32
    } else {
        number.trunc().rem_euclid(4294967296.0) as u32
    }
}

const TAG_BASE: u64 = 0x7ffc_0000_0000_0000;
const TAG_MASK: u64 = 0x7ffc_0000_0000_0000;
const TAG_NULL: u64 = TAG_BASE | 0x01;
const TAG_FALSE: u64 = TAG_BASE | 0x02;
const TAG_TRUE: u64 = TAG_BASE | 0x03;
const TAG_HEAP: u64 = 0xfffc_0000_0000_0000;
const TAG_HEAP_MASK: u64 = 0xffff_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

impl Value {
    pub const NULL: Self = Self(TAG_NULL);
    pub const FALSE: Self = Self(TAG_FALSE);
    pub const TRUE: Self = Self(TAG_TRUE);

    #[inline(always)]
    pub const fn raw_bits(self) -> u64 { self.0 }

    /// Construct a Value from raw NaN-box bits.
    ///
    /// # Safety
    /// Heap-tagged values must reference a live Wren heap handle owned by the
    /// same interpreter. Prefer values obtained from the VM/HostContext.
    #[inline(always)]
    pub const unsafe fn from_raw_bits_unchecked(bits: u64) -> Self { Self(bits) }

    #[inline(always)]
    pub const fn from_bool(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }

    #[inline(always)]
    pub fn from_num(value: f64) -> Self {
        let bits = if value.is_nan() { CANONICAL_NAN } else { value.to_bits() };
        Self(bits)
    }

    #[inline(always)]
    pub const fn from_heap(handle: u32) -> Self { Self(TAG_HEAP | handle as u64) }

    #[inline(always)]
    pub const fn is_null(self) -> bool { self.0 == TAG_NULL }

    #[inline(always)]
    pub const fn is_bool(self) -> bool { self.0 == TAG_FALSE || self.0 == TAG_TRUE }

    #[inline(always)]
    pub const fn as_bool(self) -> Option<bool> {
        if self.0 == TAG_FALSE { Some(false) }
        else if self.0 == TAG_TRUE { Some(true) }
        else { None }
    }

    #[inline(always)]
    pub const fn is_heap(self) -> bool { (self.0 & TAG_HEAP_MASK) == TAG_HEAP }

    #[inline(always)]
    pub const fn heap_handle(self) -> Option<u32> {
        if self.is_heap() { Some((self.0 & PAYLOAD_MASK) as u32) } else { None }
    }

    #[inline(always)]
    pub const fn is_num(self) -> bool { (self.0 & TAG_MASK) != TAG_BASE && !self.is_heap() }

    #[inline(always)]
    pub fn as_num(self) -> Option<f64> {
        if self.is_num() { Some(f64::from_bits(self.0)) } else { None }
    }

    #[inline(always)]
    pub const fn is_falsey(self) -> bool { self.0 == TAG_FALSE || self.0 == TAG_NULL }

    #[inline(always)]
    pub const fn word(self) -> RegisterWord { RegisterWord(self.0) }

    #[inline(always)]
    pub const fn from_word(word: RegisterWord) -> Self { Self(word.0) }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            return write!(f, "null");
        }
        if let Some(value) = self.as_bool() {
            return write!(f, "{value}");
        }
        if let Some(value) = self.as_num() {
            return write!(f, "{value}");
        }
        if let Some(handle) = self.heap_handle() {
            return write!(f, "heap#{handle}");
        }
        write!(f, "value(0x{:016x})", self.0)
    }
}

#[inline(always)]
fn word_i32(word: RegisterWord) -> i32 {
    word.0 as u32 as i32
}

#[inline(always)]
fn i32_word(value: i32) -> RegisterWord {
    RegisterWord(value as u32 as u64)
}

#[inline(always)]
fn word_f64(word: RegisterWord) -> f64 {
    f64::from_bits(word.0)
}

#[inline(always)]
fn f64_word(value: f64) -> RegisterWord {
    RegisterWord(value.to_bits())
}

// =============================================================================
// Runtime heap and map keys
// =============================================================================

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 { self.0 }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 { 0x517c_c1b7_2722_0a95 } else { self.0 };
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            hash ^= word.wrapping_mul(0x9e37_79b1_85eb_ca87);
            hash = hash.rotate_left(27).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        }
        for &byte in chunks.remainder() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        self.0 = hash;
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        let mut hash = if self.0 == 0 { 0x517c_c1b7_2722_0a95 } else { self.0 };
        hash ^= value.wrapping_mul(0x9e37_79b1_85eb_ca87);
        self.0 = hash.rotate_left(27).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) { self.write_u64(value as u64); }
    #[inline]
    fn write_usize(&mut self, value: usize) { self.write_u64(value as u64); }
}

type FastBuildHasher = BuildHasherDefault<FastHasher>;
type FastMap<K, V> = HashMap<K, V, FastBuildHasher>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MapKey {
    Null,
    Bool(bool),
    Num(u64),
    String(Arc<[u8]>),
    Object(u32),
}

#[derive(Debug, Default)]
struct WrenMap {
    entries: FastMap<MapKey, Value>,
}

#[derive(Clone, Copy, Debug)]
enum PromiseWaiter {
    Resume {
        fiber: u32,
        fiber_object: Value,
        dst: Reg,
    },
    /// Promise chaining used by async return-value adoption. When the source
    /// settles, the target Promise is settled with exactly the same outcome.
    Forward {
        promise: Value,
    },
}

#[derive(Debug)]
enum PromiseState {
    Pending {
        producer: Option<Value>,
        waiters: Vec<PromiseWaiter>,
    },
    Fulfilled(Value),
    Rejected(Value),
}

struct PromiseCell {
    state: std::cell::UnsafeCell<PromiseState>,
}

impl PromiseCell {
    fn new(state: PromiseState) -> Self {
        Self { state: std::cell::UnsafeCell::new(state) }
    }

    #[inline(always)]
    unsafe fn get(&self) -> &PromiseState {
        unsafe { &*self.state.get() }
    }

    #[inline(always)]
    unsafe fn get_mut(&self) -> &mut PromiseState {
        unsafe { &mut *self.state.get() }
    }
}

unsafe impl Send for PromiseCell {}
unsafe impl Sync for PromiseCell {}

impl fmt::Debug for PromiseCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PromiseCell(..)")
    }
}

#[derive(Debug)]
enum HeapObject {
    Random { state: u64 },
    String(Arc<[u8]>),
    List(Vec<Value>),
    Map(WrenMap),
    MapEntry { key: Value, value: Value },
    MapKeys { map: Value },
    Range {
        from: f64,
        to: f64,
        inclusive: bool,
    },
    Cell(Box<RegisterWord>),
    Closure {
        function_index: u32,
        captures: Box<[RegisterWord]>,
        reference_indices: Box<[u16]>,
    },
    ClassObject {
        class_slot: u32,
    },
    BuiltinClass {
        builtin: BuiltinId,
    },
    Instance {
        class_slot: u32,
        fields: Box<[Value]>,
    },
    Foreign {
        class_slot: u32,
        bytes: Vec<u8>,
    },
    Promise {
        state: Arc<PromiseCell>,
    },
    /// Scheduler-owned execution context for an async function. It deliberately
    /// reuses FiberState's frame/register storage without being a Wren Fiber.
    AsyncTaskObject {
        task: u32,
        state: Arc<FiberCell>,
        trace_metadata: Arc<FiberTraceMetadata>,
        promise: Value,
    },
    FiberObject {
        fiber: u32,
        state: Arc<FiberCell>,
        trace_metadata: Arc<FiberTraceMetadata>,
    },
    ModuleObject {
        module_slot: u32,
    },
}

// -----------------------------------------------------------------------------
// Stable Wren handles over moving MMTk objects
// -----------------------------------------------------------------------------
//
// Wren Values intentionally keep the existing compact u32 heap handle.  The
// handle indexes a process-wide stable atomic cell containing the *current*
// MMTk object address.  GC roots and object edges expose that cell to MMTk, so a
// nursery/defrag move updates one canonical address instead of rewriting every
// NaN-boxed Value in the VM.  This preserves the fast interpreter/JIT ABI while
// still allowing StickyImmix to move objects.

const HANDLE_CHUNK_BITS: u32 = TRACE_HEAP_CHUNK_BITS;
const HANDLE_CHUNK_SIZE: usize = TRACE_HEAP_CHUNK_SIZE;
const HANDLE_CHUNK_MASK: u32 = TRACE_HEAP_CHUNK_MASK;
const MAX_HANDLE_CHUNKS: usize = 1 << 14; // 16,777,216 simultaneous handle IDs.
const FREE_HANDLE_NONE: u32 = 0;

type HandleEntry = TraceHeapEntry;

struct HandleTable {
    chunks: Box<[AtomicPtr<HandleEntry>]>,
    install_lock: StdMutex<()>,
    next: AtomicU32,
    free_head: AtomicU32,
    live: AtomicUsize,
}

impl HandleTable {
    fn new() -> Self {
        Self {
            chunks: (0..MAX_HANDLE_CHUNKS)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            install_lock: StdMutex::new(()),
            next: AtomicU32::new(0),
            free_head: AtomicU32::new(FREE_HANDLE_NONE),
            live: AtomicUsize::new(0),
        }
    }

    #[cold]
    fn install_chunk(&self, chunk_index: usize) -> *mut HandleEntry {
        let _guard = self.install_lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.chunks[chunk_index].load(Ordering::Acquire);
        if !current.is_null() {
            return current;
        }
        let chunk = (0..HANDLE_CHUNK_SIZE)
            .map(|_| HandleEntry::new(FREE_HANDLE_NONE))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let slice = Box::leak(chunk);
        let pointer = slice.as_mut_ptr();
        self.chunks[chunk_index].store(pointer, Ordering::Release);
        pointer
    }

    #[inline(always)]
    fn entry(&self, handle: u32) -> &'static HandleEntry {
        let chunk_index = (handle >> HANDLE_CHUNK_BITS) as usize;
        assert!(chunk_index < MAX_HANDLE_CHUNKS, "Wren heap handle space exhausted");
        let mut chunk = self.chunks[chunk_index].load(Ordering::Acquire);
        if chunk.is_null() {
            chunk = self.install_chunk(chunk_index);
        }
        unsafe { &*chunk.add((handle & HANDLE_CHUNK_MASK) as usize) }
    }

    #[inline(always)]
    fn existing_entry(&self, handle: u32) -> Option<&'static HandleEntry> {
        let chunk_index = (handle >> HANDLE_CHUNK_BITS) as usize;
        if chunk_index >= MAX_HANDLE_CHUNKS {
            return None;
        }
        let chunk = self.chunks[chunk_index].load(Ordering::Acquire);
        if chunk.is_null() {
            return None;
        }
        Some(unsafe { &*chunk.add((handle & HANDLE_CHUNK_MASK) as usize) })
    }

    fn allocate_handle(&self) -> u32 {
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            if head == FREE_HANDLE_NONE {
                let handle = self.next.fetch_add(1, Ordering::Relaxed);
                let max = (MAX_HANDLE_CHUNKS * HANDLE_CHUNK_SIZE) as u32;
                assert!(handle < max, "Wren heap handle space exhausted");
                let entry = self.entry(handle);
                debug_assert_eq!(entry.address.load(Ordering::Relaxed), 0);
                return handle;
            }
            let handle = head - 1;
            let entry = self.entry(handle);
            let next = entry.next_free.load(Ordering::Relaxed);
            if self
                .free_head
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                entry.next_free.store(FREE_HANDLE_NONE, Ordering::Relaxed);
                debug_assert_eq!(entry.address.load(Ordering::Relaxed), 0);
                return handle;
            }
        }
    }

    #[inline(always)]
    fn publish(&self, handle: u32, object: GcObject, heap_object: &HeapObject) {
        let entry = self.entry(handle);
        entry.address.store(object.address(), Ordering::Relaxed);
        Self::refresh_entry(entry, heap_object);
        self.live.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn refresh(&self, handle: u32, object: &HeapObject) {
        if let Some(entry) = self.existing_entry(handle) {
            Self::refresh_entry(entry, object);
        }
    }

    #[inline(always)]
    fn refresh_entry(entry: &HandleEntry, object: &HeapObject) {
        let (kind, data, len) = match object {
            HeapObject::String(bytes) => (
                TraceHeapKind::String,
                bytes.as_ptr() as usize,
                wren_code_point_count(bytes),
            ),
            HeapObject::List(values) => (
                TraceHeapKind::List,
                values.as_ptr() as usize,
                values.len(),
            ),
            HeapObject::Map(map) => (
                TraceHeapKind::Map,
                0,
                map.entries.len(),
            ),
            HeapObject::Cell(value) => (
                TraceHeapKind::Cell,
                (&**value as *const RegisterWord) as usize,
                1,
            ),
            HeapObject::Instance { fields, .. } => (
                TraceHeapKind::Instance,
                fields.as_ptr() as usize,
                fields.len(),
            ),
            _ => (TraceHeapKind::Invalid, 0, 0),
        };
        entry.data.store(data, Ordering::Relaxed);
        entry.len.store(len, Ordering::Relaxed);
        // Publish kind last so generated code never observes a partially refreshed view.
        entry.kind.store(kind as u32, Ordering::Release);
    }

    #[inline(always)]
    fn jit_chunks(&self) -> *const *mut TraceHeapEntry {
        // AtomicPtr has pointer storage and each live handle's chunk is immutable
        // after publication. Generated code only reads already-installed chunks.
        self.chunks.as_ptr().cast::<*mut TraceHeapEntry>()
    }

    #[inline(always)]
    fn resolve(&self, handle: u32) -> Option<GcObject> {
        let address = self
            .existing_entry(handle)?
            .address
            .load(Ordering::Relaxed);
        (address != 0).then(|| unsafe { GcObject::from_address(address) })
    }

    fn release(&self, handle: u32) {
        let entry = self.entry(handle);
        entry.clear_jit_view();
        entry.address.store(0, Ordering::Relaxed);
        self.live.fetch_sub(1, Ordering::Relaxed);
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            entry.next_free.store(head, Ordering::Relaxed);
            if self
                .free_head
                .compare_exchange_weak(head, handle + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    #[inline(always)]
    fn gc_slot(&self, handle: u32) -> Option<GcSlot> {
        let entry = self.existing_entry(handle)?;
        if entry.address.load(Ordering::Relaxed) == 0 {
            return None;
        }
        Some(unsafe { GcSlot::from_ptr(&entry.address, &GC_RAW_CODEC) })
    }
}

static HANDLE_TABLE: OnceLock<HandleTable> = OnceLock::new();

#[inline(always)]
fn handles() -> &'static HandleTable {
    HANDLE_TABLE.get_or_init(HandleTable::new)
}

#[repr(C)]
struct ManagedObject {
    handle: u32,
    receiver_class: Option<ReceiverClass>,
    object: HeapObject,
}

#[inline(always)]
unsafe fn managed_object(object: GcObject) -> &'static ManagedObject {
    unsafe { &*object.payload().cast::<ManagedObject>() }
}

#[inline(always)]
unsafe fn managed_object_mut(object: GcObject) -> &'static mut ManagedObject {
    unsafe { &mut *object.payload().cast::<ManagedObject>() }
}

#[inline(always)]
fn visit_handle(handle: u32, visitor: &mut dyn FnMut(GcSlot)) {
    if let Some(slot) = handles().gc_slot(handle) {
        visitor(slot);
    }
}

#[inline(always)]
fn visit_value(value: Value, visitor: &mut dyn FnMut(GcSlot)) {
    if let Some(handle) = value.heap_handle() {
        visit_handle(handle, visitor);
    }
}

#[inline(always)]
fn visit_word(word: RegisterWord, visitor: &mut dyn FnMut(GcSlot)) {
    visit_value(Value::from_word(word), visitor);
}

unsafe fn trace_list(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    if let HeapObject::List(values) = &managed.object {
        for &value in values {
            visit_value(value, visitor);
        }
    }
}

unsafe fn trace_map(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    if let HeapObject::Map(map) = &managed.object {
        for (key, &value) in &map.entries {
            if let MapKey::Object(handle) = key {
                visit_handle(*handle, visitor);
            }
            visit_value(value, visitor);
        }
    }
}

unsafe fn trace_map_view(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    match &unsafe { managed_object(object) }.object {
        HeapObject::MapEntry { key, value } => {
            visit_value(*key, visitor);
            visit_value(*value, visitor);
        }
        HeapObject::MapKeys { map } => visit_value(*map, visitor),
        _ => {}
    }
}

unsafe fn trace_cell(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    if let HeapObject::Cell(value) = &managed.object {
        // Captured mutable Wren locals are semantic Values (the lowerer creates
        // cells as Type::Top), so a heap-tagged word here is always a real edge.
        visit_word(**value, visitor);
    }
}

unsafe fn trace_closure(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    if let HeapObject::Closure {
        captures,
        reference_indices,
        ..
    } = &managed.object
    {
        for &index in reference_indices.iter() {
            if let Some(&word) = captures.get(index as usize) {
                visit_word(word, visitor);
            }
        }
    }
}

unsafe fn trace_instance(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    if let HeapObject::Instance { fields, .. } = &managed.object {
        for &value in fields.iter() {
            visit_value(value, visitor);
        }
    }
}

unsafe fn trace_fiber_state(
    state: &FiberState,
    trace_metadata: &FiberTraceMetadata,
    visitor: &mut dyn FnMut(GcSlot),
) {
    if let Some(value) = state.starter {
        visit_value(value, visitor);
    }
    if let Some(value) = state.async_promise {
        visit_value(value, visitor);
    }
    if let Some(value) = state.awaiting_promise {
        visit_value(value, visitor);
    }
    visit_value(state.last_value, visitor);
    visit_value(state.error, visitor);
    if let Some(link) = state.link {
        visit_value(link.caller_object, visitor);
    }

    for (depth, frame) in state.frames.iter().enumerate() {
        if frame.root_map == NO_INDEX {
            continue;
        }
        let Some(maps) = trace_metadata.gc_maps.get(frame.function as usize) else {
            continue;
        };
        let Some(map) = maps.get(frame.root_map as usize) else {
            continue;
        };
        let Some(banks) = state.banks.get(depth) else {
            continue;
        };
        for reg in 0..frame.reg_count as usize {
            if map.general_is_reference(Reg(reg as u8)) {
                if let Some(&word) = state.registers.get(frame.reg_base + reg) {
                    visit_word(word, visitor);
                }
            }
        }
        for index in 0..ARGUMENT_REGISTER_COUNT {
            if (map.arguments & (1u16 << index)) != 0 {
                visit_word(banks.arguments[index], visitor);
            }
        }
        for index in 0..RETURN_REGISTER_COUNT {
            if (map.returns & (1u8 << index)) != 0 {
                visit_word(banks.returns[index], visitor);
            }
        }
        for index in 0..SPECIAL_REGISTER_COUNT {
            if (map.special & (1u16 << index)) != 0 {
                visit_word(banks.special[index], visitor);
            }
        }
    }
}

unsafe fn trace_promise(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    let HeapObject::Promise { state } = &managed.object else { return; };
    // SAFETY: GC invokes custom tracing only while all mutators are stopped.
    match unsafe { state.get() } {
        PromiseState::Pending { producer, waiters } => {
            if let Some(value) = producer {
                visit_value(*value, visitor);
            }
            for waiter in waiters {
                match *waiter {
                    PromiseWaiter::Resume { fiber_object, .. } => visit_value(fiber_object, visitor),
                    PromiseWaiter::Forward { promise } => visit_value(promise, visitor),
                }
            }
        }
        PromiseState::Fulfilled(value) | PromiseState::Rejected(value) => {
            visit_value(*value, visitor);
        }
    }
}

unsafe fn trace_async_task(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    let HeapObject::AsyncTaskObject { state, trace_metadata, promise, .. } = &managed.object else {
        return;
    };
    visit_value(*promise, visitor);
    // SAFETY: AsyncTaskObject owns stable Arc storage and GC runs at STW.
    unsafe { trace_fiber_state(state.get(), trace_metadata, visitor) };
}

unsafe fn trace_fiber(object: GcObject, visitor: &mut dyn FnMut(GcSlot)) {
    let managed = unsafe { managed_object(object) };
    let HeapObject::FiberObject { state, trace_metadata, .. } = &managed.object else {
        return;
    };
    // SAFETY: FiberObject owns stable Arc storage and GC runs at STW.
    unsafe { trace_fiber_state(state.get(), trace_metadata, visitor) };
}

unsafe fn cleanup_managed_object(object: GcObject) {
    let managed = unsafe { managed_object_mut(object) };
    handles().release(managed.handle);
    unsafe { std::ptr::drop_in_place(managed) };
}

macro_rules! gc_descriptor {
    ($name:ident, $label:literal, $trace:expr) => {
        static $name: GcDescriptor = GcDescriptor {
            name: $label,
            trace: $trace,
            cleanup: Some(cleanup_managed_object),
            finalizer: None,
        };
    };
}

gc_descriptor!(GC_STRING, "Wren.String", GcTraceLayout::Leaf);
gc_descriptor!(GC_RANGE, "Wren.Range", GcTraceLayout::Leaf);
gc_descriptor!(GC_RANDOM, "Wren.Random", GcTraceLayout::Leaf);
gc_descriptor!(GC_CLASS, "Wren.Class", GcTraceLayout::Leaf);
gc_descriptor!(GC_BUILTIN_CLASS, "Wren.BuiltinClass", GcTraceLayout::Leaf);
gc_descriptor!(GC_FOREIGN, "Wren.Foreign", GcTraceLayout::Leaf);
gc_descriptor!(GC_MODULE, "Wren.Module", GcTraceLayout::Leaf);
gc_descriptor!(GC_LIST, "Wren.List", GcTraceLayout::Custom(trace_list));
gc_descriptor!(GC_MAP, "Wren.Map", GcTraceLayout::Custom(trace_map));
gc_descriptor!(GC_MAP_VIEW, "Wren.MapView", GcTraceLayout::Custom(trace_map_view));
gc_descriptor!(GC_CELL, "Wren.Cell", GcTraceLayout::Custom(trace_cell));
gc_descriptor!(GC_CLOSURE, "Wren.Closure", GcTraceLayout::Custom(trace_closure));
gc_descriptor!(GC_INSTANCE, "Wren.Instance", GcTraceLayout::Custom(trace_instance));
gc_descriptor!(GC_PROMISE, "Wren.Promise", GcTraceLayout::Custom(trace_promise));
gc_descriptor!(GC_ASYNC_TASK, "Wren.AsyncTask", GcTraceLayout::Custom(trace_async_task));
gc_descriptor!(GC_FIBER, "Wren.Fiber", GcTraceLayout::Custom(trace_fiber));

#[inline(always)]
fn descriptor_for(object: &HeapObject) -> &'static GcDescriptor {
    match object {
        HeapObject::Random { .. } => &GC_RANDOM,
        HeapObject::String(_) => &GC_STRING,
        HeapObject::List(_) => &GC_LIST,
        HeapObject::Map(_) => &GC_MAP,
        HeapObject::MapEntry { .. } | HeapObject::MapKeys { .. } => &GC_MAP_VIEW,
        HeapObject::Range { .. } => &GC_RANGE,
        HeapObject::Cell(_) => &GC_CELL,
        HeapObject::Closure { .. } => &GC_CLOSURE,
        HeapObject::ClassObject { .. } => &GC_CLASS,
        HeapObject::BuiltinClass { .. } => &GC_BUILTIN_CLASS,
        HeapObject::Instance { .. } => &GC_INSTANCE,
        HeapObject::Foreign { .. } => &GC_FOREIGN,
        HeapObject::Promise { .. } => &GC_PROMISE,
        HeapObject::AsyncTaskObject { .. } => &GC_ASYNC_TASK,
        HeapObject::FiberObject { .. } => &GC_FIBER,
        HeapObject::ModuleObject { .. } => &GC_MODULE,
    }
}

#[inline]
fn external_bytes_for(object: &HeapObject) -> usize {
    match object {
        HeapObject::String(bytes) => bytes.len(),
        HeapObject::List(values) => values.capacity() * std::mem::size_of::<Value>(),
        HeapObject::Map(map) => map.entries.capacity() * std::mem::size_of::<(MapKey, Value)>(),
        HeapObject::Cell(_) => std::mem::size_of::<RegisterWord>(),
        HeapObject::Closure { captures, reference_indices, .. } => {
            captures.len() * std::mem::size_of::<RegisterWord>()
                + reference_indices.len() * std::mem::size_of::<u16>()
        },
        HeapObject::Instance { fields, .. } => fields.len() * std::mem::size_of::<Value>(),
        HeapObject::Foreign { bytes, .. } => bytes.capacity(),
        HeapObject::Promise { state } => {
            // SAFETY: this is queried by the owning mutator.
            match unsafe { state.get() } {
                PromiseState::Pending { waiters, .. } =>
                    waiters.capacity() * std::mem::size_of::<PromiseWaiter>(),
                PromiseState::Fulfilled(_) | PromiseState::Rejected(_) => 0,
            }
        }
        HeapObject::AsyncTaskObject { state, .. } | HeapObject::FiberObject { state, .. } => {
            // SAFETY: this is queried by the owning mutator before publication or
            // while it has exclusive VM access. GC workers never call this helper.
            fiber_state_external_bytes(unsafe { state.get() })
        }
        _ => 0,
    }
}

#[inline]
fn fiber_state_external_bytes(state: &FiberState) -> usize {
    std::mem::size_of::<FiberState>()
        + state.frames.capacity() * std::mem::size_of::<Frame>()
        + state.banks.capacity() * std::mem::size_of::<FrameBanks>()
        + state.registers.capacity() * std::mem::size_of::<RegisterWord>()
}

/// Per-mutator scratch roots used while constructing a managed object.
///
/// MMTk may collect at the beginning of an allocation. References already copied
/// into the Rust-side HeapObject being constructed are therefore not necessarily
/// visible in the VM stack map anymore. This stable provider protects exactly
/// those handles for the duration of the allocation without allocating a new
/// RootFrame/Arc on the hot path. The mutator writes the Vec only while Running;
/// MMTk reads it only after the stop-the-world rendezvous.
struct AllocationRootBuffer {
    handles: std::cell::UnsafeCell<Vec<u32>>,
}

impl AllocationRootBuffer {
    fn new() -> Self {
        Self { handles: std::cell::UnsafeCell::new(Vec::with_capacity(32)) }
    }

    #[inline]
    fn prepare(&self, object: &HeapObject) {
        // SAFETY: only the owning mutator writes this buffer while Running. Root
        // providers are visited only after all Running mutators have parked.
        let roots = unsafe { &mut *self.handles.get() };
        roots.clear();
        collect_heap_object_handles(object, roots);
    }

    #[inline]
    fn clear(&self) {
        // SAFETY: same ownership protocol as prepare().
        unsafe { &mut *self.handles.get() }.clear();
    }
}

unsafe impl Send for AllocationRootBuffer {}
unsafe impl Sync for AllocationRootBuffer {}

unsafe impl GcRootProvider for AllocationRootBuffer {
    fn visit(&self, visitor: &mut GcRootVisitor<'_>) {
        // SAFETY: collection only calls providers after the owning mutator has
        // parked, so the Vec cannot be concurrently mutated here.
        let roots = unsafe { &*self.handles.get() };
        for &handle in roots {
            if let Some(slot) = handles().gc_slot(handle) {
                visitor.slot(slot);
            }
        }
    }
}

#[inline]
fn push_value_handle(value: Value, roots: &mut Vec<u32>) {
    if let Some(handle) = value.heap_handle() {
        roots.push(handle);
    }
}

fn collect_fiber_state_handles(
    state: &FiberState,
    trace_metadata: &FiberTraceMetadata,
    roots: &mut Vec<u32>,
) {
    if let Some(value) = state.starter { push_value_handle(value, roots); }
    if let Some(value) = state.async_promise { push_value_handle(value, roots); }
    if let Some(value) = state.awaiting_promise { push_value_handle(value, roots); }
    push_value_handle(state.last_value, roots);
    push_value_handle(state.error, roots);
    if let Some(link) = state.link { push_value_handle(link.caller_object, roots); }
    for (depth, frame) in state.frames.iter().enumerate() {
        if frame.root_map == NO_INDEX { continue; }
        let Some(maps) = trace_metadata.gc_maps.get(frame.function as usize) else { continue; };
        let Some(map) = maps.get(frame.root_map as usize) else { continue; };
        let Some(banks) = state.banks.get(depth) else { continue; };
        for reg in 0..frame.reg_count as usize {
            if map.general_is_reference(Reg(reg as u8)) {
                if let Some(&word) = state.registers.get(frame.reg_base + reg) {
                    push_value_handle(Value::from_word(word), roots);
                }
            }
        }
        for index in 0..ARGUMENT_REGISTER_COUNT {
            if (map.arguments & (1u16 << index)) != 0 {
                push_value_handle(Value::from_word(banks.arguments[index]), roots);
            }
        }
        for index in 0..RETURN_REGISTER_COUNT {
            if (map.returns & (1u8 << index)) != 0 {
                push_value_handle(Value::from_word(banks.returns[index]), roots);
            }
        }
        for index in 0..SPECIAL_REGISTER_COUNT {
            if (map.special & (1u16 << index)) != 0 {
                push_value_handle(Value::from_word(banks.special[index]), roots);
            }
        }
    }
}

fn collect_heap_object_handles(object: &HeapObject, roots: &mut Vec<u32>) {
    match object {
        HeapObject::MapEntry { key, value } => {
            push_value_handle(*key, roots);
            push_value_handle(*value, roots);
        }
        HeapObject::MapKeys { map } => push_value_handle(*map, roots),
        HeapObject::List(values) => {
            for &value in values { push_value_handle(value, roots); }
        }
        HeapObject::Map(map) => {
            for (key, &value) in &map.entries {
                if let MapKey::Object(handle) = key { roots.push(*handle); }
                push_value_handle(value, roots);
            }
        }
        HeapObject::Cell(word) => push_value_handle(Value::from_word(**word), roots),
        HeapObject::Closure {
            captures,
            reference_indices,
            ..
        } => {
            for &index in reference_indices.iter() {
                if let Some(&word) = captures.get(index as usize) {
                    push_value_handle(Value::from_word(word), roots);
                }
            }
        }
        HeapObject::Instance { fields, .. } => {
            for &value in fields.iter() { push_value_handle(value, roots); }
        }
        HeapObject::Promise { state } => {
            // SAFETY: allocation root collection runs on the owning mutator.
            match unsafe { state.get() } {
                PromiseState::Pending { producer, waiters } => {
                    if let Some(value) = producer { push_value_handle(*value, roots); }
                    for waiter in waiters {
                        match *waiter {
                            PromiseWaiter::Resume { fiber_object, .. } => push_value_handle(fiber_object, roots),
                            PromiseWaiter::Forward { promise } => push_value_handle(promise, roots),
                        }
                    }
                }
                PromiseState::Fulfilled(value) | PromiseState::Rejected(value) => {
                    push_value_handle(*value, roots);
                }
            }
        }
        HeapObject::AsyncTaskObject { state, trace_metadata, promise, .. } => {
            push_value_handle(*promise, roots);
            collect_fiber_state_handles(unsafe { state.get() }, trace_metadata, roots);
        }
        HeapObject::FiberObject { state, trace_metadata, .. } => {
            // SAFETY: allocation happens on the owning mutator before publication.
            collect_fiber_state_handles(unsafe { state.get() }, trace_metadata, roots);
        }
        HeapObject::String(_)
        | HeapObject::Random { .. }
        | HeapObject::Range { .. }
        | HeapObject::ClassObject { .. }
        | HeapObject::BuiltinClass { .. }
        | HeapObject::Foreign { .. }
        | HeapObject::ModuleObject { .. } => {}
    }
}

struct Heap {
    collector: &'static GcCollector,
    mutator: GcMutator,
    allocation_roots: Arc<AllocationRootBuffer>,
    _allocation_root_registration: GcRootRegistration,
    /// Current fiber stack object. It is re-logged after every safepoint so all
    /// register/bank writes until the next safepoint are covered by StickyImmix's
    /// object remembered set without putting a barrier in every register store.
    stack_owner: Value,
    #[cfg(test)]
    allocations: Vec<Value>,
}

fn wren_gc_config() -> GcConfig {
    fn setting(name: &str, default: usize) -> usize {
        match std::env::var(name) {
            Ok(value) => value.parse().unwrap_or_else(|_| {
                panic!("{name} must be a non-negative integer, got {value:?}")
            }),
            Err(std::env::VarError::NotPresent) => default,
            Err(error) => panic!("invalid {name}: {error}"),
        }
    }
    fn megabytes(name: &str, default: usize) -> usize {
        setting(name, default >> 20).checked_mul(1 << 20)
            .unwrap_or_else(|| panic!("{name} overflows the address space"))
    }
    let mut config = GcConfig::default();
    config.heap_bytes = megabytes("WREN_GC_HEAP_MB", config.heap_bytes);
    config.workers = setting("WREN_GC_WORKERS", config.workers);
    config.external_gc_threshold = megabytes("WREN_GC_EXTERNAL_MB", config.external_gc_threshold);
    assert!(config.heap_bytes >= 8 << 20, "WREN_GC_HEAP_MB must be at least 8");
    assert!(config.workers > 0, "WREN_GC_WORKERS must be at least 1");
    config
}

impl Default for Heap {
    fn default() -> Self {
        let collector = GcCollector::initialize(wren_gc_config());
        let mut mutator = collector.attach();
        let allocation_roots = Arc::new(AllocationRootBuffer::new());
        let provider: Arc<dyn GcRootProvider> = allocation_roots.clone();
        let allocation_root_registration = mutator.register_roots(provider);
        Self {
            collector,
            mutator,
            allocation_roots,
            _allocation_root_registration: allocation_root_registration,
            stack_owner: Value::NULL,
            #[cfg(test)]
            allocations: Vec::new(),
        }
    }
}

impl Heap {
    #[inline]
    fn alloc(&mut self, object: HeapObject) -> Value {
        // Protect references which currently live only in the Rust-side object
        // being constructed. Mutator::allocate() may rendezvous for GC before it
        // returns the new MMTk object.
        self.allocation_roots.prepare(&object);
        let handle = handles().allocate_handle();
        let receiver_class = heap_object_receiver_class(&object);
        let descriptor = descriptor_for(&object);
        let external = external_bytes_for(&object);
        let gc_object = unsafe {
            self.mutator
                .allocate(descriptor, std::mem::size_of::<ManagedObject>())
        };
        unsafe {
            std::ptr::write(
                gc_object.payload().cast::<ManagedObject>(),
                ManagedObject {
                    handle,
                    receiver_class,
                    object,
                },
            );
        }
        handles().publish(handle, gc_object, unsafe { &managed_object(gc_object).object });
        self.allocation_roots.clear();
        if external != 0 {
            unsafe { self.mutator.set_external_bytes(gc_object, external) };
        }
        // Allocation can itself trigger a collection. Re-log the active fiber
        // before the interpreter performs any subsequent register writes.
        self.remember_stack_owner();
        let value = Value::from_heap(handle);
        #[cfg(test)]
        self.allocations.push(value);
        value
    }

    #[inline(always)]
    fn object(&self, value: Value) -> Option<GcObject> {
        handles().resolve(value.heap_handle()?)
    }

    #[inline(always)]
    fn get(&self, value: Value) -> Option<&HeapObject> {
        let object = self.object(value)?;
        Some(unsafe { &managed_object(object).object })
    }

    #[inline(always)]
    fn get_mut(&mut self, value: Value) -> Option<&mut HeapObject> {
        let object = self.object(value)?;
        Some(unsafe { &mut managed_object_mut(object).object })
    }

    #[inline(always)]
    fn receiver_class(&self, value: Value) -> Option<ReceiverClass> {
        let object = self.object(value)?;
        unsafe { managed_object(object).receiver_class }
    }

    #[inline(always)]
    fn refresh_jit_view(&self, value: Value) {
        let Some(handle) = value.heap_handle() else { return; };
        if let Some(object) = self.get(value) {
            handles().refresh(handle, object);
        }
    }

    #[inline]
    fn remember_if_reference(&mut self, owner: Value, value: Value) {
        if !value.is_heap() {
            return;
        }
        if let Some(owner) = self.object(owner) {
            unsafe { self.mutator.probable_write(owner) };
        }
    }

    #[inline]
    fn remember_owner(&mut self, owner: Value) {
        if let Some(owner) = self.object(owner) {
            unsafe { self.mutator.probable_write(owner) };
        }
    }

    #[inline]
    fn set_external_bytes(&mut self, owner: Value, bytes: usize) {
        if let Some(owner) = self.object(owner) {
            unsafe { self.mutator.set_external_bytes(owner, bytes) };
        }
    }

    #[inline]
    fn set_stack_owner(&mut self, owner: Value) {
        self.stack_owner = owner;
        self.remember_stack_owner();
    }

    #[inline]
    fn clear_stack_owner(&mut self) {
        self.stack_owner = Value::NULL;
    }

    #[inline]
    fn remember_stack_owner(&mut self) {
        if self.stack_owner.is_heap() {
            self.remember_owner(self.stack_owner);
        }
    }

    #[inline]
    fn safepoint(&mut self) {
        self.mutator.safepoint();
        self.remember_stack_owner();
    }

    fn collect_full(&mut self) {
        self.mutator.collect_full();
        self.remember_stack_owner();
    }

    fn register_roots(&mut self, provider: Arc<dyn GcRootProvider>) -> GcRootRegistration {
        self.mutator.register_roots(provider)
    }

    #[inline(always)]
    fn collection_count(&self) -> u64 {
        self.collector.collection_count()
    }

}

#[derive(Default)]
struct HandleRootSet {
    handles: StdMutex<Vec<u32>>,
}

impl HandleRootSet {
    fn push(&self, value: Value) {
        if let Some(handle) = value.heap_handle() {
            self.handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(handle);
        }
    }

    fn replace(&self, handles: Vec<u32>) {
        *self
            .handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = handles;
    }
}

unsafe impl GcRootProvider for HandleRootSet {
    fn visit(&self, visitor: &mut GcRootVisitor<'_>) {
        let roots = self
            .handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for &handle in roots.iter() {
            if let Some(slot) = handles().gc_slot(handle) {
                visitor.slot(slot);
            }
        }
    }
}

struct SingleHandleRoot {
    handle: u32,
}

unsafe impl GcRootProvider for SingleHandleRoot {
    fn visit(&self, visitor: &mut GcRootVisitor<'_>) {
        if let Some(slot) = handles().gc_slot(self.handle) {
            visitor.slot(slot);
        }
    }
}

/// RAII root for embedding/native code that keeps a Wren Value across another
/// managed allocation. Dropping the guard immediately removes the root.
pub struct HostRoot {
    value: Value,
    _registration: Option<GcRootRegistration>,
}

impl HostRoot {
    #[inline(always)]
    pub const fn value(&self) -> Value {
        self.value
    }
}

// =============================================================================
// Public host interface

// =============================================================================

/// Mutable runtime view exposed only on slow native/foreign paths.
pub struct HostContext<'a> {
    heap: &'a mut Heap,
}

impl HostContext<'_> {
    /// Keep `value` alive across subsequent HostContext allocations. Scalars are
    /// returned in a zero-cost guard with no collector registration.
    pub fn root(&mut self, value: Value) -> HostRoot {
        let registration = value.heap_handle().map(|handle| {
            let provider: Arc<dyn GcRootProvider> = Arc::new(SingleHandleRoot { handle });
            self.heap.register_roots(provider)
        });
        HostRoot { value, _registration: registration }
    }

    pub fn string_bytes(&self, value: Value) -> Option<&[u8]> {
        match self.heap.get(value) {
            Some(HeapObject::String(string)) => Some(string.as_ref()),
            _ => None,
        }
    }

    pub fn string_utf8(&self, value: Value) -> Option<&str> {
        std::str::from_utf8(self.string_bytes(value)?).ok()
    }

    pub fn foreign_bytes(&self, value: Value) -> Option<&[u8]> {
        match self.heap.get(value) {
            Some(HeapObject::Foreign { bytes, .. }) => Some(bytes.as_slice()),
            _ => None,
        }
    }

    pub fn foreign_bytes_mut(&mut self, value: Value) -> Option<&mut [u8]> {
        match self.heap.get_mut(value) {
            Some(HeapObject::Foreign { bytes, .. }) => Some(bytes.as_mut_slice()),
            _ => None,
        }
    }

    pub fn allocate_string(&mut self, value: impl AsRef<str>) -> Value {
        self.heap.alloc(HeapObject::String(Arc::<[u8]>::from(value.as_ref().as_bytes())))
    }

    pub fn allocate_bytes(&mut self, value: impl AsRef<[u8]>) -> Value {
        self.heap.alloc(HeapObject::String(Arc::<[u8]>::from(value.as_ref())))
    }

    pub fn allocate_list(&mut self, values: Vec<Value>) -> Value {
        self.heap.alloc(HeapObject::List(values))
    }

    pub fn allocate_map(&mut self) -> Value {
        self.heap.alloc(HeapObject::Map(WrenMap::default()))
    }

    pub fn allocate_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value {
        self.heap.alloc(HeapObject::Range { from, to, inclusive })
    }

    pub fn allocate_random(&mut self, state: u64) -> Value {
        self.heap.alloc(HeapObject::Random { state })
    }

    pub fn random_u64(&mut self, random: Value) -> Result<u64, RuntimeError> {
        let state = match self.heap.get_mut(random) {
            Some(HeapObject::Random { state }) => state,
            _ => return Err(RuntimeError::new("receiver is not Random")),
        };
        // SplitMix64: per-instance state; wrapping arithmetic is intentional.
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        Ok(value ^ (value >> 31))
    }

    #[inline]
    pub fn list_len(&self, value: Value) -> Option<usize> {
        match self.heap.get(value) { Some(HeapObject::List(list)) => Some(list.len()), _ => None }
    }

    #[inline]
    pub fn list_get(&self, value: Value, index: usize) -> Option<Value> {
        match self.heap.get(value) { Some(HeapObject::List(list)) => list.get(index).copied(), _ => None }
    }

    pub fn list_set(&mut self, value: Value, index: usize, item: Value) -> Result<Value, RuntimeError> {
        self.heap.remember_if_reference(value, item);
        match self.heap.get_mut(value) {
            Some(HeapObject::List(list)) => {
                let slot = list.get_mut(index).ok_or_else(|| RuntimeError::new("list index out of bounds"))?;
                *slot = item;
                Ok(item)
            }
            _ => Err(RuntimeError::new("receiver is not a List")),
        }
    }

    pub fn list_push(&mut self, value: Value, item: Value) -> Result<Value, RuntimeError> {
        self.heap.remember_if_reference(value, item);
        let resized = match self.heap.get_mut(value) {
            Some(HeapObject::List(list)) => {
                let old = list.capacity();
                list.push(item);
                (list.capacity() != old).then_some(list.capacity())
            }
            _ => return Err(RuntimeError::new("receiver is not a List")),
        };
        if let Some(capacity) = resized {
            self.heap.set_external_bytes(value, capacity * std::mem::size_of::<Value>());
        }
        self.heap.refresh_jit_view(value);
        Ok(item)
    }

    pub fn list_insert(&mut self, value: Value, index: usize, item: Value) -> Result<Value, RuntimeError> {
        self.heap.remember_if_reference(value, item);
        let resized = match self.heap.get_mut(value) {
            Some(HeapObject::List(list)) if index <= list.len() => {
                let old = list.capacity();
                list.insert(index, item);
                (list.capacity() != old).then_some(list.capacity())
            }
            Some(HeapObject::List(_)) => return Err(RuntimeError::new("list index out of bounds")),
            _ => return Err(RuntimeError::new("receiver is not a List")),
        };
        if let Some(capacity) = resized {
            self.heap.set_external_bytes(value, capacity * std::mem::size_of::<Value>());
        }
        self.heap.refresh_jit_view(value);
        Ok(item)
    }

    pub fn list_remove_at(&mut self, value: Value, index: usize) -> Result<Value, RuntimeError> {
        let removed = match self.heap.get_mut(value) {
            Some(HeapObject::List(list)) if index < list.len() => list.remove(index),
            Some(HeapObject::List(_)) => return Err(RuntimeError::new("list index out of bounds")),
            _ => return Err(RuntimeError::new("receiver is not a List")),
        };
        self.heap.refresh_jit_view(value);
        Ok(removed)
    }

    pub fn list_clear(&mut self, value: Value) -> Result<(), RuntimeError> {
        match self.heap.get_mut(value) {
            Some(HeapObject::List(list)) => list.clear(),
            _ => return Err(RuntimeError::new("receiver is not a List")),
        }
        self.heap.refresh_jit_view(value);
        Ok(())
    }

    fn map_key(&self, value: Value) -> Result<MapKey, RuntimeError> {
        if value.is_null() { return Ok(MapKey::Null); }
        if let Some(value) = value.as_bool() { return Ok(MapKey::Bool(value)); }
        if let Some(value) = value.as_num() {
            let normalized = if value == 0.0 { 0.0 } else { value };
            return Ok(MapKey::Num(normalized.to_bits()));
        }
        if let Some(handle) = value.heap_handle() {
            if let Some(HeapObject::String(string)) = self.heap.get(value) {
                return Ok(MapKey::String(string.clone()));
            }
            return Ok(MapKey::Object(handle));
        }
        Err(RuntimeError::new("value cannot be used as a map key"))
    }

    pub fn map_len(&self, value: Value) -> Option<usize> {
        match self.heap.get(value) { Some(HeapObject::Map(map)) => Some(map.entries.len()), _ => None }
    }

    pub fn map_get(&self, value: Value, key: Value) -> Result<Value, RuntimeError> {
        let key = self.map_key(key)?;
        match self.heap.get(value) {
            Some(HeapObject::Map(map)) => Ok(map.entries.get(&key).copied().unwrap_or(Value::NULL)),
            _ => Err(RuntimeError::new("receiver is not a Map")),
        }
    }

    pub fn map_set(&mut self, value: Value, key: Value, item: Value) -> Result<Value, RuntimeError> {
        let encoded_key = self.map_key(key)?;
        self.heap.remember_if_reference(value, key);
        self.heap.remember_if_reference(value, item);
        let resized = match self.heap.get_mut(value) {
            Some(HeapObject::Map(map)) => {
                let old = map.entries.capacity();
                map.entries.insert(encoded_key, item);
                (map.entries.capacity() != old).then_some(map.entries.capacity())
            }
            _ => return Err(RuntimeError::new("receiver is not a Map")),
        };
        if let Some(capacity) = resized {
            self.heap.set_external_bytes(
                value,
                capacity * std::mem::size_of::<(MapKey, Value)>(),
            );
        }
        self.heap.refresh_jit_view(value);
        Ok(item)
    }

    pub fn map_remove(&mut self, value: Value, key: Value) -> Result<Value, RuntimeError> {
        let key = self.map_key(key)?;
        let removed = match self.heap.get_mut(value) {
            Some(HeapObject::Map(map)) => map.entries.remove(&key).unwrap_or(Value::NULL),
            _ => return Err(RuntimeError::new("receiver is not a Map")),
        };
        self.heap.refresh_jit_view(value);
        Ok(removed)
    }

    pub fn map_clear(&mut self, value: Value) -> Result<(), RuntimeError> {
        match self.heap.get_mut(value) {
            Some(HeapObject::Map(map)) => map.entries.clear(),
            _ => return Err(RuntimeError::new("receiver is not a Map")),
        }
        self.heap.refresh_jit_view(value);
        Ok(())
    }

    pub fn map_contains_key(&self, value: Value, key: Value) -> Result<bool, RuntimeError> {
        let key = self.map_key(key)?;
        match self.heap.get(value) {
            Some(HeapObject::Map(map)) => Ok(map.entries.contains_key(&key)),
            _ => Err(RuntimeError::new("receiver is not a Map")),
        }
    }

    pub fn map_nth_key(&mut self, value: Value, index: usize) -> Result<Value, RuntimeError> {
        let map = match self.heap.get(value) {
            Some(HeapObject::Map(map)) => map,
            _ => return Err(RuntimeError::new("receiver is not a Map")),
        };
        let key = map.entries.keys().nth(index).cloned().ok_or_else(|| RuntimeError::new("map iterator out of range"))?;
        Ok(match key {
            MapKey::Null => Value::NULL,
            MapKey::Bool(value) => Value::from_bool(value),
            MapKey::Num(bits) => Value::from_num(f64::from_bits(bits)),
            MapKey::String(bytes) => self.allocate_bytes(bytes),
            MapKey::Object(handle) => Value::from_heap(handle),
        })
    }

    pub fn map_nth_entry(&mut self, map: Value, index: usize) -> Result<Value, RuntimeError> {
        let value = self.map_nth_value(map, index)?;
        let value = self.root(value);
        let key = self.map_nth_key(map, index)?;
        Ok(self.heap.alloc(HeapObject::MapEntry { key, value: value.value() }))
    }

    pub fn map_keys(&mut self, map: Value) -> Result<Value, RuntimeError> {
        self.map_len(map).ok_or_else(|| RuntimeError::new("receiver is not a Map"))?;
        Ok(self.heap.alloc(HeapObject::MapKeys { map }))
    }

    pub fn map_keys_source(&self, value: Value) -> Result<Value, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::MapKeys { map }) => Ok(*map),
            _ => Err(RuntimeError::new("receiver is not a map keys view")),
        }
    }

    pub fn map_entry_parts(&self, entry: Value) -> Result<(Value, Value), RuntimeError> {
        match self.heap.get(entry) {
            Some(HeapObject::MapEntry { key, value }) => Ok((*key, *value)),
            _ => Err(RuntimeError::new("receiver is not a map entry")),
        }
    }

    pub fn map_nth_value(&self, value: Value, index: usize) -> Result<Value, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::Map(map)) => map.entries.values().nth(index).copied().ok_or_else(|| RuntimeError::new("map iterator out of range")),
            _ => Err(RuntimeError::new("receiver is not a Map")),
        }
    }

    pub fn range_parts(&self, value: Value) -> Option<(f64, f64, bool)> {
        match self.heap.get(value) {
            Some(HeapObject::Range { from, to, inclusive }) => Some((*from, *to, *inclusive)),
            _ => None,
        }
    }

    pub fn format_value(&self, value: Value) -> String {
        let mut seen = HashSet::new();
        self.format_value_inner(value, &mut seen)
    }

    fn format_value_inner(&self, value: Value, seen: &mut HashSet<u32>) -> String {
        if value.is_null() { return "null".to_owned(); }
        if let Some(value) = value.as_bool() { return value.to_string(); }
        if let Some(value) = value.as_num() {
            if value.fract() == 0.0 && value.is_finite() { return format!("{value:.0}"); }
            return value.to_string();
        }
        let Some(handle) = value.heap_handle() else { return format!("{value:?}"); };
        match self.heap.get(value) {
            Some(HeapObject::Random { .. }) => "<random>".to_owned(),
            Some(HeapObject::String(bytes)) => String::from_utf8_lossy(bytes).into_owned(),
            Some(HeapObject::List(values)) => {
                if !seen.insert(handle) { return "[...]".to_owned(); }
                let text = values.iter().map(|v| self.format_value_inner(*v, seen)).collect::<Vec<_>>().join(", " );
                seen.remove(&handle);
                format!("[{text}]")
            }
            Some(HeapObject::Map(map)) => {
                if !seen.insert(handle) { return "{...}".to_owned(); }
                let text = map.entries.values().map(|v| self.format_value_inner(*v, seen)).collect::<Vec<_>>().join(", " );
                seen.remove(&handle);
                format!("{{{text}}}")
            }
            Some(HeapObject::Range { from, to, inclusive }) => {
                let op = if *inclusive { ".." } else { "..." };
                format!("{from}{op}{to}")
            }
            Some(HeapObject::MapEntry { key, value }) => {
                if !seen.insert(handle) { return "<map entry ...>".to_owned(); }
                let text = format!("{}: {}", self.format_value_inner(*key, seen), self.format_value_inner(*value, seen));
                seen.remove(&handle);
                text
            }
            Some(HeapObject::MapKeys { .. }) => "<map keys>".to_owned(),
            Some(HeapObject::Closure { .. }) => "<fn>".to_owned(),
            Some(HeapObject::ClassObject { class_slot }) => format!("<class#{class_slot}>"),
            Some(HeapObject::BuiltinClass { builtin }) => builtin.name().to_owned(),
            Some(HeapObject::Instance { class_slot, .. }) => format!("<instance class#{class_slot}>"),
            Some(HeapObject::Foreign { class_slot, .. }) => format!("<foreign class#{class_slot}>"),
            Some(HeapObject::Promise { state }) => match unsafe { state.get() } {
                PromiseState::Pending { .. } => "<promise:pending>".to_owned(),
                PromiseState::Fulfilled(_) => "<promise:fulfilled>".to_owned(),
                PromiseState::Rejected(_) => "<promise:rejected>".to_owned(),
            },
            Some(HeapObject::AsyncTaskObject { task, .. }) => format!("<async-task#{task}>"),
            Some(HeapObject::FiberObject { fiber, .. }) => format!("<fiber#{fiber}>"),
            Some(HeapObject::ModuleObject { module_slot }) => format!("<module#{module_slot}>"),
            Some(HeapObject::Cell(_)) => "<cell>".to_owned(),
            None => "<dead object>".to_owned(),
        }
    }
}

pub trait Host {
    fn standard_numeric_method(&self, _primitive: PrimitiveId, _name: &str, _arity: u8) -> bool { false }
    fn call_primitive(
        &mut self,
        _context: &mut HostContext<'_>,
        primitive: PrimitiveId,
        _args: &[Value],
    ) -> Result<Value, RuntimeError> {
        Err(RuntimeError::new(format!(
            "primitive {:?} is not registered",
            primitive
        )))
    }

    fn call_foreign(
        &mut self,
        _context: &mut HostContext<'_>,
        foreign: ForeignFnId,
        _args: &[Value],
    ) -> Result<Value, RuntimeError> {
        Err(RuntimeError::new(format!(
            "foreign function {:?} is not registered",
            foreign
        )))
    }

    fn safepoint(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn osr_point(&mut self, _function: FuncId, _word_pc: BytecodePc) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn breakpoint(&mut self, _function: FuncId, _word_pc: BytecodePc) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[derive(Default)]
pub struct NoHost;
impl Host for NoHost {}

// =============================================================================
// Errors and execution result
// =============================================================================

#[derive(Clone, Debug)]
pub struct RuntimeError {
    pub message: String,
    pub function: Option<FuncId>,
    pub pc: Option<BytecodePc>,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            function: None,
            pc: None,
        }
    }

    fn at(mut self, function: FuncId, pc: BytecodePc) -> Self {
        self.function = Some(function);
        self.pc = Some(pc);
        self
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.function, self.pc) {
            (Some(function), Some(pc)) => write!(f, "{} at {:?}:{:?}", self.message, function, pc),
            _ => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for RuntimeError {}

#[derive(Clone, Debug)]
pub enum ExecutionResult {
    Returned(Value),
    Suspended(Value),
}

// =============================================================================
// Prepared / threaded bytecode
// =============================================================================

type Handler = unsafe fn(&mut Interpreter, *const PreparedOp);
const NO_INDEX: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
struct PreparedOp {
    a: u8,
    b: u8,
    c: u8,
    flags: u8,
    x: u32,
    y: u32,
    next: u32,
    args_offset: u32,
    raw_pc: u32,
    gc_map: u32,
}

impl Default for PreparedOp {
    fn default() -> Self {
        Self {
            a: 0,
            b: 0,
            c: 0,
            flags: 0,
            x: 0,
            y: 0,
            next: 0,
            args_offset: 0,
            raw_pc: 0,
            gc_map: NO_INDEX,
        }
    }
}

#[derive(Clone, Debug)]
enum PreparedSwitch {
    Linear {
        default_target: u32,
        cases: Vec<(i64, u32)>,
    },
    Dense {
        default_target: u32,
        min: i64,
        targets: Vec<u32>,
    },
    Binary {
        default_target: u32,
        cases: Vec<(i64, u32)>,
    },
}

impl PreparedSwitch {
    #[inline]
    fn target(&self, value: i64) -> u32 {
        match self {
            Self::Linear {
                default_target,
                cases,
            } => {
                for &(candidate, target) in cases {
                    if candidate == value {
                        return target;
                    }
                }
                *default_target
            }
            Self::Dense {
                default_target,
                min,
                targets,
            } => {
                let index = value.wrapping_sub(*min);
                if index >= 0 && (index as usize) < targets.len() {
                    targets[index as usize]
                } else {
                    *default_target
                }
            }
            Self::Binary {
                default_target,
                cases,
            } => cases
                .binary_search_by_key(&value, |&(candidate, _)| candidate)
                .map(|index| cases[index].1)
                .unwrap_or(*default_target),
        }
    }
}

struct PreparedFunction {
    handlers: Vec<Handler>,
    ops: Vec<PreparedOp>,
    original_ops: Vec<PreparedOp>,
    call_args: Vec<Reg>,
    switches: Vec<PreparedSwitch>,
    gc_maps: Vec<ReferenceMap>,
    /// Synthetic precise map for a frame that has been initialized but has not
    /// executed its first GC-capable opcode yet. Async tasks can remain queued in
    /// exactly this state, so NO_INDEX would make their entry values invisible.
    entry_gc_map: u32,
    word_to_op: Vec<u32>,
    deopt_resume: FastMap<u32, u32>,
    frame: FrameLayout,
    id: FuncId,
    /// Class whose LoadField/StoreField operands are still class-local.
    field_owner: Option<ClassId>,
    /// Prepared-op indices whose x operand contains an instance FieldSlot.
    field_access_ops: Vec<u32>,
    /// Set after CreateClass relocates local slots by the inherited field base.
    field_slots_relocated: bool,
    /// One byte per prepared loop header: 0 enables entry, 1..254 counts down
    /// baseline backedges during a temporary cooldown, 255 permanently disables
    /// an unsupported header. Skipped entries bypass the JIT manager entirely.
    jit_disabled_headers: Vec<u8>,
    /// Dense TraceId cached for a compiled loop header. NO_INDEX means the header
    /// is still cold/uncompiled. This bypasses the loop-state HashMap on every
    /// subsequent hot backedge once tiering succeeds.
    jit_trace_headers: Vec<u32>,
    /// Invoke op indices patched to the stock-Num fast handler.
    /// Method rebinding restores these sites to the generic invoke handler.
    quickened_numeric_ops: Vec<u32>,
    /// Generic monomorphic invoke fast paths. Indexed by prepared-op index so a
    /// hot call site can guard one ReceiverClass and call its target without
    /// touching the function/IC metadata hierarchy.
    quick_invokes: Vec<QuickInvokeSite>,
    quickened_invoke_ops: Vec<u32>,
    /// Predecoded frame-entry destinations. u16::MAX means the corresponding
    /// source value was dead after bytecode allocation and needs no GPR copy.
    entry_this_dst: u16,
    entry_arg_dsts: [u16; ARGUMENT_REGISTER_COUNT],
    needs_argument_bank: bool,
    needs_return_bank: bool,
    module_slot: u32,
    kind: FunctionKind,
    is_async: bool,
    arity: u8,
    return_kind: RegisterKind,
    capture_count: u16,
    /// Capture indices whose physical register words contain tagged Wren Values.
    /// The Box keeps this table at a stable address for moving GC object tracing.
    capture_reference_indices: Box<[u16]>,
}

struct PreparedProgram {
    functions: Vec<PreparedFunction>,
    function_by_id: FastMap<FuncId, u32>,
    class_by_id: FastMap<ClassId, u32>,
    module_by_id: FastMap<ModuleId, u32>,
    module_by_name: FastMap<StringId, u32>,
}

#[derive(Clone, Copy)]
struct TempOp {
    opcode: Opcode,
    a: u8,
    b: u8,
    c: u8,
    x: u32,
    y: u32,
    raw_pc: u32,
    raw_len: u32,
    args_offset: u32,
    gc_map: u32,
}

impl PreparedProgram {
    fn prepare(program: &BytecodeProgram) -> Result<Self, RuntimeError> {
        let function_by_id: FastMap<_, _> = program
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.id, index as u32))
            .collect();
        let class_by_id: FastMap<_, _> = program
            .metadata
            .classes
            .iter()
            .enumerate()
            .map(|(index, class)| (class.id, index as u32))
            .collect();
        let module_by_id: FastMap<_, _> = program
            .metadata
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| (module.id, index as u32))
            .collect();
        let module_by_name: FastMap<_, _> = program
            .metadata
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| (module.name, index as u32))
            .collect();

        let mut functions = Vec::with_capacity(program.functions.len());
        for function in &program.functions {
            functions.push(Self::prepare_function(
                function,
                &function_by_id,
                &class_by_id,
                &module_by_id,
            )?);
        }

        Ok(Self {
            functions,
            function_by_id,
            class_by_id,
            module_by_id,
            module_by_name,
        })
    }

    fn prepare_function(
        function: &BytecodeFunction,
        function_by_id: &FastMap<FuncId, u32>,
        class_by_id: &FastMap<ClassId, u32>,
        module_by_id: &FastMap<ModuleId, u32>,
    ) -> Result<PreparedFunction, RuntimeError> {
        let code = &function.chunk.code;
        let mut word_to_op = vec![NO_INDEX; code.len() + 1];
        let mut temps = Vec::<TempOp>::new();
        let mut call_args = Vec::<Reg>::new();

        let mut safepoint_by_pc = FastMap::<u32, u32>::default();
        let mut gc_maps = Vec::with_capacity(function.chunk.safepoints.len());
        for entry in &function.chunk.safepoints {
            let index = gc_maps.len() as u32;
            gc_maps.push(entry.references.clone());
            safepoint_by_pc.insert(entry.pc.0, index);
        }

        let mut raw_pc = 0usize;
        while raw_pc < code.len() {
            let first = code[raw_pc];
            let opcode = first.opcode().ok_or_else(|| {
                RuntimeError::new(format!("invalid opcode {} in {:?}", first.0 as u8, function.id))
            })?;
            let len = instruction_word_len(first).map_err(|error| {
                RuntimeError::new(format!("bytecode decode error in {:?}: {}", function.id, error.message))
            })?;
            if raw_pc + len > code.len() {
                return Err(RuntimeError::new(format!(
                    "truncated instruction {:?} at word {}",
                    opcode, raw_pc
                )));
            }

            let op_index = temps.len() as u32;
            word_to_op[raw_pc] = op_index;
            let x = if len >= 2 { code[raw_pc + 1].0 } else { 0 };
            let y = if len >= 3 { code[raw_pc + 2].0 } else { 0 };

            // Calls with up to four arguments carry their register IDs directly
            // in PreparedOp::args_offset. This removes the call-argument side-table
            // load from the overwhelmingly common 0..=4 arity case without growing
            // PreparedOp. Larger calls keep the old side-table representation.
            let is_call = matches!(
                opcode,
                Opcode::Invoke
                    | Opcode::InvokeSuper
                    | Opcode::CallDirect
                    | Opcode::CallClosure
                    | Opcode::CallPrimitive
                    | Opcode::CallForeign
            );
            let args_offset = if is_call {
                let count = first.c() as usize;
                if count <= 4 {
                    let mut packed_regs = 0u32;
                    for index in 0..count {
                        let packed = code[raw_pc + 2 + index / 4].0;
                        let reg = ((packed >> ((index % 4) * 8)) & 0xff) as u8;
                        packed_regs |= (reg as u32) << (index * 8);
                    }
                    packed_regs
                } else {
                    let offset = call_args.len() as u32;
                    for index in 0..count {
                        let packed = code[raw_pc + 2 + index / 4].0;
                        call_args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
                    }
                    offset
                }
            } else if opcode == Opcode::MakeClosure {
                // Closure capture lists may be large and use b/c as a 16-bit count,
                // so keep them in the side table regardless of size.
                let offset = call_args.len() as u32;
                let count = first.bx() as usize;
                for index in 0..count {
                    let packed = code[raw_pc + 2 + index / 4].0;
                    call_args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
                }
                offset
            } else {
                0
            };

            temps.push(TempOp {
                opcode,
                a: first.a(),
                b: first.b(),
                c: first.c(),
                x,
                y,
                raw_pc: raw_pc as u32,
                raw_len: len as u32,
                args_offset,
                gc_map: safepoint_by_pc.get(&(raw_pc as u32)).copied().unwrap_or(NO_INDEX),
            });
            raw_pc += len;
        }
        word_to_op[code.len()] = temps.len() as u32;

        let map_word_target = |word: u32| -> Result<u32, RuntimeError> {
            word_to_op
                .get(word as usize)
                .copied()
                .filter(|&index| index != NO_INDEX)
                .ok_or_else(|| RuntimeError::new(format!("branch target word {word} is not an instruction boundary")))
        };

        let mut switches = Vec::with_capacity(function.chunk.switch_tables.len());
        for table in &function.chunk.switch_tables {
            switches.push(prepare_switch(table, &map_word_target)?);
        }

        let mut deopt_resume = FastMap::default();
        for point in &function.chunk.deopts {
            deopt_resume.insert(point.id.0, map_word_target(point.resume_pc.0)?);
        }

        let mut handlers = Vec::with_capacity(temps.len());
        let mut ops = Vec::with_capacity(temps.len());
        for (index, temp) in temps.iter().copied().enumerate() {
            let mut op = PreparedOp {
                a: temp.a,
                b: temp.b,
                c: temp.c,
                flags: 0,
                x: temp.x,
                y: temp.y,
                next: (index + 1) as u32,
                args_offset: temp.args_offset,
                raw_pc: temp.raw_pc,
                gc_map: temp.gc_map,
            };

            match temp.opcode {
                Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue => {
                    let end = temp.raw_pc.wrapping_add(temp.raw_len);
                    let target_word = (end as i64 + temp.x as i32 as i64) as u32;
                    op.x = map_word_target(target_word)?;
                }
                Opcode::CallDirect | Opcode::MakeClosure => {
                    let function = FuncId(temp.x);
                    op.x = *function_by_id.get(&function).ok_or_else(|| {
                        RuntimeError::new(format!("unknown function {:?}", function))
                    })?;
                }
                Opcode::CreateClass | Opcode::AllocInstance | Opcode::AllocForeign => {
                    let class = ClassId(temp.x);
                    op.x = *class_by_id.get(&class).ok_or_else(|| {
                        RuntimeError::new(format!("unknown class {:?}", class))
                    })?;
                }
                Opcode::LoadModuleVar | Opcode::StoreModuleVar => {
                    let module = ModuleId(temp.x);
                    op.x = *module_by_id.get(&module).ok_or_else(|| {
                        RuntimeError::new(format!("unknown module {:?}", module))
                    })?;
                }
                _ => {}
            }

            handlers.push(handler_for_prepared(temp.opcode, temp.c));
            ops.push(op);
        }

        // Superinstructions are installed only when the second instruction is
        // not a branch/switch/deopt/resume target. They keep the same op index
        // space, so no target patching is required.
        let mut target_ops = HashSet::<u32>::new();
        for (index, op) in ops.iter().enumerate() {
            if matches!(temps[index].opcode, Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue) {
                target_ops.insert(op.x);
            }
        }
        for switch in &switches {
            match switch {
                PreparedSwitch::Linear { default_target, cases }
                | PreparedSwitch::Binary { default_target, cases } => {
                    target_ops.insert(*default_target);
                    for &(_, target) in cases { target_ops.insert(target); }
                }
                PreparedSwitch::Dense { default_target, targets, .. } => {
                    target_ops.insert(*default_target);
                    target_ops.extend(targets.iter().copied());
                }
            }
        }
        target_ops.extend(deopt_resume.values().copied());
        for resume in &function.chunk.resume_points {
            if let Ok(target) = map_word_target(resume.pc.0) { target_ops.insert(target); }
        }

        let original_ops = ops.clone();
        let field_access_ops: Vec<u32> = temps
            .iter()
            .enumerate()
            .filter_map(|(index, temp)| {
                matches!(temp.opcode, Opcode::LoadField | Opcode::StoreField)
                    .then_some(index as u32)
            })
            .collect();

        install_superinstructions(&temps, &mut handlers, &mut ops, &target_ops);

        // Switch tables are immutable after preparation. Store their direct
        // address in the executable op stream; original_ops retains the logical
        // table index for trace recording. Moving the Vec later does not move its
        // heap buffer, so these pointers remain stable for PreparedFunction life.
        for (index, temp) in temps.iter().enumerate() {
            if temp.opcode != Opcode::Switch { continue; }
            let table_index = ops[index].x as usize;
            let pointer = unsafe { switches.get_unchecked(table_index) as *const PreparedSwitch } as usize as u64;
            ops[index].x = pointer as u32;
            ops[index].y = (pointer >> 32) as u32;
        }

        let module_slot = *module_by_id.get(&function.module).ok_or_else(|| {
            RuntimeError::new(format!("unknown module {:?} for {:?}", function.module, function.id))
        })?;
        let op_count = ops.len();
        let jit_disabled_headers = vec![0; op_count];
        let jit_trace_headers = vec![NO_INDEX; op_count];

        // FrameLayout bindings are immutable. Decode their tiny IncomingSource
        // enum once so every call can copy receiver/arguments straight into the
        // callee GPR bank with no Vec walk and no per-binding match.
        let mut entry_this_dst = u16::MAX;
        let mut entry_arg_dsts = [u16::MAX; ARGUMENT_REGISTER_COUNT];
        for binding in &function.frame.parameters {
            match binding.source {
                IncomingSource::This => entry_this_dst = binding.destination.0 as u16,
                IncomingSource::Argument(arg) => {
                    entry_arg_dsts[arg.index()] = binding.destination.0 as u16;
                }
            }
        }

        // Async calls create their frame before the task is scheduled. A GC can
        // therefore observe this frame before any bytecode safepoint has installed
        // a normal stack map. Build one precise entry map from the physical frame
        // bindings plus the special registers initialized by push_frame().
        let mut entry_references = ReferenceMap::default();
        for binding in &function.frame.parameters {
            if binding.kind == RegisterKind::Value {
                entry_references.set_general(binding.destination, true);
            }
        }
        for binding in &function.frame.captures {
            if binding.kind == RegisterKind::Value {
                entry_references.set_general(binding.destination, true);
            }
        }
        if function_kind_uses_this(function.kind) {
            entry_references.special |= 1u16 << SpecialReg::This.index();
        }
        if function.kind == FunctionKind::Closure {
            entry_references.special |= 1u16 << SpecialReg::Closure.index();
        }
        entry_references.special |= 1u16 << SpecialReg::Module.index();
        entry_references.special |= 1u16 << SpecialReg::Fiber.index();
        let entry_gc_map = gc_maps.len() as u32;
        gc_maps.push(entry_references);

        let deopt_uses_argument = function.chunk.deopts.iter().any(|point| {
            point.slots.iter().chain(point.inlined_frames.iter().flat_map(|frame| frame.slots.iter()))
                .any(|slot| matches!(slot.location, DeoptLocation::Argument(_)))
        });
        let deopt_uses_return = function.chunk.deopts.iter().any(|point| {
            point.slots.iter().chain(point.inlined_frames.iter().flat_map(|frame| frame.slots.iter()))
                .any(|slot| matches!(slot.location, DeoptLocation::Return(_)))
        });
        let needs_argument_bank = temps.iter().any(|temp| {
            matches!(temp.opcode, Opcode::MoveFromArg | Opcode::MoveToArg)
        }) || gc_maps.iter().any(|map| map.arguments != 0) || deopt_uses_argument;
        let needs_return_bank = temps.iter().any(|temp| {
            matches!(temp.opcode, Opcode::MoveFromRet | Opcode::MoveToRet)
        }) || gc_maps.iter().any(|map| map.returns != 0) || deopt_uses_return;

        let capture_reference_indices = function
            .frame
            .captures
            .iter()
            .filter(|binding| binding.kind == RegisterKind::Value)
            .map(|binding| binding.capture_index)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(PreparedFunction {
            handlers,
            ops,
            original_ops,
            call_args,
            switches,
            gc_maps,
            entry_gc_map,
            word_to_op,
            deopt_resume,
            frame: function.frame.clone(),
            id: function.id,
            field_owner: function.field_owner,
            field_access_ops,
            field_slots_relocated: function.field_owner.is_none(),
            jit_disabled_headers,
            jit_trace_headers,
            quickened_numeric_ops: Vec::new(),
            quick_invokes: vec![QuickInvokeSite::default(); op_count],
            quickened_invoke_ops: Vec::new(),
            entry_this_dst,
            entry_arg_dsts,
            needs_argument_bank,
            needs_return_bank,
            module_slot,
            kind: function.kind,
            is_async: function.is_async,
            arity: function.arity,
            return_kind: function.frame.return_kind,
            capture_count: function.capture_count,
            capture_reference_indices,
        })
    }
}

fn prepare_switch(
    table: &SwitchTable,
    map_word_target: &impl Fn(u32) -> Result<u32, RuntimeError>,
) -> Result<PreparedSwitch, RuntimeError> {
    let default_target = map_word_target(table.default_target.0)?;
    let mut cases = Vec::with_capacity(table.cases.len());
    for case in &table.cases {
        cases.push((case.value, map_word_target(case.target.0)?));
    }

    if cases.len() <= 6 {
        return Ok(PreparedSwitch::Linear { default_target, cases });
    }

    cases.sort_unstable_by_key(|&(value, _)| value);
    let min = cases.first().map(|x| x.0).unwrap_or(0);
    let max = cases.last().map(|x| x.0).unwrap_or(0);
    let span = max.saturating_sub(min).saturating_add(1);
    if span > 0 && span <= 4096 && span as usize <= cases.len() * 2 {
        let mut targets = vec![default_target; span as usize];
        for &(value, target) in &cases {
            targets[(value - min) as usize] = target;
        }
        Ok(PreparedSwitch::Dense {
            default_target,
            min,
            targets,
        })
    } else {
        Ok(PreparedSwitch::Binary { default_target, cases })
    }
}

fn install_superinstructions(
    temps: &[TempOp],
    handlers: &mut [Handler],
    ops: &mut [PreparedOp],
    target_ops: &HashSet<u32>,
) {
    if temps.len() < 2 { return; }
    for index in 0..temps.len() - 1 {
        if handlers[index] as usize == (h_unreachable_fused as Handler) as usize { continue; }
        if target_ops.contains(&((index + 1) as u32)) { continue; }
        let first = temps[index].opcode;
        let second = temps[index + 1].opcode;

        // compare -> conditional branch is extremely common in loops. Preserve
        // the Bool result for exact SSA/debug semantics, but remove one dispatch.
        if is_compare_opcode(first)
            && matches!(second, Opcode::JumpIfFalse | Opcode::JumpIfTrue)
            && temps[index].a == temps[index + 1].a
        {
            ops[index].y = ops[index + 1].x;
            ops[index].flags = if second == Opcode::JumpIfTrue { 1 } else { 0 };
            ops[index].next = (index + 2) as u32;
            handlers[index] = fused_compare_branch_handler(first);
            handlers[index + 1] = h_unreachable_fused;
            continue;
        }

        // Parallel-copy lowering can leave short MOVE runs after register
        // coalescing. Execute two sequential copies in one handler. Reading the
        // second source after the first write preserves exact overlap semantics.
        if first == Opcode::Move && second == Opcode::Move {
            ops[index].x = (temps[index + 1].a as u32)
                | ((temps[index + 1].b as u32) << 8);
            ops[index].next = (index + 2) as u32;
            handlers[index] = h_move_move;
            handlers[index + 1] = h_unreachable_fused;
            continue;
        }

        if first == Opcode::Move && second == Opcode::Return && temps[index].a == temps[index + 1].a {
            ops[index].next = (index + 2) as u32;
            handlers[index] = h_move_return;
            handlers[index + 1] = h_unreachable_fused;
            continue;
        }

        // Edge-copy tails frequently end in MOVE; JUMP. Fuse the final copy
        // with the branch so block-parameter lowering pays one less dispatch.
        if first == Opcode::Move && second == Opcode::Jump {
            ops[index].y = ops[index + 1].x;
            handlers[index] = h_move_jump;
            handlers[index + 1] = h_unreachable_fused;
            continue;
        }

        // Tiny leaf functions and getters often materialize a constant and
        // immediately return it.
        if second == Opcode::Return && temps[index].a == temps[index + 1].a {
            let fused = match first {
                Opcode::LoadNull => Some(h_load_null_return as Handler),
                Opcode::LoadFalse => Some(h_load_false_return as Handler),
                Opcode::LoadTrue => Some(h_load_true_return as Handler),
                Opcode::LoadI8 => Some(h_load_i8_return as Handler),
                Opcode::LoadI32 => Some(h_load_i32_return as Handler),
                Opcode::LoadConst => Some(h_load_const_return as Handler),
                Opcode::MoveFromSpecial => Some(h_move_from_special_return as Handler),
                Opcode::LoadField => Some(h_load_field_return as Handler),
                Opcode::LoadStaticField => Some(h_load_static_field_return as Handler),
                Opcode::LoadModuleVar => Some(h_load_module_var_return as Handler),
                Opcode::LoadCell => Some(h_load_cell_return as Handler),
                _ => None,
            };
            if let Some(handler) = fused {
                handlers[index] = handler;
                handlers[index + 1] = h_unreachable_fused;
                continue;
            }
        }

        // Managed mutation handlers execute StickyImmix's object barrier
        // directly. Root-table stores (module/static fields) need no remembered
        // set entry because VM roots are scanned on every collection. Therefore
        // the following explicit WriteBarrier is redundant in the prepared
        // stream and can still be folded away without weakening generational GC.
        if second == Opcode::WriteBarrier
            && matches!(
                first,
                Opcode::StoreField
                    | Opcode::StoreStaticField
                    | Opcode::StoreModuleVar
                    | Opcode::StoreCell
                    | Opcode::ListSet
                    | Opcode::ListAppend
                    | Opcode::ListInsert
                    | Opcode::MapSet
            )
        {
            ops[index].next = (index + 2) as u32;
            handlers[index + 1] = h_unreachable_fused;
        }
    }
}

#[inline]
const fn is_compare_opcode(opcode: Opcode) -> bool {
    matches!(opcode,
        Opcode::ICmpEq | Opcode::ICmpNe | Opcode::ICmpLt | Opcode::ICmpLe |
        Opcode::ICmpGt | Opcode::ICmpGe | Opcode::FCmpEq | Opcode::FCmpNe |
        Opcode::FCmpLt | Opcode::FCmpLe | Opcode::FCmpGt | Opcode::FCmpGe |
        Opcode::RefEq)
}

// =============================================================================
// Runtime classes, modules and inline caches
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallTarget {
    Bytecode(u32),
    Primitive(PrimitiveId),
    Foreign(ForeignFnId),
    NativeBuiltin(NativeBuiltin),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBuiltin {
    FnNew,
    FnArity,
    FnCall,
    FiberCurrent,
    FiberNew,
    FiberCall,
    FiberTry,
    FiberTransfer,
    FiberTransferError,
    FiberYield,
    FiberSuspend,
    FiberAbort,
    FiberError,
    FiberIsDone,
    PromiseResolve,
    PromiseReject,
    PromiseIsPending,
    PromiseIsFulfilled,
    PromiseIsRejected,
    SystemGc,
}

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
    Promise,
    MapEntry,
    MapKeys,
    Random,
}

const BUILTIN_CLASS_COUNT: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReceiverClass(u64);

const CLASS_KIND_INSTANCE: u64 = 1 << 56;
const CLASS_KIND_STATIC: u64 = 2 << 56;
const CLASS_KIND_BUILTIN: u64 = 3 << 56;
const CLASS_KIND_FOREIGN: u64 = 4 << 56;
const CLASS_KIND_CORE_STATIC: u64 = 5 << 56;

#[inline(always)]
fn heap_object_receiver_class(object: &HeapObject) -> Option<ReceiverClass> {
    Some(match object {
        HeapObject::Random { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Random as u64),
        HeapObject::String(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::String as u64),
        HeapObject::List(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::List as u64),
        HeapObject::Map(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Map as u64),
        HeapObject::MapEntry { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::MapEntry as u64),
        HeapObject::MapKeys { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::MapKeys as u64),
        HeapObject::Range { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Range as u64),
        HeapObject::Closure { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Closure as u64),
        HeapObject::Promise { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Promise as u64),
        HeapObject::FiberObject { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Fiber as u64),
        HeapObject::ClassObject { class_slot } =>
            ReceiverClass(CLASS_KIND_STATIC | *class_slot as u64),
        HeapObject::BuiltinClass { builtin } =>
            ReceiverClass(CLASS_KIND_CORE_STATIC | builtin.index() as u64),
        HeapObject::Instance { class_slot, .. } =>
            ReceiverClass(CLASS_KIND_INSTANCE | *class_slot as u64),
        HeapObject::Foreign { class_slot, .. } =>
            ReceiverClass(CLASS_KIND_FOREIGN | *class_slot as u64),
        HeapObject::AsyncTaskObject { .. }
        | HeapObject::Cell(_)
        | HeapObject::ModuleObject { .. } => return None,
    })
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericQuickOp {
    Neg = 1,
    BitNot,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    ShrU,
}

const QUICK_OP_MASK: u8 = 0x1f;
const QUICK_BRANCH_TRUE: u8 = 0x20;
const QUICK_BRANCH_FUSED: u8 = 0x40;
/// Set after a generic monomorphic quickening site proves polymorphic. The bit
/// prevents patch/depatch thrashing until a method-table invalidation occurs.
const QUICK_GENERIC_DISABLED: u8 = 0x80;

impl NumericQuickOp {
    #[inline]
    fn from_flag(flag: u8) -> Option<Self> {
        Some(match flag & QUICK_OP_MASK {
            1 => Self::Neg,
            2 => Self::BitNot,
            3 => Self::Add,
            4 => Self::Sub,
            5 => Self::Mul,
            6 => Self::Div,
            7 => Self::Rem,
            8 => Self::Lt,
            9 => Self::Le,
            10 => Self::Gt,
            11 => Self::Ge,
            12 => Self::Eq,
            13 => Self::Ne,
            14 => Self::BitAnd,
            15 => Self::BitOr,
            16 => Self::BitXor,
            17 => Self::Shl,
            18 => Self::ShrU,
            _ => return None,
        })
    }

    #[inline(always)]
    fn is_comparison(self) -> bool {
        matches!(self, Self::Lt | Self::Le | Self::Gt | Self::Ge | Self::Eq | Self::Ne)
    }

    #[inline]
    fn from_method(name: &str, arity: u8) -> Option<Self> {
        Some(match (name, arity) {
            ("-", 0) => Self::Neg,
            ("~", 0) => Self::BitNot,
            ("+", 1) => Self::Add,
            ("-", 1) => Self::Sub,
            ("*", 1) => Self::Mul,
            ("/", 1) => Self::Div,
            ("%", 1) => Self::Rem,
            ("<", 1) => Self::Lt,
            ("<=", 1) => Self::Le,
            (">", 1) => Self::Gt,
            (">=", 1) => Self::Ge,
            ("==", 1) => Self::Eq,
            ("!=", 1) => Self::Ne,
            ("&", 1) => Self::BitAnd,
            ("|", 1) => Self::BitOr,
            ("^", 1) => Self::BitXor,
            ("<<", 1) => Self::Shl,
            (">>", 1) => Self::ShrU,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct IcEntry {
    class: ReceiverClass,
    target: CallTarget,
}

impl Default for IcEntry {
    fn default() -> Self {
        Self { class: ReceiverClass(0), target: CallTarget::Bytecode(0) }
    }
}

struct InlineCacheState {
    // `len` is authoritative, so entries below it are always initialized. Using
    // plain entries instead of Option<IcEntry> removes a tag/unwrap from every
    // generic method-cache probe.
    entries: [IcEntry; 4],
    len: u8,
    megamorphic: bool,
}

impl Default for InlineCacheState {
    fn default() -> Self { Self { entries: [IcEntry::default(); 4], len: 0, megamorphic: false } }
}

impl InlineCacheState {
    #[inline(always)]
    fn probe(&self, class: ReceiverClass) -> Option<CallTarget> {
        // Four entries is small enough that an unrolled probe beats loop control
        // and gives the branch predictor a stable branch for each PIC position.
        if self.len > 0 {
            let entry = unsafe { *self.entries.get_unchecked(0) };
            if entry.class == class { return Some(entry.target); }
        }
        if self.len > 1 {
            let entry = unsafe { *self.entries.get_unchecked(1) };
            if entry.class == class { return Some(entry.target); }
        }
        if self.len > 2 {
            let entry = unsafe { *self.entries.get_unchecked(2) };
            if entry.class == class { return Some(entry.target); }
        }
        if self.len > 3 {
            let entry = unsafe { *self.entries.get_unchecked(3) };
            if entry.class == class { return Some(entry.target); }
        }
        None
    }

    #[inline]
    fn insert(&mut self, class: ReceiverClass, target: CallTarget) {
        if self.probe(class).is_some() { return; }
        if (self.len as usize) < self.entries.len() {
            self.entries[self.len as usize] = IcEntry { class, target };
            self.len += 1;
        } else {
            self.megamorphic = true;
        }
    }
}

#[derive(Clone, Copy)]
struct QuickInvokeSite {
    class: ReceiverClass,
    target: CallTarget,
}

impl Default for QuickInvokeSite {
    fn default() -> Self {
        Self { class: ReceiverClass(0), target: CallTarget::Bytecode(0) }
    }
}

struct RuntimeClass {
    id: ClassId,
    name: StringId,
    superclass: Option<u32>,
    /// Fields declared directly by this class.
    own_instance_field_count: u16,
    /// First absolute field slot owned by this class.
    field_base: u16,
    /// Complete fixed instance layout after superclass fields are copied down.
    total_instance_field_count: u16,
    layout_finalized: bool,
    is_foreign: bool,
    /// Dense MethodSymbol-indexed tables. MethodSymbol is interned densely,
    /// so direct indexing is substantially cheaper than hashing on IC misses.
    instance_methods: Vec<Option<CallTarget>>,
    static_methods: Vec<Option<CallTarget>>,
    static_fields: Vec<Value>,
    class_value: Value,
}

struct RuntimeModule {
    id: ModuleId,
    name: StringId,
    variables: Vec<Value>,
    exports: FastMap<StringId, u32>,
    module_value: Value,
    body: Option<u32>,
    initialized: bool,
    initializing: bool,
}

// =============================================================================
// Frames and fibers
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FiberStatus {
    New,
    Ready,
    Running,
    Suspended,
    Done,
    Failed,
}

#[derive(Clone, Copy)]
struct FiberLink {
    caller: u32,
    caller_object: Value,
    dst: Reg,
    catch_error: bool,
}

/// Architectural banks are kept out of `Frame`. A Frame is pushed/popped on
/// every Wren call, while these banks are retained at the fiber stack high-water
/// mark and reused by depth. This keeps the hot call-frame object tiny instead of
/// copying/zeroing 30 x 64-bit register words per call.
struct FrameBanks {
    arguments: [RegisterWord; ARGUMENT_REGISTER_COUNT],
    returns: [RegisterWord; RETURN_REGISTER_COUNT],
    special: [RegisterWord; SPECIAL_REGISTER_COUNT],
}

impl Default for FrameBanks {
    fn default() -> Self {
        Self {
            arguments: [RegisterWord(0); ARGUMENT_REGISTER_COUNT],
            returns: [RegisterWord(0); RETURN_REGISTER_COUNT],
            special: [RegisterWord(0); SPECIAL_REGISTER_COUNT],
        }
    }
}

struct Frame {
    function: u32,
    pc: u32,
    reg_base: usize,
    reg_count: u16,
    return_dst: Option<Reg>,
    root_map: u32,
}

impl Frame {
    #[inline(always)]
    fn new(function: u32, reg_base: usize, reg_count: u16, return_dst: Option<Reg>) -> Self {
        Self { function, pc: 0, reg_base, reg_count, return_dst, root_map: NO_INDEX }
    }
}

struct FiberState {
    object: Value,
    status: FiberStatus,
    frames: Vec<Frame>,
    /// Parallel architectural banks, one slot per frame depth. Entries are kept
    /// after pop and reused, just like the GPR high-water storage below.
    banks: Vec<FrameBanks>,
    /// Backing storage is retained at its high-water mark. `register_top`
    /// tracks the live prefix so frame reuse does not zero GPRs on every call.
    registers: Vec<RegisterWord>,
    register_top: usize,
    starter: Option<Value>,
    link: Option<FiberLink>,
    suspend_dst: Option<Reg>,
    /// Promise resolved/rejected when this internal async task completes.
    async_promise: Option<Value>,
    /// Promise whose settlement will make this execution runnable again.
    awaiting_promise: Option<Value>,
    last_value: Value,
    error: Value,
}

impl FiberState {
    fn new(object: Value, status: FiberStatus) -> Self {
        Self {
            object,
            status,
            frames: Vec::with_capacity(64),
            banks: Vec::with_capacity(64),
            registers: Vec::with_capacity(1024),
            register_top: 0,
            starter: None,
            link: None,
            suspend_dst: None,
            async_promise: None,
            awaiting_promise: None,
            last_value: Value::NULL,
            error: Value::NULL,
        }
    }

    #[inline(always)]
    fn top_banks(&self) -> Option<&FrameBanks> {
        self.frames.len().checked_sub(1).and_then(|index| self.banks.get(index))
    }

    #[inline(always)]
    fn top_banks_mut(&mut self) -> Option<&mut FrameBanks> {
        let index = self.frames.len().checked_sub(1)?;
        self.banks.get_mut(index)
    }
}

/// Stable native storage for a fiber stack. The interpreter is the only mutator;
/// GC workers read it exclusively while all mutators are stopped. Keeping this
/// behind Arc makes FiberObject self-contained even if a native root outlives the
/// Interpreter instance that originally created it.
struct FiberCell {
    state: std::cell::UnsafeCell<FiberState>,
}

impl FiberCell {
    fn new(state: FiberState) -> Self {
        Self { state: std::cell::UnsafeCell::new(state) }
    }

    #[inline(always)]
    unsafe fn get(&self) -> &FiberState {
        unsafe { &*self.state.get() }
    }

    #[inline(always)]
    unsafe fn get_mut(&self) -> &mut FiberState {
        unsafe { &mut *self.state.get() }
    }
}

unsafe impl Send for FiberCell {}
unsafe impl Sync for FiberCell {}

impl fmt::Debug for FiberCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FiberCell(..)")
    }
}

#[derive(Debug)]
struct FiberTraceMetadata {
    gc_maps: Box<[Box<[ReferenceMap]>]>,
}

// =============================================================================
// Interpreter
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunState {
    Running,
    Returned,
    Suspended,
    Failed,
}

pub struct Interpreter<'p> {
    jit: Option<Box<TracingJit>>,
    program: &'p BytecodeProgram,
    prepared: Box<PreparedProgram>,
    host: Box<dyn Host>,
    heap: Heap,
    classes: Vec<RuntimeClass>,
    modules: Vec<RuntimeModule>,
    strings: Vec<Value>,
    builtin_values: Vec<Value>,
    builtin_static_methods: Vec<Vec<Option<CallTarget>>>,
    /// Constants pre-resolved to physical register words once at VM startup.
    /// Hot LOAD_CONST never reinterprets Constant or performs metadata lookup.
    resolved_constants: Vec<Vec<RegisterWord>>,
    fibers: Vec<Option<Arc<FiberCell>>>,
    free_fibers: Vec<u32>,
    /// Cooperative FIFO run queue for internal async tasks and resumed awaiters.
    ready_async: VecDeque<u32>,
    fiber_trace_metadata: Arc<FiberTraceMetadata>,
    current_fiber: u32,
    run_state: RunState,
    result: Value,
    error: Option<RuntimeError>,
    inline_caches: Vec<Vec<InlineCacheState>>,
    builtin_methods: Vec<Vec<Option<CallTarget>>>,
    megamorphic_cache: FastMap<(ReceiverClass, MethodSymbol, Option<u32>), CallTarget>,
    profile_counters: Vec<Vec<u64>>,
    roots: Vec<Option<Value>>,
    free_roots: Vec<u32>,
    /// Stable idle snapshot. Active execution additionally installs an exact
    /// pointer-based provider, but this snapshot keeps VM-owned values alive
    /// between execution entrypoints.
    persistent_roots: Arc<HandleRootSet>,
    _persistent_root_registration: GcRootRegistration,
    gc_enabled: bool,
    last_gc_collections: u64,

    // Hot execution cache. These pointers are refreshed only when frames or
    // fibers change. They turn every GPR access into one pointer add/load.
    hot_frame: *mut Frame,
    hot_banks: *mut FrameBanks,
    hot_registers: *mut RegisterWord,
    hot_function: *const PreparedFunction,
    hot_ops: *const PreparedOp,
    hot_handlers: *const Handler,
    hot_constants: *const RegisterWord,
    hot_inline_caches: *mut InlineCacheState,
    /// Prepared-op PC cached alongside the hot frame. While the interpreter is
    /// running this is authoritative; Frame::pc is synchronized only at runtime,
    /// GC, JIT and frame/fiber boundaries.
    hot_pc: u32,
}

/// Precise VM roots exposed to MMTk while an Interpreter execution entrypoint
/// is active. The pointer is valid for the lifetime of run_function()/resume();
/// MMTk only invokes providers after the mutator has reached the STW rendezvous.
struct VmRootProvider {
    vm: usize,
}

unsafe impl Send for VmRootProvider {}
unsafe impl Sync for VmRootProvider {}

unsafe impl GcRootProvider for VmRootProvider {
    fn visit(&self, visitor: &mut GcRootVisitor<'_>) {
        // Interpreter<'p> has the same runtime layout for every 'p. The provider
        // never outlives the execution method that registered this pointer.
        let vm = unsafe { &*(self.vm as *const Interpreter<'static>) };
        let mut root = |value: Value| {
            if let Some(handle) = value.heap_handle() {
                if let Some(slot) = handles().gc_slot(handle) {
                    visitor.slot(slot);
                }
            }
        };

        for &value in &vm.strings { root(value); }
        for &value in &vm.builtin_values { root(value); }
        for value in vm.roots.iter().filter_map(|slot| *slot) { root(value); }
        for class in &vm.classes {
            root(class.class_value);
            for &value in &class.static_fields { root(value); }
        }
        for module in &vm.modules {
            root(module.module_value);
            for &value in &module.variables { root(value); }
        }
        if let Some(fiber) = vm
            .fibers
            .get(vm.current_fiber as usize)
            .and_then(|slot| slot.as_ref())
        {
            // SAFETY: provider runs only while mutators are stopped.
            root(unsafe { fiber.get() }.object);
        }
        for &id in &vm.ready_async {
            if let Some(fiber) = vm.fibers.get(id as usize).and_then(|slot| slot.as_ref()) {
                // SAFETY: provider runs only while mutators are stopped.
                root(unsafe { fiber.get() }.object);
            }
        }
        root(vm.result);
    }
}

impl Drop for Interpreter<'_> {
    fn drop(&mut self) {
        // Remove every VM-owned strong root and force a full collection while
        // the mutator and its native resources are still alive. Managed fibers
        // and closures are self-contained (Arc/Box-backed), so externally rooted
        // objects may survive this VM; this collection primarily releases the
        // ordinary VM-owned heap promptly and recycles stable handle IDs.
        self.persistent_roots.replace(Vec::new());
        self.roots.clear();
        self.heap.clear_stack_owner();
        self.heap.allocation_roots.clear();
        self.heap.collect_full();
    }
}

impl<'p> Interpreter<'p> {
    pub fn new(program: &'p BytecodeProgram) -> Result<Self, RuntimeError> {
        Self::with_host(program, Box::<NoHost>::default())
    }

    pub fn with_host(program: &'p BytecodeProgram, host: Box<dyn Host>) -> Result<Self, RuntimeError> {
        // All hot-path unchecked indexing and threaded dispatch rely on this
        // one-time validation barrier. Never execute unverified bytecode.
        BytecodeVerifier::verify_program(program)
            .map_err(|error| RuntimeError::new(format!("bytecode verification failed: {error}")))?;
        let mut prepared = Box::new(PreparedProgram::prepare(program)?);
        let fiber_trace_metadata = Arc::new(FiberTraceMetadata {
            gc_maps: prepared
                .functions
                .iter()
                .map(|function| function.gc_maps.clone().into_boxed_slice())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        let mut heap = Heap::default();
        // VM construction itself may allocate enough metadata to trigger MMTk.
        // Keep every object created so far live through one temporary provider;
        // handles are canonical MMTk slots and are updated if a collection moves
        // an object during initialization.
        let init_roots = Arc::new(HandleRootSet::default());
        let persistent_root_registration = heap.register_roots(init_roots.clone());

        let mut strings = Vec::with_capacity(program.metadata.strings.len());
        for string in &program.metadata.strings {
            let value = heap.alloc(HeapObject::String(Arc::<[u8]>::from(string.as_bytes())));
            init_roots.push(value);
            strings.push(value);
        }

        let mut builtin_values = Vec::with_capacity(BuiltinId::COUNT);
        for builtin in BuiltinId::ALL {
            let value = heap.alloc(HeapObject::BuiltinClass { builtin });
            init_roots.push(value);
            builtin_values.push(value);
        }

        let mut modules = Vec::with_capacity(program.metadata.modules.len());
        for (slot, info) in program.metadata.modules.iter().enumerate() {
            let module_value = heap.alloc(HeapObject::ModuleObject { module_slot: slot as u32 });
            init_roots.push(module_value);
            let mut variables = vec![Value::NULL; info.variable_count as usize];
            for binding in &info.builtin_bindings {
                let Some(target) = variables.get_mut(binding.slot.0 as usize) else {
                    return Err(RuntimeError::new("builtin module binding slot out of range"));
                };
                *target = builtin_values[binding.builtin.index()];
            }
            modules.push(RuntimeModule {
                id: info.id,
                name: info.name,
                variables,
                exports: FastMap::default(),
                module_value,
                body: None,
                initialized: false,
                initializing: false,
            });
        }

        let mut classes = Vec::with_capacity(program.metadata.classes.len());
        for (slot, info) in program.metadata.classes.iter().enumerate() {
            let class_value = heap.alloc(HeapObject::ClassObject { class_slot: slot as u32 });
            init_roots.push(class_value);
            classes.push(RuntimeClass {
                id: info.id,
                name: info.name,
                // Runtime CreateClass is authoritative. Even when metadata knows
                // a static superclass, copy-down and field relocation happen
                // only after the superclass object has actually been created.
                superclass: None,
                own_instance_field_count: info.own_instance_field_count,
                field_base: 0,
                total_instance_field_count: info.own_instance_field_count,
                layout_finalized: false,
                is_foreign: info.is_foreign,
                instance_methods: vec![None; program.metadata.methods.len()],
                static_methods: vec![None; program.metadata.methods.len()],
                static_fields: vec![Value::NULL; info.static_field_count as usize],
                class_value,
            });
        }

        // Module variable arrays are fixed-size for the life of the VM. Patch the
        // executable PreparedOp stream with the stable Value* of each global slot.
        // `original_ops` deliberately keeps logical module/slot operands for trace
        // recording and diagnostics.
        for (function_index, function) in prepared.functions.iter_mut().enumerate() {
            let bytecode = &program.functions[function_index].chunk.code;
            for index in 0..function.ops.len() {
                let original = function.original_ops[index];
                let Some(opcode) = bytecode[original.raw_pc as usize].opcode() else { continue; };
                if !matches!(opcode, Opcode::LoadModuleVar | Opcode::StoreModuleVar) { continue; }
                let module = unsafe { modules.get_unchecked_mut(original.x as usize) };
                let pointer = unsafe { module.variables.as_mut_ptr().add(original.y as usize) } as usize as u64;
                function.ops[index].x = pointer as u32;
                function.ops[index].y = (pointer >> 32) as u32;
            }
        }

        // Resolve module body entry points once. ImportModule can then jump to
        // a prepared function index with no FuncId lookup on the hot path.
        for (function_index, function) in program.functions.iter().enumerate() {
            if function.kind == FunctionKind::ModuleBody {
                if let Some(&module_slot) = prepared.module_by_id.get(&function.module) {
                    let runtime_module = &mut modules[module_slot as usize];
                    if runtime_module.body.replace(function_index as u32).is_some() {
                        return Err(RuntimeError::new(format!(
                            "module {:?} has more than one ModuleBody",
                            function.module
                        )));
                    }
                }
            }
        }

        // Resolve constants once. String constants become stable heap handles,
        // numeric constants become their exact physical register representation.
        let mut resolved_constants = Vec::with_capacity(program.functions.len());
        for function in &program.functions {
            let mut constants = Vec::with_capacity(function.chunk.constants.len());
            for constant in &function.chunk.constants {
                let word = match constant {
                    Constant::Null => Value::NULL.word(),
                    Constant::Bool(value) => Value::from_bool(*value).word(),
                    Constant::Num(value) => Value::from_num(*value).word(),
                    Constant::String(id) => strings
                        .get(id.0 as usize)
                        .copied()
                        .ok_or_else(|| RuntimeError::new("string constant index out of range"))?
                        .word(),
                    Constant::Int32(value) => i32_word(*value),
                };
                constants.push(word);
            }
            resolved_constants.push(constants);
        }

        let inline_caches = program.functions.iter().map(|function| {
            (0..function.chunk.inline_caches.len()).map(|_| InlineCacheState::default()).collect()
        }).collect();
        let profile_counters = program.functions.iter().map(|function| {
            vec![0; function.chunk.profile_counters.len()]
        }).collect();

        // FiberState lives in stable Arc storage. The managed Fiber object owns
        // another Arc plus immutable GC-map metadata, so tracing never depends on
        // raw pointers into Interpreter fields.
        let mut main_state = FiberState::new(Value::NULL, FiberStatus::Running);
        main_state.registers.reserve(4096);
        let main = Arc::new(FiberCell::new(main_state));
        let main_object = heap.alloc(HeapObject::FiberObject {
            fiber: 0,
            state: main.clone(),
            trace_metadata: fiber_trace_metadata.clone(),
        });
        init_roots.push(main_object);
        unsafe { main.get_mut() }.object = main_object;
        heap.set_stack_owner(main_object);
        let last_gc_collections = heap.collection_count();

        Ok(Self {
            jit: Some(Box::new(TracingJit::default())),
            program,
            prepared,
            host,
            heap,
            classes,
            modules,
            strings,
            builtin_values,
            builtin_static_methods: vec![vec![None; program.metadata.methods.len()]; BuiltinId::COUNT],
            resolved_constants,
            fibers: vec![Some(main)],
            free_fibers: Vec::new(),
            ready_async: VecDeque::new(),
            fiber_trace_metadata,
            current_fiber: 0,
            run_state: RunState::Running,
            result: Value::NULL,
            error: None,
            inline_caches,
            builtin_methods: vec![vec![None; program.metadata.methods.len()]; BUILTIN_CLASS_COUNT],
            megamorphic_cache: FastMap::default(),
            profile_counters,
            roots: Vec::new(),
            free_roots: Vec::new(),
            persistent_roots: init_roots,
            _persistent_root_registration: persistent_root_registration,
            gc_enabled: true,
            last_gc_collections,
            hot_frame: std::ptr::null_mut(),
            hot_banks: std::ptr::null_mut(),
            hot_registers: std::ptr::null_mut(),
            hot_function: std::ptr::null(),
            hot_ops: std::ptr::null(),
            hot_handlers: std::ptr::null(),
            hot_constants: std::ptr::null(),
            hot_inline_caches: std::ptr::null_mut(),
            hot_pc: NO_INDEX,
        })
    }

    pub fn set_gc_enabled(&mut self, enabled: bool) {
        self.gc_enabled = enabled;
    }

    #[inline]
    fn gc_root_registration(&mut self) -> GcRootRegistration {
        let provider: Arc<dyn GcRootProvider> = Arc::new(VmRootProvider {
            vm: self as *mut Self as usize,
        });
        self.heap.register_roots(provider)
    }

    fn sync_persistent_roots(&self) {
        let mut roots = Vec::with_capacity(
            self.strings.len()
                + self.builtin_values.len()
                + self.classes.len()
                + self.modules.len()
                + self.roots.len()
                + 8,
        );
        let mut push = |value: Value| {
            if let Some(handle) = value.heap_handle() { roots.push(handle); }
        };
        for &value in &self.strings { push(value); }
        for &value in &self.builtin_values { push(value); }
        for value in self.roots.iter().filter_map(|slot| *slot) { push(value); }
        for class in &self.classes {
            push(class.class_value);
            for &value in &class.static_fields { push(value); }
        }
        for module in &self.modules {
            push(module.module_value);
            for &value in &module.variables { push(value); }
        }
        if let Some(fiber) = self
            .fibers
            .get(self.current_fiber as usize)
            .and_then(|slot| slot.as_deref())
        {
            // SAFETY: sync_persistent_roots runs on the owning mutator thread.
            push(unsafe { fiber.get() }.object);
        }
        for &id in &self.ready_async {
            if let Some(fiber) = self.fibers.get(id as usize).and_then(|slot| slot.as_deref()) {
                // SAFETY: owning mutator thread; task object is stable native storage owner.
                push(unsafe { fiber.get() }.object);
            }
        }
        push(self.result);
        self.persistent_roots.replace(roots);
    }

    /// Snapshot of collector counters for benchmarking/tuning.
    pub fn gc_statistics(&self) -> crate::wren_gc::Statistics {
        self.heap.mutator.statistics()
    }

        /// Set to None for an interpreter-only run, or provide tracing thresholds.
    pub fn set_jit_options(&mut self, options: Option<TracingJitOptions>) {
        // A new tiering policy gets a clean opportunity to observe every loop.
        for function in &mut self.prepared.functions {
            function.jit_disabled_headers.fill(0);
            function.jit_trace_headers.fill(NO_INDEX);
        }
        self.jit = options.map(|options| Box::new(TracingJit::new(options)));
    }

    pub fn jit_stats(&self) -> Option<&TracingJitStats> {
        self.jit.as_ref().map(|jit| jit.stats())
    }

    fn invalidate_method_traces(&mut self) {
        if let Some(options) = self.jit.as_ref().map(|jit| jit.options().clone()) {
            self.jit = Some(Box::new(TracingJit::new(options)));
        }
        for function in &mut self.prepared.functions {
            function.jit_disabled_headers.fill(0);
            function.jit_trace_headers.fill(NO_INDEX);
            for op_index in std::mem::take(&mut function.quickened_numeric_ops) {
                let index = op_index as usize;
                if index < function.handlers.len() && index < function.ops.len() {
                    function.handlers[index] = h_invoke;
                    function.ops[index].flags = 0;
                }
            }
            for op_index in std::mem::take(&mut function.quickened_invoke_ops) {
                let index = op_index as usize;
                if index < function.handlers.len() && index < function.ops.len() {
                    function.quick_invokes[index] = QuickInvokeSite::default();
                    function.handlers[index] = h_invoke;
                    function.ops[index].flags = 0;
                }
            }
        }
        self.megamorphic_cache.clear();
        for function in &mut self.inline_caches {
            for cache in function { *cache = InlineCacheState::default(); }
        }
    }

    pub fn bind_builtin_function(&mut self, class: BuiltinClass, method: MethodSymbol, function: FuncId) {
        let function = *self.prepared.function_by_id.get(&function)
            .expect("builtin function must exist in the compiled program");
        self.invalidate_method_traces();
        let slot = self.builtin_methods.get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
            .expect("builtin method must exist in the compiled program");
        *slot = Some(CallTarget::Bytecode(function));
    }

    pub fn bind_builtin_primitive(&mut self, class: BuiltinClass, method: MethodSymbol, primitive: PrimitiveId) {
        self.invalidate_method_traces();

        if let Some(slot) = self
            .builtin_methods
            .get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Primitive(primitive));
        }
    }

    pub fn bind_builtin_static_primitive(
        &mut self,
        builtin: BuiltinId,
        method: MethodSymbol,
        primitive: PrimitiveId,
    ) {
        if let Some(slot) = self
            .builtin_static_methods
            .get_mut(builtin.index())
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Primitive(primitive));
        }
    }

    pub fn bind_builtin_static_foreign(
        &mut self,
        builtin: BuiltinId,
        method: MethodSymbol,
        foreign: ForeignFnId,
    ) {
        if let Some(slot) = self
            .builtin_static_methods
            .get_mut(builtin.index())
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Foreign(foreign));
        }
    }

    #[inline]
    pub fn builtin_value(&self, builtin: BuiltinId) -> Value {
        self.builtin_values[builtin.index()]
    }

    pub fn bind_builtin_native(&mut self, class: BuiltinClass, method: MethodSymbol, native: NativeBuiltin) {
        self.invalidate_method_traces();

        if let Some(slot) = self
            .builtin_methods
            .get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::NativeBuiltin(native));
        }
    }

    pub fn bind_builtin_static_native(&mut self, builtin: BuiltinId, method: MethodSymbol, native: NativeBuiltin) {
        if let Some(slot) = self
            .builtin_static_methods
            .get_mut(builtin.index())
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::NativeBuiltin(native));
        }
    }

    pub fn bind_builtin_foreign(&mut self, class: BuiltinClass, method: MethodSymbol, foreign: ForeignFnId) {
        self.invalidate_method_traces();

        if let Some(slot) = self
            .builtin_methods
            .get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Foreign(foreign));
        }
    }

    pub fn bind_module_variable_name(&mut self, module: ModuleId, name: StringId, slot: u32) -> Result<(), RuntimeError> {
        let module_slot = *self.prepared.module_by_id.get(&module).ok_or_else(|| RuntimeError::new("unknown module"))?;
        if slot as usize >= self.modules[module_slot as usize].variables.len() { return Err(RuntimeError::new("module slot out of range")); }
        self.modules[module_slot as usize].exports.insert(name, slot);
        Ok(())
    }

    pub fn add_root(&mut self, value: Value) -> u32 {
        let index = if let Some(index) = self.free_roots.pop() {
            self.roots[index as usize] = Some(value);
            index
        } else {
            let index = self.roots.len() as u32;
            self.roots.push(Some(value));
            index
        };
        self.sync_persistent_roots();
        index
    }

    pub fn set_root(&mut self, root: u32, value: Value) -> Result<(), RuntimeError> {
        let slot = self.roots.get_mut(root as usize).ok_or_else(|| RuntimeError::new("invalid root handle"))?;
        if slot.is_none() { return Err(RuntimeError::new("root handle has been released")); }
        *slot = Some(value);
        self.sync_persistent_roots();
        Ok(())
    }

    pub fn release_root(&mut self, root: u32) {
        if let Some(slot) = self.roots.get_mut(root as usize) {
            if slot.take().is_some() { self.free_roots.push(root); }
        }
        self.sync_persistent_roots();
    }

    pub fn run_function(&mut self, function: FuncId, args: &[Value]) -> Result<ExecutionResult, RuntimeError> {
        let _gc_roots = self.gc_root_registration();
        let function_index = *self.prepared.function_by_id.get(&function)
            .ok_or_else(|| RuntimeError::new(format!("unknown function {:?}", function)))?;
        if !self.fiber(self.current_fiber).frames.is_empty() {
            return Err(RuntimeError::new("run_function requires an idle main fiber"));
        }
        if args.len() > ARGUMENT_REGISTER_COUNT {
            return Err(RuntimeError::new("too many entry arguments"));
        }

        self.run_state = RunState::Running;
        self.result = Value::NULL;
        self.error = None;
        let current_fiber = self.current_fiber;
        self.fiber_mut(current_fiber).status = FiberStatus::Running;

        // Mark directly executed module bodies as initializing too. This keeps
        // cyclic imports from recursively entering the same body.
        let entry = unsafe { self.prepared.functions.get_unchecked(function_index as usize) };
        if entry.kind == FunctionKind::ModuleBody {
            let module = unsafe { self.modules.get_unchecked_mut(entry.module_slot as usize) };
            module.initializing = true;
        }
        let mut words = [RegisterWord(0); ARGUMENT_REGISTER_COUNT];
        for (index, value) in args.iter().copied().enumerate() {
            words[index] = value.word();
        }
        self.push_frame_direct(function_index, &words[..args.len()], None, None)?;
        self.dispatch_loop();
        let result = self.finish_result();
        self.sync_persistent_roots();
        result
    }

    pub fn run_module(&mut self, function: FuncId) -> Result<ExecutionResult, RuntimeError> {
        self.run_function(function, &[])
    }

    /// Resume a VM stopped by Fiber.suspend() or by yielding without a caller.
    ///
    /// The value becomes the result of the suspended Fiber operation.
    pub fn resume(&mut self, value: Value) -> Result<ExecutionResult, RuntimeError> {
        let _gc_roots = self.gc_root_registration();
        if self.run_state != RunState::Suspended {
            return Err(RuntimeError::new("resume requires a suspended VM"));
        }
        let current = self.current_fiber;
        if self.fiber(current).awaiting_promise.is_some() {
            return Err(RuntimeError::new(
                "a Promise-suspended execution can only resume when that Promise settles",
            ));
        }
        let dst = self.fiber_mut(current).suspend_dst.take();
        if let Some(dst) = dst {
            let fiber = self.fiber_mut(current);
            let frame = fiber.frames.last().ok_or_else(|| RuntimeError::new("suspended fiber has no frame"))?;
            let index = frame.reg_base + dst.index();
            unsafe {
                *fiber.registers.get_unchecked_mut(index) = value.word();
            }
        }
        if let Some(banks) = self.fiber_mut(current).top_banks_mut() {
            banks.special[SpecialReg::ResumeValue.index()] = value.word();
        }
        self.fiber_mut(current).status = FiberStatus::Running;
        self.refresh_hot_frame();
        self.run_state = RunState::Running;
        self.result = Value::NULL;
        self.error = None;
        self.dispatch_loop();
        let result = self.finish_result();
        self.sync_persistent_roots();
        result
    }

    fn finish_result(&mut self) -> Result<ExecutionResult, RuntimeError> {
        match self.run_state {
            RunState::Returned => Ok(ExecutionResult::Returned(self.result)),
            RunState::Suspended => Ok(ExecutionResult::Suspended(self.result)),
            RunState::Failed => Err(self.error.take().unwrap_or_else(|| RuntimeError::new("interpreter failed"))),
            RunState::Running => Err(RuntimeError::new("interpreter stopped while still running")),
        }
    }

    #[inline(never)]
    fn dispatch_loop(&mut self) {
        while self.run_state == RunState::Running {
            if self.hot_frame.is_null() {
                self.fail(RuntimeError::new("running fiber has no frame"));
                break;
            }

            let pc = self.hot_pc as usize;

            let prepared = unsafe { &*self.hot_function };
            if pc >= prepared.ops.len() {
                self.fail(RuntimeError::new("program counter is outside function"));
                break;
            }
            // The vector bases are cached when a frame becomes hot, mirroring
            // Wren's cached ip/fn/stackStart locals. Normal dispatch is now two
            // pointer adds and one indirect call, with no Vec traversal.
            let handler = unsafe { *self.hot_handlers.add(pc) };
            let op = unsafe { self.hot_ops.add(pc) };
            unsafe { handler(self, op) };
        }
    }

    #[inline(always)]
    fn fiber(&self, id: u32) -> &FiberState {
        let cell = unsafe { self.fibers.get_unchecked(id as usize).as_ref().unwrap_unchecked() };
        unsafe { cell.get() }
    }

    #[inline(always)]
    fn fiber_mut(&mut self, id: u32) -> &mut FiberState {
        let cell = unsafe { self.fibers.get_unchecked(id as usize).as_ref().unwrap_unchecked() };
        unsafe { cell.get_mut() }
    }

    #[inline]
    fn switch_current_fiber(&mut self, id: u32) {
        self.current_fiber = id;
        let owner = self.fiber(id).object;
        self.heap.set_stack_owner(owner);
    }

    #[inline(always)]
    fn current_frame(&self) -> &Frame {
        debug_assert!(!self.hot_frame.is_null());
        unsafe { &*self.hot_frame }
    }

    #[inline(always)]
    fn current_frame_mut(&mut self) -> &mut Frame {
        debug_assert!(!self.hot_frame.is_null());
        unsafe { &mut *self.hot_frame }
    }

    #[inline(always)]
    fn current_banks(&self) -> &FrameBanks {
        debug_assert!(!self.hot_banks.is_null());
        unsafe { &*self.hot_banks }
    }

    #[inline(always)]
    fn current_banks_mut(&mut self) -> &mut FrameBanks {
        debug_assert!(!self.hot_banks.is_null());
        unsafe { &mut *self.hot_banks }
    }

    #[inline(always)]
    unsafe fn reg(&self, reg: u8) -> RegisterWord {
        debug_assert!(!self.hot_registers.is_null());
        *self.hot_registers.add(reg as usize)
    }

    #[inline(always)]
    unsafe fn set_reg(&mut self, reg: u8, value: RegisterWord) {
        debug_assert!(!self.hot_registers.is_null());
        *self.hot_registers.add(reg as usize) = value;
    }

    /// Refresh the cached frame and GPR base after any operation that can move
    /// the register/frame vectors or switch the current fiber.
    #[inline]
    fn refresh_hot_frame(&mut self) {
        let current = self.current_fiber as usize;
        let Some(cell) = self.fibers.get(current).and_then(Option::as_ref) else {
            self.clear_hot_frame();
            return;
        };
        let fiber = unsafe { cell.get_mut() };
        let depth = fiber.frames.len();
        if depth == 0 {
            self.clear_hot_frame();
            return;
        }
        let frame = unsafe { fiber.frames.get_unchecked_mut(depth - 1) };
        let banks = unsafe { fiber.banks.get_unchecked_mut(depth - 1) };
        self.hot_pc = frame.pc;
        self.hot_frame = frame as *mut Frame;
        self.hot_banks = banks as *mut FrameBanks;
        self.hot_registers = unsafe { fiber.registers.as_mut_ptr().add(frame.reg_base) };
        self.hot_function = unsafe { self.prepared.functions.as_ptr().add(frame.function as usize) };
        let prepared = unsafe { &*self.hot_function };
        self.hot_ops = prepared.ops.as_ptr();
        self.hot_handlers = prepared.handlers.as_ptr();
        self.hot_constants = unsafe {
            self.resolved_constants.get_unchecked(frame.function as usize).as_ptr()
        };
        self.hot_inline_caches = unsafe {
            self.inline_caches.get_unchecked_mut(frame.function as usize).as_mut_ptr()
        };
    }

    /// Synchronize the cached PC only at boundaries which expose Frame state to
    /// slow/runtime code. Normal bytecode dispatch never writes Frame::pc.
    #[inline(always)]
    fn flush_hot_pc(&mut self) {
        if !self.hot_frame.is_null() && self.hot_pc != NO_INDEX {
            unsafe { (*self.hot_frame).pc = self.hot_pc; }
        }
    }

    #[inline(always)]
    fn clear_hot_frame(&mut self) {
        // The current frame may already have been popped when this is called. Do
        // not flush through hot_frame here; callers that expose a live frame flush
        // before mutating the frame/fiber stack.
        self.hot_frame = std::ptr::null_mut();
        self.hot_banks = std::ptr::null_mut();
        self.hot_registers = std::ptr::null_mut();
        self.hot_function = std::ptr::null();
        self.hot_ops = std::ptr::null();
        self.hot_handlers = std::ptr::null();
        self.hot_constants = std::ptr::null();
        self.hot_inline_caches = std::ptr::null_mut();
        self.hot_pc = NO_INDEX;
    }

    #[inline(always)]
    fn current_prepared(&self) -> &PreparedFunction {
        debug_assert!(!self.hot_function.is_null());
        unsafe { &*self.hot_function }
    }

    #[inline(always)]
    unsafe fn advance(&mut self, op: &PreparedOp) {
        self.hot_pc = op.next;
    }

    #[inline(always)]
    unsafe fn set_pc(&mut self, pc: u32) {
        self.hot_pc = pc;
    }

    /// Take a real backwards CFG edge and give the tracing tier a chance to
    /// count/record/enter the loop.  Keeping this off the main dispatch path is
    /// important: ordinary opcodes pay zero HashMap/tiering cost.
    #[inline(never)]
    unsafe fn take_backedge(&mut self, op: &PreparedOp, target: u32) {
        let from_index = self.hot_pc;
        let function_index = self.current_frame().function as usize;
        debug_assert!(target < from_index);

        let cached_trace = self
            .prepared
            .functions
            .get(function_index)
            .and_then(|function| function.jit_trace_headers.get(target as usize))
            .copied()
            .unwrap_or(NO_INDEX);

        self.set_pc(target);

        // During recording/native runtime helpers the manager is temporarily
        // taken out of the Interpreter, which also prevents recursive tiering.
        // The manager itself is boxed, so this hot transition moves one pointer
        // instead of the whole JIT state.
        let Some(mut jit) = self.jit.take() else { return; };

        let (result, uncached_target) = if cached_trace != NO_INDEX {
            // Compiled headers need neither raw-PC lookup nor TraceLocation
            // construction nor a loop-state hash probe.
            (jit.enter_trace(self, TraceId(cached_trace)), None)
        } else {
            let function = self.prepared.functions[function_index].id;
            let target_raw_pc = match self.prepared.functions[function_index]
                .original_ops
                .get(target as usize)
            {
                Some(target_op) => target_op.raw_pc,
                None => {
                    self.jit = Some(jit);
                    self.fail(RuntimeError::new("invalid prepared backedge target"));
                    return;
                }
            };
            let from = TraceLocation::new(function, BytecodePc(op.raw_pc));
            let to = TraceLocation::new(function, BytecodePc(target_raw_pc));
            (jit.on_backedge(self, from, to), Some(to))
        };

        // Cold/uncompiled headers need a loop-state lookup to learn whether a
        // sequence of recording failures finally blacklisted the header.
        let disable_header = if matches!(&result, Ok(JitAction::Blacklisted)) {
            true
        } else if let Some(to) = uncached_target {
            jit.is_blacklisted(to)
        } else {
            false
        };
        let newly_compiled = match &result {
            Ok(JitAction::Compiled { trace })
            | Ok(JitAction::EnteredNative { trace })
            | Ok(JitAction::CoolingDown { trace })
            | Ok(JitAction::SideExit { trace, .. }) => Some(trace.0),
            _ => None,
        };
        self.jit = Some(jit);

        if let Some(function) = self.prepared.functions.get_mut(function_index) {
            if disable_header {
                if let Some(disabled) = function.jit_disabled_headers.get_mut(target as usize) {
                    *disabled = u8::MAX;
                }
                if let Some(cached) = function.jit_trace_headers.get_mut(target as usize) {
                    *cached = NO_INDEX;
                }
            } else if cached_trace == NO_INDEX {
                if let Some(trace) = newly_compiled {
                    if let Some(cached) = function.jit_trace_headers.get_mut(target as usize) {
                        *cached = trace;
                    }
                }
            }
            if matches!(&result, Ok(JitAction::CoolingDown { .. })) {
                function.jit_disabled_headers[target as usize] = 64;
            }
        }

        if let Err(error) = result {
            self.fail(RuntimeError::new(error.to_string()));
        }
    }

    #[inline(always)]
    unsafe fn branch_to(&mut self, op: &PreparedOp, target: u32) {
        let backwards = target < self.hot_pc;
        if target < self.hot_pc && self.jit.is_some() {
            // Once a loop has been blacklisted, never enter the tiering manager
            // from this header again. A byte lookup is substantially cheaper than
            // taking/moving the JIT manager and probing its loop-state HashMap on
            // every baseline iteration.
            let cooldown = self
                .current_prepared()
                .jit_disabled_headers
                .get(target as usize)
                .copied()
                .unwrap_or(0);
            if cooldown != 0 {
                if cooldown != u8::MAX {
                    let function = self.current_frame().function as usize;
                    self.prepared.functions[function].jit_disabled_headers[target as usize] -= 1;
                }
                self.set_pc(target);
            } else {
                self.take_backedge(op, target);
            }
        } else {
            self.set_pc(target);
        }
        if backwards && self.heap.mutator.collection_requested()
            && self.run_state == RunState::Running && !self.hot_frame.is_null()
        {
            let current = self.current_prepared().original_ops[self.hot_pc as usize];
            if current.gc_map != NO_INDEX {
                self.mark_gc_point(&current);
                self.maybe_collect();
            }
        }
    }

    #[inline(always)]
    fn current_location(&self, op: &PreparedOp) -> (FuncId, BytecodePc) {
        (self.current_prepared().id, BytecodePc(op.raw_pc))
    }

    #[inline(always)]
    fn mark_gc_point(&mut self, op: &PreparedOp) {
        // Frame state becomes externally observable at GC/runtime boundaries.
        self.flush_hot_pc();
        if op.gc_map != NO_INDEX {
            self.current_frame_mut().root_map = op.gc_map;
        }
    }

    #[inline]
    fn fail_terminal(&mut self, error: RuntimeError) {
        self.error = Some(error);
        self.run_state = RunState::Failed;
    }

    /// Route an ordinary VM runtime error through Wren's Fiber caller chain.
    ///
    /// We only materialize the Wren String after reaching a `try()` caller. The
    /// caller's FiberTry opcode is a GC safepoint, so its exact root map is
    /// already published. Allocating the String at the original failing opcode
    /// would be unsound because arbitrary non-GC bytecodes do not carry stack
    /// maps for a collection triggered by that allocation.
    fn catch_runtime_error_in_fiber(&mut self, error: &RuntimeError) -> bool {
        let start = self.current_fiber;
        if start == 0 || self.fiber(start).async_promise.is_some() {
            return false;
        }

        // First prove there is a try() boundary. Do not mutate the fiber graph
        // for an uncaught/internal VM failure; the terminal path will stop the VM.
        let mut probe = start;
        let mut catches = false;
        while probe != 0 {
            let Some(link) = self.fiber(probe).link else { break; };
            if link.catch_error {
                catches = true;
                break;
            }
            probe = link.caller;
        }
        if !catches {
            return false;
        }

        let mut failed = Vec::with_capacity(4);
        let mut current = start;
        loop {
            debug_assert_ne!(current, 0, "pre-scan guaranteed a Fiber.try boundary");
            let link = {
                let fiber = self.fiber_mut(current);
                fiber.status = FiberStatus::Failed;
                fiber.awaiting_promise = None;
                fiber.suspend_dst = None;
                fiber.link.take().expect("pre-scanned fiber chain changed")
            };
            failed.push(current);

            if !link.catch_error {
                current = link.caller;
                continue;
            }

            // The FiberTry call site published its precise GC map before the
            // switch. Make all aborted FiberObjects temporary roots as well so
            // nested called fibers remain alive while the error String allocates.
            self.switch_current_fiber(link.caller);
            self.fiber_mut(link.caller).status = FiberStatus::Running;
            self.refresh_hot_frame();

            let failed_objects = failed
                .iter()
                .map(|&id| self.fiber(id).object)
                .collect::<Vec<_>>();
            let roots = failed_objects
                .iter()
                .copied()
                .map(|object| self.add_root(object))
                .collect::<Vec<_>>();

            let bytes: Arc<[u8]> = Arc::from(error.message.as_bytes());
            let error_value = self.heap.alloc(HeapObject::String(bytes));

            for (&id, &object) in failed.iter().zip(&failed_objects) {
                self.heap.remember_if_reference(object, error_value);
                self.fiber_mut(id).error = error_value;
            }

            {
                let caller = self.fiber_mut(link.caller);
                if let Some(frame) = caller.frames.last() {
                    let index = frame.reg_base + link.dst.index();
                    caller.registers[index] = error_value.word();
                }
            }

            for root in roots {
                self.release_root(root);
            }
            self.refresh_hot_frame();
            return true;
        }
    }

    fn fail(&mut self, error: RuntimeError) {
        if self.catch_runtime_error_in_fiber(&error) {
            return;
        }
        self.fail_terminal(error);
    }

    fn fail_at(&mut self, op: &PreparedOp, message: impl Into<String>) {
        let (function, pc) = self.current_location(op);
        self.fail(RuntimeError::new(message).at(function, pc));
    }

    #[inline(always)]
    unsafe fn packed_call_reg(op: &PreparedOp, index: usize) -> u8 {
        debug_assert!((op.c as usize) <= 4 && index < op.c as usize);
        ((op.args_offset >> (index * 8)) & 0xff) as u8
    }

    /// Unrolled 0..=4 argument gather. Register IDs are embedded in the prepared
    /// opcode, so the common call path touches no side-table memory.
    #[inline(always)]
    unsafe fn gather_call_words_small(&self, op: &PreparedOp, out: &mut [RegisterWord; 4]) -> usize {
        match op.c {
            0 => 0,
            1 => {
                out[0] = self.reg(Self::packed_call_reg(op, 0));
                1
            }
            2 => {
                out[0] = self.reg(Self::packed_call_reg(op, 0));
                out[1] = self.reg(Self::packed_call_reg(op, 1));
                2
            }
            3 => {
                out[0] = self.reg(Self::packed_call_reg(op, 0));
                out[1] = self.reg(Self::packed_call_reg(op, 1));
                out[2] = self.reg(Self::packed_call_reg(op, 2));
                3
            }
            4 => {
                out[0] = self.reg(Self::packed_call_reg(op, 0));
                out[1] = self.reg(Self::packed_call_reg(op, 1));
                out[2] = self.reg(Self::packed_call_reg(op, 2));
                out[3] = self.reg(Self::packed_call_reg(op, 3));
                4
            }
            _ => std::hint::unreachable_unchecked(),
        }
    }

    #[inline(always)]
    unsafe fn gather_call_words_const<const N: usize>(&self, op: &PreparedOp) -> [RegisterWord; 4] {
        debug_assert!(N <= 4);
        let mut out = [RegisterWord(0); 4];
        if N > 0 { out[0] = self.reg(Self::packed_call_reg(op, 0)); }
        if N > 1 { out[1] = self.reg(Self::packed_call_reg(op, 1)); }
        if N > 2 { out[2] = self.reg(Self::packed_call_reg(op, 2)); }
        if N > 3 { out[3] = self.reg(Self::packed_call_reg(op, 3)); }
        out
    }

    #[inline(always)]
    unsafe fn gather_call_words_large(&self, op: &PreparedOp, out: &mut [RegisterWord; 17]) -> usize {
        let function = &*self.hot_function;
        let count = op.c as usize;
        debug_assert!(count > 4);
        let regs = function.call_args.get_unchecked(op.args_offset as usize..op.args_offset as usize + count);
        let mut index = 0usize;
        while index < count {
            out[index] = self.reg(regs.get_unchecked(index).0);
            index += 1;
        }
        count
    }

    fn push_frame_direct(
        &mut self,
        function_index: u32,
        sources: &[RegisterWord],
        closure: Option<Value>,
        return_dst: Option<Reg>,
    ) -> Result<(), RuntimeError> {
        let prepared = &self.prepared.functions[function_index as usize];
        let uses_this = function_kind_uses_this(prepared.kind);
        let (receiver, args) = if uses_this {
            let (first, rest) = sources.split_first().ok_or_else(|| RuntimeError::new("method-like call has no receiver"))?;
            (Some(*first), rest)
        } else {
            (None, sources)
        };
        self.push_frame(function_index, receiver, args, closure, return_dst)
    }

    fn push_frame(
        &mut self,
        function_index: u32,
        receiver: Option<RegisterWord>,
        args: &[RegisterWord],
        closure: Option<Value>,
        return_dst: Option<Reg>,
    ) -> Result<(), RuntimeError> {
        if args.len() > ARGUMENT_REGISTER_COUNT {
            return Err(RuntimeError::new("too many call arguments"));
        }

        // Prepared function metadata is immutable after VM construction. Keep a
        // raw pointer so frame construction can mutate the current fiber without
        // cloning FrameLayout or fighting borrow splitting on `self`.
        let prepared_ptr = unsafe {
            self.prepared.functions.as_ptr().add(function_index as usize)
        };
        let prepared = unsafe { &*prepared_ptr };
        let layout = &prepared.frame;
        let reg_count = layout.general_register_count as usize;
        let module_value = self.modules[prepared.module_slot as usize].module_value;
        let fiber_value = self.fiber(self.current_fiber).object;

        // Closure captures live in a stable heap slot. The heap is not mutated
        // while constructing the frame, so retaining this slice is safe here.
        let captures: &[RegisterWord] = if let Some(closure_value) = closure {
            match self.heap.get(closure_value) {
                Some(HeapObject::Closure { function_index: actual, captures, .. })
                    if *actual == function_index => captures.as_ref(),
                Some(HeapObject::Closure { .. }) => {
                    return Err(RuntimeError::new("closure/function mismatch"));
                }
                _ => return Err(RuntimeError::new("CallClosure receiver is not a closure")),
            }
        } else {
            &[]
        };

        let current = self.current_fiber;
        let cell = self.fibers[current as usize].as_ref().expect("live fiber");
        let fiber = unsafe { cell.get_mut() };
        let depth = fiber.frames.len();
        let base = fiber.register_top;
        let needed = base + reg_count;
        // External stack storage is already charged at its current capacities.
        // Re-account only when this push can force one of those Vecs to grow.
        let may_grow_external = depth == fiber.frames.capacity()
            || (fiber.banks.len() <= depth && fiber.banks.len() == fiber.banks.capacity())
            || needed > fiber.registers.capacity();
        if fiber.registers.len() < needed {
            fiber.registers.resize(needed, RegisterWord(0));
        }
        fiber.register_top = needed;

        // FrameBanks are retained at the high-water mark instead of being
        // embedded in Frame. Ordinary calls therefore push only the tiny Frame
        // metadata object; the 30-word architectural bank is reused in place.
        if fiber.banks.len() <= depth {
            fiber.banks.push(FrameBanks::default());
        }
        let banks = unsafe { fiber.banks.get_unchecked_mut(depth) };

        // Valid bytecode never reads an undefined transfer/special register, so
        // reused slots do not need to be bulk-zeroed. Initialize only values that
        // are architecturally defined on frame entry.
        if let Some(receiver) = receiver {
            banks.special[SpecialReg::This.index()] = receiver;
        }
        if let Some(closure) = closure {
            banks.special[SpecialReg::Closure.index()] = closure.word();
        }
        banks.special[SpecialReg::Module.index()] = module_value.word();
        banks.special[SpecialReg::Fiber.index()] = fiber_value.word();

        let mut index = 0usize;
        while index < args.len() {
            let value = unsafe { *args.get_unchecked(index) };
            if prepared.needs_argument_bank {
                banks.arguments[index] = value;
            }
            let dst = unsafe { *prepared.entry_arg_dsts.get_unchecked(index) };
            if dst != u16::MAX {
                unsafe {
                    *fiber.registers.get_unchecked_mut(base + dst as usize) = value;
                }
            }
            index += 1;
        }
        if let Some(receiver) = receiver {
            if prepared.entry_this_dst != u16::MAX {
                unsafe {
                    *fiber
                        .registers
                        .get_unchecked_mut(base + prepared.entry_this_dst as usize) = receiver;
                }
            }
        }

        // Captures are less common and may have arbitrary sparse destinations;
        // retain the compact precomputed binding list for them.
        for binding in &layout.captures {
            let value = captures
                .get(binding.capture_index as usize)
                .copied()
                .ok_or_else(|| RuntimeError::new("closure capture binding is out of range"))?;
            unsafe {
                *fiber.registers.get_unchecked_mut(base + binding.destination.index()) = value;
            }
        }

        let mut frame = Frame::new(
            function_index,
            base,
            reg_count as u16,
            return_dst,
        );
        frame.root_map = prepared.entry_gc_map;
        fiber.frames.push(frame);
        let external_update = may_grow_external.then(|| {
            (fiber.object, fiber_state_external_bytes(fiber))
        });
        if let Some((owner, bytes)) = external_update {
            self.heap.set_external_bytes(owner, bytes);
        }
        self.refresh_hot_frame();
        Ok(())
    }

    fn return_from_frame(&mut self, value0: RegisterWord, value1: Option<RegisterWord>) {
        let current = self.current_fiber;
        let (return_dst, base, finished_function) = {
            let fiber = self.fiber_mut(current);
            let frame = match fiber.frames.pop() {
                Some(frame) => frame,
                None => { self.fail(RuntimeError::new("return with empty frame stack")); return; }
            };
            (frame.return_dst, frame.reg_base, frame.function)
        };

        // A module becomes initialized exactly when its ModuleBody returns.
        // Cyclic imports observe the partially initialized module object, as Wren
        // requires, but subsequent imports do not execute the body again.
        let (finished_kind, finished_module) = {
            let prepared = unsafe {
                self.prepared.functions.get_unchecked(finished_function as usize)
            };
            (prepared.kind, prepared.module_slot)
        };
        if finished_kind == FunctionKind::ModuleBody {
            let module = unsafe {
                self.modules.get_unchecked_mut(finished_module as usize)
            };
            module.initialized = true;
            module.initializing = false;
        }

        let prepared_functions = self.prepared.functions.as_ptr();
        let has_caller = {
            let fiber = self.fiber_mut(current);
            fiber.register_top = base;
            let depth = fiber.frames.len();
            if depth != 0 {
                let caller = unsafe { fiber.frames.get_unchecked(depth - 1) };
                let caller_base = caller.reg_base;
                let caller_function = caller.function as usize;
                if unsafe { (*prepared_functions.add(caller_function)).needs_return_bank } {
                    let banks = unsafe { fiber.banks.get_unchecked_mut(depth - 1) };
                    banks.returns[0] = value0;
                    if let Some(value1) = value1 { banks.returns[1] = value1; }
                }
                if let Some(dst) = return_dst {
                    unsafe {
                        *fiber.registers.get_unchecked_mut(caller_base + dst.index()) = value0;
                    }
                }
                true
            } else {
                false
            }
        };
        if has_caller {
            self.refresh_hot_frame();
            return;
        }

        let return_kind = unsafe {
            self.prepared.functions.get_unchecked(finished_function as usize).return_kind
        };
        let value = match return_kind {
            RegisterKind::Value => Value::from_word(value0),
            RegisterKind::Int32 => Value::from_num(word_i32(value0) as f64),
            RegisterKind::F64 => Value::from_num(word_f64(value0)),
            RegisterKind::RawPtr => {
                self.fail(RuntimeError::new("raw pointer escaped through a Wren-visible return"));
                return;
            }
        };
        self.finish_fiber(value);
    }

    #[inline]
    fn allocate_promise(&mut self, state: PromiseState) -> Value {
        self.heap.alloc(HeapObject::Promise {
            state: Arc::new(PromiseCell::new(state)),
        })
    }

    #[inline]
    fn promise_cell(&self, promise: Value) -> Result<Arc<PromiseCell>, RuntimeError> {
        match self.heap.get(promise) {
            Some(HeapObject::Promise { state }) => Ok(state.clone()),
            _ => Err(RuntimeError::new("await expects a Promise")),
        }
    }

    #[inline]
    fn enqueue_ready_execution(&mut self, id: u32) {
        let Some(cell) = self.fibers.get(id as usize).and_then(|slot| slot.as_ref()) else {
            return;
        };
        // SAFETY: scheduler mutation is confined to the owning interpreter thread.
        let state = unsafe { cell.get_mut() };
        if matches!(state.status, FiberStatus::Done | FiberStatus::Failed | FiberStatus::Ready) {
            return;
        }
        state.status = FiberStatus::Ready;
        self.ready_async.push_back(id);
    }

    /// Select the next runnable async continuation. The public Fiber API never
    /// uses this queue; it keeps its explicit call/transfer graph unchanged.
    fn schedule_next_ready(&mut self) -> bool {
        while let Some(id) = self.ready_async.pop_front() {
            let is_ready = self
                .fibers
                .get(id as usize)
                .and_then(|slot| slot.as_ref())
                .is_some_and(|cell| unsafe { cell.get() }.status == FiberStatus::Ready);
            if !is_ready {
                continue;
            }
            self.switch_current_fiber(id);
            let state = self.fiber_mut(id);
            state.status = FiberStatus::Running;
            state.awaiting_promise = None;
            self.refresh_hot_frame();
            self.run_state = RunState::Running;
            return true;
        }
        false
    }

    fn install_execution_cell(&mut self, id: u32, cell: Arc<FiberCell>) {
        if id as usize == self.fibers.len() {
            self.fibers.push(Some(cell));
        } else {
            self.fibers[id as usize] = Some(cell);
        }
    }

    /// Create an internal task for an async bytecode function and immediately
    /// return its result Promise to the caller. Execution is cooperative: the
    /// task starts at the next await/scheduler boundary rather than recursively
    /// entering a second interpreter dispatch loop.
    fn spawn_async_task(
        &mut self,
        function_index: u32,
        receiver: Option<RegisterWord>,
        args: &[RegisterWord],
        closure: Option<Value>,
    ) -> Result<Value, RuntimeError> {
        let promise = self.allocate_promise(PromiseState::Pending {
            producer: None,
            waiters: Vec::new(),
        });
        // A second managed allocation follows before the Promise becomes reachable
        // from either caller registers or its producer task.
        let promise_root = self.add_root(promise);

        let id = if let Some(id) = self.free_fibers.pop() {
            id
        } else {
            self.fibers.len() as u32
        };
        let mut task_state = FiberState::new(Value::NULL, FiberStatus::Ready);
        task_state.async_promise = Some(promise);
        let task = Arc::new(FiberCell::new(task_state));
        let task_object = self.heap.alloc(HeapObject::AsyncTaskObject {
            task: id,
            state: task.clone(),
            trace_metadata: self.fiber_trace_metadata.clone(),
            promise,
        });
        unsafe { task.get_mut() }.object = task_object;
        self.install_execution_cell(id, task);

        // The pending Promise owns its producer while the task is runnable or
        // suspended. This edge is removed when the Promise settles.
        self.heap.remember_owner(promise);
        let promise_cell = self.promise_cell(promise)?;
        if let PromiseState::Pending { producer, .. } = unsafe { promise_cell.get_mut() } {
            *producer = Some(task_object);
        }

        let source = self.current_fiber;
        self.switch_current_fiber(id);
        let pushed = self.push_frame(function_index, receiver, args, closure, None);
        self.fiber_mut(id).status = FiberStatus::Ready;
        self.switch_current_fiber(source);
        self.refresh_hot_frame();

        if let Err(error) = pushed {
            self.fiber_mut(id).status = FiberStatus::Failed;
            self.release_root(promise_root);
            return Err(error);
        }

        self.ready_async.push_back(id);
        self.release_root(promise_root);
        Ok(promise)
    }

    /// Common bytecode-call boundary. Async callees return a Promise immediately;
    /// synchronous callees retain the existing frame-push fast path.
    fn enter_bytecode_call(
        &mut self,
        op: &PreparedOp,
        function: u32,
        receiver: Option<RegisterWord>,
        args: &[RegisterWord],
        closure: Option<Value>,
    ) {
        self.mark_gc_point(op);
        unsafe { self.advance(op); }
        self.flush_hot_pc();

        if self.prepared.functions[function as usize].is_async {
            self.maybe_collect();
            match self.spawn_async_task(function, receiver, args, closure) {
                Ok(promise) => unsafe { self.set_reg(op.a, promise.word()) },
                Err(error) => self.fail_at(op, error.message),
            }
        } else if let Err(error) = self.push_frame(
            function,
            receiver,
            args,
            closure,
            Some(Reg(op.a)),
        ) {
            self.fail_at(op, error.message);
        }
    }

    fn enter_direct_bytecode_call(
        &mut self,
        op: &PreparedOp,
        function: u32,
        sources: &[RegisterWord],
        closure: Option<Value>,
    ) {
        let uses_this = function_kind_uses_this(self.prepared.functions[function as usize].kind);
        let (receiver, args) = if uses_this {
            let Some((first, rest)) = sources.split_first() else {
                self.fail_at(op, "method-like call has no receiver");
                return;
            };
            (Some(*first), rest)
        } else {
            (None, sources)
        };
        self.enter_bytecode_call(op, function, receiver, args, closure);
    }

    fn resume_promise_waiter(
        &mut self,
        promise: Value,
        fiber: u32,
        fiber_object: Value,
        dst: Reg,
        value: Value,
    ) {
        // The suspended execution object can be mature while `value` is young.
        // Its register stack lives in external Arc storage but is traced as part
        // of that managed owner, so the owner barrier must precede the write.
        self.heap.remember_if_reference(fiber_object, value);
        let Some(cell) = self.fibers.get(fiber as usize).and_then(|slot| slot.as_ref()) else {
            return;
        };
        // SAFETY: scheduler and Promise settlement run on the owning interpreter thread.
        let state = unsafe { cell.get_mut() };
        if state.object != fiber_object
            || state.awaiting_promise != Some(promise)
            || matches!(state.status, FiberStatus::Done | FiberStatus::Failed)
        {
            return;
        }
        let Some(frame) = state.frames.last() else { return; };
        let index = frame.reg_base + dst.index();
        if index >= state.registers.len() { return; }
        state.registers[index] = value.word();
        state.awaiting_promise = None;
        state.suspend_dst = None;
        state.status = FiberStatus::Ready;
        self.ready_async.push_back(fiber);
    }

    fn reject_execution(&mut self, id: u32, reason: Value) {
        let owner = match self.fibers.get(id as usize).and_then(|slot| slot.as_ref()) {
            Some(cell) => unsafe { cell.get() }.object,
            None => return,
        };
        self.heap.remember_if_reference(owner, reason);
        let result_promise = {
            let Some(cell) = self.fibers.get(id as usize).and_then(|slot| slot.as_ref()) else {
                return;
            };
            // SAFETY: scheduler mutation is confined to the owning interpreter thread.
            let state = unsafe { cell.get_mut() };
            state.status = FiberStatus::Failed;
            state.error = reason;
            state.awaiting_promise = None;
            state.suspend_dst = None;
            state.frames.clear();
            state.register_top = 0;
            state.async_promise
        };

        if id == 0 {
            self.switch_current_fiber(0);
            self.error = Some(RuntimeError::new(format!(
                "awaited Promise rejected: {:?}", reason
            )));
            self.run_state = RunState::Failed;
            self.clear_hot_frame();
            return;
        }

        if let Some(promise) = result_promise {
            let _ = self.settle_promise(promise, true, reason);
        }
    }

    fn reject_promise_waiter(
        &mut self,
        promise: Value,
        fiber: u32,
        fiber_object: Value,
        reason: Value,
    ) {
        let valid = self
            .fibers
            .get(fiber as usize)
            .and_then(|slot| slot.as_ref())
            .is_some_and(|cell| {
                let state = unsafe { cell.get() };
                state.object == fiber_object && state.awaiting_promise == Some(promise)
            });
        if valid {
            self.reject_execution(fiber, reason);
        }
    }

    fn settle_promise(
        &mut self,
        promise: Value,
        rejected: bool,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let cell = self.promise_cell(promise)?;
        self.heap.remember_owner(promise);
        let replacement = if rejected {
            PromiseState::Rejected(value)
        } else {
            PromiseState::Fulfilled(value)
        };
        let previous = unsafe { std::mem::replace(cell.get_mut(), replacement) };
        let waiters = match previous {
            PromiseState::Pending { waiters, .. } => waiters,
            settled => {
                unsafe { *cell.get_mut() = settled; }
                return Err(RuntimeError::new("Promise has already been settled"));
            }
        };
        self.heap.set_external_bytes(promise, 0);

        for waiter in waiters {
            match waiter {
                PromiseWaiter::Resume { fiber, fiber_object, dst } => {
                    if rejected {
                        self.reject_promise_waiter(promise, fiber, fiber_object, value);
                    } else {
                        self.resume_promise_waiter(promise, fiber, fiber_object, dst, value);
                    }
                }
                PromiseWaiter::Forward { promise: target } => {
                    // A Forward waiter is installed only on a still-pending target.
                    // If user/runtime code has already settled it, preserve that
                    // first settlement and ignore the stale forwarding edge.
                    let _ = self.settle_promise(target, rejected, value);
                }
            }
        }
        Ok(())
    }

    /// Resolve an async function's result Promise. Returning another Promise
    /// adopts its state instead of producing a nested Promise.
    fn resolve_async_result(&mut self, promise: Value, value: Value) -> Result<(), RuntimeError> {
        if promise == value {
            // Self-resolution would otherwise create a permanently pending cycle.
            return self.settle_promise(promise, true, Value::NULL);
        }

        let Some(inner_cell) = (match self.heap.get(value) {
            Some(HeapObject::Promise { state }) => Some(state.clone()),
            _ => None,
        }) else {
            return self.settle_promise(promise, false, value);
        };

        // Promise state belongs to this interpreter thread. GC only observes it
        // after STW, so it is safe to inspect/mutate through the stable Arc cell.
        match unsafe { inner_cell.get_mut() } {
            PromiseState::Fulfilled(result) => self.settle_promise(promise, false, *result),
            PromiseState::Rejected(reason) => self.settle_promise(promise, true, *reason),
            PromiseState::Pending { waiters, .. } => {
                self.heap.remember_owner(value);
                waiters.push(PromiseWaiter::Forward { promise });
                self.heap.set_external_bytes(
                    value,
                    waiters.capacity() * std::mem::size_of::<PromiseWaiter>(),
                );

                // Replace the completed task as the outer Promise's dependency.
                // This keeps the adopted inner Promise alive until forwarding fires.
                self.heap.remember_if_reference(promise, value);
                let outer = self.promise_cell(promise)?;
                match unsafe { outer.get_mut() } {
                    PromiseState::Pending { producer, .. } => *producer = Some(value),
                    _ => return Err(RuntimeError::new("async result Promise settled before adoption")),
                }
                Ok(())
            }
        }
    }

    fn await_promise(&mut self, op: &PreparedOp, promise: Value, dst: Reg) {
        let cell = match self.promise_cell(promise) {
            Ok(cell) => cell,
            Err(error) => {
                self.fail_at(op, error.message);
                return;
            }
        };

        // SAFETY: Promise state is mutated only by this interpreter thread. GC
        // observes it only after a stop-the-world rendezvous.
        match unsafe { cell.get_mut() } {
            PromiseState::Fulfilled(value) => unsafe {
                self.set_reg(dst.0, (*value).word());
                self.advance(op);
            },
            PromiseState::Rejected(reason) => {
                let reason = *reason;
                self.mark_gc_point(op);
                unsafe { self.advance(op); }
                self.flush_hot_pc();
                let current = self.current_fiber;
                self.reject_execution(current, reason);
                if self.run_state != RunState::Failed && !self.schedule_next_ready() {
                    self.switch_current_fiber(0);
                    self.run_state = if self.fiber(0).status == FiberStatus::Done {
                        RunState::Returned
                    } else {
                        RunState::Suspended
                    };
                    self.clear_hot_frame();
                }
            }
            PromiseState::Pending { waiters, .. } => {
                self.mark_gc_point(op);
                unsafe { self.advance(op); }
                self.flush_hot_pc();

                let current = self.current_fiber;
                let fiber_object = self.fiber(current).object;
                self.heap.remember_owner(promise);
                waiters.push(PromiseWaiter::Resume { fiber: current, fiber_object, dst });
                let external = waiters.capacity() * std::mem::size_of::<PromiseWaiter>();
                self.heap.set_external_bytes(promise, external);

                // The suspended execution object may already be mature while the
                // Promise is young. The Promise becomes an outgoing edge of the
                // execution state, so log the managed owner before publishing it.
                self.heap.remember_if_reference(fiber_object, promise);
                let state = self.fiber_mut(current);
                state.status = FiberStatus::Suspended;
                state.awaiting_promise = Some(promise);
                state.suspend_dst = Some(dst);

                if !self.schedule_next_ready() {
                    // Park only when the public execution itself is waiting. A
                    // detached async task must not keep an already-completed main
                    // module artificially suspended just because its Promise is
                    // still pending.
                    self.switch_current_fiber(0);
                    if self.fiber(0).status == FiberStatus::Done {
                        self.result = self.fiber(0).last_value;
                        self.run_state = RunState::Returned;
                    } else {
                        self.result = promise;
                        self.run_state = RunState::Suspended;
                    }
                    self.clear_hot_frame();
                }
            }
        }
    }

    fn finish_fiber(&mut self, value: Value) {
        let current = self.current_fiber;
        let async_promise = self.fiber(current).async_promise;

        if let Some(promise) = async_promise {
            {
                let task = self.fiber_mut(current);
                task.status = FiberStatus::Done;
                task.last_value = value;
                task.awaiting_promise = None;
            }
            if let Err(error) = self.resolve_async_result(promise, value) {
                self.fail(error);
                return;
            }
            if self.run_state == RunState::Failed {
                return;
            }
            if self.schedule_next_ready() {
                return;
            }
            // No async continuation is runnable. Return control to the public
            // main execution object before exposing an idle VM state; internal
            // AsyncTask objects must never become the externally current Fiber.
            self.switch_current_fiber(0);
            if self.fiber(0).status == FiberStatus::Done {
                self.run_state = RunState::Returned;
            } else {
                self.result = self.fiber(0).awaiting_promise.unwrap_or(Value::NULL);
                self.run_state = RunState::Suspended;
            }
            self.clear_hot_frame();
            return;
        }

        if current == 0 {
            let main = self.fiber_mut(0);
            main.status = FiberStatus::Done;
            main.last_value = value;
            self.result = value;
            if self.schedule_next_ready() {
                return;
            }
            self.run_state = RunState::Returned;
            self.clear_hot_frame();
            return;
        }

        let link = {
            let fiber = self.fiber_mut(current);
            fiber.status = FiberStatus::Done;
            fiber.last_value = value;
            fiber.link.take()
        };
        if let Some(link) = link {
            self.switch_current_fiber(link.caller);
            let caller = self.fiber_mut(link.caller);
            caller.status = FiberStatus::Running;
            if let Some(frame) = caller.frames.last() {
                let index = frame.reg_base + link.dst.index();
                caller.registers[index] = value.word();
            }
            self.refresh_hot_frame();
        } else {
            self.result = value;
            self.run_state = RunState::Returned;
            self.clear_hot_frame();
        }
    }
}

impl Interpreter<'_> {
    #[inline(always)]
    #[inline(always)]
    fn receiver_matches_class(&self, value: Value, expected: ReceiverClass) -> bool {
        let kind = expected.0 >> 56;
        if kind == (CLASS_KIND_BUILTIN >> 56) {
            return match (expected.0 & 0xff) as usize {
                x if x == BuiltinClass::Null as usize => value.is_null(),
                x if x == BuiltinClass::Bool as usize => value.is_bool(),
                x if x == BuiltinClass::Num as usize => value.is_num(),
                _ => self.heap.receiver_class(value) == Some(expected),
            };
        }
        self.heap.receiver_class(value) == Some(expected)
    }

    #[inline(always)]
    fn receiver_class(&self, value: Value) -> Result<ReceiverClass, RuntimeError> {
        // Numbers dominate operator dispatch and are the cheapest tag to classify.
        // Test them first; null/bool remain two direct-word comparisons, while all
        // Heap objects share the ReceiverClass cached in their managed payload.
        if value.is_num() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Num as u64));
        }
        if value.is_null() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Null as u64));
        }
        if value.is_bool() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Bool as u64));
        }
        self.heap
            .receiver_class(value)
            .ok_or_else(|| RuntimeError::new("value has no Wren class"))
    }

    /// Finalizes a Wren class's immutable runtime shape.
    ///
    /// This performs both of Wren's cheap-object optimizations in one cold
    /// class-linking step: copy-down inheritance and fixed-layout field relocation.
    fn link_class_layout(
        &mut self,
        class_slot: u32,
        superclass: Option<u32>,
    ) -> Result<(), RuntimeError> {
        let class_index = class_slot as usize;
        if class_index >= self.classes.len() {
            return Err(RuntimeError::new("CreateClass class slot out of range"));
        }

        if self.classes[class_index].layout_finalized {
            if self.classes[class_index].superclass != superclass {
                return Err(RuntimeError::new("class was relinked with a different superclass"));
            }
            return Ok(());
        }

        let (field_base, inherited_instance_methods) =
            if let Some(super_slot) = superclass {
                let super_class = self
                    .classes
                    .get(super_slot as usize)
                    .ok_or_else(|| RuntimeError::new("superclass slot out of range"))?;
                if !super_class.layout_finalized {
                    return Err(RuntimeError::new(
                        "superclass layout must be finalized before subclass creation",
                    ));
                }
                if self.classes[class_index].is_foreign
                    && super_class.total_instance_field_count != 0
                {
                    return Err(RuntimeError::new(
                        "a foreign class cannot inherit from a class with instance fields",
                    ));
                }
                (
                    super_class.total_instance_field_count,
                    Some(super_class.instance_methods.clone()),
                )
            } else {
                (0, None)
            };

        let own = self.classes[class_index].own_instance_field_count;
        let total = field_base
            .checked_add(own)
            .ok_or_else(|| RuntimeError::new("class instance field layout exceeds 65535 slots"))?;

        {
            let class = &mut self.classes[class_index];
            class.superclass = superclass;
            class.field_base = field_base;
            class.total_instance_field_count = total;
            if let Some(methods) = inherited_instance_methods {
                class.instance_methods = methods;
            }
            class.layout_finalized = true;
        }

        let owner = self.classes[class_index].id;
        let base = field_base as u32;
        for function in &mut self.prepared.functions {
            if function.field_owner != Some(owner) || function.field_slots_relocated {
                continue;
            }
            for &op_index in &function.field_access_ops {
                let op = function
                    .ops
                    .get_mut(op_index as usize)
                    .ok_or_else(|| RuntimeError::new("field relocation op index out of range"))?;
                let relocated = op
                    .x
                    .checked_add(base)
                    .ok_or_else(|| RuntimeError::new("field slot relocation overflow"))?;
                if relocated > u16::MAX as u32 {
                    return Err(RuntimeError::new(
                        "absolute instance field slot exceeds FieldSlot range",
                    ));
                }
                op.x = relocated;
            }
            function.field_slots_relocated = true;
        }

        Ok(())
    }

    fn lookup_method(
        &self,
        class: ReceiverClass,
        method: MethodSymbol,
        home_class: Option<u32>,
    ) -> Option<CallTarget> {
        let kind = class.0 >> 56;
        let payload = (class.0 & 0x00ff_ffff_ffff_ffff) as u32;
        if kind == (CLASS_KIND_BUILTIN >> 56) {
            return self
                .builtin_methods
                .get(payload as usize)?
                .get(method.0 as usize)
                .copied()
                .flatten();
        }
        if kind == (CLASS_KIND_CORE_STATIC >> 56) {
            return self
                .builtin_static_methods
                .get(payload as usize)?
                .get(method.0 as usize)
                .copied()
                .flatten();
        }

        let is_static = kind == (CLASS_KIND_STATIC >> 56);
        let slot = if let Some(home) = home_class {
            // `super` starts at the superclass of the lexical home class.
            // That superclass already contains copy-down entries from all of
            // its ancestors, so one indexed lookup is sufficient.
            self.classes.get(home as usize)?.superclass?
        } else {
            payload
        };
        let class = self.classes.get(slot as usize)?;
        let table = if is_static {
            &class.static_methods
        } else {
            &class.instance_methods
        };
        table.get(method.0 as usize).copied().flatten()
    }

    #[inline(always)]
    unsafe fn probe_invoke_cache(
        &self,
        function_index: u32,
        ic_slot: u32,
        class: ReceiverClass,
    ) -> Option<CallTarget> {
        debug_assert_eq!((*self.hot_frame).function, function_index);
        (*self.hot_inline_caches.add(ic_slot as usize)).probe(class)
    }

    #[cold]
    #[inline(never)]
    fn resolve_invoke_miss(
        &mut self,
        function_index: u32,
        ic_slot: u32,
        class: ReceiverClass,
    ) -> Result<CallTarget, RuntimeError> {
        let descriptor = self.program.functions[function_index as usize]
            .chunk.inline_caches.get(ic_slot as usize)
            .ok_or_else(|| RuntimeError::new("inline cache slot out of range"))?;
        let home_slot = descriptor.home_class.and_then(|id| self.prepared.class_by_id.get(&id).copied());
        let mega_key = (class, descriptor.method, home_slot);
        if self.inline_caches[function_index as usize][ic_slot as usize].megamorphic {
            if let Some(target) = self.megamorphic_cache.get(&mega_key).copied() {
                return Ok(target);
            }
        }

        let target = self.lookup_method(class, descriptor.method, home_slot).ok_or_else(|| {
            RuntimeError::new(format!("method {:?} is not defined for receiver", descriptor.method))
        })?;
        let cache = &mut self.inline_caches[function_index as usize][ic_slot as usize];
        cache.insert(class, target);
        if cache.megamorphic {
            self.megamorphic_cache.insert(mega_key, target);
        }
        Ok(target)
    }

    #[inline]
    fn maybe_quicken_numeric_invoke(
        &mut self,
        function_index: u32,
        op_index: u32,
        ic_slot: u32,
        class: ReceiverClass,
        target: CallTarget,
    ) -> bool {
        let expected_num = ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Num as u64);
        if class != expected_num { return false; }
        let CallTarget::Primitive(primitive) = target else { return false; };

        let Some(descriptor) = self
            .program
            .functions
            .get(function_index as usize)
            .and_then(|function| function.chunk.inline_caches.get(ic_slot as usize))
        else { return false; };
        let Some(signature) = self.program.metadata.method(descriptor.method) else { return false; };
        let Some(name) = self.program.metadata.string(signature.name) else { return false; };
        if !self.host.standard_numeric_method(primitive, name, signature.arity) { return false; }
        let Some(quick) = NumericQuickOp::from_method(name, signature.arity) else { return false; };

        let index = op_index as usize;
        let mut quick_flags = quick as u8;

        // If a numeric comparison feeds the immediately following conditional
        // branch and the Bool temporary is dead on both successors, fuse the pair
        // after quickening. The fast handler then branches on the host Bool without
        // NaN-boxing it or paying another threaded dispatch.
        if quick.is_comparison() {
            if let Some(function) = self.prepared.functions.get(function_index as usize) {
                if index + 1 < function.ops.len() && index + 1 < function.handlers.len() {
                    let branch = &function.ops[index + 1];
                    let handler = function.handlers[index + 1] as usize;
                    let is_false = handler == (h_jump_if_false as Handler) as usize;
                    let is_true = handler == (h_jump_if_true as Handler) as usize;
                    if (is_false || is_true) && branch.a == function.ops[index].a {
                        let bytecode = &self.program.functions[function_index as usize].chunk;
                        let result_reg = Reg(function.ops[index].a);
                        let successor_dead = |prepared_index: u32| -> bool {
                            let Some(successor) = function.original_ops.get(prepared_index as usize) else {
                                return true;
                            };
                            bytecode
                                .live_registers_at(BytecodePc(successor.raw_pc))
                                .is_some_and(|live| !live.general_is_live(result_reg))
                        };
                        if successor_dead(branch.x) && successor_dead(branch.next) {
                            quick_flags |= QUICK_BRANCH_FUSED;
                            if is_true {
                                quick_flags |= QUICK_BRANCH_TRUE;
                            }
                        }
                    }
                }
            }
        }

        let Some(function) = self.prepared.functions.get_mut(function_index as usize) else { return false; };
        if index >= function.handlers.len() || index >= function.ops.len() { return false; }
        if function.ops[index].flags != 0 { return false; }
        if signature.arity == 1 {
            let rhs = if function.ops[index].c <= 4 {
                Reg((function.ops[index].args_offset & 0xff) as u8)
            } else {
                let Some(rhs) = function.call_args.get(function.ops[index].args_offset as usize).copied() else {
                    return false;
                };
                rhs
            };
            // `y` is not consumed by generic Invoke execution. Cache the first
            // argument register here so the quickened handler avoids all argument
            // metadata on every arithmetic operation.
            function.ops[index].y = rhs.0 as u32;
        }
        function.ops[index].flags = quick_flags;
        function.handlers[index] = numeric_quick_handler(quick, quick_flags);
        function.quickened_numeric_ops.push(op_index);
        true
    }

    #[inline]
    fn maybe_quicken_monomorphic_invoke(
        &mut self,
        function_index: u32,
        op_index: u32,
        class: ReceiverClass,
        target: CallTarget,
    ) {
        let index = op_index as usize;
        let Some(function) = self.prepared.functions.get_mut(function_index as usize) else { return; };
        if index >= function.handlers.len() || index >= function.ops.len() { return; }
        if function.ops[index].flags != 0 { return; }

        function.quick_invokes[index] = QuickInvokeSite { class, target };
        function.handlers[index] = match target {
            CallTarget::Bytecode(_) => match function.ops[index].c {
                0 => h_invoke_mono_bytecode_small::<0>,
                1 => h_invoke_mono_bytecode_small::<1>,
                2 => h_invoke_mono_bytecode_small::<2>,
                3 => h_invoke_mono_bytecode_small::<3>,
                4 => h_invoke_mono_bytecode_small::<4>,
                _ => h_invoke_mono_bytecode,
            },
            _ => h_invoke_mono,
        };
        function.quickened_invoke_ops.push(op_index);
    }

    #[inline(always)]
    fn current_quick_invoke(&self) -> QuickInvokeSite {
        let index = self.hot_pc as usize;
        unsafe {
            let quick_invokes = &(*self.hot_function).quick_invokes;
            *quick_invokes.get_unchecked(index)
        }
    }

    #[cold]
    #[inline(never)]
    fn disable_current_monomorphic_invoke(&mut self) {
        let function_index = unsafe { (*self.hot_frame).function as usize };
        let index = self.hot_pc as usize;
        let function = unsafe { self.prepared.functions.get_unchecked_mut(function_index) };
        if index < function.handlers.len() {
            function.quick_invokes[index] = QuickInvokeSite::default();
            function.handlers[index] = h_invoke;
            function.ops[index].flags |= QUICK_GENERIC_DISABLED;
        }
    }

    /// Execute the stock Num primitives directly in the interpreter when the
    /// embedding host explicitly certifies that the binding has the standard
    /// Wren numeric semantics. This avoids routing every arithmetic operator
    /// through Box<dyn Host>, HostContext construction and GC polling when a
    /// trace is cold, rejected or deliberately left in baseline (notably u32
    /// bitwise operations).
    #[inline(always)]
    fn try_standard_numeric_primitive(
        &self,
        op: &PreparedOp,
        primitive: PrimitiveId,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) -> Option<Result<Value, RuntimeError>> {
        let receiver = Value::from_word(receiver);
        let lhs = receiver.as_num()?;

        let function_index = unsafe { (*self.hot_frame).function as usize };
        let descriptor = self
            .program
            .functions
            .get(function_index)?
            .chunk
            .inline_caches
            .get(op.x as usize)?;
        let signature = self.program.metadata.method(descriptor.method)?;
        let name = self.program.metadata.string(signature.name)?;

        if !self
            .host
            .standard_numeric_method(primitive, name, signature.arity)
        {
            return None;
        }

        let argument_value = || -> Result<Value, RuntimeError> {
            args.first()
                .copied()
                .map(Value::from_word)
                .ok_or_else(|| RuntimeError::new("builtin argument missing"))
        };
        let argument_num = || -> Result<f64, RuntimeError> {
            argument_value()?
                .as_num()
                .ok_or_else(|| RuntimeError::new("argument must be a Num"))
        };
        let result = match (name, signature.arity) {
            ("-", 0) => Ok(Value::from_num(-lhs)),
            ("~", 0) => Ok(Value::from_num((!num_to_u32(lhs)) as f64)),

            ("+", 1) => argument_num().map(|rhs| Value::from_num(lhs + rhs)),
            ("-", 1) => argument_num().map(|rhs| Value::from_num(lhs - rhs)),
            ("*", 1) => argument_num().map(|rhs| Value::from_num(lhs * rhs)),
            ("/", 1) => argument_num().map(|rhs| Value::from_num(lhs / rhs)),
            ("%", 1) => argument_num().map(|rhs| Value::from_num(lhs % rhs)),

            ("<", 1) => argument_num().map(|rhs| Value::from_bool(lhs < rhs)),
            ("<=", 1) => argument_num().map(|rhs| Value::from_bool(lhs <= rhs)),
            (">", 1) => argument_num().map(|rhs| Value::from_bool(lhs > rhs)),
            (">=", 1) => argument_num().map(|rhs| Value::from_bool(lhs >= rhs)),
            ("==", 1) => argument_value().map(|rhs| {
                Value::from_bool(rhs.as_num().is_some_and(|rhs| lhs == rhs))
            }),
            ("!=", 1) => argument_value().map(|rhs| {
                Value::from_bool(!rhs.as_num().is_some_and(|rhs| lhs == rhs))
            }),

            ("&", 1) | ("|", 1) | ("^", 1) | ("<<", 1) | (">>", 1) => {
                let left = num_to_u32(lhs);
                argument_num().map(|rhs| {
                    let right = num_to_u32(rhs);
                    Value::from_num(match name {
                        "&" => (left & right) as f64,
                        "|" => (left | right) as f64,
                        "^" => (left ^ right) as f64,
                        "<<" => left.wrapping_shl(right) as f64,
                        ">>" => left.wrapping_shr(right) as f64,
                        _ => unreachable!(),
                    })
                })
            }

            // The host may certify more numeric primitives in the future. Leave
            // those on the ordinary Host path until this direct implementation is
            // extended with identical semantics.
            _ => return None,
        };

        Some(result)
    }

    fn invoke_target(
        &mut self,
        op: &PreparedOp,
        target: CallTarget,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        match target {
            CallTarget::Bytecode(function) => {
                self.enter_bytecode_call(op, function, Some(receiver), args, None);
            }
            CallTarget::Primitive(primitive) => {
                // Stock numeric primitives are pure, non-allocating operations.
                // Keep their baseline path as cheap as an interpreter primitive
                // instead of paying a dyn Host call for every arithmetic opcode.
                if let Some(result) =
                    self.try_standard_numeric_primitive(op, primitive, receiver, args)
                {
                    match result {
                        Ok(result) => unsafe {
                            self.set_reg(op.a, result.word());
                            self.advance(op);
                        },
                        Err(error) => self.fail_at(op, error.message),
                    }
                    return;
                }

                self.mark_gc_point(op);
                self.stage_invoke_args(receiver, args);
                if args.len() <= 4 {
                    let mut values = [Value::NULL; 5];
                    values[0] = Value::from_word(receiver);
                    for (index, word) in args.iter().copied().enumerate() {
                        values[index + 1] = Value::from_word(word);
                    }
                    match self.call_host_primitive(primitive, &values[..args.len() + 1]) {
                        Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                        Err(error) => self.fail_at(op, error.message),
                    }
                } else {
                    let mut values = [Value::NULL; 17];
                    values[0] = Value::from_word(receiver);
                    for (index, word) in args.iter().copied().enumerate() {
                        values[index + 1] = Value::from_word(word);
                    }
                    match self.call_host_primitive(primitive, &values[..args.len() + 1]) {
                        Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                        Err(error) => self.fail_at(op, error.message),
                    }
                }
            }
            CallTarget::Foreign(foreign) => {
                self.mark_gc_point(op);
                self.stage_invoke_args(receiver, args);
                if args.len() <= 4 {
                    let mut values = [Value::NULL; 5];
                    values[0] = Value::from_word(receiver);
                    for (index, word) in args.iter().copied().enumerate() { values[index + 1] = Value::from_word(word); }
                    match self.call_host_foreign(foreign, &values[..args.len() + 1]) {
                        Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                        Err(error) => self.fail_at(op, error.message),
                    }
                } else {
                    let mut values = [Value::NULL; 17];
                    values[0] = Value::from_word(receiver);
                    for (index, word) in args.iter().copied().enumerate() { values[index + 1] = Value::from_word(word); }
                    match self.call_host_foreign(foreign, &values[..args.len() + 1]) {
                        Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                        Err(error) => self.fail_at(op, error.message),
                    }
                }
            }
            CallTarget::NativeBuiltin(native) => self.invoke_native_builtin(op, native, receiver, args),
        }
    }

    fn invoke_native_builtin(
        &mut self,
        op: &PreparedOp,
        native: NativeBuiltin,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        if self.fiber(self.current_fiber).async_promise.is_some()
            && matches!(
                native,
                NativeBuiltin::FiberCall
                    | NativeBuiltin::FiberTry
                    | NativeBuiltin::FiberTransfer
                    | NativeBuiltin::FiberTransferError
                    | NativeBuiltin::FiberYield
                    | NativeBuiltin::FiberSuspend
                    | NativeBuiltin::FiberAbort
            )
        {
            self.fail_at(op, "Fiber control-flow operations are unavailable inside async tasks");
            return;
        }
        match native {
            NativeBuiltin::FnNew => {
                let closure = arg(0);
                if !matches!(self.heap.get(closure), Some(HeapObject::Closure { .. })) {
                    self.fail_at(op, "Fn.new expects a function");
                    return;
                }
                unsafe { self.set_reg(op.a, closure.word()); self.advance(op); }
            }
            NativeBuiltin::FnArity => {
                let closure = Value::from_word(receiver);
                let function = match self.heap.get(closure) {
                    Some(HeapObject::Closure { function_index, .. }) => *function_index,
                    _ => { self.fail_at(op, "Fn.arity receiver is not a closure"); return; }
                };
                let arity = self.prepared.functions[function as usize].arity;
                unsafe { self.set_reg(op.a, Value::from_num(arity as f64).word()); self.advance(op); }
            }
            NativeBuiltin::FnCall => {
                let closure = Value::from_word(receiver);
                let function = match self.heap.get(closure) {
                    Some(HeapObject::Closure { function_index, .. }) => *function_index,
                    _ => { self.fail_at(op, "Fn.call receiver is not a closure"); return; }
                };
                self.enter_bytecode_call(op, function, None, args, Some(closure));
            }
            NativeBuiltin::FiberCurrent => {
                let current = self.fiber(self.current_fiber);
                // Async tasks reuse FiberState storage but are deliberately not
                // public Wren Fiber objects. Do not leak the internal task handle.
                let value = if current.async_promise.is_some() {
                    Value::NULL
                } else {
                    current.object
                };
                unsafe { self.set_reg(op.a, value.word()); self.advance(op); }
            }
            NativeBuiltin::FiberNew => {
                self.mark_gc_point(op);
                self.maybe_collect();
                match self.allocate_fiber(arg(0)) {
                    Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
                    Err(error) => self.fail_at(op, error.message),
                }
            }
            NativeBuiltin::FiberCall => self.activate_fiber(op, Value::from_word(receiver), arg(0), true, false),
            NativeBuiltin::FiberTry => self.activate_fiber(op, Value::from_word(receiver), arg(0), true, true),
            NativeBuiltin::FiberTransfer => self.activate_fiber(op, Value::from_word(receiver), arg(0), false, false),
            NativeBuiltin::FiberTransferError => self.transfer_error_to_fiber(op, Value::from_word(receiver), arg(0)),
            NativeBuiltin::FiberYield => self.yield_current_fiber(op, arg(0), Some(Reg(op.a))),
            NativeBuiltin::FiberSuspend => self.suspend_vm(op, Reg(op.a)),
            NativeBuiltin::FiberAbort => {
                let error = arg(0);
                if error.is_null() {
                    unsafe { self.set_reg(op.a, Value::NULL.word()); self.advance(op); }
                } else {
                    self.abort_current_fiber(op, error);
                }
            }
            NativeBuiltin::FiberError => {
                let fiber = Value::from_word(receiver);
                let id = match self.heap.get(fiber) {
                    Some(HeapObject::FiberObject { fiber, .. }) => *fiber,
                    _ => { self.fail_at(op, "Fiber.error receiver is not a Fiber"); return; }
                };
                let value = self.fibers
                    .get(id as usize)
                    .and_then(|slot| slot.as_ref())
                    .map(|state| unsafe { state.get() }.error)
                    .unwrap_or(Value::NULL);
                unsafe { self.set_reg(op.a, value.word()); self.advance(op); }
            }
            NativeBuiltin::FiberIsDone => {
                let fiber = Value::from_word(receiver);
                let id = match self.heap.get(fiber) {
                    Some(HeapObject::FiberObject { fiber, .. }) => *fiber,
                    _ => { self.fail_at(op, "Fiber.isDone receiver is not a Fiber"); return; }
                };
                let done = self
                    .fibers
                    .get(id as usize)
                    .and_then(|slot| slot.as_ref())
                    .map_or(true, |state| matches!(unsafe { state.get() }.status, FiberStatus::Done | FiberStatus::Failed));
                unsafe { self.set_reg(op.a, Value::from_bool(done).word()); self.advance(op); }
            }
            NativeBuiltin::PromiseResolve => {
                let value = arg(0);
                if matches!(self.heap.get(value), Some(HeapObject::Promise { .. })) {
                    unsafe { self.set_reg(op.a, value.word()); self.advance(op); }
                    return;
                }
                self.mark_gc_point(op);
                self.maybe_collect();
                let promise = self.allocate_promise(PromiseState::Fulfilled(value));
                unsafe { self.set_reg(op.a, promise.word()); self.advance(op); }
            }
            NativeBuiltin::PromiseReject => {
                let reason = arg(0);
                self.mark_gc_point(op);
                self.maybe_collect();
                let promise = self.allocate_promise(PromiseState::Rejected(reason));
                unsafe { self.set_reg(op.a, promise.word()); self.advance(op); }
            }
            NativeBuiltin::PromiseIsPending
            | NativeBuiltin::PromiseIsFulfilled
            | NativeBuiltin::PromiseIsRejected => {
                let promise = Value::from_word(receiver);
                let state = match self.heap.get(promise) {
                    Some(HeapObject::Promise { state }) => state.clone(),
                    _ => { self.fail_at(op, "Promise state getter receiver is not a Promise"); return; }
                };
                let matches = match native {
                    NativeBuiltin::PromiseIsPending => matches!(unsafe { state.get() }, PromiseState::Pending { .. }),
                    NativeBuiltin::PromiseIsFulfilled => matches!(unsafe { state.get() }, PromiseState::Fulfilled(_)),
                    NativeBuiltin::PromiseIsRejected => matches!(unsafe { state.get() }, PromiseState::Rejected(_)),
                    _ => false,
                };
                unsafe { self.set_reg(op.a, Value::from_bool(matches).word()); self.advance(op); }
            }
            NativeBuiltin::SystemGc => {
                // System.gc() is itself a GC-capable call. Publish the exact
                // stack map for this Invoke before asking MMTk to stop the world.
                self.mark_gc_point(op);
                self.collect_garbage();
                unsafe { self.set_reg(op.a, Value::NULL.word()); self.advance(op); }
            }
        }
    }

    #[inline]
    fn stage_runtime_call_args(&mut self, args: &[RegisterWord]) {
        let banks = self.current_banks_mut();
        for (index, word) in args.iter().copied().enumerate() {
            banks.arguments[index] = word;
        }
    }

    #[inline]
    fn stage_invoke_args(&mut self, receiver: RegisterWord, args: &[RegisterWord]) {
        let banks = self.current_banks_mut();
        banks.special[SpecialReg::This.index()] = receiver;
        for (index, word) in args.iter().copied().enumerate() {
            banks.arguments[index] = word;
        }
    }

    fn call_host_primitive(&mut self, primitive: PrimitiveId, args: &[Value]) -> Result<Value, RuntimeError> {
        self.maybe_collect();
        let (host, heap) = (&mut self.host, &mut self.heap);
        let mut context = HostContext { heap };
        host.call_primitive(&mut context, primitive, args)
    }

    fn call_host_foreign(&mut self, foreign: ForeignFnId, args: &[Value]) -> Result<Value, RuntimeError> {
        self.maybe_collect();
        let (host, heap) = (&mut self.host, &mut self.heap);
        let mut context = HostContext { heap };
        host.call_foreign(&mut context, foreign, args)
    }

    #[inline(always)]
    fn constant_word(&self, function: u32, index: u32) -> Result<RegisterWord, RuntimeError> {
        self.resolved_constants
            .get(function as usize)
            .and_then(|constants| constants.get(index as usize))
            .copied()
            .ok_or_else(|| RuntimeError::new("constant index out of range"))
    }

    #[inline(always)]
    unsafe fn constant_word_unchecked(&self, function: u32, index: u32) -> RegisterWord {
        *self
            .resolved_constants
            .get_unchecked(function as usize)
            .get_unchecked(index as usize)
    }

    #[inline(always)]
    unsafe fn hot_constant_word_unchecked(&self, index: u32) -> RegisterWord {
        *self.hot_constants.add(index as usize)
    }

    #[inline]
    fn value_to_map_key(&self, value: Value) -> Result<MapKey, RuntimeError> {
        if value.is_null() { return Ok(MapKey::Null); }
        if let Some(value) = value.as_bool() { return Ok(MapKey::Bool(value)); }
        if let Some(number) = value.as_num() {
            let bits = if number == 0.0 { 0.0f64.to_bits() } else if number.is_nan() { CANONICAL_NAN } else { number.to_bits() };
            return Ok(MapKey::Num(bits));
        }
        if let Some(handle) = value.heap_handle() {
            if let Some(HeapObject::String(string)) = self.heap.get(value) {
                return Ok(MapKey::String(string.clone()));
            }
            return Ok(MapKey::Object(handle));
        }
        Err(RuntimeError::new("invalid map key"))
    }

    fn type_matches(&self, word: RegisterWord, expected: &Type) -> bool {
        match expected {
            Type::Bottom => false,
            Type::Top => true,
            Type::Int32 => true, // representation is guaranteed by verifier at typed specialized uses
            Type::F64 => true,
            Type::RawPtr => true,
            Type::Null => Value::from_word(word).is_null(),
            Type::Bool => Value::from_word(word).is_bool(),
            Type::Num => Value::from_word(word).is_num(),
            Type::String => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::String(_))),
            Type::List(_) => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::List(_))),
            Type::Map => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::Map(_))),
            Type::Range => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::Range { .. })),
            Type::Closure(function) => match self.heap.get(Value::from_word(word)) {
                Some(HeapObject::Closure { function_index, .. }) => function.map_or(true, |id| self.prepared.function_by_id.get(&id).copied() == Some(*function_index)),
                _ => false,
            },
            Type::Fiber => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::FiberObject { .. })),
            Type::Module => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::ModuleObject { .. })),
            Type::Cell(_) => matches!(self.heap.get(Value::from_word(word)), Some(HeapObject::Cell(_))),
            Type::Class(set) => self.class_set_matches(Value::from_word(word), set, true),
            Type::Instance(set) => self.class_set_matches(Value::from_word(word), set, false),
            Type::Foreign(class) => match self.heap.get(Value::from_word(word)) {
                Some(HeapObject::Foreign { class_slot, .. }) => class.map_or(true, |id| self.prepared.class_by_id.get(&id).copied() == Some(*class_slot)),
                _ => false,
            },
            Type::Union(types) => types.iter().any(|ty| self.type_matches(word, ty)),
        }
    }

    fn class_set_matches(&self, value: Value, set: &ClassSet, want_class_object: bool) -> bool {
        let slot = match self.heap.get(value) {
            Some(HeapObject::ClassObject { class_slot }) if want_class_object => *class_slot,
            Some(HeapObject::Instance { class_slot, .. }) if !want_class_object => *class_slot,
            Some(HeapObject::Foreign { class_slot, .. }) if !want_class_object => *class_slot,
            _ => return false,
        };
        if set.is_unknown() { return true; }
        let id = self.classes[slot as usize].id;
        set.contains(id)
    }

    fn deopt(&mut self, op: &PreparedOp, deopt_id: u32) {
        if let Some(&resume) = self.current_prepared().deopt_resume.get(&deopt_id) {
            unsafe { self.set_pc(resume); }
        } else {
            self.fail_at(op, format!("missing deopt target for DeoptId#{deopt_id}"));
        }
    }

    #[inline]
    fn maybe_collect(&mut self) {
        // MMTk may request a rendezvous independently of the language-level
        // System.gc switch. Polling is extremely cheap when no GC is requested.
        self.heap.safepoint();
        let collections = self.heap.collection_count();
        if collections != self.last_gc_collections {
            self.last_gc_collections = collections;
            self.reap_dead_fibers();
            // Drop stale idle handles promptly. During active execution the
            // pointer-based provider remains the authoritative exact root set.
            self.sync_persistent_roots();
        }
    }

    pub fn collect_garbage(&mut self) {
        if !self.gc_enabled {
            self.heap.safepoint();
            return;
        }
        // During dispatch run_function()/resume() already owns the provider. If
        // collection is requested while idle, install one for this call only.
        let _idle_gc_roots = if self.hot_frame.is_null() {
            Some(self.gc_root_registration())
        } else {
            None
        };
        self.flush_hot_pc();
        self.heap.collect_full();
        self.last_gc_collections = self.heap.collection_count();
        self.reap_dead_fibers();
        self.sync_persistent_roots();
    }

    fn reap_dead_fibers(&mut self) {
        // Fiber native state is intentionally outside the moving heap. The
        // Managed FiberObject cleanup clears the canonical handle on a GC worker.
        // Reclaim native FiberState storage only after mutators resume, so
        // trace_fiber can never race the Box drop.
        for index in 1..self.fibers.len() {
            let dead = self.fibers[index].as_ref().is_some_and(|cell| {
                let fiber = unsafe { cell.get() };
                let live = match self.heap.get(fiber.object) {
                    Some(HeapObject::FiberObject { fiber: live_id, .. }) =>
                        *live_id == index as u32,
                    Some(HeapObject::AsyncTaskObject { task: live_id, .. }) =>
                        *live_id == index as u32,
                    _ => false,
                };
                !live
            });
            if dead && self.fibers[index].take().is_some() {
                self.free_fibers.push(index as u32);
            }
        }
    }
}

impl Interpreter<'_> {
    fn is_subclass(&self, mut class: u32, target: u32) -> bool {
        loop {
            if class == target { return true; }
            match self.classes[class as usize].superclass {
                Some(parent) => class = parent,
                None => return false,
            }
        }
    }

    fn allocate_fiber(&mut self, closure: Value) -> Result<Value, RuntimeError> {
        let function_index = match self.heap.get(closure) {
            Some(HeapObject::Closure { function_index, .. }) => *function_index,
            _ => return Err(RuntimeError::new("Fiber.new requires a closure")),
        };
        let Some(function) = self.program.functions.get(function_index as usize) else {
            return Err(RuntimeError::new("Fiber.new closure has an invalid function"));
        };
        if function.is_async {
            return Err(RuntimeError::new(
                "Fiber.new cannot wrap an async closure; call the async closure directly",
            ));
        }
        if function.arity > 1 {
            return Err(RuntimeError::new("Fiber function cannot take more than one parameter"));
        }

        let id = if let Some(id) = self.free_fibers.pop() {
            id
        } else {
            self.fibers.len() as u32
        };
        let mut state = FiberState::new(Value::NULL, FiberStatus::New);
        state.starter = Some(closure);
        let fiber = Arc::new(FiberCell::new(state));
        let object = self.heap.alloc(HeapObject::FiberObject {
            fiber: id,
            state: fiber.clone(),
            trace_metadata: self.fiber_trace_metadata.clone(),
        });
        unsafe { fiber.get_mut() }.object = object;
        if id as usize == self.fibers.len() {
            self.fibers.push(Some(fiber));
        } else {
            self.fibers[id as usize] = Some(fiber);
        }
        Ok(object)
    }

    fn fiber_id_from_value(&self, value: Value) -> Result<u32, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::FiberObject { fiber, .. }) => Ok(*fiber),
            _ => Err(RuntimeError::new("value is not a Fiber")),
        }
    }

    fn activate_fiber(
        &mut self,
        op: &PreparedOp,
        fiber_value: Value,
        argument: Value,
        is_call: bool,
        catch_error: bool,
    ) {
        let target = match self.fiber_id_from_value(fiber_value) {
            Ok(id) => id,
            Err(error) => { self.fail_at(op, error.message); return; }
        };
        if target == self.current_fiber {
            self.fail_at(op, "fiber cannot call/transfer to itself");
            return;
        }
        // Wren's root fiber may be transferred to, but it cannot be called.
        // Calling it would create a structured caller edge into the VM's root
        // execution, which the reference implementation explicitly rejects.
        if is_call && target == 0 {
            self.fail_at(op, "Cannot call root fiber.");
            return;
        }

        let status = self.fiber(target).status;
        if matches!(status, FiberStatus::Done | FiberStatus::Failed) {
            self.fail_at(op, "fiber is not resumable");
            return;
        }
        if status == FiberStatus::Running {
            self.fail_at(op, "fiber is already running");
            return;
        }

        // Wren only creates a caller edge for call()/try(). transfer() is a
        // symmetric coroutine switch and deliberately does not remember the
        // source fiber.
        if is_call && self.fiber(target).link.is_some() {
            self.fail_at(op, "fiber has already been called");
            return;
        }

        self.mark_gc_point(op);
        let source = self.current_fiber;
        unsafe { self.advance(op); }
        self.flush_hot_pc();
        self.fiber_mut(source).status = FiberStatus::Suspended;

        if is_call {
            let caller_object = self.fiber(source).object;
            let target_object = self.fiber(target).object;
            self.heap.remember_if_reference(target_object, caller_object);
            self.fiber_mut(target).link = Some(FiberLink {
                caller: source,
                caller_object,
                dst: Reg(op.a),
                catch_error,
            });
        } else {
            // The transfer expression in the source only obtains a value if
            // some fiber explicitly transfers/calls back to it later.
            self.fiber_mut(source).suspend_dst = Some(Reg(op.a));
        }

        self.resume_target_fiber(op, target, argument, if is_call { Some(source) } else { None });
    }

    fn resume_target_fiber(
        &mut self,
        op: &PreparedOp,
        target: u32,
        argument: Value,
        _caller: Option<u32>,
    ) {
        let status = self.fiber(target).status;

        if status == FiberStatus::New {
            let closure = self.fiber(target).starter.unwrap_or(Value::NULL);
            let function_index = match self.heap.get(closure) {
                Some(HeapObject::Closure { function_index, .. }) => *function_index,
                _ => { self.fail_at(op, "new fiber has invalid starter closure"); return; }
            };

            self.switch_current_fiber(target);
            self.fiber_mut(target).status = FiberStatus::Running;
            if let Err(error) = self.push_frame(
                function_index,
                None,
                &[argument.word()],
                Some(closure),
                None,
            ) {
                self.fail_at(op, error.message);
                return;
            }

            let caller_value = self.fiber(target)
                .link
                .map(|link| self.fiber(link.caller).object)
                .unwrap_or(Value::NULL);
            if let Some(banks) = self.fiber_mut(target).top_banks_mut() {
                banks.special[SpecialReg::CallerFiber.index()] = caller_value.word();
                banks.special[SpecialReg::ResumeValue.index()] = argument.word();
            }
            return;
        }

        let suspend_dst = self.fiber_mut(target).suspend_dst.take();
        self.switch_current_fiber(target);
        self.fiber_mut(target).status = FiberStatus::Running;

        if let Some(dst) = suspend_dst {
            let fiber = self.fiber_mut(target);
            if let Some(frame) = fiber.frames.last() {
                let index = frame.reg_base + dst.index();
                unsafe {
                    *fiber.registers.get_unchecked_mut(index) = argument.word();
                }
            }
        }

        let caller_value = self.fiber(target)
            .link
            .map(|link| self.fiber(link.caller).object)
            .unwrap_or(Value::NULL);
        if let Some(banks) = self.fiber_mut(target).top_banks_mut() {
            banks.special[SpecialReg::ResumeValue.index()] = argument.word();
            banks.special[SpecialReg::CallerFiber.index()] = caller_value.word();
        }
        self.refresh_hot_frame();
    }

    fn transfer_error_to_fiber(
        &mut self,
        op: &PreparedOp,
        fiber_value: Value,
        error: Value,
    ) {
        let target = match self.fiber_id_from_value(fiber_value) {
            Ok(id) => id,
            Err(runtime_error) => { self.fail_at(op, runtime_error.message); return; }
        };
        if target == self.current_fiber {
            self.fail_at(op, "fiber cannot transferError to itself");
            return;
        }

        let status = self.fiber(target).status;
        if matches!(status, FiberStatus::Done | FiberStatus::Failed) {
            self.fail_at(op, "fiber is not resumable");
            return;
        }
        if status == FiberStatus::Running {
            self.fail_at(op, "fiber is already running");
            return;
        }

        self.mark_gc_point(op);
        let source = self.current_fiber;
        unsafe { self.advance(op); }
        self.flush_hot_pc();
        self.fiber_mut(source).status = FiberStatus::Suspended;
        self.fiber_mut(source).suspend_dst = Some(Reg(op.a));

        // transferError() switches to the target without becoming its caller,
        // then immediately aborts that target. Error propagation therefore
        // follows the caller chain the target already had (typically from
        // try()), exactly like Wren's runFiber(..., isCall=false) + error.
        self.switch_current_fiber(target);
        self.fiber_mut(target).status = FiberStatus::Running;
        self.refresh_hot_frame();
        self.abort_current_fiber(op, error);
    }

    fn yield_current_fiber(&mut self, op: &PreparedOp, value: Value, suspend_dst: Option<Reg>) {
        let current = self.current_fiber;
        self.mark_gc_point(op);
        unsafe { self.advance(op); }
        self.flush_hot_pc();

        let link = self.fiber_mut(current).link.take();
        self.fiber_mut(current).status = FiberStatus::Suspended;
        self.fiber_mut(current).suspend_dst = suspend_dst;

        if let Some(link) = link {
            self.switch_current_fiber(link.caller);
            self.fiber_mut(link.caller).status = FiberStatus::Running;
            let caller = self.fiber_mut(link.caller);
            if let Some(frame) = caller.frames.last() {
                let index = frame.reg_base + link.dst.index();
                unsafe {
                    *caller.registers.get_unchecked_mut(index) = value.word();
                }
            }
            self.refresh_hot_frame();
        } else {
            // Yielding from a fiber that was reached only via transfer has no
            // parent to return to. Wren exits the interpreter in this case.
            self.result = value;
            self.run_state = RunState::Suspended;
        }
    }

    fn suspend_vm(&mut self, op: &PreparedOp, dst: Reg) {
        self.mark_gc_point(op);
        unsafe { self.advance(op); }
        self.flush_hot_pc();
        let current = self.current_fiber;
        self.fiber_mut(current).status = FiberStatus::Suspended;
        self.fiber_mut(current).suspend_dst = Some(dst);
        self.result = Value::NULL;
        self.run_state = RunState::Suspended;
    }

    fn abort_current_fiber(&mut self, op: &PreparedOp, error: Value) {
        let current = self.current_fiber;
        if current == 0 {
            let (function, pc) = self.current_location(op);
            self.fail_terminal(
                RuntimeError::new(format!("fiber aborted: {:?}", error)).at(function, pc),
            );
            return;
        }

        let owner = self.fiber(current).object;
        self.heap.remember_if_reference(owner, error);
        self.fiber_mut(current).status = FiberStatus::Failed;
        self.fiber_mut(current).error = error;
        self.fiber_mut(current).awaiting_promise = None;
        self.fiber_mut(current).suspend_dst = None;
        let link = self.fiber_mut(current).link.take();
        let Some(link) = link else {
            self.fail_terminal(RuntimeError::new(format!("unhandled fiber error: {:?}", error)));
            return;
        };
        self.switch_current_fiber(link.caller);
        self.fiber_mut(link.caller).status = FiberStatus::Running;
        self.refresh_hot_frame();
        if link.catch_error {
            let caller = self.fiber_mut(link.caller);
            if let Some(frame) = caller.frames.last() {
                let index = frame.reg_base + link.dst.index();
                caller.registers[index] = error.word();
            }
        } else {
            self.abort_current_fiber(op, error);
        }
    }
}

// =============================================================================
// Static token-threading handler table.
// =============================================================================

const OPCODE_COUNT: usize = 123;

static HANDLER_TABLE: [Handler; OPCODE_COUNT] = [
    h_nop, // Nop
    h_trap, // Trap
    h_move, // Move
    h_swap, // Swap
    h_load_null, // LoadNull
    h_load_false, // LoadFalse
    h_load_true, // LoadTrue
    h_load_i8, // LoadI8
    h_load_i32, // LoadI32
    h_load_const, // LoadConst
    h_move_from_arg, // MoveFromArg
    h_move_to_arg, // MoveToArg
    h_move_from_ret, // MoveFromRet
    h_move_to_ret, // MoveToRet
    h_move_from_special, // MoveFromSpecial
    h_move_to_special, // MoveToSpecial
    h_jump, // Jump
    h_jump_if_false, // JumpIfFalse
    h_jump_if_true, // JumpIfTrue
    h_switch, // Switch
    h_return, // Return
    h_return2, // Return2
    h_safepoint, // Safepoint
    h_osr_point, // OsrPoint
    h_guard_type, // GuardType
    h_guard_class, // GuardClass
    h_guard_bounds, // GuardBounds
    h_guard_not_null, // GuardNotNull
    h_guard_int32, // GuardInt32
    h_guard_f64, // GuardF64
    h_bailout, // Bailout
    h_invoke, // Invoke
    h_invoke, // InvokeSuper
    h_call_direct, // CallDirect
    h_call_closure, // CallClosure
    h_call_primitive, // CallPrimitive
    h_call_foreign, // CallForeign
    h_create_class, // CreateClass
    h_bind_method, // BindMethod
    h_bind_foreign_method, // BindForeignMethod
    h_alloc_instance, // AllocInstance
    h_alloc_foreign, // AllocForeign
    h_load_field, // LoadField
    h_store_field, // StoreField
    h_load_static_field, // LoadStaticField
    h_store_static_field, // StoreStaticField
    h_is_instance, // IsInstance
    h_import_module, // ImportModule
    h_import_variable, // ImportVariable
    h_load_module_var, // LoadModuleVar
    h_store_module_var, // StoreModuleVar
    h_alloc_cell, // AllocCell
    h_load_cell, // LoadCell
    h_store_cell, // StoreCell
    h_make_closure, // MakeClosure
    h_alloc_list, // AllocList
    h_alloc_list_cap, // AllocListCap
    h_list_len, // ListLen
    h_list_get, // ListGet
    h_list_set, // ListSet
    h_list_append, // ListAppend
    h_list_insert, // ListInsert
    h_list_remove_at, // ListRemoveAt
    h_alloc_map, // AllocMap
    h_map_len, // MapLen
    h_map_get, // MapGet
    h_map_set, // MapSet
    h_map_remove, // MapRemove
    h_string_len, // StringLen
    h_string_concat, // StringConcat
    h_string_slice, // StringSlice
    h_string_code_point_at, // StringCodePointAt
    h_make_range, // MakeRange
    h_ineg, // INeg
    h_ibitnot, // IBitNot
    h_iadd, // IAdd
    h_isub, // ISub
    h_imul, // IMul
    h_idiv, // IDivS
    h_irem, // IRemS
    h_iand, // IBitAnd
    h_ior, // IBitOr
    h_ixor, // IBitXor
    h_ishl, // IShl
    h_ishru, // IShrU
    h_isar, // ISar
    h_fneg, // FNeg
    h_fadd, // FAdd
    h_fsub, // FSub
    h_fmul, // FMul
    h_fdiv, // FDiv
    h_frem, // FRem
    h_icmp_eq, // ICmpEq
    h_icmp_ne, // ICmpNe
    h_icmp_lt, // ICmpLt
    h_icmp_le, // ICmpLe
    h_icmp_gt, // ICmpGt
    h_icmp_ge, // ICmpGe
    h_fcmp_eq, // FCmpEq
    h_fcmp_ne, // FCmpNe
    h_fcmp_lt, // FCmpLt
    h_fcmp_le, // FCmpLe
    h_fcmp_gt, // FCmpGt
    h_fcmp_ge, // FCmpGe
    h_ref_eq, // RefEq
    h_num_to_i32, // NumToInt32
    h_num_to_f64, // NumToF64
    h_i32_to_num, // Int32ToNum
    h_f64_to_num, // F64ToNum
    h_make_fiber, // MakeFiber
    h_fiber_call, // FiberCall
    h_fiber_try, // FiberTry
    h_fiber_transfer, // FiberTransfer
    h_fiber_transfer_error, // FiberTransferError
    h_fiber_yield, // FiberYield
    h_fiber_suspend, // FiberSuspend
    h_fiber_abort, // FiberAbort
    h_write_barrier, // WriteBarrier
    h_read_barrier, // ReadBarrier
    h_profile_block, // ProfileBlock
    h_profile_numeric, // ProfileNumeric
    h_breakpoint, // Breakpoint
    h_await, // Await
];

#[inline(always)]
fn handler_for(opcode: Opcode) -> Handler {
    // Opcode values are deliberately dense 0..122. Preparation has already
    // validated the byte stream, so dispatch selection is a single indexed
    // load instead of a generated 123-way match.
    unsafe { *HANDLER_TABLE.get_unchecked(opcode as usize) }
}

#[inline(always)]
fn handler_for_prepared(opcode: Opcode, arity: u8) -> Handler {
    match opcode {
        Opcode::CallDirect => match arity {
            0 => h_call_direct_small::<0>,
            1 => h_call_direct_small::<1>,
            2 => h_call_direct_small::<2>,
            3 => h_call_direct_small::<3>,
            4 => h_call_direct_small::<4>,
            _ => h_call_direct,
        },
        Opcode::CallClosure => match arity {
            0 => h_call_closure_small::<0>,
            1 => h_call_closure_small::<1>,
            2 => h_call_closure_small::<2>,
            3 => h_call_closure_small::<3>,
            4 => h_call_closure_small::<4>,
            _ => h_call_closure,
        },
        _ => handler_for(opcode),
    }
}

fn fused_compare_branch_handler(opcode: Opcode) -> Handler {
    match opcode {
        Opcode::ICmpEq => h_fused_icmp_eq_branch,
        Opcode::ICmpNe => h_fused_icmp_ne_branch,
        Opcode::ICmpLt => h_fused_icmp_lt_branch,
        Opcode::ICmpLe => h_fused_icmp_le_branch,
        Opcode::ICmpGt => h_fused_icmp_gt_branch,
        Opcode::ICmpGe => h_fused_icmp_ge_branch,
        Opcode::FCmpEq => h_fused_fcmp_eq_branch,
        Opcode::FCmpNe => h_fused_fcmp_ne_branch,
        Opcode::FCmpLt => h_fused_fcmp_lt_branch,
        Opcode::FCmpLe => h_fused_fcmp_le_branch,
        Opcode::FCmpGt => h_fused_fcmp_gt_branch,
        Opcode::FCmpGe => h_fused_fcmp_ge_branch,
        Opcode::RefEq => h_fused_ref_eq_branch,
        _ => h_unreachable_fused,
    }
}

// =============================================================================
// Tiny hot handlers
// =============================================================================

unsafe fn h_nop(vm: &mut Interpreter, ptr: *const PreparedOp) { vm.advance(&*ptr); }
unsafe fn h_trap(vm: &mut Interpreter, ptr: *const PreparedOp) { vm.fail_at(&*ptr, "TRAP executed"); }
unsafe fn h_unreachable_fused(vm: &mut Interpreter, ptr: *const PreparedOp) { vm.fail_at(&*ptr, "entered fused-away instruction"); }

unsafe fn h_move(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr; let value = vm.reg(op.b); vm.set_reg(op.a, value); vm.advance(op);
}
unsafe fn h_swap(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op=&*ptr; let a=vm.reg(op.a); let b=vm.reg(op.b); vm.set_reg(op.a,b); vm.set_reg(op.b,a); vm.advance(op);
}
unsafe fn h_load_null(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,Value::NULL.word());vm.advance(op)}
unsafe fn h_load_false(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,Value::FALSE.word());vm.advance(op)}
unsafe fn h_load_true(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,Value::TRUE.word());vm.advance(op)}
unsafe fn h_load_i8(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,i32_word(op.b as i8 as i32));vm.advance(op)}
unsafe fn h_load_i32(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,i32_word(op.x as i32));vm.advance(op)}
unsafe fn h_load_const(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let value=vm.hot_constant_word_unchecked(op.x);
    vm.set_reg(op.a,value);
    vm.advance(op)
}
unsafe fn h_move_from_arg(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*vm.current_banks().arguments.get_unchecked(op.b as usize);vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_arg(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);*vm.current_banks_mut().arguments.get_unchecked_mut(op.a as usize)=v;vm.advance(op)}
unsafe fn h_move_from_ret(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*vm.current_banks().returns.get_unchecked(op.b as usize);vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_ret(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);*vm.current_banks_mut().returns.get_unchecked_mut(op.a as usize)=v;vm.advance(op)}
unsafe fn h_move_from_special(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*vm.current_banks().special.get_unchecked(op.b as usize);vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_special(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);*vm.current_banks_mut().special.get_unchecked_mut(op.a as usize)=v;vm.advance(op)}

unsafe fn h_jump(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.branch_to(op,op.x)}
unsafe fn h_jump_if_false(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;if Value::from_word(vm.reg(op.a)).is_falsey(){vm.branch_to(op,op.x)}else{vm.advance(op)}}
unsafe fn h_jump_if_true(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;if !Value::from_word(vm.reg(op.a)).is_falsey(){vm.branch_to(op,op.x)}else{vm.advance(op)}}
unsafe fn h_switch(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let word=vm.reg(op.a); let value=Value::from_word(word);
    let v=if value.is_num(){f64::from_bits(word.0) as i64}else{word_i32(word) as i64};
    let table=(((op.y as u64)<<32)|op.x as u64) as usize as *const PreparedSwitch;
    let target=(*table).target(v); vm.branch_to(op,target)
}
unsafe fn h_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.a);vm.return_from_frame(v,None)}
unsafe fn h_return2(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let a=vm.reg(op.a);let b=vm.reg(op.b);vm.return_from_frame(a,Some(b))}
unsafe fn h_move_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.return_from_frame(v,None)}
unsafe fn h_move_jump(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.set_reg(op.a,v);vm.branch_to(op,op.y)}
unsafe fn h_move_move(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let first=vm.reg(op.b);vm.set_reg(op.a,first);let dst=(op.x&0xff)as u8;let src=((op.x>>8)&0xff)as u8;let second=vm.reg(src);vm.set_reg(dst,second);vm.advance(op)}
unsafe fn h_load_null_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::NULL.word(),None)}
unsafe fn h_load_false_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::FALSE.word(),None)}
unsafe fn h_load_true_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::TRUE.word(),None)}
unsafe fn h_load_i8_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.return_from_frame(i32_word(op.b as i8 as i32),None)}
unsafe fn h_load_i32_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.return_from_frame(i32_word(op.x as i32),None)}
unsafe fn h_load_const_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.hot_constant_word_unchecked(op.x);vm.return_from_frame(v,None)}
unsafe fn h_move_from_special_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*vm.current_banks().special.get_unchecked(op.b as usize);vm.return_from_frame(v,None)}

unsafe fn h_safepoint(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; vm.mark_gc_point(op); vm.maybe_collect(); match vm.host.safepoint(){Ok(())=>vm.advance(op),Err(e)=>vm.fail_at(op,e.message)}
}
unsafe fn h_osr_point(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let (f,pc)=vm.current_location(op); match vm.host.osr_point(f,pc){Ok(())=>vm.advance(op),Err(e)=>vm.fail_at(op,e.message)}
}

unsafe fn h_guard_type(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let f=vm.current_frame().function as usize; let desc:&TypeGuardDescriptor=&vm.program.functions[f].chunk.type_guards[op.x as usize]; let src=vm.reg(op.b);
    if vm.type_matches(src,&desc.expected){vm.set_reg(op.a,src);vm.advance(op)}else{vm.deopt(op,op.y)}
}
unsafe fn h_guard_class(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let f=vm.current_frame().function as usize; let desc:&ClassGuardDescriptor=&vm.program.functions[f].chunk.class_guards[op.x as usize]; let value=Value::from_word(vm.reg(op.b));
    let ok=match vm.heap.get(value){Some(HeapObject::Instance{class_slot,..})|Some(HeapObject::Foreign{class_slot,..})=>{if desc.expected.is_unknown(){true}else{desc.expected.contains(vm.classes[*class_slot as usize].id)}},Some(HeapObject::ClassObject{class_slot})=>{if desc.expected.is_unknown(){true}else{desc.expected.contains(vm.classes[*class_slot as usize].id)}},_=>false};
    if ok{vm.set_reg(op.a,value.word());vm.advance(op)}else{vm.deopt(op,op.y)}
}
unsafe fn h_guard_bounds(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let i=word_i32(vm.reg(op.a));let n=word_i32(vm.reg(op.b));if i>=0&&i<n{vm.advance(op)}else{vm.deopt(op,op.x)}}
unsafe fn h_guard_not_null(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);if !Value::from_word(v).is_null(){vm.set_reg(op.a,v);vm.advance(op)}else{vm.deopt(op,op.x)}}
unsafe fn h_guard_int32(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let v=Value::from_word(vm.reg(op.b)); if let Some(n)=v.as_num(){if n.is_finite()&&n.fract()==0.0&&n>=i32::MIN as f64&&n<=i32::MAX as f64{vm.set_reg(op.a,i32_word(n as i32));vm.advance(op);return;}} vm.deopt(op,op.x)
}
unsafe fn h_guard_f64(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=Value::from_word(vm.reg(op.b));if let Some(n)=v.as_num(){vm.set_reg(op.a,f64_word(n));vm.advance(op)}else{vm.deopt(op,op.x)}}
unsafe fn h_bailout(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let _reason=decode_bailout_reason(op.a).unwrap_or(BailoutReason::Explicit);vm.deopt(op,op.x)}

unsafe fn h_invoke(vm:&mut Interpreter, ptr:*const PreparedOp){
    // Copy the descriptor so quickening may mutate the prepared opcode/handler
    // without aliasing the currently executing PreparedOp reference.
    let op=*ptr;
    let function=(*vm.hot_frame).function;
    let op_index=vm.hot_pc;
    let receiver=vm.reg(op.b);
    let receiver_value=Value::from_word(receiver);
    let class=match vm.receiver_class(receiver_value){
        Ok(class)=>class,
        Err(error)=>{vm.fail_at(&op,error.message);return;}
    };
    let target=match vm.probe_invoke_cache(function,op.x,class){
        Some(target)=>target,
        None=>match vm.resolve_invoke_miss(function,op.x,class){
            Ok(target)=>target,
            Err(error)=>{vm.fail_at(&op,error.message);return;}
        },
    };
    if !vm.maybe_quicken_numeric_invoke(function,op_index,op.x,class,target) {
        vm.maybe_quicken_monomorphic_invoke(function,op_index,class,target);
    }
    if op.c <= 4 {
        let mut args=[RegisterWord(0);4];
        let count=vm.gather_call_words_small(&op,&mut args);
        vm.invoke_target(&op,target,receiver,&args[..count]);
    } else {
        let mut args=[RegisterWord(0);17];
        let count=vm.gather_call_words_large(&op,&mut args);
        vm.invoke_target(&op,target,receiver,&args[..count]);
    }
}

#[inline(always)]
unsafe fn invoke_args_and_target(
    vm: &mut Interpreter,
    op: &PreparedOp,
    target: CallTarget,
    receiver: RegisterWord,
) {
    if op.c <= 4 {
        let mut args=[RegisterWord(0);4];
        let count=vm.gather_call_words_small(op,&mut args);
        vm.invoke_target(op,target,receiver,&args[..count]);
    } else {
        let mut args=[RegisterWord(0);17];
        let count=vm.gather_call_words_large(op,&mut args);
        vm.invoke_target(op,target,receiver,&args[..count]);
    }
}

unsafe fn h_invoke_mono(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=*ptr;
    let receiver=vm.reg(op.b);
    let receiver_value=Value::from_word(receiver);
    let site=vm.current_quick_invoke();
    if !vm.receiver_matches_class(receiver_value,site.class) {
        vm.disable_current_monomorphic_invoke();
        h_invoke(vm,ptr);
        return;
    }
    invoke_args_and_target(vm,&op,site.target,receiver);
}

unsafe fn h_invoke_mono_bytecode_small<const N: usize>(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=*ptr;
    let receiver=vm.reg(op.b);
    let receiver_value=Value::from_word(receiver);
    let site=vm.current_quick_invoke();
    if !vm.receiver_matches_class(receiver_value,site.class) {
        vm.disable_current_monomorphic_invoke();
        h_invoke(vm,ptr);
        return;
    }
    let CallTarget::Bytecode(function)=site.target else {
        h_invoke_mono(vm,ptr);
        return;
    };
    let args=vm.gather_call_words_const::<N>(&op);
    vm.enter_bytecode_call(&op,function,Some(receiver),&args[..N],None);
}

unsafe fn h_invoke_mono_bytecode(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=*ptr;
    let receiver=vm.reg(op.b);
    let receiver_value=Value::from_word(receiver);
    let site=vm.current_quick_invoke();
    if !vm.receiver_matches_class(receiver_value,site.class) {
        vm.disable_current_monomorphic_invoke();
        h_invoke(vm,ptr);
        return;
    }
    let CallTarget::Bytecode(function)=site.target else {
        h_invoke_mono(vm,ptr);
        return;
    };

    // The receiver/argument registers must be read before push_frame can resize
    // the current fiber register stack and invalidate the hot GPR base.
    if op.c <= 4 {
        let mut args=[RegisterWord(0);4];
        let count=vm.gather_call_words_small(&op,&mut args);
        vm.enter_bytecode_call(&op,function,Some(receiver),&args[..count],None);
    } else {
        let mut args=[RegisterWord(0);17];
        let count=vm.gather_call_words_large(&op,&mut args);
        vm.enter_bytecode_call(&op,function,Some(receiver),&args[..count],None);
    }
}

/// Select a monomorphized numeric quick handler once when the Invoke site is
/// patched. The hot path then has one stable indirect branch per operation and
/// no secondary `match` on `+`, `*`, `<`, ... on every execution.
#[inline]
fn numeric_quick_handler(quick: NumericQuickOp, flags: u8) -> Handler {
    const NO_BRANCH: u8 = 0;
    const BRANCH_FALSE: u8 = 1;
    const BRANCH_TRUE: u8 = 2;

    let branch = if (flags & QUICK_BRANCH_FUSED) == 0 {
        NO_BRANCH
    } else if (flags & QUICK_BRANCH_TRUE) != 0 {
        BRANCH_TRUE
    } else {
        BRANCH_FALSE
    };

    macro_rules! comparison_handler {
        ($op:expr) => {{
            match branch {
                NO_BRANCH => h_invoke_num_quick_specialized::<$op, NO_BRANCH> as Handler,
                BRANCH_FALSE => h_invoke_num_quick_specialized::<$op, BRANCH_FALSE> as Handler,
                BRANCH_TRUE => h_invoke_num_quick_specialized::<$op, BRANCH_TRUE> as Handler,
                _ => unreachable!(),
            }
        }};
    }

    match quick {
        NumericQuickOp::Neg => h_invoke_num_quick_specialized::<1, NO_BRANCH> as Handler,
        NumericQuickOp::BitNot => h_invoke_num_quick_specialized::<2, NO_BRANCH> as Handler,
        NumericQuickOp::Add => h_invoke_num_quick_specialized::<3, NO_BRANCH> as Handler,
        NumericQuickOp::Sub => h_invoke_num_quick_specialized::<4, NO_BRANCH> as Handler,
        NumericQuickOp::Mul => h_invoke_num_quick_specialized::<5, NO_BRANCH> as Handler,
        NumericQuickOp::Div => h_invoke_num_quick_specialized::<6, NO_BRANCH> as Handler,
        NumericQuickOp::Rem => h_invoke_num_quick_specialized::<7, NO_BRANCH> as Handler,
        NumericQuickOp::Lt => comparison_handler!(8),
        NumericQuickOp::Le => comparison_handler!(9),
        NumericQuickOp::Gt => comparison_handler!(10),
        NumericQuickOp::Ge => comparison_handler!(11),
        NumericQuickOp::Eq => comparison_handler!(12),
        NumericQuickOp::Ne => comparison_handler!(13),
        NumericQuickOp::BitAnd => h_invoke_num_quick_specialized::<14, NO_BRANCH> as Handler,
        NumericQuickOp::BitOr => h_invoke_num_quick_specialized::<15, NO_BRANCH> as Handler,
        NumericQuickOp::BitXor => h_invoke_num_quick_specialized::<16, NO_BRANCH> as Handler,
        NumericQuickOp::Shl => h_invoke_num_quick_specialized::<17, NO_BRANCH> as Handler,
        NumericQuickOp::ShrU => h_invoke_num_quick_specialized::<18, NO_BRANCH> as Handler,
    }
}

/// Monomorphized stock-Num Invoke. OP is NumericQuickOp as u8. BRANCH is
/// 0=ordinary result, 1=fused JumpIfFalse, 2=fused JumpIfTrue. LLVM sees both
/// constants and removes all operation/branch selection from each instantiation.
#[inline(always)]
unsafe fn h_invoke_num_quick_specialized<const OP: u8, const BRANCH: u8>(
    vm: &mut Interpreter,
    ptr: *const PreparedOp,
) {
    let op = *ptr;
    let receiver = Value::from_word(vm.reg(op.b));
    let Some(lhs) = receiver.as_num() else {
        h_invoke(vm, ptr);
        return;
    };

    // Unary quick operations never read argument metadata. Every binary numeric
    // site caches its first argument register in `y` when quickened.
    let rhs_value = if OP >= 3 {
        Some(Value::from_word(vm.reg(op.y as u8)))
    } else {
        None
    };

    if BRANCH != 0 {
        debug_assert!((8..=13).contains(&OP));
        let condition = match OP {
            8 | 9 | 10 | 11 => {
                let Some(rhs) = rhs_value.and_then(Value::as_num) else {
                    h_invoke(vm, ptr);
                    return;
                };
                match OP {
                    8 => lhs < rhs,
                    9 => lhs <= rhs,
                    10 => lhs > rhs,
                    11 => lhs >= rhs,
                    _ => std::hint::unreachable_unchecked(),
                }
            }
            12 => rhs_value.and_then(Value::as_num).is_some_and(|rhs| lhs == rhs),
            13 => !rhs_value.and_then(Value::as_num).is_some_and(|rhs| lhs == rhs),
            _ => std::hint::unreachable_unchecked(),
        };

        let branch = &*ptr.add(1);
        let take = if BRANCH == 2 { condition } else { !condition };
        // Make the cached PC match the logical branch instruction before taking
        // a possible backedge so tiering observes the correct from-index/PC.
        vm.set_pc(op.next);
        if take {
            vm.branch_to(branch, branch.x);
        } else {
            vm.set_pc(branch.next);
        }
        return;
    }

    let result = match OP {
        1 => Value::from_num(-lhs),
        2 => Value::from_num((!num_to_u32(lhs)) as f64),
        12 => Value::from_bool(rhs_value.and_then(Value::as_num).is_some_and(|rhs| lhs == rhs)),
        13 => Value::from_bool(!rhs_value.and_then(Value::as_num).is_some_and(|rhs| lhs == rhs)),
        3..=11 | 14..=18 => {
            let Some(rhs) = rhs_value.and_then(Value::as_num) else {
                h_invoke(vm, ptr);
                return;
            };
            match OP {
                3 => Value::from_num(lhs + rhs),
                4 => Value::from_num(lhs - rhs),
                5 => Value::from_num(lhs * rhs),
                6 => Value::from_num(lhs / rhs),
                7 => Value::from_num(lhs % rhs),
                8 => Value::from_bool(lhs < rhs),
                9 => Value::from_bool(lhs <= rhs),
                10 => Value::from_bool(lhs > rhs),
                11 => Value::from_bool(lhs >= rhs),
                14 | 15 | 16 | 17 | 18 => {
                    let left = num_to_u32(lhs);
                    let right = num_to_u32(rhs);
                    Value::from_num(match OP {
                        14 => (left & right) as f64,
                        15 => (left | right) as f64,
                        16 => (left ^ right) as f64,
                        17 => left.wrapping_shl(right) as f64,
                        18 => left.wrapping_shr(right) as f64,
                        _ => std::hint::unreachable_unchecked(),
                    })
                }
                _ => std::hint::unreachable_unchecked(),
            }
        }
        _ => std::hint::unreachable_unchecked(),
    };

    vm.set_reg(op.a, result.word());
    vm.advance(&op);
}

unsafe fn h_call_direct_small<const N: usize>(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let words=vm.gather_call_words_const::<N>(op);
    vm.enter_direct_bytecode_call(op,op.x,&words[..N],None);
}

unsafe fn h_call_closure_small<const N: usize>(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let closure=Value::from_word(vm.reg(op.b));
    let function=match vm.heap.get(closure){
        Some(HeapObject::Closure{function_index,..})=>*function_index,
        _=>{vm.fail_at(op,"CallClosure operand is not a closure");return;}
    };
    let words=vm.gather_call_words_const::<N>(op);
    vm.enter_bytecode_call(op,function,None,&words[..N],Some(closure));
}

unsafe fn h_call_direct(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    if op.c <= 4 {
        let mut words=[RegisterWord(0);4];
        let count=vm.gather_call_words_small(op,&mut words);
        vm.enter_direct_bytecode_call(op,op.x,&words[..count],None);
    } else {
        let mut words=[RegisterWord(0);17];
        let count=vm.gather_call_words_large(op,&mut words);
        vm.enter_direct_bytecode_call(op,op.x,&words[..count],None);
    }
}
unsafe fn h_call_closure(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let closure=Value::from_word(vm.reg(op.b));
    let function=match vm.heap.get(closure){
        Some(HeapObject::Closure{function_index,..})=>*function_index,
        _=>{vm.fail_at(op,"CallClosure operand is not a closure");return;}
    };
    if op.c <= 4 {
        let mut words=[RegisterWord(0);4];
        let count=vm.gather_call_words_small(op,&mut words);
        vm.enter_bytecode_call(op,function,None,&words[..count],Some(closure));
    } else {
        let mut words=[RegisterWord(0);17];
        let count=vm.gather_call_words_large(op,&mut words);
        vm.enter_bytecode_call(op,function,None,&words[..count],Some(closure));
    }
}
unsafe fn h_call_primitive(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; vm.mark_gc_point(op);
    if op.c <= 4 {
        let mut words=[RegisterWord(0);4]; let count=vm.gather_call_words_small(op,&mut words); vm.stage_runtime_call_args(&words[..count]);
        let mut args=[Value::NULL;4]; for i in 0..count{args[i]=Value::from_word(words[i]);}
        match vm.call_host_primitive(PrimitiveId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
    } else {
        let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words_large(op,&mut words); vm.stage_runtime_call_args(&words[..count]);
        let mut args=[Value::NULL;17]; for i in 0..count{args[i]=Value::from_word(words[i]);}
        match vm.call_host_primitive(PrimitiveId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
    }
}
unsafe fn h_call_foreign(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; vm.mark_gc_point(op);
    if op.c <= 4 {
        let mut words=[RegisterWord(0);4]; let count=vm.gather_call_words_small(op,&mut words); vm.stage_runtime_call_args(&words[..count]);
        let mut args=[Value::NULL;4]; for i in 0..count{args[i]=Value::from_word(words[i]);}
        match vm.call_host_foreign(ForeignFnId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
    } else {
        let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words_large(op,&mut words); vm.stage_runtime_call_args(&words[..count]);
        let mut args=[Value::NULL;17]; for i in 0..count{args[i]=Value::from_word(words[i]);}
        match vm.call_host_foreign(ForeignFnId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
    }
}

unsafe fn h_create_class(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let slot=op.x;
    let superclass=if op.c!=0{
        let super_value=Value::from_word(vm.reg(op.b));
        match vm.heap.get(super_value){
            Some(HeapObject::ClassObject{class_slot})=>Some(*class_slot),
            _=>{vm.fail_at(op,"CreateClass superclass is not a class");return;}
        }
    }else{None};
    if let Err(error)=vm.link_class_layout(slot,superclass){
        vm.fail_at(op,error.message);
        return;
    }
    let value=vm.classes[slot as usize].class_value;
    vm.set_reg(op.a,value.word());
    vm.advance(op)
}
unsafe fn h_bind_method(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class_value=Value::from_word(vm.reg(op.a));let slot=match vm.heap.get(class_value){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"BindMethod target is not a class");return;}};let function=match vm.prepared.function_by_id.get(&FuncId(op.y)).copied(){Some(v)=>v,None=>{vm.fail_at(op,"BindMethod function does not exist");return;}};let table=if op.b!=0{&mut vm.classes[slot as usize].static_methods}else{&mut vm.classes[slot as usize].instance_methods};match table.get_mut(op.x as usize){Some(entry)=>{*entry=Some(CallTarget::Bytecode(function));vm.advance(op)},None=>vm.fail_at(op,"BindMethod symbol out of range")}}
unsafe fn h_bind_foreign_method(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class_value=Value::from_word(vm.reg(op.a));let slot=match vm.heap.get(class_value){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"BindForeignMethod target is not a class");return;}};let table=if op.b!=0{&mut vm.classes[slot as usize].static_methods}else{&mut vm.classes[slot as usize].instance_methods};match table.get_mut(op.x as usize){Some(entry)=>{*entry=Some(CallTarget::Foreign(ForeignFnId(op.y)));vm.advance(op)},None=>vm.fail_at(op,"BindForeignMethod symbol out of range")}}
unsafe fn h_alloc_instance(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let Some(class)=vm.classes.get(op.x as usize) else{
        vm.fail_at(op,"AllocInstance class slot out of range");
        return;
    };
    if !class.layout_finalized{
        vm.fail_at(op,"AllocInstance used before class layout was finalized");
        return;
    }
    let count=class.total_instance_field_count as usize;
    let value=vm.heap.alloc(HeapObject::Instance{
        class_slot:op.x,
        fields:vec![Value::NULL;count].into_boxed_slice()
    });
    vm.set_reg(op.a,value.word());
    vm.advance(op)
}
unsafe fn h_alloc_foreign(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let size=word_i32(vm.reg(op.b));if size<0{vm.fail_at(op,"negative foreign allocation size");return;}let value=vm.heap.alloc(HeapObject::Foreign{class_slot:op.x,bytes:vec![0;size as usize]});vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_load_field(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let obj=Value::from_word(vm.reg(op.b));
    let Some(HeapObject::Instance{fields,..})=vm.heap.get(obj) else {vm.fail_at(op,"LoadField object is not an instance");return;};
    let slot=op.x as usize;
    debug_assert!(slot < fields.len(), "verified LoadField slot is out of range");
    let value=*fields.get_unchecked(slot);vm.set_reg(op.a,value.word());vm.advance(op)
}
unsafe fn h_load_field_return(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let obj=Value::from_word(vm.reg(op.b));
    let Some(HeapObject::Instance{fields,..})=vm.heap.get(obj) else {vm.fail_at(op,"LoadField object is not an instance");return;};
    let slot=op.x as usize;
    debug_assert!(slot < fields.len(), "verified LoadField slot is out of range");
    let value=*fields.get_unchecked(slot);vm.return_from_frame(value.word(),None)
}
unsafe fn h_store_field(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let obj=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));
    vm.heap.remember_if_reference(obj, value);
    let Some(HeapObject::Instance{fields,..})=vm.heap.get_mut(obj) else {vm.fail_at(op,"StoreField object is not an instance");return;};
    let index=op.x as usize;
    debug_assert!(index < fields.len(), "verified StoreField slot is out of range");
    *fields.get_unchecked_mut(index)=value;vm.advance(op)
}
unsafe fn h_load_static_field(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(class){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"LoadStaticField target is not class");return;}};match vm.classes[slot as usize].static_fields.get(op.x as usize).copied(){Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"static field slot out of range")}}
unsafe fn h_load_static_field_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(class){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"LoadStaticField target is not class");return;}};match vm.classes[slot as usize].static_fields.get(op.x as usize).copied(){Some(v)=>vm.return_from_frame(v.word(),None),None=>vm.fail_at(op,"static field slot out of range")}}
unsafe fn h_store_static_field(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(class){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"StoreStaticField target is not class");return;}};if let Some(field)=vm.classes[slot as usize].static_fields.get_mut(op.x as usize){*field=value;vm.advance(op)}else{vm.fail_at(op,"static field slot out of range")}}
unsafe fn h_is_instance(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let value=Value::from_word(vm.reg(op.b));let class=Value::from_word(vm.reg(op.c));let target=match vm.heap.get(class){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.set_reg(op.a,Value::FALSE.word());vm.advance(op);return;}};let actual=match vm.heap.get(value){Some(HeapObject::Instance{class_slot,..})|Some(HeapObject::Foreign{class_slot,..})=>Some(*class_slot),_=>None};vm.set_reg(op.a,Value::from_bool(actual.map_or(false,|slot|vm.is_subclass(slot,target))).word());vm.advance(op)}

unsafe fn h_import_module(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;
    let slot=match vm.prepared.module_by_name.get(&StringId(op.x)).copied(){
        Some(v)=>v,
        None=>{vm.fail_at(op,"module name not found");return;}
    };

    let (module_value, body, initialized, initializing)={
        let module=vm.modules.get_unchecked(slot as usize);
        (module.module_value,module.body,module.initialized,module.initializing)
    };
    vm.set_reg(op.a,module_value.word());

    // Already initialized, or currently being initialized through a cyclic
    // import: expose the module immediately without running its body again.
    if initialized || initializing || body.is_none() {
        vm.advance(op);
        return;
    }

    vm.mark_gc_point(op);
    vm.advance(op);
    vm.flush_hot_pc();
    vm.modules.get_unchecked_mut(slot as usize).initializing=true;

    // No return destination is needed: the module object was written above.
    // Returning from ModuleBody marks the module initialized and resumes here.
    if let Err(error)=vm.push_frame_direct(body.unwrap_unchecked(),&[],None,None){
        vm.modules.get_unchecked_mut(slot as usize).initializing=false;
        vm.fail_at(op,error.message);
    }
}
unsafe fn h_import_variable(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let module=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(module){Some(HeapObject::ModuleObject{module_slot})=>*module_slot,_=>{vm.fail_at(op,"ImportVariable operand is not module");return;}};let name=StringId(op.x);let index=vm.modules[slot as usize].exports.get(&name).copied();match index.and_then(|i|vm.modules[slot as usize].variables.get(i as usize).copied()){Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"imported variable name is not bound")}}
#[inline(always)]
unsafe fn module_slot_ptr(op: &PreparedOp) -> *mut Value {
    (((op.y as u64) << 32) | op.x as u64) as usize as *mut Value
}

unsafe fn h_load_module_var(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*module_slot_ptr(op);vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_load_module_var_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=*module_slot_ptr(op);vm.return_from_frame(v.word(),None)}
unsafe fn h_store_module_var(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let value=Value::from_word(vm.reg(op.a));*module_slot_ptr(op)=value;vm.advance(op)}

unsafe fn h_alloc_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let value=vm.heap.alloc(HeapObject::Cell(Box::new(vm.reg(op.b))));vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_load_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cell=Value::from_word(vm.reg(op.b));match vm.heap.get(cell){Some(HeapObject::Cell(value))=>{vm.set_reg(op.a,**value);vm.advance(op)},_=>vm.fail_at(op,"LoadCell operand is not cell")}}
unsafe fn h_load_cell_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cell=Value::from_word(vm.reg(op.b));match vm.heap.get(cell){Some(HeapObject::Cell(value))=>vm.return_from_frame(**value,None),_=>vm.fail_at(op,"LoadCell operand is not cell")}}
unsafe fn h_store_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cell=Value::from_word(vm.reg(op.a));let word=vm.reg(op.b);let value=Value::from_word(word);vm.heap.remember_if_reference(cell,value);match vm.heap.get_mut(cell){Some(HeapObject::Cell(slot))=>{**slot=word;vm.advance(op)},_=>vm.fail_at(op,"StoreCell operand is not cell")}}
unsafe fn h_make_closure(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let count=((op.b as u16)|((op.c as u16)<<8)) as usize;let current=vm.current_frame().function as usize;let start=op.args_offset as usize;let mut captures=Vec::with_capacity(count);for index in 0..count{let reg=vm.prepared.functions[current].call_args[start+index];captures.push(vm.reg(reg.0));}let reference_indices=vm.prepared.functions[op.x as usize].capture_reference_indices.clone();let value=vm.heap.alloc(HeapObject::Closure{function_index:op.x,captures:captures.into_boxed_slice(),reference_indices});vm.set_reg(op.a,value.word());vm.advance(op)}

#[inline]
fn normalize_index(index: i32, len: usize) -> Option<usize> {
    let index = if index < 0 { len as i64 + index as i64 } else { index as i64 };
    if index >= 0 && (index as usize) < len { Some(index as usize) } else { None }
}

unsafe fn h_alloc_list(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::List(Vec::new()));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_alloc_list_cap(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cap=word_i32(vm.reg(op.b));if cap<0{vm.fail_at(op,"negative list capacity");return;}vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::List(Vec::with_capacity(cap as usize)));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_list_len(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));match vm.heap.get(list){Some(HeapObject::List(values))=>{vm.set_reg(op.a,i32_word(values.len() as i32));vm.advance(op)},_=>vm.fail_at(op,"ListLen operand is not list")}}
unsafe fn h_list_get(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));let index=word_i32(vm.reg(op.c));let value=match vm.heap.get(list){Some(HeapObject::List(values))=>normalize_index(index,values.len()).map(|i|*values.get_unchecked(i)),_=>None};match value{Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"list index out of bounds")}}
unsafe fn h_list_set(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let index=word_i32(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));vm.heap.remember_if_reference(list,value);match vm.heap.get_mut(list){Some(HeapObject::List(values))=>if let Some(i)=normalize_index(index,values.len()){*values.get_unchecked_mut(i)=value;vm.advance(op)}else{vm.fail_at(op,"list index out of bounds")},_=>vm.fail_at(op,"ListSet operand is not list")}}
unsafe fn h_list_append(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));
    vm.mark_gc_point(op);vm.maybe_collect();vm.heap.remember_if_reference(list,value);
    let resized=match vm.heap.get_mut(list){
        Some(HeapObject::List(values))=>{let old=values.capacity();values.push(value);(values.capacity()!=old).then_some(values.capacity())},
        _=>{vm.fail_at(op,"ListAppend operand is not list");return;}
    };
    if let Some(capacity)=resized{vm.heap.set_external_bytes(list,capacity*std::mem::size_of::<Value>());}
    vm.heap.refresh_jit_view(list);
    vm.advance(op)
}
unsafe fn h_list_insert(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let index=word_i32(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));
    vm.mark_gc_point(op);vm.maybe_collect();vm.heap.remember_if_reference(list,value);
    let resized=match vm.heap.get_mut(list){
        Some(HeapObject::List(values))=>{
            let mut i=index as i64;if i<0{i=values.len() as i64+i+1;}
            if i<0||i>values.len() as i64{vm.fail_at(op,"list insertion index out of bounds");return;}
            let old=values.capacity();values.insert(i as usize,value);(values.capacity()!=old).then_some(values.capacity())
        },
        _=>{vm.fail_at(op,"ListInsert operand is not list");return;}
    };
    if let Some(capacity)=resized{vm.heap.set_external_bytes(list,capacity*std::mem::size_of::<Value>());}
    vm.heap.refresh_jit_view(list);
    vm.advance(op)
}
unsafe fn h_list_remove_at(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));let index=word_i32(vm.reg(op.c));let removed=match vm.heap.get_mut(list){Some(HeapObject::List(values))=>if let Some(i)=normalize_index(index,values.len()){Some(values.remove(i))}else{vm.fail_at(op,"list removal index out of bounds");None},_=>{vm.fail_at(op,"ListRemoveAt operand is not list");None}};if let Some(value)=removed{vm.heap.refresh_jit_view(list);vm.set_reg(op.a,value.word());vm.advance(op)}}

unsafe fn h_alloc_map(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::Map(WrenMap::default()));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_map_len(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map=Value::from_word(vm.reg(op.b));match vm.heap.get(map){Some(HeapObject::Map(map))=>{vm.set_reg(op.a,i32_word(map.entries.len() as i32));vm.advance(op)},_=>vm.fail_at(op,"MapLen operand is not map")}}
unsafe fn h_map_get(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map=Value::from_word(vm.reg(op.b));let key_value=Value::from_word(vm.reg(op.c));let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};match vm.heap.get(map){Some(HeapObject::Map(map))=>{let value=map.entries.get(&key).copied().unwrap_or(Value::NULL);vm.set_reg(op.a,value.word());vm.advance(op)},_=>vm.fail_at(op,"MapGet operand is not map")}}
unsafe fn h_map_set(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr;let map_value=Value::from_word(vm.reg(op.a));let key_value=Value::from_word(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));
    let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};
    vm.mark_gc_point(op);vm.maybe_collect();vm.heap.remember_if_reference(map_value,key_value);vm.heap.remember_if_reference(map_value,value);
    let resized=match vm.heap.get_mut(map_value){
        Some(HeapObject::Map(map))=>{let old=map.entries.capacity();map.entries.insert(key,value);(map.entries.capacity()!=old).then_some(map.entries.capacity())},
        _=>{vm.fail_at(op,"MapSet operand is not map");return;}
    };
    if let Some(capacity)=resized{vm.heap.set_external_bytes(map_value,capacity*std::mem::size_of::<(MapKey,Value)>());}
    vm.heap.refresh_jit_view(map_value);
    vm.advance(op)
}
unsafe fn h_map_remove(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map_value=Value::from_word(vm.reg(op.b));let key_value=Value::from_word(vm.reg(op.c));let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};let value=match vm.heap.get_mut(map_value){Some(HeapObject::Map(map))=>Some(map.entries.remove(&key).unwrap_or(Value::NULL)),_=>{vm.fail_at(op,"MapRemove operand is not map");None}};if let Some(v)=value{vm.heap.refresh_jit_view(map_value);vm.set_reg(op.a,v.word());vm.advance(op)}}

#[inline]
fn decode_wren_code_point(bytes: &[u8], index: usize) -> (u32, usize) {
    let first = bytes[index];
    if first < 0x80 { return (first as u32, 1); }
    let (needed, mut code, min) = match first {
        0xC2..=0xDF => (2usize, (first & 0x1f) as u32, 0x80u32),
        0xE0..=0xEF => (3usize, (first & 0x0f) as u32, 0x800u32),
        0xF0..=0xF4 => (4usize, (first & 0x07) as u32, 0x10000u32),
        _ => return (first as u32, 1),
    };
    if index + needed > bytes.len() { return (first as u32, 1); }
    for offset in 1..needed {
        let byte = bytes[index + offset];
        if (byte & 0xC0) != 0x80 { return (first as u32, 1); }
        code = (code << 6) | (byte & 0x3f) as u32;
    }
    if code < min || code > 0x10ffff || (0xd800..=0xdfff).contains(&code) {
        return (first as u32, 1);
    }
    (code, needed)
}

#[inline]
fn wren_code_point_count(bytes: &[u8]) -> usize {
    if bytes.is_ascii() { return bytes.len(); }
    let mut index = 0usize;
    let mut count = 0usize;
    while index < bytes.len() {
        let (_, width) = decode_wren_code_point(bytes, index);
        index += width;
        count += 1;
    }
    count
}

#[inline]
fn normalize_byte_index(index: i32, len: usize, allow_end: bool) -> Option<usize> {
    let index = if index < 0 { len as i64 + index as i64 } else { index as i64 };
    let upper = if allow_end { len as i64 } else { len.saturating_sub(1) as i64 };
    if index >= 0 && index <= upper && (allow_end || index < len as i64) { Some(index as usize) } else { None }
}

unsafe fn h_string_len(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let string=Value::from_word(vm.reg(op.b));match vm.heap.get(string){Some(HeapObject::String(s))=>{vm.set_reg(op.a,i32_word(wren_code_point_count(s) as i32));vm.advance(op)},_=>vm.fail_at(op,"StringLen operand is not string")}}
unsafe fn h_string_concat(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let left=Value::from_word(vm.reg(op.b));let right=Value::from_word(vm.reg(op.c));let (a,b)=match(vm.heap.get(left),vm.heap.get(right)){(Some(HeapObject::String(a)),Some(HeapObject::String(b)))=>(a.clone(),b.clone()),_=>{vm.fail_at(op,"StringConcat operands must be strings");return;}};vm.mark_gc_point(op);vm.maybe_collect();let mut out=Vec::with_capacity(a.len()+b.len());out.extend_from_slice(&a);out.extend_from_slice(&b);let value=vm.heap.alloc(HeapObject::String(Arc::<[u8]>::from(out)));vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_string_slice(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let string=Value::from_word(vm.reg(op.b));let start=word_i32(vm.reg(op.c));let count=word_i32(vm.reg(op.x as u8));if count<0{vm.fail_at(op,"negative string slice length");return;}let source=match vm.heap.get(string){Some(HeapObject::String(s))=>s.clone(),_=>{vm.fail_at(op,"StringSlice operand is not string");return;}};let Some(begin)=normalize_byte_index(start,source.len(),true)else{vm.fail_at(op,"string byte offset out of bounds");return;};let Some(end)=begin.checked_add(count as usize).filter(|end|*end<=source.len())else{vm.fail_at(op,"string slice out of bounds");return;};vm.mark_gc_point(op);vm.maybe_collect();let value=vm.heap.alloc(HeapObject::String(Arc::<[u8]>::from(&source[begin..end])));vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_string_code_point_at(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let string=Value::from_word(vm.reg(op.b));let index=word_i32(vm.reg(op.c));let code=match vm.heap.get(string){Some(HeapObject::String(bytes))=>normalize_byte_index(index,bytes.len(),false).map(|i|decode_wren_code_point(bytes,i).0 as i32),_=>None};match code{Some(c)=>{vm.set_reg(op.a,i32_word(c));vm.advance(op)},None=>vm.fail_at(op,"string byte offset out of bounds")}}
unsafe fn h_make_range(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let from=Value::from_word(vm.reg(op.b)).as_num();let to=Value::from_word(vm.reg(op.c)).as_num();let(Some(from),Some(to))=(from,to)else{vm.fail_at(op,"range endpoints must be numbers");return;};vm.mark_gc_point(op);vm.maybe_collect();let value=vm.heap.alloc(HeapObject::Range{from,to,inclusive:op.x!=0});vm.set_reg(op.a,value.word());vm.advance(op)}

macro_rules! int_bin_handler {($name:ident,$expr:expr)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_i32(vm.reg(op.b));let b=word_i32(vm.reg(op.c));let r: i32=($expr)(a,b);vm.set_reg(op.a,i32_word(r));vm.advance(op)}}}
unsafe fn h_ineg(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_i32(vm.reg(op.b));vm.set_reg(op.a,i32_word(a.wrapping_neg()));vm.advance(op)}
unsafe fn h_ibitnot(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_i32(vm.reg(op.b));vm.set_reg(op.a,i32_word(!a));vm.advance(op)}
int_bin_handler!(h_iadd,|a:i32,b:i32|a.wrapping_add(b));
int_bin_handler!(h_isub,|a:i32,b:i32|a.wrapping_sub(b));
int_bin_handler!(h_imul,|a:i32,b:i32|a.wrapping_mul(b));
unsafe fn h_idiv(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_i32(vm.reg(op.b));let b=word_i32(vm.reg(op.c));if b==0{vm.fail_at(op,"integer division by zero")}else if a==i32::MIN&&b==-1{vm.set_reg(op.a,i32_word(i32::MIN));vm.advance(op)}else{vm.set_reg(op.a,i32_word(a/b));vm.advance(op)}}
unsafe fn h_irem(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_i32(vm.reg(op.b));let b=word_i32(vm.reg(op.c));if b==0{vm.fail_at(op,"integer remainder by zero")}else{vm.set_reg(op.a,i32_word(a.wrapping_rem(b)));vm.advance(op)}}
int_bin_handler!(h_iand,|a:i32,b:i32|a&b); int_bin_handler!(h_ior,|a:i32,b:i32|a|b); int_bin_handler!(h_ixor,|a:i32,b:i32|a^b);
int_bin_handler!(h_ishl,|a:i32,b:i32|a.wrapping_shl((b as u32)&31));
int_bin_handler!(h_ishru,|a:i32,b:i32|((a as u32)>>((b as u32)&31)) as i32);
int_bin_handler!(h_isar,|a:i32,b:i32|a>>((b as u32)&31));

macro_rules! f_bin_handler {($name:ident,$op:tt)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let a=word_f64(vm.reg(i.b));let b=word_f64(vm.reg(i.c));vm.set_reg(i.a,f64_word(a $op b));vm.advance(i)}}}
unsafe fn h_fneg(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_f64(vm.reg(op.b));vm.set_reg(op.a,f64_word(-a));vm.advance(op)}
f_bin_handler!(h_fadd,+); f_bin_handler!(h_fsub,-); f_bin_handler!(h_fmul,*); f_bin_handler!(h_fdiv,/);
unsafe fn h_frem(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let a=word_f64(vm.reg(op.b));let b=word_f64(vm.reg(op.c));vm.set_reg(op.a,f64_word(a%b));vm.advance(op)}

macro_rules! i_cmp_handler {($name:ident,$op:tt)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let a=word_i32(vm.reg(i.b));let b=word_i32(vm.reg(i.c));vm.set_reg(i.a,Value::from_bool(a $op b).word());vm.advance(i)}}}
i_cmp_handler!(h_icmp_eq,==);i_cmp_handler!(h_icmp_ne,!=);i_cmp_handler!(h_icmp_lt,<);i_cmp_handler!(h_icmp_le,<=);i_cmp_handler!(h_icmp_gt,>);i_cmp_handler!(h_icmp_ge,>=);
macro_rules! f_cmp_handler {($name:ident,$op:tt)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let a=word_f64(vm.reg(i.b));let b=word_f64(vm.reg(i.c));vm.set_reg(i.a,Value::from_bool(a $op b).word());vm.advance(i)}}}
f_cmp_handler!(h_fcmp_eq,==);f_cmp_handler!(h_fcmp_ne,!=);f_cmp_handler!(h_fcmp_lt,<);f_cmp_handler!(h_fcmp_le,<=);f_cmp_handler!(h_fcmp_gt,>);f_cmp_handler!(h_fcmp_ge,>=);
unsafe fn h_ref_eq(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;vm.set_reg(op.a,Value::from_bool(vm.reg(op.b).0==vm.reg(op.c).0).word());vm.advance(op)}
unsafe fn h_num_to_i32(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;match Value::from_word(vm.reg(op.b)).as_num(){Some(v)if v.is_finite()&&v.fract()==0.0&&v>=i32::MIN as f64&&v<=i32::MAX as f64=>{vm.set_reg(op.a,i32_word(v as i32));vm.advance(op)},_=>vm.fail_at(op,"NumToInt32 conversion failed")}}
unsafe fn h_num_to_f64(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;match Value::from_word(vm.reg(op.b)).as_num(){Some(v)=>{vm.set_reg(op.a,f64_word(v));vm.advance(op)},None=>vm.fail_at(op,"NumToF64 operand is not Num")}}
unsafe fn h_i32_to_num(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let v=word_i32(vm.reg(op.b));vm.set_reg(op.a,Value::from_num(v as f64).word());vm.advance(op)}
unsafe fn h_f64_to_num(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let v=word_f64(vm.reg(op.b));vm.set_reg(op.a,Value::from_num(v).word());vm.advance(op)}

unsafe fn h_make_fiber(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let closure=Value::from_word(vm.reg(op.b));vm.mark_gc_point(op);vm.maybe_collect();match vm.allocate_fiber(closure){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}}
unsafe fn h_fiber_call(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let fiber=Value::from_word(vm.reg(op.b));let arg=Value::from_word(vm.reg(op.c));vm.activate_fiber(op,fiber,arg,true,false)}
unsafe fn h_fiber_try(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let fiber=Value::from_word(vm.reg(op.b));let arg=Value::from_word(vm.reg(op.c));vm.activate_fiber(op,fiber,arg,true,true)}
unsafe fn h_fiber_transfer(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let fiber=Value::from_word(vm.reg(op.b));let arg=Value::from_word(vm.reg(op.c));vm.activate_fiber(op,fiber,arg,false,false)}
unsafe fn h_fiber_transfer_error(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let fiber=Value::from_word(vm.reg(op.b));let error=Value::from_word(vm.reg(op.c));vm.transfer_error_to_fiber(op,fiber,error)}
unsafe fn h_fiber_yield(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let value=Value::from_word(vm.reg(op.b));vm.yield_current_fiber(op,value,Some(Reg(op.a)))}
unsafe fn h_fiber_suspend(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;vm.suspend_vm(op,Reg(op.a))}
unsafe fn h_fiber_abort(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let error=Value::from_word(vm.reg(op.a));vm.abort_current_fiber(op,error)}

unsafe fn h_write_barrier(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let owner=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));vm.heap.remember_if_reference(owner,value);vm.advance(op)}
unsafe fn h_read_barrier(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let value=vm.reg(op.b);vm.set_reg(op.a,value);vm.advance(op)}
unsafe fn h_profile_block(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let f=vm.current_frame().function as usize;if let Some(counter)=vm.profile_counters[f].get_mut(op.x as usize){*counter=counter.saturating_add(1);}vm.advance(op)}
unsafe fn h_profile_numeric(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let f=vm.current_frame().function as usize;if let Some(counter)=vm.profile_counters[f].get_mut(op.x as usize){*counter=counter.saturating_add(1);}vm.advance(op)}
unsafe fn h_breakpoint(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let(f,pc)=vm.current_location(op);match vm.host.breakpoint(f,pc){Ok(())=>vm.advance(op),Err(e)=>vm.fail_at(op,e.message)}}
unsafe fn h_await(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let promise=Value::from_word(vm.reg(op.b));vm.await_promise(op,promise,Reg(op.a))}

#[inline(always)]
unsafe fn fused_branch(vm:&mut Interpreter,op:&PreparedOp,condition:bool){let take=if op.flags!=0{condition}else{!condition};let target=if take{op.y}else{op.next};vm.branch_to(op,target)}
macro_rules! fused_i_cmp {($name:ident,$op:tt)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let a=word_i32(vm.reg(i.b));let b=word_i32(vm.reg(i.c));let condition=a $op b;vm.set_reg(i.a,Value::from_bool(condition).word());fused_branch(vm,i,condition)}}}
fused_i_cmp!(h_fused_icmp_eq_branch,==);fused_i_cmp!(h_fused_icmp_ne_branch,!=);fused_i_cmp!(h_fused_icmp_lt_branch,<);fused_i_cmp!(h_fused_icmp_le_branch,<=);fused_i_cmp!(h_fused_icmp_gt_branch,>);fused_i_cmp!(h_fused_icmp_ge_branch,>=);
macro_rules! fused_f_cmp {($name:ident,$op:tt)=>{unsafe fn $name(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let a=word_f64(vm.reg(i.b));let b=word_f64(vm.reg(i.c));let condition=a $op b;vm.set_reg(i.a,Value::from_bool(condition).word());fused_branch(vm,i,condition)}}}
fused_f_cmp!(h_fused_fcmp_eq_branch,==);fused_f_cmp!(h_fused_fcmp_ne_branch,!=);fused_f_cmp!(h_fused_fcmp_lt_branch,<);fused_f_cmp!(h_fused_fcmp_le_branch,<=);fused_f_cmp!(h_fused_fcmp_gt_branch,>);fused_f_cmp!(h_fused_fcmp_ge_branch,>=);
unsafe fn h_fused_ref_eq_branch(vm:&mut Interpreter,ptr:*const PreparedOp){let i=&*ptr;let condition=vm.reg(i.b).0==vm.reg(i.c).0;vm.set_reg(i.a,Value::from_bool(condition).word());fused_branch(vm,i,condition)}

// =============================================================================
// Introspection / diagnostics
// =============================================================================

impl Interpreter<'_> {
    pub fn profile_counter(&self, function: FuncId, counter: u32) -> Option<u64> {
        let function = *self.prepared.function_by_id.get(&function)? as usize;
        self.profile_counters.get(function)?.get(counter as usize).copied()
    }

    pub fn heap_live_objects(&self) -> usize { handles().live.load(Ordering::Relaxed) }

    pub fn class_value(&self, class: ClassId) -> Option<Value> {
        let slot = *self.prepared.class_by_id.get(&class)?;
        Some(self.classes[slot as usize].class_value)
    }

    pub fn module_value(&self, module: ModuleId) -> Option<Value> {
        let slot = *self.prepared.module_by_id.get(&module)?;
        Some(self.modules[slot as usize].module_value)
    }

    pub fn value_to_string(&self, value: Value) -> String {
        if value.is_null() { return "null".into(); }
        if let Some(boolean) = value.as_bool() { return boolean.to_string(); }
        if let Some(number) = value.as_num() { return number.to_string(); }
        match self.heap.get(value) {
            Some(HeapObject::Random { .. }) => "<random>".into(),
            Some(HeapObject::String(string)) => String::from_utf8_lossy(string).into_owned(),
            Some(HeapObject::List(values)) => format!("<list:{}>", values.len()),
            Some(HeapObject::Map(map)) => format!("<map:{}>", map.entries.len()),
            Some(HeapObject::MapEntry { .. }) => "<map entry>".into(),
            Some(HeapObject::MapKeys { .. }) => "<map keys>".into(),
            Some(HeapObject::Range { from, to, inclusive }) => format!("{}{}{}", from, if *inclusive { ".." } else { "..." }, to),
            Some(HeapObject::Closure { function_index, .. }) => format!("<closure {:?}>", self.prepared.functions[*function_index as usize].id),
            Some(HeapObject::ClassObject { class_slot }) => self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("<class>").to_owned(),
            Some(HeapObject::BuiltinClass { builtin }) => builtin.name().to_owned(),
            Some(HeapObject::Instance { class_slot, .. }) => format!("<{} instance>", self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("class")),
            Some(HeapObject::Foreign { class_slot, .. }) => format!("<foreign {}>", self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("class")),
            Some(HeapObject::Promise { state }) => match unsafe { state.get() } {
                PromiseState::Pending { .. } => "<promise:pending>".into(),
                PromiseState::Fulfilled(_) => "<promise:fulfilled>".into(),
                PromiseState::Rejected(_) => "<promise:rejected>".into(),
            },
            Some(HeapObject::AsyncTaskObject { task, .. }) => format!("<async-task#{task}>"),
            Some(HeapObject::FiberObject { fiber, .. }) => format!("<fiber:{fiber}>") ,
            Some(HeapObject::ModuleObject { module_slot }) => format!("<module:{}>", self.program.metadata.string(self.modules[*module_slot as usize].name).unwrap_or("module")),
            Some(HeapObject::Cell(_)) => "<cell>".into(),
            None => "<invalid value>".into(),
        }
    }
}

// =============================================================================
// Trace runtime adapter. Prepared op indexes and bytecode word PCs are distinct;
// all transitions between the two use the validated preparation tables.
// =============================================================================

impl TraceRuntime for Interpreter<'_> {
    type RuntimeError = RuntimeError;

    fn trace_program(&self) -> &BytecodeProgram { self.program }

    fn trace_standard_numeric_method(&self, function: FuncId, cache: u32) -> bool {
        let Some(function) = self.program.function(function) else { return false; };
        let Some(cache) = function.chunk.inline_caches.get(cache as usize) else { return false; };
        let Some(CallTarget::Primitive(primitive)) = self.builtin_methods[BuiltinClass::Num as usize][cache.method.0 as usize] else { return false; };
        let Some(signature) = self.program.metadata.method(cache.method) else { return false; };
        let name = self.program.metadata.string(signature.name).unwrap_or("");
        matches!(signature.kind, crate::wren_ir::MethodKind::Method | crate::wren_ir::MethodKind::Getter)
            && self.host.standard_numeric_method(primitive, name, signature.arity)
    }

    fn trace_current_location(&self) -> TraceLocation {
        if self.hot_frame.is_null() {
            return TraceLocation::new(FuncId(u32::MAX), BytecodePc(u32::MAX));
        }
        let frame = self.current_frame();
        let function = &self.prepared.functions[frame.function as usize];
        let pc = function.original_ops.get(self.hot_pc as usize).map_or(u32::MAX, |op| op.raw_pc);
        TraceLocation::new(function.id, BytecodePc(pc))
    }

    fn trace_read_slot(&self, slot: FrameSlot) -> Result<RegisterWord, RuntimeError> {
        let frame = self.current_frame();
        let banks = self.current_banks();
        let value = match slot {
            FrameSlot::General(reg) if reg.0 as u16 >= frame.reg_count => None,
            FrameSlot::General(reg) => Some(unsafe { &*self.hot_registers.add(reg.index()) }),
            FrameSlot::Argument(reg) => banks.arguments.get(reg.0 as usize),
            FrameSlot::Return(reg) => banks.returns.get(reg.0 as usize),
            FrameSlot::Special(reg) => banks.special.get(reg.index()),
        };
        value.copied().ok_or_else(|| RuntimeError::new("trace slot out of bounds"))
    }

    fn trace_write_slot(&mut self, slot: FrameSlot, value: RegisterWord) -> Result<(), RuntimeError> {
        let reg_bank = self.hot_registers;
        let reg_count = self.current_frame().reg_count;
        let banks = self.current_banks_mut();
        let target = match slot {
            FrameSlot::General(reg) if reg.0 as u16 >= reg_count => None,
            FrameSlot::General(reg) => Some(unsafe { &mut *reg_bank.add(reg.index()) }),
            FrameSlot::Argument(reg) => banks.arguments.get_mut(reg.0 as usize),
            FrameSlot::Return(reg) => banks.returns.get_mut(reg.0 as usize),
            FrameSlot::Special(reg) => banks.special.get_mut(reg.index()),
        }.ok_or_else(|| RuntimeError::new("trace slot out of bounds"))?;
        *target = value; Ok(())
    }

    fn trace_set_location(&mut self, location: TraceLocation) -> Result<(), RuntimeError> {
        let prepared = self.current_prepared();
        if prepared.id != location.function { return Err(RuntimeError::new("trace function mismatch")); }
        let pc = prepared.word_to_op.get(location.pc.0 as usize).copied()
            .filter(|pc| *pc != NO_INDEX && (*pc as usize) < prepared.original_ops.len())
            .ok_or_else(|| RuntimeError::new("invalid trace resume PC"))?;
        unsafe { self.set_pc(pc); } Ok(())
    }

    fn trace_execute_unfused_step(&mut self) -> Result<TraceStep, RuntimeError> {
        let before = self.trace_current_location();
        let fiber = self.current_fiber;
        let depth = self.fiber(fiber).frames.len();
        let function = self.current_frame().function;
        let mut op = self.current_prepared().original_ops[self.hot_pc as usize];
        let opcode = self.program.functions[function as usize].chunk.code[op.raw_pc as usize].opcode().unwrap();
        if matches!(opcode, Opcode::Switch | Opcode::LoadModuleVar | Opcode::StoreModuleVar) {
            // These handlers consume patched pointers, while original_ops keeps
            // logical indices. Preserve the unfused continuation and copy only
            // the pointer operands from the executable instruction.
            let prepared_op = self.current_prepared().ops[self.hot_pc as usize];
            op.x = prepared_op.x;
            op.y = prepared_op.y;
        }
        unsafe { handler_for(opcode)(self, &op); }
        let running = self.run_state == RunState::Running;
        let after = self.trace_current_location();
        let frame_changed = self.current_fiber != fiber || self.fiber(self.current_fiber).frames.len() != depth
            || self.hot_frame.is_null() || self.current_frame().function != function;
        Ok(TraceStep { before, after, frame_changed, running })
    }

    fn trace_activation_inputs(&mut self) -> Result<TraceActivationInputs, RuntimeError> {
        self.flush_hot_pc();
        let general = self.hot_registers;
        let runtime = self as *mut Self as *mut std::ffi::c_void;
        let banks = self.current_banks_mut();
        Ok(TraceActivationInputs { general, arguments: banks.arguments.as_mut_ptr(),
            returns: banks.returns.as_mut_ptr(), special: banks.special.as_mut_ptr(), runtime,
            execute: execute_trace_runtime, heap_chunks: handles().jit_chunks(),
            heap_write_barrier: trace_heap_write_barrier })
    }

    fn trace_materialize_constant(&self, constant: TraceConstant) -> Result<RegisterWord, RuntimeError> {
        Ok(match constant {
            TraceConstant::Null => Value::NULL.word(),
            TraceConstant::Bool(v) => Value::from_bool(v).word(),
            TraceConstant::NumBits(v) | TraceConstant::F64Bits(v) => RegisterWord(v),
            TraceConstant::Int32(v) => i32_word(v),
            TraceConstant::String(id) => self.strings.get(id.0 as usize).ok_or_else(|| RuntimeError::new("unknown string"))?.word(),
            TraceConstant::BuiltinClass(id) => self.builtin_value(id).word(),
        })
    }

    fn trace_unsupported_constant_error(&self, _: TraceConstant) -> RuntimeError {
        RuntimeError::new("unsupported trace constant")
    }
}

/// StickyImmix post barrier used by native heap stores. Generated code calls this
/// only after all kind/bounds guards have succeeded and only when the stored value
/// is heap-tagged. It cannot allocate or safepoint.
unsafe extern "C" fn trace_heap_write_barrier(runtime: *mut std::ffi::c_void, object: usize) {
    if object == 0 { return; }
    let vm = unsafe { &mut *(runtime as *mut Interpreter<'_>) };
    let object = unsafe { GcObject::from_address(object) };
    unsafe { vm.heap.mutator.probable_write(object) };
}

/// The native caller has committed all dirty registers. A helper may collect,
/// throw, suspend or replace the frame: only a matching continuation may resume.
unsafe extern "C" fn execute_trace_runtime(activation: *mut TraceActivation, function: u32, pc: u32, next: u32) -> u32 {
    // Rust unwinding must never cross generated code (it has no unwind metadata).
    let activation = unsafe { &mut *activation };
    let vm = unsafe { &mut *(activation.runtime as *mut Interpreter<'_>) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        vm.trace_set_location(TraceLocation::new(FuncId(function), BytecodePc(pc)))?;
        vm.trace_execute_unfused_step()
    }));
    match result {
        Ok(Ok(step)) if step.running && !step.frame_changed
            && step.after == TraceLocation::new(FuncId(function), BytecodePc(next)) => {
                let inputs = vm.trace_activation_inputs().unwrap();
                activation.general = inputs.general;
                activation.arguments = inputs.arguments;
                activation.returns = inputs.returns;
                activation.special = inputs.special;
                0
            },
        Ok(Ok(_)) => 1,
        Ok(Err(error)) => { vm.fail(error); 1 },
        Err(_) => { vm.fail(RuntimeError::new("panic in JIT runtime helper")); 1 },
    }
}

// =============================================================================
// Tests that require only hand-built bytecode.
// =============================================================================

#[cfg(test)]
#[path = "wren_jit_tests.rs"]
mod jit_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{
        Assembler, BytecodeFunction, SafepointEntry, SafepointKind,
    };
    use crate::wren_builtin::{self, BuiltinHost, BuiltinManifest};
    use crate::wren_pipeline::compile_program;
    use crate::wren_front::parse;
    use crate::wren_ir::{FunctionKind, ModuleInfo, Program};
    use crate::wren_lowerer::lower_module;

    fn simple_program(chunk: crate::wren_bytecode::BytecodeChunk, gprs: u16) -> BytecodeProgram {
        let mut program = BytecodeProgram::new();
        program.metadata.modules.push(ModuleInfo { id: ModuleId(0), name: StringId(0), variable_count: 0, builtin_bindings: Vec::new() });
        program.metadata.strings.push("main".to_owned());
        program.push_function(BytecodeFunction {
            id: FuncId(0), module: ModuleId(0), kind: FunctionKind::ModuleBody,
            field_owner: None,
            name: StringId(0), is_async: false, arity: 0, capture_count: 0,
            frame: FrameLayout { general_register_count: gprs, parameters: vec![], captures: vec![], return_kind: RegisterKind::Value },
            chunk,
        }).unwrap();
        program.rebuild_indexes().unwrap();
        program
    }

    fn compile_wren_module(source: &str) -> (BytecodeProgram, FuncId, BuiltinManifest) {
        let parsed = parse("jit_test.wren", source);
        assert!(!parsed.has_errors(), "parse diagnostics: {:#?}", parsed.diagnostics);

        // Use the exact production pipeline from main.rs. The previous JIT source
        // tests lowered into an empty Program and then ran with NoHost, so ordinary
        // Num operators were never backed by the stock primitive manifest. Those
        // tests therefore did not exercise the same Invoke -> Num specialization
        // path as real .wren files.
        let mut hir = Program::new();
        let manifest = wren_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(
            &mut hir,
            "jit_test.wren",
            source,
            "jit_test",
            &parsed.module,
        );
        assert!(!lowered.has_errors(), "lowering diagnostics: {:#?}", lowered.diagnostics);
        let entry = lowered
            .lowered
            .as_ref()
            .expect("module should lower successfully")
            .entry;
        let bytecode = compile_program(&mut hir).expect("compilation pipeline should succeed");
        (bytecode, entry, manifest)
    }

    fn source_vm<'a>(
        program: &'a BytecodeProgram,
        manifest: &BuiltinManifest,
    ) -> Interpreter<'a> {
        let host = Box::new(BuiltinHost::new(manifest.clone()));
        let mut vm = Interpreter::with_host(program, host).expect("VM should initialize");
        wren_builtin::install_vm_builtins(&mut vm, manifest);
        vm
    }

    fn compile_local_wren(source: &str) -> (BytecodeProgram, FuncId, BuiltinManifest) {
        // Numeric traces operate on locals. Module variables currently use
        // runtime opcodes, so putting these fixtures at module scope never
        // exercises the native numeric tier used by scripts/jit.wren.
        compile_wren_module(&format!(
            "class JitTest {{\n  static run() {{\n{source}\n  }}\n}}\nvar result = JitTest.run()\n"
        ))
    }

    fn source_result(vm: &Interpreter<'_>) -> f64 {
        vm.modules.last().unwrap().variables.last().unwrap().as_num().unwrap()
    }

    include!("wren_class_tests.rs");

    #[test]
    fn jit_numeric_math_getters_and_remainder_stay_in_native_loops() {
        for expression in [
            "x.abs", "x.floor", "x.ceil", "x.sqrt", "x.round",
            "x.sin", "x.cos", "x.tan", "x.log", "x.exp", "x % 3.25",
            "x.isNan", "x.isInfinity", "x.isInteger",
            "i.isInteger", "(i - 256).isInteger",
            "(x / 0).isInfinity", "(0 / 0 + x).isNan",
            "(x / 0).isInteger", "(0 / 0 + x).isInteger",
        ] {
            let boolean = expression.contains(".is");
            let accumulation = if boolean {
                "if (value) { total = total + 1 }"
            } else {
                "total = total + value"
            };
            let source = format!(r#"
                var i = 0
                var total = 0
                while (i < 512) {{
                    var x = (i - 16.5) / 8
                    var value = {expression}
                    {accumulation}
                    i = i + 1
                }}
                return total
            "#);
            let (program, entry, manifest) = compile_local_wren(&source);
            let mut baseline = source_vm(&program, &manifest);
            baseline.set_jit_options(None);
            baseline.run_module(entry).unwrap();
            let expected = source_result(&baseline);
            for optimize_trace_ir in [false, true] {
                let mut native = source_vm(&program, &manifest);
                native.set_jit_options(Some(TracingJitOptions {
                    hot_loop_threshold: 2,
                    optimize_trace_ir,
                    no_runtime_calls_in_native_trace: true,
                    ..Default::default()
                }));
                native.run_module(entry).unwrap();
                let actual = source_result(&native);
                assert!(if expected.is_nan() { actual.is_nan() }
                    else { actual.to_bits() == expected.to_bits() },
                    "{expression}, optimize={optimize_trace_ir}: {actual:?} != {expected:?}");
                let stats = native.jit_stats().unwrap();
                assert!(stats.native_iterations > 400,
                    "{expression}, optimize={optimize_trace_ir}: {stats:?}");
            }
        }
    }

    #[test]
    fn source_random_int_respects_bounds_and_survives_gc() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var random = Random.new()
            var other = Random.new()
            var seen = [false, false, false, false]
            var i = 0
            while (i < 2000) {
                var value = random.int(3, 10)
                if (!value.isInteger || value < 3 || value >= 10) return -1
                var small = random.int(4)
                if (!small.isInteger || small < 0 || small >= 4) return -2
                seen[small] = true
                var negative = other.int(-10, -3)
                if (negative < -10 || negative >= -3) return -3
                if (random.int(-1, 0) != -1) return -4
                if (random.int(1) != 0) return -5
                var wide = random.int(-9007199254740991, 9007199254740991)
                if (!wide.isInteger || wide < -9007199254740991 || wide >= 9007199254740991) return -6
                i = i + 1
            }
            System.gc()
            if (random.int(1) != 0) return -7
            for (found in seen) {
                if (!found) return -8
            }
            return 42
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 42.0);
        }
    }

    #[test]
    fn source_random_rejects_invalid_bounds() {
        for (expression, expected) in [
            ("random.int(0)", "upper bound must be greater"),
            ("random.int(-4)", "upper bound must be greater"),
            ("random.int(3, 3)", "upper bound must be greater"),
            ("random.int(10, 3)", "upper bound must be greater"),
            ("random.int(1.5)", "safe finite integers"),
            ("random.int(Num.infinity)", "safe finite integers"),
            ("random.int(Num.nan)", "safe finite integers"),
            ("random.int(9007199254740992)", "safe finite integers"),
            ("random.int(null)", "upper bound must be a Num"),
            ("random.int(\"4\", 5)", "lower bound must be a Num"),
        ] {
            let (program, entry, manifest) = compile_wren_module(
                &format!("var random = Random.new()\n{expression}\n"));
            let mut vm = source_vm(&program, &manifest);
            let error = vm.run_module(entry).err().expect(expression);
            assert!(error.message.contains(expected), "{expression}: {error:?}");
        }
    }

    #[test]
    fn source_collection_getters_require_no_parentheses() {
        for (expression, expected) in [
            ("\"string\".count", "6"),
            ("(1..10).count", "10"),
            ("(1...10).count", "9"),
            ("(10..1).count", "10"),
            ("(10...1).count", "9"),
            ("(2..2).count", "1"),
            ("(2...2).count", "0"),
            ("(0.5..2.75).count", "3"),
            ("(0.5...2.75).count", "3"),
            ("(1..10).min", "1"),
            ("[1, 2, 3].isEmpty", "false"),
            ("[].isEmpty", "true"),
            ("\"\".isEmpty", "true"),
            ("\"a\".isEmpty", "false"),
            ("{}.isEmpty", "true"),
            ("{\"a\": null}.isEmpty", "false"),
            ("{}.keys.isEmpty", "true"),
        ] {
            let (program, entry, manifest) = compile_wren_module(&format!("var result = {expression}\n"));
            let mut vm = source_vm(&program, &manifest);
            vm.run_module(entry).unwrap();
            let value = *vm.modules.last().unwrap().variables.last().unwrap();
            let context = HostContext { heap: &mut vm.heap };
            assert_eq!(context.format_value(value), expected, "{expression}");
        }
        let (program, entry, manifest) = compile_wren_module("var result = 1.23.sin\n");
        let mut vm = source_vm(&program, &manifest);
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 1.23f64.sin());
    }

    #[test]
    fn source_map_entries_and_keys_iterate_with_matching_values() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var birds = {"Georgia": "Brown Thrasher", "Idaho": "Mountain Bluebird"}
            var entries = 0
            for (bird in birds) {
                if (birds[bird.key] != bird.value) return -1
                var sentence = "The state bird of %(bird.key) is %(bird.value)"
                if (sentence.count == 0) return -2
                entries = entries + 1
            }
            var keys = 0
            for (state in birds.keys) {
                if (!birds.containsKey(state)) return -3
                var sentence = "The state bird of %(state) is " + birds[state]
                if (sentence.count == 0) return -4
                keys = keys + 1
            }
            for (entry in {}) return -5
            for (key in {}.keys) return -6
            return entries * 10 + keys
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 22.0);
        }
    }

    #[test]
    fn source_map_keys_are_live_and_entries_survive_collection() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var map = {"Georgia": [1], "Idaho": [2]}
            var keys = map.keys
            var saved = []
            for (entry in map) saved.add(entry)
            map.clear()
            if (keys.count != 0) return -1
            map["Maine"] = [3]
            System.gc()
            var visits = 0
            for (key in keys) {
                if (key != "Maine") return -2
                visits = visits + 1
            }
            var sum = 0
            for (entry in saved) {
                System.gc()
                if (entry.key != "Georgia" && entry.key != "Idaho") return -3
                sum = sum + entry.value[0]
            }
            var detached = {"kept": [4]}.keys
            System.gc()
            for (key in detached) {
                if (key != "kept") return -4
                sum = sum + 4
            }
            return sum * 10 + visits
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 71.0);
        }
    }

    #[test]
    fn source_map_iteration_supports_non_string_keys_and_invalid_iterators() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var object = [1]
            var map = {null: 1, true: 2, 3: 4}
            map[object] = 8
            var sum = 0
            for (entry in map) {
                if (map[entry.key] != entry.value) return -1
                sum = sum + entry.value
            }
            for (key in map.keys) sum = sum + map[key]
            return sum
        "#);
        let mut vm = source_vm(&program, &manifest);
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 30.0);
        for expression in [
            r#"({"key": 1}).iteratorValue(5)"#,
            r#"({"key": 1}).keys.iteratorValue(-1)"#,
        ] {
            let (program, entry, manifest) = compile_wren_module(&format!("{expression}\n"));
            let mut vm = source_vm(&program, &manifest);
            let error = vm.run_module(entry).err().expect(expression);
            assert!(error.message.contains("map iterator out of range"), "{error:?}");
        }
    }

    #[test]
    fn source_map_remove_returns_the_stored_value() {
        for (source, expected) in [
            (r#"var capitals = {"Georgia": "Atlanta", "Idaho": "Boise"}
                var removed = capitals.remove("Georgia")
                return removed + ":" + capitals.count.toString + ":" + capitals.containsKey("Georgia").toString"#,
                "Atlanta:1:false"),
            (r#"var capitals = {"Georgia": null}
                var removed = capitals.remove("Georgia")
                return removed.toString + ":" + capitals.count.toString + ":" + capitals.containsKey("Georgia").toString"#,
                "null:0:false"),
            (r#"var capitals = {"Georgia": "Atlanta"}
                var removed = capitals.remove("Idaho")
                return removed.toString + ":" + capitals.count.toString"#, "null:1"),
            (r#"var capitals = {}
                return capitals.remove("Georgia").toString"#, "null"),
            (r#"var value = [1]
                var map = {"key": value}
                var removed = map.remove("key")
                removed.add(2)
                System.gc()
                return value.toString + ":" + map.count.toString"#, "[1, 2]:0"),
        ] {
            let (program, entry, manifest) = compile_local_wren(source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{source}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{source}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_map_clear_empties_aliases_and_allows_reuse() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var capitals = {"Georgia": "Atlanta", "Idaho": null, "Maine": [1, 2]}
            var alias = capitals
            var result = capitals.clear()
            if (result != null) return -1
            if (capitals.count != 0) return -2
            if (alias.count != 0) return -3
            if (capitals.containsKey("Georgia")) return -4
            if (capitals.containsKey("Idaho")) return -5
            if (capitals.containsKey("Maine")) return -6
            if (capitals["Georgia"] != null) return -7
            capitals.clear()
            System.gc()
            capitals["Georgia"] = "Atlanta"
            if (alias["Georgia"] != "Atlanta") return -8
            if (alias.remove("Georgia") != "Atlanta") return -9
            if (capitals.count != 0) return -10
            var empty = {}
            empty.clear()
            return empty.count + 42
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 42.0, "jit={jit_enabled}");
        }
    }

    #[test]
    fn source_list_remove_returns_first_match_or_null() {
        for (source, expected) in [
            (r#"var xs = ["a", "b", "c", "d"]
                var removed = xs.remove("a")
                return removed + xs.toString"#, "a[b, c, d]"),
            (r#"var xs = [1, 2, 1]
                var removed = xs.remove(1)
                return removed.toString + xs.toString"#, "1[2, 1]"),
            (r#"var xs = [1, 2, 3]
                var removed = xs.remove(3)
                return removed.toString + xs.toString"#, "3[1, 2]"),
            (r#"var xs = ["a"]
                var removed = xs.remove("not found")
                return removed.toString + xs.toString"#, "null[a]"),
            (r#"var xs = []
                return xs.remove(1).toString + xs.toString"#, "null[]"),
            (r#"var xs = [false, null, true]
                xs.remove(null)
                xs.remove(true)
                return xs.toString"#, "[false]"),
            (r#"var xs = [0, 1]
                xs.remove(-0)
                return xs.toString"#, "[1]"),
            (r#"var xs = [Num.nan, 1]
                xs.remove(Num.nan)
                return xs.count.toString"#, "2"),
            (r#"var xs = ["a" + "b", "tail"]
                xs.remove("ab")
                return xs.toString"#, "[tail]"),
            (r#"var item = [1]
                var xs = [[1], item]
                xs.remove(item)
                return xs.count.toString"#, "1"),
        ] {
            let (program, entry, manifest) = compile_local_wren(source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{source}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{source}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_list_remove_dispatches_user_equality() {
        let source = r#"
            class Match {
                construct new(value) { _value = value }
                ==(other) {
                    System.gc()
                    return _value == other
                }
            }
            var first = Match.new(1)
            var second = Match.new(2)
            var items = [first, second, Match.new(2)]
            var removed = items.remove(2)
            var result = items.count * 10
            if (removed == 2) result = result + 1
        "#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 21.0);
        }
    }

    #[test]
    fn source_list_and_string_concatenation() {
        for (expression, expected) in [
            ("[\"a\", \"b\", \"c\"] + [\"d\", \"e\", \"f\"]", "[a, b, c, d, e, f]"),
            ("[] + []", "[]"),
            ("[] + [1, null, true]", "[1, null, true]"),
            ("[1, null, true] + []", "[1, null, true]"),
            ("[1] + [2] + [3]", "[1, 2, 3]"),
            ("[[1]] + [[2]]", "[[1], [2]]"),
            ("\"hello\" + \" world\"", "hello world"),
            ("\"\" + \"\"", ""),
            ("\"\" + \"hello\"", "hello"),
            ("\"hello\" + \"\"", "hello"),
            ("\"a\" + \"b\" + \"c\"", "abc"),
            ("\"\\u00e1\" + \"\\u03bb\"", "\u{e1}\u{3bb}"),
        ] {
            let (program, entry, manifest) = compile_wren_module(&format!("var result = {expression}\n"));
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{expression}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{expression}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_list_concatenation_is_a_shallow_copy() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var left = [[1], 2]
            var right = [3]
            var combined = left + right
            combined[1] = 99
            combined[2] = 88
            combined[0][0] = 5
            combined.add(10)
            var twice = right + right
            twice[0] = 100
            var empty = []
            var copy = empty + empty
            copy.add(7)
            return left[0][0] + left[1] + right[0] + left.count + right.count + twice[1] + empty.count
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 16.0);
        }
    }

    #[test]
    fn source_concatenation_rejects_wrong_operand_types() {
        for (expression, expected) in [
            ("[] + 1", "List.+ expects List"),
            ("[] + null", "List.+ expects List"),
            ("[] + \"text\"", "List.+ expects List"),
            ("\"text\" + []", "String.+ expects String"),
            ("\"text\" + 1", "String.+ expects String"),
            ("\"text\" + null", "String.+ expects String"),
        ] {
            let (program, entry, manifest) = compile_wren_module(&format!("{expression}\n"));
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(None);
            let error = vm.run_module(entry).err().expect(expression);
            assert!(error.message.contains(expected), "{expression}: {error:?}");
        }
    }

    #[test]
    fn source_list_multidimensional_subscripts() {
        for (expression, expected) in [
            ("matrix[0, 1]", "2"),
            ("matrix[1, 2]", "6"),
            ("matrix[2, 0]", "7"),
            ("matrix[-1, -1]", "9"),
            ("matrix[-3, -2]", "2"),
            ("matrix[0, 1] == matrix[0][1]", "true"),
            ("[[1], [2, 3]][1, -1]", "3"),
            ("[[[10, 20]], [[30, 40]]][1, 0, 1]", "40"),
            ("[[[10, 20]]][0, 0]", "[10, 20]"),
            ("[[null, true, \"text\"]][0, 0]", "null"),
            ("[[null, true, \"text\"]][0, 1]", "true"),
            ("[[null, true, \"text\"]][0, 2]", "text"),
            ("matrix[1, 0..1]", "[4, 5]"),
            ("matrix[1, -1..0]", "[6, 5, 4]"),
            ("[[]][0, 0...0]", "[]"),
        ] {
            let source = format!(
                "var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nvar result = {expression}\n"
            );
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{expression}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{expression}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_list_multidimensional_assignment_preserves_aliases_and_references() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var matrix = [[1, 2], [3, 4]]
            var row = matrix[0]
            var assigned = (matrix[0, 1] = 20)
            matrix[-1, -1] = 40
            var cube = [[[5]]]
            cube[0, 0, 0] = 50
            var slice = matrix[0, 0..1]
            slice[0] = 999
            System.gc()
            matrix[1, 0] = [60]
            System.gc()
            return assigned + row[1] + matrix[1, 1] + cube[0, 0, 0] + matrix[1, 0, 0] + matrix[0, 0]
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 191.0, "jit={jit_enabled}");
        }
    }

    #[test]
    fn source_list_multidimensional_subscripts_reject_invalid_accesses() {
        for (expression, expected) in [
            ("matrix[3, 5]", "index out of bounds"),
            ("matrix[0, 3]", "index out of bounds"),
            ("matrix[-4, 0]", "index out of bounds"),
            ("matrix[0, -4]", "index out of bounds"),
            ("[][0, 0]", "index out of bounds"),
            ("[[]][0, 0]", "index out of bounds"),
            ("matrix[0.5, 0]", "index must be an integer"),
            ("matrix[0, Num.nan]", "index must be an integer"),
            ("matrix[Num.infinity, 0]", "index must be an integer"),
            ("matrix[true, 0]", "index must be a Num"),
            ("matrix[0, null]", "index must be a Num"),
            ("matrix[0..1, 0]", "index must be a Num"),
            ("matrix[0, 0, 0]", "receiver is not List"),
            ("[null][0, 0]", "receiver is not List"),
            ("matrix[3, 5] = 10", "index out of bounds"),
            ("matrix[0, 3] = 10", "index out of bounds"),
            ("matrix[0, -4] = 10", "index out of bounds"),
            ("matrix[0, 0.5] = 10", "index must be an integer"),
            ("matrix[0, 0..1] = 10", "index must be a Num"),
            ("matrix[0, 0, 0] = 10", "receiver is not List"),
        ] {
            let source = format!(
                "var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n{expression}\n"
            );
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                let error = vm.run_module(entry).err().expect(expression);
                assert!(error.message.contains(expected), "{expression}, jit={jit_enabled}: {error:?}");
                let matrix = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(matrix), "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]");
            }
        }
    }

    #[test]
    fn source_list_multidimensional_subscripts_support_maximum_arity() {
        for dimensions in [2, 3, 15, 16] {
            let indices = vec!["0"; dimensions].join(", ");
            let initial = if dimensions < 16 { 0 } else { 42 };
            let assignment = if dimensions < 16 {
                format!("matrix[{indices}] = 42\n")
            } else {
                String::new()
            };
            let source = format!(
                "var matrix = {}{initial}{}\n{assignment}var result = matrix[{indices}]\n",
                "[".repeat(dimensions), "]".repeat(dimensions),
            );
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap();
                assert_eq!(source_result(&vm), 42.0, "dimensions={dimensions}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_mixed_multidimensional_subscripts() {
        for (expression, expected) in [
            (r#"map_list[1, "source"]"#, "[3, 4]"),
            (r#"map_list[-1, "source", 0]"#, "3"),
            (r#"map_list[0, "missing"]"#, "null"),
            (r#"map["rows", 0, "source", -1]"#, "2"),
            (r#"map["meta", "version"]"#, "7"),
            (r#"map["meta", "missing"]"#, "null"),
            (r#"map[1.5, null]"#, "9"),
            (r#"map[true, -1]"#, "10"),
            (r#"map["rows", 1, "source", 1..0]"#, "[4, 3]"),
            (r#"map["rows", 1, "source", 0] == map["rows"][1]["source"][0]"#, "true"),
        ] {
            let source = format!(r#"
                var map_list = [{{"source": [1, 2]}}, {{"source": [3, 4]}}]
                var map = {{"rows": map_list, "meta": {{"version": 7}}, 1.5: {{null: 9}}, true: {{-1: 10}}}}
                var result = {expression}
            "#);
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{expression}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{expression}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_mixed_multidimensional_assignment_preserves_keys_aliases_and_gc() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var map_list = [{}, {}]
            var map = {"rows": map_list, "meta": {}}
            var alias = map_list[1]
            System.gc()
            var assigned = (map_list[1, "source"] = [1, 2, 3])
            map["rows", 1, "source", -1] = 9
            map["meta", "version"] = 7
            map["rows", 0, "source"] = {"nested": [11]}
            map["rows", 0, "source", "nested", 0] = 12
            var objectKey = [100]
            map["meta", objectKey] = [13]
            var rangeKey = 1..3
            map["meta", rangeKey] = 14
            var i = 0
            while (i < 100) {
                map["meta", i] = [i]
                i = i + 1
            }
            System.gc()
            return alias["source", 2] + assigned[2] + map["meta", "version"] +
                map["rows", 0, "source", "nested", 0] + map["meta", objectKey, 0] +
                map["meta", rangeKey] + map["meta", 99, 0]
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 163.0, "jit={jit_enabled}");
        }
    }

    #[test]
    fn source_mixed_multidimensional_subscripts_reject_invalid_paths() {
        for (expression, expected) in [
            (r#"data["missing", "source"]"#, "receiver is not List or Map"),
            (r#"data["empty", "missing", "source"]"#, "receiver is not List or Map"),
            (r#"data["null", "source"]"#, "receiver is not List or Map"),
            (r#"data["scalar", 0]"#, "receiver is not List or Map"),
            (r#"data["rows", 1, "source"]"#, "index out of bounds"),
            (r#"data["rows", "bad", "source"]"#, "index must be a Num"),
            (r#"data["rows", 0.5, "source"]"#, "index must be an integer"),
        ] {
            for suffix in ["", " = [42]"] {
                let source = format!(r#"
                    var data = {{"rows": [{{}}], "empty": {{}}, "null": null, "scalar": 7}}
                    {expression}{suffix}
                "#);
                let (program, entry, manifest) = compile_wren_module(&source);
                for jit_enabled in [false, true] {
                    let mut vm = source_vm(&program, &manifest);
                    vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                    let error = vm.run_module(entry).err().expect(expression);
                    assert!(error.message.contains(expected), "{expression}{suffix}, jit={jit_enabled}: {error:?}");
                    let data = *vm.modules.last().unwrap().variables.last().unwrap();
                    let context = HostContext { heap: &mut vm.heap };
                    assert_eq!(context.map_len(data), Some(4));
                    let HeapObject::Map(map) = context.heap.get(data).unwrap() else { panic!("expected Map") };
                    let empty = map.entries[&MapKey::String(Arc::from(b"empty".as_slice()))];
                    let rows = map.entries[&MapKey::String(Arc::from(b"rows".as_slice()))];
                    assert_eq!(context.map_len(empty), Some(0));
                    assert_eq!(context.map_len(context.list_get(rows, 0).unwrap()), Some(0));
                }
            }
        }
    }

    #[test]
    fn source_map_multidimensional_subscripts_support_maximum_arity() {
        for dimensions in [2, 15, 16] {
            let indices = vec![r#""next""#; dimensions].join(", ");
            let initial = if dimensions < 16 { "0" } else { "42" };
            let nested = (0..dimensions).fold(initial.to_owned(), |inner, _| {
                format!(r#"{{"next": {inner}}}"#)
            });
            let assignment = if dimensions < 16 {
                format!("data[{indices}] = 42\n")
            } else {
                String::new()
            };
            let source = format!("var data = {nested}\n{assignment}var result = data[{indices}]\n");
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
                vm.run_module(entry).unwrap();
                assert_eq!(source_result(&vm), 42.0, "dimensions={dimensions}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_custom_multidimensional_subscripts_keep_method_dispatch() {
        let (program, entry, manifest) = compile_wren_module(r#"
            class Grid {
                construct new() { _rows = [[1, 2], [3, 4]] }
                [column, row] { _rows[row, column] }
                [column, row]=(value) { _rows[row, column] = value }
            }
            var grid = Grid.new()
            var before = grid[0, 1]
            grid[1, 0] = 20
            var result = before + grid[1, 0]
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 23.0, "jit={jit_enabled}");
        }
    }

    #[test]
    fn source_ranges_and_list_slices() {
        let cases = [
            ("trees[1..2]", "[birch, oak]"),
            ("trees[1..-1]", "[birch, oak, willow]"),
            ("trees[1...3]", "[birch, oak]"),
            ("trees[0...4]", "[cedar, birch, oak, willow]"),
            ("trees[0..-1]", "[cedar, birch, oak, willow]"),
            ("trees[-3..-2]", "[birch, oak]"),
            ("trees[3..1]", "[willow, oak, birch]"),
            ("trees[3...1]", "[willow, oak]"),
            ("trees[-1..0]", "[willow, oak, birch, cedar]"),
            ("trees[-1...-4]", "[willow, oak, birch]"),
            ("trees[2..2]", "[oak]"),
            ("trees[2...2]", "[]"),
            ("trees[4...4]", "[]"),
            ("trees[4..-1]", "[]"),
            ("trees[0...-5]", "[cedar]"),
            ("[][0..-1]", "[]"),
            ("[][0...0]", "[]"),
            ("trees[1..3][1..-1]", "[oak, willow]"),
            ("trees[0]", "cedar"),
            ("trees[-1]", "willow"),
            ("(1..3).toString", "1..3"),
            ("(3...1).toString", "3...1"),
            ("(1..3).from", "1"),
            ("(1..3).to", "3"),
            ("(3...1).min", "1"),
            ("(3...1).max", "3"),
            ("(1..3).isInclusive", "true"),
            ("(1...3).isInclusive", "false"),
        ];
        for (expression, expected) in cases {
            let source = format!(
                "var trees = [\"cedar\", \"birch\", \"oak\", \"willow\"]\nvar result = {expression}\n"
            );
            let (program, entry, manifest) = compile_wren_module(&source);
            for jit_enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                vm.set_jit_options(jit_enabled.then(|| TracingJitOptions {
                    hot_loop_threshold: 2,
                    ..Default::default()
                }));
                vm.run_module(entry).unwrap_or_else(|error| panic!("{expression}: {error:?}"));
                let value = *vm.modules.last().unwrap().variables.last().unwrap();
                let context = HostContext { heap: &mut vm.heap };
                assert_eq!(context.format_value(value), expected, "{expression}, jit={jit_enabled}");
            }
        }
    }

    #[test]
    fn source_range_iteration_and_slice_copy_semantics() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var total = 0
            for (i in 1..3) { total = total + i }
            for (i in 3...0) { total = total + i }
            for (i in 2...2) { total = total + 1000 }
            for (i in 0.5..2.5) { total = total + i }
            var original = [10, 20, 30]
            var copy = original[0..-1]
            copy[0] = 99
            original[1] = 77
            var nested = [[1]]
            var shallow = nested[0..0]
            shallow[0][0] = 5
            var bounds = 1...3
            var selected = original[bounds]
            return total + original[0] + copy[1] + nested[0][0] + selected.count
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(|| TracingJitOptions {
                hot_loop_threshold: 2,
                ..Default::default()
            }));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 53.5, "jit={jit_enabled}");
        }
    }

    #[test]
    fn source_list_slices_reject_invalid_bounds() {
        let cases = [
            ("trees[4..4]", "range start out of bounds"),
            ("trees[-5..1]", "range start out of bounds"),
            ("trees[0..4]", "range end out of bounds"),
            ("trees[0...5]", "range end out of bounds"),
            ("trees[0..-5]", "range end out of bounds"),
            ("trees[1.5..2]", "range start must be an integer"),
            ("trees[0..1.5]", "range end must be an integer"),
            ("trees[Num.infinity..1]", "range start must be an integer"),
            ("trees[0..Num.nan]", "range end must be an integer"),
            ("trees[1e30..1]", "range start out of bounds"),
            ("trees[0..1e30]", "range end out of bounds"),
            ("[][0..0]", "range start out of bounds"),
            ("trees[true]", "index must be a Num"),
            ("trees[1..2] = []", "index must be a Num"),
            ("1..\"two\"", "range end must be a Num"),
            ("trees[4]", "index out of bounds"),
            ("trees[0.5]", "index must be an integer"),
        ];
        for (expression, expected) in cases {
            let source = format!("var trees = [0, 1, 2, 3]\n{expression}\n");
            let (program, entry, manifest) = compile_wren_module(&source);
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(None);
            let error = vm.run_module(entry).err().expect(expression);
            assert!(error.message.contains(expected), "{expression}: {error:?}");
        }
    }

    #[test]
    fn optimized_source_preserves_closures_branches_and_side_effects() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var value = 0
            var update = Fn.new {
                if (false) { value = 1000 } else { value = value + 1 }
                return value
            }
            var i = 0
            while (i < 20) {
                update.call()
                i = i + 1
            }
            if (true) {
                return value + 22
            } else {
                return 0
            }
        "#);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(|| TracingJitOptions {
                hot_loop_threshold: 2,
                ..Default::default()
            }));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 42.0);
        }
    }

    #[test]
    fn scalar_replacement_preserves_object_cell_and_closure_semantics() {
        let source = r#"
class Point {
  construct new(x, y) {
    _x = x
    _y = y
    _sum = _x + _y
  }
  x { _x }
  sum { _sum }
  shift(dx) {
    _x = _x + dx
    return _x * 2 + _y
  }
}

class Sroa {
  static run() {
    var total = 0
    var counter = 0
    var bump = Fn.new { counter = counter + 1 }
    var i = 0
    while (i < 30) {
      var p = Point.new(i, 2)
      total = total + p.sum
      if (i % 3 == 0) {
        total = total + p.shift(1)
      }
      var unused = Point.new(i, i)
      var dead = Fn.new { unused }
      var acc = 0
      var add = Fn.new { |v| acc = acc + v }
      if (i % 2 == 0) {
        acc = acc + i
      } else {
        acc = acc - 1
      }
      total = total + acc
      bump.call()
      i = i + 1
    }
    return total + counter
  }
}
var result = Sroa.run()
"#;
        // Reference value computed by hand from the Wren semantics above.
        let mut expected = 0.0f64;
        for i in 0..30 {
            let i = i as f64;
            expected += i + 2.0;
            if i % 3.0 == 0.0 { expected += (i + 1.0) * 2.0 + 2.0; }
            expected += if i % 2.0 == 0.0 { i } else { -1.0 };
        }
        expected += 30.0;

        let (program, entry, manifest) = compile_wren_module(source);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(|| TracingJitOptions {
                hot_loop_threshold: 2,
                ..Default::default()
            }));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), expected, "jit_enabled={jit_enabled}");
        }
    }

    #[test]
    fn executes_threaded_integer_program() {
        let mut asm = Assembler::new();
        asm.emit_ab(Opcode::LoadI8, 0, 20);
        asm.emit_ab(Opcode::LoadI8, 1, 22);
        asm.emit_abc(Opcode::IAdd, 2, 0, 1);
        asm.emit_ab(Opcode::Int32ToNum, 3, 2);
        asm.emit_a(Opcode::Return, 3);
        let program = simple_program(asm.finish().unwrap(), 4);
        let mut vm = Interpreter::new(&program).unwrap();
        let result = vm.run_function(FuncId(0), &[]).unwrap();
        match result { ExecutionResult::Returned(value) => assert_eq!(value.as_num(), Some(42.0)), _ => panic!() }
    }

        #[test]
    fn jit_executes_native_return_and_integer_loop() {
        let mut asm = Assembler::new();
        asm.emit_ab(Opcode::LoadI8, 0, 0);
        asm.emit_ab(Opcode::LoadI8, 1, 1);
        asm.emit_ab(Opcode::LoadI8, 2, 100);
        let head = asm.label();
        let done = asm.label();
        asm.bind(head);
        asm.emit_abc(Opcode::ICmpLt, 3, 0, 2);
        asm.emit_jump_if_false(Reg(3), done);
        asm.emit_abc(Opcode::IAdd, 0, 0, 1);
        asm.emit_jump(head);
        asm.bind(done);
        asm.emit_ab(Opcode::Int32ToNum, 4, 0);
        asm.emit_a(Opcode::Return, 4);
        let program = simple_program(asm.finish().unwrap(), 5);
        let mut vm = Interpreter::new(&program).unwrap();
        vm.set_jit_options(Some(TracingJitOptions { hot_loop_threshold: 2, ..Default::default() }));
        for iteration in 0..3 {
            let ExecutionResult::Returned(value) = vm.run_function(FuncId(0), &[]).unwrap() else { panic!() };
            assert_eq!(value.as_num(), Some(100.0), "iteration {iteration}: {:?}\n{:#?}", vm.jit_stats(), vm.jit.as_ref().unwrap().trace(crate::wren_trace_ir::TraceId(0)));
        }
        let stats = vm.jit_stats().unwrap();
        assert!(stats.native_entries > 0, "{stats:?}");
        assert_eq!(stats.recordings_aborted, 0, "{stats:?}");
    }

    #[test]
    fn jit_executes_runtime_helpers_and_gc() {
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::AllocList, 0);
        asm.emit_a(Opcode::LoadTrue, 1);
        asm.emit_ab(Opcode::ListAppend, 0, 1);
        asm.emit_ab(Opcode::ListLen, 2, 0);
        asm.emit_ab(Opcode::Int32ToNum, 3, 2);
        asm.emit_a(Opcode::Return, 3);
        let mut chunk = asm.finish().unwrap();
        chunk.safepoints.push(SafepointEntry { pc: BytecodePc(0), kind: SafepointKind::Allocation, references: ReferenceMap::default() });
        let mut roots = ReferenceMap::default();
        roots.set_general(Reg(0), true);
        chunk.safepoints.push(SafepointEntry { pc: BytecodePc(2), kind: SafepointKind::RuntimeMutation, references: roots });
        let program = simple_program(chunk, 4);
        let mut vm = Interpreter::new(&program).unwrap();
        vm.set_jit_options(Some(TracingJitOptions { hot_loop_threshold: 1, ..Default::default() }));
        for _ in 0..3 {
            vm.collect_garbage();
            let ExecutionResult::Returned(value) = vm.run_function(FuncId(0), &[]).unwrap() else { panic!() };
            assert_eq!(value.as_num(), Some(1.0));
        }
        let stats = vm.jit_stats().unwrap();
        assert_eq!(stats.native_entries, 0, "straight-line code must not be mistaken for a hot loop: {stats:?}");
        assert_eq!(stats.recordings_started, 0, "{stats:?}");
    }

    #[test]
    fn jit_completes_dynamic_numeric_invoke_loop() {
        let source = r#"
            var i = 0
            while (i < 20000) {
                i = i + 1
            }
        "#;
        let (program, entry, manifest) = compile_local_wren(source);
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            native_loop_iteration_budget: 1 << 18,
            ..Default::default()
        }));

        let result = vm.run_module(entry);
        assert!(result.is_ok(), "dynamic numeric loop failed: {result:?}");
        let stats = vm.jit_stats().unwrap();
        assert!(stats.recordings_started > 0, "{stats:?}");
        assert!(stats.native_entries > 0, "{stats:?}");
        assert_eq!(stats.native_budget_exits, 0, "loop must take its real native exit before the watchdog: {stats:?}");
        assert_eq!(stats.blacklisted_loops, 0, "correct trace must not be blacklisted: {stats:?}");
    }

    #[test]
    fn jit_completes_mixed_int_f64_loop() {
        let source = r#"
            var sum = 0.0
            var i = 0
            var steps = 20000
            while (i < steps) {
                var x = (i + 0.5) / steps
                sum = sum + 4.0 / (1.0 + x * x)
                i = i + 1
            }
        "#;
        let (program, entry, manifest) = compile_local_wren(source);
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            native_loop_iteration_budget: 1 << 18,
            ..Default::default()
        }));

        let result = vm.run_module(entry);
        assert!(result.is_ok(), "mixed Int32/F64 loop failed: {result:?}");
        let stats = vm.jit_stats().unwrap();
        assert!(stats.recordings_started > 0, "{stats:?}");
        assert!(stats.native_entries > 0, "{stats:?}");
        assert_eq!(stats.native_budget_exits, 0, "mixed numeric loop must take its real native exit before the watchdog: {stats:?}");
        assert_eq!(stats.blacklisted_loops, 0, "correct mixed numeric trace must not be blacklisted: {stats:?}");
    }

    #[test]
    fn jit_does_not_install_finite_prefix_for_outer_nested_loop() {
        let source = r#"
            var total = 0
            var y = 0
            while (y < 40) {
                var x = 0
                while (x < 200) {
                    total = total + x
                    x = x + 1
                }
                y = y + 1
            }
        "#;
        let (program, entry, manifest) = compile_local_wren(source);
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            max_failures_before_blacklist: 2,
            native_loop_iteration_budget: 1 << 18,
            ..Default::default()
        }));

        let result = vm.run_module(entry);
        assert!(result.is_ok(), "nested numeric loop failed: {result:?}");
        let stats = vm.jit_stats().unwrap();
        // Only the innermost loop should become a root native loop. The outer
        // loop necessarily encounters the inner cycle before it can close.
        assert_eq!(
            stats.traces_compiled.saturating_sub(stats.side_traces_compiled),
            1,
            "outer nested loop must not install a second root trace: {stats:?}",
        );
        assert!(stats.recordings_aborted > 0, "outer recording should abort on the nested cycle: {stats:?}");
        assert!(stats.blacklisted_loops > 0, "repeatedly untraceable outer loop should fall back to baseline: {stats:?}");
        assert_eq!(stats.trace_cooldowns, 0, "nested-cycle rejection should happen before native thrashing: {stats:?}");
    }


    #[test]
    fn jit_completes_mandelbrot_like_multi_guard_inner_loop() {
        // This is deliberately shaped like jit.wren: two outer loops and an
        // inner loop with a short-circuit condition over several loop-carried
        // floating-point values. It catches stale LoopPhi/backedge lowering much
        // more reliably than a single induction-variable loop.
        let source = r#"
            var checksum = 0
            var escaped = 0
            var y = 0
            var width = 24
            var height = 16
            var maxIter = 120

            while (y < height) {
                var ci = (y / height) * 2.0 - 1.0
                var x = 0
                while (x < width) {
                    var cr = (x / width) * 3.5 - 2.5
                    var zr = 0.0
                    var zi = 0.0
                    var zr2 = 0.0
                    var zi2 = 0.0
                    var i = 0

                    while (i < maxIter && (zr2 + zi2) <= 4.0) {
                        zi = 2.0 * zr * zi + ci
                        zr = zr2 - zi2 + cr
                        zr2 = zr * zr
                        zi2 = zi * zi
                        i = i + 1
                    }

                    if (i < maxIter) { escaped = escaped + 1 }
                    checksum = checksum + i * (x + 1) * (y + 1)
                    x = x + 1
                }
                y = y + 1
            }
        "#;

        let (program, entry, manifest) = compile_local_wren(source);
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            native_loop_iteration_budget: 4096,
            ..Default::default()
        }));

        let result = vm.run_module(entry);
        assert!(result.is_ok(), "mandelbrot-like loop failed: {result:?}");
        let stats = vm.jit_stats().unwrap();
        assert!(stats.native_entries > 0, "inner numeric loop should enter native code: {stats:?}");
        assert_eq!(
            stats.native_budget_exits,
            0,
            "mandelbrot-like loop must exit through its real guards, not the watchdog: {stats:?}"
        );
    }

    #[test]
    fn jit_budget_yields_preserve_results_and_native_execution() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var i = 0
            var sum = 0
            while (i < 20000) {
                sum = sum + i
                i = i + 1
            }
            return sum
        "#);
        let mut baseline = source_vm(&program, &manifest);
        baseline.set_jit_options(None);
        baseline.run_module(entry).unwrap();
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            native_loop_iteration_budget: 128,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 199990000.0);
        assert_eq!(source_result(&vm), source_result(&baseline));
        let stats = vm.jit_stats().unwrap();
        assert!(stats.native_budget_exits > 100, "{stats:?}");
        assert!(stats.native_iterations > 19000, "{stats:?}");
        assert_eq!(stats.blacklisted_loops, 0, "{stats:?}");
        assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    }

    #[test]
    fn jit_short_inner_loops_can_become_hot_later() {
        // Start hot, cool down over many short invocations, then become hot
        // again. A permanent blacklist loses all the work in the last phase.
        let (program, entry, manifest) = compile_local_wren(r#"
            var y = 0
            var total = 0
            while (y < 150) {
                var limit = 2
                if (y < 10 || y >= 100) { limit = 1000 }
                var i = 0
                while (i < limit) {
                    total = total + i
                    i = i + 1
                }
                y = y + 1
            }
            return total
        "#);
        let mut baseline = source_vm(&program, &manifest);
        baseline.set_jit_options(None);
        baseline.run_module(entry).unwrap();
        let mut vm = source_vm(&program, &manifest);
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 29970090.0);
        assert_eq!(source_result(&vm), source_result(&baseline));
        let stats = vm.jit_stats().unwrap();
        assert!(stats.trace_cooldowns > 0, "{stats:?}");
        assert!(stats.native_iterations > 58000, "{stats:?}");
    }

    #[test]
    fn jit_does_not_record_enclosing_backedge_as_inner_iteration() {
        let (program, entry, manifest) = compile_local_wren(r#"
            var y = 0
            var total = 0
            while (y < 500) {
                var limit = 2
                if (y >= 450) { limit = 1000 }
                var i = 0
                while (i < limit) {
                    total = total + i
                    i = i + 1
                }
                y = y + 1
            }
            return total
        "#);
        let mut vm = source_vm(&program, &manifest);
        // The first recording starts at i == limit, after a short invocation.
        vm.set_jit_options(Some(TracingJitOptions {
            hot_loop_threshold: 2,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 24975450.0);
        let stats = vm.jit_stats().unwrap();
        assert!(stats.recordings_aborted > 0, "{stats:?}");
        assert!(stats.native_iterations > 45000, "{stats:?}");
        assert_eq!(
            stats.traces_compiled.saturating_sub(stats.side_traces_compiled),
            1,
            "{stats:?}",
        );
    }

    #[test]
    fn jit_benchmark_checksums_match_original_wren() {
        let benchmark = include_str!("../scripts/jit.wren");
        let definitions = benchmark.split("JitBench.integratePi(1000000)").next().unwrap();
        // Reference values from the bundled original Wren executable.
        for (expression, expected) in [
            ("JitBench.integratePi(10000)", 3.1415926544231336),
            ("JitBench.xorshift(10000)", 695495781.0),
            ("JitBench.mandelbrot(80, 60, 100)", 196065774.0),
        ] {
            let source = format!("{definitions}\nvar result = {expression}\n");
            let (program, entry, manifest) = compile_wren_module(&source);
            for enabled in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                if !enabled { vm.set_jit_options(None); }
                vm.run_module(entry).unwrap();
                let result = source_result(&vm);
                assert!((result - expected).abs() < 1e-12,
                    "{expression}, JIT={enabled}: {result} != {expected}");
                if enabled {
                    let stats = vm.jit_stats().unwrap();
                    assert!(stats.native_iterations > 9000, "{expression}: {stats:?}");
                    if expression.contains("xorshift") {
                        assert!(stats.promoted_u32_loop_phis > 0, "{stats:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn bitwise_primitives_match_wren_unsigned_semantics() {
        struct HostWithoutNumericFastPath(BuiltinHost);
        impl Host for HostWithoutNumericFastPath {
            fn call_primitive(
                &mut self, context: &mut HostContext<'_>, primitive: PrimitiveId, args: &[Value],
            ) -> Result<Value, RuntimeError> {
                self.0.call_primitive(context, primitive, args)
            }
        }
        for (expression, expected) in [
            ("~0", 4294967295.0),
            ("1 << 31", 2147483648.0),
            ("4294967295 >> 1", 2147483647.0),
            ("2147483648 << 1", 0.0),
            ("-1 & 255", 255.0),
            ("3.75 ^ 1", 2.0),
            ("4294967297 | 0", 1.0),
        ] {
            let (program, entry, manifest) = compile_local_wren(&format!("return {expression}"));
            for fast_path in [false, true] {
                let mut vm = source_vm(&program, &manifest);
                if !fast_path {
                    vm.host = Box::new(HostWithoutNumericFastPath(BuiltinHost::new(manifest.clone())));
                }
                vm.run_module(entry).unwrap();
                assert_eq!(source_result(&vm), expected, "{expression}, fast_path={fast_path}");
            }
        }
    }

    #[test]
    fn sticky_binding_promotes_edges_and_reclaims_mature_resources() {
        use crate::wren_gc::{Descriptor, Field, TraceLayout, RAW_CODEC};
        static LEAF: Descriptor = Descriptor {
            name: "test.leaf", trace: TraceLayout::Leaf, cleanup: None, finalizer: None,
        };
        static FIELDS: [Field; 1] = [Field { offset: 0, codec: &RAW_CODEC }];
        static PARENT: Descriptor = Descriptor {
            name: "test.parent", trace: TraceLayout::Fields(&FIELDS), cleanup: None, finalizer: None,
        };
        static CLEANED: AtomicUsize = AtomicUsize::new(0);
        unsafe fn cleanup(_: GcObject) { CLEANED.fetch_add(1, Ordering::Relaxed); }
        static RESOURCE: Descriptor = Descriptor {
            name: "test.resource", trace: TraceLayout::Leaf, cleanup: Some(cleanup), finalizer: None,
        };
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::LoadNull, 0);
        asm.emit_a(Opcode::Return, 0);
        let program = simple_program(asm.finish().unwrap(), 1);
        let mut vm = Interpreter::new(&program).unwrap();
        let mutator = &mut vm.heap.mutator;
        let parent = unsafe { mutator.allocate(&PARENT, std::mem::size_of::<usize>()) };
        let root = unsafe { mutator.root(parent) };
        mutator.collect();
        let child = unsafe { mutator.allocate(&LEAF, 16) };
        let weak = unsafe { mutator.downgrade(child) };
        let parent = root.get(mutator, 0).unwrap();
        unsafe { mutator.write(parent, parent.slot(0, &RAW_CODEC), child.address()) };
        mutator.collect();
        let child_root = weak.upgrade(mutator).expect("old-to-young edge was lost");
        let parent = root.get(mutator, 0).unwrap();
        unsafe { mutator.verify_object(parent) }.unwrap();
        drop(child_root);
        unsafe { mutator.write(parent, parent.slot(0, &RAW_CODEC), 0) };
        mutator.collect_full();
        assert!(weak.upgrade(mutator).is_none());
        if std::env::var_os("MMTK_IMMIX_DEFRAG_EVERY_BLOCK").is_some() {
            assert_ne!(parent.address(), root.get(mutator, 0).unwrap().address());
        }

        let resource = unsafe { mutator.allocate(&RESOURCE, 0) };
        let resource_root = unsafe { mutator.root(resource) };
        let resource_weak = unsafe { mutator.downgrade(resource) };
        unsafe { mutator.set_external_bytes(resource, 1234) };
        mutator.collect();
        let full_before = mutator.statistics().full_collections;
        drop(resource_root);
        mutator.collect();
        // Another test's full-GC request may subsume this nursery request.
        if mutator.statistics().full_collections == full_before {
            assert_eq!(CLEANED.load(Ordering::Relaxed), 0, "nursery must retain mature objects");
            assert!(resource_weak.upgrade(mutator).is_some());
        }
        mutator.collect_full();
        assert_eq!(CLEANED.load(Ordering::Relaxed), 1);
        assert!(resource_weak.upgrade(mutator).is_none());
    }

    #[test]
    fn sticky_binding_large_objects_and_deferred_finalizers() {
        use crate::wren_gc::{Descriptor, Mutator, RootFrame, TraceLayout, RAW_CODEC};
        static FINALIZED: AtomicUsize = AtomicUsize::new(0);
        static CLEANED: AtomicUsize = AtomicUsize::new(0);
        static RESURRECTED: StdMutex<Option<crate::wren_gc::GlobalRoot>> = StdMutex::new(None);
        unsafe fn cleanup(_: GcObject) { CLEANED.fetch_add(1, Ordering::Relaxed); }
        fn finalize(mutator: &mut Mutator, root: &RootFrame) {
            assert!(root.get(mutator, 0).is_some());
            mutator.collect_full();
            assert!(root.get(mutator, 0).is_some(), "finalizer argument must stay rooted");
            *RESURRECTED.lock().unwrap() = Some(unsafe {
                mutator.global_root(root.get(mutator, 0).unwrap())
            });
            FINALIZED.fetch_add(1, Ordering::Relaxed);
        }
        static RESOURCE: Descriptor = Descriptor {
            name: "test.finalizable", trace: TraceLayout::Leaf,
            cleanup: Some(cleanup), finalizer: Some(finalize),
        };
        static ARRAY: Descriptor = Descriptor {
            name: "test.large", trace: TraceLayout::Words { offset: 0, codec: &RAW_CODEC },
            cleanup: None, finalizer: None,
        };
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::LoadNull, 0);
        asm.emit_a(Opcode::Return, 0);
        let program = simple_program(asm.finish().unwrap(), 1);
        let mut vm = Interpreter::new(&program).unwrap();
        let mutator = &mut vm.heap.mutator;
        let array = unsafe { mutator.allocate(&ARRAY, 128 << 10) };
        let array_root = unsafe { mutator.root(array) };
        mutator.collect();
        let child = unsafe { mutator.allocate(&RESOURCE, 0) };
        let weak = unsafe { mutator.downgrade(child) };
        let array = array_root.get(mutator, 0).unwrap();
        unsafe { mutator.write(array, array.slot(0, &RAW_CODEC), child.address()) };
        mutator.collect();
        assert!(weak.upgrade(mutator).is_some());
        drop(array_root);
        mutator.collect_full();
        assert_eq!(CLEANED.load(Ordering::Relaxed), 0);
        assert_eq!(mutator.run_finalizers(1), 1);
        assert_eq!(FINALIZED.load(Ordering::Relaxed), 1);
        mutator.collect_full();
        assert_eq!(CLEANED.load(Ordering::Relaxed), 0, "resurrection must retain native resources");
        RESURRECTED.lock().unwrap().take();
        mutator.collect_full();
        assert_eq!(CLEANED.load(Ordering::Relaxed), 1);
        assert_eq!(mutator.run_finalizers(1), 0);
        assert!(weak.upgrade(mutator).is_none());
    }

    #[test]
    fn sticky_managed_mutations_and_allocation_roots_survive() {
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::LoadNull, 0);
        asm.emit_a(Opcode::Return, 0);
        let program = simple_program(asm.finish().unwrap(), 1);
        let mut vm = Interpreter::new(&program).unwrap();
        let list = vm.heap.alloc(HeapObject::List(Vec::new()));
        let list_root = vm.add_root(list);
        let map = vm.heap.alloc(HeapObject::Map(WrenMap::default()));
        let map_root = vm.add_root(map);
        vm.heap.mutator.collect();
        let list_object = vm.heap.object(list).unwrap();
        vm.heap.remember_owner(list);
        {
            use mmtk::vm::ObjectModel;
            let reference = mmtk::util::ObjectReference::from_raw_address(unsafe {
                mmtk::util::Address::from_usize(list_object.address())
            }).unwrap();
            assert_eq!(
                crate::wren_gc::Model::GLOBAL_LOG_BIT_SPEC.as_spec()
                    .load_atomic::<crate::wren_gc::Binding, u8>(reference, None, Ordering::SeqCst),
                0,
                "probable-write must claim the log bit, not enqueue duplicate owners",
            );
        }
        for i in 0..32 {
            let child = vm.heap.alloc(HeapObject::String(Arc::from(format!("child{i}").as_bytes())));
            let mut context = HostContext { heap: &mut vm.heap };
            context.list_push(list, child).unwrap();
            context.map_set(map, list, child).unwrap();
            vm.heap.mutator.collect();
            assert!(matches!(vm.heap.get(child), Some(HeapObject::String(_))));
        }
        let child = match vm.heap.get(list).unwrap() {
            HeapObject::List(items) => items[0],
            _ => unreachable!(),
        };
        let closure = vm.heap.alloc(HeapObject::Closure {
            function_index: 0,
            captures: vec![child.word(), RegisterWord(TAG_HEAP | u32::MAX as u64)].into_boxed_slice(),
            reference_indices: vec![0].into_boxed_slice(),
        });
        let closure_root = vm.add_root(closure);
        vm.release_root(list_root);
        vm.release_root(map_root);
        vm.collect_garbage();
        assert!(matches!(vm.heap.get(child), Some(HeapObject::String(_))));
        // Force collection inside Heap::alloc while the child is only in its
        // Rust-side constructor, no longer in any VM root or reference map.
        let child_object = vm.heap.object(child).unwrap();
        vm.release_root(closure_root);
        let threshold = vm.heap.collector.config_external_threshold_for_test();
        if threshold != 0 {
            let bytes = threshold * 2 + vm.gc_statistics().external_bytes;
            unsafe { vm.heap.mutator.set_external_bytes(child_object, bytes) };
            let before = vm.heap.collection_count();
            let wrapped = vm.heap.alloc(HeapObject::List(vec![child]));
            let root = vm.add_root(wrapped);
            assert!(vm.heap.collection_count() > before, "allocation must actually collect");
            assert!(matches!(vm.heap.get(child), Some(HeapObject::String(_))));
            vm.release_root(root);
        }
    }

    #[test]
    fn wren_fiber_call_yield_resume_and_values() {
        let source = r#"
var fiber = Fiber.new {|first|
  var resumed = Fiber.yield(first + 1)
  return resumed + 2
}
var first = fiber.call(10)
var second = fiber.call(20)
var result = first * 100 + second
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 1122.0, "jit={jit}");
        }
    }

    #[test]
    fn wren_fiber_try_catches_ordinary_runtime_errors_through_call_chain() {
        let source = r#"
var child = Fiber.new {
  123.methodThatDoesNotExist()
}
var parent = Fiber.new {
  child.call()
}
var caught = parent.try()
var result = (caught == parent.error && caught == child.error && parent.isDone && child.isDone) ? 1 : 0
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 1.0, "jit={jit}");
        }
    }

    #[test]
    fn wren_fiber_root_cannot_be_called_but_can_be_transferred_to() {
        let source = r#"
var root = Fiber.current
var inner = Fiber.new {
  root.call()
}
var outer = Fiber.new {
  return inner.try()
}
var error = outer.call()
var result = error != null ? 1 : 0
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 1.0, "jit={jit}");
        }
    }

    #[test]
    fn wren_fiber_transfer_is_peer_to_peer_and_passes_resume_values() {
        let source = r#"
var main = Fiber.current
var fiber = Fiber.new {|first|
  var resumed = main.transfer(first + 1)
  main.transfer(resumed + 1)
}
var first = fiber.transfer(5)
var second = fiber.transfer(10)
var result = first * 100 + second
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 611.0, "jit={jit}");
        }
    }

    #[test]
    fn wren_fiber_transfer_error_reaches_existing_try_caller() {
        let source = r#"
var a
var b

a = Fiber.new {
  b.transferError("boom")
}

b = Fiber.new {
  a.transfer()
}

var caught = b.try()
var result = (caught == "boom" && b.error == "boom" && b.isDone) ? 1 : 0
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 1.0, "jit={jit}");
        }
    }

    #[test]
    fn wren_fiber_abort_null_is_a_noop() {
        let source = r#"
Fiber.abort(null)
var result = 1
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        let mut vm = source_vm(&program, &manifest);
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 1.0);
    }

    #[test]
    fn wren_fiber_suspend_returns_control_to_host_and_resume_supplies_value() {
        let source = r#"
var resumed = Fiber.suspend()
var result = resumed
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(None);
        let first = vm.run_module(entry).unwrap();
        assert!(matches!(first, ExecutionResult::Suspended(value) if value.is_null()));
        let second = vm.resume(Value::from_num(42.0)).unwrap();
        assert!(matches!(second, ExecutionResult::Returned(value) if value.as_num() == Some(42.0)));
        assert_eq!(source_result(&vm), 42.0);
    }

    #[test]
    fn sticky_source_closures_fibers_and_pressure() {
        let source = r#"
class GcTest {
  static run() {
    var keep = []
    var value = "captured"
    var closure = Fn.new { value }
    var fiber = Fiber.new {
      var local = ["fiber"]
      Fiber.yield(local)
      System.gc()
      return local[0]
    }
    var first = fiber.call()
    System.gc()
    var i = 0
    while (i < 50000) {
      var garbage = [i, i + 1, i + 2]
      if (i < 100) keep.add([closure, first])
      i = i + 1
    }
    System.gc()
    if (closure.call() != "captured") Fiber.abort("lost capture")
    if (fiber.call() != "fiber") Fiber.abort("lost suspended fiber")
    return keep.count
  }
}
var result = GcTest.run()
"#;
        let (program, entry, manifest) = compile_wren_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 100.0);
            assert!(vm.gc_statistics().full_collections > 0);
        }
    }

    #[test]
    fn sticky_numeric_loops_rendezvous_with_another_mutator() {
        let (program, entry, manifest) = compile_local_wren(
            "var i = 0\nwhile (i < 2000000) { i = i + 1 }\nreturn i",
        );
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            let collector = vm.heap.collector;
            let (ready_tx, ready_rx) = std::sync::mpsc::channel();
            let (go_tx, go_rx) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let mut mutator = collector.attach();
                unsafe {
                    mutator.with_native(|| {
                        ready_tx.send(()).unwrap();
                        go_rx.recv().unwrap();
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    });
                }
                mutator.collect_full();
            });
            unsafe { vm.heap.mutator.with_native(|| ready_rx.recv().unwrap()) };
            go_tx.send(()).unwrap();
            vm.run_module(entry).unwrap();
            // Joining while Running could block a worker whose request arrived
            // just after the finite loop returned.
            unsafe { vm.heap.mutator.with_native(|| worker.join().unwrap()) };
            assert_eq!(source_result(&vm), 2000000.0);
        }
    }

    #[test]
    fn sticky_attach_parks_idle_same_thread_peers() {
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::LoadNull, 0);
        asm.emit_a(Opcode::Return, 0);
        let program = simple_program(asm.finish().unwrap(), 1);
        let mut vm = Interpreter::new(&program).unwrap();
        let collector = vm.heap.collector;
        let worker = std::thread::spawn(move || {
            let mut mutator = collector.attach();
            mutator.collect_full();
        });
        while !vm.heap.mutator.collection_requested() {
            std::thread::yield_now();
        }
        let second = Heap::default();
        drop(second);
        unsafe { vm.heap.mutator.with_native(|| worker.join().unwrap()) };
    }

    #[test]
    fn list_fast_path_and_gc_survive() {
        let mut asm = Assembler::new();
        asm.emit_a(Opcode::AllocList, 0);
        asm.emit_a(Opcode::LoadTrue, 1);
        asm.emit_ab(Opcode::ListAppend, 0, 1);
        asm.emit_ab(Opcode::ListLen, 2, 0);
        asm.emit_ab(Opcode::Int32ToNum, 3, 2);
        asm.emit_a(Opcode::Return, 3);
        let mut chunk = asm.finish().unwrap();
        chunk.safepoints.push(SafepointEntry {
            pc: BytecodePc(0),
            kind: SafepointKind::Allocation,
            references: ReferenceMap::default(),
        });
        let mut append_roots = ReferenceMap::default();
        append_roots.set_general(Reg(0), true);
        chunk.safepoints.push(SafepointEntry {
            pc: BytecodePc(2),
            kind: SafepointKind::RuntimeMutation,
            references: append_roots,
        });
        let program = simple_program(chunk, 4);
        let mut vm = Interpreter::new(&program).unwrap();
        let result = vm.run_function(FuncId(0), &[]).unwrap();
        match result { ExecutionResult::Returned(value) => assert_eq!(value.as_num(), Some(1.0)), _ => panic!() }
    }
}
