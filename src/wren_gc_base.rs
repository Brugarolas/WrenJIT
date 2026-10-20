//! Runtime-neutral MMTk 0.33.0 StickyImmix binding.
//!
//! All GC references must be rooted across allocation, collection and safepoints.
//! Object addresses are transient; reload roots after every possible collection.
//! See README.md for the complete embedding and unsafe contracts.
#![deny(unsafe_op_in_unsafe_fn)]

use mmtk::memory_manager as mm;
use mmtk::util::alloc::{Allocator, ImmixAllocator};
use mmtk::util::copy::{CopySemantics, GCWorkerCopyContext};
use mmtk::util::{Address, ObjectReference, opaque_pointer::*};
use mmtk::vm::slot::{MemorySlice, Slot as MmtkSlot};
use mmtk::vm::{self, ObjectModel, ObjectTracer, ObjectTracerContext, RootsWorkFactory};
use mmtk::{AllocationSemantics, MMTK, MMTKBuilder};
use parking_lot::{Condvar, Mutex};
use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Instant;

const WORD: usize = size_of::<usize>();
const ROOT_BATCH: usize = 512;
const MAX_ALIGNMENT: usize = 16;
static GC: OnceLock<Collector> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());
thread_local! {
    static CURRENT: Cell<*const ThreadRecord> = const { Cell::new(ptr::null()) };
}

/// An unrooted object address. Copying this value does not keep the object alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Object(ObjectReference);

impl Object {
    pub fn address(self) -> usize {
        self.0.to_raw_address().as_usize()
    }

    /// # Safety
    /// `address` must be a live object base allocated by this binding.
    pub unsafe fn from_address(address: usize) -> Self {
        Self(
            ObjectReference::from_raw_address(unsafe { Address::from_usize(address) })
                .expect("null object address"),
        )
    }

    /// # Safety
    /// The object must remain live and unmoved throughout the access. No GC may
    /// occur while a pointer or Rust reference into this payload is in use.
    pub unsafe fn payload(self) -> *mut u8 {
        unsafe { (self.address() as *mut u8).add(size_of::<Header>()) }
    }

    unsafe fn header(self) -> &'static Header {
        unsafe { &*(self.address() as *const Header) }
    }

    /// # Safety
    /// The object must be live and current.
    pub unsafe fn payload_bytes(self) -> usize {
        unsafe { self.header().payload_bytes }
    }

    /// # Safety
    /// The object must be live and `offset` must identify an initialized atomic
    /// reference slot. The codec must describe that slot's representation.
    pub unsafe fn slot(self, offset: usize, codec: &'static SlotCodec) -> Slot {
        assert!(
            offset
                .checked_add(WORD)
                .is_some_and(|n| n <= unsafe { self.payload_bytes() })
        );
        unsafe { Slot::from_ptr(self.payload().add(offset).cast(), codec) }
    }
}

/// A word-sized atomic slot codec. Functions must be pure, thread-safe and must
/// neither allocate GC objects nor dereference candidate object addresses.
/// `decode(0)` MUST return None. `relocate` preserves non-address tag bits.
pub struct SlotCodec {
    pub decode: fn(usize) -> Option<usize>,
    pub relocate: fn(usize, usize) -> usize,
}

pub static RAW_CODEC: SlotCodec = SlotCodec {
    decode: |word| if word == 0 { None } else { Some(word) },
    relocate: |_, address| address,
};

/// A stable address and representation for a reference field. Slots may be
/// processed concurrently by GC workers; all accesses use AtomicUsize.
#[derive(Clone, Copy)]
pub struct Slot {
    address: usize,
    codec: &'static SlotCodec,
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("address", &self.address)
            .finish()
    }
}
impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address && ptr::eq(self.codec, other.codec)
    }
}
impl Eq for Slot {}
impl std::hash::Hash for Slot {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.address, state);
        std::hash::Hash::hash(&(self.codec as *const _ as usize), state);
    }
}
impl Slot {
    /// # Safety
    /// The pointer must address initialized, aligned AtomicUsize storage that
    /// outlives all uses of this slot. Every decoded pointer must be a live GC
    /// object. No overlapping non-atomic access is allowed during collection.
    pub unsafe fn from_ptr(pointer: *const AtomicUsize, codec: &'static SlotCodec) -> Self {
        assert!(!pointer.is_null() && (pointer as usize).is_multiple_of(align_of::<AtomicUsize>()));
        Self {
            address: pointer as usize,
            codec,
        }
    }
    fn atomic(self) -> &'static AtomicUsize {
        unsafe { &*(self.address as *const AtomicUsize) }
    }
    fn word(self) -> usize {
        self.atomic().load(Ordering::Relaxed)
    }
    fn decode(self, word: usize) -> Option<ObjectReference> {
        (self.codec.decode)(word).map(|address| {
            ObjectReference::from_raw_address(unsafe { Address::from_usize(address) })
                .expect("codec decoded a null object")
        })
    }
}
impl MmtkSlot for Slot {
    #[inline]
    fn load(&self) -> Option<ObjectReference> {
        self.decode(self.word())
    }
    #[inline]
    fn store(&self, object: ObjectReference) {
        let old = self.word();
        let new = (self.codec.relocate)(old, object.to_raw_address().as_usize());
        self.atomic().store(new, Ordering::Relaxed);
    }
}

/// Heap reference slice. Construction is unsafe because its lifetime is erased.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceSlice {
    start: Slot,
    count: usize,
    owner: Object,
}
impl ReferenceSlice {
    /// # Safety
    /// All `count` words must be valid slots with the same codec, owned by
    /// `owner`, and stable until GC finishes processing any submitted work.
    pub unsafe fn new(owner: Object, start: Slot, count: usize) -> Self {
        assert!(
            count
                .checked_mul(WORD)
                .and_then(|n| start.address.checked_add(n))
                .is_some()
        );
        Self {
            start,
            count,
            owner,
        }
    }
}
pub struct SlotIter {
    next: Slot,
    remaining: usize,
}
impl Iterator for SlotIter {
    type Item = Slot;
    fn next(&mut self) -> Option<Slot> {
        if self.remaining == 0 {
            return None;
        }
        let slot = self.next;
        self.remaining -= 1;
        self.next.address += WORD;
        Some(slot)
    }
}
impl MemorySlice for ReferenceSlice {
    type SlotType = Slot;
    type SlotIterator = SlotIter;
    fn iter_slots(&self) -> SlotIter {
        SlotIter {
            next: self.start,
            remaining: self.count,
        }
    }
    fn object(&self) -> Option<ObjectReference> {
        Some(self.owner.0)
    }
    fn start(&self) -> Address {
        unsafe { Address::from_usize(self.start.address) }
    }
    fn bytes(&self) -> usize {
        self.count * WORD
    }
    fn copy(source: &Self, target: &Self) {
        assert_eq!(source.count, target.count);
        assert!(ptr::eq(source.start.codec, target.start.codec));
        // Memmove semantics with atomic word accesses, including overlapping slices.
        let copy_one = |i| {
            let src = Slot {
                address: source.start.address + i * WORD,
                ..source.start
            };
            let dst = Slot {
                address: target.start.address + i * WORD,
                ..target.start
            };
            dst.atomic().store(src.word(), Ordering::Relaxed);
        };
        if target.start.address > source.start.address {
            for i in (0..source.count).rev() {
                copy_one(i);
            }
        } else {
            for i in 0..source.count {
                copy_one(i);
            }
        }
    }
}

/// A field offset is relative to the payload, never to the GC header.
#[derive(Clone, Copy)]
pub struct Field {
    pub offset: usize,
    pub codec: &'static SlotCodec,
}
pub enum TraceLayout {
    Leaf,
    Fields(&'static [Field]),
    /// Every word from `offset` to the end of the payload is a reference slot.
    Words {
        offset: usize,
        codec: &'static SlotCodec,
    },
    /// Must emit all strong reference slots, without mutable references to the
    /// whole object. Callbacks may run concurrently, including for one object.
    Custom(unsafe fn(Object, &mut dyn FnMut(Slot))),
}

/// Immutable runtime-supplied layout. All payloads must be byte-relocatable.
/// Address-sensitive resources must be kept behind pinned roots or represented
/// by stable native handles. No Rust references into a payload may cross a GC.
pub struct Descriptor {
    pub name: &'static str,
    pub trace: TraceLayout,
    /// Native resource cleanup, called once before reclaiming an unreachable
    /// object. Must not panic, allocate GC objects, resurrect or inspect children.
    pub cleanup: Option<unsafe fn(Object)>,
    /// Deferred language finalizer, executed only by Mutator::run_finalizers.
    /// Its rooted argument may be retained to resurrect the object. Runs once.
    pub finalizer: Option<fn(&mut Mutator, &RootFrame)>,
}

#[repr(C, align(16))]
struct Header {
    forwarding: AtomicUsize,
    bytes: usize,
    payload_bytes: usize,
    descriptor: &'static Descriptor,
    external_bytes: AtomicUsize,
}

fn trace<V: FnMut(Slot)>(object: Object, visitor: &mut V) {
    let header = unsafe { object.header() };
    match &header.descriptor.trace {
        TraceLayout::Leaf => {}
        TraceLayout::Fields(fields) => {
            let base = unsafe { object.payload() } as usize;
            for field in *fields {
                visitor(Slot {
                    address: base + field.offset,
                    codec: field.codec,
                });
            }
        }
        TraceLayout::Words { offset, codec } => {
            let mut address = unsafe { object.payload().add(*offset) } as usize;
            let end = unsafe { object.payload() } as usize + header.payload_bytes;
            while address < end {
                visitor(Slot { address, codec });
                address += WORD;
            }
        }
        TraceLayout::Custom(callback) => unsafe { callback(object, visitor) },
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub heap_bytes: usize,
    pub workers: usize,
    /// External-byte growth triggering a collection before the next allocation.
    /// Zero disables this trigger; byte accounting remains enabled.
    pub external_gc_threshold: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            heap_bytes: 256 << 20,
            workers: 2,
            external_gc_threshold: 64 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Statistics {
    pub collections: u64,
    pub pause_ns: u64,
    pub rendezvous_ns: u64,
    pub external_bytes: usize,
    pub allocated_bytes: u64,
    pub barrier_slow_calls: u64,
    pub copied_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationError {
    InvalidObject(usize),
    InvalidReference { slot: usize, target: usize },
}
#[derive(Default)]
struct Counters {
    collections: AtomicU64,
    pause_ns: AtomicU64,
    rendezvous_ns: AtomicU64,
    external_bytes: AtomicUsize,
    allocated_bytes: AtomicU64,
    barrier_slow_calls: AtomicU64,
    copied_bytes: AtomicU64,
}

pub struct Collector {
    mmtk: &'static MMTK<Binding>,
    config: Config,
    requested: AtomicBool,
    world: Mutex<World>,
    changed: Condvar,
    roots: Mutex<Vec<Weak<RootStorage>>>,
    providers: Mutex<Vec<Weak<dyn RootProvider>>>,
    weak: Mutex<Vec<Weak<AtomicUsize>>>,
    tracked: Mutex<Vec<Tracked>>,
    pending: Mutex<Vec<Arc<RootStorage>>>,
    keep_roots: Mutex<Vec<Arc<RootStorage>>>,
    keep_providers: Mutex<Vec<Arc<dyn RootProvider>>>,
    external_at_gc: AtomicUsize,
    counters: Counters,
}
struct Tracked {
    object: Object,
    finalized: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThreadState {
    Running,
    Parked,
    Native,
}
struct ThreadEntry {
    record: Arc<ThreadRecord>,
    state: ThreadState,
}
struct World {
    threads: Vec<ThreadEntry>,
    epoch: u64,
    completed: u64,
    active: bool,
    started: Option<Instant>,
}
struct ThreadRecord {
    context: UnsafeCell<Option<Box<mmtk::Mutator<Binding>>>>,
    roots: Mutex<Vec<Weak<RootStorage>>>,
}
// The owner accesses context while Running; GC accesses it only after rendezvous.
// World membership prevents destruction until all GC work has completed.
unsafe impl Send for ThreadRecord {}
unsafe impl Sync for ThreadRecord {}
impl ThreadRecord {
    fn token(&self) -> VMMutatorThread {
        VMMutatorThread(VMThread(OpaquePointer::from_address(Address::from_ptr(
            self,
        ))))
    }
    unsafe fn context_ptr(&self) -> *mut mmtk::Mutator<Binding> {
        unsafe {
            (*self.context.get())
                .as_deref_mut()
                .expect("detached mutator")
        }
    }
}

impl Collector {
    /// Initializes the process-wide collector once. Repeated calls must use
    /// identical configuration. MMTk 0.33 does not support teardown/restart here.
    pub fn initialize(config: Config) -> &'static Self {
        let _init = INIT.lock();
        if let Some(gc) = GC.get() {
            assert_eq!(gc.config.heap_bytes, config.heap_bytes);
            assert_eq!(gc.config.workers, config.workers);
            assert_eq!(
                gc.config.external_gc_threshold,
                config.external_gc_threshold
            );
            return gc;
        }
        assert!(config.heap_bytes >= 8 << 20 && config.workers > 0);
        let mut builder = MMTKBuilder::new();
        assert!(builder.set_option("plan", "StickyImmix"));
        assert!(builder.set_option("threads", &config.workers.to_string()));
        assert!(builder.set_option(
            "gc_trigger",
            &format!("FixedHeapSize:{}", config.heap_bytes)
        ));
        let mmtk = Box::leak(mm::mmtk_init::<Binding>(&builder));
        let collector = Self {
            mmtk,
            config,
            requested: AtomicBool::new(false),
            changed: Condvar::new(),
            world: Mutex::new(World {
                threads: Vec::new(),
                epoch: 0,
                completed: 0,
                active: false,
                started: None,
            }),
            roots: Mutex::new(Vec::new()),
            providers: Mutex::new(Vec::new()),
            weak: Mutex::new(Vec::new()),
            tracked: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            keep_roots: Mutex::new(Vec::new()),
            keep_providers: Mutex::new(Vec::new()),
            external_at_gc: AtomicUsize::new(0),
            counters: Counters::default(),
        };
        assert!(GC.set(collector).is_ok());
        let gc = GC.get().unwrap();
        mm::initialize_collection(mmtk, VMThread::UNINITIALIZED);
        gc
    }

    pub fn attach(&'static self) -> Mutator {
        CURRENT.with(|current| assert!(current.get().is_null(), "thread already attached"));
        let mut world = self.world.lock();
        while world.active {
            self.changed.wait(&mut world);
        }
        let record = Arc::new(ThreadRecord {
            context: UnsafeCell::new(None),
            roots: Mutex::new(Vec::new()),
        });
        let context = mm::bind_mutator(self.mmtk, record.token());
        unsafe {
            *record.context.get() = Some(context);
        }
        let pointer = NonNull::new(unsafe { record.context_ptr() }).unwrap();
        world.threads.push(ThreadEntry {
            record: record.clone(),
            state: ThreadState::Running,
        });
        CURRENT.with(|current| current.set(Arc::as_ptr(&record)));
        Mutator {
            gc: self,
            record,
            pointer,
            local_bytes: 0,
            local_barriers: 0,
            native: Cell::new(false),
            max_non_los: self
                .mmtk
                .get_plan()
                .constraints()
                .max_non_los_default_alloc_bytes,
            _not_send: PhantomData,
        }
    }

    pub fn statistics(&self) -> Statistics {
        let c = &self.counters;
        Statistics {
            collections: c.collections.load(Ordering::Relaxed),
            pause_ns: c.pause_ns.load(Ordering::Relaxed),
            rendezvous_ns: c.rendezvous_ns.load(Ordering::Relaxed),
            external_bytes: c.external_bytes.load(Ordering::Relaxed),
            allocated_bytes: c.allocated_bytes.load(Ordering::Relaxed),
            barrier_slow_calls: c.barrier_slow_calls.load(Ordering::Relaxed),
            copied_bytes: c.copied_bytes.load(Ordering::Relaxed),
        }
    }

    fn wait(&self, record: &ThreadRecord, for_collection: bool) {
        let mut world = self.world.lock();
        if !for_collection && !world.active {
            return;
        }
        let target = if world.active {
            world.epoch
        } else {
            world.completed + 1
        };
        let entry = world
            .threads
            .iter_mut()
            .find(|e| ptr::eq(Arc::as_ptr(&e.record), record))
            .unwrap();
        assert!(entry.state == ThreadState::Running);
        entry.state = ThreadState::Parked;
        self.changed.notify_all();
        while world.completed < target || world.active {
            self.changed.wait(&mut world);
        }
        world
            .threads
            .iter_mut()
            .find(|e| ptr::eq(Arc::as_ptr(&e.record), record))
            .unwrap()
            .state = ThreadState::Running;
    }
}

fn gc() -> &'static Collector {
    GC.get().expect("collector not initialized")
}

/// Thread-affine GC context. Dropping it detaches and flushes MMTk state.
pub struct Mutator {
    gc: &'static Collector,
    record: Arc<ThreadRecord>,
    pointer: NonNull<mmtk::Mutator<Binding>>,
    local_bytes: u64,
    local_barriers: u64,
    native: Cell<bool>,
    max_non_los: usize,
    _not_send: PhantomData<Rc<()>>,
}

impl Mutator {
    /// Checks one object's outgoing references against MMTk's valid-object
    /// metadata. This is a diagnostic, not a replacement for correct rooting.
    /// # Safety
    /// The object's descriptor and emitted slot storage must be valid. Other
    /// mutators must not resize or destroy its external buffers during checking.
    pub unsafe fn verify_object(&self, object: Object) -> Result<(), VerificationError> {
        self.check_running();
        if mm::is_mmtk_object(object.0.to_raw_address()) != Some(object.0) {
            return Err(VerificationError::InvalidObject(object.address()));
        }
        let mut error = None;
        trace(object, &mut |slot| {
            if let Some(target) = (slot.codec.decode)(slot.word())
                && (target == 0
                    || !target.is_multiple_of(ObjectReference::ALIGNMENT)
                    || mm::is_mmtk_object(unsafe { Address::from_usize(target) }).is_none())
            {
                error = Some(VerificationError::InvalidReference {
                    slot: slot.address,
                    target,
                });
            }
        });
        error.map_or(Ok(()), Err)
    }

    #[inline]
    fn check_running(&self) {
        assert!(!self.native.get(), "heap access in native-safe section");
    }
    #[inline]
    pub fn safepoint(&mut self) {
        self.check_running();
        if self.gc.requested.load(Ordering::Acquire) {
            self.flush_statistics();
            self.gc.wait(&self.record, false);
        }
    }
    pub fn collect(&mut self) {
        self.check_running();
        self.flush_statistics();
        mm::handle_user_collection_request(self.gc.mmtk, self.record.token());
    }
    /// Explicit full-heap collection, including unreachable mature objects.
    pub fn collect_full(&mut self) {
        self.check_running();
        self.flush_statistics();
        self.gc
            .mmtk
            .handle_user_collection_request(self.record.token(), true, true);
    }
    /// The callback may block without delaying collection.
    /// # Safety
    /// The callback must not access managed memory, alter root-provider storage,
    /// or invoke any GC/mutator API. All live managed references must be rooted.
    /// After return, reload all addresses from their roots.
    pub unsafe fn with_native<R>(&mut self, callback: impl FnOnce() -> R) -> R {
        self.safepoint();
        self.flush_statistics();
        {
            let mut world = self.gc.world.lock();
            world
                .threads
                .iter_mut()
                .find(|e| Arc::ptr_eq(&e.record, &self.record))
                .unwrap()
                .state = ThreadState::Native;
            self.native.set(true);
            self.gc.changed.notify_all();
        }
        struct Exit<'a>(&'a Mutator);
        impl Drop for Exit<'_> {
            fn drop(&mut self) {
                let mut world = self.0.gc.world.lock();
                while world.active {
                    self.0.gc.changed.wait(&mut world);
                }
                world
                    .threads
                    .iter_mut()
                    .find(|e| Arc::ptr_eq(&e.record, &self.0.record))
                    .unwrap()
                    .state = ThreadState::Running;
                self.0.native.set(false);
            }
        }
        let _exit = Exit(self);
        callback()
    }
    pub fn flush_statistics(&mut self) {
        self.gc
            .counters
            .allocated_bytes
            .fetch_add(std::mem::take(&mut self.local_bytes), Ordering::Relaxed);
        self.gc
            .counters
            .barrier_slow_calls
            .fetch_add(std::mem::take(&mut self.local_barriers), Ordering::Relaxed);
    }

    /// Allocates a zeroed payload and publishes a fully initialized header.
    /// # Safety
    /// All existing live objects must be rooted before this call. Custom layouts
    /// must tolerate a zeroed payload until initialization finishes. Initialize
    /// any external Rust resources without a safepoint and before publishing
    /// the object. Root the result before another allocating operation.
    pub unsafe fn allocate(
        &mut self,
        descriptor: &'static Descriptor,
        payload_bytes: usize,
    ) -> Object {
        self.safepoint();
        let external = self.gc.counters.external_bytes.load(Ordering::Relaxed);
        let baseline = self.gc.external_at_gc.load(Ordering::Relaxed);
        if self.gc.config.external_gc_threshold != 0
            && external.saturating_sub(baseline) >= self.gc.config.external_gc_threshold
        {
            self.collect();
        }
        match &descriptor.trace {
            TraceLayout::Fields(fields) => {
                for field in *fields {
                    assert!(field.offset.is_multiple_of(WORD));
                    assert!(
                        field
                            .offset
                            .checked_add(WORD)
                            .is_some_and(|n| n <= payload_bytes)
                    );
                }
            }
            TraceLayout::Words { offset, .. } => {
                assert!(offset.is_multiple_of(WORD) && *offset <= payload_bytes);
                assert!((payload_bytes - offset).is_multiple_of(WORD));
            }
            _ => {}
        }
        let bytes = size_of::<Header>()
            .checked_add(payload_bytes)
            .and_then(|n| n.checked_add(WORD - 1))
            .expect("allocation size overflow")
            & !(WORD - 1);
        assert!(bytes <= isize::MAX as usize);
        let semantics = if bytes > self.max_non_los {
            AllocationSemantics::Los
        } else {
            AllocationSemantics::Default
        };
        let context = unsafe { self.pointer.as_mut() };
        // Statically select the Immix allocator. Its real bump-pointer fast
        // path and hole/overflow slow paths retain all MMTk invariants.
        let address = if semantics == AllocationSemantics::Default {
            unsafe { context.allocator_impl_mut_for_semantic::<ImmixAllocator<Binding>>(semantics) }
                .alloc(bytes, MAX_ALIGNMENT, 0)
        } else {
            mm::alloc(context, bytes, MAX_ALIGNMENT, 0, semantics)
        };
        assert!(!address.is_zero(), "MMTk returned a null allocation");
        let object = Object(ObjectReference::from_raw_address(address).unwrap());
        unsafe {
            ptr::write_bytes(address.to_mut_ptr::<u8>(), 0, bytes);
            ptr::write(
                address.to_mut_ptr::<Header>(),
                Header {
                    forwarding: AtomicUsize::new(0),
                    bytes,
                    payload_bytes,
                    descriptor,
                    external_bytes: AtomicUsize::new(0),
                },
            );
        }
        mm::post_alloc(context, object.0, bytes, semantics);
        if descriptor.cleanup.is_some() || descriptor.finalizer.is_some() {
            self.gc.tracked.lock().push(Tracked {
                object,
                finalized: false,
            });
        }
        self.local_bytes = self.local_bytes.saturating_add(bytes as u64);
        object
    }

    /// Stores an encoded word and runs the StickyImmix object barrier.
    /// # Safety
    /// `owner` is the current live owner of `slot`; every decoded reference is
    /// live. No GC may intervene between obtaining these addresses and calling
    /// this method. Mutator-to-mutator publication needs separate synchronization.
    #[inline]
    pub unsafe fn write(&mut self, owner: Object, slot: Slot, value: usize) {
        self.check_running();
        slot.atomic().store(value, Ordering::Relaxed);
        // StickyImmix uses an object-remembering POST barrier. Its pre-barrier
        // is empty. Preserve the plan's atomic unlog-bit test and claim logic.
        if Model::GLOBAL_LOG_BIT_SPEC
            .as_spec()
            .load_atomic::<Binding, u8>(owner.0, None, Ordering::SeqCst)
            != 0
        {
            self.local_barriers = self.local_barriers.saturating_add(1);
            unsafe { self.pointer.as_mut() }
                .barrier
                .object_reference_write_slow(owner.0, slot, slot.decode(value));
        }
    }

    /// Bulk reference copy with overlap handling and MMTk's region barrier.
    /// # Safety
    /// Both regions and owners must remain live and stable; the destination must
    /// enumerate these slots in its tracing descriptor.
    pub unsafe fn copy_references(&mut self, source: ReferenceSlice, target: ReferenceSlice) {
        self.check_running();
        unsafe { self.pointer.as_mut() }
            .barrier
            .memory_region_copy(source, target);
    }

    /// Updates external-memory accounting for a tracked native-resource owner.
    /// # Safety
    /// The object must be live and have a cleanup callback. `bytes` must reflect
    /// its current external ownership; synchronize concurrent resizing yourself.
    pub unsafe fn set_external_bytes(&mut self, object: Object, bytes: usize) {
        self.check_running();
        let header = unsafe { object.header() };
        assert!(
            header.descriptor.cleanup.is_some(),
            "external owner needs cleanup"
        );
        let old = header.external_bytes.swap(bytes, Ordering::Relaxed);
        if bytes >= old {
            self.gc
                .counters
                .external_bytes
                .fetch_add(bytes - old, Ordering::Relaxed);
        } else {
            self.gc
                .counters
                .external_bytes
                .fetch_sub(old - bytes, Ordering::Relaxed);
        }
    }
}

impl Drop for Mutator {
    fn drop(&mut self) {
        self.check_running();
        self.flush_statistics();
        let mut world = self.gc.world.lock();
        while world.active {
            drop(world);
            self.gc.wait(&self.record, false);
            world = self.gc.world.lock();
        }
        // Holding World prevents a new rendezvous from taking a stale snapshot.
        let index = world
            .threads
            .iter()
            .position(|e| Arc::ptr_eq(&e.record, &self.record))
            .unwrap();
        // Preserve any root frames that outlive the attached thread.
        self.gc
            .roots
            .lock()
            .extend(self.record.roots.lock().drain(..));
        unsafe {
            mm::destroy_mutator(self.pointer.as_mut());
            (*self.record.context.get()).take();
        }
        world.threads.swap_remove(index);
        CURRENT.with(|current| current.set(ptr::null()));
        self.gc.changed.notify_all();
    }
}

#[derive(Clone, Copy)]
enum RootKind {
    Precise,
    Pinned,
    Conservative { max_search_bytes: usize },
}
struct RootStorage {
    words: Box<[AtomicUsize]>,
    kind: RootKind,
}
impl RootStorage {
    fn new(count: usize, kind: RootKind) -> Arc<Self> {
        Arc::new(Self {
            words: (0..count).map(|_| AtomicUsize::new(0)).collect(),
            kind,
        })
    }
}

/// Owned, stable, precise root slots. Moving the frame does not move its slots.
/// Frame access requires an active mutator; the frame itself is thread-affine.
pub struct RootFrame {
    storage: Arc<RootStorage>,
    _not_send: PhantomData<Rc<()>>,
}
impl RootFrame {
    pub fn len(&self) -> usize {
        self.storage.words.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, mutator: &Mutator, index: usize) -> Option<Object> {
        mutator.check_running();
        let word = self.storage.words[index].load(Ordering::Relaxed);
        if word == 0 {
            None
        } else {
            Some(unsafe { Object::from_address(word) })
        }
    }
    /// # Safety
    /// `object` must be live at its current address. This operation has no
    /// safepoint, so rooting a just-allocated object is permitted.
    pub unsafe fn set(&self, mutator: &Mutator, index: usize, object: Option<Object>) {
        mutator.check_running();
        self.storage.words[index].store(object.map_or(0, Object::address), Ordering::Relaxed);
    }
    pub fn clear(&self, mutator: &Mutator, index: usize) {
        mutator.check_running();
        self.storage.words[index].store(0, Ordering::Relaxed);
    }
}

/// Sendable strong handle. Read it through an attached mutator and reload after GC.
#[derive(Clone)]
pub struct GlobalRoot {
    storage: Arc<RootStorage>,
}
impl GlobalRoot {
    pub fn get(&self, mutator: &Mutator) -> Object {
        mutator.check_running();
        unsafe { Object::from_address(self.storage.words[0].load(Ordering::Relaxed)) }
    }
}

/// A weak handle is cleared only after the referent is collected. Nursery
/// collections may retain unreachable mature objects until a full collection.
#[derive(Clone)]
pub struct WeakRoot {
    cell: Arc<AtomicUsize>,
}
impl WeakRoot {
    pub fn upgrade(&self, mutator: &mut Mutator) -> Option<RootFrame> {
        mutator.check_running();
        let address = self.cell.load(Ordering::Relaxed);
        if address == 0 {
            None
        } else {
            Some(unsafe { mutator.root(Object::from_address(address)) })
        }
    }
}

/// Owned conservative candidates. These are addresses, not tagged runtime values.
/// A validated interior pointer pins its containing object for that collection.
pub struct ConservativeRoots {
    storage: Arc<RootStorage>,
    _not_send: PhantomData<Rc<()>>,
}
impl ConservativeRoots {
    pub fn set_candidate(&self, mutator: &Mutator, index: usize, candidate: usize) {
        mutator.check_running();
        self.storage.words[index].store(candidate, Ordering::Relaxed);
    }
}

/// Implementors publish roots from globals, JIT stack maps or external modules.
/// Providers may be called repeatedly and must not allocate GC objects or panic.
/// Their destructor may run on a GC worker during pause completion; it must not
/// call collector APIs or wait for mutators to resume.
/// # Safety
/// Emitted slots must remain valid and stable throughout the collection, must
/// follow Slot's atomic/codec contracts, and must include all live strong roots.
/// Any mutating thread must participate in this collector's safepoint protocol.
pub unsafe trait RootProvider: Send + Sync + 'static {
    fn visit(&self, visitor: &mut RootVisitor<'_>);
}
pub struct RootRegistration {
    _provider: Arc<dyn RootProvider>,
}
pub struct RootVisitor<'a> {
    slots: &'a mut Vec<Slot>,
    pinned: &'a mut Vec<ObjectReference>,
}
impl RootVisitor<'_> {
    pub fn slot(&mut self, slot: Slot) {
        self.slots.push(slot);
    }
    /// # Safety
    /// The object must be a current live reference.
    pub unsafe fn pin(&mut self, object: Object) {
        self.pinned.push(object.0);
    }
    pub fn conservative(&mut self, candidate: usize, max_search_bytes: usize) {
        if candidate != 0
            && max_search_bytes != 0
            && let Some(object) = mm::find_object_from_internal_pointer(
                unsafe { Address::from_usize(candidate) },
                max_search_bytes,
            )
        {
            self.pinned.push(object);
        }
    }
}

impl Mutator {
    pub fn roots(&mut self, count: usize) -> RootFrame {
        self.check_running();
        let storage = RootStorage::new(count, RootKind::Precise);
        self.record.roots.lock().push(Arc::downgrade(&storage));
        RootFrame {
            storage,
            _not_send: PhantomData,
        }
    }
    /// # Safety
    /// The object must be live at its current address.
    pub unsafe fn root(&mut self, object: Object) -> RootFrame {
        let root = self.roots(1);
        unsafe {
            root.set(self, 0, Some(object));
        }
        root
    }
    /// Keeps the object's address stable, but does not pin its children.
    /// # Safety
    /// The object must be live at its current address. Rooting has no safepoint.
    pub unsafe fn pinned_root(&mut self, object: Object) -> RootFrame {
        self.check_running();
        let storage = RootStorage::new(1, RootKind::Pinned);
        storage.words[0].store(object.address(), Ordering::Relaxed);
        self.record.roots.lock().push(Arc::downgrade(&storage));
        RootFrame {
            storage,
            _not_send: PhantomData,
        }
    }
    /// # Safety
    /// The object must be live at its current address.
    pub unsafe fn global_root(&mut self, object: Object) -> GlobalRoot {
        self.check_running();
        let storage = RootStorage::new(1, RootKind::Precise);
        storage.words[0].store(object.address(), Ordering::Relaxed);
        self.gc.roots.lock().push(Arc::downgrade(&storage));
        GlobalRoot { storage }
    }
    /// # Safety
    /// The object must be live at its current address.
    pub unsafe fn downgrade(&mut self, object: Object) -> WeakRoot {
        self.check_running();
        let cell = Arc::new(AtomicUsize::new(object.address()));
        self.gc.weak.lock().push(Arc::downgrade(&cell));
        WeakRoot { cell }
    }
    pub fn conservative_roots(
        &mut self,
        count: usize,
        max_search_bytes: usize,
    ) -> ConservativeRoots {
        self.check_running();
        assert!(max_search_bytes > 0);
        let storage = RootStorage::new(count, RootKind::Conservative { max_search_bytes });
        self.record.roots.lock().push(Arc::downgrade(&storage));
        ConservativeRoots {
            storage,
            _not_send: PhantomData,
        }
    }
    pub fn register_roots(&mut self, provider: Arc<dyn RootProvider>) -> RootRegistration {
        self.check_running();
        self.gc.providers.lock().push(Arc::downgrade(&provider));
        RootRegistration {
            _provider: provider,
        }
    }

    /// Executes up to `limit` deferred finalizers. Finalizer arguments stay
    /// rooted during nested allocations/collections. A panicking finalizer is
    /// not retried. Native cleanup remains deferred until actual reclamation.
    pub fn run_finalizers(&mut self, limit: usize) -> usize {
        self.check_running();
        let mut done = 0;
        while done < limit {
            let Some(storage) = self.gc.pending.lock().pop() else {
                break;
            };
            let frame = RootFrame {
                storage,
                _not_send: PhantomData,
            };
            let object = frame.get(self, 0).unwrap();
            let callback = unsafe { object.header().descriptor.finalizer }.unwrap();
            callback(self, &frame);
            done += 1;
        }
        done
    }
}

fn emit_storage(factory: &mut impl RootsWorkFactory<Slot>, storage: Arc<RootStorage>) {
    let mut slots = Vec::with_capacity(ROOT_BATCH);
    let mut pinned = Vec::new();
    for word in &storage.words {
        match storage.kind {
            RootKind::Precise => {
                slots.push(unsafe { Slot::from_ptr(word, &RAW_CODEC) });
                if slots.len() == ROOT_BATCH {
                    factory.create_process_roots_work(std::mem::replace(
                        &mut slots,
                        Vec::with_capacity(ROOT_BATCH),
                    ));
                }
            }
            RootKind::Pinned => {
                let address = word.load(Ordering::Relaxed);
                if address != 0 {
                    pinned.push(unsafe { Object::from_address(address) }.0);
                }
            }
            RootKind::Conservative { max_search_bytes } => {
                RootVisitor {
                    slots: &mut slots,
                    pinned: &mut pinned,
                }
                .conservative(word.load(Ordering::Relaxed), max_search_bytes);
            }
        }
    }
    if !slots.is_empty() {
        factory.create_process_roots_work(slots);
    }
    if !pinned.is_empty() {
        factory.create_process_pinning_roots_work(pinned);
    }
    // Queued work contains raw slot addresses. Keep storage alive even if the
    // last public handle is dropped by a native-safe/unattached thread.
    gc().keep_roots.lock().push(storage);
}
fn emit_registry(
    factory: &mut impl RootsWorkFactory<Slot>,
    registry: &Mutex<Vec<Weak<RootStorage>>>,
) {
    registry.lock().retain(|weak| {
        if let Some(storage) = weak.upgrade() {
            emit_storage(factory, storage);
            true
        } else {
            false
        }
    });
}

#[derive(Default)]
pub struct Binding;
pub struct Model;
pub struct Scanning;
pub struct Collection;
pub struct ActivePlan;
pub struct References;

impl vm::VMBinding for Binding {
    type VMObjectModel = Model;
    type VMScanning = Scanning;
    type VMCollection = Collection;
    type VMActivePlan = ActivePlan;
    type VMReferenceGlue = References;
    type VMSlot = Slot;
    type VMMemorySlice = ReferenceSlice;
    const MIN_ALIGNMENT: usize = WORD;
    const MAX_ALIGNMENT: usize = MAX_ALIGNMENT;
    const USE_ALLOCATION_OFFSET: bool = false;
    const ALLOC_END_ALIGNMENT: usize = WORD;
}

impl ObjectModel<Binding> for Model {
    const GLOBAL_LOG_BIT_SPEC: vm::VMGlobalLogBitSpec = vm::VMGlobalLogBitSpec::side_first();
    const LOCAL_FORWARDING_POINTER_SPEC: vm::VMLocalForwardingPointerSpec =
        vm::VMLocalForwardingPointerSpec::in_header(0);
    const LOCAL_FORWARDING_BITS_SPEC: vm::VMLocalForwardingBitsSpec =
        vm::VMLocalForwardingBitsSpec::side_first();
    const LOCAL_MARK_BIT_SPEC: vm::VMLocalMarkBitSpec =
        vm::VMLocalMarkBitSpec::side_after(Self::LOCAL_FORWARDING_BITS_SPEC.as_spec());
    const LOCAL_PINNING_BIT_SPEC: vm::VMLocalPinningBitSpec =
        vm::VMLocalPinningBitSpec::side_after(Self::LOCAL_MARK_BIT_SPEC.as_spec());
    const LOCAL_LOS_MARK_NURSERY_SPEC: vm::VMLocalLOSMarkNurserySpec =
        vm::VMLocalLOSMarkNurserySpec::side_after(Self::LOCAL_PINNING_BIT_SPEC.as_spec());
    const NEED_VO_BITS_DURING_TRACING: bool = true;
    const OBJECT_REF_OFFSET_LOWER_BOUND: isize = 0;
    const UNIFIED_OBJECT_REFERENCE_ADDRESS: bool = true;

    fn copy(
        from: ObjectReference,
        semantics: CopySemantics,
        context: &mut GCWorkerCopyContext<Binding>,
    ) -> ObjectReference {
        let bytes = Self::get_current_size(from);
        let destination = context.alloc_copy(from, bytes, MAX_ALIGNMENT, 0, semantics);
        let to = Self::get_reference_when_copied_to(from, destination);
        Self::copy_to(from, to, destination);
        context.post_copy(to, bytes, semantics);
        gc().counters
            .copied_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        to
    }
    fn copy_to(from: ObjectReference, _to: ObjectReference, region: Address) -> Address {
        let bytes = Self::get_current_size(from);
        // Forwarding is claimed by MMTk. The reserved forwarding word may be
        // concurrently inspected, so never copy it as ordinary payload bytes.
        unsafe {
            ptr::write(region.to_mut_ptr::<AtomicUsize>(), AtomicUsize::new(0));
            ptr::copy_nonoverlapping(
                (from.to_raw_address() + WORD).to_ptr::<u8>(),
                (region + WORD).to_mut_ptr::<u8>(),
                bytes - WORD,
            );
        }
        region + bytes
    }
    fn get_reference_when_copied_to(_from: ObjectReference, to: Address) -> ObjectReference {
        ObjectReference::from_raw_address(to).expect("null copy destination")
    }
    fn get_current_size(object: ObjectReference) -> usize {
        unsafe { Object(object).header().bytes }
    }
    fn get_size_when_copied(object: ObjectReference) -> usize {
        Self::get_current_size(object)
    }
    fn get_align_when_copied(_object: ObjectReference) -> usize {
        MAX_ALIGNMENT
    }
    fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
        0
    }
    fn get_type_descriptor(_object: ObjectReference) -> &'static [i8] {
        &[0]
    }
    fn ref_to_object_start(object: ObjectReference) -> Address {
        object.to_raw_address()
    }
    fn ref_to_header(object: ObjectReference) -> Address {
        object.to_raw_address()
    }
    fn dump_object(object: ObjectReference) {
        let header = unsafe { Object(object).header() };
        eprintln!(
            "{} @ {:?}: {} bytes",
            header.descriptor.name, object, header.bytes
        );
    }
}

impl vm::Scanning<Binding> for Scanning {
    fn scan_object<SV: vm::SlotVisitor<Slot>>(
        _tls: VMWorkerThread,
        object: ObjectReference,
        visitor: &mut SV,
    ) {
        trace(Object(object), &mut |slot| visitor.visit_slot(slot));
    }
    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        mutator: &'static mut mmtk::Mutator<Binding>,
        mut factory: impl RootsWorkFactory<Slot>,
    ) {
        let record = unsafe { record_from_token(mutator.mutator_tls) };
        emit_registry(&mut factory, &record.roots);
    }
    fn scan_vm_specific_roots(_tls: VMWorkerThread, mut factory: impl RootsWorkFactory<Slot>) {
        emit_registry(&mut factory, &gc().roots);
        gc().providers.lock().retain(|weak| {
            if let Some(provider) = weak.upgrade() {
                let mut slots = Vec::new();
                let mut pinned = Vec::new();
                provider.visit(&mut RootVisitor {
                    slots: &mut slots,
                    pinned: &mut pinned,
                });
                if !slots.is_empty() {
                    factory.create_process_roots_work(slots);
                }
                if !pinned.is_empty() {
                    factory.create_process_pinning_roots_work(pinned);
                }
                gc().keep_providers.lock().push(provider);
                true
            } else {
                false
            }
        });
    }
    fn notify_initial_thread_scan_complete(_partial: bool, _tls: VMWorkerThread) {}
    fn supports_return_barrier() -> bool {
        false
    }
    fn prepare_for_roots_re_scanning() {}

    fn process_weak_refs(
        worker: &mut mmtk::scheduler::GCWorker<Binding>,
        context: impl ObjectTracerContext<Binding>,
    ) -> bool {
        let mut tracked = gc().tracked.lock();
        let mut discovered = Vec::new();
        for entry in tracked.iter_mut() {
            if !entry.object.0.is_reachable()
                && !entry.finalized
                && unsafe { entry.object.header().descriptor.finalizer.is_some() }
            {
                entry.finalized = true;
                discovered.push(entry as *mut Tracked);
            }
        }
        if !discovered.is_empty() {
            context.with_tracer(worker, |tracer| {
                for entry in discovered {
                    let entry = unsafe { &mut *entry };
                    entry.object = Object(tracer.trace_object(entry.object.0));
                    let storage = RootStorage::new(1, RootKind::Precise);
                    storage.words[0].store(entry.object.address(), Ordering::Relaxed);
                    gc().roots.lock().push(Arc::downgrade(&storage));
                    gc().pending.lock().push(storage);
                }
            });
            // Finish tracing finalizer reachability before clearing weak handles
            // or releasing native resources owned by their descendants.
            return true;
        }
        tracked.retain_mut(|entry| {
            if entry.object.0.is_reachable() {
                entry.object = Object(
                    entry
                        .object
                        .0
                        .get_forwarded_object()
                        .unwrap_or(entry.object.0),
                );
                true
            } else {
                let header = unsafe { entry.object.header() };
                let external = header.external_bytes.load(Ordering::Relaxed);
                if let Some(cleanup) = header.descriptor.cleanup {
                    unsafe {
                        cleanup(entry.object);
                    }
                }
                gc().counters
                    .external_bytes
                    .fetch_sub(external, Ordering::Relaxed);
                false
            }
        });
        drop(tracked);
        gc().weak.lock().retain(|weak| {
            if let Some(cell) = weak.upgrade() {
                let address = cell.load(Ordering::Relaxed);
                if address != 0 {
                    let object = unsafe { Object::from_address(address) }.0;
                    let address = if object.is_reachable() {
                        object
                            .get_forwarded_object()
                            .unwrap_or(object)
                            .to_raw_address()
                            .as_usize()
                    } else {
                        0
                    };
                    cell.store(address, Ordering::Relaxed);
                }
                true
            } else {
                false
            }
        });
        false
    }
}

unsafe fn record_from_token(token: VMMutatorThread) -> &'static ThreadRecord {
    assert!(!token.0.0.is_null());
    unsafe { &*token.0.0.to_address().to_ptr::<ThreadRecord>() }
}

impl vm::Collection<Binding> for Collection {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mut visitor: F)
    where
        F: FnMut(&'static mut mmtk::Mutator<Binding>),
    {
        let gc = gc();
        let start = Instant::now();
        let mut world = gc.world.lock();
        assert!(!world.active);
        world.active = true;
        world.epoch += 1;
        gc.requested.store(true, Ordering::Release);
        while world
            .threads
            .iter()
            .any(|entry| entry.state == ThreadState::Running)
        {
            gc.changed.wait(&mut world);
        }
        gc.counters
            .rendezvous_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        world.started = Some(Instant::now());
        let records: Vec<_> = world.threads.iter().map(|e| e.record.clone()).collect();
        drop(world);
        for record in records {
            let pointer = unsafe { record.context_ptr() };
            // World stays active until all GC work ends. No context is touched
            // by a mutator or destroyed while MMTk owns this reference.
            visitor(unsafe { &mut *pointer });
        }
    }
    fn resume_mutators(_tls: VMWorkerThread) {
        let gc = gc();
        // Work packets no longer hold slot pointers when this callback runs.
        gc.keep_roots.lock().clear();
        gc.keep_providers.lock().clear();
        let mut world = gc.world.lock();
        assert!(world.active);
        if let Some(start) = world.started.take() {
            gc.counters
                .pause_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        gc.external_at_gc.store(
            gc.counters.external_bytes.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        gc.counters.collections.fetch_add(1, Ordering::Relaxed);
        world.completed = world.epoch;
        world.active = false;
        gc.requested.store(false, Ordering::Release);
        gc.changed.notify_all();
    }
    fn block_for_gc(token: VMMutatorThread) {
        gc().wait(unsafe { record_from_token(token) }, true);
    }
    fn spawn_gc_thread(_parent: VMThread, context: vm::GCThreadContext<Binding>) {
        std::thread::Builder::new()
            .name("stickyimmix-worker".into())
            .spawn(move || {
                let token = Box::new(0u8);
                let tls = VMWorkerThread(VMThread(OpaquePointer::from_address(Address::from_ptr(
                    &*token,
                ))));
                match context {
                    vm::GCThreadContext::Worker(worker) => worker.run(tls, gc().mmtk),
                }
            })
            .expect("cannot start GC worker");
    }
    fn vm_live_bytes() -> usize {
        gc().counters.external_bytes.load(Ordering::Relaxed)
    }
}

impl vm::ActivePlan<Binding> for ActivePlan {
    fn is_mutator(tls: VMThread) -> bool {
        gc().world
            .lock()
            .threads
            .iter()
            .any(|e| e.record.token().0 == tls)
    }
    fn mutator(tls: VMMutatorThread) -> &'static mut mmtk::Mutator<Binding> {
        unsafe { &mut *record_from_token(tls).context_ptr() }
    }
    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut mmtk::Mutator<Binding>> + 'a> {
        let world = gc().world.lock();
        assert!(
            world.active
                && world
                    .threads
                    .iter()
                    .all(|e| e.state != ThreadState::Running)
        );
        let pointers: Vec<_> = world
            .threads
            .iter()
            .map(|e| unsafe { e.record.context_ptr() })
            .collect();
        Box::new(pointers.into_iter().map(|pointer| unsafe { &mut *pointer }))
    }
    fn number_of_mutators() -> usize {
        gc().world.lock().threads.len()
    }
}

// Weak references and finalization use Scanning::process_weak_refs exclusively.
// No legacy MMTk reference objects are registered by this binding.
impl vm::ReferenceGlue<Binding> for References {
    type FinalizableType = ObjectReference;
    fn clear_referent(_object: ObjectReference) {
        unreachable!("legacy reference processor is disabled")
    }
    fn get_referent(_object: ObjectReference) -> Option<ObjectReference> {
        unreachable!("legacy reference processor is disabled")
    }
    fn set_referent(_object: ObjectReference, _referent: ObjectReference) {
        unreachable!("legacy reference processor is disabled")
    }
    fn enqueue_references(objects: &[ObjectReference], _tls: VMWorkerThread) {
        assert!(objects.is_empty());
    }
}
