//! Wren tracing JIT.
//!
//! This module connects the register interpreter to the backend-agnostic trace IR
//! (`wren_trace_ir.rs`) and lowers optimized traces to Cranelift IR.
//!
//! The implementation deliberately separates four concerns:
//!
//! 1. `TraceRuntime` is the very small adapter the interpreter implements.
//!    `wren_tracingjit` never reaches into private `Frame`, `FiberState` or heap
//!    structures directly.
//! 2. `TraceRecorder` observes exactly one execution of original (unfused)
//!    bytecode and builds a linear SSA `Trace`.
//! 3. `CraneliftTraceCompiler` lowers that trace to native code. Wren architectural
//!    registers remain virtual while native code runs.
//! 4. `TracingJit` owns hotness, compiled-code lifetime, side-exit accounting and
//!    future trace-stitching metadata.
//!
//! Important ABI property:
//!
//! Native traces receive one pointer, `*mut TraceActivation`.  The structure
//! contains pointers to the current Wren register banks plus a compact side-exit
//! result area. Cold Cranelift side-exit blocks materialize *only* values needed
//! by the failing exit into that area, return to Rust, and Rust writes the sparse
//! snapshot back to the interpreter frame.
//!
//! Consequently, a loop such as:
//!
//! ```text
//!   sum = sum + i
//!   i   = i + 1
//! ```
//!
//! can keep `sum` and `i` in native registers for the whole native loop.  There is
//! no per-iteration store back to Wren R-registers.
//!
//! Numeric/control-flow operations lower directly to Cranelift. Non-GC heap/module/
//! container operations may use synchronized runtime leaf calls that preserve unrelated
//! SSA values across the call. Full runtime calls still form a memory epoch. Calls which
//! change frames/fibers terminate a trace AFTER executing the opcode; the manager can
//! compile the continuation.
//! No architectural register pointer or uncommitted GC root crosses such a call.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use hashbrown::{HashMap, HashSet};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::mem::offset_of;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

mod inlining;
#[cfg(test)]
mod heap_proof_tests;

use cranelift::codegen::ir::{Type as ClifType, UserFuncName};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{default_libcall_names, FuncId as ClifFuncId, Module};
use cranelift::prelude::*;

use crate::wren_bytecode::{
    instruction_word_len, ArgReg, BytecodeFunction, BytecodePc, BytecodeProgram,
    Opcode, Reg, RegisterKind, RegisterWord, ReturnReg,
    SpecialReg,
};
use crate::wren_ir::{
    CmpKind, Constant, F64BinOp, F64UnaryOp, FuncId, IntBinOp, IntUnaryOp, MethodKind, Type,
};
use crate::wren_trace_ir::{
    verify_trace, ExitReason, ExitTarget, FrameDisposition, FrameSlot,
    SnapshotValue, SideExit, SideExitId, Trace, TraceAnchor, TraceClass,
    TraceConstant, TraceHeapObjectKind, TraceId, TraceInst, TraceKind, TraceLocation,
    TraceRep, TraceSwitchGuard, TraceMathOp, TraceTerminator, TraceType, TraceValueId,
    TraceValueOrigin,
};

// =============================================================================
// Wren NaN-box representation constants
// =============================================================================
//
// These deliberately mirror `wren_interpreter::Value`. Keeping the bit-level
// constants here avoids making the Trace IR depend on the interpreter module and
// lets native guards reproduce `Value::is_num()` / Wren truthiness exactly.

const TAG_BASE: u64 = 0x7ffc_0000_0000_0000;
const TAG_MASK: u64 = 0x7ffc_0000_0000_0000;
const TAG_NULL: u64 = TAG_BASE | 0x01;
const TAG_FALSE: u64 = TAG_BASE | 0x02;
const TAG_TRUE: u64 = TAG_BASE | 0x03;
const TAG_HEAP: u64 = 0xfffc_0000_0000_0000;
const TAG_HEAP_MASK: u64 = 0xffff_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

#[inline]
fn word_is_num(word: RegisterWord) -> bool {
    (word.0 & TAG_MASK) != TAG_BASE && (word.0 & TAG_HEAP_MASK) != TAG_HEAP
}

#[inline]
fn word_num(word: RegisterWord) -> Option<f64> {
    word_is_num(word).then(|| f64::from_bits(word.0))
}

#[inline]
fn word_exact_i32(word: RegisterWord) -> Option<i32> {
    let number = word_num(word)?;
    // Preserve Wren/IEEE-754 signed-zero semantics. Integer speculation is only
    // valid when converting to i32 and back produces the identical numeric value.
    if number == 0.0 && number.to_bits() != 0.0f64.to_bits() {
        return None;
    }
    if !number.is_finite()
        || number.fract() != 0.0
        || number < i32::MIN as f64
        || number > i32::MAX as f64
    {
        return None;
    }
    Some(number as i32)
}

#[inline]
fn word_is_falsey(word: RegisterWord) -> bool {
    word.0 == TAG_FALSE || word.0 == TAG_NULL
}

#[inline]
fn int_word(value: i32) -> RegisterWord {
    RegisterWord(value as u32 as u64)
}

#[inline]
fn bool_word(value: bool) -> RegisterWord {
    RegisterWord(if value { TAG_TRUE } else { TAG_FALSE })
}

#[inline]
fn num_word(value: f64) -> RegisterWord {
    RegisterWord(if value.is_nan() {
        CANONICAL_NAN
    } else {
        value.to_bits()
    })
}

// Scalar-only helpers: these cannot allocate, access VM state or trigger GC.
extern "C" fn native_frem(lhs: f64, rhs: f64) -> f64 { lhs % rhs }
extern "C" fn native_round(value: f64) -> f64 { value.round() }
extern "C" fn native_sin(value: f64) -> f64 { value.sin() }
extern "C" fn native_cos(value: f64) -> f64 { value.cos() }
extern "C" fn native_tan(value: f64) -> f64 { value.tan() }
extern "C" fn native_log(value: f64) -> f64 { value.ln() }
extern "C" fn native_exp(value: f64) -> f64 { value.exp() }
unsafe extern "C" fn noop_heap_write_barrier(_: *mut c_void, _: usize) {}

// =============================================================================
// Native trace ABI
// =============================================================================

pub const TRACE_RESULT_CONTINUE: u32 = 0;
pub const TRACE_RESULT_SIDE_EXIT: u32 = 1;
pub const TRACE_RESULT_RETURNED: u32 = 2;
pub const TRACE_RESULT_LINK: u32 = 3;
pub const TRACE_RESULT_RUNTIME: u32 = 4;
/// High bit in the runtime-helper `next` operand. It requests call-to-return
/// execution instead of exactly one bytecode step. Bytecode PCs are u32 word
/// offsets and the verifier never emits code streams approaching this boundary.
pub const TRACE_RUNTIME_CALL_FLAG: u32 = 1 << 31;

/// Maximum number of distinct dynamic values one side-exit snapshot may
/// describe. Native exits now commit these values directly to architectural
/// register banks, so this is a structural verifier limit rather than a large
/// per-entry TraceActivation buffer.
pub const MAX_TRACE_EXIT_VALUES: usize = 768;

/// Handle-table geometry shared with the interpreter. A live heap Value can only
/// reference an already-installed chunk, and chunks are never relocated.
pub const TRACE_HEAP_CHUNK_BITS: u32 = 10;
pub const TRACE_HEAP_CHUNK_SIZE: usize = 1usize << TRACE_HEAP_CHUNK_BITS;
pub const TRACE_HEAP_CHUNK_MASK: u32 = (TRACE_HEAP_CHUNK_SIZE as u32) - 1;
pub const TRACE_HEAP_KIND_MASK: u32 = 0xff;
pub const TRACE_HEAP_METADATA_SHIFT: u32 = 8;
pub const TRACE_HEAP_METADATA_MAX: u32 = 0x00ff_ffff;
pub const TRACE_MAP_DICTIONARY_SHAPE: u32 = TRACE_HEAP_METADATA_MAX;

/// Promise state is packed into the high metadata bits of TraceHeapEntry::kind.
pub const TRACE_PROMISE_PENDING: u32 = 1;
pub const TRACE_PROMISE_FULFILLED: u32 = 2;
pub const TRACE_PROMISE_REJECTED: u32 = 3;
pub const TRACE_PROMISE_CANCELLED: u32 = 4;

/// Stable object kinds exposed to generated code. This is deliberately separate
/// from the Rust `HeapObject` enum: Cranelift must never depend on Rust enum layout.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceHeapKind {
    Invalid = 0,
    String = 1,
    List = 2,
    Map = 3,
    Cell = 4,
    Instance = 5,
    Promise = 6,
}

/// Stable JIT-visible sidecar for one Wren heap handle.
///
/// `address` is the moving MMTk object address and is rewritten by GC through the
/// existing precise handle slot. `data` points only at stable external allocations
/// (Box/Vec/Arc payloads), never into the moving GC object. `len` is the logical
/// length used by fast paths. `receiver_class` is the immutable semantic class key
/// (zero means no Wren class). All fields are atomics so the GC/handle machinery can retain its
/// existing shared handle-table representation.
#[repr(C)]
pub struct TraceHeapEntry {
    pub(crate) address: AtomicUsize,
    pub(crate) next_free: AtomicU32,
    /// Packed metadata: low 8 bits are TraceHeapKind. The upper 24 bits are a
    /// type-specific payload: ShapeId for shaped Maps, UTF-8 byte length for
    /// Strings, and settlement state for Promises. Packing avoids extra sidecar
    /// fields for small type-specific metadata.
    pub(crate) kind: AtomicU32,
    pub(crate) receiver_class: AtomicU64,
    pub(crate) data: AtomicUsize,
    pub(crate) len: AtomicUsize,
    /// Settled Promise payload. Promise state itself is packed in `kind` metadata.
    pub(crate) promise_value: AtomicU64,
}

impl TraceHeapEntry {
    pub(crate) fn new(free_none: u32) -> Self {
        Self {
            address: AtomicUsize::new(0),
            next_free: AtomicU32::new(free_none),
            kind: AtomicU32::new(TraceHeapKind::Invalid as u32),
            receiver_class: AtomicU64::new(0),
            data: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            promise_value: AtomicU64::new(0),
        }
    }

    #[inline]
    pub(crate) fn clear_jit_view(&self) {
        self.kind.store(TraceHeapKind::Invalid as u32, Ordering::Relaxed);
        self.receiver_class.store(0, Ordering::Relaxed);
        self.data.store(0, Ordering::Relaxed);
        self.len.store(0, Ordering::Relaxed);
        self.promise_value.store(0, Ordering::Relaxed);
    }
}

pub type HeapWriteBarrierFn = unsafe extern "C" fn(*mut c_void, usize);

#[repr(C, align(16))]
pub struct TraceActivation {
    pub method_epoch: u64,
    /// Base of the current Wren frame's contiguous GPR bank.
    pub general: *mut RegisterWord,
    pub arguments: *mut RegisterWord,
    pub returns: *mut RegisterWord,
    pub special: *mut RegisterWord,

    /// Interpreter pointer/context reserved for later runtime-helper calls.
    pub runtime: *mut c_void,
    pub execute: RuntimeTraceFn,
    /// Stable handle-table chunk array used by native heap fast paths.
    pub heap_chunks: *const *mut TraceHeapEntry,
    /// Non-allocating StickyImmix post-barrier slow path.
    pub heap_write_barrier: HeapWriteBarrierFn,

    /// Dense native entry table and per-trace link-table pointers. The Cranelift
    /// dispatcher uses these to follow stitched trace-tree edges without returning
    /// to the Rust tiering manager. They remain stable for one dispatcher call.
    pub trace_entries: *const usize,
    pub trace_link_tables: *const usize,
    pub current_links: *const u32,
    pub trace_entries_len: u32,
    pub next_trace: u32,
    /// Link transitions left before the native dispatcher cooperatively yields.
    /// u32::MAX disables dispatcher-level yielding.
    pub dispatch_links_remaining: u32,

    /// Written only by native side exits.
    pub exit_id: u32,
    pub result_code: u32,
    /// Number of complete native loop iterations before the side exit. This is
    /// used only by the manager's anti-thrashing policy and is not architectural
    /// VM state.
    pub native_iterations: u32,
}

impl TraceActivation {
    pub fn new(inputs: TraceActivationInputs) -> Self {
        Self {
            method_epoch: 0,
            general: inputs.general,
            arguments: inputs.arguments,
            returns: inputs.returns,
            special: inputs.special,
            runtime: inputs.runtime,
            execute: inputs.execute,
            heap_chunks: inputs.heap_chunks,
            heap_write_barrier: inputs.heap_write_barrier,
            trace_entries: std::ptr::null(),
            trace_link_tables: std::ptr::null(),
            current_links: std::ptr::null(),
            trace_entries_len: 0,
            next_trace: u32::MAX,
            dispatch_links_remaining: u32::MAX,
            exit_id: u32::MAX,
            result_code: TRACE_RESULT_CONTINUE,
            native_iterations: 0,
        }
    }

    #[inline]
    pub fn reset_result(&mut self) {
        self.exit_id = u32::MAX;
        self.result_code = TRACE_RESULT_CONTINUE;
        self.native_iterations = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TraceActivationInputs {
    pub general: *mut RegisterWord,
    pub arguments: *mut RegisterWord,
    pub returns: *mut RegisterWord,
    pub special: *mut RegisterWord,
    pub runtime: *mut c_void,
    pub execute: RuntimeTraceFn,
    pub heap_chunks: *const *mut TraceHeapEntry,
    pub heap_write_barrier: HeapWriteBarrierFn,
}

pub type NativeTraceFn = unsafe extern "C" fn(*mut TraceActivation) -> u32;
pub type RuntimeTraceFn = unsafe extern "C" fn(*mut TraceActivation, u32, u32, u32) -> u32;

// =============================================================================
// Interpreter adapter
// =============================================================================

/// Result of executing exactly one original bytecode instruction while recording.
///
/// "Original" matters: the threaded interpreter may fuse two bytecodes into a
/// superinstruction, but a recorder needs the logical instruction boundary.
#[derive(Clone, Copy, Debug)]
pub struct TraceStep {
    pub before: TraceLocation,
    pub after: TraceLocation,
    /// True when the step pushed/popped/switched frames or fibers.
    pub frame_changed: bool,
    /// False if execution returned/suspended/failed instead of remaining runnable.
    pub running: bool,
}

impl TraceStep {
    pub const fn simple(before: TraceLocation, after: TraceLocation) -> Self {
        Self {
            before,
            after,
            frame_changed: false,
            running: true,
        }
    }
}

/// Runtime classification of a tagged value.
///
/// The recorder only requires heap classification when it wants to specialize a
/// heap operation. The initial numeric tier can classify Null/Bool/Num from bits
/// without asking the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedClass {
    Known(TraceClass),
    UnknownHeap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapedMapAccess {
    pub shape: u32,
    pub slot: u16,
    /// Canonical shape-key bytes. Recording copies them into Trace IR; generated
    /// code never embeds the recyclable heap handle of the observed String.
    pub key_bytes: Vec<u8>,
}

/// Thin interface implemented by `wren_interpreter::Interpreter`.
///
/// Keeping this trait here makes the JIT independently testable. The interpreter
/// implementation should:
///
/// - expose pointers for the *current* frame in `trace_activation_inputs`;
/// - execute `trace_execute_unfused_step` using the original opcode handler, not
///   an installed superinstruction;
/// - keep activation pointers stable between synchronized runtime boundaries.
///   Non-GC leaf calls preserve them; full/runtime-call boundaries may move the
///   register stack or heap and must refresh `TraceActivation` before native code
///   resumes.
pub trait TraceRuntime {
    type RuntimeError: Error + Send + Sync + 'static;

    fn trace_program(&self) -> &BytecodeProgram;

    /// Only specialize a numeric Invoke when its actual method target has the
    /// standard semantics. A method name alone is insufficient for host bindings.
    fn trace_standard_numeric_method(&self, _function: FuncId, _cache: u32) -> bool { false }

    fn trace_method_epoch(&self) -> u64 { 0 }
    /// Actual bytecode method selected for this receiver, plus up to four
    /// observed receiver classes sharing that target. The native trace must
    /// guard both the class set and method-table epoch against rebinding.
    fn trace_inline_target(&self, _function: FuncId, _cache: u32, _receiver: RegisterWord) -> Option<(FuncId, Vec<u64>)> { None }

    /// Resolve a GuardClass descriptor to stable runtime receiver-class keys.
    fn trace_class_guard(&self, _function: FuncId, _guard: u32) -> Option<Vec<u64>> { None }
    fn trace_receiver_class(&self, _value: RegisterWord) -> Option<u64> { None }
    fn trace_is_loop_header(&self, _location: TraceLocation) -> bool { false }
    /// Field slots are relocated when a class is linked to its superclass.
    fn trace_field_slot(&self, _location: TraceLocation, local_slot: u32) -> u32 { local_slot }

    /// Stable address of the module-global slot referenced by this bytecode.
    /// Module variable arrays are fixed-size for the lifetime of an interpreter.
    fn trace_module_slot_address(&self, _location: TraceLocation) -> Option<usize> { None }

    /// Observe one class static-field access. Returns the exact receiver-class
    /// key of the class object plus the lifetime-stable address of the slot.
    fn trace_static_field_access(
        &self,
        _class: RegisterWord,
        _slot: u32,
    ) -> Option<(u64, usize)> {
        None
    }

    /// Observe the class operand of IsInstance. The first element is the exact
    /// receiver-class key of the class object; the second contains every finalized
    /// instance/foreign receiver-class key accepted by that class, and the third
    /// contains every finalized instance/foreign receiver-class key currently known.
    /// Unknown future dynamic classes therefore side-exit instead of reusing a stale
    /// negative hierarchy observation.
    fn trace_is_instance_observation(
        &self,
        _class: RegisterWord,
    ) -> Option<(u64, Vec<u64>, Vec<u64>)> {
        None
    }

    fn trace_current_location(&self) -> TraceLocation;

    fn trace_read_slot(&self, slot: FrameSlot) -> Result<RegisterWord, Self::RuntimeError>;

    fn trace_write_slot(
        &mut self,
        slot: FrameSlot,
        value: RegisterWord,
    ) -> Result<(), Self::RuntimeError>;

    fn trace_set_location(
        &mut self,
        location: TraceLocation,
    ) -> Result<(), Self::RuntimeError>;

    fn trace_execute_unfused_step(&mut self) -> Result<TraceStep, Self::RuntimeError>;

    /// Execute the call-like opcode at the current location and, for ordinary
    /// synchronous bytecode calls, continue the baseline interpreter until the
    /// callee returns to the same caller frame. Implementations must stop early
    /// if execution suspends/fails/switches fiber or otherwise cannot resume at
    /// the requested continuation.
    fn trace_execute_call_to_return(&mut self) -> Result<TraceStep, Self::RuntimeError>;

    fn trace_activation_inputs(
        &mut self,
    ) -> Result<TraceActivationInputs, Self::RuntimeError>;

    /// Observe a Map/key pair without mutating VM state. Returning Some allows
    /// the recorder to specialize this site to shape+slot native code.
    fn trace_shaped_map_access(
        &self,
        _map: RegisterWord,
        _key: RegisterWord,
    ) -> Option<ShapedMapAccess> {
        None
    }

    /// Observe an already-fulfilled Promise while recording `Await`. Pending and
    /// rejected Promises remain runtime boundaries because they can suspend/fail.
    fn trace_fulfilled_promise(
        &self,
        _value: RegisterWord,
    ) -> Result<Option<RegisterWord>, Self::RuntimeError> {
        Ok(None)
    }

    /// Resolve a heap object's semantic class if the recorder asks for one.
    fn trace_observe_heap_class(
        &self,
        _value: RegisterWord,
    ) -> Result<ObservedClass, Self::RuntimeError> {
        Ok(ObservedClass::UnknownHeap)
    }

    /// Materialize constants whose physical value is runtime-owned (notably
    /// interned strings and builtin class objects).
    fn trace_materialize_constant(
        &self,
        constant: TraceConstant,
    ) -> Result<RegisterWord, Self::RuntimeError> {
        match constant {
            TraceConstant::Null => Ok(RegisterWord(TAG_NULL)),
            TraceConstant::Bool(value) => Ok(bool_word(value)),
            TraceConstant::NumBits(bits) => Ok(RegisterWord(bits)),
            TraceConstant::Int32(value) => Ok(int_word(value)),
            TraceConstant::F64Bits(bits) => Ok(RegisterWord(bits)),
            TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                // Runtime-owned heap values need the interpreter's interning
                // tables. An implementation that allows them in traces should
                // override this method.
                Err(self.trace_unsupported_constant_error(constant))
            }
        }
    }

    /// This default is only used to make the previous method ergonomic. Runtime
    /// implementations should normally override `trace_materialize_constant`
    /// instead of this method.
    fn trace_unsupported_constant_error(
        &self,
        constant: TraceConstant,
    ) -> Self::RuntimeError;
}

// =============================================================================
// Configuration / diagnostics
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeOptLevel {
    None,
    Speed,
    SpeedAndSize,
}

impl NativeOptLevel {
    #[inline]
    const fn as_setting(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Speed => "speed",
            Self::SpeedAndSize => "speed_and_size",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TracingJitOptions {
    /// Number of visits to a loop header before starting a recording.
    pub hot_loop_threshold: u32,
    /// Visits to a function entry/call continuation before recording. Zero disables this tier.
    pub hot_entry_threshold: u32,
    /// Do not install acyclic entry traces shorter than this. Tiny forwarding
    /// methods usually lose to the token-threaded baseline until native call
    /// linking/inlining removes the transition itself.
    pub min_entry_trace_bytecodes: u32,
    /// Hard recorder limit for one loop iteration.
    pub max_trace_steps: usize,
    /// Repeated recording/lowering failures before a loop is permanently ignored.
    pub max_failures_before_blacklist: u16,
    /// Start considering a side exit for a child trace after this many hits.
    pub hot_side_exit_threshold: u32,
    /// Maximum native loop iterations executed by one trace entry before a
    /// cooperative side exit. Zero disables the budget. Exhaustion resumes the
    /// interpreter at the header; the compiled trace remains available.
    pub native_loop_iteration_budget: u32,

    pub optimize_trace_ir: bool,
    /// Cranelift optimization mode for compiled traces.
    pub cranelift_opt_level: NativeOptLevel,
    /// Run Cranelift's verifier before native lowering. Recommended in development.
    pub cranelift_enable_verifier: bool,
    /// Print CLIF before native compilation and retain it for `disassembly()`.
    pub cranelift_dump_ir: bool,
    /// Ask Cranelift for post-regalloc VCode and retain/print it.
    pub cranelift_dump_disassembly: bool,

    /// Reject the generic full-state runtime fallback for unsupported opcodes.
    /// Explicitly whitelisted non-GC leaf operations use a small internal budget.
    pub no_runtime_calls_in_native_trace: bool,
}

impl Default for TracingJitOptions {
    fn default() -> Self {
        Self {
            hot_loop_threshold: 64,
            hot_entry_threshold: 64,
            min_entry_trace_bytecodes: 6,
            max_trace_steps: 4096,
            max_failures_before_blacklist: 8,
            hot_side_exit_threshold: 32,
            // Periodically return control to the interpreter so embedders retain
            // a cooperative safepoint even for a pure source-level infinite loop.
            native_loop_iteration_budget: 1 << 16,

            optimize_trace_ir: true,
            cranelift_opt_level: NativeOptLevel::Speed,
            cranelift_enable_verifier: cfg!(debug_assertions),
            cranelift_dump_ir: false,
            cranelift_dump_disassembly: false,

            // Until an opcode has a real native lowering, do not use the generic
            // full-state fallback. A small number of non-GC leaf helpers is still
            // profitable in mixed traces because unaffected SSA values stay native.
            no_runtime_calls_in_native_trace: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TracingJitStats {
    pub inlined_calls: u64,
    pub devirtualized_calls: u64,
    pub loop_header_hits: u64,
    pub entry_hits: u64,
    pub entry_traces_compiled: u64,
    pub entry_traces_too_small: u64,
    pub entry_native_entries: u64,
    pub blacklisted_entries: u64,
    pub recordings_started: u64,
    pub recordings_succeeded: u64,
    pub recordings_aborted: u64,
    pub traces_compiled: u64,
    pub native_entries: u64,
    pub side_exits: u64,
    pub native_budget_exits: u64,
    /// Total complete loop iterations executed inside native traces.
    pub native_iterations: u64,
    /// Side exits taken before enough native work was done to amortize the
    /// interpreter -> JIT -> interpreter transition.
    pub short_native_exits: u64,
    /// Temporary cooldowns after repeated short native entries.
    pub trace_cooldowns: u64,
    pub blacklisted_loops: u64,
    pub stitched_exits: u64,
    pub side_traces_compiled: u64,
    pub side_recordings_aborted: u64,
    pub recorded_bytecodes: u64,
    pub promoted_f64_loop_phis: u64,
    pub promoted_i32_loop_phis: u64,
    pub promoted_u32_loop_phis: u64,
    pub pruned_snapshot_writes: u64,
    pub eliminated_trace_guards: u64,
    pub eliminated_trace_boxes: u64,
    pub eliminated_trace_copies: u64,
    pub trace_gvn_eliminated: u64,
    pub trace_licm_hoisted: u64,
    pub trace_pre_eliminated: u64,
    pub last_abort: Option<String>,
}

#[derive(Debug)]
pub enum TracingJitError {
    Runtime(String),
    Decode {
        location: TraceLocation,
        message: String,
    },
    Record {
        location: TraceLocation,
        message: String,
    },
    TraceVerification(Vec<String>),
    CraneliftLowering {
        trace: TraceId,
        message: String,
    },
    InvalidSideExit {
        trace: TraceId,
        exit: u32,
    },
    Activation(String),
}

impl fmt::Display for TracingJitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(message) => write!(f, "trace runtime error: {message}"),
            Self::Decode { location, message } => write!(
                f,
                "trace decode error in {:?} at word {}: {message}",
                location.function, location.pc.0
            ),
            Self::Record { location, message } => write!(
                f,
                "trace recording error in {:?} at word {}: {message}",
                location.function, location.pc.0
            ),
            Self::TraceVerification(errors) => {
                writeln!(f, "trace verification failed")?;
                for error in errors {
                    writeln!(f, "  - {error}")?;
                }
                Ok(())
            }
            Self::CraneliftLowering { trace, message } => {
                write!(f, "Cranelift lowering failed for {trace:?}: {message}")
            }
            Self::InvalidSideExit { trace, exit } => {
                write!(f, "{trace:?} returned invalid side exit {exit}")
            }
            Self::Activation(message) => f.write_str(message),
        }
    }
}

impl Error for TracingJitError {}

fn runtime_error<E: Error>(error: E) -> TracingJitError {
    TracingJitError::Runtime(error.to_string())
}

#[derive(Clone, Debug)]
pub enum RecordingAbort {
    UnsupportedOpcode(Opcode),
    UnsupportedConstant,
    UnsupportedCall,
    FrameChanged,
    LeftFunction,
    PathDidNotClose,
    EnclosingLoop,
    NestedCycle(TraceLocation),
    TraceTooLong,
    RepresentationConflict,
    InterpreterStopped,
    GuardFailedWhileRecording,
}

impl fmt::Display for RecordingAbort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOpcode(op) => write!(f, "unsupported opcode {}", op.name()),
            Self::UnsupportedConstant => f.write_str("unsupported runtime-owned constant"),
            Self::UnsupportedCall => f.write_str("call cannot yet stay inside native trace"),
            Self::FrameChanged => f.write_str("bytecode step changed frame/fiber"),
            Self::LeftFunction => f.write_str("recorded path left the root function"),
            Self::PathDidNotClose => f.write_str("recorded path did not return to trace head"),
            Self::EnclosingLoop => f.write_str("recorded path took an enclosing loop backedge"),
            Self::NestedCycle(location) => write!(
                f,
                "recording encountered nested/re-entrant cycle at {:?}:{} before returning to trace head",
                location.function,
                location.pc.0,
            ),
            Self::TraceTooLong => f.write_str("trace exceeded configured instruction limit"),
            Self::RepresentationConflict => {
                f.write_str("one loop-entry register requires incompatible representations")
            }
            Self::InterpreterStopped => f.write_str("interpreter stopped while recording"),
            Self::GuardFailedWhileRecording => {
                f.write_str("speculative bytecode guard failed on recording iteration")
            }
        }
    }
}

// =============================================================================
// Decoded bytecode view used by the recorder
// =============================================================================

#[derive(Clone, Debug)]
struct RecordedOp {
    location: TraceLocation,
    opcode: Opcode,
    a: u8,
    b: u8,
    c: u8,
    x: u32,
    y: u32,
    next_pc: BytecodePc,
    args: Vec<Reg>,
}

fn decode_at(
    program: &BytecodeProgram,
    location: TraceLocation,
) -> Result<RecordedOp, TracingJitError> {
    let function = program
        .function(location.function)
        .ok_or_else(|| TracingJitError::Decode {
            location,
            message: "unknown bytecode function".to_owned(),
        })?;

    let raw_pc = location.pc.0 as usize;
    let first = *function
        .chunk
        .code
        .get(raw_pc)
        .ok_or_else(|| TracingJitError::Decode {
            location,
            message: "program counter is outside bytecode".to_owned(),
        })?;

    let opcode = first.opcode().ok_or_else(|| TracingJitError::Decode {
        location,
        message: format!("invalid opcode {}", first.0 as u8),
    })?;

    let len = instruction_word_len(first).map_err(|error| TracingJitError::Decode {
        location,
        message: error.message,
    })?;

    if raw_pc + len > function.chunk.code.len() {
        return Err(TracingJitError::Decode {
            location,
            message: "truncated bytecode instruction".to_owned(),
        });
    }

    let x = if len >= 2 {
        function.chunk.code[raw_pc + 1].0
    } else {
        0
    };
    let y = if len >= 3 {
        function.chunk.code[raw_pc + 2].0
    } else {
        0
    };

    let mut args = Vec::new();
    if matches!(
        opcode,
        Opcode::Invoke
            | Opcode::InvokeSuper
            | Opcode::CallDirect
            | Opcode::CallClosure
            | Opcode::CallPrimitive
            | Opcode::CallForeign
    ) {
        let count = first.c() as usize;
        args.reserve(count);
        for index in 0..count {
            let packed = function.chunk.code[raw_pc + 2 + index / 4].0;
            args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
        }
    } else if opcode == Opcode::MakeClosure {
        let count = first.bx() as usize;
        args.reserve(count);
        for index in 0..count {
            let packed = function.chunk.code[raw_pc + 2 + index / 4].0;
            args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
        }
    }

    Ok(RecordedOp {
        location,
        opcode,
        a: first.a(),
        b: first.b(),
        c: first.c(),
        x,
        y,
        next_pc: BytecodePc(location.pc.0 + len as u32),
        args,
    })
}

#[inline]
fn relative_target(next: BytecodePc, encoded_delta: u32) -> BytecodePc {
    let delta = encoded_delta as i32 as i64;
    BytecodePc((next.0 as i64 + delta) as u32)
}

fn deopt_resume(
    function: &BytecodeFunction,
    deopt: u32,
) -> Option<BytecodePc> {
    function
        .chunk
        .deopts
        .iter()
        .find(|point| point.id.0 == deopt)
        .map(|point| point.resume_pc)
}

// =============================================================================
// Recorder helpers
// =============================================================================

/// Keep mixed traces from degenerating into a native-to-Rust call trampoline.
/// The whitelist is intentionally small: leaf helpers are for occasional heap or
/// module operations between larger native regions, not for replacing baseline
/// dispatch one opcode at a time.
const MAX_RUNTIME_LEAF_OPS_PER_TRACE: usize = 8;

struct TraceRecorder<'a> {
    options: &'a TracingJitOptions,
    trace_id: TraceId,
    anchor: TraceLocation,
}

#[derive(Clone, Copy)]
enum RecordingMode {
    RootLoop,
    Entry,
    Side { close_at: TraceLocation, link_to: TraceId },
}

impl<'a> TraceRecorder<'a> {
    fn new(
        options: &'a TracingJitOptions,
        trace_id: TraceId,
        anchor: TraceLocation,
    ) -> Self {
        Self {
            options,
            trace_id,
            anchor,
        }
    }

    fn read_gpr(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        reg: u8,
        ty: TraceType,
        rep: TraceRep,
    ) -> Result<TraceValueId, RecordingAbort> {
        self.read_slot(builder, FrameSlot::General(Reg(reg)), ty, rep)
    }

    fn read_slot(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        slot: FrameSlot,
        ty: TraceType,
        rep: TraceRep,
    ) -> Result<TraceValueId, RecordingAbort> {
        if let Some(existing) = builder.slot_value(slot) {
            let canonical = builder.trace().canonical_value(existing);
            let (current_ty, current_rep, current_origin) = {
                let current = builder.trace().value(canonical).unwrap();
                (current.ty, current.rep, current.origin.clone())
            };

            if current_rep == rep
                && (current_ty == ty
                    || current_ty == TraceType::Any
                    || ty == TraceType::Any)
            {
                // Refine a still-generic entry value once an opcode tells us the
                // physical representation.
                if current_ty == TraceType::Any && ty != TraceType::Any {
                    let value = builder.trace_mut().value_mut(canonical).unwrap();
                    value.ty = ty;
                    value.rep = rep;
                }
                return Ok(existing);
            }

            // Entry values created through a bit-preserving Move may start as
            // Any/Tagged. If no opcode has specialized them yet, refine in-place.
            if matches!(
                current_origin,
                TraceValueOrigin::EntrySlot { .. }
            ) && current_ty == TraceType::Any
            {
                let value = builder.trace_mut().value_mut(canonical).unwrap();
                value.ty = ty;
                value.rep = rep;
                return Ok(existing);
            }

            // Register moves preserve physical bits, including internal i32/f64
            // words. A representation bridge must not numerically box them.
            return Ok(builder.emit1(TraceInst::Reinterpret { value: existing }, ty, rep));
        }

        Ok(builder.read_slot(slot, ty, rep))
    }

    fn write_gpr(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        reg: u8,
        value: TraceValueId,
    ) {
        builder.write_slot(FrameSlot::General(Reg(reg)), value);
    }

    #[inline]
    fn guard_heap_object(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        value: TraceValueId,
        kind: TraceHeapObjectKind,
        exit: SideExitId,
    ) -> TraceValueId {
        builder.emit1(
            TraceInst::GuardHeapObject { value, kind, exit },
            TraceType::Any,
            TraceRep::RawPtr,
        )
    }

    fn tagged_bool(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        condition: TraceValueId,
    ) -> TraceValueId {
        let t = builder.constant(TraceConstant::Bool(true));
        let f = builder.constant(TraceConstant::Bool(false));
        builder.emit1(
            TraceInst::Select {
                condition,
                if_true: t,
                if_false: f,
            },
            TraceType::Bool,
            TraceRep::Tagged,
        )
    }

    /// Recover the raw 0/1 comparison feeding a tagged Boolean produced by
    /// `tagged_bool`. Branch guards can use that value directly instead of
    /// re-decoding the NaN-boxed true/false representation.
    fn raw_recorded_bool(
        &self,
        builder: &crate::wren_trace_ir::TraceBuilder,
        value: TraceValueId,
    ) -> Option<TraceValueId> {
        let trace = builder.trace();
        let value = trace.canonical_value(value);
        let node = match &trace.value(value)?.origin {
            TraceValueOrigin::InstResult { node } => *node,
            _ => return None,
        };
        let TraceInst::Select {
            condition,
            if_true,
            if_false,
        } = &trace.nodes.get(node)?.kind
        else {
            return None;
        };
        let condition = *condition;
        let if_true = *if_true;
        let if_false = *if_false;

        let is_bool_constant = |id: TraceValueId, expected: bool| {
            let id = trace.canonical_value(id);
            matches!(
                trace.value(id).map(|value| &value.origin),
                Some(TraceValueOrigin::Constant(TraceConstant::Bool(actual))) if *actual == expected
            )
        };

        if is_bool_constant(if_true, true) && is_bool_constant(if_false, false) {
            Some(trace.canonical_value(condition))
        } else {
            None
        }
    }

    fn num_to_f64_guard(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        value: TraceValueId,
        resume: BytecodePc,
    ) -> TraceValueId {
        let exit = builder.add_side_exit(ExitReason::GuardNumber, resume);
        builder.emit1(
            TraceInst::GuardF64 { value, exit },
            TraceType::F64,
            TraceRep::F64,
        )
    }

    fn num_to_i32_guard(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        value: TraceValueId,
        resume: BytecodePc,
    ) -> TraceValueId {
        let exit = builder.add_side_exit(ExitReason::GuardInteger, resume);
        builder.emit1(
            TraceInst::GuardInt32 { value, exit },
            TraceType::Int32,
            TraceRep::Int32,
        )
    }

    fn num_to_u32_guard(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        value: TraceValueId,
        resume: BytecodePc,
    ) -> TraceValueId {
        let exit = builder.add_side_exit(ExitReason::GuardUnsignedInteger, resume);
        builder.emit1(
            TraceInst::GuardUInt32 { value, exit },
            TraceType::UInt32,
            TraceRep::UInt32,
        )
    }

    /// Trace through a VM operation that is guaranteed by the bytecode contract
    /// to neither GC nor change frames. Dirty architectural registers are committed
    /// for precise exceptional state, but successful execution keeps unrelated SSA
    /// values alive across the helper call.
    fn runtime_leaf(
        &self,
        op: &RecordedOp,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
        outputs: &[(Reg, TraceType, TraceRep)],
    ) -> Result<ExpectedStep, RecordingAbort> {
        if op.opcode.may_gc() {
            return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
        }

        let used = builder
            .trace()
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. }))
            .count();
        if used >= MAX_RUNTIME_LEAF_OPS_PER_TRACE {
            return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
        }

        builder.runtime_leaf(op.location, op.next_pc);
        for &(reg, ty, rep) in outputs {
            builder.reload_slot_after_runtime(FrameSlot::General(reg), ty, rep);
        }

        // Treat the recorded execution as a runtime step as well. On the hot
        // successful path it must advance to `next_pc`; if the opcode throws,
        // deoptimizes or otherwise changes control flow while recording, the
        // recorder ends the trace cleanly instead of reporting a path mismatch.
        Ok(ExpectedStep::Runtime { terminal: false })
    }

    fn record_num_invoke<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        function: &BytecodeFunction,
        op: &RecordedOp,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<bool, RecordingAbort> {
        if op.opcode != Opcode::Invoke {
            return Ok(false);
        }
        if !runtime.trace_standard_numeric_method(op.location.function, op.x) { return Ok(false); }

        let Some(ic) = function.chunk.inline_caches.get(op.x as usize) else {
            return Ok(false);
        };
        let Some(signature) = program.metadata.method(ic.method) else {
            return Ok(false);
        };
        let Some(name) = program.metadata.string(signature.name) else {
            return Ok(false);
        };
        // Arithmetic stays in f64, while Wren bitwise operators use the exact
        // unsigned-u32 coercion domain. Out-of-domain values side-exit to baseline.
        if !matches!(name, "+" | "-" | "*" | "/" | "%" | "<" | "<=" | ">" | ">=" | "==" | "!="
            | "~" | "&" | "|" | "^" | "<<" | ">>"
            | "abs" | "floor" | "ceil" | "sqrt" | "round" | "sin" | "cos" | "tan" | "log" | "exp"
            | "isNan" | "isInfinity" | "isInteger") { return Ok(false); }

        let receiver_word = runtime
            .trace_read_slot(FrameSlot::General(Reg(op.b)))
            .map_err(|_| RecordingAbort::UnsupportedCall)?;
        if !word_is_num(receiver_word) {
            return Ok(false);
        }

        let receiver = self.read_gpr(
            builder,
            op.b,
            TraceType::Any,
            TraceRep::Tagged,
        )?;

        let resume = op.location.pc;

        match (signature.kind, signature.arity, name) {
            (MethodKind::Getter, 0, "abs" | "floor" | "ceil" | "sqrt" | "round"
                | "sin" | "cos" | "tan" | "log" | "exp") => {
                let math_op = match name {
                    "abs" => TraceMathOp::Abs,
                    "floor" => TraceMathOp::Floor,
                    "ceil" => TraceMathOp::Ceil,
                    "sqrt" => TraceMathOp::Sqrt,
                    "round" => TraceMathOp::Round,
                    "sin" => TraceMathOp::Sin,
                    "cos" => TraceMathOp::Cos,
                    "tan" => TraceMathOp::Tan,
                    "log" => TraceMathOp::Log,
                    "exp" => TraceMathOp::Exp,
                    _ => unreachable!(),
                };
                let value = self.num_to_f64_guard(builder, receiver, resume);
                let raw = builder.emit1(TraceInst::F64Math { op: math_op, value }, TraceType::F64, TraceRep::F64);
                let boxed = builder.emit1(TraceInst::BoxF64 { value: raw }, TraceType::Num, TraceRep::Tagged);
                self.write_gpr(builder, op.a, boxed);
                return Ok(true);
            }
            (MethodKind::Getter, 0, "isNan" | "isInfinity" | "isInteger") => {
                let raw = self.num_to_f64_guard(builder, receiver, resume);
                let condition = if name == "isNan" {
                    builder.emit1(TraceInst::F64Compare { op: CmpKind::Ne, lhs: raw, rhs: raw },
                        TraceType::Bool, TraceRep::Bool)
                } else {
                    let abs = builder.emit1(TraceInst::F64Math { op: TraceMathOp::Abs, value: raw },
                        TraceType::F64, TraceRep::F64);
                    let infinity = builder.constant(TraceConstant::F64Bits(f64::INFINITY.to_bits()));
                    if name == "isInfinity" {
                        builder.emit1(TraceInst::F64Compare { op: CmpKind::Eq, lhs: abs, rhs: infinity },
                            TraceType::Bool, TraceRep::Bool)
                    } else {
                        let finite = builder.emit1(TraceInst::F64Compare { op: CmpKind::Lt, lhs: abs, rhs: infinity },
                            TraceType::Bool, TraceRep::Bool);
                        let floor = builder.emit1(TraceInst::F64Math { op: TraceMathOp::Floor, value: raw },
                            TraceType::F64, TraceRep::F64);
                        let integral = builder.emit1(TraceInst::F64Compare { op: CmpKind::Eq, lhs: raw, rhs: floor },
                            TraceType::Bool, TraceRep::Bool);
                        let false_value = builder.emit1(TraceInst::F64Compare { op: CmpKind::Lt, lhs: raw, rhs: raw },
                            TraceType::Bool, TraceRep::Bool);
                        builder.emit1(TraceInst::Select { condition: finite, if_true: integral, if_false: false_value },
                            TraceType::Bool, TraceRep::Bool)
                    }
                };
                let boxed = self.tagged_bool(builder, condition);
                self.write_gpr(builder, op.a, boxed);
                return Ok(true);
            }
            // Unary numeric methods.
            (MethodKind::Method, 0, "-") => {
                let raw = self.num_to_f64_guard(builder, receiver, resume);
                let result = builder.emit1(
                    TraceInst::F64Unary {
                        op: F64UnaryOp::Neg,
                        value: raw,
                    },
                    TraceType::F64,
                    TraceRep::F64,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxF64 { value: result },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, boxed);
                return Ok(true);
            }
            (MethodKind::Method, 0, "~") => {
                let raw = self.num_to_u32_guard(builder, receiver, resume);
                let result = builder.emit1(
                    TraceInst::IntUnary {
                        op: IntUnaryOp::BitNot,
                        value: raw,
                    },
                    TraceType::UInt32,
                    TraceRep::UInt32,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxUInt32 { value: result },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, boxed);
                return Ok(true);
            }
            _ => {}
        }

        if signature.arity != 1 || op.args.len() != 1 {
            return Ok(false);
        }

        let rhs_word = runtime
            .trace_read_slot(FrameSlot::General(op.args[0]))
            .map_err(|_| RecordingAbort::UnsupportedCall)?;
        if !word_is_num(rhs_word) {
            return Ok(false);
        }

        let rhs = self.read_gpr(
            builder,
            op.args[0].0,
            TraceType::Any,
            TraceRep::Tagged,
        )?;

        match name {
            "+" | "-" | "*" | "/" | "%" => {
                // Wren-visible Num is semantically f64. Keep dynamic Invoke
                // specialization in that representation until the Cranelift tier
                // is proven correct with mixed Tagged/Int32 loop-carried values.
                // Typed I* bytecodes still use the integer native lowering below.
                let lhs = self.num_to_f64_guard(builder, receiver, resume);
                let rhs = self.num_to_f64_guard(builder, rhs, resume);
                let bin = match name {
                    "+" => F64BinOp::Add,
                    "-" => F64BinOp::Sub,
                    "*" => F64BinOp::Mul,
                    "/" => F64BinOp::Div,
                    "%" => F64BinOp::Rem,
                    _ => unreachable!(),
                };
                let raw = builder.emit1(
                    TraceInst::F64Binary { op: bin, lhs, rhs },
                    TraceType::F64,
                    TraceRep::F64,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxF64 { value: raw },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, boxed);
                Ok(true)
            }

            "<" | "<=" | ">" | ">=" | "==" | "!=" => {
                let cmp = match name {
                    "<" => CmpKind::Lt,
                    "<=" => CmpKind::Le,
                    ">" => CmpKind::Gt,
                    ">=" => CmpKind::Ge,
                    "==" => CmpKind::Eq,
                    "!=" => CmpKind::Ne,
                    _ => unreachable!(),
                };

                // Use the source-language representation for dynamic Num
                // comparisons. This avoids changing a loop-carried Num between
                // Int32 and F64 depending on the values observed while recording.
                let lhs = self.num_to_f64_guard(builder, receiver, resume);
                let rhs = self.num_to_f64_guard(builder, rhs, resume);
                let condition = builder.emit1(
                    TraceInst::F64Compare { op: cmp, lhs, rhs },
                    TraceType::Bool,
                    TraceRep::Bool,
                );
                let boxed = self.tagged_bool(builder, condition);
                self.write_gpr(builder, op.a, boxed);
                Ok(true)
            }

            "&" | "|" | "^" | "<<" | ">>" => {
                let lhs = self.num_to_u32_guard(builder, receiver, resume);
                let rhs = self.num_to_u32_guard(builder, rhs, resume);
                let bin = match name {
                    "&" => IntBinOp::BitAnd,
                    "|" => IntBinOp::BitOr,
                    "^" => IntBinOp::BitXor,
                    "<<" => IntBinOp::Shl,
                    ">>" => IntBinOp::ShrU,
                    _ => unreachable!(),
                };
                let raw = builder.emit1(
                    TraceInst::IntBinary {
                        op: bin,
                        lhs,
                        rhs,
                    },
                    TraceType::UInt32,
                    TraceRep::UInt32,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxUInt32 { value: raw },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, boxed);
                Ok(true)
            }

            _ => Ok(false),
        }
    }

    fn translate_op<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, function: &BytecodeFunction,
        op: &RecordedOp, builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        // A speculative lowering may have emitted guards before discovering a
        // representation conflict. Roll back that attempt before using a helper.
        let checkpoint = (!self.options.no_runtime_calls_in_native_trace)
            .then(|| builder.clone());
        let attempt = self.translate_native(runtime, program, function, op, builder);
        match attempt {
            Ok(step) => Ok(step),
            Err(reason) if self.options.no_runtime_calls_in_native_trace => Err(reason),
            Err(_) => {
                *builder = checkpoint.unwrap();
                builder.set_source(op.location);
                let terminal = matches!(op.opcode,
                    Opcode::Trap | Opcode::Return | Opcode::Return2 | Opcode::Bailout
                    | Opcode::Switch | Opcode::Invoke | Opcode::InvokeSuper
                    | Opcode::CallDirect | Opcode::CallClosure | Opcode::ImportModule
                    | Opcode::FiberCall | Opcode::FiberTry | Opcode::FiberTransfer
                    | Opcode::FiberTransferError | Opcode::FiberYield | Opcode::FiberSuspend
                    | Opcode::FiberAbort | Opcode::Await);
                builder.runtime(op.location, op.next_pc, terminal);
                Ok(ExpectedStep::Runtime { terminal })
            }
        }
    }

    fn inline_or_runtime_call<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        op: &RecordedOp,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        // Inlining is preferable, but it is speculative. If the callee is too
        // large/complex (or contains an opcode the bounded inliner refuses), roll
        // back every emitted guard/node and keep the call inside the native trace
        // as a synchronized RuntimeCall instead of aborting the whole trace.
        let checkpoint = builder.clone();
        match self.inline_call(runtime, program, op, builder) {
            Ok(step) => Ok(step),
            Err(RecordingAbort::UnsupportedCall | RecordingAbort::UnsupportedOpcode(_)) => {
                *builder = checkpoint;
                builder.runtime_call(op.location, op.next_pc);
                Ok(ExpectedStep::RuntimeCall)
            }
            Err(error) => Err(error),
        }
    }

    fn translate_native<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        function: &BytecodeFunction,
        op: &RecordedOp,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        use Opcode::*;

        builder.set_source(op.location);

        let next = TraceLocation::new(op.location.function, op.next_pc);

        match op.opcode {
            Nop | OsrPoint => Ok(ExpectedStep::Exact(next)),

            Move => {
                let src = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, src);
                Ok(ExpectedStep::Exact(next))
            }

            Swap => {
                let a = self.read_gpr(
                    builder,
                    op.a,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let b = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, b);
                self.write_gpr(builder, op.b, a);
                Ok(ExpectedStep::Exact(next))
            }

            LoadNull => {
                let value = builder.constant(TraceConstant::Null);
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }
            LoadFalse => {
                let value = builder.constant(TraceConstant::Bool(false));
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }
            LoadTrue => {
                let value = builder.constant(TraceConstant::Bool(true));
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }
            LoadI8 => {
                let value = builder.constant(TraceConstant::Int32(op.b as i8 as i32));
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }
            LoadI32 => {
                let value = builder.constant(TraceConstant::Int32(op.x as i32));
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }
            LoadConst => {
                let Some(constant) = function.chunk.constants.get(op.x as usize) else {
                    return Err(RecordingAbort::UnsupportedConstant);
                };
                let value = match constant {
                    Constant::Null => builder.constant(TraceConstant::Null),
                    Constant::Bool(value) => builder.constant(TraceConstant::Bool(*value)),
                    Constant::Num(value) => {
                        builder.constant(TraceConstant::NumBits(
                            if value.is_nan() { CANONICAL_NAN } else { value.to_bits() }
                        ))
                    }
                    Constant::Int32(value) => {
                        builder.constant(TraceConstant::Int32(*value))
                    }
                    Constant::String(_) => {
                        return self.runtime_leaf(
                            op,
                            builder,
                            &[(Reg(op.a), TraceType::Any, TraceRep::Tagged)],
                        );
                    }
                };
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }

            MoveFromArg => {
                let src = self.read_slot(
                    builder,
                    FrameSlot::Argument(ArgReg(op.b)),
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, src);
                Ok(ExpectedStep::Exact(next))
            }
            MoveToArg => {
                let src = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                builder.write_slot(FrameSlot::Argument(ArgReg(op.a)), src);
                Ok(ExpectedStep::Exact(next))
            }
            MoveFromRet => {
                let src = self.read_slot(
                    builder,
                    FrameSlot::Return(ReturnReg(op.b)),
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, src);
                Ok(ExpectedStep::Exact(next))
            }
            MoveToRet => {
                let src = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                builder.write_slot(FrameSlot::Return(ReturnReg(op.a)), src);
                Ok(ExpectedStep::Exact(next))
            }
            MoveFromSpecial => {
                let Some(special) = special_reg(op.b) else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };
                let src = self.read_slot(
                    builder,
                    FrameSlot::Special(special),
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, src);
                Ok(ExpectedStep::Exact(next))
            }
            MoveToSpecial => {
                let Some(special) = special_reg(op.a) else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };
                let src = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                builder.write_slot(FrameSlot::Special(special), src);
                Ok(ExpectedStep::Exact(next))
            }

            Jump => {
                let target = TraceLocation::new(
                    op.location.function,
                    relative_target(op.next_pc, op.x),
                );
                Ok(ExpectedStep::Exact(target))
            }

            JumpIfFalse | JumpIfTrue => {
                let condition = self.read_gpr(
                    builder,
                    op.a,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.a)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let truthy = !word_is_falsey(physical);

                let branch_target = relative_target(op.next_pc, op.x);
                let taken = if op.opcode == JumpIfTrue {
                    truthy
                } else {
                    !truthy
                };
                let observed = if taken { branch_target } else { op.next_pc };
                let alternate = if taken { op.next_pc } else { branch_target };

                let exit = builder.add_side_exit(
                    if truthy {
                        ExitReason::GuardTruthy
                    } else {
                        ExitReason::GuardFalsey
                    },
                    alternate,
                );

                if let Some(raw_bool) = self.raw_recorded_bool(builder, condition) {
                    // Numeric comparisons in Wren are recorded as raw Bool ->
                    // tagged Bool. Guard the raw comparison result directly. This
                    // keeps loop-exit control flow independent from boxing/select
                    // lowering and makes the backedge condition explicit in CLIF.
                    builder.emit0(TraceInst::GuardSwitch {
                        tag: raw_bool,
                        observed: TraceSwitchGuard::Case(if truthy { 1 } else { 0 }),
                        exit,
                    });
                } else if truthy {
                    builder.emit0(TraceInst::GuardTruthy {
                        value: condition,
                        exit,
                    });
                } else {
                    builder.emit0(TraceInst::GuardFalsey {
                        value: condition,
                        exit,
                    });
                }

                Ok(ExpectedStep::Exact(TraceLocation::new(
                    op.location.function,
                    observed,
                )))
            }

            Switch => {
                let slot = FrameSlot::General(Reg(op.a));
                let physical = runtime
                    .trace_read_slot(slot)
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;

                // A trace only needs to prove that the switch keeps taking the
                // recorded path. Guard the exact observed value instead of
                // reproducing the whole jump table in native code. This is both
                // cheaper and works for tagged and unboxed switch tags.
                let known = builder.slot_value(slot).and_then(|value| {
                    let value = builder.trace().canonical_value(value);
                    builder.trace().value(value).map(|desc| (desc.ty, desc.rep))
                });

                let (tag, observed_value, key) = match known {
                    Some((ty, TraceRep::Int32)) => {
                        let raw = physical.0 as u32 as i32;
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::Int32)?;
                        let constant = builder.constant(TraceConstant::Int32(raw));
                        (tag, constant, raw as i64)
                    }
                    Some((ty, TraceRep::UInt32)) => {
                        let raw = physical.0 as u32;
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::UInt32)?;
                        let constant = builder.constant(TraceConstant::Int32(raw as i32));
                        let constant = builder.emit1(
                            TraceInst::Reinterpret { value: constant },
                            TraceType::UInt32,
                            TraceRep::UInt32,
                        );
                        (tag, constant, raw as i64)
                    }
                    Some((ty, TraceRep::Bool)) => {
                        let raw = physical.0 as u32 as i32;
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::Bool)?;
                        let constant = builder.constant(TraceConstant::Int32(raw));
                        let constant = builder.emit1(
                            TraceInst::Reinterpret { value: constant },
                            TraceType::Bool,
                            TraceRep::Bool,
                        );
                        (tag, constant, raw as i64)
                    }
                    Some((ty, TraceRep::F64)) => {
                        let number = f64::from_bits(physical.0);
                        if number.is_nan() {
                            return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                        }
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::F64)?;
                        let constant = builder.constant(TraceConstant::F64Bits(physical.0));
                        (tag, constant, number as i64)
                    }
                    Some((ty, TraceRep::Tagged))
                        if matches!(ty, TraceType::Any | TraceType::Num) && word_is_num(physical) =>
                    {
                        let number = f64::from_bits(physical.0);
                        let tag = self.read_gpr(
                            builder,
                            op.a,
                            TraceType::Num,
                            TraceRep::Tagged,
                        )?;
                        let constant = builder.constant(TraceConstant::NumBits(physical.0));
                        (tag, constant, number as i64)
                    }
                    Some((ty, TraceRep::Tagged)) if physical.0 == TAG_NULL => {
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::Tagged)?;
                        let constant = builder.constant(TraceConstant::Null);
                        (tag, constant, physical.0 as u32 as i32 as i64)
                    }
                    Some((ty, TraceRep::Tagged))
                        if physical.0 == TAG_FALSE || physical.0 == TAG_TRUE =>
                    {
                        let tag = self.read_gpr(builder, op.a, ty, TraceRep::Tagged)?;
                        let constant = builder.constant(TraceConstant::Bool(physical.0 == TAG_TRUE));
                        (tag, constant, physical.0 as u32 as i32 as i64)
                    }
                    None if word_is_num(physical) => {
                        let number = f64::from_bits(physical.0);
                        let tag = self.read_gpr(
                            builder,
                            op.a,
                            TraceType::Num,
                            TraceRep::Tagged,
                        )?;
                        let constant = builder.constant(TraceConstant::NumBits(physical.0));
                        (tag, constant, number as i64)
                    }
                    None if physical.0 == TAG_NULL => {
                        let tag = self.read_gpr(builder, op.a, TraceType::Null, TraceRep::Tagged)?;
                        let constant = builder.constant(TraceConstant::Null);
                        (tag, constant, physical.0 as u32 as i32 as i64)
                    }
                    None if physical.0 == TAG_FALSE || physical.0 == TAG_TRUE => {
                        let tag = self.read_gpr(builder, op.a, TraceType::Bool, TraceRep::Tagged)?;
                        let constant = builder.constant(TraceConstant::Bool(physical.0 == TAG_TRUE));
                        (tag, constant, physical.0 as u32 as i32 as i64)
                    }
                    _ => return Err(RecordingAbort::UnsupportedOpcode(op.opcode)),
                };

                let Some(table) = function.chunk.switch_tables.get(op.x as usize) else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };

                let selected = table
                    .cases
                    .iter()
                    .find(|case| case.value == key)
                    .map(|case| case.target)
                    .unwrap_or(table.default_target);

                // On a different value, reexecute the original switch in baseline.
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                builder.emit0(TraceInst::GuardEq {
                    lhs: tag,
                    rhs: observed_value,
                    exit,
                });

                Ok(ExpectedStep::Exact(TraceLocation::new(
                    op.location.function,
                    selected,
                )))
            }

            GuardNotNull => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                if physical.0 == TAG_NULL {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }
                let resume = deopt_resume(function, op.x).unwrap_or(op.location.pc);
                let exit = builder.add_side_exit(ExitReason::GuardNotNull, resume);
                let result = builder.emit1(
                    TraceInst::GuardNotNull {
                        value: source,
                        exit,
                    },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            GuardInt32 => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some(number) = word_num(physical) else {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                };
                if !number.is_finite()
                    || number.fract() != 0.0
                    || number < i32::MIN as f64
                    || number > i32::MAX as f64
                {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }

                let resume = deopt_resume(function, op.x).unwrap_or(op.location.pc);
                let exit = builder.add_side_exit(ExitReason::GuardInteger, resume);
                let result = builder.emit1(
                    TraceInst::GuardInt32 {
                        value: source,
                        exit,
                    },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            GuardF64 => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                if !word_is_num(physical) {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }
                let resume = deopt_resume(function, op.x).unwrap_or(op.location.pc);
                let exit = builder.add_side_exit(ExitReason::GuardNumber, resume);
                let result = builder.emit1(
                    TraceInst::GuardF64 {
                        value: source,
                        exit,
                    },
                    TraceType::F64,
                    TraceRep::F64,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            GuardBounds => {
                let index = self.read_gpr(
                    builder,
                    op.a,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let length = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;

                let i = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.a)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?
                    .0 as u32 as i32;
                let n = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?
                    .0 as u32 as i32;
                if i < 0 || i >= n {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }

                let resume = deopt_resume(function, op.x).unwrap_or(op.location.pc);
                let exit = builder.add_side_exit(ExitReason::GuardBounds, resume);
                builder.emit0(TraceInst::GuardBounds {
                    index,
                    length,
                    exit,
                });
                Ok(ExpectedStep::Exact(next))
            }

            GuardType => {
                let Some(descriptor) = function.chunk.type_guards.get(op.x as usize) else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };
                let expected = match &descriptor.expected {
                    Type::Top => TraceType::Any,
                    Type::Null => TraceType::Null,
                    Type::Bool => TraceType::Bool,
                    Type::Num => TraceType::Num,
                    // Heap/reference guards still use the VM's private class
                    // representation, but GuardType itself is non-GC. Trace it
                    // through the synchronized leaf boundary and preserve the
                    // rest of the native SSA state on the successful path.
                    _ => {
                        return self.runtime_leaf(
                            op,
                            builder,
                            &[(Reg(op.a), TraceType::Any, TraceRep::Tagged)],
                        );
                    }
                };

                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let matches = match expected {
                    TraceType::Any => true,
                    TraceType::Null => physical.0 == TAG_NULL,
                    TraceType::Bool => physical.0 == TAG_FALSE || physical.0 == TAG_TRUE,
                    TraceType::Num => word_is_num(physical),
                    _ => false,
                };
                if !matches {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }

                let source = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let result = if expected == TraceType::Any {
                    source
                } else {
                    let resume = deopt_resume(function, op.y).unwrap_or(op.location.pc);
                    let reason = match expected {
                        TraceType::Null | TraceType::Bool => ExitReason::GuardValue,
                        TraceType::Num => ExitReason::GuardNumber,
                        _ => ExitReason::GuardValue,
                    };
                    let exit = builder.add_side_exit(reason, resume);
                    builder.emit1(
                        TraceInst::GuardType { value: source, expected, exit },
                        expected,
                        TraceRep::Tagged,
                    )
                };
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            GuardClass => {
                let classes = runtime.trace_class_guard(op.location.function, op.x)
                    .ok_or(RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let physical = runtime.trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                if !runtime.trace_receiver_class(physical).is_some_and(|class| classes.contains(&class)) {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                // Replay the guard on failure so the VM owns deoptimization.
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let result = builder.emit1(TraceInst::GuardClass { value, classes, exit },
                    TraceType::Any, TraceRep::Tagged);
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            Bailout => Err(RecordingAbort::UnsupportedOpcode(op.opcode)),

            Invoke => {
                if self.record_num_invoke(runtime, program, function, op, builder)? {
                    Ok(ExpectedStep::Exact(next))
                } else {
                    self.inline_or_runtime_call(runtime, program, op, builder)
                }
            }
            CallDirect => self.inline_or_runtime_call(runtime, program, op, builder),

            // Specialized integer unary.
            INeg | IBitNot => {
                let value = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let unary = if op.opcode == INeg {
                    IntUnaryOp::Neg
                } else {
                    IntUnaryOp::BitNot
                };
                let result = builder.emit1(
                    TraceInst::IntUnary { op: unary, value },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            IAdd | ISub | IMul | IBitAnd | IBitOr | IBitXor | IShl | IShrU | ISar => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let bin = match op.opcode {
                    IAdd => IntBinOp::Add,
                    ISub => IntBinOp::Sub,
                    IMul => IntBinOp::Mul,
                    IBitAnd => IntBinOp::BitAnd,
                    IBitOr => IntBinOp::BitOr,
                    IBitXor => IntBinOp::BitXor,
                    IShl => IntBinOp::Shl,
                    IShrU => IntBinOp::ShrU,
                    ISar => IntBinOp::Sar,
                    _ => unreachable!(),
                };
                let result = builder.emit1(
                    TraceInst::IntBinary { op: bin, lhs, rhs },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            IDivS | IRemS => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let exit = builder.add_side_exit(ExitReason::GuardInteger, op.location.pc);
                let operation = if op.opcode == IDivS {
                    IntBinOp::DivS
                } else {
                    IntBinOp::RemS
                };
                let result = builder.emit1(
                    TraceInst::IntBinaryChecked {
                        op: operation,
                        lhs,
                        rhs,
                        exit,
                    },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            FNeg => {
                let value = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let result = builder.emit1(
                    TraceInst::F64Unary {
                        op: F64UnaryOp::Neg,
                        value,
                    },
                    TraceType::F64,
                    TraceRep::F64,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            FAdd | FSub | FMul | FDiv | FRem => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let bin = match op.opcode {
                    FAdd => F64BinOp::Add,
                    FSub => F64BinOp::Sub,
                    FMul => F64BinOp::Mul,
                    FDiv => F64BinOp::Div,
                    FRem => F64BinOp::Rem,
                    _ => unreachable!(),
                };
                let result = builder.emit1(
                    TraceInst::F64Binary { op: bin, lhs, rhs },
                    TraceType::F64,
                    TraceRep::F64,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let cmp = cmp_from_opcode(op.opcode);
                let raw = builder.emit1(
                    TraceInst::IntCompare { op: cmp, lhs, rhs },
                    TraceType::Bool,
                    TraceRep::Bool,
                );
                let boxed = self.tagged_bool(builder, raw);
                self.write_gpr(builder, op.a, boxed);
                Ok(ExpectedStep::Exact(next))
            }

            FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let cmp = cmp_from_opcode(op.opcode);
                let raw = builder.emit1(
                    TraceInst::F64Compare { op: cmp, lhs, rhs },
                    TraceType::Bool,
                    TraceRep::Bool,
                );
                let boxed = self.tagged_bool(builder, raw);
                self.write_gpr(builder, op.a, boxed);
                Ok(ExpectedStep::Exact(next))
            }

            RefEq => {
                let lhs = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let rhs = self.read_gpr(
                    builder,
                    op.c,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let raw = builder.emit1(
                    TraceInst::RefEq { lhs, rhs },
                    TraceType::Bool,
                    TraceRep::Bool,
                );
                let boxed = self.tagged_bool(builder, raw);
                self.write_gpr(builder, op.a, boxed);
                Ok(ExpectedStep::Exact(next))
            }

            NumToInt32 => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some(number) = word_num(physical) else {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                };
                if !number.is_finite()
                    || number.fract() != 0.0
                    || number < i32::MIN as f64
                    || number > i32::MAX as f64
                {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }
                let exit = builder.add_side_exit(ExitReason::GuardInteger, op.location.pc);
                let result = builder.emit1(
                    TraceInst::GuardInt32 {
                        value: source,
                        exit,
                    },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            NumToF64 => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                if !word_is_num(physical) {
                    return Err(RecordingAbort::GuardFailedWhileRecording);
                }
                let exit = builder.add_side_exit(ExitReason::GuardNumber, op.location.pc);
                let result = builder.emit1(
                    TraceInst::GuardF64 {
                        value: source,
                        exit,
                    },
                    TraceType::F64,
                    TraceRep::F64,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            Int32ToNum => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Int32,
                    TraceRep::Int32,
                )?;
                let result = builder.emit1(
                    TraceInst::BoxInt32 { value: source },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            F64ToNum => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::F64,
                    TraceRep::F64,
                )?;
                let result = builder.emit1(
                    TraceInst::BoxF64 { value: source },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            LoadField => {
                let object = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let object = self.guard_heap_object(builder, object, TraceHeapObjectKind::Instance, exit);
                let result = builder.emit1(
                    TraceInst::HeapLoadField { object, slot: runtime.trace_field_slot(op.location, op.x), exit },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            StoreField => {
                let object = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let object = self.guard_heap_object(builder, object, TraceHeapObjectKind::Instance, exit);
                builder.emit0(TraceInst::HeapStoreField {
                    object, value, slot: runtime.trace_field_slot(op.location, op.x), exit });
                Ok(ExpectedStep::Exact(next))
            }

            LoadCell => {
                let cell = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let cell = self.guard_heap_object(builder, cell, TraceHeapObjectKind::Cell, exit);
                let result = builder.emit1(
                    TraceInst::HeapLoadCell { cell, exit },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            StoreCell => {
                let cell = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let cell = self.guard_heap_object(builder, cell, TraceHeapObjectKind::Cell, exit);
                builder.emit0(TraceInst::HeapStoreCell { cell, value, exit });
                Ok(ExpectedStep::Exact(next))
            }

            ListLen => {
                let list = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let list = self.guard_heap_object(builder, list, TraceHeapObjectKind::List, exit);
                let result = builder.emit1(
                    TraceInst::HeapListLen { list, exit },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            ListGet => {
                let list = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let index = self.read_gpr(builder, op.c, TraceType::Int32, TraceRep::Int32)?;
                let exit = builder.add_side_exit(ExitReason::GuardBounds, op.location.pc);
                let list = self.guard_heap_object(builder, list, TraceHeapObjectKind::List, exit);
                let result = builder.emit1(
                    TraceInst::HeapListGet { list, index, exit },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            ListSet => {
                let list = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                let index = self.read_gpr(builder, op.b, TraceType::Int32, TraceRep::Int32)?;
                let value = self.read_gpr(builder, op.c, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardBounds, op.location.pc);
                let list = self.guard_heap_object(builder, list, TraceHeapObjectKind::List, exit);
                builder.emit0(TraceInst::HeapListSet { list, index, value, exit });
                Ok(ExpectedStep::Exact(next))
            }

            MapLen => {
                let map = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let map = self.guard_heap_object(builder, map, TraceHeapObjectKind::Map, exit);
                let result = builder.emit1(
                    TraceInst::HeapMapLen { map, exit },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            MapGet => {
                let map = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let key = self.read_gpr(builder, op.c, TraceType::Any, TraceRep::Tagged)?;
                let map_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let key_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.c)))
                    .map_err(|_| RecordingAbort::UnsupportedOpcode(op.opcode))?;
                if let Some(access) = runtime.trace_shaped_map_access(map_word, key_word) {
                    let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                    let map = self.guard_heap_object(builder, map, TraceHeapObjectKind::Map, exit);
                    let result = builder.emit1(
                        TraceInst::HeapMapGetShaped {
                            map,
                            key,
                            expected_key: access.key_bytes,
                            shape: access.shape,
                            slot: access.slot as u32,
                            exit,
                        },
                        TraceType::Any,
                        TraceRep::Tagged,
                    );
                    self.write_gpr(builder, op.a, result);
                    Ok(ExpectedStep::Exact(next))
                } else {
                    self.runtime_leaf(
                        op,
                        builder,
                        &[(Reg(op.a), TraceType::Any, TraceRep::Tagged)],
                    )
                }
            }

            MapSet => {
                let map = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                let key = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let value = self.read_gpr(builder, op.c, TraceType::Any, TraceRep::Tagged)?;
                let map_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.a)))
                    .map_err(|_| RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let key_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let Some(access) = runtime.trace_shaped_map_access(map_word, key_word) else {
                    // Inserts, dictionary Maps and non-string keys remain runtime
                    // operations because they can change representation/storage.
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let map = self.guard_heap_object(builder, map, TraceHeapObjectKind::Map, exit);
                builder.emit0(TraceInst::HeapMapSetShaped {
                    map,
                    key,
                    value,
                    expected_key: access.key_bytes,
                    shape: access.shape,
                    slot: access.slot as u32,
                    exit,
                });
                Ok(ExpectedStep::Exact(next))
            }

            StringLen => {
                let string = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let string = self.guard_heap_object(builder, string, TraceHeapObjectKind::String, exit);
                let result = builder.emit1(
                    TraceInst::HeapStringLen { string, exit },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            ReadBarrier => {
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let result = builder.emit1(
                    TraceInst::ReadBarrier { value },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            LoadModuleVar => {
                let address = runtime
                    .trace_module_slot_address(op.location)
                    .ok_or(RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let value = builder.emit1(
                    TraceInst::LoadAbsoluteValue { address: address as u64 },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }

            StoreModuleVar => {
                let address = runtime
                    .trace_module_slot_address(op.location)
                    .ok_or(RecordingAbort::UnsupportedOpcode(op.opcode))?;
                let value = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                builder.emit0(TraceInst::StoreAbsoluteValue {
                    address: address as u64,
                    value,
                });
                Ok(ExpectedStep::Exact(next))
            }

            LoadStaticField => {
                let class_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some((class_key, address)) = runtime
                    .trace_static_field_access(class_word, op.x)
                else {
                    return self.runtime_leaf(
                        op,
                        builder,
                        &[(Reg(op.a), TraceType::Any, TraceRep::Tagged)],
                    );
                };
                let class = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let _guarded = builder.emit1(
                    TraceInst::GuardClass {
                        value: class,
                        classes: vec![class_key],
                        exit,
                    },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                let value = builder.emit1(
                    TraceInst::LoadAbsoluteValue { address: address as u64 },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }

            StoreStaticField => {
                let class_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.a)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some((class_key, address)) = runtime
                    .trace_static_field_access(class_word, op.x)
                else {
                    return self.runtime_leaf(op, builder, &[]);
                };
                let class = self.read_gpr(builder, op.a, TraceType::Any, TraceRep::Tagged)?;
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let _guarded = builder.emit1(
                    TraceInst::GuardClass {
                        value: class,
                        classes: vec![class_key],
                        exit,
                    },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                builder.emit0(TraceInst::StoreAbsoluteValue {
                    address: address as u64,
                    value,
                });
                Ok(ExpectedStep::Exact(next))
            }

            IsInstance => {
                let class_word = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.c)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some((expected_class, instance_classes, known_classes)) = runtime
                    .trace_is_instance_observation(class_word)
                else {
                    return self.runtime_leaf(
                        op,
                        builder,
                        &[(Reg(op.a), TraceType::Bool, TraceRep::Tagged)],
                    );
                };
                let value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let class = self.read_gpr(builder, op.c, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let result = builder.emit1(
                    TraceInst::IsInstanceObserved {
                        value,
                        class,
                        expected_class,
                        instance_classes,
                        known_classes,
                        exit,
                    },
                    TraceType::Bool,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, result);
                Ok(ExpectedStep::Exact(next))
            }

            // Remaining non-GC runtime leaf operations preserve unrelated native
            // SSA values instead of forcing the full runtime memory epoch.
            ListRemoveAt | MapRemove => self.runtime_leaf(
                op,
                builder,
                &[(Reg(op.a), TraceType::Any, TraceRep::Tagged)],
            ),

            StringCodePointAt => self.runtime_leaf(
                op,
                builder,
                &[(Reg(op.a), TraceType::Int32, TraceRep::Int32)],
            ),

            WriteBarrier | ProfileBlock | ProfileNumeric => self.runtime_leaf(op, builder, &[]),

            Await => {
                let observed = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.b)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;
                let Some(_fulfilled) = runtime
                    .trace_fulfilled_promise(observed)
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?
                else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };
                let promise = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let promise = self.guard_heap_object(builder, promise, TraceHeapObjectKind::Promise, exit);
                let value = builder.emit1(
                    TraceInst::HeapAwaitFulfilled { promise, exit },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, value);
                Ok(ExpectedStep::Exact(next))
            }

            Breakpoint => Err(RecordingAbort::UnsupportedOpcode(op.opcode)),

            // Full call boundaries remain inside the native trace. The runtime
            // helper executes a synchronous bytecode callee until it returns to
            // this frame, or reports a terminal/frame-changing condition. This
            // removes the tier-manager round trip without pretending that host
            // or foreign calls are pure or non-GC.
            InvokeSuper | CallDirect | CallClosure | CallPrimitive | CallForeign | ImportModule => {
                builder.runtime_call(op.location, op.next_pc);
                Ok(ExpectedStep::RuntimeCall)
            }

            // GC-capable operations that return to the same frame are explicit
            // synchronized slow paths rather than unsupported opcodes. They still
            // form a full heap/register epoch, but the surrounding arithmetic and
            // control flow remains in the compiled trace even when the default
            // `no_runtime_calls_in_native_trace` policy is enabled.
            CreateClass | BindMethod | BindForeignMethod | AllocInstance | AllocForeign
            | ImportVariable | AllocCell | MakeClosure | AllocList | AllocListCap
            | ListAppend | ListInsert | AllocMap | StringConcat | StringSlice
            | MakeRange | MakeFiber | Safepoint | Catch => {
                builder.runtime(op.location, op.next_pc, false);
                Ok(ExpectedStep::Runtime { terminal: false })
            }

            Return
            | Return2
            | Trap
            | FiberCall
            | FiberTry
            | FiberTransfer
            | FiberTransferError
            | FiberYield
            | FiberSuspend
            | FiberAbort => Err(RecordingAbort::UnsupportedOpcode(op.opcode)),
        }
    }

    fn install_loop_phis(
        &self,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<(), RecordingAbort> {
        builder.ensure_loop_entries();
        // Collect root-frame entry values first. We must not mutate the vector
        // while iterating it.
        let entries: Vec<(FrameSlot, TraceValueId, TraceType, TraceRep)> = builder
            .trace()
            .values
            .iter()
            .filter_map(|value| match &value.origin {
                TraceValueOrigin::EntrySlot { frame, slot } if frame.0 == 0 => {
                    Some((*slot, value.id, value.ty, value.rep))
                }
                _ => None,
            })
            .collect();

        for (slot, initial, ty, rep) in entries {
            let Some(final_value) = builder.slot_value(slot) else {
                continue;
            };
            let mut final_value = builder.trace().canonical_value(final_value);
            let initial_canonical = builder.trace().canonical_value(initial);
            if final_value == initial_canonical {
                continue;
            }

            let final_rep = builder.trace().value(final_value).unwrap().rep;
            if final_rep != rep {
                // Reinterpret preserves physical bits; it is not a semantic
                // conversion between Tagged/Int32/F64. Compiling such a backedge
                // can feed invalid bits into the next native iteration. Until the
                // trace IR grows an explicit representation bridge, keep this loop
                // in the baseline interpreter.
                return Err(RecordingAbort::RepresentationConflict);
            }

            let phi = builder.add_loop_phi(slot, initial, ty, rep);

            // Rewrite uses *inside the iteration* from the original entry value to
            // the loop-carried value. Do not rewrite LoopPhi.initial itself.
            {
                let trace = builder.trace_mut();

                for node in &mut trace.nodes {
                    node.kind.replace_operand(initial, phi);
                }
                for exit in &mut trace.exits {
                    for frame in &mut exit.frames {
                        if !frame.writes.iter().any(|write| write.slot == slot) {
                            frame.writes.push(crate::wren_trace_ir::SnapshotWrite {
                                slot,
                                kind: RegisterKind::Value,
                                target_rep: rep,
                                value: SnapshotValue::Value(phi),
                            });
                        }
                        for write in &mut frame.writes {
                            if write.value == SnapshotValue::Value(initial) {
                                write.value = SnapshotValue::Value(phi);
                            }
                        }
                    }
                }
                for existing in &mut trace.loop_phis {
                    if existing.value == phi {
                        continue;
                    }
                    if existing.backedge == Some(initial) {
                        existing.backedge = Some(phi);
                    }
                }
                trace.terminator.replace_operand(initial, phi);
            }

            builder
                .set_loop_backedge(phi, final_value)
                .map_err(|_| RecordingAbort::RepresentationConflict)?;
        }

        Ok(())
    }

    fn record<R: TraceRuntime>(
        &self,
        runtime: &mut R,
    ) -> Result<Trace, RecordOutcome> {
        self.record_impl(runtime, RecordingMode::RootLoop)
    }

    fn record_side<R: TraceRuntime>(
        &self,
        runtime: &mut R,
        close_at: TraceLocation,
        link_to: TraceId,
    ) -> Result<Trace, RecordOutcome> {
        self.record_impl(runtime, RecordingMode::Side { close_at, link_to })
    }

    fn record_impl<R: TraceRuntime>(
        &self,
        runtime: &mut R,
        mode: RecordingMode,
    ) -> Result<Trace, RecordOutcome> {
        let program_ptr = runtime.trace_program() as *const BytecodeProgram;
        let program = unsafe { &*program_ptr };

        program
            .function(self.anchor.function)
            .ok_or_else(|| RecordOutcome::Fatal(TracingJitError::Record {
                location: self.anchor,
                message: "trace anchor references unknown function".to_owned(),
            }))?;

        let (kind, close_at) = match mode {
            RecordingMode::RootLoop => (TraceKind::Loop, self.anchor),
            RecordingMode::Entry => (TraceKind::Entry, self.anchor),
            RecordingMode::Side { close_at, .. } => (TraceKind::Side, close_at),
        };
        if close_at.function != self.anchor.function {
            return Err(RecordOutcome::Abort(RecordingAbort::LeftFunction));
        }

        let mut builder = crate::wren_trace_ir::TraceBuilder::new(
            self.trace_id,
            TraceAnchor {
                kind,
                location: self.anchor,
            },
        );

        let mut steps = 0usize;
        let mut runtime_end = false;
        let mut visited = HashSet::new();

        loop {
            if steps >= self.options.max_trace_steps {
                if matches!(mode, RecordingMode::Entry) { break; }
                return Err(RecordOutcome::Abort(RecordingAbort::TraceTooLong));
            }

            let location = runtime.trace_current_location();
            if location.function != self.anchor.function {
                return Err(RecordOutcome::Abort(RecordingAbort::LeftFunction));
            }

            if matches!(mode, RecordingMode::Entry) && runtime.trace_is_loop_header(location) {
                break;
            }
            if steps > 0 && location == close_at {
                break;
            }
            if !visited.insert(location) {
                // Side traces may join a parent root, but they still must not
                // contain an unrelated cycle: that would require a region CFG.
                return Err(RecordOutcome::Abort(RecordingAbort::NestedCycle(location)));
            }

            let function = program
                .function(location.function)
                .ok_or_else(|| RecordOutcome::Fatal(TracingJitError::Record {
                    location,
                    message: "current function disappeared during recording".to_owned(),
                }))?;

            let op = decode_at(program, location).map_err(RecordOutcome::Fatal)?;

            let expected = if matches!(mode, RecordingMode::Entry) {
                // A boundary may be discovered after speculative guards have been
                // emitted. Keep supported synchronous boundaries in the entry trace;
                // otherwise roll the speculative lowering back and close the prefix.
                let checkpoint = builder.clone();
                match self.translate_native(runtime, program, function, &op, &mut builder) {
                    Ok(step @ ExpectedStep::Exact(next)) if next.pc.0 > location.pc.0
                        && !visited.contains(&next) => step,
                    Ok(step @ ExpectedStep::Inlined { next, .. }) if next.pc.0 > location.pc.0
                        && !visited.contains(&next) => step,
                    // Synchronous call/runtime boundaries can remain inside an
                    // entry trace as long as the helper returns to the same frame.
                    // The recording loop below validates the exact continuation
                    // after executing the boundary and stops cleanly otherwise.
                    Ok(step @ ExpectedStep::RuntimeCall) => step,
                    Ok(step @ ExpectedStep::Runtime { terminal: false }) => step,
                    Ok(_) | Err(_) => {
                        builder = checkpoint;
                        break;
                    }
                }
            } else {
                self.translate_op(runtime, program, function, &op, &mut builder)
                    .map_err(RecordOutcome::Abort)?
            };

            let mut step = match expected {
                ExpectedStep::RuntimeCall => runtime.trace_execute_call_to_return(),
                _ => runtime.trace_execute_unfused_step(),
            }
            .map_err(|error| RecordOutcome::Fatal(runtime_error(error)))?;

            if let ExpectedStep::Inlined { next, callee, steps: callee_steps } = expected {
                if !step.running || step.after != TraceLocation::new(callee, BytecodePc(0)) {
                    return Err(RecordOutcome::Abort(RecordingAbort::FrameChanged));
                }
                for _ in 0..callee_steps {
                    if !step.running || runtime.trace_current_location().function != callee {
                        return Err(RecordOutcome::Abort(RecordingAbort::FrameChanged));
                    }
                    step = runtime.trace_execute_unfused_step()
                        .map_err(|error| RecordOutcome::Fatal(runtime_error(error)))?;
                }
                if !step.running || step.after != next {
                    return Err(RecordOutcome::Abort(RecordingAbort::FrameChanged));
                }
                // The bounded leaf returned to the original caller. Its frame
                // transitions are absent from native code and all deopts replay
                // the original call with the caller snapshot.
                step = TraceStep::simple(location, next);
                builder.trace_mut().stats.recorded_bytecodes += callee_steps;
            }

            steps += 1;
            builder.trace_mut().stats.recorded_bytecodes += 1;

            if let ExpectedStep::Runtime { terminal } = expected {
                if terminal || !step.running || step.frame_changed
                    || step.after != TraceLocation::new(location.function, op.next_pc)
                {
                    runtime_end = true;
                    break;
                }
                continue;
            }
            if matches!(expected, ExpectedStep::RuntimeCall) {
                if !step.running || step.frame_changed
                    || step.after != TraceLocation::new(location.function, op.next_pc)
                {
                    runtime_end = true;
                    break;
                }
                continue;
            }

            if !step.running {
                return Err(RecordOutcome::Abort(RecordingAbort::InterpreterStopped));
            }
            if step.frame_changed {
                return Err(RecordOutcome::Abort(RecordingAbort::FrameChanged));
            }
            if matches!(mode, RecordingMode::RootLoop)
                && step.after.pc.0 < self.anchor.pc.0
                && step.after.pc.0 < location.pc.0
            {
                // An inner loop can finish on the recording iteration. Walking
                // through its enclosing loop and back to this header does not
                // record an iteration of the inner loop.
                return Err(RecordOutcome::Abort(RecordingAbort::EnclosingLoop));
            }
            if step.before != location {
                return Err(RecordOutcome::Fatal(TracingJitError::Record {
                    location,
                    message: "runtime executed a different bytecode than recorder decoded".to_owned(),
                }));
            }

            if let ExpectedStep::Exact(expected_location) | ExpectedStep::Inlined { next: expected_location, .. } = expected {
                if step.after != expected_location {
                    return Err(RecordOutcome::Fatal(TracingJitError::Record {
                        location,
                        message: format!(
                            "recorded path mismatch: expected {:?}:{}, runtime reached {:?}:{}",
                            expected_location.function,
                            expected_location.pc.0,
                            step.after.function,
                            step.after.pc.0
                        ),
                    }));
                }
            }
        }

        if steps == 0 {
            return Err(RecordOutcome::Abort(RecordingAbort::PathDidNotClose));
        }

        if builder.has_runtime_calls() {
            builder.prepare_memory_loop();
        }

        let side_terminal_exit = match mode {
            RecordingMode::Entry => {
                if runtime_end {
                    // The recorded boundary did not synchronously return to the
                    // expected continuation (throw/suspend/frame change/etc.).
                    // RuntimeCall already returns directly on that cold path; if
                    // a later execution does return normally, stop at the same
                    // epoch instead of using a snapshot from a different frame.
                    builder.set_terminator(TraceTerminator::RuntimeReturn);
                    None
                } else {
                    let exit = builder.add_side_exit(
                        ExitReason::GuardValue,
                        runtime.trace_current_location().pc,
                    );
                    builder.set_terminator(TraceTerminator::Exit(exit));
                    None
                }
            }
            RecordingMode::RootLoop => {
                if !builder.has_runtime_calls() {
                    self.install_loop_phis(&mut builder).map_err(RecordOutcome::Abort)?;
                    if self.options.native_loop_iteration_budget != 0 {
                        builder.add_native_budget_exit(self.anchor.pc);
                    }
                }
                builder.set_terminator(if runtime_end {
                    TraceTerminator::RuntimeReturn
                } else {
                    TraceTerminator::LoopBack
                });
                None
            }
            RecordingMode::Side { close_at, .. } => {
                if runtime_end {
                    builder.set_terminator(TraceTerminator::RuntimeReturn);
                    None
                } else {
                    // Keep the interpreter location until liveness pruning has run;
                    // afterwards this exit is retargeted to the parent native trace.
                    let exit = builder.add_side_exit(ExitReason::GuardValue, close_at.pc);
                    builder.set_terminator(TraceTerminator::Exit(exit));
                    Some(exit)
                }
            }
        };

        let mut trace = builder.finish().map_err(|errors| {
            RecordOutcome::Fatal(TracingJitError::TraceVerification(
                errors.into_iter().map(|error| error.to_string()).collect(),
            ))
        })?;

        prune_dead_snapshot_writes(program, &mut trace);

        if self.options.optimize_trace_ir {
            crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(&mut trace).map_err(|errors| {
                RecordOutcome::Fatal(TracingJitError::TraceVerification(
                    errors.into_iter().map(|error| error.to_string()).collect(),
                ))
            })?;
        }

        if let (RecordingMode::Side { link_to, .. }, Some(exit)) = (mode, side_terminal_exit) {
            if let Some(side_exit) = trace.exits.get_mut(exit.0 as usize) {
                side_exit.target = ExitTarget::Trace(link_to);
                side_exit.linked_trace = Some(link_to);
            }
            verify_trace(&trace).map_err(|errors| {
                RecordOutcome::Fatal(TracingJitError::TraceVerification(errors))
            })?;
        }

        Ok(trace)
    }
}

fn prune_dead_snapshot_writes(program: &BytecodeProgram, trace: &mut Trace) {
    let mut pruned = 0usize;
    for exit in &mut trace.exits {
        let ExitTarget::Interpreter(location) = exit.target else { continue; };
        let Some(live) = program
            .function(location.function)
            .and_then(|function| function.chunk.live_registers_at(location.pc))
        else {
            // Missing metadata must be conservative: preserve the whole snapshot.
            continue;
        };

        for frame in &mut exit.frames {
            let before = frame.writes.len();
            frame.writes.retain(|write| match write.slot {
                FrameSlot::General(reg) => live.general_is_live(reg),
                // ARG/RET/special banks do not currently have SSA-allocation
                // liveness metadata. Keep them conservatively.
                FrameSlot::Argument(_) | FrameSlot::Return(_) | FrameSlot::Special(_) => true,
            });
            pruned += before - frame.writes.len();
        }
    }
    trace.stats.pruned_snapshot_writes = trace.stats.pruned_snapshot_writes.saturating_add(pruned);
}

enum RecordOutcome {
    Abort(RecordingAbort),
    Fatal(TracingJitError),
}

#[derive(Clone, Copy, Debug)]
enum ExpectedStep {
    Exact(TraceLocation),
    Inlined { next: TraceLocation, callee: FuncId, steps: usize },
    Runtime { terminal: bool },
    RuntimeCall,
}

fn special_reg(index: u8) -> Option<SpecialReg> {
    match index {
        0 => Some(SpecialReg::This),
        1 => Some(SpecialReg::Closure),
        2 => Some(SpecialReg::Module),
        3 => Some(SpecialReg::Fiber),
        4 => Some(SpecialReg::CallerFiber),
        5 => Some(SpecialReg::Error),
        6 => Some(SpecialReg::ResumeValue),
        7 => Some(SpecialReg::ForeignContext),
        8 => Some(SpecialReg::Scratch0),
        9 => Some(SpecialReg::Scratch1),
        10 => Some(SpecialReg::Scratch2),
        11 => Some(SpecialReg::Scratch3),
        _ => None,
    }
}

fn cmp_from_opcode(opcode: Opcode) -> CmpKind {
    match opcode {
        Opcode::ICmpEq | Opcode::FCmpEq => CmpKind::Eq,
        Opcode::ICmpNe | Opcode::FCmpNe => CmpKind::Ne,
        Opcode::ICmpLt | Opcode::FCmpLt => CmpKind::Lt,
        Opcode::ICmpLe | Opcode::FCmpLe => CmpKind::Le,
        Opcode::ICmpGt | Opcode::FCmpGt => CmpKind::Gt,
        Opcode::ICmpGe | Opcode::FCmpGe => CmpKind::Ge,
        _ => unreachable!("non-comparison opcode"),
    }
}

// =============================================================================
// Sparse side-exit plans
// =============================================================================

#[derive(Clone, Copy, Debug)]
enum ExitWriteSource {
    Constant(TraceConstant),
    Dynamic(u16),
}

#[derive(Clone, Copy, Debug)]
struct CompiledExitValue {
    value: TraceValueId,
    target_rep: TraceRep,
}

#[derive(Clone, Copy, Debug)]
struct CompiledExitWrite {
    slot: FrameSlot,
    kind: RegisterKind,
    target_rep: TraceRep,
    source: ExitWriteSource,
}

#[derive(Clone, Debug)]
struct CompiledExitPlan {
    id: SideExitId,
    target: ExitTarget,
    writes: Vec<CompiledExitWrite>,
    dynamic_values: Vec<CompiledExitValue>,
    /// Runtime-owned String/BuiltinClass constants cannot be reconstructed by
    /// generated code, so exits containing them keep manager-level stitching.
    native_linkable: bool,
}

impl CompiledExitPlan {
    fn from_exit(trace: &Trace, exit: &SideExit) -> Result<Self, TracingJitError> {
        if exit.frames.len() != 1 {
            return Err(TracingJitError::CraneliftLowering {
                trace: trace.id,
                message: format!(
                    "native tier currently supports one existing frame per exit, got {}",
                    exit.frames.len()
                ),
            });
        }

        let frame = &exit.frames[0];
        if !matches!(
            frame.disposition,
            FrameDisposition::Existing { depth_from_top: 0 }
        ) {
            return Err(TracingJitError::CraneliftLowering {
                trace: trace.id,
                message: "native tier cannot yet recreate inlined Wren frames".to_owned(),
            });
        }

        let mut dynamic_values = Vec::<CompiledExitValue>::new();
        let mut dynamic_index = HashMap::<(TraceValueId, TraceRep), u16>::new();
        let mut writes = Vec::with_capacity(frame.writes.len());

        for write in &frame.writes {
            let source = match write.value {
                SnapshotValue::Constant(constant) => ExitWriteSource::Constant(constant),
                SnapshotValue::Value(value) => {
                    let value = trace.canonical_value(value);
                    let key = (value, write.target_rep);
                    let index = if let Some(index) = dynamic_index.get(&key).copied() {
                        index
                    } else {
                        if dynamic_values.len() >= MAX_TRACE_EXIT_VALUES {
                            return Err(TracingJitError::CraneliftLowering {
                                trace: trace.id,
                                message: "side exit exceeds TraceActivation exit buffer".to_owned(),
                            });
                        }
                        let index = dynamic_values.len() as u16;
                        dynamic_values.push(CompiledExitValue {
                            value,
                            target_rep: write.target_rep,
                        });
                        dynamic_index.insert(key, index);
                        index
                    };
                    ExitWriteSource::Dynamic(index)
                }
            };

            writes.push(CompiledExitWrite {
                slot: write.slot,
                kind: write.kind,
                target_rep: write.target_rep,
                source,
            });
        }

        let native_linkable = writes.iter().all(|write| {
            !matches!(
                write.source,
                ExitWriteSource::Constant(TraceConstant::String(_) | TraceConstant::BuiltinClass(_))
            )
        });

        Ok(Self {
            id: exit.id,
            target: exit.target,
            writes,
            dynamic_values,
            native_linkable,
        })
    }
}

// =============================================================================
// Cranelift lowering
// =============================================================================

struct CompiledTrace {
    trace: Trace,
    function: ClifFuncId,
    code_ptr: *const u8,
    entry: NativeTraceFn,
    exits: Vec<CompiledExitPlan>,
    exit_hits: Vec<u32>,
    side_trace_attempted: Vec<bool>,
    /// Dense child TraceId per exit. u32::MAX means manager-level side exit. The
    /// boxed allocation has a stable address consumed by generated code.
    native_links: Box<[u32]>,
    /// Root loop owning this trace tree and depth from that root.
    root_trace: TraceId,
    tree_depth: u8,
    /// Consecutive entries that did too little work to amortize a transition.
    /// A long streak requests a temporary cooldown in the interpreter.
    short_exit_streak: u16,
    debug_text: Option<String>,
}

impl CompiledTrace {
    #[inline]
    fn code_ptr(&self) -> *const u8 {
        self.code_ptr
    }
}

struct CraneliftBackend {
    module: JITModule,
    context: codegen::Context,
    function_context: FunctionBuilderContext,
    dispatcher: NativeTraceFn,
}

impl CraneliftBackend {
    fn new(options: &TracingJitOptions) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", options.cranelift_opt_level.as_setting())
            .map_err(|error| format!("invalid Cranelift opt_level: {error}"))?;
        flag_builder
            .set(
                "enable_verifier",
                if options.cranelift_enable_verifier { "true" } else { "false" },
            )
            .map_err(|error| format!("cannot configure Cranelift verifier: {error}"))?;
        flag_builder
            .set("enable_alias_analysis", "true")
            .map_err(|error| format!("cannot enable Cranelift alias analysis: {error}"))?;
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|error| format!("cannot configure Cranelift libcalls: {error}"))?;
        flag_builder
            .set(
                "is_pic",
                if cfg!(target_arch = "x86_64") { "true" } else { "false" },
            )
            .map_err(|error| format!("cannot configure Cranelift PIC mode: {error}"))?;

        let isa_builder = cranelift::native::builder()
            .map_err(|error| format!("Cranelift does not support this host: {error}"))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|error| format!("cannot build Cranelift host ISA: {error}"))?;

        let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        if module.target_config().pointer_type() != types::I64 {
            return Err("WrenJIT's NaN-boxed native tier currently requires a 64-bit host".to_owned());
        }

        let dispatcher = Self::compile_dispatcher(&mut module)?;
        let context = module.make_context();
        let function_context = FunctionBuilderContext::new();
        Ok(Self { module, context, function_context, dispatcher })
    }

    /// Native trace-tree dispatcher. Individual traces keep the simple C ABI
    /// `fn(*mut TraceActivation) -> u32`; linked exits return TRACE_RESULT_LINK
    /// after writing `next_trace`. This loop immediately calls the child trace in
    /// generated code, avoiding a Rust-manager round trip and avoiding recursive
    /// native calls for root <-> side cycles.
    fn compile_dispatcher(module: &mut JITModule) -> Result<NativeTraceFn, String> {
        let pointer_type = module.target_config().pointer_type();
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer_type));
        signature.returns.push(AbiParam::new(types::I32));
        let function = module
            .declare_anonymous_function(&signature)
            .map_err(|error| format!("cannot declare native trace dispatcher: {error}"))?;

        let trace_signature = signature.clone();
        let mut context = module.make_context();
        context.func.signature = signature;
        context.func.name = UserFuncName::user(0, function.as_u32());
        let target_config = module.target_config();
        let mut function_context = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
            let entry = builder.create_block();
            let dispatch = builder.create_block();
            let linked_result = builder.create_block();
            let budgeted_link = builder.create_block();
            let done = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.append_block_param(linked_result, types::I32);
            builder.append_block_param(budgeted_link, types::I32);
            builder.append_block_param(done, types::I32);

            builder.switch_to_block(entry);
            let activation = builder.block_params(entry)[0];
            let trace_signature = builder.import_signature(trace_signature);
            builder.ins().jump(dispatch, &[]);

            builder.switch_to_block(dispatch);
            let trace_id = builder.ins().load(
                types::I32,
                MemFlagsData::trusted(),
                activation,
                offset_of!(TraceActivation, next_trace) as i32,
            );
            let trace_index = builder.ins().uextend(types::I64, trace_id);
            let byte_offset = builder.ins().ishl_imm(trace_index, 3);

            let entries = builder.ins().load(
                pointer_type,
                MemFlagsData::trusted(),
                activation,
                offset_of!(TraceActivation, trace_entries) as i32,
            );
            let entry_addr = builder.ins().iadd(entries, byte_offset);
            let callee = builder.ins().load(
                pointer_type,
                MemFlagsData::trusted(),
                entry_addr,
                0,
            );

            let link_tables = builder.ins().load(
                pointer_type,
                MemFlagsData::trusted(),
                activation,
                offset_of!(TraceActivation, trace_link_tables) as i32,
            );
            let link_addr = builder.ins().iadd(link_tables, byte_offset);
            let current_links = builder.ins().load(
                pointer_type,
                MemFlagsData::trusted(),
                link_addr,
                0,
            );
            builder.ins().store(
                MemFlagsData::trusted(),
                current_links,
                activation,
                offset_of!(TraceActivation, current_links) as i32,
            );

            let call = builder.ins().call_indirect(trace_signature, callee, &[activation]);
            let result = builder.inst_results(call)[0];
            let linked = builder.ins().icmp_imm(
                IntCC::Equal,
                result,
                TRACE_RESULT_LINK as i64,
            );
            builder.ins().brif(linked, linked_result, &[result.into()], done, &[result.into()]);

            // A trace-tree cycle can otherwise remain in generated code forever
            // while each individual trace resets its own loop watchdog. Bound the
            // number of native-to-native links as a second cooperative safepoint.
            builder.switch_to_block(linked_result);
            let result = builder.block_params(linked_result)[0];
            let remaining = builder.ins().load(
                types::I32,
                MemFlagsData::trusted(),
                activation,
                offset_of!(TraceActivation, dispatch_links_remaining) as i32,
            );
            let unlimited = builder.ins().icmp_imm(IntCC::Equal, remaining, -1);
            builder.ins().brif(unlimited, dispatch, &[], budgeted_link, &[result.into()]);

            builder.switch_to_block(budgeted_link);
            let result = builder.block_params(budgeted_link)[0];
            let remaining = builder.ins().load(
                types::I32,
                MemFlagsData::trusted(),
                activation,
                offset_of!(TraceActivation, dispatch_links_remaining) as i32,
            );
            let remaining = builder.ins().iadd_imm(remaining, -1);
            builder.ins().store(
                MemFlagsData::trusted(),
                remaining,
                activation,
                offset_of!(TraceActivation, dispatch_links_remaining) as i32,
            );
            let exhausted = builder.ins().icmp_imm(IntCC::Equal, remaining, 0);
            builder.ins().brif(exhausted, done, &[result.into()], dispatch, &[]);

            builder.switch_to_block(done);
            let result = builder.block_params(done)[0];
            builder.ins().return_(&[result]);
            builder.seal_all_blocks();
            builder.finalize(target_config);
        }

        module
            .define_function(function, &mut context)
            .map_err(|error| format!("cannot compile native trace dispatcher: {error}"))?;
        module
            .finalize_definitions()
            .map_err(|error| format!("cannot finalize native trace dispatcher: {error}"))?;
        let code_ptr = module.get_finalized_function(function);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }
}

struct CraneliftTraceCompiler<'a> {
    trace: &'a Trace,
    options: &'a TracingJitOptions,
    exit_plans: Vec<CompiledExitPlan>,
}

struct CraneliftLoweringState<'a, 'p> {
    builder: FunctionBuilder<'a>,
    trace: &'p Trace,
    activation: Value,
    pointer_type: ClifType,
    runtime_signature: codegen::ir::SigRef,
    heap_barrier_signature: codegen::ir::SigRef,
    values: Vec<Option<Value>>,
    /// Current moving-object address for a GuardHeapObject result. Valid only
    /// until the next full runtime/GC epoch. This is a backend-local lowering
    /// cache, not a Trace IR optimization pass.
    heap_addresses: Vec<Option<Value>>,
    /// Dominating kind/liveness proofs for tagged operands within a GC epoch.
    /// Metadata is reloaded separately because promise state can change.
    heap_entries: HashMap<(TraceValueId, u32), (Value, Value)>,
    /// Successful normalized list indices. Native element stores preserve length;
    /// all runtime calls invalidate these facts, including non-GC leaf calls.
    list_indices: HashMap<(TraceValueId, TraceValueId), Value>,
    list_lengths: HashMap<TraceValueId, TraceValueId>,
    exit_plans: &'p [CompiledExitPlan],
    /// Current loop budget block parameter. The manager derives completed native
    /// iterations from it only when a side exit is materialized.
    native_budget_remaining: Option<Value>,
    native_budget_initial: u32,
}

impl<'a> CraneliftTraceCompiler<'a> {
    fn new(trace: &'a Trace, options: &'a TracingJitOptions) -> Result<Self, TracingJitError> {
        verify_trace(trace).map_err(|errors| {
            TracingJitError::TraceVerification(
                errors.into_iter().map(|error| error.to_string()).collect(),
            )
        })?;

        // A loop phi whose backedge canonicalizes to itself cannot make
        // architectural progress. Reject it before native code generation rather
        // than relying on the watchdog to discover a non-advancing loop at run
        // time. `install_loop_phis` normally never creates such a phi; this also
        // protects us from an optimizer accidentally collapsing the update.
        for phi in &trace.loop_phis {
            if let Some(backedge) = phi.backedge {
                if trace.canonical_value(backedge) == trace.canonical_value(phi.value) {
                    return Err(TracingJitError::CraneliftLowering {
                        trace: trace.id,
                        message: format!(
                            "loop phi {:?} has a self backedge and cannot make progress",
                            phi.value
                        ),
                    });
                }
            }
        }

        let mut exit_plans = Vec::with_capacity(trace.exits.len());
        for exit in &trace.exits {
            exit_plans.push(CompiledExitPlan::from_exit(trace, exit)?);
        }

        Ok(Self {
            trace,
            options,
            exit_plans,
        })
    }

    fn compile(self, backend: &mut CraneliftBackend) -> Result<CompiledTrace, TracingJitError> {
        let trace_id = self.trace.id;
        let CraneliftBackend { module, context, function_context, .. } = backend;
        context.clear();
        let pointer_type = module.target_config().pointer_type();

        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer_type));
        signature.returns.push(AbiParam::new(types::I32));

        let function = module
            .declare_anonymous_function(&signature)
            .map_err(|error| TracingJitError::CraneliftLowering {
                trace: trace_id,
                message: format!("cannot declare native trace: {error}"),
            })?;

        let mut runtime_signature = module.make_signature();
        runtime_signature.params.push(AbiParam::new(pointer_type));
        runtime_signature.params.push(AbiParam::new(types::I32));
        runtime_signature.params.push(AbiParam::new(types::I32));
        runtime_signature.params.push(AbiParam::new(types::I32));
        runtime_signature.returns.push(AbiParam::new(types::I32));

        let mut heap_barrier_signature = module.make_signature();
        heap_barrier_signature.params.push(AbiParam::new(pointer_type));
        heap_barrier_signature.params.push(AbiParam::new(pointer_type));

        context.func.signature = signature;
        context.func.name = UserFuncName::user(0, function.as_u32());
        context.set_disasm(self.options.cranelift_dump_disassembly);

        let target_config = module.target_config();

        {
            let mut builder = FunctionBuilder::new(&mut context.func, function_context);
            let entry_block = builder.create_block();
            let loop_block = builder.create_block();

            builder.append_block_params_for_function_params(entry_block);
            let native_budget_enabled = self.options.native_loop_iteration_budget != 0
                && matches!(self.trace.terminator, TraceTerminator::LoopBack);

            builder.switch_to_block(entry_block);
            let activation = builder.block_params(entry_block)[0];
            let runtime_signature = builder.import_signature(runtime_signature);
            let heap_barrier_signature = builder.import_signature(heap_barrier_signature);

            let mut state = CraneliftLoweringState {
                builder,
                trace: self.trace,
                activation,
                pointer_type,
                runtime_signature,
                heap_barrier_signature,
                values: vec![None; self.trace.values.len()],
                heap_addresses: vec![None; self.trace.values.len()],
                heap_entries: HashMap::new(),
                list_indices: HashMap::new(),
                list_lengths: HashMap::new(),
                exit_plans: &self.exit_plans,
                native_budget_remaining: None,
                native_budget_initial: self.options.native_loop_iteration_budget,
            };

            // Entry loads/constants are invariant unless represented by a LoopPhi.
            let value_ids: Vec<_> = self.trace.values.iter().map(|value| value.id).collect();
            for id in value_ids {
                let origin = self.trace.value(id).unwrap().origin.clone();
                if matches!(
                    origin,
                    TraceValueOrigin::EntrySlot { .. }
                        | TraceValueOrigin::GuardedEntry { .. }
                        | TraceValueOrigin::Constant(_)
                ) {
                    let _ = state.lower_value(id)?;
                }
            }

            let mut preheader_nodes = vec![false; self.trace.nodes.len()];
            for &id in &self.trace.preheader {
                preheader_nodes[id] = true;
                let node = &self.trace.nodes[id];
                if !matches!(node.kind, TraceInst::Nop) {
                    state.lower_node(node)?;
                }
            }

            // Mirror the old B3 backend exactly: loop-carried values are mutable
            // frontend variables. Cranelift's frontend SSA builder inserts and
            // wires the required block parameters itself when `use_var` sees a
            // value with definitions from the preheader and backedge. This avoids
            // maintaining a second, hand-written phi/BlockArg implementation in
            // the tracing tier.
            let mut phi_vars = Vec::with_capacity(self.trace.loop_phis.len());
            for phi in &self.trace.loop_phis {
                let variable = state.builder.declare_var(clif_type(phi.rep));
                let mut initial = state.lower_value(phi.initial)?;
                if let Some(exit) = phi.entry_exit {
                    initial = match phi.rep {
                        TraceRep::F64 => state.guard_numeric(initial, exit)?,
                        TraceRep::Int32 => state.guard_int32_numeric(initial, exit)?,
                        TraceRep::UInt32 => state.guard_uint32_numeric(initial, exit)?,
                        _ => {
                            return Err(TracingJitError::CraneliftLowering {
                                trace: trace_id,
                                message: format!(
                                    "unsupported promoted LoopPhi representation {:?}",
                                    phi.rep
                                ),
                            });
                        }
                    };
                }
                state.builder.def_var(variable, initial);
                phi_vars.push(variable);
            }

            // Treat the watchdog counter as loop-carried state too. Keeping it in
            // the same SSA mechanism makes it impossible for the explicit block
            // argument list to get out of sync with the Trace IR phi list.
            let budget_var = if native_budget_enabled {
                let variable = state.builder.declare_var(types::I64);
                let initial = state.builder.ins().iconst(
                    types::I64,
                    self.options.native_loop_iteration_budget as i64,
                );
                state.builder.def_var(variable, initial);
                Some(variable)
            } else {
                None
            };

            state.builder.ins().jump(loop_block, &[]);
            state.builder.switch_to_block(loop_block);

            // A runtime operation in the body may invalidate preheader proofs on
            // the backedge, even though it follows their first body consumer.
            if self.trace.nodes.iter().any(|node| matches!(node.kind,
                TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. } | TraceInst::ReadBarrier { .. }))
            {
                state.invalidate_heap_proofs();
            }

            for (phi, variable) in self.trace.loop_phis.iter().zip(phi_vars.iter().copied()) {
                let value = state.builder.use_var(variable);
                state.cache_value(phi.value, value);
            }
            state.native_budget_remaining = budget_var.map(|variable| state.builder.use_var(variable));

            for node in &self.trace.nodes {
                if !preheader_nodes[node.id] && !matches!(node.kind, TraceInst::Nop) {
                    state.lower_node(node)?;
                }
            }

            match &self.trace.terminator {
                TraceTerminator::RuntimeReturn => {
                    let result = state
                        .builder
                        .ins()
                        .iconst(types::I32, TRACE_RESULT_RUNTIME as i64);
                    state.builder.ins().return_(&[result]);
                }
                TraceTerminator::LoopBack => {
                    // Define every loop-carried variable with the value produced by
                    // this iteration. `use_var` at the top of `loop_block` then
                    // observes the appropriate SSA phi on the next iteration.
                    for (phi, variable) in self.trace.loop_phis.iter().zip(phi_vars.iter().copied()) {
                        let backedge = phi.backedge.ok_or_else(|| {
                            TracingJitError::CraneliftLowering {
                                trace: trace_id,
                                message: format!("{:?} has no backedge", phi.value),
                            }
                        })?;
                        let value = state.lower_value(backedge)?;
                        state.builder.def_var(variable, value);
                    }

                    if let Some(variable) = budget_var {
                        let budget = state
                            .native_budget_remaining
                            .ok_or_else(|| TracingJitError::CraneliftLowering {
                                trace: trace_id,
                                message: "native loop budget variable has no current SSA value".to_owned(),
                            })?;
                        let next_budget = state.builder.ins().iadd_imm(budget, -1);
                        state.builder.def_var(variable, next_budget);
                        let exhausted = state.builder.ins().icmp_imm(
                            IntCC::Equal,
                            next_budget,
                            0,
                        );
                        let exhausted = state.bool32(exhausted);
                        let stop = state.builder.create_block();
                        state.builder.set_cold_block(stop);
                        state.builder.ins().brif(
                            exhausted,
                            stop,
                            &[],
                            loop_block,
                            &[],
                        );
                        state.builder.switch_to_block(stop);
                        let exit = self.trace.native_budget_exit.ok_or_else(|| {
                            TracingJitError::CraneliftLowering {
                                trace: trace_id,
                                message: "native loop budget enabled without watchdog side exit".to_owned(),
                            }
                        })?;
                        state.native_budget_remaining = Some(next_budget);
                        state.emit_unconditional_side_exit(exit)?;
                    } else {
                        state.builder.ins().jump(loop_block, &[]);
                    }
                }
                TraceTerminator::Exit(exit) => {
                    state.emit_unconditional_side_exit(*exit)?;
                }
                TraceTerminator::Return => {
                    return Err(TracingJitError::CraneliftLowering {
                        trace: trace_id,
                        message: "native Return terminator is not enabled in the first loop tier"
                            .to_owned(),
                    });
                }
                TraceTerminator::Link => {
                    return Err(TracingJitError::CraneliftLowering {
                        trace: trace_id,
                        message: "direct native trace links are installed after compilation"
                            .to_owned(),
                    });
                }
                TraceTerminator::Unreachable => {
                    return Err(TracingJitError::CraneliftLowering {
                        trace: trace_id,
                        message: "trace has unreachable terminator".to_owned(),
                    });
                }
            }

            state.builder.seal_all_blocks();
            state.builder.finalize(target_config);
        }

        let clif_text = self
            .options
            .cranelift_dump_ir
            .then(|| context.func.display().to_string());
        if let Some(text) = &clif_text {
            eprintln!("{text}");
        }

        module
            .define_function(function, context)
            .map_err(|error| TracingJitError::CraneliftLowering {
                trace: trace_id,
                message: format!("native code generation failed: {error:?}"),
            })?;

        let disassembly = if self.options.cranelift_dump_disassembly {
            context
                .compiled_code()
                .and_then(|compiled| compiled.vcode.clone())
        } else {
            None
        };
        if let Some(text) = &disassembly {
            eprintln!("{text}");
        }

        module
            .finalize_definitions()
            .map_err(|error| TracingJitError::CraneliftLowering {
                trace: trace_id,
                message: format!("cannot finalize native trace: {error}"),
            })?;

        let code_ptr = module.get_finalized_function(function);
        let entry: NativeTraceFn = unsafe { std::mem::transmute(code_ptr) };

        Ok(CompiledTrace {
            trace: self.trace.clone(),
            function,
            code_ptr,
            entry,
            exit_hits: vec![0; self.exit_plans.len()],
            side_trace_attempted: vec![false; self.exit_plans.len()],
            native_links: vec![u32::MAX; self.exit_plans.len()].into_boxed_slice(),
            root_trace: trace_id,
            tree_depth: 0,
            exits: self.exit_plans,
            short_exit_streak: 0,
            debug_text: disassembly.or(clif_text),
        })
    }
}

impl CraneliftLoweringState<'_, '_> {
    #[inline(always)]
    fn cached_value(&self, id: TraceValueId) -> Option<Value> {
        self.values.get(id.0 as usize).and_then(|value| *value)
    }

    #[inline(always)]
    fn cache_value(&mut self, id: TraceValueId, value: Value) {
        let slot = self.values.get_mut(id.0 as usize).expect("trace value id out of range");
        *slot = Some(value);
    }

    #[inline]
    fn trusted_mem() -> MemFlagsData {
        MemFlagsData::trusted()
    }

    #[inline]
    fn plain_mem() -> MemFlagsData {
        MemFlagsData::new()
    }

    fn error(&self, message: impl Into<String>) -> TracingJitError {
        TracingJitError::CraneliftLowering {
            trace: self.trace.id,
            message: message.into(),
        }
    }

    fn value_rep(&self, value: TraceValueId) -> Result<TraceRep, TracingJitError> {
        self.trace
            .value(value)
            .map(|value| value.rep)
            .ok_or_else(|| self.error(format!("unknown trace value {value:?}")))
    }

    fn lower_value(&mut self, value: TraceValueId) -> Result<Value, TracingJitError> {
        let value = self.trace.canonical_value(value);
        if let Some(existing) = self.cached_value(value) {
            return Ok(existing);
        }

        let (origin, value_rep) = {
            let trace_value = self
                .trace
                .value(value)
                .ok_or_else(|| self.error(format!("unknown trace value {value:?}")))?;
            (trace_value.origin.clone(), trace_value.rep)
        };

        let lowered = match origin {
            TraceValueOrigin::EntrySlot { frame, slot } => {
                if frame.0 != 0 {
                    return Err(self.error(
                        "Cranelift tier cannot load recreated inlined frames yet",
                    ));
                }
                self.load_frame_slot(slot, value_rep)
            }
            TraceValueOrigin::GuardedEntry {
                value: source,
                exit,
            } => {
                let tagged = self.lower_value(source)?;
                match value_rep {
                    TraceRep::F64 => self.guard_numeric(tagged, exit)?,
                    TraceRep::Int32 => self.guard_int32_numeric(tagged, exit)?,
                    TraceRep::UInt32 => self.guard_uint32_numeric(tagged, exit)?,
                    _ => {
                        return Err(self.error(format!(
                            "unsupported guarded-entry representation {value_rep:?}"
                        )));
                    }
                }
            }
            TraceValueOrigin::Constant(constant) => self.lower_constant(constant)?,
            TraceValueOrigin::InstResult { .. } => {
                return Err(self.error(format!(
                    "instruction result {value:?} requested before producer was lowered"
                )));
            }
            TraceValueOrigin::LoopPhi { .. } | TraceValueOrigin::LoopTemporary => {
                return Err(self.error(format!(
                    "loop value {value:?} has no Cranelift block parameter"
                )));
            }
            TraceValueOrigin::Alias(alias) => return self.lower_value(alias),
        };

        self.cache_value(value, lowered);
        Ok(lowered)
    }

    fn base_for_slot(&mut self, slot: FrameSlot) -> (Value, i32) {
        let (bank_offset, slot_offset) = match slot {
            FrameSlot::General(reg) => (
                offset_of!(TraceActivation, general),
                reg.0 as i32 * std::mem::size_of::<RegisterWord>() as i32,
            ),
            FrameSlot::Argument(reg) => (
                offset_of!(TraceActivation, arguments),
                reg.0 as i32 * std::mem::size_of::<RegisterWord>() as i32,
            ),
            FrameSlot::Return(reg) => (
                offset_of!(TraceActivation, returns),
                reg.0 as i32 * std::mem::size_of::<RegisterWord>() as i32,
            ),
            FrameSlot::Special(reg) => (
                offset_of!(TraceActivation, special),
                reg as u8 as i32 * std::mem::size_of::<RegisterWord>() as i32,
            ),
        };

        // Runtime helpers may replace the active frame/fiber and refresh these
        // pointers in TraceActivation. Reload the bank pointer for every access.
        let base = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            self.activation,
            bank_offset as i32,
        );
        (base, slot_offset)
    }

    fn load_frame_slot(&mut self, slot: FrameSlot, rep: TraceRep) -> Value {
        let (base, offset) = self.base_for_slot(slot);
        let raw = self
            .builder
            .ins()
            .load(types::I64, Self::trusted_mem(), base, offset);

        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => raw,
            TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => self.builder.ins().ireduce(types::I32, raw),
            TraceRep::F64 => self
                .builder
                .ins()
                .bitcast(types::F64, Self::plain_mem(), raw),
        }
    }

    fn store_frame_slot(&mut self, slot: FrameSlot, value: Value, rep: TraceRep) {
        let raw = self.normalize_to_word(value, rep);
        self.store_frame_word(slot, raw);
    }

    #[inline]
    fn store_frame_word(&mut self, slot: FrameSlot, word: Value) {
        let (base, offset) = self.base_for_slot(slot);
        self.builder
            .ins()
            .store(Self::trusted_mem(), word, base, offset);
    }

    fn lower_constant(&mut self, constant: TraceConstant) -> Result<Value, TracingJitError> {
        Ok(match constant {
            TraceConstant::Null => self.builder.ins().iconst(types::I64, TAG_NULL as i64),
            TraceConstant::Bool(value) => self.builder.ins().iconst(
                types::I64,
                if value { TAG_TRUE } else { TAG_FALSE } as i64,
            ),
            TraceConstant::NumBits(bits) => self.builder.ins().iconst(types::I64, bits as i64),
            TraceConstant::Int32(value) => {
                self.builder.ins().iconst(types::I32, value as i64)
            }
            TraceConstant::F64Bits(bits) => self
                .builder
                .ins()
                .f64const(Ieee64::with_bits(bits)),
            TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                return Err(self.error("runtime-owned constant reached native tier"));
            }
        })
    }

    fn normalize_to_word(&mut self, value: Value, rep: TraceRep) -> Value {
        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => value,
            TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => self.builder.ins().uextend(types::I64, value),
            TraceRep::F64 => self
                .builder
                .ins()
                .bitcast(types::I64, Self::plain_mem(), value),
        }
    }

    fn materialize_exit_word(
        &mut self,
        value: Value,
        source_rep: TraceRep,
        target_rep: TraceRep,
    ) -> Result<Value, TracingJitError> {
        if source_rep == target_rep {
            return Ok(self.normalize_to_word(value, source_rep));
        }

        Ok(match (source_rep, target_rep) {
            (TraceRep::F64, TraceRep::Tagged) => self.box_f64(value),
            (TraceRep::Int32, TraceRep::Tagged) => {
                let number = self.builder.ins().fcvt_from_sint(types::F64, value);
                self.builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number)
            }
            (TraceRep::UInt32, TraceRep::Tagged) => {
                let wide = self.builder.ins().uextend(types::I64, value);
                let number = self.builder.ins().fcvt_from_uint(types::F64, wide);
                self.builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number)
            }
            (TraceRep::Bool, TraceRep::Tagged) => self.box_bool(value),
            (TraceRep::HeapHandle, TraceRep::Tagged) => {
                let tag = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
                self.builder.ins().bor(value, tag)
            }
            (TraceRep::Int32, TraceRep::F64) => {
                let number = self.builder.ins().fcvt_from_sint(types::F64, value);
                self.builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number)
            }
            (TraceRep::UInt32, TraceRep::F64) => {
                let wide = self.builder.ins().uextend(types::I64, value);
                let number = self.builder.ins().fcvt_from_uint(types::F64, wide);
                self.builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number)
            }
            (TraceRep::Bool, TraceRep::Int32) => self.builder.ins().uextend(types::I64, value),
            (TraceRep::Tagged, TraceRep::HeapHandle) => {
                let payload = self
                    .builder
                    .ins()
                    .iconst(types::I64, PAYLOAD_MASK as i64);
                self.builder.ins().band(value, payload)
            }
            _ => {
                return Err(self.error(format!(
                    "cannot materialize side-exit value from {source_rep:?} to {target_rep:?}"
                )));
            }
        })
    }

    #[inline]
    fn bool32(&mut self, value: Value) -> Value {
        let ty = self.builder.func.dfg.value_type(value);
        if ty == types::I32 {
            value
        } else {
            self.builder.ins().uextend(types::I32, value)
        }
    }

    fn bool_not(&mut self, value: Value) -> Value {
        let equal = self.builder.ins().icmp_imm_s(IntCC::Equal, value, 0);
        self.bool32(equal)
    }

    fn bool_and(&mut self, lhs: Value, rhs: Value) -> Value {
        self.builder.ins().band(lhs, rhs)
    }

    fn bool_or(&mut self, lhs: Value, rhs: Value) -> Value {
        self.builder.ins().bor(lhs, rhs)
    }

    fn int_compare(&mut self, op: CmpKind, lhs: Value, rhs: Value) -> Value {
        let cond = match op {
            CmpKind::Eq => IntCC::Equal,
            CmpKind::Ne => IntCC::NotEqual,
            CmpKind::Lt => IntCC::SignedLessThan,
            CmpKind::Le => IntCC::SignedLessThanOrEqual,
            CmpKind::Gt => IntCC::SignedGreaterThan,
            CmpKind::Ge => IntCC::SignedGreaterThanOrEqual,
        };
        let result = self.builder.ins().icmp(cond, lhs, rhs);
        self.bool32(result)
    }

    fn f64_compare(&mut self, op: CmpKind, lhs: Value, rhs: Value) -> Value {
        let cond = match op {
            CmpKind::Eq => FloatCC::Equal,
            CmpKind::Ne => FloatCC::NotEqual,
            CmpKind::Lt => FloatCC::LessThan,
            CmpKind::Le => FloatCC::LessThanOrEqual,
            CmpKind::Gt => FloatCC::GreaterThan,
            CmpKind::Ge => FloatCC::GreaterThanOrEqual,
        };
        let result = self.builder.ins().fcmp(cond, lhs, rhs);
        self.bool32(result)
    }

    fn tagged_is_num(&mut self, bits: Value) -> Value {
        let mask = self.builder.ins().iconst(types::I64, TAG_MASK as i64);
        let base = self.builder.ins().iconst(types::I64, TAG_BASE as i64);
        let masked = self.builder.ins().band(bits, mask);
        let not_tag = self
            .builder
            .ins()
            .icmp(IntCC::NotEqual, masked, base);
        let not_tag = self.bool32(not_tag);

        let heap_mask = self
            .builder
            .ins()
            .iconst(types::I64, TAG_HEAP_MASK as i64);
        let heap_tag = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
        let heap_masked = self.builder.ins().band(bits, heap_mask);
        let not_heap = self
            .builder
            .ins()
            .icmp(IntCC::NotEqual, heap_masked, heap_tag);
        let not_heap = self.bool32(not_heap);

        self.bool_and(not_tag, not_heap)
    }

    fn tagged_falsey(&mut self, bits: Value) -> Value {
        let null = self.builder.ins().iconst(types::I64, TAG_NULL as i64);
        let false_ = self.builder.ins().iconst(types::I64, TAG_FALSE as i64);
        let is_null = self.builder.ins().icmp(IntCC::Equal, bits, null);
        let is_false = self.builder.ins().icmp(IntCC::Equal, bits, false_);
        let is_null = self.bool32(is_null);
        let is_false = self.bool32(is_false);
        self.bool_or(is_null, is_false)
    }

    fn box_bool(&mut self, condition: Value) -> Value {
        let bit = self.builder.ins().uextend(types::I64, condition);
        let base = self.builder.ins().iconst(types::I64, TAG_FALSE as i64);
        self.builder.ins().iadd(base, bit)
    }

    fn box_f64(&mut self, value: Value) -> Value {
        let is_nan = self.builder.ins().fcmp(FloatCC::Unordered, value, value);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, Self::plain_mem(), value);
        let canonical = self
            .builder
            .ins()
            .iconst(types::I64, CANONICAL_NAN as i64);
        self.builder.ins().select(is_nan, canonical, bits)
    }

    fn guard_numeric(
        &mut self,
        tagged: Value,
        exit: SideExitId,
    ) -> Result<Value, TracingJitError> {
        let ok = self.tagged_is_num(tagged);
        let failure = self.bool_not(ok);
        self.emit_side_exit(exit, failure)?;
        Ok(self
            .builder
            .ins()
            .bitcast(types::F64, Self::plain_mem(), tagged))
    }

    fn guard_int32_numeric(
        &mut self,
        tagged: Value,
        exit: SideExitId,
    ) -> Result<Value, TracingJitError> {
        let number = self.guard_numeric(tagged, exit)?;

        let negative_zero = self
            .builder
            .ins()
            .iconst(types::I64, (-0.0f64).to_bits() as i64);
        let is_negative_zero = self
            .builder
            .ins()
            .icmp(IntCC::Equal, tagged, negative_zero);
        let is_negative_zero = self.bool32(is_negative_zero);
        self.emit_side_exit(exit, is_negative_zero)?;

        let min = self
            .builder
            .ins()
            .f64const(Ieee64::with_bits((i32::MIN as f64).to_bits()));
        let max = self
            .builder
            .ins()
            .f64const(Ieee64::with_bits((i32::MAX as f64).to_bits()));
        let ge_min = self
            .builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, number, min);
        let le_max = self
            .builder
            .ins()
            .fcmp(FloatCC::LessThanOrEqual, number, max);
        let ge_min = self.bool32(ge_min);
        let le_max = self.bool32(le_max);
        let in_range = self.bool_and(ge_min, le_max);
        let failure = self.bool_not(in_range);
        self.emit_side_exit(exit, failure)?;

        // This conversion is non-trapping on the surviving path because the
        // preceding ordered range checks reject NaN and out-of-range values.
        let i64_value = self.builder.ins().fcvt_to_sint(types::I64, number);
        let back = self
            .builder
            .ins()
            .fcvt_from_sint(types::F64, i64_value);
        let exact = self.builder.ins().fcmp(FloatCC::Equal, number, back);
        let exact = self.bool32(exact);
        let not_exact = self.bool_not(exact);
        self.emit_side_exit(exit, not_exact)?;

        Ok(self.builder.ins().ireduce(types::I32, i64_value))
    }

    fn guard_uint32_numeric(
        &mut self,
        tagged: Value,
        exit: SideExitId,
    ) -> Result<Value, TracingJitError> {
        let number = self.guard_numeric(tagged, exit)?;

        let zero = self.builder.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
        let max = self.builder.ins().f64const(Ieee64::with_bits((u32::MAX as f64).to_bits()));
        let ge_zero = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, number, zero);
        let le_max = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, number, max);
        let ge_zero = self.bool32(ge_zero);
        let le_max = self.bool32(le_max);
        let in_range = self.bool_and(ge_zero, le_max);
        let failure = self.bool_not(in_range);
        self.emit_side_exit(exit, failure)?;

        // Range checks make the unsigned conversion non-trapping. Verify the
        // round-trip so fractional values side-exit to Wren's full ToUInt32 path.
        let wide = self.builder.ins().fcvt_to_uint(types::I64, number);
        let back = self.builder.ins().fcvt_from_uint(types::F64, wide);
        let exact = self.builder.ins().fcmp(FloatCC::Equal, number, back);
        let exact = self.bool32(exact);
        let not_exact = self.bool_not(exact);
        self.emit_side_exit(exit, not_exact)?;

        Ok(self.builder.ins().ireduce(types::I32, wide))
    }

    fn emit_side_exit(
        &mut self,
        exit_id: SideExitId,
        failure_condition: Value,
    ) -> Result<(), TracingJitError> {
        let plan = self
            .exit_plans
            .get(exit_id.0 as usize)
            .ok_or_else(|| self.error(format!("unknown side exit {exit_id:?}")))?
            .clone();

        let stop = self.builder.create_block();
        let resume = self.builder.create_block();
        self.builder.set_cold_block(stop);
        let failure_condition = self.bool32(failure_condition);
        self.builder
            .ins()
            .brif(failure_condition, stop, &[], resume, &[]);

        self.builder.switch_to_block(stop);
        self.materialize_exit_or_native_link(&plan)?;

        self.builder.switch_to_block(resume);
        Ok(())
    }

    fn emit_unconditional_side_exit(
        &mut self,
        exit_id: SideExitId,
    ) -> Result<(), TracingJitError> {
        let plan = self
            .exit_plans
            .get(exit_id.0 as usize)
            .ok_or_else(|| self.error(format!("unknown side exit {exit_id:?}")))?
            .clone();
        self.materialize_exit_or_native_link(&plan)
    }

    /// Materialize an exit either into a stitched child trace or back to Rust.
    /// This is shared by conditional guard exits and unconditional terminators so
    /// hot branch side traces get the same native-to-native path as loop tails.
    fn materialize_exit_or_native_link(
        &mut self,
        plan: &CompiledExitPlan,
    ) -> Result<(), TracingJitError> {
        if !plan.native_linkable {
            return self.materialize_side_exit(plan);
        }

        // Stitching updates a stable per-trace u32 child table. Generated code
        // probes its own exit slot directly; u32::MAX means the edge still needs
        // manager-level processing. Once stitched, frame state is materialized and
        // TRACE_RESULT_LINK returns to the native dispatcher, not to Rust.
        let links = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            self.activation,
            offset_of!(TraceActivation, current_links) as i32,
        );
        let child = self.builder.ins().load(
            types::I32,
            Self::trusted_mem(),
            links,
            (plan.id.0 as i32).saturating_mul(std::mem::size_of::<u32>() as i32),
        );
        let linked = self.builder.ins().icmp_imm(IntCC::NotEqual, child, -1);
        let linked_block = self.builder.create_block();
        let slow_exit = self.builder.create_block();
        self.builder.set_cold_block(slow_exit);
        self.builder.ins().brif(linked, linked_block, &[], slow_exit, &[]);

        self.builder.switch_to_block(linked_block);
        self.materialize_native_link(plan, child)?;

        self.builder.switch_to_block(slow_exit);
        self.materialize_side_exit(plan)
    }

    fn materialize_native_link(
        &mut self,
        plan: &CompiledExitPlan,
        child: Value,
    ) -> Result<(), TracingJitError> {
        let completed = if let Some(remaining) = self.native_budget_remaining {
            let initial = self.builder.ins().iconst(
                types::I64,
                self.native_budget_initial as i64,
            );
            let completed = self.builder.ins().isub(initial, remaining);
            self.builder.ins().ireduce(types::I32, completed)
        } else {
            self.builder.ins().iconst(types::I32, 0)
        };
        self.builder.ins().store(
            Self::trusted_mem(),
            completed,
            self.activation,
            offset_of!(TraceActivation, native_iterations) as i32,
        );

        for write in &plan.writes {
            let word = match write.source {
                ExitWriteSource::Dynamic(index) => {
                    let dynamic = plan
                        .dynamic_values
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| self.error("side-exit dynamic value index out of bounds"))?;
                    let source_rep = self.value_rep(dynamic.value)?;
                    let value = self.lower_value(dynamic.value)?;
                    self.materialize_exit_word(value, source_rep, dynamic.target_rep)?
                }
                ExitWriteSource::Constant(constant) => {
                    let source_rep = match constant {
                        TraceConstant::Int32(_) => TraceRep::Int32,
                        TraceConstant::F64Bits(_) => TraceRep::F64,
                        TraceConstant::Null | TraceConstant::Bool(_) | TraceConstant::NumBits(_) => TraceRep::Tagged,
                        TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                            return Err(self.error("runtime-owned constant reached native trace link"));
                        }
                    };
                    let value = self.lower_constant(constant)?;
                    self.materialize_exit_word(value, source_rep, write.target_rep)?
                }
            };
            self.store_frame_word(write.slot, word);
        }

        self.builder.ins().store(
            Self::trusted_mem(),
            child,
            self.activation,
            offset_of!(TraceActivation, next_trace) as i32,
        );
        let result = self.builder.ins().iconst(types::I32, TRACE_RESULT_LINK as i64);
        self.builder.ins().return_(&[result]);
        Ok(())
    }

    fn materialize_side_exit(&mut self, plan: &CompiledExitPlan) -> Result<(), TracingJitError> {
        // Preserve a cheap measure of how much useful native work this entry did.
        // The counter stays in SSA; only cold side exits write it to activation.
        let completed = if let Some(remaining) = self.native_budget_remaining {
            let initial = self.builder.ins().iconst(
                types::I64,
                self.native_budget_initial as i64,
            );
            let completed = self.builder.ins().isub(initial, remaining);
            self.builder.ins().ireduce(types::I32, completed)
        } else {
            self.builder.ins().iconst(types::I32, 0)
        };
        self.builder.ins().store(
            Self::trusted_mem(),
            completed,
            self.activation,
            offset_of!(TraceActivation, native_iterations) as i32,
        );

        // Dynamic snapshot values are already available in native SSA. Commit
        // them directly to the architectural register banks on the cold exit path
        // instead of bouncing through TraceActivation::exit_values and copying them
        // again in Rust. Runtime-owned constants remain a Rust-side concern.
        for write in &plan.writes {
            let ExitWriteSource::Dynamic(index) = write.source else { continue; };
            let dynamic = plan
                .dynamic_values
                .get(index as usize)
                .copied()
                .ok_or_else(|| self.error("side-exit dynamic value index out of bounds"))?;
            let source_rep = self.value_rep(dynamic.value)?;
            let value = self.lower_value(dynamic.value)?;
            let word = self.materialize_exit_word(value, source_rep, dynamic.target_rep)?;
            self.store_frame_word(write.slot, word);
        }

        let id = self
            .builder
            .ins()
            .iconst(types::I32, plan.id.0 as i64);
        self.builder.ins().store(
            Self::trusted_mem(),
            id,
            self.activation,
            offset_of!(TraceActivation, exit_id) as i32,
        );
        let result = self
            .builder
            .ins()
            .iconst(types::I32, TRACE_RESULT_SIDE_EXIT as i64);
        self.builder.ins().return_(&[result]);
        Ok(())
    }

    fn call_scalar_helper(&mut self, address: usize, args: &[Value]) -> Value {
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        signature.params.extend(args.iter().map(|_| AbiParam::new(types::F64)));
        signature.returns.push(AbiParam::new(types::F64));
        let signature = self.builder.import_signature(signature);
        let callee = self.builder.ins().iconst(self.pointer_type, address as i64);
        let call = self.builder.ins().call_indirect(signature, callee, args);
        self.builder.inst_results(call)[0]
    }

    fn trace_value_may_be_heap(&self, value: TraceValueId) -> Result<bool, TracingJitError> {
        let value = self.trace.value(value).ok_or_else(|| self.error("unknown heap-store value"))?;
        Ok(matches!(value.ty, TraceType::Any))
    }

    /// Guard a tagged Wren value as a particular JIT-visible heap kind and return
    /// `(entry_ptr, current_gc_object_address, packed_metadata)`. The handle-table
    /// entry is stable; the MMTk object address itself is reloaded on every native
    /// operation.
    fn guard_heap_entry_with_metadata(
        &mut self,
        tagged: Value,
        expected: TraceHeapKind,
        exit: SideExitId,
    ) -> Result<(Value, Value, Value), TracingJitError> {
        let (entry, address) = self.guard_live_heap_entry(tagged, exit)?;
        let metadata = self.builder.ins().load(
            types::I32,
            Self::trusted_mem(),
            entry,
            offset_of!(TraceHeapEntry, kind) as i32,
        );
        let kind_mask = self
            .builder
            .ins()
            .iconst(types::I32, TRACE_HEAP_KIND_MASK as i64);
        let kind = self.builder.ins().band(metadata, kind_mask);
        let matches_kind = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::Equal, kind, expected as i64);
        let matches_kind = self.bool32(matches_kind);
        let wrong_kind = self.bool_not(matches_kind);
        self.emit_side_exit(exit, wrong_kind)?;
        Ok((entry, address, metadata))
    }

    fn guard_heap_entry(
        &mut self,
        tagged: Value,
        expected: TraceHeapKind,
        exit: SideExitId,
    ) -> Result<(Value, Value), TracingJitError> {
        let (entry, address, _) =
            self.guard_heap_entry_with_metadata(tagged, expected, exit)?;
        Ok((entry, address))
    }

    #[inline]
    fn runtime_heap_kind(kind: TraceHeapObjectKind) -> TraceHeapKind {
        match kind {
            TraceHeapObjectKind::String => TraceHeapKind::String,
            TraceHeapObjectKind::List => TraceHeapKind::List,
            TraceHeapObjectKind::Map => TraceHeapKind::Map,
            TraceHeapObjectKind::Cell => TraceHeapKind::Cell,
            TraceHeapObjectKind::Instance => TraceHeapKind::Instance,
            TraceHeapObjectKind::Promise => TraceHeapKind::Promise,
        }
    }

    fn proven_heap_entry_kind(&self, value: TraceValueId) -> Option<TraceHeapObjectKind> {
        let value = self.trace.canonical_value(value);
        let TraceValueOrigin::InstResult { node } = &self.trace.value(value)?.origin else {
            return None;
        };
        match &self.trace.nodes.get(*node)?.kind {
            TraceInst::GuardHeapObject { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    /// Resolve a heap operand that may already be a stable sidecar entry produced
    /// by GuardHeapObject. The backend may reuse the current moving address inside
    /// one GC epoch; every full runtime/call boundary invalidates that local cache.
    fn heap_entry_for_value(
        &mut self,
        value: TraceValueId,
        expected: TraceHeapKind,
        exit: SideExitId,
    ) -> Result<(Value, Value), TracingJitError> {
        let canonical = self.trace.canonical_value(value);
        if let Some(&entry) = self.heap_entries.get(&(canonical, expected as u32)) {
            return Ok(entry);
        }
        let proven = self
            .proven_heap_entry_kind(canonical)
            .map(Self::runtime_heap_kind)
            == Some(expected);
        if proven {
            let entry = self.lower_value(canonical)?;
            if let Some(address) = self
                .heap_addresses
                .get(canonical.0 as usize)
                .and_then(|address| *address)
            {
                return Ok((entry, address));
            }
            // This path is used only when a structural transform separated the
            // proof from its consumer without an available backend cache. Reload
            // and validate liveness from the stable handle-table entry.
            let address = self.builder.ins().load(
                self.pointer_type,
                Self::trusted_mem(),
                entry,
                offset_of!(TraceHeapEntry, address) as i32,
            );
            let dead = self.builder.ins().icmp_imm_u(IntCC::Equal, address, 0);
            let dead = self.bool32(dead);
            self.emit_side_exit(exit, dead)?;
            return Ok((entry, address));
        }
        let tagged = self.lower_value(canonical)?;
        let entry = self.guard_heap_entry(tagged, expected, exit)?;
        self.heap_entries.insert((canonical, expected as u32), entry);
        Ok(entry)
    }

    fn heap_entry_with_metadata_for_value(
        &mut self,
        value: TraceValueId,
        expected: TraceHeapKind,
        exit: SideExitId,
    ) -> Result<(Value, Value, Value), TracingJitError> {
        let (entry, address) = self.heap_entry_for_value(value, expected, exit)?;
        let metadata = self.builder.ins().load(types::I32, Self::trusted_mem(), entry,
            offset_of!(TraceHeapEntry, kind) as i32);
        Ok((entry, address, metadata))
    }

    /// Resolve a value already proven to carry the heap tag. The returned
    /// sidecar entry is stable, while `address` is valid only until the next full
    /// runtime/GC epoch.
    fn live_heap_entry_from_heap_tagged(
        &mut self,
        tagged: Value,
        exit: SideExitId,
    ) -> Result<(Value, Value), TracingJitError> {
        let payload_mask = self.builder.ins().iconst(types::I64, PAYLOAD_MASK as i64);
        let handle = self.builder.ins().band(tagged, payload_mask);
        let chunk_index = self
            .builder
            .ins()
            .ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
        let entry_index_mask = self
            .builder
            .ins()
            .iconst(types::I64, TRACE_HEAP_CHUNK_MASK as i64);
        let entry_index = self.builder.ins().band(handle, entry_index_mask);

        let chunk_stride = self
            .builder
            .ins()
            .iconst(types::I64, std::mem::size_of::<usize>() as i64);
        let chunk_offset = self.builder.ins().imul(chunk_index, chunk_stride);
        let chunks = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            self.activation,
            offset_of!(TraceActivation, heap_chunks) as i32,
        );
        let chunk_slot = self.builder.ins().iadd(chunks, chunk_offset);
        let chunk = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            chunk_slot,
            0,
        );

        let entry_stride = self
            .builder
            .ins()
            .iconst(types::I64, std::mem::size_of::<TraceHeapEntry>() as i64);
        let entry_offset = self.builder.ins().imul(entry_index, entry_stride);
        let entry = self.builder.ins().iadd(chunk, entry_offset);

        let address = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            entry,
            offset_of!(TraceHeapEntry, address) as i32,
        );
        let dead = self.builder.ins().icmp_imm_u(IntCC::Equal, address, 0);
        let dead = self.bool32(dead);
        self.emit_side_exit(exit, dead)?;

        Ok((entry, address))
    }

    fn guard_live_heap_entry(
        &mut self,
        tagged: Value,
        exit: SideExitId,
    ) -> Result<(Value, Value), TracingJitError> {
        let heap_mask = self.builder.ins().iconst(types::I64, TAG_HEAP_MASK as i64);
        let heap_tag = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
        let masked = self.builder.ins().band(tagged, heap_mask);
        let is_heap = self.builder.ins().icmp(IntCC::Equal, masked, heap_tag);
        let is_heap = self.bool32(is_heap);
        let not_heap = self.bool_not(is_heap);
        self.emit_side_exit(exit, not_heap)?;

        self.live_heap_entry_from_heap_tagged(tagged, exit)
    }

    #[inline(always)]
    fn heap_metadata_payload(&mut self, metadata: Value) -> Value {
        self.builder
            .ins()
            .ushr_imm_u(metadata, TRACE_HEAP_METADATA_SHIFT as i64)
    }

    fn guard_map_shape_metadata(
        &mut self,
        metadata: Value,
        expected: u32,
        exit: SideExitId,
    ) -> Result<(), TracingJitError> {
        let shape = self.heap_metadata_payload(metadata);
        let matches = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::Equal, shape, expected as i64);
        let matches = self.bool32(matches);
        let fail = self.bool_not(matches);
        self.emit_side_exit(exit, fail)
    }

    /// Guard a dynamic Wren String against canonical shape-key bytes. The heap
    /// handle itself is deliberately not compared: handles are recyclable. The
    /// String sidecar packs UTF-8 byte length in the metadata high bits and keeps
    /// an Arc-backed stable data pointer, so the guard is exact and GC-safe.
    fn guard_string_bytes(
        &mut self,
        tagged: Value,
        expected: &[u8],
        exit: SideExitId,
    ) -> Result<(), TracingJitError> {
        debug_assert!(expected.len() <= 64);
        let (entry, _, metadata) =
            self.guard_heap_entry_with_metadata(tagged, TraceHeapKind::String, exit)?;
        let byte_len = self.heap_metadata_payload(metadata);
        let same_len = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::Equal, byte_len, expected.len() as i64);
        let same_len = self.bool32(same_len);
        let wrong_len = self.bool_not(same_len);
        self.emit_side_exit(exit, wrong_len)?;

        let data = self.heap_entry_data(entry);
        let mut offset = 0usize;
        while offset + 8 <= expected.len() {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&expected[offset..offset + 8]);
            let actual = self.builder.ins().load(
                types::I64,
                Self::trusted_mem(),
                data,
                offset as i32,
            );
            let wanted = self
                .builder
                .ins()
                .iconst(types::I64, u64::from_ne_bytes(chunk) as i64);
            let equal = self.builder.ins().icmp(IntCC::Equal, actual, wanted);
            let equal = self.bool32(equal);
            let fail = self.bool_not(equal);
            self.emit_side_exit(exit, fail)?;
            offset += 8;
        }
        while offset < expected.len() {
            let actual = self.builder.ins().load(
                types::I8,
                Self::trusted_mem(),
                data,
                offset as i32,
            );
            let equal = self
                .builder
                .ins()
                .icmp_imm_u(IntCC::Equal, actual, expected[offset] as i64);
            let equal = self.bool32(equal);
            let fail = self.bool_not(equal);
            self.emit_side_exit(exit, fail)?;
            offset += 1;
        }
        Ok(())
    }

    fn heap_entry_data(&mut self, entry: Value) -> Value {
        self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            entry,
            offset_of!(TraceHeapEntry, data) as i32,
        )
    }

    fn heap_entry_len(&mut self, entry: Value) -> Value {
        self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            entry,
            offset_of!(TraceHeapEntry, len) as i32,
        )
    }

    fn heap_entry_promise_value(&mut self, entry: Value) -> Value {
        self.builder.ins().load(
            types::I64,
            Self::trusted_mem(),
            entry,
            offset_of!(TraceHeapEntry, promise_value) as i32,
        )
    }

    fn guard_constant_index(
        &mut self,
        len: Value,
        index: u32,
        exit: SideExitId,
    ) -> Result<(), TracingJitError> {
        let in_bounds = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::UnsignedGreaterThan, len, index as i64);
        let in_bounds = self.bool32(in_bounds);
        let fail = self.bool_not(in_bounds);
        self.emit_side_exit(exit, fail)
    }

    fn normalize_list_index(
        &mut self,
        index: Value,
        len: Value,
        exit: SideExitId,
    ) -> Result<Value, TracingJitError> {
        let index = self.builder.ins().sextend(types::I64, index);
        let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, index, 0);
        let from_end = self.builder.ins().iadd(len, index);
        let normalized = self.builder.ins().select(negative, from_end, index);

        let non_negative = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, normalized, 0);
        let below_len = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, normalized, len);
        let non_negative = self.bool32(non_negative);
        let below_len = self.bool32(below_len);
        let in_bounds = self.bool_and(non_negative, below_len);
        let fail = self.bool_not(in_bounds);
        self.emit_side_exit(exit, fail)?;
        Ok(normalized)
    }

    fn invalidate_heap_proofs(&mut self) {
        self.heap_addresses.fill(None);
        self.heap_entries.clear();
        self.list_indices.clear();
        self.list_lengths.clear();
    }

    fn validated_list_index(
        &mut self, list: TraceValueId, index: TraceValueId, entry: Value, exit: SideExitId,
    ) -> Result<Value, TracingJitError> {
        let key = (self.trace.canonical_value(list), self.trace.canonical_value(index));
        if let Some(&normalized) = self.list_indices.get(&key) {
            return Ok(normalized);
        }
        let value = self.lower_value(index)?;
        let len = self.heap_entry_len(entry);
        let normalized = self.normalize_list_index(value, len, exit)?;
        self.list_indices.insert(key, normalized);
        Ok(normalized)
    }

    fn heap_element_address(&mut self, data: Value, index: Value) -> Value {
        let scale = self.builder.ins().iconst(types::I64, std::mem::size_of::<RegisterWord>() as i64);
        let offset = self.builder.ins().imul(index, scale);
        self.builder.ins().iadd(data, offset)
    }

    /// Emit the StickyImmix probable-write barrier only when a tagged value is a
    /// heap reference. Known scalar trace types elide the barrier completely.
    fn emit_heap_write_barrier(
        &mut self,
        owner_address: Value,
        value_id: TraceValueId,
        tagged: Value,
    ) -> Result<(), TracingJitError> {
        if !self.trace_value_may_be_heap(value_id)? {
            return Ok(());
        }

        let mask = self.builder.ins().iconst(types::I64, TAG_HEAP_MASK as i64);
        let tag = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
        let masked = self.builder.ins().band(tagged, mask);
        let is_heap = self.builder.ins().icmp(IntCC::Equal, masked, tag);

        let barrier = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(is_heap, barrier, &[], done, &[]);

        self.builder.switch_to_block(barrier);
        let callee = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            self.activation,
            offset_of!(TraceActivation, heap_write_barrier) as i32,
        );
        let runtime = self.builder.ins().load(
            self.pointer_type,
            Self::trusted_mem(),
            self.activation,
            offset_of!(TraceActivation, runtime) as i32,
        );
        self.builder.ins().call_indirect(
            self.heap_barrier_signature,
            callee,
            &[runtime, owner_address],
        );
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(done);
        Ok(())
    }

    fn lower_node(
        &mut self,
        node: &crate::wren_trace_ir::TraceNode,
    ) -> Result<(), TracingJitError> {
        use TraceInst::*;

        let one_result = |node: &crate::wren_trace_ir::TraceNode| -> Result<TraceValueId, TracingJitError> {
            if node.results.len() == 1 {
                Ok(node.results[0])
            } else {
                Err(TracingJitError::CraneliftLowering {
                    trace: self.trace.id,
                    message: format!("trace node {} expected exactly one result", node.id),
                })
            }
        };

        match &node.kind {
            Nop => {}

            Runtime {
                location,
                next,
                terminal,
            } => {
                // A full runtime boundary may collect and move objects. Do not let
                // any backend-local moving address flow past this epoch.
                self.invalidate_heap_proofs();
                let callee = self.builder.ins().load(
                    self.pointer_type,
                    Self::trusted_mem(),
                    self.activation,
                    offset_of!(TraceActivation, execute) as i32,
                );
                let function = self
                    .builder
                    .ins()
                    .iconst(types::I32, location.function.0 as i64);
                let pc = self
                    .builder
                    .ins()
                    .iconst(types::I32, location.pc.0 as i64);
                let next = self.builder.ins().iconst(types::I32, next.0 as i64);
                let call = self.builder.ins().call_indirect(
                    self.runtime_signature,
                    callee,
                    &[self.activation, function, pc, next],
                );
                let status = self.builder.inst_results(call)[0];

                let stop = self.builder.create_block();
                let resume = self.builder.create_block();
                self.builder.set_cold_block(stop);
                if *terminal {
                    self.builder.ins().jump(stop, &[]);
                } else {
                    self.builder.ins().brif(status, stop, &[], resume, &[]);
                }

                self.builder.switch_to_block(stop);
                let result = self
                    .builder
                    .ins()
                    .iconst(types::I32, TRACE_RESULT_RUNTIME as i64);
                self.builder.ins().return_(&[result]);

                // Keep a construction block even for terminal runtime steps. It is
                // unreachable in that case, but lets the trace terminator close the
                // CLIF function without appending to an already-terminated block.
                self.builder.switch_to_block(resume);
            }

            RuntimeCall { location, next } => {
                // Calls may allocate/collect even when they synchronously return to
                // the same Wren frame, so moving addresses die at this boundary.
                self.invalidate_heap_proofs();
                if next.0 & TRACE_RUNTIME_CALL_FLAG != 0 {
                    return Err(self.error("RuntimeCall continuation PC uses reserved high bit"));
                }
                let callee = self.builder.ins().load(
                    self.pointer_type,
                    Self::trusted_mem(),
                    self.activation,
                    offset_of!(TraceActivation, execute) as i32,
                );
                let function = self.builder.ins().iconst(types::I32, location.function.0 as i64);
                let pc = self.builder.ins().iconst(types::I32, location.pc.0 as i64);
                let next = self.builder.ins().iconst(
                    types::I32,
                    (next.0 | TRACE_RUNTIME_CALL_FLAG) as i64,
                );
                let call = self.builder.ins().call_indirect(
                    self.runtime_signature,
                    callee,
                    &[self.activation, function, pc, next],
                );
                let status = self.builder.inst_results(call)[0];

                let stop = self.builder.create_block();
                let resume = self.builder.create_block();
                self.builder.set_cold_block(stop);
                self.builder.ins().brif(status, stop, &[], resume, &[]);

                self.builder.switch_to_block(stop);
                let result = self.builder.ins().iconst(types::I32, TRACE_RESULT_RUNTIME as i64);
                self.builder.ins().return_(&[result]);

                self.builder.switch_to_block(resume);
            }

            LoadAbsoluteValue { address } => {
                let address = self.builder.ins().iconst(self.pointer_type, *address as i64);
                let value = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    address,
                    0,
                );
                self.cache_value(one_result(node)?, value);
            }

            StoreAbsoluteValue { address, value } => {
                let value = self.lower_value(*value)?;
                let address = self.builder.ins().iconst(self.pointer_type, *address as i64);
                self.builder
                    .ins()
                    .store(Self::trusted_mem(), value, address, 0);
            }

            IsInstanceObserved {
                value,
                class,
                expected_class,
                instance_classes,
                known_classes,
                exit,
            } => {
                // First guard the dynamic class operand against the class object
                // observed while recording. If it changes, baseline owns the
                // full dynamic semantics.
                let class = self.lower_value(*class)?;
                let (class_entry, _) = self.guard_live_heap_entry(class, *exit)?;
                let class_key = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    class_entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32,
                );
                let expected = self
                    .builder
                    .ins()
                    .iconst(types::I64, *expected_class as i64);
                let same_class = self
                    .builder
                    .ins()
                    .icmp(IntCC::Equal, class_key, expected);
                let same_class = self.bool32(same_class);
                let wrong_class = self.bool_not(same_class);
                self.emit_side_exit(*exit, wrong_class)?;

                let tagged = self.lower_value(*value)?;
                let heap_mask = self
                    .builder
                    .ins()
                    .iconst(types::I64, TAG_HEAP_MASK as i64);
                let heap_tag = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
                let masked = self.builder.ins().band(tagged, heap_mask);
                let is_heap = self.builder.ins().icmp(IntCC::Equal, masked, heap_tag);
                let is_heap = self.bool32(is_heap);

                let heap_block = self.builder.create_block();
                let non_heap_block = self.builder.create_block();
                let done = self.builder.create_block();
                self.builder.append_block_param(done, types::I64);
                self.builder
                    .ins()
                    .brif(is_heap, heap_block, &[], non_heap_block, &[]);

                self.builder.switch_to_block(non_heap_block);
                let false_value = self.builder.ins().iconst(types::I64, TAG_FALSE as i64);
                self.builder.ins().jump(done, &[false_value.into()]);

                self.builder.switch_to_block(heap_block);
                let (entry, _) = self.live_heap_entry_from_heap_tagged(tagged, *exit)?;
                let actual = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    entry,
                    offset_of!(TraceHeapEntry, receiver_class) as i32,
                );
                let mut matches = self.builder.ins().iconst(types::I32, 0);
                for &candidate in instance_classes {
                    let candidate = self
                        .builder
                        .ins()
                        .iconst(types::I64, candidate as i64);
                    let equal = self.builder.ins().icmp(IntCC::Equal, actual, candidate);
                    let equal = self.bool32(equal);
                    matches = self.bool_or(matches, equal);
                }

                // Existing finalized classes have immutable superclass links. If
                // a class is linked after recording, its receiver-class key is not
                // in `known_classes`; side-exit rather than returning a stale false.
                let mut known = self.builder.ins().iconst(types::I32, 0);
                for &candidate in known_classes {
                    let candidate = self.builder.ins().iconst(types::I64, candidate as i64);
                    let equal = self.builder.ins().icmp(IntCC::Equal, actual, candidate);
                    let equal = self.bool32(equal);
                    known = self.bool_or(known, equal);
                }
                let kind = self.builder.ins().ushr_imm_u(actual, 56);
                let instance_kind = self.builder.ins().icmp_imm_u(IntCC::Equal, kind, 1);
                let foreign_kind = self.builder.ins().icmp_imm_u(IntCC::Equal, kind, 4);
                let instance_kind = self.bool32(instance_kind);
                let foreign_kind = self.bool32(foreign_kind);
                let dynamic_instance = self.bool_or(instance_kind, foreign_kind);
                let not_known = self.bool_not(known);
                let unknown = self.bool_and(dynamic_instance, not_known);
                self.emit_side_exit(*exit, unknown)?;

                let boxed = self.box_bool(matches);
                self.builder.ins().jump(done, &[boxed.into()]);

                self.builder.switch_to_block(done);
                let result = self.builder.block_params(done)[0];
                self.cache_value(one_result(node)?, result);
            }

            Copy { value } => {
                let result = one_result(node)?;
                let value = self.lower_value(*value)?;
                self.cache_value(result, value);
            }

            Reinterpret { value } => {
                let result = one_result(node)?;
                let source_rep = self.value_rep(*value)?;
                let target_rep = self.value_rep(result)?;
                let value = self.lower_value(*value)?;
                let word = self.normalize_to_word(value, source_rep);
                let value = match target_rep {
                    TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => {
                        self.builder.ins().ireduce(types::I32, word)
                    }
                    TraceRep::F64 => self
                        .builder
                        .ins()
                        .bitcast(types::F64, Self::plain_mem(), word),
                    _ => word,
                };
                self.cache_value(result, value);
            }

            LoadFrameSlot { slot } => {
                let result = one_result(node)?;
                let rep = self.value_rep(result)?;
                let value = self.load_frame_slot(*slot, rep);
                self.cache_value(result, value);
            }

            CommitFrameSlot { slot, value } => {
                let rep = self.value_rep(*value)?;
                let value = self.lower_value(*value)?;
                self.store_frame_slot(*slot, value, rep);
            }

            GuardTruthy { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let failure = self.tagged_falsey(tagged);
                self.emit_side_exit(*exit, failure)?;
            }

            GuardFalsey { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let falsey = self.tagged_falsey(tagged);
                let failure = self.bool_not(falsey);
                self.emit_side_exit(*exit, failure)?;
            }

            GuardNotNull { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let failure = self.builder.ins().icmp_imm_s(IntCC::Equal, tagged, TAG_NULL as i64);
                let failure = self.bool32(failure);
                self.emit_side_exit(*exit, failure)?;
                if let Some(result) = node.results.first().copied() {
                    self.cache_value(result, tagged);
                }
            }

            GuardF64 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_numeric(tagged, *exit)?;
                self.cache_value(one_result(node)?, raw);
            }

            GuardInt32 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_int32_numeric(tagged, *exit)?;
                self.cache_value(one_result(node)?, raw);
            }

            GuardUInt32 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_uint32_numeric(tagged, *exit)?;
                self.cache_value(one_result(node)?, raw);
            }

            GuardBounds {
                index,
                length,
                exit,
            } => {
                let index_id = self.trace.canonical_value(*index);
                let length_id = self.trace.canonical_value(*length);
                let index = self.lower_value(*index)?;
                let length = self.lower_value(*length)?;
                let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, index, 0);
                let too_large = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, index, length);
                let negative = self.bool32(negative);
                let too_large = self.bool32(too_large);
                let failure = self.bool_or(negative, too_large);
                self.emit_side_exit(*exit, failure)?;
                if let Some(&list) = self.list_lengths.get(&length_id) {
                    let normalized = self.builder.ins().uextend(types::I64, index);
                    self.list_indices.insert((list, index_id), normalized);
                }
            }

            GuardEq { lhs, rhs, exit } => {
                let lhs_value = self.lower_value(*lhs)?;
                let rhs_value = self.lower_value(*rhs)?;
                let rep = self.value_rep(*lhs)?;
                let equal = if rep == TraceRep::F64 {
                    self.f64_compare(CmpKind::Eq, lhs_value, rhs_value)
                } else {
                    let equal = self
                        .builder
                        .ins()
                        .icmp(IntCC::Equal, lhs_value, rhs_value);
                    self.bool32(equal)
                };
                let failure = self.bool_not(equal);
                self.emit_side_exit(*exit, failure)?;
            }

            GuardSwitch {
                tag,
                observed,
                exit,
            } => {
                let tag_value = self.lower_value(*tag)?;
                let rep = self.value_rep(*tag)?;
                if !matches!(rep, TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool) {
                    return Err(self.error("tagged Switch lowering is not enabled"));
                }

                match observed {
                    TraceSwitchGuard::Case(case) => {
                        let equal = self.builder.ins().icmp_imm_s(
                            IntCC::Equal,
                            tag_value,
                            *case,
                        );
                        let equal = self.bool32(equal);
                        let failure = self.bool_not(equal);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    TraceSwitchGuard::Default { excluded } => {
                        let mut matches_any = self.builder.ins().iconst(types::I32, 0);
                        for case in excluded {
                            let equal = self.builder.ins().icmp_imm_s(
                                IntCC::Equal,
                                tag_value,
                                *case,
                            );
                            let equal = self.bool32(equal);
                            matches_any = self.bool_or(matches_any, equal);
                        }
                        self.emit_side_exit(*exit, matches_any)?;
                    }
                }
            }

            GuardType {
                value,
                expected,
                exit,
            } => {
                let value = self.lower_value(*value)?;
                match expected {
                    TraceType::Any => {}
                    TraceType::Num => {
                        let ok = self.tagged_is_num(value);
                        let failure = self.bool_not(ok);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    TraceType::Null => {
                        let ok = self.builder.ins().icmp_imm_s(IntCC::Equal, value, TAG_NULL as i64);
                        let ok = self.bool32(ok);
                        let failure = self.bool_not(ok);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    TraceType::Bool => {
                        let false_ = self.builder.ins().icmp_imm_s(IntCC::Equal, value, TAG_FALSE as i64);
                        let true_ = self.builder.ins().icmp_imm_s(IntCC::Equal, value, TAG_TRUE as i64);
                        let false_ = self.bool32(false_);
                        let true_ = self.bool32(true_);
                        let ok = self.bool_or(false_, true_);
                        let failure = self.bool_not(ok);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    _ => {
                        return Err(self.error(format!(
                            "unsupported native GuardType {expected:?}"
                        )));
                    }
                }

                if let Some(result) = node.results.first().copied() {
                    self.cache_value(result, value);
                }
            }

            GuardClass { value, classes, exit } => {
                let value = self.lower_value(*value)?;
                let (entry, _) = self.guard_live_heap_entry(value, *exit)?;
                let class = self.builder.ins().load(types::I64, Self::trusted_mem(),
                    entry, offset_of!(TraceHeapEntry, receiver_class) as i32);
                let mut matches = self.builder.ins().iconst(types::I32, 0);
                for &expected in classes {
                    let expected = self.builder.ins().iconst(types::I64, expected as i64);
                    let equal = self.builder.ins().icmp(IntCC::Equal, class, expected);
                    let equal = self.bool32(equal);
                    matches = self.bool_or(matches, equal);
                }
                let failure = self.bool_not(matches);
                self.emit_side_exit(*exit, failure)?;
                self.cache_value(one_result(node)?, value);
            }

            GuardMethodEpoch { expected, exit } => {
                let epoch = self.builder.ins().load(types::I64, Self::trusted_mem(), self.activation,
                    offset_of!(TraceActivation, method_epoch) as i32);
                let expected = self.builder.ins().iconst(types::I64, *expected as i64);
                let failure = self.builder.ins().icmp(IntCC::NotEqual, epoch, expected);
                self.emit_side_exit(*exit, failure)?;
            }

            IntUnary { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    IntUnaryOp::Neg => self.builder.ins().ineg(value),
                    IntUnaryOp::BitNot => self.builder.ins().bnot(value),
                };
                self.cache_value(one_result(node)?, result);
            }

            IntBinary { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let mut rhs = self.lower_value(*rhs)?;
                let result = match op {
                    IntBinOp::Add => self.builder.ins().iadd(lhs, rhs),
                    IntBinOp::Sub => self.builder.ins().isub(lhs, rhs),
                    IntBinOp::Mul => self.builder.ins().imul(lhs, rhs),
                    IntBinOp::BitAnd => self.builder.ins().band(lhs, rhs),
                    IntBinOp::BitOr => self.builder.ins().bor(lhs, rhs),
                    IntBinOp::BitXor => self.builder.ins().bxor(lhs, rhs),
                    IntBinOp::Shl => {
                        let mask = self.builder.ins().iconst(types::I32, 31);
                        rhs = self.builder.ins().band(rhs, mask);
                        self.builder.ins().ishl(lhs, rhs)
                    }
                    IntBinOp::ShrU => {
                        let mask = self.builder.ins().iconst(types::I32, 31);
                        rhs = self.builder.ins().band(rhs, mask);
                        self.builder.ins().ushr(lhs, rhs)
                    }
                    IntBinOp::Sar => {
                        let mask = self.builder.ins().iconst(types::I32, 31);
                        rhs = self.builder.ins().band(rhs, mask);
                        self.builder.ins().sshr(lhs, rhs)
                    }
                    IntBinOp::DivS | IntBinOp::RemS => {
                        return Err(self.error(
                            "unchecked integer division/remainder disabled",
                        ));
                    }
                };
                self.cache_value(one_result(node)?, result);
            }

            IntBinaryChecked {
                op,
                lhs,
                rhs,
                exit,
            } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = match op {
                    IntBinOp::Add => {
                        let (result, overflow) = self.builder.ins().sadd_overflow(lhs, rhs);
                        self.emit_side_exit(*exit, overflow)?;
                        result
                    }
                    IntBinOp::Sub => {
                        let (result, overflow) = self.builder.ins().ssub_overflow(lhs, rhs);
                        self.emit_side_exit(*exit, overflow)?;
                        result
                    }
                    IntBinOp::Mul => {
                        let (result, overflow) = self.builder.ins().smul_overflow(lhs, rhs);
                        self.emit_side_exit(*exit, overflow)?;
                        result
                    }
                    IntBinOp::DivS | IntBinOp::RemS => {
                        let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, rhs, 0);
                        let zero = self.bool32(zero);
                        self.emit_side_exit(*exit, zero)?;

                        let is_min = self.builder.ins().icmp_imm_s(
                            IntCC::Equal,
                            lhs,
                            i32::MIN as i64,
                        );
                        let is_minus_one = self.builder.ins().icmp_imm_s(IntCC::Equal, rhs, -1);
                        let is_min = self.bool32(is_min);
                        let is_minus_one = self.bool32(is_minus_one);
                        let overflow = self.bool_and(is_min, is_minus_one);
                        self.emit_side_exit(*exit, overflow)?;

                        if matches!(op, IntBinOp::DivS) {
                            self.builder.ins().sdiv(lhs, rhs)
                        } else {
                            self.builder.ins().srem(lhs, rhs)
                        }
                    }
                    _ => {
                        return Err(self.error(format!(
                            "checked {op:?} is not meaningful for this integer operation"
                        )));
                    }
                };
                self.cache_value(one_result(node)?, result);
            }

            F64Unary { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    F64UnaryOp::Neg => self.builder.ins().fneg(value),
                };
                self.cache_value(one_result(node)?, result);
            }

            F64Math { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    TraceMathOp::Abs => self.builder.ins().fabs(value),
                    TraceMathOp::Floor => self.builder.ins().floor(value),
                    TraceMathOp::Ceil => self.builder.ins().ceil(value),
                    TraceMathOp::Sqrt => self.builder.ins().sqrt(value),
                    _ => {
                        let helper: extern "C" fn(f64) -> f64 = match op {
                            TraceMathOp::Round => native_round,
                            TraceMathOp::Sin => native_sin,
                            TraceMathOp::Cos => native_cos,
                            TraceMathOp::Tan => native_tan,
                            TraceMathOp::Log => native_log,
                            TraceMathOp::Exp => native_exp,
                            _ => unreachable!(),
                        };
                        self.call_scalar_helper(helper as usize, &[value])
                    }
                };
                self.cache_value(one_result(node)?, result);
            }

            F64Binary { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = match op {
                    F64BinOp::Add => self.builder.ins().fadd(lhs, rhs),
                    F64BinOp::Sub => self.builder.ins().fsub(lhs, rhs),
                    F64BinOp::Mul => self.builder.ins().fmul(lhs, rhs),
                    F64BinOp::Div => self.builder.ins().fdiv(lhs, rhs),
                    F64BinOp::Rem => {
                        self.call_scalar_helper(native_frem as *const () as usize, &[lhs, rhs])
                    }
                };
                self.cache_value(one_result(node)?, result);
            }

            IntCompare { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.int_compare(*op, lhs, rhs);
                self.cache_value(one_result(node)?, result);
            }

            F64Compare { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.f64_compare(*op, lhs, rhs);
                self.cache_value(one_result(node)?, result);
            }

            RefEq { lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.builder.ins().icmp(IntCC::Equal, lhs, rhs);
                let result = self.bool32(result);
                self.cache_value(one_result(node)?, result);
            }

            IsNull { value } => {
                let value = self.lower_value(*value)?;
                let result = self.builder.ins().icmp_imm_s(IntCC::Equal, value, TAG_NULL as i64);
                let result = self.bool32(result);
                self.cache_value(one_result(node)?, result);
            }

            Select {
                condition,
                if_true,
                if_false,
            } => {
                let condition = self.lower_value(*condition)?;
                let if_true = self.lower_value(*if_true)?;
                let if_false = self.lower_value(*if_false)?;
                let result = self.builder.ins().select(condition, if_true, if_false);
                self.cache_value(one_result(node)?, result);
            }

            BoxInt32 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.builder.ins().fcvt_from_sint(types::F64, value);
                let tagged = self
                    .builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number);
                self.cache_value(one_result(node)?, tagged);
            }

            BoxUInt32 { value } => {
                let value = self.lower_value(*value)?;
                let wide = self.builder.ins().uextend(types::I64, value);
                let number = self.builder.ins().fcvt_from_uint(types::F64, wide);
                let tagged = self
                    .builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number);
                self.cache_value(one_result(node)?, tagged);
            }

            BoxF64 { value } => {
                let value = self.lower_value(*value)?;
                let tagged = self.box_f64(value);
                self.cache_value(one_result(node)?, tagged);
            }

            Int32ToF64 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.builder.ins().fcvt_from_sint(types::F64, value);
                self.cache_value(one_result(node)?, number);
            }

            UInt32ToF64 { value } => {
                let value = self.lower_value(*value)?;
                let wide = self.builder.ins().uextend(types::I64, value);
                let number = self.builder.ins().fcvt_from_uint(types::F64, wide);
                self.cache_value(one_result(node)?, number);
            }

            F64ToInt32 { value, exit } => {
                let number = self.lower_value(*value)?;

                let bits = self
                    .builder
                    .ins()
                    .bitcast(types::I64, Self::plain_mem(), number);
                let negative_zero = self
                    .builder
                    .ins()
                    .iconst(types::I64, (-0.0f64).to_bits() as i64);
                let is_negative_zero = self
                    .builder
                    .ins()
                    .icmp(IntCC::Equal, bits, negative_zero);
                let is_negative_zero = self.bool32(is_negative_zero);
                self.emit_side_exit(*exit, is_negative_zero)?;

                let min = self
                    .builder
                    .ins()
                    .f64const(Ieee64::with_bits((i32::MIN as f64).to_bits()));
                let max = self
                    .builder
                    .ins()
                    .f64const(Ieee64::with_bits((i32::MAX as f64).to_bits()));
                let ge = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::GreaterThanOrEqual, number, min);
                let le = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, number, max);
                let ge = self.bool32(ge);
                let le = self.bool32(le);
                let in_range = self.bool_and(ge, le);
                let fail = self.bool_not(in_range);
                self.emit_side_exit(*exit, fail)?;

                let wide = self.builder.ins().fcvt_to_sint(types::I64, number);
                let back = self
                    .builder
                    .ins()
                    .fcvt_from_sint(types::F64, wide);
                let exact = self.builder.ins().fcmp(FloatCC::Equal, number, back);
                let exact = self.bool32(exact);
                let not_exact = self.bool_not(exact);
                self.emit_side_exit(*exit, not_exact)?;

                let result = self.builder.ins().ireduce(types::I32, wide);
                self.cache_value(one_result(node)?, result);
            }

            TaggedToHeapHandle { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let mask = self
                    .builder
                    .ins()
                    .iconst(types::I64, TAG_HEAP_MASK as i64);
                let expected = self.builder.ins().iconst(types::I64, TAG_HEAP as i64);
                let masked = self.builder.ins().band(tagged, mask);
                let is_heap = self.builder.ins().icmp(IntCC::Equal, masked, expected);
                let is_heap = self.bool32(is_heap);
                let fail = self.bool_not(is_heap);
                self.emit_side_exit(*exit, fail)?;

                let payload = self
                    .builder
                    .ins()
                    .iconst(types::I64, PAYLOAD_MASK as i64);
                let handle = self.builder.ins().band(tagged, payload);
                self.cache_value(one_result(node)?, handle);
            }

            GuardHeapObject { value, kind, exit } => {
                let (entry, address) = self.heap_entry_for_value(
                    *value,
                    Self::runtime_heap_kind(*kind),
                    *exit,
                )?;
                let result = one_result(node)?;
                self.cache_value(result, entry);
                self.heap_addresses[result.0 as usize] = Some(address);
            }

            HeapLoadField { object, slot, exit } => {
                let (entry, _) = self.heap_entry_for_value(*object, TraceHeapKind::Instance, *exit)?;
                let len = self.heap_entry_len(entry);
                self.guard_constant_index(len, *slot, *exit)?;
                let data = self.heap_entry_data(entry);
                let index = self.builder.ins().iconst(types::I64, *slot as i64);
                let address = self.heap_element_address(data, index);
                let result = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    address,
                    0,
                );
                self.cache_value(one_result(node)?, result);
            }

            HeapStoreField { object, value, slot, exit } => {
                let tagged_value = self.lower_value(*value)?;
                let (entry, owner_address) =
                    self.heap_entry_for_value(*object, TraceHeapKind::Instance, *exit)?;
                let len = self.heap_entry_len(entry);
                self.guard_constant_index(len, *slot, *exit)?;
                let data = self.heap_entry_data(entry);
                let index = self.builder.ins().iconst(types::I64, *slot as i64);
                let address = self.heap_element_address(data, index);
                self.emit_heap_write_barrier(owner_address, *value, tagged_value)?;
                self.builder.ins().store(Self::trusted_mem(), tagged_value, address, 0);
            }

            HeapLoadCell { cell, exit } => {
                let (entry, _) = self.heap_entry_for_value(*cell, TraceHeapKind::Cell, *exit)?;
                let data = self.heap_entry_data(entry);
                let result = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    data,
                    0,
                );
                self.cache_value(one_result(node)?, result);
            }

            HeapStoreCell { cell, value, exit } => {
                let tagged_value = self.lower_value(*value)?;
                let (entry, owner_address) =
                    self.heap_entry_for_value(*cell, TraceHeapKind::Cell, *exit)?;
                let data = self.heap_entry_data(entry);
                self.emit_heap_write_barrier(owner_address, *value, tagged_value)?;
                self.builder.ins().store(Self::trusted_mem(), tagged_value, data, 0);
            }

            HeapListLen { list, exit } => {
                let (entry, _) = self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                let len = self.heap_entry_len(entry);
                let len = self.builder.ins().ireduce(types::I32, len);
                self.cache_value(one_result(node)?, len);
                self.list_lengths.insert(one_result(node)?, self.trace.canonical_value(*list));
            }

            HeapListGet { list, index, exit } => {
                let (entry, _) = self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                let index = self.validated_list_index(*list, *index, entry, *exit)?;
                let data = self.heap_entry_data(entry);
                let address = self.heap_element_address(data, index);
                let result = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    address,
                    0,
                );
                self.cache_value(one_result(node)?, result);
            }

            HeapListSet { list, index, value, exit } => {
                let tagged_value = self.lower_value(*value)?;
                let (entry, owner_address) =
                    self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                let index = self.validated_list_index(*list, *index, entry, *exit)?;
                let data = self.heap_entry_data(entry);
                let address = self.heap_element_address(data, index);
                self.emit_heap_write_barrier(owner_address, *value, tagged_value)?;
                self.builder.ins().store(Self::trusted_mem(), tagged_value, address, 0);
            }

            HeapMapLen { map, exit } => {
                let (entry, _) = self.heap_entry_for_value(*map, TraceHeapKind::Map, *exit)?;
                let len = self.heap_entry_len(entry);
                let len = self.builder.ins().ireduce(types::I32, len);
                self.cache_value(one_result(node)?, len);
            }

            HeapMapGetShaped { map, key, expected_key, shape, slot, exit } => {
                let tagged_key = self.lower_value(*key)?;
                let (entry, _, metadata) =
                    self.heap_entry_with_metadata_for_value(*map, TraceHeapKind::Map, *exit)?;
                self.guard_map_shape_metadata(metadata, *shape, *exit)?;
                self.guard_string_bytes(tagged_key, expected_key, *exit)?;
                let len = self.heap_entry_len(entry);
                self.guard_constant_index(len, *slot, *exit)?;
                let data = self.heap_entry_data(entry);
                let index = self.builder.ins().iconst(types::I64, *slot as i64);
                let address = self.heap_element_address(data, index);
                let result = self.builder.ins().load(
                    types::I64,
                    Self::trusted_mem(),
                    address,
                    0,
                );
                self.cache_value(one_result(node)?, result);
            }

            HeapMapSetShaped { map, key, value, expected_key, shape, slot, exit } => {
                let tagged_key = self.lower_value(*key)?;
                let tagged_value = self.lower_value(*value)?;
                let (entry, owner_address, metadata) =
                    self.heap_entry_with_metadata_for_value(*map, TraceHeapKind::Map, *exit)?;
                self.guard_map_shape_metadata(metadata, *shape, *exit)?;
                self.guard_string_bytes(tagged_key, expected_key, *exit)?;
                let len = self.heap_entry_len(entry);
                self.guard_constant_index(len, *slot, *exit)?;
                let data = self.heap_entry_data(entry);
                let index = self.builder.ins().iconst(types::I64, *slot as i64);
                let address = self.heap_element_address(data, index);
                self.emit_heap_write_barrier(owner_address, *value, tagged_value)?;
                self.builder.ins().store(Self::trusted_mem(), tagged_value, address, 0);
            }

            HeapStringLen { string, exit } => {
                let (entry, _) = self.heap_entry_for_value(*string, TraceHeapKind::String, *exit)?;
                let len = self.heap_entry_len(entry);
                let len = self.builder.ins().ireduce(types::I32, len);
                self.cache_value(one_result(node)?, len);
            }

            HeapAwaitFulfilled { promise, exit } => {
                let (entry, _, metadata) = self.heap_entry_with_metadata_for_value(
                    *promise,
                    TraceHeapKind::Promise,
                    *exit,
                )?;
                let state = self.heap_metadata_payload(metadata);
                let fulfilled = self.builder.ins().icmp_imm_u(
                    IntCC::Equal,
                    state,
                    TRACE_PROMISE_FULFILLED as i64,
                );
                let fulfilled = self.bool32(fulfilled);
                let not_fulfilled = self.bool_not(fulfilled);
                self.emit_side_exit(*exit, not_fulfilled)?;
                let value = self.heap_entry_promise_value(entry);
                self.cache_value(one_result(node)?, value);
            }

            ReadBarrier { value } => {
                self.invalidate_heap_proofs();
                let value = self.lower_value(*value)?;
                self.cache_value(one_result(node)?, value);
            }
        }

        Ok(())
    }
}

fn clif_type(rep: TraceRep) -> ClifType {
    match rep {
        TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => types::I64,
        TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => types::I32,
        TraceRep::F64 => types::F64,
    }
}

// =============================================================================
// Hotness / compilation manager
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LoopKey {
    location: TraceLocation,
}

#[derive(Clone, Debug, Default)]
struct LoopState {
    hotness: u32,
    retry_after: u32,
    failures: u16,
    trace: Option<TraceId>,
    blacklisted: bool,
}

#[derive(Clone, Debug)]
pub enum JitAction {
    Interpreted,
    RecordingAborted {
        reason: String,
    },
    Compiled {
        trace: TraceId,
    },
    EnteredNative {
        trace: TraceId,
    },
    SideExit {
        trace: TraceId,
        exit: SideExitId,
        target: TraceLocation,
    },
    Blacklisted,
    CoolingDown {
        trace: TraceId,
    },
}

#[derive(Clone, Debug)]
enum SideExitDispatch {
    Action(JitAction),
    Link(TraceId),
}

pub struct TracingJit {
    options: TracingJitOptions,
    backend: Option<CraneliftBackend>,
    loops: HashMap<LoopKey, LoopState>,
    entries: HashMap<TraceLocation, LoopState>,
    traces: Vec<Option<CompiledTrace>>,
    /// Dense tables consumed by the generated native dispatcher. They may move
    /// while compiling new traces, but no compilation occurs during one native
    /// dispatcher invocation, so pointers installed in TraceActivation stay valid.
    entry_table: Vec<usize>,
    link_table_ptrs: Vec<usize>,
    next_trace_id: u32,
    stats: TracingJitStats,
}

impl Default for TracingJit {
    fn default() -> Self {
        Self::new(TracingJitOptions::default())
    }
}

#[inline]
fn jit_events_enabled() -> bool {
    std::env::var_os("WREN_JIT_EVENTS").is_some()
}

impl TracingJit {
    #[cfg(test)]
    pub(crate) fn execute_one_for_test<R: TraceRuntime>(&mut self, runtime: &mut R) -> Result<(), TracingJitError> {
        let location = runtime.trace_current_location();
        let program = runtime.trace_program();
        let function = program.function(location.function).unwrap();
        let op = decode_at(program, location)?;
        let recorder = TraceRecorder::new(&self.options, TraceId(0), location);
        let mut builder = crate::wren_trace_ir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Loop, location });
        let expected = recorder.translate_op(runtime, program, function, &op, &mut builder)
            .map_err(|reason| TracingJitError::Record { location, message: reason.to_string() })?;
        match expected {
            ExpectedStep::Exact(next) | ExpectedStep::Inlined { next, .. } => {
                let exit = builder.add_side_exit(ExitReason::GuardValue, next.pc);
                builder.set_terminator(TraceTerminator::Exit(exit));
            },
            ExpectedStep::Runtime { .. } | ExpectedStep::RuntimeCall => {
                builder.prepare_memory_loop();
                builder.set_terminator(TraceTerminator::RuntimeReturn);
            },
        }
        let trace = builder.finish().map_err(TracingJitError::TraceVerification)?;
        let compiled = self.compile_trace(&trace)?;
        self.insert_compiled(TraceId(0), compiled);
        self.enter_compiled(runtime, TraceId(0))?;
        Ok(())
    }

    pub fn new(options: TracingJitOptions) -> Self {
        Self {
            options,
            backend: None,
            loops: HashMap::new(),
            entries: HashMap::new(),
            traces: Vec::new(),
            entry_table: Vec::new(),
            link_table_ptrs: Vec::new(),
            next_trace_id: 0,
            stats: TracingJitStats::default(),
        }
    }

    pub fn options(&self) -> &TracingJitOptions {
        &self.options
    }

    pub fn stats(&self) -> &TracingJitStats {
        &self.stats
    }

    #[inline]
    fn compiled(&self, id: TraceId) -> Option<&CompiledTrace> {
        self.traces.get(id.0 as usize).and_then(Option::as_ref)
    }

    #[inline]
    fn compiled_mut(&mut self, id: TraceId) -> Option<&mut CompiledTrace> {
        self.traces.get_mut(id.0 as usize).and_then(Option::as_mut)
    }

    fn insert_compiled(&mut self, id: TraceId, mut compiled: CompiledTrace) {
        let index = id.0 as usize;

        // Side traces are born with their closing edge already targeting the root.
        // Prime such links immediately; ordinary root exits start unlinked and are
        // updated later by stitch().
        for exit_index in 0..compiled.exits.len() {
            let native_target = {
                let plan = &compiled.exits[exit_index];
                if plan.native_linkable {
                    match plan.target {
                        ExitTarget::Trace(child) => Some(child),
                        _ => None,
                    }
                } else {
                    None
                }
            };
            if let Some(child) = native_target {
                if let Some(slot) = compiled.native_links.get_mut(exit_index) {
                    *slot = child.0;
                }
            }
        }

        if self.traces.len() <= index {
            self.traces.resize_with(index + 1, || None);
        }
        if self.entry_table.len() <= index {
            self.entry_table.resize(index + 1, 0);
            self.link_table_ptrs.resize(index + 1, 0);
        }
        self.entry_table[index] = compiled.entry as usize;
        self.link_table_ptrs[index] = compiled.native_links.as_ptr() as usize;
        self.traces[index] = Some(compiled);
    }

    fn compile_trace(&mut self, trace: &Trace) -> Result<CompiledTrace, TracingJitError> {
        let options = self.options.clone();
        if self.backend.is_none() {
            let backend = CraneliftBackend::new(&options).map_err(|message| {
                TracingJitError::CraneliftLowering {
                    trace: trace.id,
                    message,
                }
            })?;
            self.backend = Some(backend);
        }

        let compiler = CraneliftTraceCompiler::new(trace, &options)?;
        compiler.compile(self.backend.as_mut().unwrap())
    }

    pub fn trace(&self, id: TraceId) -> Option<&Trace> {
        self.compiled(id).map(|compiled| &compiled.trace)
    }

    pub fn disassembly(&self, id: TraceId) -> Option<String> {
        self.compiled(id)
            .and_then(|compiled| compiled.debug_text.clone())
    }

    /// Prime manager-side hotness after the interpreter's dense per-PC counter
    /// has already paid the cold-tier threshold. This keeps HashMap traffic out
    /// of cold calls without changing retry/blacklist semantics here.
    pub(crate) fn preheat_entry(&mut self, location: TraceLocation) {
        let threshold = self.options.hot_entry_threshold.max(1);
        let state = self.entries.entry(location).or_default();
        state.hotness = state.hotness.max(threshold.max(state.retry_after));
    }

    /// Record a bounded acyclic path. Recording executes the prefix once, so
    /// newly compiled entries are first used on a subsequent visit.
    pub fn on_entry<R: TraceRuntime>(&mut self, runtime: &mut R)
        -> Result<JitAction, TracingJitError>
    {
        if self.options.hot_entry_threshold == 0 { return Ok(JitAction::Interpreted); }
        let location = runtime.trace_current_location();
        self.stats.entry_hits += 1;
        let state = self.entries.entry(location).or_default();
        if state.blacklisted { return Ok(JitAction::Blacklisted); }
        if let Some(trace) = state.trace { return self.enter_compiled(runtime, trace); }
        state.hotness = state.hotness.saturating_add(1);
        if state.hotness < self.options.hot_entry_threshold.max(state.retry_after) {
            return Ok(JitAction::Interpreted);
        }
        let id = TraceId(self.next_trace_id);
        self.next_trace_id = self.next_trace_id.wrapping_add(1);
        self.stats.recordings_started += 1;
        let recorder = TraceRecorder::new(&self.options, id, location);
        let result = match recorder.record_impl(runtime, RecordingMode::Entry) {
            Ok(trace) => {
                self.stats.recordings_succeeded += 1;
                self.stats.recorded_bytecodes += trace.stats.recorded_bytecodes as u64;
                self.stats.inlined_calls += trace.stats.inlined_calls as u64;
                self.stats.devirtualized_calls += trace.stats.devirtualized_calls as u64;
                self.stats.pruned_snapshot_writes += trace.stats.pruned_snapshot_writes as u64;
                self.stats.eliminated_trace_guards += trace.stats.eliminated_guards as u64;
                self.stats.eliminated_trace_boxes += trace.stats.eliminated_boxes as u64;
                self.stats.eliminated_trace_copies += trace.stats.eliminated_copies as u64;
                self.stats.trace_gvn_eliminated += trace.stats.gvn_eliminated as u64;
                self.stats.trace_licm_hoisted += trace.stats.licm_hoisted as u64;
                self.stats.trace_pre_eliminated += trace.stats.pre_eliminated as u64;
                if trace.stats.recorded_bytecodes < self.options.min_entry_trace_bytecodes as usize {
                    let state = self.entries.get_mut(&location).unwrap();
                    state.blacklisted = true;
                    self.stats.blacklisted_entries = self.stats.blacklisted_entries.saturating_add(1);
                    self.stats.entry_traces_too_small = self.stats.entry_traces_too_small.saturating_add(1);
                    self.stats.last_abort = Some(format!(
                        "entry region too small: {} bytecodes < {}",
                        trace.stats.recorded_bytecodes,
                        self.options.min_entry_trace_bytecodes,
                    ));
                    return Ok(JitAction::Blacklisted);
                }
                self.compile_trace(&trace).map_err(|error| error.to_string())
            }
            Err(RecordOutcome::Abort(reason)) => Err(reason.to_string()),
            Err(RecordOutcome::Fatal(error)) => return Err(error),
        };
        match result {
            Ok(compiled) => {
                self.insert_compiled(id, compiled);
                self.entries.get_mut(&location).unwrap().trace = Some(id);
                self.stats.traces_compiled += 1;
                self.stats.entry_traces_compiled += 1;
                if jit_events_enabled() {
                    eprintln!("[wren-jit] compiled entry {:?} at {:?}:{}", id, location.function, location.pc.0);
                }
                Ok(JitAction::Compiled { trace: id })
            }
            Err(reason) => {
                self.stats.recordings_aborted += 1;
                self.stats.last_abort = Some(reason.clone());
                let state = self.entries.get_mut(&location).unwrap();
                state.failures = state.failures.saturating_add(1);
                state.hotness = 0;
                state.retry_after = self.options.hot_entry_threshold.max(state.retry_after)
                    .saturating_mul(2);
                if state.failures >= self.options.max_failures_before_blacklist {
                    state.blacklisted = true;
                    self.stats.blacklisted_entries += 1;
                }
                if jit_events_enabled() {
                    eprintln!("[wren-jit] entry abort {:?}:{}: {}", location.function, location.pc.0, reason);
                }
                Ok(JitAction::RecordingAborted { reason })
            }
        }
    }

    pub(crate) fn entry_trace(&self, location: TraceLocation) -> Option<TraceId> {
        self.entries.get(&location).and_then(|state| state.trace)
    }

    pub(crate) fn entry_blacklisted(&self, location: TraceLocation) -> bool {
        self.entries.get(&location).is_some_and(|state| state.blacklisted)
    }

    /// Called when baseline execution arrives at a known loop header.
    ///
    /// The recorder executes one real iteration when the threshold is crossed.
    /// Therefore, after successful recording the interpreter is again at the loop
    /// header and the just-compiled native trace can immediately take over.
    pub fn on_loop_header<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
    ) -> Result<JitAction, TracingJitError> {
        let location = runtime.trace_current_location();
        let key = LoopKey { location };

        self.stats.loop_header_hits += 1;

        let state = self.loops.entry(key).or_default();
        if state.blacklisted {
            return Ok(JitAction::Blacklisted);
        }

        if let Some(trace) = state.trace {
            return self.enter_compiled(runtime, trace);
        }

        state.hotness = state.hotness.saturating_add(1);
        if state.hotness < self.options.hot_loop_threshold.max(state.retry_after) {
            return Ok(JitAction::Interpreted);
        }

        let trace_id = TraceId(self.next_trace_id);
        self.next_trace_id = self.next_trace_id.wrapping_add(1);
        self.stats.recordings_started += 1;

        let recorder = TraceRecorder::new(&self.options, trace_id, location);
        let trace = match recorder.record(runtime) {
            Ok(trace) => trace,
            Err(RecordOutcome::Abort(reason)) => {
                self.stats.last_abort = Some(reason.to_string());
                self.stats.recordings_aborted += 1;
                if jit_events_enabled() {
                    eprintln!(
                        "[wren-jit] record abort {:?}:{}: {}",
                        location.function, location.pc.0, reason
                    );
                }
                let state = self.loops.get_mut(&key).unwrap();
                if matches!(reason, RecordingAbort::EnclosingLoop) {
                    // This path may be short only for the current invocation
                    // (e.g. Mandelbrot's early pixels). Retry less often, but do
                    // not permanently disable a loop that can become hot later.
                    state.retry_after = self.options.hot_loop_threshold.max(state.retry_after)
                        .max(1).saturating_mul(2).min(4096);
                    state.hotness = 0;
                    return Ok(JitAction::RecordingAborted { reason: reason.to_string() });
                }
                state.failures = state.failures.saturating_add(1);
                // Back off instead of trying to record every iteration.
                state.hotness = self.options.hot_loop_threshold / 2;
                if state.failures >= self.options.max_failures_before_blacklist {
                    state.blacklisted = true;
                    self.stats.blacklisted_loops += 1;
                }
                return Ok(JitAction::RecordingAborted {
                    reason: reason.to_string(),
                });
            }
            Err(RecordOutcome::Fatal(error)) => return Err(error),
        };

        self.stats.recordings_succeeded += 1;
        self.stats.recorded_bytecodes += trace.stats.recorded_bytecodes as u64;
        self.stats.inlined_calls += trace.stats.inlined_calls as u64;
        self.stats.devirtualized_calls += trace.stats.devirtualized_calls as u64;
        self.stats.promoted_f64_loop_phis += trace.stats.promoted_f64_loop_phis as u64;
        self.stats.promoted_i32_loop_phis += trace.stats.promoted_i32_loop_phis as u64;
        self.stats.promoted_u32_loop_phis += trace.stats.promoted_u32_loop_phis as u64;
        self.stats.pruned_snapshot_writes += trace.stats.pruned_snapshot_writes as u64;
        self.stats.eliminated_trace_guards += trace.stats.eliminated_guards as u64;
        self.stats.eliminated_trace_boxes += trace.stats.eliminated_boxes as u64;
        self.stats.eliminated_trace_copies += trace.stats.eliminated_copies as u64;
        self.stats.trace_gvn_eliminated += trace.stats.gvn_eliminated as u64;
        self.stats.trace_licm_hoisted += trace.stats.licm_hoisted as u64;
        self.stats.trace_pre_eliminated += trace.stats.pre_eliminated as u64;

        let compiled = match self.compile_trace(&trace) {
            Ok(compiled) => compiled,
            Err(error) => {
                self.stats.last_abort = Some(error.to_string());
                if jit_events_enabled() {
                    eprintln!(
                        "[wren-jit] compile abort {:?}:{}: {}",
                        location.function, location.pc.0, error
                    );
                }
                // A trace-tier lowering failure is not a language/runtime error.
                // Baseline execution remains authoritative, so back off and let
                // the interpreter continue.
                let state = self.loops.get_mut(&key).unwrap();
                state.failures = state.failures.saturating_add(1);
                state.hotness = self.options.hot_loop_threshold / 2;
                if state.failures >= self.options.max_failures_before_blacklist {
                    state.blacklisted = true;
                    self.stats.blacklisted_loops += 1;
                }
                return Ok(JitAction::RecordingAborted {
                    reason: error.to_string(),
                });
            }
        };

        self.stats.traces_compiled += 1;
        self.loops.get_mut(&key).unwrap().trace = Some(trace_id);
        self.insert_compiled(trace_id, compiled);
        if jit_events_enabled() {
            eprintln!(
                "[wren-jit] compiled {:?} at {:?}:{} (bytecodes={}, nodes={}, exits={}, phis={})",
                trace_id,
                location.function,
                location.pc.0,
                trace.stats.recorded_bytecodes,
                trace.nodes.len(),
                trace.exits.len(),
                trace.loop_phis.len(),
            );
        }

        // Recording one iteration ended at the same root header.
        if runtime.trace_current_location() == location {
            self.enter_compiled(runtime, trace_id)
        } else {
            Ok(JitAction::Compiled { trace: trace_id })
        }
    }

    /// Enter an already compiled trace by dense TraceId. The interpreter caches
    /// this ID beside the prepared loop header, bypassing the loop-state HashMap
    /// after tiering has succeeded.
    #[inline]
    pub(crate) fn enter_trace<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        trace: TraceId,
    ) -> Result<JitAction, TracingJitError> {
        self.enter_compiled(runtime, trace)
    }

    /// Convenience hook for a baseline backward jump.
    ///
    /// The interpreter can call this after it has set its PC to `to`. Forward
    /// jumps are ignored. This avoids adding counters to every bytecode dispatch.
    pub fn on_backedge<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        from: TraceLocation,
        to: TraceLocation,
    ) -> Result<JitAction, TracingJitError> {
        if from.function != to.function || to.pc.0 >= from.pc.0 {
            return Ok(JitAction::Interpreted);
        }
        if runtime.trace_current_location() != to {
            return Err(TracingJitError::Activation(
                "on_backedge must be called after setting the interpreter PC to target".to_owned(),
            ));
        }
        self.on_loop_header(runtime)
    }

    fn enter_compiled<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        initial_trace: TraceId,
    ) -> Result<JitAction, TracingJitError> {
        let mut trace_id = initial_trace;

        // A generated Cranelift dispatcher follows already-stitched trace-tree
        // edges without re-entering this Rust manager. Rust is involved only when
        // an exit is still unlinked, a runtime helper terminates the trace, or the
        // source program itself returns/suspends.
        loop {
            let inputs = runtime
                .trace_activation_inputs()
                .map_err(runtime_error)?;

            if inputs.general.is_null()
                || inputs.arguments.is_null()
                || inputs.returns.is_null()
                || inputs.special.is_null()
            {
                return Err(TracingJitError::Activation(
                    "runtime returned null trace register-bank pointer".to_owned(),
                ));
            }
            if self.compiled(trace_id).is_none() {
                return Err(TracingJitError::CraneliftLowering {
                    trace: trace_id,
                    message: "compiled trace disappeared".to_owned(),
                });
            }

            let mut activation = TraceActivation::new(inputs);
            activation.method_epoch = runtime.trace_method_epoch();
            activation.trace_entries = self.entry_table.as_ptr();
            activation.trace_link_tables = self.link_table_ptrs.as_ptr();
            activation.trace_entries_len = self.entry_table.len() as u32;
            activation.next_trace = trace_id.0;
            activation.dispatch_links_remaining = if self.options.native_loop_iteration_budget == 0 {
                u32::MAX
            } else {
                self.options.native_loop_iteration_budget
            };

            let dispatcher = self
                .backend
                .as_ref()
                .ok_or_else(|| TracingJitError::CraneliftLowering {
                    trace: trace_id,
                    message: "native backend disappeared".to_owned(),
                })?
                .dispatcher;

            self.stats.native_entries = self.stats.native_entries.saturating_add(1);
            if matches!(self.compiled(trace_id).unwrap().trace.anchor.kind, TraceKind::Entry) {
                self.stats.entry_native_entries += 1;
            }
            let result = unsafe { dispatcher(&mut activation) };
            let producer = TraceId(activation.next_trace);

            match result {
                TRACE_RESULT_SIDE_EXIT => {
                    self.stats.side_exits = self.stats.side_exits.saturating_add(1);
                    match self.apply_side_exit(runtime, producer, &activation)? {
                        SideExitDispatch::Action(action) => return Ok(action),
                        SideExitDispatch::Link(child) => {
                            // This path is normally taken only once while a hot edge
                            // is being stitched. Subsequent executions are consumed by
                            // the native dispatcher directly. Rebuild activation here
                            // because side-trace compilation may have reallocated the
                            // dense entry/link pointer tables.
                            trace_id = child;
                            continue;
                        }
                    }
                }

                TRACE_RESULT_RETURNED | TRACE_RESULT_RUNTIME => {
                    return Ok(JitAction::EnteredNative { trace: producer });
                }

                TRACE_RESULT_LINK => {
                    // Dispatcher fuel expired after the linked exit had already
                    // materialized the child entry state. Resume baseline at that
                    // exact trace anchor to preserve cooperative runtime/GC access.
                    let next = TraceId(activation.next_trace);
                    let location = self
                        .compiled(next)
                        .ok_or_else(|| TracingJitError::CraneliftLowering {
                            trace: next,
                            message: "dispatcher yielded to an unknown child trace".to_owned(),
                        })?
                        .trace
                        .anchor
                        .location;
                    runtime.trace_set_location(location).map_err(runtime_error)?;
                    self.stats.native_budget_exits = self.stats.native_budget_exits.saturating_add(1);
                    return Ok(JitAction::EnteredNative { trace: next });
                }

                other => {
                    return Err(TracingJitError::Activation(format!(
                        "native trace {:?} returned unknown result code {}",
                        producer, other
                    )));
                }
            }
        }
    }

    fn try_record_side_trace<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        parent: TraceId,
        parent_exit: SideExitId,
        start: TraceLocation,
        close_at: TraceLocation,
        link_to: TraceId,
    ) -> Result<Option<TraceId>, TracingJitError> {
        let trace_id = TraceId(self.next_trace_id);
        self.next_trace_id = self.next_trace_id.wrapping_add(1);
        self.stats.recordings_started = self.stats.recordings_started.saturating_add(1);

        let recorder = TraceRecorder::new(&self.options, trace_id, start);
        let trace = match recorder.record_side(runtime, close_at, link_to) {
            Ok(trace) => trace,
            Err(RecordOutcome::Abort(reason)) => {
                self.stats.recordings_aborted = self.stats.recordings_aborted.saturating_add(1);
                self.stats.side_recordings_aborted = self.stats.side_recordings_aborted.saturating_add(1);
                self.stats.last_abort = Some(reason.to_string());
                if jit_events_enabled() {
                    eprintln!(
                        "[wren-jit] side record abort {:?}:{} -> {:?}:{}: {}",
                        start.function, start.pc.0, close_at.function, close_at.pc.0, reason
                    );
                }
                return Ok(None);
            }
            Err(RecordOutcome::Fatal(error)) => return Err(error),
        };

        self.stats.recordings_succeeded = self.stats.recordings_succeeded.saturating_add(1);
        self.stats.inlined_calls += trace.stats.inlined_calls as u64;
        self.stats.devirtualized_calls += trace.stats.devirtualized_calls as u64;
        self.stats.recorded_bytecodes = self
            .stats
            .recorded_bytecodes
            .saturating_add(trace.stats.recorded_bytecodes as u64);
        self.stats.promoted_f64_loop_phis = self
            .stats
            .promoted_f64_loop_phis
            .saturating_add(trace.stats.promoted_f64_loop_phis as u64);
        self.stats.promoted_i32_loop_phis = self
            .stats
            .promoted_i32_loop_phis
            .saturating_add(trace.stats.promoted_i32_loop_phis as u64);
        self.stats.promoted_u32_loop_phis = self
            .stats
            .promoted_u32_loop_phis
            .saturating_add(trace.stats.promoted_u32_loop_phis as u64);
        self.stats.pruned_snapshot_writes = self
            .stats
            .pruned_snapshot_writes
            .saturating_add(trace.stats.pruned_snapshot_writes as u64);
        self.stats.eliminated_trace_guards = self
            .stats
            .eliminated_trace_guards
            .saturating_add(trace.stats.eliminated_guards as u64);
        self.stats.eliminated_trace_boxes = self
            .stats
            .eliminated_trace_boxes
            .saturating_add(trace.stats.eliminated_boxes as u64);
        self.stats.eliminated_trace_copies = self
            .stats
            .eliminated_trace_copies
            .saturating_add(trace.stats.eliminated_copies as u64);
        self.stats.trace_gvn_eliminated = self
            .stats
            .trace_gvn_eliminated
            .saturating_add(trace.stats.gvn_eliminated as u64);
        self.stats.trace_licm_hoisted = self
            .stats
            .trace_licm_hoisted
            .saturating_add(trace.stats.licm_hoisted as u64);
        self.stats.trace_pre_eliminated = self
            .stats
            .trace_pre_eliminated
            .saturating_add(trace.stats.pre_eliminated as u64);

        let mut compiled = match self.compile_trace(&trace) {
            Ok(compiled) => compiled,
            Err(error) => {
                self.stats.recordings_aborted = self.stats.recordings_aborted.saturating_add(1);
                self.stats.side_recordings_aborted = self.stats.side_recordings_aborted.saturating_add(1);
                self.stats.last_abort = Some(error.to_string());
                if jit_events_enabled() {
                    eprintln!(
                        "[wren-jit] side compile abort {:?}:{}: {}",
                        start.function, start.pc.0, error
                    );
                }
                return Ok(None);
            }
        };

        let parent_depth = self
            .compiled(parent)
            .map(|compiled| compiled.tree_depth)
            .unwrap_or(0);
        compiled.root_trace = link_to;
        compiled.tree_depth = parent_depth.saturating_add(1);
        self.insert_compiled(trace_id, compiled);
        self.stitch(parent, parent_exit, trace_id)?;
        self.stats.traces_compiled = self.stats.traces_compiled.saturating_add(1);
        self.stats.side_traces_compiled = self.stats.side_traces_compiled.saturating_add(1);

        if jit_events_enabled() {
            eprintln!(
                "[wren-jit] compiled side {:?} at {:?}:{} -> root {:?}:{} (bytecodes={}, exits={})",
                trace_id,
                start.function,
                start.pc.0,
                close_at.function,
                close_at.pc.0,
                trace.stats.recorded_bytecodes,
                trace.exits.len(),
            );
        }

        Ok(Some(trace_id))
    }

    fn apply_side_exit<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        trace_id: TraceId,
        activation: &TraceActivation,
    ) -> Result<SideExitDispatch, TracingJitError> {
        const MIN_PROFITABLE_NATIVE_ITERATIONS: u32 = 8;

        let exit_index = activation.exit_id as usize;
        // Snapshot option values before borrowing a compiled trace mutably below.
        // Besides satisfying Rust's aliasing rules, this keeps option loads out of
        // the mutable side-exit accounting scope.
        let hot_side_exit_threshold = self.options.hot_side_exit_threshold.max(1);
        let cooldown_threshold = hot_side_exit_threshold.min(u16::MAX as u32) as u16;
        let is_entry = self.compiled(trace_id)
            .is_some_and(|compiled| matches!(compiled.trace.anchor.kind, TraceKind::Entry));
        let budget_tracking = !is_entry && self.options.native_loop_iteration_budget != 0;

        // Update mutable exit accounting first, but do not clone the exit plan.
        // The old implementation cloned two Vecs on every side exit. A nested
        // numeric loop such as Mandelbrot can exit hundreds of thousands of times,
        // turning a cold-path metadata operation into allocator traffic on the hot
        // path. Keep only Copy metadata here and borrow the plan below.
        const MAX_SIDE_TRACE_DEPTH: u8 = 4;

        let (root_trace, tree_depth, root_anchor) = {
            let compiled = self
                .compiled(trace_id)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;
            let root_trace = compiled.root_trace;
            let tree_depth = compiled.tree_depth;
            let fallback_anchor = compiled.trace.anchor.location;
            let root_anchor = self
                .compiled(root_trace)
                .map(|root| root.trace.anchor.location)
                .unwrap_or(fallback_anchor);
            (root_trace, tree_depth, root_anchor)
        };

        let (plan_id, target, watchdog_anchor, cooldown, attempt_side_trace) = {
            let compiled = self
                .compiled_mut(trace_id)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;

            let plan = compiled
                .exits
                .get(exit_index)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;
            let plan_id = plan.id;
            let target = plan.target;

            let hits = if let Some(hit) = compiled.exit_hits.get_mut(exit_index) {
                *hit = hit.saturating_add(1);
                *hit
            } else {
                0
            };

            let watchdog_anchor = (compiled.trace.native_budget_exit == Some(plan_id))
                .then_some(root_anchor);

            // Native entry has a fixed transition cost. Repeatedly entering a trace
            // for only a handful of iterations is slower than baseline even when the
            // generated machine code is good. This matters for inner Mandelbrot
            // loops, which may be entered once per pixel and escape very quickly.
            let unprofitable_exit = budget_tracking
                && watchdog_anchor.is_none()
                && matches!(target, ExitTarget::Interpreter(_))
                && activation.native_iterations < MIN_PROFITABLE_NATIVE_ITERATIONS;
            if unprofitable_exit {
                compiled.short_exit_streak = compiled.short_exit_streak.saturating_add(1);
            } else {
                compiled.short_exit_streak = 0;
            }
            // A repeatedly short exit is precisely where a side trace can remove
            // the native -> baseline -> native transition. Give trace-tree formation
            // priority over cooldown once the exit itself is hot.
            let side_trace_candidate = !is_entry && watchdog_anchor.is_none()
                && tree_depth < MAX_SIDE_TRACE_DEPTH
                && matches!(target, ExitTarget::Interpreter(_))
                && hits >= hot_side_exit_threshold
                && compiled
                    .side_trace_attempted
                    .get(exit_index)
                    .copied()
                    .is_some_and(|attempted| !attempted);
            let cooldown = compiled.short_exit_streak >= cooldown_threshold
                && !side_trace_candidate;
            if cooldown {
                compiled.short_exit_streak = 0;
            }

            let attempt_side_trace = side_trace_candidate;
            if attempt_side_trace {
                if let Some(attempted) = compiled.side_trace_attempted.get_mut(exit_index) {
                    *attempted = true;
                }
            }

            (plan_id, target, watchdog_anchor, cooldown, attempt_side_trace)
        };

        self.stats.native_iterations = self
            .stats
            .native_iterations
            .saturating_add(activation.native_iterations as u64);
        if budget_tracking
            && activation.native_iterations < MIN_PROFITABLE_NATIVE_ITERATIONS
            && watchdog_anchor.is_none()
        {
            self.stats.short_native_exits = self.stats.short_native_exits.saturating_add(1);
        }

        // Borrow the plan in place. No Vec clone, no heap allocation, no temporary
        // ownership transfer on the side-exit path.
        {
            let compiled = self
                .compiled(trace_id)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;
            let plan = compiled
                .exits
                .get(exit_index)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;

            for write in &plan.writes {
                let ExitWriteSource::Constant(constant) = write.source else {
                    // Dynamic values were committed directly by the native cold
                    // exit block. Avoid a second copy through Rust.
                    continue;
                };
                let value = runtime
                    .trace_materialize_constant(constant)
                    .map_err(runtime_error)?;
                runtime
                    .trace_write_slot(write.slot, value)
                    .map_err(runtime_error)?;
            }
        }

        if watchdog_anchor.is_some() {
            self.stats.native_budget_exits = self.stats.native_budget_exits.saturating_add(1);
        }
        if cooldown {
            self.stats.trace_cooldowns = self.stats.trace_cooldowns.saturating_add(1);
        }

        match target {
            ExitTarget::Interpreter(location) => {
                runtime.trace_set_location(location).map_err(runtime_error)?;

                if cooldown {
                    return Ok(SideExitDispatch::Action(JitAction::CoolingDown { trace: trace_id }));
                }

                // If this hot exit lands on a loop that already owns native code,
                // stitch it immediately. This removes the manager/baseline round
                // trip for common trace-tree edges.
                let hits = self.exit_hits(trace_id, plan_id).unwrap_or(0);
                if !is_entry && hits >= hot_side_exit_threshold {
                    let child = self
                        .loops
                        .get(&LoopKey { location })
                        .and_then(|state| state.trace)
                        .filter(|child| *child != trace_id);
                    if let Some(child) = child {
                        self.stitch(trace_id, plan_id, child)?;
                        self.stats.stitched_exits = self.stats.stitched_exits.saturating_add(1);
                        return Ok(SideExitDispatch::Link(child));
                    }
                }

                // A hot trace-tree exit gets one attempt to record the alternate
                // path until it rejoins the root header. Bounded recursive side
                // traces cover common branch shapes without requiring a region CFG.
                if attempt_side_trace {
                    if self
                        .try_record_side_trace(
                            runtime,
                            trace_id,
                            plan_id,
                            location,
                            root_anchor,
                            root_trace,
                        )?
                        .is_some()
                    {
                        // Recording executed the side path once and stopped at the
                        // root header, so continue with the parent root now rather
                        // than executing the newly compiled side twice.
                        if runtime.trace_current_location() == root_anchor {
                            return Ok(SideExitDispatch::Link(root_trace));
                        }
                    }
                }

                Ok(SideExitDispatch::Action(JitAction::SideExit {
                    trace: trace_id,
                    exit: plan_id,
                    target: runtime.trace_current_location(),
                }))
            }

            ExitTarget::Trace(child) => {
                self.stats.stitched_exits = self.stats.stitched_exits.saturating_add(1);
                Ok(SideExitDispatch::Link(child))
            }

            ExitTarget::Return => Err(TracingJitError::CraneliftLowering {
                trace: trace_id,
                message: "side-exit Return target is not enabled".to_owned(),
            }),
        }
    }

    /// Link a side exit to an already compiled child trace.
    ///
    /// Native-linkable exits update a stable per-trace child table consumed by the
    /// generated Cranelift dispatcher. Runtime-owned constants keep the manager
    /// fallback because they cannot be reconstructed without VM access.
    pub fn stitch(
        &mut self,
        parent: TraceId,
        exit: SideExitId,
        child: TraceId,
    ) -> Result<(), TracingJitError> {
        if self.compiled(child).is_none() {
            return Err(TracingJitError::CraneliftLowering {
                trace: parent,
                message: format!("cannot stitch to unknown child {child:?}"),
            });
        }

        let parent_trace = self
            .compiled_mut(parent)
            .ok_or_else(|| TracingJitError::CraneliftLowering {
                trace: parent,
                message: "unknown parent trace".to_owned(),
            })?;

        if parent_trace.trace.native_budget_exit == Some(exit) {
            return Err(TracingJitError::CraneliftLowering {
                trace: parent,
                message: "native watchdog exits cannot be stitched to child traces".to_owned(),
            });
        }

        let exit_index = exit.0 as usize;
        let native_linkable = {
            let plan = parent_trace
                .exits
                .get_mut(exit_index)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: parent,
                    exit: exit.0,
                })?;
            plan.target = ExitTarget::Trace(child);
            plan.native_linkable
        };
        if native_linkable {
            if let Some(slot) = parent_trace.native_links.get_mut(exit_index) {
                *slot = child.0;
            }
        }

        if let Some(side_exit) = parent_trace.trace.exits.get_mut(exit_index) {
            side_exit.target = ExitTarget::Trace(child);
            side_exit.linked_trace = Some(child);
        }

        Ok(())
    }

    pub fn exit_hits(&self, trace: TraceId, exit: SideExitId) -> Option<u32> {
        self.compiled(trace)?
            .exit_hits
            .get(exit.0 as usize)
            .copied()
    }

    pub fn should_record_side_trace(
        &self,
        trace: TraceId,
        exit: SideExitId,
    ) -> bool {
        if self
            .compiled(trace)
            .is_some_and(|compiled| compiled.trace.native_budget_exit == Some(exit))
        {
            return false;
        }
        self.exit_hits(trace, exit)
            .is_some_and(|hits| hits >= self.options.hot_side_exit_threshold)
    }

    pub fn is_blacklisted(&self, location: TraceLocation) -> bool {
        self.loops
            .get(&LoopKey { location })
            .is_some_and(|state| state.blacklisted)
    }
}

// =============================================================================
// Tests that do not require executing generated native code
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn run_scalar_trace(trace: &Trace, registers: &mut [RegisterWord], epoch: u64, budget: u32) -> u32 {
        unsafe extern "C" fn unexpected_runtime(_: *mut TraceActivation, _: u32, _: u32, _: u32) -> u32 { panic!("unexpected runtime") }
        let mut jit = TracingJit::new(TracingJitOptions { native_loop_iteration_budget: budget, ..Default::default() });
        let compiled = jit.compile_trace(trace).unwrap();
        let mut activation = TraceActivation::new(TraceActivationInputs {
            general: registers.as_mut_ptr(), arguments: std::ptr::null_mut(), returns: std::ptr::null_mut(),
            special: std::ptr::null_mut(), runtime: std::ptr::null_mut(), execute: unexpected_runtime,
            heap_chunks: std::ptr::null(), heap_write_barrier: noop_heap_write_barrier,
        });
        activation.current_links = compiled.native_links.as_ptr();
        activation.method_epoch = epoch;
        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        activation.exit_id
    }

    #[test]
    fn optimized_affine_loop_matches_native_wraparound_and_budget_snapshots() {
        let mut b = crate::wren_trace_ir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Loop, location: TraceLocation::new(FuncId(0), BytecodePc(0)) });
        let slot = FrameSlot::General(Reg(0));
        let start = b.read_slot(slot, TraceType::Int32, TraceRep::Int32);
        let phi = b.add_loop_phi(slot, start, TraceType::Int32, TraceRep::Int32);
        let three = b.constant(TraceConstant::Int32(3));
        let seven = b.constant(TraceConstant::Int32(7));
        let product = b.emit1(TraceInst::IntBinary { op: IntBinOp::Mul, lhs: phi, rhs: seven }, TraceType::Int32, TraceRep::Int32);
        let next = b.emit1(TraceInst::IntBinary { op: IntBinOp::Add, lhs: phi, rhs: three }, TraceType::Int32, TraceRep::Int32);
        b.write_slot(slot, next);
        b.write_slot(FrameSlot::General(Reg(1)), product);
        b.set_loop_backedge(phi, next).unwrap();
        b.add_native_budget_exit(BytecodePc(0));
        b.set_terminator(TraceTerminator::LoopBack);
        let original = b.finish().unwrap();
        let mut optimized = original.clone();
        crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(&mut optimized).unwrap();
        assert_eq!(optimized.stats.strength_reduced, 1);
        for input in [0, -1, i32::MIN, i32::MAX - 2] {
            for budget in [1, 3, 17] {
                let mut baseline = [RegisterWord(input as u32 as u64), RegisterWord(0)];
                let mut actual = baseline;
                for _ in 0..3 {
                    run_scalar_trace(&original, &mut baseline, 0, budget);
                    run_scalar_trace(&optimized, &mut actual, 0, budget);
                    assert_eq!(actual, baseline, "input={input} budget={budget}");
                }
            }
        }
    }

    #[test]
    fn method_epoch_guard_deopts_before_inlined_effects() {
        let mut b = crate::wren_trace_ir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Side, location: TraceLocation::new(FuncId(0), BytecodePc(0)) });
        let failure = b.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        b.emit0(TraceInst::GuardMethodEpoch { expected: 42, exit: failure });
        let result = b.constant(TraceConstant::Int32(7));
        b.write_slot(FrameSlot::General(Reg(0)), result);
        let success = b.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        b.set_terminator(TraceTerminator::Exit(success));
        let trace = b.finish().unwrap();
        for epoch in [41, 42, 43] {
            let mut registers = [RegisterWord(99)];
            let exit = run_scalar_trace(&trace, &mut registers, epoch, 0);
            assert_eq!(exit, if epoch == 42 { success.0 } else { failure.0 });
            assert_eq!(registers[0].0, if epoch == 42 { 7 } else { 99 });
        }
    }

    #[test]
    fn loop_memory_pre_matches_native_first_iteration_and_reentry() {
        let mut b = crate::wren_trace_ir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Loop, location: TraceLocation::new(FuncId(0), BytecodePc(0)) });
        let object = b.read_slot(FrameSlot::General(Reg(0)), TraceType::Any, TraceRep::Tagged);
        let exit = b.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        let value = b.emit1(TraceInst::HeapLoadCell { cell: object, exit }, TraceType::Any, TraceRep::Tagged);
        let seven = b.constant(TraceConstant::NumBits(7.0f64.to_bits()));
        b.emit0(TraceInst::HeapStoreCell { cell: object, value: seven, exit });
        b.write_slot(FrameSlot::General(Reg(1)), value);
        b.add_native_budget_exit(BytecodePc(0));
        b.set_terminator(TraceTerminator::LoopBack);
        let original = b.finish().unwrap();
        let mut optimized = original.clone();
        crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(&mut optimized).unwrap();
        assert_eq!(optimized.stats.loop_loads_reused, 1);
        unsafe extern "C" fn unexpected_runtime(_: *mut TraceActivation, _: u32, _: u32, _: u32) -> u32 { TRACE_RESULT_RUNTIME }
        for budget in [1, 2, 5] {
            for trace in [&original, &optimized] {
                let mut jit = TracingJit::new(TracingJitOptions { native_loop_iteration_budget: budget, ..Default::default() });
                let compiled = jit.compile_trace(trace).unwrap();
                let mut cell = RegisterWord(0);
                let mut chunk: Vec<_> = (0..TRACE_HEAP_CHUNK_SIZE).map(|_| TraceHeapEntry::new(0)).collect();
                chunk[0].address.store(1, Ordering::Relaxed);
                chunk[0].kind.store(TraceHeapKind::Cell as u32, Ordering::Relaxed);
                chunk[0].data.store(&mut cell as *mut RegisterWord as usize, Ordering::Relaxed);
                let chunks = [chunk.as_mut_ptr()];
                let mut registers = [RegisterWord(TAG_HEAP), RegisterWord(0)];
                let mut activation = TraceActivation::new(TraceActivationInputs {
                    general: registers.as_mut_ptr(), arguments: std::ptr::null_mut(), returns: std::ptr::null_mut(),
                    special: std::ptr::null_mut(), runtime: std::ptr::null_mut(), execute: unexpected_runtime,
                    heap_chunks: chunks.as_ptr(), heap_write_barrier: noop_heap_write_barrier,
                });
                activation.current_links = compiled.native_links.as_ptr();
                for initial in [42.0f64, -10.0] {
                    cell.0 = initial.to_bits();
                    activation.reset_result();
                    assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
                    assert_eq!(f64::from_bits(cell.0), 7.0);
                    assert_eq!(f64::from_bits(registers[1].0), if budget == 1 { initial } else { 7.0 });
                }
            }
        }
    }

    #[test]
    fn range_optimization_preserves_native_failure_snapshots() {
        let mut b = crate::wren_trace_ir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Side, location: TraceLocation::new(FuncId(0), BytecodePc(0)) });
        let input = b.read_slot(FrameSlot::General(Reg(0)), TraceType::Int32, TraceRep::Int32);
        let mask = b.constant(TraceConstant::Int32(15));
        let length = b.constant(TraceConstant::Int32(16));
        let masked = b.emit1(TraceInst::IntBinary { op: IntBinOp::BitAnd, lhs: input, rhs: mask }, TraceType::Int32, TraceRep::Int32);
        b.write_slot(FrameSlot::General(Reg(1)), masked);
        let failure = b.add_side_exit(ExitReason::GuardBounds, BytecodePc(1));
        b.emit0(TraceInst::GuardBounds { index: masked, length, exit: failure });
        b.emit0(TraceInst::GuardBounds { index: input, length, exit: failure });
        let sum = b.emit1(TraceInst::IntBinaryChecked { op: IntBinOp::Add, lhs: input, rhs: length, exit: failure }, TraceType::Int32, TraceRep::Int32);
        b.write_slot(FrameSlot::General(Reg(2)), sum);
        let success = b.add_side_exit(ExitReason::GuardValue, BytecodePc(2));
        b.set_terminator(TraceTerminator::Exit(success));
        let original = b.finish().unwrap();
        let mut optimized = original.clone();
        crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(&mut optimized).unwrap();
        assert_eq!(optimized.stats.eliminated_bounds_checks, 1);
        assert_eq!(optimized.stats.eliminated_overflow_checks, 1);
        for input in [i32::MIN, -1, 0, 15, 16, i32::MAX] {
            let mut baseline = [RegisterWord(input as u32 as u64), RegisterWord(0), RegisterWord(99)];
            let mut actual = baseline;
            assert_eq!(run_scalar_trace(&original, &mut baseline, 0, 0), run_scalar_trace(&optimized, &mut actual, 0, 0));
            assert_eq!(actual, baseline);
        }
    }

    #[test]
    fn native_math_matches_rust_including_ieee_edge_cases() {
        unsafe extern "C" fn unexpected_runtime(
            _: *mut TraceActivation, _: u32, _: u32, _: u32,
        ) -> u32 { TRACE_RESULT_RUNTIME }

        let inputs = [
            -0.0, 0.0, -0.5, 0.5, -1.5, 1.5, -2.5, 2.5,
            f64::MIN_POSITIVE, f64::from_bits(1), -f64::from_bits(1),
            f64::MAX, -f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
        ];
        for math_op in [
            TraceMathOp::Abs, TraceMathOp::Floor, TraceMathOp::Ceil, TraceMathOp::Sqrt,
            TraceMathOp::Round, TraceMathOp::Sin, TraceMathOp::Cos, TraceMathOp::Tan,
            TraceMathOp::Log, TraceMathOp::Exp,
        ] {
            for optimize in [false, true] {
                let location = TraceLocation::new(FuncId(0), BytecodePc(0));
                let mut builder = crate::wren_trace_ir::TraceBuilder::new(
                    TraceId(0), TraceAnchor { kind: TraceKind::Side, location });
                let value = builder.read_slot(FrameSlot::General(Reg(0)), TraceType::F64, TraceRep::F64);
                let result = builder.emit1(TraceInst::F64Math { op: math_op, value },
                    TraceType::F64, TraceRep::F64);
                builder.write_slot(FrameSlot::General(Reg(1)), result);
                let exit = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
                builder.set_terminator(TraceTerminator::Exit(exit));
                let mut trace = builder.finish().unwrap();
                if optimize {
                    crate::wren_trace_ir_optimizing::optimization_passes().optimize_trace(&mut trace).unwrap();
                }
                let mut jit = TracingJit::new(TracingJitOptions::default());
                let compiled = jit.compile_trace(&trace).unwrap();
                for input in inputs {
                    let expected = match math_op {
                        TraceMathOp::Abs => input.abs(),
                        TraceMathOp::Floor => input.floor(),
                        TraceMathOp::Ceil => input.ceil(),
                        TraceMathOp::Sqrt => input.sqrt(),
                        TraceMathOp::Round => input.round(),
                        TraceMathOp::Sin => input.sin(),
                        TraceMathOp::Cos => input.cos(),
                        TraceMathOp::Tan => input.tan(),
                        TraceMathOp::Log => input.ln(),
                        TraceMathOp::Exp => input.exp(),
                    };
                    let mut general = [RegisterWord(input.to_bits()), RegisterWord(0)];
                    let mut activation = TraceActivation::new(TraceActivationInputs {
                        general: general.as_mut_ptr(),
                        arguments: std::ptr::null_mut(), returns: std::ptr::null_mut(),
                        special: std::ptr::null_mut(), runtime: std::ptr::null_mut(),
                        execute: unexpected_runtime,
                        heap_chunks: std::ptr::null(),
                        heap_write_barrier: noop_heap_write_barrier,
                    });
                    activation.current_links = compiled.native_links.as_ptr();
                    assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
                    let actual = f64::from_bits(general[1].0);
                    assert!(if expected.is_nan() { actual.is_nan() }
                        else { actual.to_bits() == expected.to_bits() },
                        "{math_op:?}({input:?}), optimize={optimize}: {actual:?} != {expected:?}");
                }
            }
        }
    }

    #[test]
    fn trace_passes_preserve_native_loop_exits_and_budget_reentry() {
        unsafe extern "C" fn unexpected_runtime(
            _: *mut TraceActivation, _: u32, _: u32, _: u32,
        ) -> u32 { TRACE_RESULT_RUNTIME }

        for tagged in [false, true] {
            let original = crate::wren_trace_ir_optimizing::tests::loop_trace(tagged);
            let mut optimized = original.clone();
            crate::wren_trace_ir_optimizing::optimization_passes()
                .optimize_trace(&mut optimized).unwrap();
            assert_eq!(optimized.stats.pre_eliminated, 1);
            for budget in [0, 2] {
                let mut outputs = Vec::new();
                for trace in [&original, &optimized] {
                    let mut jit = TracingJit::new(TracingJitOptions {
                        native_loop_iteration_budget: budget,
                        ..Default::default()
                    });
                    let compiled = jit.compile_trace(trace)
                        .unwrap_or_else(|error| panic!("tagged={tagged}, budget={budget}, stats={:?}: {error}", trace.stats));
                    let mut general = [RegisterWord(0); 8];
                    general[1] = RegisterWord(10);
                    let mut activation = TraceActivation::new(TraceActivationInputs {
                        general: general.as_mut_ptr(),
                        arguments: std::ptr::null_mut(),
                        returns: std::ptr::null_mut(),
                        special: std::ptr::null_mut(),
                        runtime: std::ptr::null_mut(),
                        execute: unexpected_runtime,
                        heap_chunks: std::ptr::null(),
                        heap_write_barrier: noop_heap_write_barrier,
                    });
                    activation.current_links = compiled.native_links.as_ptr();
                    if tagged {
                        // Failed entry specialization must leave all VM state intact.
                        general[0] = RegisterWord(TAG_TRUE);
                        let before = general.map(|v| v.0);
                        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
                        assert_eq!(general.map(|v| v.0), before);
                        general[0] = RegisterWord(0.0f64.to_bits());
                    }
                    let mut snapshots = Vec::new();
                    // Finite test bound also catches a broken loop-carried PRE value.
                    for _ in 0..6 {
                        activation.reset_result();
                        let result = unsafe { (compiled.entry)(&mut activation) };
                        assert_eq!(result, TRACE_RESULT_SIDE_EXIT);
                        snapshots.push(general.map(|v| v.0));
                        let counter = if tagged { f64::from_bits(general[0].0) as u64 }
                            else { general[0].0 };
                        if counter == 6 { break; }
                    }
                    let last = snapshots.last().unwrap();
                    assert_eq!(last[2..6], [10, 10, 12, 22]);
                    assert_eq!(snapshots.len(), if budget == 0 { 1 } else { 3 });
                    outputs.push(snapshots);
                }
                assert_eq!(outputs[0], outputs[1], "tagged={tagged}, budget={budget}");
            }
        }
    }

    #[test]
    fn nan_box_helpers_match_runtime_layout() {
        assert!(word_is_num(RegisterWord(1.0f64.to_bits())));
        assert!(word_is_num(RegisterWord(CANONICAL_NAN)));
        assert!(!word_is_num(RegisterWord(TAG_NULL)));
        assert!(!word_is_num(RegisterWord(TAG_FALSE)));
        assert!(!word_is_num(RegisterWord(TAG_HEAP | 7)));
    }

    #[test]
    fn falsey_is_exactly_null_or_false() {
        assert!(word_is_falsey(RegisterWord(TAG_NULL)));
        assert!(word_is_falsey(RegisterWord(TAG_FALSE)));
        assert!(!word_is_falsey(RegisterWord(TAG_TRUE)));
        assert!(!word_is_falsey(RegisterWord(0.0f64.to_bits())));
    }

    #[test]
    fn exact_i32_speculation_rejects_negative_zero_and_fractional_values() {
        assert_eq!(word_exact_i32(RegisterWord(42.0f64.to_bits())), Some(42));
        assert_eq!(word_exact_i32(RegisterWord(0.0f64.to_bits())), Some(0));
        assert_eq!(word_exact_i32(RegisterWord((-0.0f64).to_bits())), None);
        assert_eq!(word_exact_i32(RegisterWord(1.5f64.to_bits())), None);
        assert_eq!(word_exact_i32(RegisterWord((i32::MAX as f64 + 1.0).to_bits())), None);
    }

    #[test]
    fn relative_branch_target_is_measured_from_instruction_end() {
        assert_eq!(relative_target(BytecodePc(100), 10).0, 110);
        assert_eq!(relative_target(BytecodePc(100), (-10i32) as u32).0, 90);
    }

    #[test]
    fn activation_exit_buffer_is_large_enough_for_vm_register_file() {
        assert!(MAX_TRACE_EXIT_VALUES > 256 + 16 + 2 + 12);
    }

    #[test]
    fn trace_activation_stays_small() {
        // Side exits commit dynamic values directly to architectural register
        // banks. Reintroducing a large per-entry scratch buffer here would make
        // short/nested loops pathologically expensive again.
        assert!(std::mem::size_of::<TraceActivation>() <= 128);
    }

    #[test]
    fn loop_phi_representation_conflict_aborts_recording() {
        let options = TracingJitOptions::default();
        let anchor = TraceLocation::new(FuncId(0), BytecodePc(10));
        let recorder = TraceRecorder::new(&options, TraceId(0), anchor);
        let mut builder = crate::wren_trace_ir::TraceBuilder::new(
            TraceId(0),
            TraceAnchor { kind: TraceKind::Loop, location: anchor },
        );
        let slot = FrameSlot::General(Reg(0));
        let _initial = builder.read_slot(slot, TraceType::Num, TraceRep::Tagged);
        let raw = builder.constant(TraceConstant::Int32(1));
        builder.write_slot(slot, raw);

        let result = recorder.install_loop_phis(&mut builder);
        assert!(matches!(result, Err(RecordingAbort::RepresentationConflict)));
    }

    #[test]
    fn default_native_loop_budget_is_finite_and_can_be_disabled() {
        let default = TracingJitOptions::default();
        assert!(default.native_loop_iteration_budget > 0);
        let disabled = TracingJitOptions {
            native_loop_iteration_budget: 0,
            ..default
        };
        assert_eq!(disabled.native_loop_iteration_budget, 0);
    }

    #[test]
    fn runtime_leaf_preserves_unrelated_ssa_and_reloads_output() {
        unsafe extern "C" fn runtime_leaf(
            activation: *mut TraceActivation,
            _: u32,
            _: u32,
            _: u32,
        ) -> u32 {
            let activation = unsafe { &mut *activation };
            let general = activation.general;
            unsafe {
                *general.add(1) = RegisterWord(6);
            }
            TRACE_RESULT_CONTINUE
        }

        let location = TraceLocation::new(FuncId(0), BytecodePc(0));
        let mut builder = crate::wren_trace_ir::TraceBuilder::new(
            TraceId(76),
            TraceAnchor { kind: TraceKind::Side, location },
        );
        let preserved = builder.constant(TraceConstant::Int32(40));
        builder.write_slot(FrameSlot::General(Reg(0)), preserved);
        builder.runtime_leaf(location, BytecodePc(1));
        let runtime_result = builder.reload_slot_after_runtime(
            FrameSlot::General(Reg(1)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let sum = builder.emit1(
            TraceInst::IntBinary {
                op: IntBinOp::Add,
                lhs: preserved,
                rhs: runtime_result,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        builder.write_slot(FrameSlot::General(Reg(2)), sum);
        let success = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(2));
        builder.set_terminator(TraceTerminator::Exit(success));
        let trace = builder.finish().unwrap();

        let mut jit = TracingJit::new(TracingJitOptions::default());
        let compiled = jit.compile_trace(&trace).unwrap();
        let mut general = [RegisterWord(0); 4];
        let mut activation = TraceActivation::new(TraceActivationInputs {
            general: general.as_mut_ptr(),
            arguments: std::ptr::null_mut(),
            returns: std::ptr::null_mut(),
            special: std::ptr::null_mut(),
            runtime: std::ptr::null_mut(),
            execute: runtime_leaf,
            heap_chunks: std::ptr::null(),
            heap_write_barrier: noop_heap_write_barrier,
        });
        activation.current_links = compiled.native_links.as_ptr();

        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, success.0);
        assert_eq!(general[0].0 as u32 as i32, 40);
        assert_eq!(general[1].0 as u32 as i32, 6);
        assert_eq!(general[2].0 as u32 as i32, 46);
    }

    #[test]
    fn fulfilled_await_loads_result_natively_and_pending_promise_side_exits() {
        unsafe extern "C" fn unexpected_runtime(
            _: *mut TraceActivation, _: u32, _: u32, _: u32,
        ) -> u32 { panic!("fulfilled Await fast path must not call the runtime") }

        let location = TraceLocation::new(FuncId(0), BytecodePc(0));
        let mut builder = crate::wren_trace_ir::TraceBuilder::new(
            TraceId(79),
            TraceAnchor { kind: TraceKind::Side, location },
        );
        let promise = builder.read_slot(
            FrameSlot::General(Reg(0)),
            TraceType::Any,
            TraceRep::Tagged,
        );
        let slow = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(0));
        let value = builder.emit1(
            TraceInst::HeapAwaitFulfilled { promise, exit: slow },
            TraceType::Any,
            TraceRep::Tagged,
        );
        builder.write_slot(FrameSlot::General(Reg(1)), value);
        let success = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        builder.set_terminator(TraceTerminator::Exit(success));
        let trace = builder.finish().unwrap();

        let mut jit = TracingJit::new(TracingJitOptions::default());
        let compiled = jit.compile_trace(&trace).unwrap();
        let mut chunk = (0..TRACE_HEAP_CHUNK_SIZE)
            .map(|_| TraceHeapEntry::new(0))
            .collect::<Vec<_>>();
        let mut chunks = vec![chunk.as_mut_ptr()];
        let entry = &chunk[0];
        entry.address.store(1, Ordering::Relaxed);
        entry.promise_value.store(42.0f64.to_bits(), Ordering::Relaxed);
        entry.kind.store(
            (TraceHeapKind::Promise as u32)
                | (TRACE_PROMISE_FULFILLED << TRACE_HEAP_METADATA_SHIFT),
            Ordering::Relaxed,
        );

        let answer = RegisterWord(42.0f64.to_bits());
        let mut general = [RegisterWord(TAG_HEAP), RegisterWord(0)];
        let mut activation = TraceActivation::new(TraceActivationInputs {
            general: general.as_mut_ptr(),
            arguments: std::ptr::null_mut(),
            returns: std::ptr::null_mut(),
            special: std::ptr::null_mut(),
            runtime: std::ptr::null_mut(),
            execute: unexpected_runtime,
            heap_chunks: chunks.as_mut_ptr(),
            heap_write_barrier: noop_heap_write_barrier,
        });
        activation.current_links = compiled.native_links.as_ptr();

        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, success.0);
        assert_eq!(general[1], answer);

        entry.kind.store(
            (TraceHeapKind::Promise as u32)
                | (TRACE_PROMISE_PENDING << TRACE_HEAP_METADATA_SHIFT),
            Ordering::Relaxed,
        );
        activation.reset_result();
        general[1] = RegisterWord(0);
        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, slow.0);
    }

    #[test]
    fn checked_integer_division_lowers_and_exits_on_invalid_inputs() {
        unsafe extern "C" fn unexpected_runtime(
            _: *mut TraceActivation, _: u32, _: u32, _: u32,
        ) -> u32 { TRACE_RESULT_RUNTIME }

        let location = TraceLocation::new(FuncId(0), BytecodePc(0));
        let mut builder = crate::wren_trace_ir::TraceBuilder::new(
            TraceId(77),
            TraceAnchor { kind: TraceKind::Side, location },
        );
        let lhs = builder.read_slot(
            FrameSlot::General(Reg(0)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let rhs = builder.read_slot(
            FrameSlot::General(Reg(1)),
            TraceType::Int32,
            TraceRep::Int32,
        );
        let invalid = builder.add_side_exit(ExitReason::GuardInteger, BytecodePc(0));
        let quotient = builder.emit1(
            TraceInst::IntBinaryChecked {
                op: IntBinOp::DivS,
                lhs,
                rhs,
                exit: invalid,
            },
            TraceType::Int32,
            TraceRep::Int32,
        );
        builder.write_slot(FrameSlot::General(Reg(2)), quotient);
        let success = builder.add_side_exit(ExitReason::GuardValue, BytecodePc(1));
        builder.set_terminator(TraceTerminator::Exit(success));
        let trace = builder.finish().unwrap();

        let mut jit = TracingJit::new(TracingJitOptions::default());
        let compiled = jit.compile_trace(&trace).unwrap();
        let mut general = [RegisterWord(0); 4];
        let mut activation = TraceActivation::new(TraceActivationInputs {
            general: general.as_mut_ptr(),
            arguments: std::ptr::null_mut(),
            returns: std::ptr::null_mut(),
            special: std::ptr::null_mut(),
            runtime: std::ptr::null_mut(),
            execute: unexpected_runtime,
            heap_chunks: std::ptr::null(),
            heap_write_barrier: noop_heap_write_barrier,
        });
        activation.current_links = compiled.native_links.as_ptr();

        general[0] = RegisterWord(21);
        general[1] = RegisterWord(3);
        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, success.0);
        assert_eq!(general[2].0 as u32 as i32, 7);

        activation.reset_result();
        general[0] = RegisterWord(21);
        general[1] = RegisterWord(0);
        general[2] = RegisterWord(0xdead_beef);
        assert_eq!(unsafe { (compiled.entry)(&mut activation) }, TRACE_RESULT_SIDE_EXIT);
        assert_eq!(activation.exit_id, invalid.0);
        assert_eq!(general[2].0, 0xdead_beef);
    }

}
