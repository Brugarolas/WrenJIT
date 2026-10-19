//! Wren tracing JIT.
//!
//! This module connects the register interpreter to the backend-agnostic trace IR
//! (`wren_traceir.rs`) and lowers optimized traces to B3.
//!
//! The implementation deliberately separates four concerns:
//!
//! 1. `TraceRuntime` is the very small adapter the interpreter implements.
//!    `wren_tracingjit` never reaches into private `Frame`, `FiberState` or heap
//!    structures directly.
//! 2. `TraceRecorder` observes exactly one execution of original (unfused)
//!    bytecode and builds a linear SSA `Trace`.
//! 3. `B3TraceCompiler` lowers that trace to native code.  Wren architectural
//!    registers remain virtual while native code runs.
//! 4. `TracingJit` owns hotness, compiled-code lifetime, side-exit accounting and
//!    future trace-stitching metadata.
//!
//! Important ABI property:
//!
//! Native traces receive one pointer, `*mut TraceActivation`.  The structure
//! contains pointers to the current Wren register banks plus a compact side-exit
//! result area.  B3 `check` stack-map generators materialize *only* values needed
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
//! Numeric/control-flow operations lower directly to B3. Other bytecodes lower
//! to synchronized runtime calls. Calls which change frames/fibers terminate a
//! trace AFTER executing the opcode; the manager can compile the continuation.
//! No architectural register pointer or uncommitted GC root crosses such a call.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::mem::offset_of;
use std::rc::Rc;

use b3::air::generate::emit_function_epilogue;
use b3::jit::compilation::Compilation;
use b3::jit::reg::Reg as B3Reg;
use b3::macroassembler::{
    assembler::abstract_macro_assembler::Address,
    jit::gpr_info::{ARGUMENT_GPR0, CALL_FRAME_REGISTER, RETURN_VALUE_GPR, T1},
};
use b3::{BasicBlockBuilder, Frequency, OptLevel, ValueRep, ValueRepKind};

use crate::wren_bytecode::{
    instruction_word_len, ArgReg, BytecodeFunction, BytecodePc, BytecodeProgram,
    Opcode, Reg, RegisterKind, RegisterWord, ReturnReg,
    SpecialReg,
};
use crate::wren_ir::{
    CmpKind, Constant, F64BinOp, F64UnaryOp, FuncId, IntBinOp, IntUnaryOp, MethodKind,
};
use crate::wren_traceir::{
    verify_trace, ExitReason, ExitTarget, FrameDisposition, FrameSlot,
    SnapshotValue, SideExit, SideExitId, Trace, TraceAnchor, TraceClass,
    TraceConstant, TraceId, TraceInst, TraceKind, TraceLocation, TraceOptimizer,
    TraceRep, TraceSwitchGuard, TraceTerminator, TraceType, TraceValueId,
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

// =============================================================================
// Native trace ABI
// =============================================================================

pub const TRACE_RESULT_CONTINUE: u32 = 0;
pub const TRACE_RESULT_SIDE_EXIT: u32 = 1;
pub const TRACE_RESULT_RETURNED: u32 = 2;
pub const TRACE_RESULT_LINK: u32 = 3;
pub const TRACE_RESULT_RUNTIME: u32 = 4;

/// Maximum number of *dynamic* values one side exit may materialize.
///
/// Snapshots are sparse. A normal loop exit normally uses only a handful of
/// entries. This is deliberately larger than the Wren GPR count to leave room
/// for future inlined-frame reconstruction.
pub const MAX_TRACE_EXIT_VALUES: usize = 768;

#[repr(C, align(16))]
pub struct TraceActivation {
    /// Base of the current Wren frame's contiguous GPR bank.
    pub general: *mut RegisterWord,
    pub arguments: *mut RegisterWord,
    pub returns: *mut RegisterWord,
    pub special: *mut RegisterWord,

    /// Interpreter pointer/context reserved for later runtime-helper calls.
    pub runtime: *mut c_void,
    pub execute: RuntimeTraceFn,

    /// Written only by native side exits.
    pub exit_id: u32,
    pub result_code: u32,

    /// Raw `RegisterWord` representation of dynamic side-exit values.
    pub exit_values: [u64; MAX_TRACE_EXIT_VALUES],
}

impl TraceActivation {
    pub fn new(inputs: TraceActivationInputs) -> Self {
        Self {
            general: inputs.general,
            arguments: inputs.arguments,
            returns: inputs.returns,
            special: inputs.special,
            runtime: inputs.runtime,
            execute: inputs.execute,
            exit_id: u32::MAX,
            result_code: TRACE_RESULT_CONTINUE,
            exit_values: [0; MAX_TRACE_EXIT_VALUES],
        }
    }

    #[inline]
    pub fn reset_result(&mut self) {
        self.exit_id = u32::MAX;
        self.result_code = TRACE_RESULT_CONTINUE;
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

/// Thin interface implemented by `wren_interpreter::Interpreter`.
///
/// Keeping this trait here makes the JIT independently testable. The interpreter
/// implementation should:
///
/// - expose pointers for the *current* frame in `trace_activation_inputs`;
/// - execute `trace_execute_unfused_step` using the original opcode handler, not
///   an installed superinstruction;
/// - keep those activation pointers stable while native code executes. The first
///   native tier emits no allocating/runtime calls, so this naturally holds.
pub trait TraceRuntime {
    type RuntimeError: Error + Send + Sync + 'static;

    fn trace_program(&self) -> &BytecodeProgram;

    /// Only specialize a numeric Invoke when its actual method target has the
    /// standard semantics. A method name alone is insufficient for host bindings.
    fn trace_standard_numeric_method(&self, _function: FuncId, _cache: u32) -> bool { false }

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

    fn trace_activation_inputs(
        &mut self,
    ) -> Result<TraceActivationInputs, Self::RuntimeError>;

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

#[derive(Clone, Debug)]
pub struct TracingJitOptions {
    /// Number of visits to a loop header before starting a recording.
    pub hot_loop_threshold: u32,
    /// Hard recorder limit for one loop iteration.
    pub max_trace_steps: usize,
    /// Repeated recording/lowering failures before a loop is permanently ignored.
    pub max_failures_before_blacklist: u16,
    /// Start considering a side exit for a child trace after this many hits.
    pub hot_side_exit_threshold: u32,

    pub optimize_trace_ir: bool,
    pub b3_opt_level: OptLevel,
    pub b3_enable_sccp: bool,
    pub b3_enable_licm: bool,
    pub b3_dump_ir: bool,
    pub b3_dump_air: bool,

    /// Restrict compilation to numeric traces, for diagnostics/comparison.
    pub no_runtime_calls_in_native_trace: bool,
}

impl Default for TracingJitOptions {
    fn default() -> Self {
        Self {
            hot_loop_threshold: 64,
            max_trace_steps: 4096,
            max_failures_before_blacklist: 8,
            hot_side_exit_threshold: 32,

            optimize_trace_ir: true,
            b3_opt_level: OptLevel::O3,
            b3_enable_sccp: true,
            b3_enable_licm: true,
            b3_dump_ir: false,
            b3_dump_air: false,

            no_runtime_calls_in_native_trace: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TracingJitStats {
    pub loop_header_hits: u64,
    pub recordings_started: u64,
    pub recordings_succeeded: u64,
    pub recordings_aborted: u64,
    pub traces_compiled: u64,
    pub native_entries: u64,
    pub side_exits: u64,
    pub blacklisted_loops: u64,
    pub stitched_exits: u64,
    pub recorded_bytecodes: u64,
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
    B3Lowering {
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
            Self::B3Lowering { trace, message } => {
                write!(f, "B3 lowering failed for {trace:?}: {message}")
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

struct TraceRecorder<'a> {
    options: &'a TracingJitOptions,
    trace_id: TraceId,
    anchor: TraceLocation,
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
        builder: &mut crate::wren_traceir::TraceBuilder,
        reg: u8,
        ty: TraceType,
        rep: TraceRep,
    ) -> Result<TraceValueId, RecordingAbort> {
        self.read_slot(builder, FrameSlot::General(Reg(reg)), ty, rep)
    }

    fn read_slot(
        &self,
        builder: &mut crate::wren_traceir::TraceBuilder,
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
        builder: &mut crate::wren_traceir::TraceBuilder,
        reg: u8,
        value: TraceValueId,
    ) {
        builder.write_slot(FrameSlot::General(Reg(reg)), value);
    }

    fn tagged_bool(
        &self,
        builder: &mut crate::wren_traceir::TraceBuilder,
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

    fn num_to_f64_guard(
        &self,
        builder: &mut crate::wren_traceir::TraceBuilder,
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
        builder: &mut crate::wren_traceir::TraceBuilder,
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

    fn record_num_invoke<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        function: &BytecodeFunction,
        op: &RecordedOp,
        builder: &mut crate::wren_traceir::TraceBuilder,
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
        // Wren's host bitwise operators use i64, whereas typed I* bytecodes use
        // i32. Keep the former in the runtime, including wide shift counts.
        if !matches!(name, "+" | "-" | "*" | "/" | "<" | "<=" | ">" | ">=" | "==" | "!=") { return Ok(false); }

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
                let raw = self.num_to_i32_guard(builder, receiver, resume);
                let result = builder.emit1(
                    TraceInst::IntUnary {
                        op: IntUnaryOp::BitNot,
                        value: raw,
                    },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxInt32 { value: result },
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
            "+" | "-" | "*" | "/" => {
                let lhs = self.num_to_f64_guard(builder, receiver, resume);
                let rhs = self.num_to_f64_guard(builder, rhs, resume);
                let bin = match name {
                    "+" => F64BinOp::Add,
                    "-" => F64BinOp::Sub,
                    "*" => F64BinOp::Mul,
                    "/" => F64BinOp::Div,
                    _ => unreachable!(),
                };
                let raw = builder.emit1(
                    TraceInst::F64Binary {
                        op: bin,
                        lhs,
                        rhs,
                    },
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
                let lhs = self.num_to_f64_guard(builder, receiver, resume);
                let rhs = self.num_to_f64_guard(builder, rhs, resume);
                let cmp = match name {
                    "<" => CmpKind::Lt,
                    "<=" => CmpKind::Le,
                    ">" => CmpKind::Gt,
                    ">=" => CmpKind::Ge,
                    "==" => CmpKind::Eq,
                    "!=" => CmpKind::Ne,
                    _ => unreachable!(),
                };
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
                let lhs = self.num_to_i32_guard(builder, receiver, resume);
                let rhs = self.num_to_i32_guard(builder, rhs, resume);
                let bin = match name {
                    "&" => IntBinOp::BitAnd,
                    "|" => IntBinOp::BitOr,
                    "^" => IntBinOp::BitXor,
                    "<<" => IntBinOp::Shl,
                    ">>" => IntBinOp::Sar,
                    _ => unreachable!(),
                };
                let raw = builder.emit1(
                    TraceInst::IntBinary {
                        op: bin,
                        lhs,
                        rhs,
                    },
                    TraceType::Int32,
                    TraceRep::Int32,
                );
                let boxed = builder.emit1(
                    TraceInst::BoxInt32 { value: raw },
                    TraceType::Num,
                    TraceRep::Tagged,
                );
                self.write_gpr(builder, op.a, boxed);
                Ok(true)
            }

            // `%` is intentionally not emitted yet. B3's public `Mod` lowering
            // has target-specific corner semantics; using a runtime helper later
            // is safer than silently changing Wren's floating remainder.
            _ => Ok(false),
        }
    }

    fn translate_op<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, function: &BytecodeFunction,
        op: &RecordedOp, builder: &mut crate::wren_traceir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        // A speculative lowering may have emitted guards before discovering a
        // representation conflict. Roll back that attempt before using a helper.
        let checkpoint = builder.clone();
        let attempt = if matches!(op.opcode, Opcode::Switch | Opcode::OsrPoint | Opcode::ReadBarrier) {
            Err(RecordingAbort::UnsupportedOpcode(op.opcode))
        } else {
            self.translate_native(runtime, program, function, op, builder)
        };
        match attempt {
            Ok(step) => Ok(step),
            Err(reason) if self.options.no_runtime_calls_in_native_trace => Err(reason),
            Err(_) => {
                *builder = checkpoint;
                builder.set_source(op.location);
                let terminal = matches!(op.opcode,
                    Opcode::Trap | Opcode::Return | Opcode::Return2 | Opcode::Bailout
                    | Opcode::Switch | Opcode::Invoke | Opcode::InvokeSuper
                    | Opcode::CallDirect | Opcode::CallClosure | Opcode::ImportModule
                    | Opcode::FiberCall | Opcode::FiberTry | Opcode::FiberTransfer
                    | Opcode::FiberTransferError | Opcode::FiberYield | Opcode::FiberSuspend
                    | Opcode::FiberAbort);
                builder.runtime(op.location, op.next_pc, terminal);
                Ok(ExpectedStep::Runtime { terminal })
            }
        }
    }

    fn translate_native<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        function: &BytecodeFunction,
        op: &RecordedOp,
        builder: &mut crate::wren_traceir::TraceBuilder,
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
                    Constant::String(_) => return Err(RecordingAbort::UnsupportedConstant),
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

                if truthy {
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
                let tag = self.read_gpr(
                    builder,
                    op.a,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                let physical = runtime
                    .trace_read_slot(FrameSlot::General(Reg(op.a)))
                    .map_err(|_| RecordingAbort::GuardFailedWhileRecording)?;

                let key = if let Some(number) = word_num(physical) {
                    number as i64
                } else {
                    physical.0 as u32 as i32 as i64
                };

                let Some(table) = function.chunk.switch_tables.get(op.x as usize) else {
                    return Err(RecordingAbort::UnsupportedOpcode(op.opcode));
                };

                let mut selected = table.default_target;
                let mut is_case = false;
                for case in &table.cases {
                    if case.value == key {
                        selected = case.target;
                        is_case = true;
                        break;
                    }
                }

                // If the speculation fails, reexecute the original switch. This
                // handles all alternate cases without one exit per case.
                let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
                let observed = if is_case {
                    TraceSwitchGuard::Case(key)
                } else {
                    TraceSwitchGuard::Default {
                        excluded: table.cases.iter().map(|case| case.value).collect(),
                    }
                };
                builder.emit0(TraceInst::GuardSwitch {
                    tag,
                    observed,
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

            GuardType | GuardClass | Bailout => {
                // Type/Class guards need the runtime's heap/class representation.
                // The IR supports them, but the first native tier deliberately
                // does not make that private representation part of its ABI.
                Err(RecordingAbort::UnsupportedOpcode(op.opcode))
            }

            Invoke => {
                if self.record_num_invoke(runtime, program, function, op, builder)? {
                    Ok(ExpectedStep::Exact(next))
                } else {
                    Err(RecordingAbort::UnsupportedCall)
                }
            }

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
                // Public B3 documents division-by-zero and MIN/-1 as unrestricted
                // for raw integer Div/Mod. Keep semantics exact by not tracing
                // these until we lower explicit checks.
                Err(RecordingAbort::UnsupportedOpcode(op.opcode))
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

            FAdd | FSub | FMul | FDiv => {
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

            FRem => Err(RecordingAbort::UnsupportedOpcode(op.opcode)),

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

            ReadBarrier => {
                let source = self.read_gpr(
                    builder,
                    op.b,
                    TraceType::Any,
                    TraceRep::Tagged,
                )?;
                self.write_gpr(builder, op.a, source);
                Ok(ExpectedStep::Exact(next))
            }

            WriteBarrier => {
                // The current interpreter barrier is a no-op. Do not bake that
                // collector detail into native code yet.
                Err(RecordingAbort::UnsupportedOpcode(op.opcode))
            }

            ProfileBlock | ProfileNumeric | Safepoint | Breakpoint => {
                Err(RecordingAbort::UnsupportedOpcode(op.opcode))
            }

            Return
            | Return2
            | Trap
            | InvokeSuper
            | CallDirect
            | CallClosure
            | CallPrimitive
            | CallForeign
            | CreateClass
            | BindMethod
            | BindForeignMethod
            | AllocInstance
            | AllocForeign
            | LoadField
            | StoreField
            | LoadStaticField
            | StoreStaticField
            | IsInstance
            | ImportModule
            | ImportVariable
            | LoadModuleVar
            | StoreModuleVar
            | AllocCell
            | LoadCell
            | StoreCell
            | MakeClosure
            | AllocList
            | AllocListCap
            | ListLen
            | ListGet
            | ListSet
            | ListAppend
            | ListInsert
            | ListRemoveAt
            | AllocMap
            | MapLen
            | MapGet
            | MapSet
            | MapRemove
            | StringLen
            | StringConcat
            | StringSlice
            | StringCodePointAt
            | MakeRange
            | MakeFiber
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
        builder: &mut crate::wren_traceir::TraceBuilder,
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
                // A physical VM register whose representation changes across the
                // backedge needs an explicit representation bridge. Keep the
                // first recorder simple and let baseline execute this loop.
                final_value = builder.emit1(TraceInst::Reinterpret { value: final_value }, ty, rep);
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
                            frame.writes.push(crate::wren_traceir::SnapshotWrite {
                                slot, kind: RegisterKind::Value, value: SnapshotValue::Value(phi),
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
        let program_ptr = runtime.trace_program() as *const BytecodeProgram;
        let program = unsafe { &*program_ptr };

        let anchor_function = program
            .function(self.anchor.function)
            .ok_or_else(|| RecordOutcome::Fatal(TracingJitError::Record {
                location: self.anchor,
                message: "trace anchor references unknown function".to_owned(),
            }))?;

        let _ = anchor_function;

        let mut builder = crate::wren_traceir::TraceBuilder::new(
            self.trace_id,
            TraceAnchor {
                kind: TraceKind::Loop,
                location: self.anchor,
            },
        );

        let mut steps = 0usize;
        let mut runtime_end = false;
        let mut end_location = None;
        let mut visited = HashSet::new();

        loop {
            if steps >= self.options.max_trace_steps {
                return Err(RecordOutcome::Abort(RecordingAbort::TraceTooLong));
            }

            let location = runtime.trace_current_location();
            if location.function != self.anchor.function {
                return Err(RecordOutcome::Abort(RecordingAbort::LeftFunction));
            }

            if steps > 0 && location == self.anchor {
                break;
            }
            if !visited.insert(location) {
                end_location = Some(location);
                break;
            }

            let function = program
                .function(location.function)
                .ok_or_else(|| RecordOutcome::Fatal(TracingJitError::Record {
                    location,
                    message: "current function disappeared during recording".to_owned(),
                }))?;

            let op = decode_at(program, location).map_err(RecordOutcome::Fatal)?;

            let expected = self
                .translate_op(runtime, program, function, &op, &mut builder)
                .map_err(RecordOutcome::Abort)?;

            let step = runtime
                .trace_execute_unfused_step()
                .map_err(|error| RecordOutcome::Fatal(runtime_error(error)))?;

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

            if !step.running {
                return Err(RecordOutcome::Abort(RecordingAbort::InterpreterStopped));
            }
            if step.frame_changed {
                return Err(RecordOutcome::Abort(RecordingAbort::FrameChanged));
            }
            if step.before != location {
                return Err(RecordOutcome::Fatal(TracingJitError::Record {
                    location,
                    message: "runtime executed a different bytecode than recorder decoded".to_owned(),
                }));
            }

            if let ExpectedStep::Exact(expected_location) = expected {
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

        let finite_exit = end_location.map(|location| builder.add_side_exit(ExitReason::GuardValue, location.pc));
        if builder.has_runtime_calls() {
            builder.prepare_memory_loop();
        } else if finite_exit.is_none() {
            self.install_loop_phis(&mut builder).map_err(RecordOutcome::Abort)?;
        }
        builder.set_terminator(if runtime_end {
            TraceTerminator::RuntimeReturn
        } else if let Some(exit) = finite_exit {
            TraceTerminator::Exit(exit)
        } else {
            TraceTerminator::LoopBack
        });

        let mut trace = builder.finish().map_err(|errors| {
            RecordOutcome::Fatal(TracingJitError::TraceVerification(
                errors.into_iter().map(|error| error.to_string()).collect(),
            ))
        })?;

        if self.options.optimize_trace_ir {
            TraceOptimizer::default().optimize(&mut trace).map_err(|errors| {
                RecordOutcome::Fatal(TracingJitError::TraceVerification(
                    errors.into_iter().map(|error| error.to_string()).collect(),
                ))
            })?;
        }

        Ok(trace)
    }
}

enum RecordOutcome {
    Abort(RecordingAbort),
    Fatal(TracingJitError),
}

#[derive(Clone, Copy, Debug)]
enum ExpectedStep {
    Exact(TraceLocation),
    Runtime { terminal: bool },
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

#[derive(Clone, Debug)]
enum ExitWriteSource {
    Constant(TraceConstant),
    Dynamic(u16),
}

#[derive(Clone, Debug)]
struct CompiledExitWrite {
    slot: FrameSlot,
    kind: RegisterKind,
    source: ExitWriteSource,
}

#[derive(Clone, Debug)]
struct CompiledExitPlan {
    id: SideExitId,
    target: ExitTarget,
    writes: Vec<CompiledExitWrite>,
    dynamic_values: Vec<TraceValueId>,
}

impl CompiledExitPlan {
    fn from_exit(trace: &Trace, exit: &SideExit) -> Result<Self, TracingJitError> {
        if exit.frames.len() != 1 {
            return Err(TracingJitError::B3Lowering {
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
            return Err(TracingJitError::B3Lowering {
                trace: trace.id,
                message: "native tier cannot yet recreate inlined Wren frames".to_owned(),
            });
        }

        let mut dynamic_values = Vec::<TraceValueId>::new();
        let mut dynamic_index = HashMap::<TraceValueId, u16>::new();
        let mut writes = Vec::with_capacity(frame.writes.len());

        for write in &frame.writes {
            let source = match write.value {
                SnapshotValue::Constant(constant) => ExitWriteSource::Constant(constant),
                SnapshotValue::Value(value) => {
                    let value = trace.canonical_value(value);
                    let index = if let Some(index) = dynamic_index.get(&value).copied() {
                        index
                    } else {
                        if dynamic_values.len() >= MAX_TRACE_EXIT_VALUES {
                            return Err(TracingJitError::B3Lowering {
                                trace: trace.id,
                                message: "side exit exceeds TraceActivation exit buffer".to_owned(),
                            });
                        }
                        let index = dynamic_values.len() as u16;
                        dynamic_values.push(value);
                        dynamic_index.insert(value, index);
                        index
                    };
                    ExitWriteSource::Dynamic(index)
                }
            };

            writes.push(CompiledExitWrite {
                slot: write.slot,
                kind: write.kind,
                source,
            });
        }

        Ok(Self {
            id: exit.id,
            target: exit.target,
            writes,
            dynamic_values,
        })
    }
}

// =============================================================================
// B3 lowering
// =============================================================================

struct CompiledTrace {
    trace: Trace,
    compilation: Compilation,
    entry: NativeTraceFn,
    exits: Vec<CompiledExitPlan>,
    exit_hits: Vec<u32>,
}

impl CompiledTrace {
    fn code_ptr(&self) -> *const u8 {
        self.compilation.code_ref().start()
    }
}

struct B3TraceCompiler<'a> {
    trace: &'a Trace,
    options: &'a TracingJitOptions,
    exit_plans: Vec<CompiledExitPlan>,
}

struct B3LoweringState<'a, 'p> {
    builder: BasicBlockBuilder<'a>,
    trace: &'p Trace,
    activation: b3::ValueId,
    general: b3::ValueId,
    arguments: b3::ValueId,
    returns: b3::ValueId,
    special: b3::ValueId,
    values: HashMap<TraceValueId, b3::ValueId>,
    phi_vars: HashMap<TraceValueId, b3::variable::VariableId>,
    exit_plans: &'p [CompiledExitPlan],
}

impl<'a> B3TraceCompiler<'a> {
    fn new(trace: &'a Trace, options: &'a TracingJitOptions) -> Result<Self, TracingJitError> {
        verify_trace(trace).map_err(|errors| {
            TracingJitError::TraceVerification(
                errors.into_iter().map(|error| error.to_string()).collect(),
            )
        })?;

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

    fn compile(self) -> Result<CompiledTrace, TracingJitError> {
        let mut options = b3::Options::default();
        options.opt_level = self.options.b3_opt_level;
        options.enable_sccp = self.options.b3_enable_sccp;
        options.use_b3_hoist_loop_invariant_values = self.options.b3_enable_licm;
        options.dump_b3_at_each_phase = self.options.b3_dump_ir;
        options.dump_air_at_each_phase = self.options.b3_dump_air;

        let mut procedure = b3::Procedure::new(options);
        let entry_block = procedure.add_block(1.0);
        let loop_block = procedure.add_block(1.0);

        let mut entry = BasicBlockBuilder::new(&mut procedure, entry_block);
        let activation = entry.argument(
            B3Reg::new_gpr(ARGUMENT_GPR0),
            b3::Type::Int64,
        );

        let general = entry.load(
            b3::Type::Int64,
            activation,
            offset_of!(TraceActivation, general) as i32,
            None,
            None,
        );
        let arguments = entry.load(
            b3::Type::Int64,
            activation,
            offset_of!(TraceActivation, arguments) as i32,
            None,
            None,
        );
        let returns = entry.load(
            b3::Type::Int64,
            activation,
            offset_of!(TraceActivation, returns) as i32,
            None,
            None,
        );
        let special = entry.load(
            b3::Type::Int64,
            activation,
            offset_of!(TraceActivation, special) as i32,
            None,
            None,
        );

        let mut state = B3LoweringState {
            builder: entry,
            trace: self.trace,
            activation,
            general,
            arguments,
            returns,
            special,
            values: HashMap::new(),
            phi_vars: HashMap::new(),
            exit_plans: &self.exit_plans,
        };

        // Entry-slot loads and constants are intentionally emitted before the
        // native loop. They are invariant unless represented by a LoopPhi.
        let value_ids: Vec<_> = self.trace.values.iter().map(|value| value.id).collect();
        for id in value_ids {
            let origin = self.trace.value(id).unwrap().origin.clone();
            if matches!(
                origin,
                TraceValueOrigin::EntrySlot { .. } | TraceValueOrigin::Constant(_)
            ) {
                let _ = state.lower_value(id)?;
            }
        }

        // Each LoopPhi is represented as a B3 variable. B3's own SSA conversion
        // subsequently turns these into backend phis/upsilons.
        for phi in &self.trace.loop_phis {
            let ty = b3_type(phi.rep);
            let variable = state.builder.procedure.add_variable(ty);
            let initial = state.lower_value(phi.initial)?;
            state.builder.var_set(variable, initial);
            state.phi_vars.insert(phi.value, variable);
        }

        state.builder.jump(Some(loop_block));
        state.builder.block = loop_block;

        for phi in &self.trace.loop_phis {
            let variable = state.phi_vars[&phi.value];
            let value = state.builder.var_get(variable);
            state.values.insert(phi.value, value);
        }

        for node in &self.trace.nodes {
            if matches!(node.kind, TraceInst::Nop) {
                continue;
            }
            state.lower_node(node)?;
        }

        match &self.trace.terminator {
            TraceTerminator::RuntimeReturn => {
                let result = state.builder.const32(TRACE_RESULT_RUNTIME as i32);
                state.builder.return_(Some(result));
            }
            TraceTerminator::LoopBack => {
                for phi in &self.trace.loop_phis {
                    let backedge = phi.backedge.ok_or_else(|| {
                        TracingJitError::B3Lowering {
                            trace: self.trace.id,
                            message: format!("{:?} has no backedge", phi.value),
                        }
                    })?;
                    let value = state.lower_value(backedge)?;
                    let variable = state.phi_vars[&phi.value];
                    state.builder.var_set(variable, value);
                }
                state.builder.jump(Some(loop_block));
            }

            TraceTerminator::Exit(exit) => {
                // An unconditional trace exit is represented by a check that is
                // always true. This keeps all exit materialization in one path.
                let one = state.builder.const32(1);
                state.emit_side_exit(*exit, one)?;
                let zero = state.builder.const32(0);
                state.builder.return_(Some(zero));
            }

            TraceTerminator::Return { .. } => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: "native Return terminator is not enabled in the first loop tier".to_owned(),
                });
            }

            TraceTerminator::Link { .. } => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: "direct native trace links are installed after compilation".to_owned(),
                });
            }

            TraceTerminator::Unreachable => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: "trace has unreachable terminator".to_owned(),
                });
            }
        }

        // Release the builder's mutable borrow of `procedure` and the borrowed
        // exit-plan slice before handing the Procedure to B3.
        drop(state);

        let compilation = b3::compile(procedure);
        let entry_ptr = compilation.code_ref().start();
        let entry: NativeTraceFn = unsafe { std::mem::transmute(entry_ptr) };

        Ok(CompiledTrace {
            trace: self.trace.clone(),
            compilation,
            entry,
            exit_hits: vec![0; self.exit_plans.len()],
            exits: self.exit_plans,
        })
    }
}

impl B3LoweringState<'_, '_> {
    /// B3's generic DToI has no Air lowering in this backend. Emit the machine
    /// conversion explicitly; callers guard range and integrality separately.
    fn double_to_int64(&mut self, value: b3::ValueId) -> b3::ValueId {
        let result = self.builder.patchpoint(b3::Type::Int64);
        self.builder.procedure.stackmap_append(result, value, ValueRep::new(ValueRepKind::SomeRegister));
        self.builder.procedure.stackmap_set_generator(result, Rc::new(|jit, params| {
            jit.truncate_double_to_int64(params[1].get_reg().fpr(), params[0].get_reg().gpr());
        }));
        result
    }

    fn lower_value(&mut self, value: TraceValueId) -> Result<b3::ValueId, TracingJitError> {
        let value = self.trace.canonical_value(value);
        if let Some(existing) = self.values.get(&value).copied() {
            return Ok(existing);
        }

        if let Some(variable) = self.phi_vars.get(&value).copied() {
            let loaded = self.builder.var_get(variable);
            self.values.insert(value, loaded);
            return Ok(loaded);
        }

        let (origin, value_rep) = {
            let trace_value = self.trace.value(value).ok_or_else(|| {
                TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: format!("unknown trace value {value:?}"),
                }
            })?;
            (trace_value.origin.clone(), trace_value.rep)
        };

        let lowered = match origin {
            TraceValueOrigin::EntrySlot { frame, slot } => {
                if frame.0 != 0 {
                    return Err(TracingJitError::B3Lowering {
                        trace: self.trace.id,
                        message: "B3 tier cannot load recreated inlined frames yet".to_owned(),
                    });
                }
                self.load_frame_slot(slot, value_rep)?
            }

            TraceValueOrigin::Constant(constant) => self.lower_constant(constant)?,

            TraceValueOrigin::InstResult { .. } => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: format!(
                        "instruction result {value:?} requested before producer was lowered"
                    ),
                });
            }

            TraceValueOrigin::LoopPhi { .. } => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: format!("loop value {value:?} has no B3 variable"),
                });
            }

            TraceValueOrigin::Alias(alias) => return self.lower_value(alias),
        };

        self.values.insert(value, lowered);
        Ok(lowered)
    }

    fn base_for_slot(&mut self, slot: FrameSlot) -> (b3::ValueId, i32) {
        let (bank, offset) = match slot {
            FrameSlot::General(reg) => (offset_of!(TraceActivation, general), reg.0 as i32 * 8),
            FrameSlot::Argument(reg) => (offset_of!(TraceActivation, arguments), reg.0 as i32 * 8),
            FrameSlot::Return(reg) => (offset_of!(TraceActivation, returns), reg.0 as i32 * 8),
            FrameSlot::Special(reg) => (offset_of!(TraceActivation, special), reg as u8 as i32 * 8),
        };
        (self.builder.load(b3::Type::Int64, self.activation, bank as i32, None, None), offset)
    }

    fn load_frame_slot(
        &mut self,
        slot: FrameSlot,
        rep: TraceRep,
    ) -> Result<b3::ValueId, TracingJitError> {
        let (base, offset) = self.base_for_slot(slot);
        let raw = self.builder.load(
            b3::Type::Int64,
            base,
            offset,
            None,
            None,
        );

        Ok(match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => raw,
            TraceRep::Int32 | TraceRep::Bool => self.builder.trunc(raw),
            TraceRep::F64 => self.builder.bitwise_cast(b3::Type::Double, raw),
        })
    }

    fn store_frame_slot(
        &mut self,
        slot: FrameSlot,
        value: b3::ValueId,
        rep: TraceRep,
    ) {
        let raw = self.normalize_to_word(value, rep);
        let (base, offset) = self.base_for_slot(slot);
        self.builder.store(raw, base, offset, None, None);
    }

    fn lower_constant(
        &mut self,
        constant: TraceConstant,
    ) -> Result<b3::ValueId, TracingJitError> {
        Ok(match constant {
            TraceConstant::Null => self.builder.const64(TAG_NULL as i64),
            TraceConstant::Bool(value) => self
                .builder
                .const64(if value { TAG_TRUE } else { TAG_FALSE } as i64),
            TraceConstant::NumBits(bits) => self.builder.const64(bits as i64),
            TraceConstant::Int32(value) => self.builder.const32(value),
            TraceConstant::F64Bits(bits) => {
                self.builder.const_double(f64::from_bits(bits))
            }
            TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                return Err(TracingJitError::B3Lowering {
                    trace: self.trace.id,
                    message: "runtime-owned constant reached native tier".to_owned(),
                })
            }
        })
    }

    fn normalize_to_word(&mut self, value: b3::ValueId, rep: TraceRep) -> b3::ValueId {
        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => value,
            TraceRep::Int32 | TraceRep::Bool => self.builder.zext32(value),
            TraceRep::F64 => self.builder.bitwise_cast(b3::Type::Int64, value),
        }
    }

    fn binary_compare(
        &mut self,
        op: CmpKind,
        lhs: b3::ValueId,
        rhs: b3::ValueId,
    ) -> b3::ValueId {
        let opcode = match op {
            CmpKind::Eq => b3::Opcode::Equal,
            CmpKind::Ne => b3::Opcode::NotEqual,
            CmpKind::Lt => b3::Opcode::LessThan,
            CmpKind::Le => b3::Opcode::LessEqual,
            CmpKind::Gt => b3::Opcode::GreaterThan,
            CmpKind::Ge => b3::Opcode::GreaterEqual,
        };
        self.builder.binary(opcode, lhs, rhs)
    }

    fn bool_not(&mut self, value: b3::ValueId) -> b3::ValueId {
        let zero = self.builder.const32(0);
        self.builder.binary(b3::Opcode::Equal, value, zero)
    }

    fn bool_and(&mut self, lhs: b3::ValueId, rhs: b3::ValueId) -> b3::ValueId {
        self.builder.binary(b3::Opcode::BitAnd, lhs, rhs)
    }

    fn bool_or(&mut self, lhs: b3::ValueId, rhs: b3::ValueId) -> b3::ValueId {
        self.builder.binary(b3::Opcode::BitOr, lhs, rhs)
    }

    fn select(
        &mut self,
        condition: b3::ValueId,
        if_true: b3::ValueId,
        if_false: b3::ValueId,
        rep: TraceRep,
    ) -> b3::ValueId {
        // Branchless select using a 0/-1 mask. This avoids relying on a helper
        // method for B3 Select and maps well to both x86-64 and AArch64.
        match rep {
            TraceRep::Int32 | TraceRep::Bool => {
                let zero = self.builder.const32(0);
                let mask = self.builder.binary(b3::Opcode::Sub, zero, condition);
                let not_mask = {
                    let all = self.builder.const32(-1);
                    self.builder.binary(b3::Opcode::BitXor, mask, all)
                };
                let a = self.builder.binary(b3::Opcode::BitAnd, if_true, mask);
                let b = self.builder.binary(b3::Opcode::BitAnd, if_false, not_mask);
                self.builder.binary(b3::Opcode::BitOr, a, b)
            }

            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => {
                let zero32 = self.builder.const32(0);
                let mask32 = self.builder.binary(b3::Opcode::Sub, zero32, condition);
                let mask = self.builder.sext32(mask32);
                let all = self.builder.const64(-1);
                let not_mask = self.builder.binary(b3::Opcode::BitXor, mask, all);
                let a = self.builder.binary(b3::Opcode::BitAnd, if_true, mask);
                let b = self.builder.binary(b3::Opcode::BitAnd, if_false, not_mask);
                self.builder.binary(b3::Opcode::BitOr, a, b)
            }

            TraceRep::F64 => {
                let a_bits = self.builder.bitwise_cast(b3::Type::Int64, if_true);
                let b_bits = self.builder.bitwise_cast(b3::Type::Int64, if_false);
                let selected = self.select(condition, a_bits, b_bits, TraceRep::Tagged);
                self.builder.bitwise_cast(b3::Type::Double, selected)
            }
        }
    }

    fn tagged_is_num(&mut self, bits: b3::ValueId) -> b3::ValueId {
        let mask = self.builder.const64(TAG_MASK as i64);
        let base = self.builder.const64(TAG_BASE as i64);
        let masked = self.builder.binary(b3::Opcode::BitAnd, bits, mask);
        let not_tag = self.builder.binary(b3::Opcode::NotEqual, masked, base);

        let heap_mask = self.builder.const64(TAG_HEAP_MASK as i64);
        let heap_tag = self.builder.const64(TAG_HEAP as i64);
        let heap_masked = self.builder.binary(b3::Opcode::BitAnd, bits, heap_mask);
        let not_heap = self
            .builder
            .binary(b3::Opcode::NotEqual, heap_masked, heap_tag);

        self.bool_and(not_tag, not_heap)
    }

    fn tagged_falsey(&mut self, bits: b3::ValueId) -> b3::ValueId {
        let null = self.builder.const64(TAG_NULL as i64);
        let false_ = self.builder.const64(TAG_FALSE as i64);
        let is_null = self.builder.binary(b3::Opcode::Equal, bits, null);
        let is_false = self.builder.binary(b3::Opcode::Equal, bits, false_);
        self.bool_or(is_null, is_false)
    }

    fn box_bool(&mut self, condition: b3::ValueId) -> b3::ValueId {
        // TAG_TRUE == TAG_FALSE + 1, and comparisons produce 0/1 Int32.
        let bit = self.builder.zext32(condition);
        let base = self.builder.const64(TAG_FALSE as i64);
        self.builder.binary(b3::Opcode::Add, base, bit)
    }

    fn box_f64(&mut self, value: b3::ValueId) -> b3::ValueId {
        // Wren canonicalizes NaNs before putting them in a tagged Value. Do the
        // same without control flow:
        //
        //   mask = -(value != value)
        //   result = (bits & ~mask) | (CANONICAL_NAN & mask)
        let is_nan = self
            .builder
            .binary(b3::Opcode::NotEqual, value, value);
        let zero = self.builder.const32(0);
        let mask32 = self.builder.binary(b3::Opcode::Sub, zero, is_nan);
        let mask = self.builder.sext32(mask32);
        let all = self.builder.const64(-1);
        let not_mask = self.builder.binary(b3::Opcode::BitXor, mask, all);

        let bits = self.builder.bitwise_cast(b3::Type::Int64, value);
        let canonical = self.builder.const64(CANONICAL_NAN as i64);

        let normal = self.builder.binary(b3::Opcode::BitAnd, bits, not_mask);
        let nan = self.builder.binary(b3::Opcode::BitAnd, canonical, mask);
        self.builder.binary(b3::Opcode::BitOr, normal, nan)
    }

    fn guard_numeric(
        &mut self,
        tagged: b3::ValueId,
        exit: SideExitId,
    ) -> Result<b3::ValueId, TracingJitError> {
        let ok = self.tagged_is_num(tagged);
        let failure = self.bool_not(ok);
        self.emit_side_exit(exit, failure)?;
        Ok(self.builder.bitwise_cast(b3::Type::Double, tagged))
    }

    fn emit_side_exit(
        &mut self,
        exit_id: SideExitId,
        failure_condition: b3::ValueId,
    ) -> Result<(), TracingJitError> {
        let plan = self
            .exit_plans
            .get(exit_id.0 as usize)
            .ok_or_else(|| TracingJitError::B3Lowering {
                trace: self.trace.id,
                message: format!("unknown side exit {exit_id:?}"),
            })?
            .clone();

        let stop = self.builder.procedure.add_block(0.01);
        let resume = self.builder.procedure.add_block(1.0);
        self.builder.branch(failure_condition, stop, (resume, Frequency::Normal));
        self.builder.block = stop;
        for (index, value) in plan.dynamic_values.iter().enumerate() {
            let rep = self.trace.value(*value).unwrap().rep;
            let value = self.lower_value(*value)?;
            let word = self.normalize_to_word(value, rep);
            self.builder.store(word, self.activation,
                offset_of!(TraceActivation, exit_values) as i32 + index as i32 * 8, None, None);
        }
        let id = self.builder.const32(exit_id.0 as i32);
        self.builder.store(id, self.activation, offset_of!(TraceActivation, exit_id) as i32, None, None);
        let result = self.builder.const32(TRACE_RESULT_SIDE_EXIT as i32);
        self.builder.return_(Some(result));
        self.builder.block = resume;

        Ok(())
    }

    fn lower_node(
        &mut self,
        node: &crate::wren_traceir::TraceNode,
    ) -> Result<(), TracingJitError> {
        use TraceInst::*;

        let trace_id = self.trace.id;
        let one_result = move |node: &crate::wren_traceir::TraceNode| -> Result<TraceValueId, TracingJitError> {
            if node.results.len() != 1 {
                Err(TracingJitError::B3Lowering {
                    trace: trace_id,
                    message: format!("{:?} expected one result", node.id),
                })
            } else {
                Ok(node.results[0])
            }
        };

        let set_result = |this: &mut Self,
                          result: TraceValueId,
                          value: b3::ValueId| {
            this.values.insert(result, value);
        };

        match &node.kind {
            Nop => {}

            Runtime { location, next, terminal } => {
                let callee = self.builder.load(b3::Type::Int64, self.activation,
                    offset_of!(TraceActivation, execute) as i32, None, None);
                let function = self.builder.const32(location.function.0 as i32);
                let pc = self.builder.const32(location.pc.0 as i32);
                let next = self.builder.const32(next.0 as i32);
                let status = self.builder.ccall(b3::Type::Int32, callee,
                    &[self.activation, function, pc, next], b3::Effects::for_call());
                // Return through the normal B3 epilogue, including callee saves.
                // Never materialize a pre-call snapshot into a changed frame.
                let stop = self.builder.procedure.add_block(0.01);
                let resume = self.builder.procedure.add_block(1.0);
                if *terminal { self.builder.jump(Some(stop)); }
                else { self.builder.branch(status, stop, (resume, Frequency::Normal)); }
                self.builder.block = stop;
                let result = self.builder.const32(TRACE_RESULT_RUNTIME as i32);
                self.builder.return_(Some(result));
                self.builder.block = resume;
            }

            Copy { value } => {
                let result = one_result(node)?;
                let value = self.lower_value(*value)?;
                set_result(self, result, value);
            }

            Reinterpret { value } => {
                let result = one_result(node)?;
                let source_rep = self.trace.value(*value).unwrap().rep;
                let target_rep = self.trace.value(result).unwrap().rep;
                let value = self.lower_value(*value)?;
                let word = self.normalize_to_word(value, source_rep);
                let value = match target_rep {
                    TraceRep::Int32 | TraceRep::Bool => self.builder.trunc(word),
                    TraceRep::F64 => self.builder.bitwise_cast(b3::Type::Double, word),
                    _ => word,
                };
                set_result(self, result, value);
            }

            LoadFrameSlot { slot, .. } => {
                let result = one_result(node)?;
                let rep = self.trace.value(result).unwrap().rep;
                let value = self.load_frame_slot(*slot, rep)?;
                set_result(self, result, value);
            }

            CommitFrameSlot {
                slot,
                value,
                ..
            } => {
                let rep = self.trace.value(*value).unwrap().rep;
                let lowered = self.lower_value(*value)?;
                self.store_frame_slot(*slot, lowered, rep);
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
                let null = self.builder.const64(TAG_NULL as i64);
                let failure = self.builder.binary(b3::Opcode::Equal, tagged, null);
                self.emit_side_exit(*exit, failure)?;

                if let Some(result) = node.results.first().copied() {
                    set_result(self, result, tagged);
                }
            }

            GuardF64 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_numeric(tagged, *exit)?;
                let result = one_result(node)?;
                set_result(self, result, raw);
            }

            GuardInt32 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let number = self.guard_numeric(tagged, *exit)?;

                // Reject NaN/out-of-range before DToI.
                let min = self.builder.const_double(i32::MIN as f64);
                let max = self.builder.const_double(i32::MAX as f64);
                let ge_min = self
                    .builder
                    .binary(b3::Opcode::GreaterEqual, number, min);
                let le_max = self
                    .builder
                    .binary(b3::Opcode::LessEqual, number, max);
                let in_range = self.bool_and(ge_min, le_max);
                let failure = self.bool_not(in_range);
                self.emit_side_exit(*exit, failure)?;

                let i64_value = self.double_to_int64(number);
                let back = self.builder.i2d(i64_value);
                let exact = self.builder.binary(b3::Opcode::Equal, number, back);
                let not_exact = self.bool_not(exact);
                self.emit_side_exit(*exit, not_exact)?;

                let i32_value = self.builder.trunc(i64_value);
                let result = one_result(node)?;
                set_result(self, result, i32_value);
            }

            GuardBounds {
                index,
                length,
                exit,
            } => {
                let index = self.lower_value(*index)?;
                let length = self.lower_value(*length)?;
                let zero = self.builder.const32(0);
                let negative = self
                    .builder
                    .binary(b3::Opcode::LessThan, index, zero);
                let too_large = self
                    .builder
                    .binary(b3::Opcode::GreaterEqual, index, length);
                let failure = self.bool_or(negative, too_large);
                self.emit_side_exit(*exit, failure)?;
            }

            GuardEq { lhs, rhs, exit } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let equal = self.builder.binary(b3::Opcode::Equal, lhs, rhs);
                let failure = self.bool_not(equal);
                self.emit_side_exit(*exit, failure)?;
            }

            GuardSwitch {
                tag,
                observed,
                exit,
            } => {
                let tag_value = self.lower_value(*tag)?;
                match observed {
                    TraceSwitchGuard::Case(case) => {
                        // First tier supports raw Int32 switch tags. Tagged source
                        // switches are rare in optimized loop bytecode; if this is
                        // Tagged the types won't match B3 and the verifier/lowerer
                        // will reject the trace rather than guessing.
                        let rep = self.trace.value(*tag).unwrap().rep;
                        let expected = match rep {
                            TraceRep::Int32 | TraceRep::Bool => {
                                self.builder.const32(*case as i32)
                            }
                            _ => {
                                return Err(TracingJitError::B3Lowering {
                                    trace: self.trace.id,
                                    message: "tagged Switch lowering is not enabled".to_owned(),
                                })
                            }
                        };
                        let equal = self
                            .builder
                            .binary(b3::Opcode::Equal, tag_value, expected);
                        let failure = self.bool_not(equal);
                        self.emit_side_exit(*exit, failure)?;
                    }

                    TraceSwitchGuard::Default { excluded } => {
                        let rep = self.trace.value(*tag).unwrap().rep;
                        if !matches!(rep, TraceRep::Int32 | TraceRep::Bool) {
                            return Err(TracingJitError::B3Lowering {
                                trace: self.trace.id,
                                message: "tagged Switch lowering is not enabled".to_owned(),
                            });
                        }

                        let mut matches_any = self.builder.const32(0);
                        for case in excluded {
                            let candidate = self.builder.const32(*case as i32);
                            let equal = self
                                .builder
                                .binary(b3::Opcode::Equal, tag_value, candidate);
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
                        let expected = self.builder.const64(TAG_NULL as i64);
                        let ok = self.builder.binary(b3::Opcode::Equal, value, expected);
                        let failure = self.bool_not(ok);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    TraceType::Bool => {
                        let false_ = self.builder.const64(TAG_FALSE as i64);
                        let true_ = self.builder.const64(TAG_TRUE as i64);
                        let a = self.builder.binary(b3::Opcode::Equal, value, false_);
                        let b = self.builder.binary(b3::Opcode::Equal, value, true_);
                        let ok = self.bool_or(a, b);
                        let failure = self.bool_not(ok);
                        self.emit_side_exit(*exit, failure)?;
                    }
                    _ => {
                        return Err(TracingJitError::B3Lowering {
                            trace: self.trace.id,
                            message: format!("unsupported native GuardType {expected:?}"),
                        })
                    }
                }

                if let Some(result) = node.results.first().copied() {
                    set_result(self, result, value);
                }
            }

            IntUnary { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    IntUnaryOp::Neg => {
                        let zero = self.builder.const32(0);
                        self.builder.binary(b3::Opcode::Sub, zero, value)
                    }
                    IntUnaryOp::BitNot => {
                        let all = self.builder.const32(-1);
                        self.builder.binary(b3::Opcode::BitXor, value, all)
                    }
                };
                set_result(self, one_result(node)?, result);
            }

            IntBinary { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let mut rhs = self.lower_value(*rhs)?;
                let opcode = match op {
                    IntBinOp::Add => b3::Opcode::Add,
                    IntBinOp::Sub => b3::Opcode::Sub,
                    IntBinOp::Mul => b3::Opcode::Mul,
                    IntBinOp::BitAnd => b3::Opcode::BitAnd,
                    IntBinOp::BitOr => b3::Opcode::BitOr,
                    IntBinOp::BitXor => b3::Opcode::BitXor,
                    IntBinOp::Shl => {
                        let mask = self.builder.const32(31);
                        rhs = self.builder.binary(b3::Opcode::BitAnd, rhs, mask);
                        b3::Opcode::Shl
                    }
                    IntBinOp::ShrU => {
                        let mask = self.builder.const32(31);
                        rhs = self.builder.binary(b3::Opcode::BitAnd, rhs, mask);
                        b3::Opcode::ZShr
                    }
                    IntBinOp::Sar => {
                        let mask = self.builder.const32(31);
                        rhs = self.builder.binary(b3::Opcode::BitAnd, rhs, mask);
                        b3::Opcode::SShr
                    }
                    IntBinOp::DivS | IntBinOp::RemS => {
                        return Err(TracingJitError::B3Lowering {
                            trace: self.trace.id,
                            message: "unchecked integer division/remainder disabled".to_owned(),
                        })
                    }
                };
                let value = self.builder.binary(opcode, lhs, rhs);
                set_result(self, one_result(node)?, value);
            }

            IntBinaryChecked {
                op,
                lhs,
                rhs,
                exit,
            } => {
                let lhs32 = self.lower_value(*lhs)?;
                let rhs32 = self.lower_value(*rhs)?;

                match op {
                    IntBinOp::Add | IntBinOp::Sub | IntBinOp::Mul => {
                        let lhs64 = self.builder.sext32(lhs32);
                        let rhs64 = self.builder.sext32(rhs32);
                        let opcode = match op {
                            IntBinOp::Add => b3::Opcode::Add,
                            IntBinOp::Sub => b3::Opcode::Sub,
                            IntBinOp::Mul => b3::Opcode::Mul,
                            _ => unreachable!(),
                        };
                        let wide = self.builder.binary(opcode, lhs64, rhs64);
                        let narrow = self.builder.trunc(wide);
                        let roundtrip = self.builder.sext32(narrow);
                        let overflow = self
                            .builder
                            .binary(b3::Opcode::NotEqual, wide, roundtrip);
                        self.emit_side_exit(*exit, overflow)?;
                        set_result(self, one_result(node)?, narrow);
                    }
                    _ => {
                        return Err(TracingJitError::B3Lowering {
                            trace: self.trace.id,
                            message: format!(
                                "checked {:?} is not implemented by first B3 tier",
                                op
                            ),
                        })
                    }
                }
            }

            F64Unary { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    F64UnaryOp::Neg => {
                        let minus_one = self.builder.const_double(-1.0);
                        self.builder.binary(b3::Opcode::Mul, value, minus_one)
                    }
                };
                set_result(self, one_result(node)?, result);
            }

            F64Binary { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let opcode = match op {
                    F64BinOp::Add => b3::Opcode::Add,
                    F64BinOp::Sub => b3::Opcode::Sub,
                    F64BinOp::Mul => b3::Opcode::Mul,
                    F64BinOp::Div => b3::Opcode::Div,
                    F64BinOp::Rem => {
                        return Err(TracingJitError::B3Lowering {
                            trace: self.trace.id,
                            message: "floating remainder helper is not enabled".to_owned(),
                        })
                    }
                };
                let result = self.builder.binary(opcode, lhs, rhs);
                set_result(self, one_result(node)?, result);
            }

            IntCompare { op, lhs, rhs } | F64Compare { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.binary_compare(*op, lhs, rhs);
                set_result(self, one_result(node)?, result);
            }

            RefEq { lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.builder.binary(b3::Opcode::Equal, lhs, rhs);
                set_result(self, one_result(node)?, result);
            }

            IsNull { value } => {
                let value = self.lower_value(*value)?;
                let null = self.builder.const64(TAG_NULL as i64);
                let result = self.builder.binary(b3::Opcode::Equal, value, null);
                set_result(self, one_result(node)?, result);
            }

            Select {
                condition,
                if_true,
                if_false,
            } => {
                let condition = self.lower_value(*condition)?;
                let t = self.lower_value(*if_true)?;
                let f = self.lower_value(*if_false)?;
                let result_id = one_result(node)?;
                let rep = self.trace.value(result_id).unwrap().rep;
                let result = self.select(condition, t, f, rep);
                set_result(self, result_id, result);
            }

            BoxInt32 { value } => {
                let value = self.lower_value(*value)?;
                let double = self.builder.i2d(value);
                let tagged = self.builder.bitwise_cast(b3::Type::Int64, double);
                set_result(self, one_result(node)?, tagged);
            }

            BoxF64 { value } => {
                let value = self.lower_value(*value)?;
                let tagged = self.box_f64(value);
                set_result(self, one_result(node)?, tagged);
            }

            Int32ToF64 { value } => {
                let value = self.lower_value(*value)?;
                let double = self.builder.i2d(value);
                set_result(self, one_result(node)?, double);
            }

            F64ToInt32 { value, exit } => {
                let number = self.lower_value(*value)?;
                let min = self.builder.const_double(i32::MIN as f64);
                let max = self.builder.const_double(i32::MAX as f64);
                let ge = self.builder.binary(b3::Opcode::GreaterEqual, number, min);
                let le = self.builder.binary(b3::Opcode::LessEqual, number, max);
                let in_range = self.bool_and(ge, le);
                let fail = self.bool_not(in_range);
                self.emit_side_exit(*exit, fail)?;

                let wide = self.double_to_int64(number);
                let back = self.builder.i2d(wide);
                let exact = self.builder.binary(b3::Opcode::Equal, number, back);
                let not_exact = self.bool_not(exact);
                self.emit_side_exit(*exit, not_exact)?;

                let result = self.builder.trunc(wide);
                set_result(self, one_result(node)?, result);
            }

            TaggedToHeapHandle { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let mask = self.builder.const64(TAG_HEAP_MASK as i64);
                let expected = self.builder.const64(TAG_HEAP as i64);
                let masked = self.builder.binary(b3::Opcode::BitAnd, tagged, mask);
                let is_heap = self
                    .builder
                    .binary(b3::Opcode::Equal, masked, expected);
                let fail = self.bool_not(is_heap);
                self.emit_side_exit(*exit, fail)?;

                let payload = self.builder.const64(PAYLOAD_MASK as i64);
                let handle = self
                    .builder
                    .binary(b3::Opcode::BitAnd, tagged, payload);
                set_result(self, one_result(node)?, handle);
            }

            ReadBarrier { value } => {
                let value = self.lower_value(*value)?;
                set_result(self, one_result(node)?, value);
            }
        }

        Ok(())
    }
}

fn b3_type(rep: TraceRep) -> b3::Type {
    match rep {
        TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => b3::Type::Int64,
        TraceRep::Int32 | TraceRep::Bool => b3::Type::Int32,
        TraceRep::F64 => b3::Type::Double,
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
}

pub struct TracingJit {
    options: TracingJitOptions,
    loops: HashMap<LoopKey, LoopState>,
    traces: HashMap<TraceId, CompiledTrace>,
    next_trace_id: u32,
    stats: TracingJitStats,
}

impl Default for TracingJit {
    fn default() -> Self {
        Self::new(TracingJitOptions::default())
    }
}

impl TracingJit {
    #[cfg(test)]
    pub(crate) fn execute_one_for_test<R: TraceRuntime>(&mut self, runtime: &mut R) -> Result<(), TracingJitError> {
        let location = runtime.trace_current_location();
        let program = runtime.trace_program();
        let function = program.function(location.function).unwrap();
        let op = decode_at(program, location)?;
        let recorder = TraceRecorder::new(&self.options, TraceId(0), location);
        let mut builder = crate::wren_traceir::TraceBuilder::new(TraceId(0), TraceAnchor { kind: TraceKind::Loop, location });
        let expected = recorder.translate_op(runtime, program, function, &op, &mut builder)
            .map_err(|reason| TracingJitError::Record { location, message: reason.to_string() })?;
        match expected {
            ExpectedStep::Exact(next) => {
                let exit = builder.add_side_exit(ExitReason::GuardValue, next.pc);
                builder.set_terminator(TraceTerminator::Exit(exit));
            },
            ExpectedStep::Runtime { .. } => {
                builder.prepare_memory_loop();
                builder.set_terminator(TraceTerminator::RuntimeReturn);
            },
        }
        let trace = builder.finish().map_err(TracingJitError::TraceVerification)?;
        let compiled = B3TraceCompiler::new(&trace, &self.options)?.compile()?;
        self.traces.insert(TraceId(0), compiled);
        self.enter_compiled(runtime, TraceId(0))?;
        Ok(())
    }

    pub fn new(options: TracingJitOptions) -> Self {
        Self {
            options,
            loops: HashMap::new(),
            traces: HashMap::new(),
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

    pub fn trace(&self, id: TraceId) -> Option<&Trace> {
        self.traces.get(&id).map(|compiled| &compiled.trace)
    }

    pub fn disassembly(&self, id: TraceId) -> Option<String> {
        self.traces
            .get(&id)
            .map(|compiled| compiled.compilation.disassembly())
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
        if state.hotness < self.options.hot_loop_threshold {
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
                let state = self.loops.get_mut(&key).unwrap();
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

        let compiled = match B3TraceCompiler::new(&trace, &self.options)
            .and_then(B3TraceCompiler::compile)
        {
            Ok(compiled) => compiled,
            Err(error) => {
                self.stats.last_abort = Some(error.to_string());
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
        self.traces.insert(trace_id, compiled);

        // Recording one iteration ended at the same root header.
        if runtime.trace_current_location() == location {
            self.enter_compiled(runtime, trace_id)
        } else {
            Ok(JitAction::Compiled { trace: trace_id })
        }
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
        trace_id: TraceId,
    ) -> Result<JitAction, TracingJitError> {
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

        let mut activation = TraceActivation::new(inputs);

        // Keep the Compilation object alive in `self.traces` for the entire call.
        let (entry, result) = {
            let compiled = self
                .traces
                .get(&trace_id)
                .ok_or_else(|| TracingJitError::B3Lowering {
                    trace: trace_id,
                    message: "compiled trace disappeared".to_owned(),
                })?;
            let entry = compiled.entry;
            self.stats.native_entries += 1;
            let result = unsafe { entry(&mut activation) };
            (entry, result)
        };
        let _ = entry;

        match result {
            TRACE_RESULT_SIDE_EXIT => {
                self.stats.side_exits += 1;
                self.apply_side_exit(runtime, trace_id, &activation)
            }

            TRACE_RESULT_RETURNED => Ok(JitAction::EnteredNative { trace: trace_id }),
            TRACE_RESULT_RUNTIME => Ok(JitAction::EnteredNative { trace: trace_id }),

            TRACE_RESULT_LINK => Ok(JitAction::EnteredNative { trace: trace_id }),

            other => Err(TracingJitError::Activation(format!(
                "native trace {:?} returned unknown result code {}",
                trace_id, other
            ))),
        }
    }

    fn apply_side_exit<R: TraceRuntime>(
        &mut self,
        runtime: &mut R,
        trace_id: TraceId,
        activation: &TraceActivation,
    ) -> Result<JitAction, TracingJitError> {
        let exit_index = activation.exit_id as usize;

        let (plan, target) = {
            let compiled = self
                .traces
                .get_mut(&trace_id)
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;

            let plan = compiled
                .exits
                .get(exit_index)
                .cloned()
                .ok_or_else(|| TracingJitError::InvalidSideExit {
                    trace: trace_id,
                    exit: activation.exit_id,
                })?;

            if let Some(hit) = compiled.exit_hits.get_mut(exit_index) {
                *hit = hit.saturating_add(1);
            }

            let target = plan.target;
            (plan, target)
        };

        for write in &plan.writes {
            let value = match write.source {
                ExitWriteSource::Constant(constant) => runtime
                    .trace_materialize_constant(constant)
                    .map_err(runtime_error)?,

                ExitWriteSource::Dynamic(index) => {
                    let raw = activation.exit_values[index as usize];
                    RegisterWord(raw)
                }
            };

            runtime
                .trace_write_slot(write.slot, value)
                .map_err(runtime_error)?;
        }

        match target {
            ExitTarget::Interpreter(location) => {
                runtime.trace_set_location(location).map_err(runtime_error)?;

                Ok(JitAction::SideExit {
                    trace: trace_id,
                    exit: plan.id,
                    target: location,
                })
            }

            ExitTarget::Trace(child) => {
                // Manager-level stitching: until direct machine-code patching is
                // installed, chain without returning to the interpreter dispatch
                // loop. The child trace still uses the same architectural frame.
                self.stats.stitched_exits += 1;
                self.enter_compiled(runtime, child)
            }

            ExitTarget::Return => Err(TracingJitError::B3Lowering {
                trace: trace_id,
                message: "side-exit Return target is not enabled".to_owned(),
            }),
        }
    }

    /// Link a side exit to an already compiled child trace.
    ///
    /// This performs dispatch-free manager-level stitching immediately. A future
    /// B3 late-link patch can turn the same metadata relation into a direct
    /// machine-code jump without changing Trace IR.
    pub fn stitch(
        &mut self,
        parent: TraceId,
        exit: SideExitId,
        child: TraceId,
    ) -> Result<(), TracingJitError> {
        if !self.traces.contains_key(&child) {
            return Err(TracingJitError::B3Lowering {
                trace: parent,
                message: format!("cannot stitch to unknown child {child:?}"),
            });
        }

        let parent_trace = self
            .traces
            .get_mut(&parent)
            .ok_or_else(|| TracingJitError::B3Lowering {
                trace: parent,
                message: "unknown parent trace".to_owned(),
            })?;

        let plan = parent_trace
            .exits
            .get_mut(exit.0 as usize)
            .ok_or_else(|| TracingJitError::InvalidSideExit {
                trace: parent,
                exit: exit.0,
            })?;
        plan.target = ExitTarget::Trace(child);

        if let Some(side_exit) = parent_trace.trace.exits.get_mut(exit.0 as usize) {
            side_exit.target = ExitTarget::Trace(child);
            side_exit.linked_trace = Some(child);
        }

        Ok(())
    }

    pub fn exit_hits(&self, trace: TraceId, exit: SideExitId) -> Option<u32> {
        self.traces
            .get(&trace)?
            .exit_hits
            .get(exit.0 as usize)
            .copied()
    }

    pub fn should_record_side_trace(
        &self,
        trace: TraceId,
        exit: SideExitId,
    ) -> bool {
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
    fn relative_branch_target_is_measured_from_instruction_end() {
        assert_eq!(relative_target(BytecodePc(100), 10).0, 110);
        assert_eq!(relative_target(BytecodePc(100), (-10i32) as u32).0, 90);
    }

    #[test]
    fn activation_exit_buffer_is_large_enough_for_vm_register_file() {
        assert!(MAX_TRACE_EXIT_VALUES > 256 + 16 + 2 + 12);
    }
}
