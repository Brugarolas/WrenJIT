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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

pub use crate::wren_core::BuiltinClass;
use crate::wren_core::{self as core, NativeMethod};

use crate::wren_bytecode::{
    decode_bailout_reason, function_kind_uses_this, instruction_word_len,
    BytecodeFunction, BytecodePc, BytecodeProgram, BytecodeVerifier, ClassGuardDescriptor,
    FrameLayout, IncomingSource, Opcode, ReferenceMap, Reg, RegisterKind,
    RegisterWord, SpecialReg, SwitchTable, TypeGuardDescriptor,
    ARGUMENT_REGISTER_COUNT, RETURN_REGISTER_COUNT, SPECIAL_REGISTER_COUNT,
};
use crate::wren_ir::{
    BailoutReason, ClassId, ClassSet, Constant, ForeignFnId, FuncId, FunctionKind,
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
    pub const fn raw_bits(self) -> u64 {
        self.0
    }

    /// Construct a Value from raw NaN-box bits.
    ///
    /// # Safety
    ///
    /// Heap-tagged values must reference a live heap handle owned by the same
    /// interpreter. Prefer values obtained from the VM/HostContext.
    #[inline(always)]
    pub const unsafe fn from_raw_bits_unchecked(bits: u64) -> Self {
        Self(bits)
    }

    #[inline(always)]
    pub const fn from_bool(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }

    #[inline(always)]
    pub fn from_num(value: f64) -> Self {
        let bits = if value.is_nan() {
            CANONICAL_NAN
        } else {
            value.to_bits()
        };
        Self(bits)
    }

    #[inline(always)]
    pub const fn from_heap(handle: u32) -> Self {
        Self(TAG_HEAP | handle as u64)
    }

    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == TAG_NULL
    }

    #[inline(always)]
    pub const fn is_bool(self) -> bool {
        self.0 == TAG_FALSE || self.0 == TAG_TRUE
    }

    #[inline(always)]
    pub const fn as_bool(self) -> Option<bool> {
        if self.0 == TAG_FALSE {
            Some(false)
        } else if self.0 == TAG_TRUE {
            Some(true)
        } else {
            None
        }
    }

    #[inline(always)]
    pub const fn is_heap(self) -> bool {
        (self.0 & TAG_HEAP_MASK) == TAG_HEAP
    }

    #[inline(always)]
    pub const fn heap_handle(self) -> Option<u32> {
        if self.is_heap() {
            Some((self.0 & PAYLOAD_MASK) as u32)
        } else {
            None
        }
    }

    #[inline(always)]
    pub const fn is_num(self) -> bool {
        (self.0 & TAG_MASK) != TAG_BASE && !self.is_heap()
    }

    #[inline(always)]
    pub fn as_num(self) -> Option<f64> {
        if self.is_num() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    #[inline(always)]
    pub const fn is_falsey(self) -> bool {
        self.0 == TAG_FALSE || self.0 == TAG_NULL
    }

    #[inline(always)]
    pub const fn word(self) -> RegisterWord {
        RegisterWord(self.0)
    }

    #[inline(always)]
    pub const fn from_word(word: RegisterWord) -> Self {
        Self(word.0)
    }
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

#[derive(Debug)]
enum HeapObject {
    String(Arc<[u8]>),
    List(Vec<Value>),
    Map(WrenMap),
    Range {
        from: f64,
        to: f64,
        inclusive: bool,
    },
    Cell(RegisterWord),
    Closure {
        function_index: u32,
        captures: Box<[RegisterWord]>,
    },
    ClassObject {
        class_slot: u32,
    },
    Instance {
        class_slot: u32,
        fields: Box<[Value]>,
    },
    Foreign {
        class_slot: u32,
        bytes: Vec<u8>,
    },
    FiberObject {
        fiber: u32,
    },
    ModuleObject {
        module_slot: u32,
    },
}

#[derive(Debug)]
struct HeapSlot {
    marked: bool,
    receiver_class: Option<ReceiverClass>,
    object: HeapObject,
}

#[derive(Debug)]
struct Heap {
    slots: Vec<Option<HeapSlot>>,
    free: Vec<u32>,
    live: usize,
    allocations_since_gc: usize,
    next_gc: usize,
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            allocations_since_gc: 0,
            next_gc: 4096,
        }
    }
}

impl Heap {
    #[inline]
    fn alloc(&mut self, object: HeapObject) -> Value {
        let handle = if let Some(handle) = self.free.pop() {
            let receiver_class = heap_object_receiver_class(&object);
            self.slots[handle as usize] = Some(HeapSlot {
                marked: false,
                receiver_class,
                object,
            });
            handle
        } else {
            let handle = self.slots.len() as u32;
            let receiver_class = heap_object_receiver_class(&object);
            self.slots.push(Some(HeapSlot {
                marked: false,
                receiver_class,
                object,
            }));
            handle
        };
        self.live += 1;
        self.allocations_since_gc += 1;
        Value::from_heap(handle)
    }

    #[inline(always)]
    fn get(&self, value: Value) -> Option<&HeapObject> {
        let handle = value.heap_handle()? as usize;
        self.slots.get(handle)?.as_ref().map(|slot| &slot.object)
    }

    #[inline(always)]
    fn get_mut(&mut self, value: Value) -> Option<&mut HeapObject> {
        let handle = value.heap_handle()? as usize;
        self.slots
            .get_mut(handle)?
            .as_mut()
            .map(|slot| &mut slot.object)
    }

    #[inline(always)]
    fn receiver_class(&self, value: Value) -> Option<ReceiverClass> {
        let handle = value.heap_handle()? as usize;
        unsafe {
            self.slots
                .get_unchecked(handle)
                .as_ref()
                .and_then(|slot| slot.receiver_class)
        }
    }

    #[inline(always)]
    fn should_collect(&self) -> bool {
        self.allocations_since_gc >= self.next_gc
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
}

pub trait Host {
    /// Output from System.print/write. Embedders can capture it without
    /// replacing the core library or allocating their own primitive IDs.
    fn write(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(bytes).and_then(|()| stdout.flush())
            .map_err(|error| RuntimeError::new(format!("output failed: {error}")))
    }

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
    call_args: Vec<Reg>,
    switches: Vec<PreparedSwitch>,
    gc_maps: Vec<ReferenceMap>,
    word_to_op: Vec<u32>,
    deopt_resume: HashMap<u32, u32>,
    frame: FrameLayout,
    id: FuncId,
    module_slot: u32,
    kind: FunctionKind,
    return_kind: RegisterKind,
    capture_count: u16,
}

struct PreparedProgram {
    functions: Vec<PreparedFunction>,
    function_by_id: HashMap<FuncId, u32>,
    class_by_id: HashMap<ClassId, u32>,
    module_by_id: HashMap<ModuleId, u32>,
    module_by_name: HashMap<StringId, u32>,
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
        let function_by_id: HashMap<_, _> = program
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.id, index as u32))
            .collect();
        let class_by_id: HashMap<_, _> = program
            .metadata
            .classes
            .iter()
            .enumerate()
            .map(|(index, class)| (class.id, index as u32))
            .collect();
        let module_by_id: HashMap<_, _> = program
            .metadata
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| (module.id, index as u32))
            .collect();
        let module_by_name: HashMap<_, _> = program
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
        function_by_id: &HashMap<FuncId, u32>,
        class_by_id: &HashMap<ClassId, u32>,
        module_by_id: &HashMap<ModuleId, u32>,
    ) -> Result<PreparedFunction, RuntimeError> {
        let code = &function.chunk.code;
        let mut word_to_op = vec![NO_INDEX; code.len() + 1];
        let mut temps = Vec::<TempOp>::new();
        let mut call_args = Vec::<Reg>::new();

        let mut safepoint_by_pc = HashMap::<u32, u32>::new();
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
            let args_offset = call_args.len() as u32;
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
                for index in 0..count {
                    let packed = code[raw_pc + 2 + index / 4].0;
                    call_args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
                }
            } else if opcode == Opcode::MakeClosure {
                let count = first.bx() as usize;
                for index in 0..count {
                    let packed = code[raw_pc + 2 + index / 4].0;
                    call_args.push(Reg(((packed >> ((index % 4) * 8)) & 0xff) as u8));
                }
            }

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

        let mut deopt_resume = HashMap::new();
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

            handlers.push(handler_for(temp.opcode));
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

        install_superinstructions(&temps, &mut handlers, &mut ops, &target_ops);

        let module_slot = *module_by_id.get(&function.module).ok_or_else(|| {
            RuntimeError::new(format!("unknown module {:?} for {:?}", function.module, function.id))
        })?;

        Ok(PreparedFunction {
            handlers,
            ops,
            call_args,
            switches,
            gc_maps,
            word_to_op,
            deopt_resume,
            frame: function.frame.clone(),
            id: function.id,
            module_slot,
            kind: function.kind,
            return_kind: function.frame.return_kind,
            capture_count: function.capture_count,
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
                _ => None,
            };
            if let Some(handler) = fused {
                handlers[index] = handler;
                handlers[index + 1] = h_unreachable_fused;
                continue;
            }
        }

        // The current collector is non-generational mark/sweep. Explicit write
        // barriers are semantically empty, so merge them into the preceding
        // mutation. Keep this list explicit so switching collectors cannot
        // silently invalidate the optimization.
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
    Native(NativeMethod),
    Primitive(PrimitiveId),
    Foreign(ForeignFnId),
}

const BUILTIN_CLASS_COUNT: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReceiverClass(u64);

const CLASS_KIND_INSTANCE: u64 = 1 << 56;
const CLASS_KIND_STATIC: u64 = 2 << 56;
const CLASS_KIND_BUILTIN: u64 = 3 << 56;
const CLASS_KIND_FOREIGN: u64 = 4 << 56;

#[inline(always)]
fn heap_object_receiver_class(object: &HeapObject) -> Option<ReceiverClass> {
    Some(match object {
        HeapObject::String(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::String as u64),
        HeapObject::List(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::List as u64),
        HeapObject::Map(_) =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Map as u64),
        HeapObject::Range { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Range as u64),
        HeapObject::Closure { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Closure as u64),
        HeapObject::FiberObject { .. } =>
            ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Fiber as u64),
        HeapObject::ClassObject { class_slot } =>
            ReceiverClass(CLASS_KIND_STATIC | *class_slot as u64),
        HeapObject::Instance { class_slot, .. } =>
            ReceiverClass(CLASS_KIND_INSTANCE | *class_slot as u64),
        HeapObject::Foreign { class_slot, .. } =>
            ReceiverClass(CLASS_KIND_FOREIGN | *class_slot as u64),
        HeapObject::Cell(_) | HeapObject::ModuleObject { .. } => return None,
    })
}

#[derive(Clone, Copy)]
struct IcEntry {
    class: ReceiverClass,
    target: CallTarget,
}

struct InlineCacheState {
    entries: [Option<IcEntry>; 4],
    len: u8,
    megamorphic: bool,
}

impl Default for InlineCacheState {
    fn default() -> Self { Self { entries: [None; 4], len: 0, megamorphic: false } }
}

impl InlineCacheState {
    #[inline(always)]
    fn probe(&self, class: ReceiverClass) -> Option<CallTarget> {
        let mut index = 0usize;
        while index < self.len as usize {
            let entry = unsafe { self.entries.get_unchecked(index).unwrap_unchecked() };
            if entry.class == class { return Some(entry.target); }
            index += 1;
        }
        None
    }

    fn insert(&mut self, class: ReceiverClass, target: CallTarget) {
        if self.probe(class).is_some() { return; }
        if (self.len as usize) < self.entries.len() {
            self.entries[self.len as usize] = Some(IcEntry { class, target });
            self.len += 1;
        } else {
            self.megamorphic = true;
        }
    }
}

struct RuntimeClass {
    id: ClassId,
    name: StringId,
    superclass: Option<u32>,
    instance_field_count: u16,
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
    exports: HashMap<StringId, u32>,
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
    Running,
    Suspended,
    Done,
    Failed,
}

#[derive(Clone, Copy)]
struct FiberLink {
    caller: u32,
    dst: Reg,
    catch_error: bool,
}

struct Frame {
    function: u32,
    pc: u32,
    reg_base: usize,
    reg_count: u16,
    arguments: [RegisterWord; ARGUMENT_REGISTER_COUNT],
    returns: [RegisterWord; RETURN_REGISTER_COUNT],
    special: [RegisterWord; SPECIAL_REGISTER_COUNT],
    return_dst: Option<Reg>,
    root_map: u32,
}

impl Frame {
    fn new(function: u32, reg_base: usize, reg_count: u16, return_dst: Option<Reg>) -> Self {
        Self {
            function,
            pc: 0,
            reg_base,
            reg_count,
            arguments: [RegisterWord(0); ARGUMENT_REGISTER_COUNT],
            returns: [RegisterWord(0); RETURN_REGISTER_COUNT],
            special: [RegisterWord(0); SPECIAL_REGISTER_COUNT],
            return_dst,
            root_map: NO_INDEX,
        }
    }
}

struct FiberState {
    object: Value,
    status: FiberStatus,
    frames: Vec<Frame>,
    /// Backing storage is retained at its high-water mark. `register_top`
    /// tracks the live prefix so frame reuse does not zero GPRs on every call.
    registers: Vec<RegisterWord>,
    register_top: usize,
    starter: Option<Value>,
    link: Option<FiberLink>,
    suspend_dst: Option<Reg>,
    last_value: Value,
    error: Value,
}

impl FiberState {
    fn new(object: Value, status: FiberStatus) -> Self {
        Self {
            object,
            status,
            frames: Vec::new(),
            registers: Vec::with_capacity(1024),
            register_top: 0,
            starter: None,
            link: None,
            suspend_dst: None,
            last_value: Value::NULL,
            error: Value::NULL,
        }
    }
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
    program: &'p BytecodeProgram,
    prepared: PreparedProgram,
    host: Box<dyn Host>,
    heap: Heap,
    classes: Vec<RuntimeClass>,
    modules: Vec<RuntimeModule>,
    strings: Vec<Value>,
    /// Constants pre-resolved to physical register words once at VM startup.
    /// Hot LOAD_CONST never reinterprets Constant or performs metadata lookup.
    resolved_constants: Vec<Vec<RegisterWord>>,
    fibers: Vec<Option<FiberState>>,
    free_fibers: Vec<u32>,
    current_fiber: u32,
    run_state: RunState,
    result: Value,
    error: Option<RuntimeError>,
    inline_caches: Vec<Vec<InlineCacheState>>,
    builtin_methods: Vec<Vec<Option<CallTarget>>>,
    core_foreign_methods: HashMap<ForeignFnId, NativeMethod>,
    started_at: Instant,
    megamorphic_cache: HashMap<(ReceiverClass, MethodSymbol, Option<u32>), CallTarget>,
    profile_counters: Vec<Vec<u64>>,
    roots: Vec<Option<Value>>,
    free_roots: Vec<u32>,
    gc_enabled: bool,

    // Hot execution cache. These pointers are refreshed only when frames or
    // fibers change. They turn every GPR access into one pointer add/load.
    hot_frame: *mut Frame,
    hot_registers: *mut RegisterWord,
    hot_function: *const PreparedFunction,
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
        let prepared = PreparedProgram::prepare(program)?;
        let mut heap = Heap::default();

        let mut strings = Vec::with_capacity(program.metadata.strings.len());
        for string in &program.metadata.strings {
            strings.push(heap.alloc(HeapObject::String(Arc::<[u8]>::from(string.as_bytes()))));
        }

        let mut modules = Vec::with_capacity(program.metadata.modules.len());
        for (slot, info) in program.metadata.modules.iter().enumerate() {
            let module_value = heap.alloc(HeapObject::ModuleObject { module_slot: slot as u32 });
            modules.push(RuntimeModule {
                id: info.id,
                name: info.name,
                variables: vec![Value::NULL; info.variable_count as usize],
                exports: HashMap::new(),
                module_value,
                body: None,
                initialized: false,
                initializing: false,
            });
        }

        let mut classes = Vec::with_capacity(program.metadata.classes.len());
        for (slot, info) in program.metadata.classes.iter().enumerate() {
            let class_value = heap.alloc(HeapObject::ClassObject { class_slot: slot as u32 });
            classes.push(RuntimeClass {
                id: info.id,
                name: info.name,
                superclass: info.superclass.and_then(|id| prepared.class_by_id.get(&id).copied()),
                instance_field_count: info.instance_field_count,
                is_foreign: info.is_foreign,
                instance_methods: vec![None; program.metadata.methods.len()],
                static_methods: vec![None; program.metadata.methods.len()],
                static_fields: vec![Value::NULL; info.static_field_count as usize],
                class_value,
            });
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

        // Main fiber is a permanent VM root.
        let main_object = heap.alloc(HeapObject::FiberObject { fiber: 0 });
        let mut main = FiberState::new(main_object, FiberStatus::Running);
        main.registers.reserve(4096);

        let mut vm = Self {
            program,
            prepared,
            host,
            heap,
            classes,
            modules,
            strings,
            resolved_constants,
            fibers: vec![Some(main)],
            free_fibers: Vec::new(),
            current_fiber: 0,
            run_state: RunState::Running,
            result: Value::NULL,
            error: None,
            inline_caches,
            builtin_methods: vec![vec![None; program.metadata.methods.len()]; BUILTIN_CLASS_COUNT],
            core_foreign_methods: HashMap::new(),
            started_at: Instant::now(),
            megamorphic_cache: HashMap::new(),
            profile_counters,
            roots: Vec::new(),
            free_roots: Vec::new(),
            gc_enabled: true,
            hot_frame: std::ptr::null_mut(),
            hot_registers: std::ptr::null_mut(),
            hot_function: std::ptr::null(),
        };
        vm.install_core()?;
        Ok(vm)
    }

    fn install_core(&mut self) -> Result<(), RuntimeError> {
        use BuiltinClass::*;
        for class in [Null, Bool, Num, String, List, Map, Range, Closure, Fiber] {
            for (index, method) in self.program.metadata.methods.iter().enumerate() {
                let name = self.program.metadata.string(method.name).unwrap_or("");
                if let Some(native) = core::instance_method(class, name, method.kind, method.arity) {
                    self.builtin_methods[class as usize][index] = Some(CallTarget::Native(native));
                }
            }
        }
        for foreign in &self.program.metadata.foreign_functions {
            if self.program.metadata.string(foreign.module) != Some(core::MODULE_NAME) {
                continue;
            }
            let signature = self.program.metadata.method(foreign.signature)
                .ok_or_else(|| RuntimeError::new("missing core method signature"))?;
            let class = self.program.metadata.string(foreign.class).unwrap_or("");
            let name = self.program.metadata.string(signature.name).unwrap_or("");
            let native = core::foreign_method(class, name, signature.kind, signature.arity)
                .filter(|_| foreign.is_static)
                .ok_or_else(|| RuntimeError::new(format!("unknown core method {class}.{name}")))?;
            self.core_foreign_methods.insert(foreign.id, native);
        }
        let entry = self.program.functions.iter().find(|function| {
            function.kind == FunctionKind::ModuleBody
                && self.program.metadata.module(function.module)
                    .and_then(|module| self.program.metadata.string(module.name)) == Some(core::MODULE_NAME)
        }).map(|function| function.id);
        if let Some(entry) = entry {
            match self.run_module(entry)? {
                ExecutionResult::Returned(_) => {}
                ExecutionResult::Suspended(_) => return Err(RuntimeError::new("core initialization suspended")),
            }
        }
        Ok(())
    }

    fn invalidate_method_caches(&mut self) {
        for caches in &mut self.inline_caches {
            caches.fill_with(InlineCacheState::default);
        }
        self.megamorphic_cache.clear();
    }

    pub fn set_gc_enabled(&mut self, enabled: bool) {
        self.gc_enabled = enabled;
    }

    pub fn bind_builtin_primitive(&mut self, class: BuiltinClass, method: MethodSymbol, primitive: PrimitiveId) {
        if let Some(slot) = self
            .builtin_methods
            .get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Primitive(primitive));
            self.invalidate_method_caches();
        }
    }

    pub fn bind_builtin_foreign(&mut self, class: BuiltinClass, method: MethodSymbol, foreign: ForeignFnId) {
        if let Some(slot) = self
            .builtin_methods
            .get_mut(class as usize)
            .and_then(|table| table.get_mut(method.0 as usize))
        {
            *slot = Some(CallTarget::Foreign(foreign));
            self.invalidate_method_caches();
        }
    }

    pub fn bind_module_variable_name(&mut self, module: ModuleId, name: StringId, slot: u32) -> Result<(), RuntimeError> {
        let module_slot = *self.prepared.module_by_id.get(&module).ok_or_else(|| RuntimeError::new("unknown module"))?;
        if slot as usize >= self.modules[module_slot as usize].variables.len() { return Err(RuntimeError::new("module slot out of range")); }
        self.modules[module_slot as usize].exports.insert(name, slot);
        Ok(())
    }

    pub fn add_root(&mut self, value: Value) -> u32 {
        if let Some(index) = self.free_roots.pop() {
            self.roots[index as usize] = Some(value);
            index
        } else {
            let index = self.roots.len() as u32;
            self.roots.push(Some(value));
            index
        }
    }

    pub fn set_root(&mut self, root: u32, value: Value) -> Result<(), RuntimeError> {
        let slot = self.roots.get_mut(root as usize).ok_or_else(|| RuntimeError::new("invalid root handle"))?;
        if slot.is_none() { return Err(RuntimeError::new("root handle has been released")); }
        *slot = Some(value);
        Ok(())
    }

    pub fn release_root(&mut self, root: u32) {
        if let Some(slot) = self.roots.get_mut(root as usize) {
            if slot.take().is_some() { self.free_roots.push(root); }
        }
    }

    pub fn run_function(&mut self, function: FuncId, args: &[Value]) -> Result<ExecutionResult, RuntimeError> {
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
        self.finish_result()
    }

    pub fn run_module(&mut self, function: FuncId) -> Result<ExecutionResult, RuntimeError> {
        self.run_function(function, &[])
    }

    /// Resume a VM stopped by Fiber.suspend() or by yielding without a caller.
    ///
    /// The value becomes the result of the suspended Fiber operation.
    pub fn resume(&mut self, value: Value) -> Result<ExecutionResult, RuntimeError> {
        if self.run_state != RunState::Suspended {
            return Err(RuntimeError::new("resume requires a suspended VM"));
        }
        let current = self.current_fiber;
        let dst = self.fiber_mut(current).suspend_dst.take();
        if let Some(dst) = dst {
            let fiber = self.fiber_mut(current);
            let frame = fiber.frames.last().ok_or_else(|| RuntimeError::new("suspended fiber has no frame"))?;
            let index = frame.reg_base + dst.index();
            unsafe {
                *fiber.registers.get_unchecked_mut(index) = value.word();
            }
        }
        if let Some(frame) = self.fiber_mut(current).frames.last_mut() {
            frame.special[SpecialReg::ResumeValue.index()] = value.word();
        }
        self.fiber_mut(current).status = FiberStatus::Running;
        self.refresh_hot_frame();
        self.run_state = RunState::Running;
        self.result = Value::NULL;
        self.error = None;
        self.dispatch_loop();
        self.finish_result()
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
            let frame = unsafe { &*self.hot_frame };
            let pc = frame.pc as usize;

            let prepared = unsafe { &*self.hot_function };
            if pc >= prepared.handlers.len() {
                self.fail(RuntimeError::new("program counter is outside function"));
                break;
            }
            let handler = unsafe { *prepared.handlers.get_unchecked(pc) };
            let op = unsafe { prepared.ops.as_ptr().add(pc) };
            unsafe { handler(self, op) };
        }
    }

    #[inline(always)]
    fn fiber(&self, id: u32) -> &FiberState {
        unsafe { self.fibers.get_unchecked(id as usize).as_ref().unwrap_unchecked() }
    }

    #[inline(always)]
    fn fiber_mut(&mut self, id: u32) -> &mut FiberState {
        unsafe { self.fibers.get_unchecked_mut(id as usize).as_mut().unwrap_unchecked() }
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
        let Some(fiber) = self.fibers.get_mut(current).and_then(Option::as_mut) else {
            self.hot_frame = std::ptr::null_mut();
            self.hot_registers = std::ptr::null_mut();
            self.hot_function = std::ptr::null();
            return;
        };
        let Some(frame) = fiber.frames.last_mut() else {
            self.hot_frame = std::ptr::null_mut();
            self.hot_registers = std::ptr::null_mut();
            self.hot_function = std::ptr::null();
            return;
        };
        self.hot_frame = frame as *mut Frame;
        self.hot_registers = unsafe {
            fiber.registers.as_mut_ptr().add(frame.reg_base)
        };
        self.hot_function = unsafe {
            self.prepared.functions.as_ptr().add(frame.function as usize)
        };
    }

    #[inline(always)]
    fn clear_hot_frame(&mut self) {
        self.hot_frame = std::ptr::null_mut();
        self.hot_registers = std::ptr::null_mut();
        self.hot_function = std::ptr::null();
    }

    #[inline(always)]
    fn current_prepared(&self) -> &PreparedFunction {
        debug_assert!(!self.hot_function.is_null());
        unsafe { &*self.hot_function }
    }

    #[inline(always)]
    unsafe fn advance(&mut self, op: &PreparedOp) {
        self.current_frame_mut().pc = op.next;
    }

    #[inline(always)]
    unsafe fn set_pc(&mut self, pc: u32) {
        self.current_frame_mut().pc = pc;
    }

    #[inline(always)]
    fn current_location(&self, op: &PreparedOp) -> (FuncId, BytecodePc) {
        (self.current_prepared().id, BytecodePc(op.raw_pc))
    }

    #[inline(always)]
    fn mark_gc_point(&mut self, op: &PreparedOp) {
        if op.gc_map != NO_INDEX {
            self.current_frame_mut().root_map = op.gc_map;
        }
    }

    fn fail(&mut self, error: RuntimeError) {
        self.error = Some(error);
        self.run_state = RunState::Failed;
    }

    fn fail_at(&mut self, op: &PreparedOp, message: impl Into<String>) {
        let (function, pc) = self.current_location(op);
        self.fail(RuntimeError::new(message).at(function, pc));
    }

    #[inline(always)]
    unsafe fn gather_call_words(&self, op: &PreparedOp, out: &mut [RegisterWord; 17]) -> usize {
        let function = &*self.hot_function;
        let count = op.c as usize;
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
        let arity = self.program.functions[function_index as usize].arity as usize;
        if args.len() < arity {
            return Err(RuntimeError::new(format!(
                "function requires {arity} argument(s), got {}", args.len()
            )));
        }
        // `prepared` is immutable for the lifetime of the interpreter. Keeping a
        // raw pointer here lets us mutate disjoint VM state below without cloning
        // FrameLayout (which contains Vecs and would allocate on every call).
        let prepared_ptr = unsafe {
            self.prepared.functions.as_ptr().add(function_index as usize)
        };
        let (layout, module_slot) = unsafe {
            (&(*prepared_ptr).frame, (*prepared_ptr).module_slot)
        };
        let reg_count = layout.general_register_count as usize;
        let module_value = self.modules[module_slot as usize].module_value;
        let fiber_value = self.fiber(self.current_fiber).object;

        // Closure captures live in a stable heap slot. The heap is not mutated
        // during frame construction, so retaining this slice is safe here.
        let captures: &[RegisterWord] = if let Some(closure_value) = closure {
            match self.heap.get(closure_value) {
                Some(HeapObject::Closure { function_index: actual, captures }) if *actual == function_index => captures.as_ref(),
                Some(HeapObject::Closure { .. }) => return Err(RuntimeError::new("closure/function mismatch")),
                _ => return Err(RuntimeError::new("CallClosure receiver is not a closure")),
            }
        } else {
            &[]
        };

        let current = self.current_fiber;
        let fiber = self.fibers[current as usize].as_mut().expect("live fiber");
        let base = fiber.register_top;
        let needed = base + reg_count;
        if fiber.registers.len() < needed {
            fiber.registers.resize(needed, RegisterWord(0));
        }
        fiber.register_top = needed;
        let mut frame = Frame::new(function_index, base, reg_count as u16, return_dst);
        if let Some(receiver) = receiver { frame.special[SpecialReg::This.index()] = receiver; }
        if let Some(closure) = closure { frame.special[SpecialReg::Closure.index()] = closure.word(); }
        frame.special[SpecialReg::Module.index()] = module_value.word();
        frame.special[SpecialReg::Fiber.index()] = fiber_value.word();

        // ARG bank initialization is a tiny bounded copy (maximum 16).
        for (index, value) in args.iter().copied().enumerate() {
            frame.arguments[index] = value;
        }

        // Entry bindings write directly into their allocated GPRs. No bytecode
        // prologue and no temporary allocation is needed.
        for binding in &layout.parameters {
            let value = match binding.source {
                IncomingSource::This => frame.special[SpecialReg::This.index()],
                IncomingSource::Argument(arg) => frame.arguments[arg.index()],
            };
            unsafe {
                *fiber.registers.get_unchecked_mut(base + binding.destination.index()) = value;
            }
        }
        for binding in &layout.captures {
            let value = captures
                .get(binding.capture_index as usize)
                .copied()
                .ok_or_else(|| RuntimeError::new("closure capture binding is out of range"))?;
            unsafe {
                *fiber.registers.get_unchecked_mut(base + binding.destination.index()) = value;
            }
        }
        fiber.frames.push(frame);
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

        let has_caller = {
            let fiber = self.fiber_mut(current);
            fiber.register_top = base;
            if let Some(caller) = fiber.frames.last_mut() {
                caller.returns[0] = value0;
                if let Some(value1) = value1 { caller.returns[1] = value1; }
                let target = return_dst.map(|dst| caller.reg_base + dst.index());
                if let Some(index) = target { fiber.registers[index] = value0; }
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

    fn finish_fiber(&mut self, value: Value) {
        let current = self.current_fiber;
        let link = {
            let fiber = self.fiber_mut(current);
            fiber.status = FiberStatus::Done;
            fiber.last_value = value;
            fiber.link.take()
        };
        if let Some(link) = link {
            self.current_fiber = link.caller;
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
    fn receiver_class(&self, value: Value) -> Result<ReceiverClass, RuntimeError> {
        if value.is_null() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Null as u64));
        }
        if value.is_bool() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Bool as u64));
        }
        if value.is_num() {
            return Ok(ReceiverClass(CLASS_KIND_BUILTIN | BuiltinClass::Num as u64));
        }
        self.heap
            .receiver_class(value)
            .ok_or_else(|| RuntimeError::new("value has no Wren class"))
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

        let is_static = kind == (CLASS_KIND_STATIC >> 56);
        let mut slot = if let Some(home) = home_class {
            self.classes.get(home as usize)?.superclass?
        } else {
            payload
        };
        loop {
            let class = self.classes.get(slot as usize)?;
            let table = if is_static { &class.static_methods } else { &class.instance_methods };
            if let Some(Some(target)) = table.get(method.0 as usize) {
                return Some(*target);
            }
            slot = match class.superclass { Some(next) => next, None => break };
        }
        let signature = self.program.metadata.method(method)?;
        let name = self.program.metadata.string(signature.name)?;
        core::object_method(name, signature.kind, signature.arity).map(CallTarget::Native)
    }

    #[inline(always)]
    unsafe fn probe_invoke_cache(
        &self,
        function_index: u32,
        ic_slot: u32,
        class: ReceiverClass,
    ) -> Option<CallTarget> {
        self.inline_caches
            .get_unchecked(function_index as usize)
            .get_unchecked(ic_slot as usize)
            .probe(class)
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
            let signature = self.program.metadata.method(descriptor.method);
            let name = signature.and_then(|signature| self.program.metadata.string(signature.name)).unwrap_or("<unknown>");
            let arity = signature.map_or(0, |signature| signature.arity);
            RuntimeError::new(format!("method '{name}' with {arity} argument(s) is not defined for receiver"))
        })?;
        let cache = &mut self.inline_caches[function_index as usize][ic_slot as usize];
        cache.insert(class, target);
        if cache.megamorphic {
            self.megamorphic_cache.insert(mega_key, target);
        }
        Ok(target)
    }

    fn invoke_target(
        &mut self,
        op: &PreparedOp,
        target: CallTarget,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        match target {
            CallTarget::Native(NativeMethod::ClosureCall) => {
                let closure = Value::from_word(receiver);
                let function = match self.heap.get(closure) {
                    Some(HeapObject::Closure { function_index, .. }) => *function_index,
                    _ => { self.fail_at(op, "call receiver must be a function"); return; }
                };
                self.mark_gc_point(op);
                self.current_frame_mut().pc = op.next;
                if let Err(error) = self.push_frame(function, None, args, Some(closure), Some(Reg(op.a))) {
                    self.fail_at(op, error.message);
                }
            }
            CallTarget::Native(native) => {
                self.mark_gc_point(op);
                self.stage_invoke_args(receiver, args);
                let mut values = [Value::NULL; 17];
                values[0] = Value::from_word(receiver);
                for (index, word) in args.iter().copied().enumerate() { values[index + 1] = Value::from_word(word); }
                self.maybe_collect();
                match self.call_native(native, &values[..args.len() + 1]) {
                    Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                    Err(error) => self.fail_at(op, error.message),
                }
            }
            CallTarget::Bytecode(function) => {
                self.mark_gc_point(op);
                self.current_frame_mut().pc = op.next;
                if let Err(error) = self.push_frame(function, Some(receiver), args, None, Some(Reg(op.a))) {
                    self.fail_at(op, error.message);
                }
            }
            CallTarget::Primitive(primitive) => {
                self.mark_gc_point(op);
                self.stage_invoke_args(receiver, args);
                let mut values = [Value::NULL; 17];
                values[0] = Value::from_word(receiver);
                for (index, word) in args.iter().copied().enumerate() { values[index + 1] = Value::from_word(word); }
                match self.call_host_primitive(primitive, &values[..args.len() + 1]) {
                    Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                    Err(error) => self.fail_at(op, error.message),
                }
            }
            CallTarget::Foreign(foreign) => {
                self.mark_gc_point(op);
                self.stage_invoke_args(receiver, args);
                let mut values = [Value::NULL; 17];
                values[0] = Value::from_word(receiver);
                for (index, word) in args.iter().copied().enumerate() { values[index + 1] = Value::from_word(word); }
                match self.call_host_foreign(foreign, &values[..args.len() + 1]) {
                    Ok(result) => unsafe { self.set_reg(op.a, result.word()); self.advance(op); },
                    Err(error) => self.fail_at(op, error.message),
                }
            }
        }
    }

    #[inline]
    fn stage_runtime_call_args(&mut self, args: &[RegisterWord]) {
        let frame = self.current_frame_mut();
        for (index, word) in args.iter().copied().enumerate() {
            frame.arguments[index] = word;
        }
    }

    #[inline]
    fn stage_invoke_args(&mut self, receiver: RegisterWord, args: &[RegisterWord]) {
        let frame = self.current_frame_mut();
        frame.special[SpecialReg::This.index()] = receiver;
        for (index, word) in args.iter().copied().enumerate() {
            frame.arguments[index] = word;
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
        if let Some(native) = self.core_foreign_methods.get(&foreign).copied() {
            return self.call_native(native, args);
        }
        let (host, heap) = (&mut self.host, &mut self.heap);
        let mut context = HostContext { heap };
        host.call_foreign(&mut context, foreign, args)
    }

    fn call_native(&mut self, method: NativeMethod, args: &[Value]) -> Result<Value, RuntimeError> {
        use NativeMethod::*;
        let receiver = args.first().copied().ok_or_else(|| RuntimeError::new("missing native receiver"))?;
        let argument = || args.get(1).copied().ok_or_else(|| RuntimeError::new("missing native argument"));
        match method {
            WriteString => {
                let value = argument()?;
                let Some(HeapObject::String(bytes)) = self.heap.get(value) else {
                    return Err(RuntimeError::new("toString must return a string"));
                };
                self.host.write(bytes)?;
                Ok(Value::NULL)
            }
            Clock => Ok(Value::from_num(self.started_at.elapsed().as_secs_f64())),
            CollectGarbage => { self.collect_garbage(); Ok(Value::NULL) }
            FnNew => {
                let value = argument()?;
                if !matches!(self.heap.get(value), Some(HeapObject::Closure { .. })) {
                    return Err(RuntimeError::new("Fn.new expects a function"));
                }
                Ok(value)
            }
            ClosureArity => match self.heap.get(receiver) {
                Some(HeapObject::Closure { function_index, .. }) =>
                    Ok(Value::from_num(self.program.functions[*function_index as usize].arity as f64)),
                _ => Err(RuntimeError::new("arity receiver must be a function")),
            },
            ClosureCall => Err(RuntimeError::new("closure calls require an interpreter frame")),
            ToString => {
                if matches!(self.heap.get(receiver), Some(HeapObject::String(_))) {
                    return Ok(receiver);
                }
                let text = self.value_to_string(receiver);
                Ok(self.heap.alloc(HeapObject::String(Arc::from(text.as_bytes()))))
            }
            Not => Ok(Value::from_bool(receiver.is_null() || receiver.as_bool() == Some(false))),
            Equal | NotEqual => {
                let other = argument()?;
                let equal = match (receiver.as_num(), other.as_num()) {
                    (Some(a), Some(b)) => a == b,
                    _ => match (self.heap.get(receiver), self.heap.get(other)) {
                        (Some(HeapObject::String(a)), Some(HeapObject::String(b))) => a == b,
                        _ => receiver == other,
                    },
                };
                Ok(Value::from_bool(if matches!(method, Equal) { equal } else { !equal }))
            }
            StringConcat => {
                let other = argument()?;
                let (Some(HeapObject::String(a)), Some(HeapObject::String(b))) =
                    (self.heap.get(receiver), self.heap.get(other)) else {
                        return Err(RuntimeError::new("string '+' expects a string argument"));
                    };
                let mut bytes = Vec::with_capacity(a.len() + b.len());
                bytes.extend_from_slice(a);
                bytes.extend_from_slice(b);
                Ok(self.heap.alloc(HeapObject::String(Arc::from(bytes))))
            }
            Negate => Ok(Value::from_num(-receiver.as_num().ok_or_else(|| RuntimeError::new("expected a number"))?)),
            Add | Subtract | Multiply | Divide | Remainder | Less | LessEqual | Greater | GreaterEqual => {
                let a = receiver.as_num().ok_or_else(|| RuntimeError::new("expected a number"))?;
                let b = argument()?.as_num().ok_or_else(|| RuntimeError::new("numeric operator expects a number argument"))?;
                Ok(match method {
                    Add => Value::from_num(a + b),
                    Subtract => Value::from_num(a - b),
                    Multiply => Value::from_num(a * b),
                    Divide => Value::from_num(a / b),
                    Remainder => Value::from_num(a % b),
                    Less => Value::from_bool(a < b),
                    LessEqual => Value::from_bool(a <= b),
                    Greater => Value::from_bool(a > b),
                    GreaterEqual => Value::from_bool(a >= b),
                    _ => unreachable!(),
                })
            }
        }
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
            self.current_frame_mut().pc = resume;
        } else {
            self.fail_at(op, format!("missing deopt target for DeoptId#{deopt_id}"));
        }
    }

    fn maybe_collect(&mut self) {
        if self.gc_enabled && self.heap.should_collect() { self.collect_garbage(); }
    }

    pub fn collect_garbage(&mut self) {
        let mut work = Vec::<u32>::new();
        for &value in self.strings.iter().chain(self.roots.iter().filter_map(|x| x.as_ref())) {
            mark_value_handle(&mut self.heap, value, &mut work);
        }
        for class in &self.classes {
            mark_value_handle(&mut self.heap, class.class_value, &mut work);
            for &value in &class.static_fields { mark_value_handle(&mut self.heap, value, &mut work); }
        }
        for module in &self.modules {
            mark_value_handle(&mut self.heap, module.module_value, &mut work);
            for &value in &module.variables { mark_value_handle(&mut self.heap, value, &mut work); }
        }
        if let Some(fiber) = self.fibers.get(self.current_fiber as usize).and_then(|x| x.as_ref()) {
            mark_value_handle(&mut self.heap, fiber.object, &mut work);
        }

        let mut traced_fibers = HashSet::<u32>::new();
        while let Some(handle) = work.pop() {
            let object_ptr: *const HeapObject = match self.heap.slots.get(handle as usize).and_then(|x| x.as_ref()) {
                Some(slot) => &slot.object,
                None => continue,
            };
            // SAFETY: objects are not moved or swept until the mark worklist is empty.
            match unsafe { &*object_ptr } {
                HeapObject::List(values) => for &value in values { mark_value_handle(&mut self.heap, value, &mut work); },
                HeapObject::Map(map) => {
                    for (key, &value) in &map.entries {
                        if let MapKey::Object(handle) = key { mark_handle(&mut self.heap, *handle, &mut work); }
                        mark_value_handle(&mut self.heap, value, &mut work);
                    }
                }
                HeapObject::Cell(word) => mark_value_handle(&mut self.heap, Value::from_word(*word), &mut work),
                HeapObject::Closure { function_index, captures } => {
                    let function = &self.prepared.functions[*function_index as usize];
                    for binding in &function.frame.captures {
                        if binding.kind == RegisterKind::Value {
                            if let Some(&word) = captures.get(binding.capture_index as usize) {
                                mark_value_handle(&mut self.heap, Value::from_word(word), &mut work);
                            }
                        }
                    }
                }
                HeapObject::Instance { fields, .. } => for &value in fields.iter() { mark_value_handle(&mut self.heap, value, &mut work); },
                HeapObject::FiberObject { fiber } => {
                    if traced_fibers.insert(*fiber) { self.mark_fiber_roots(*fiber, &mut work); }
                }
                HeapObject::String(_) | HeapObject::Range { .. } | HeapObject::ClassObject { .. }
                | HeapObject::Foreign { .. } | HeapObject::ModuleObject { .. } => {}
            }
        }

        self.heap.free.clear();
        self.heap.live = 0;
        for (index, slot) in self.heap.slots.iter_mut().enumerate() {
            match slot {
                Some(object) if object.marked => {
                    object.marked = false;
                    self.heap.live += 1;
                }
                Some(_) => {
                    *slot = None;
                    self.heap.free.push(index as u32);
                }
                None => self.heap.free.push(index as u32),
            }
        }
        self.heap.allocations_since_gc = 0;
        self.heap.next_gc = self.heap.live.saturating_mul(2).max(4096);

        for index in 1..self.fibers.len() {
            let keep = self.fibers[index].as_ref().map_or(false, |fiber| {
                fiber.object.heap_handle().and_then(|handle| self.heap.slots.get(handle as usize)).and_then(|x| x.as_ref()).is_some()
            });
            if !keep && self.fibers[index].take().is_some() { self.free_fibers.push(index as u32); }
        }
    }

    fn mark_fiber_roots(&mut self, fiber_id: u32, work: &mut Vec<u32>) {
        let Some(fiber) = self.fibers.get(fiber_id as usize).and_then(|x| x.as_ref()) else { return; };
        let mut values = Vec::<Value>::new();
        if let Some(starter) = fiber.starter { values.push(starter); }
        values.push(fiber.last_value);
        values.push(fiber.error);
        // A call()/try() link is a strong runtime edge. The active callee must
        // keep every suspended caller in its chain alive even when user code
        // no longer holds an explicit Fiber reference. transfer() creates no
        // such edge by design.
        if let Some(link) = fiber.link {
            if let Some(caller) = self.fibers.get(link.caller as usize).and_then(|slot| slot.as_ref()) {
                values.push(caller.object);
            }
        }
        for frame in &fiber.frames {
            let function = &self.prepared.functions[frame.function as usize];
            if frame.root_map != NO_INDEX {
                if let Some(map) = function.gc_maps.get(frame.root_map as usize) {
                    for reg in 0..frame.reg_count as usize {
                        if map.general_is_reference(Reg(reg as u8)) {
                            values.push(Value::from_word(fiber.registers[frame.reg_base + reg]));
                        }
                    }
                    for index in 0..ARGUMENT_REGISTER_COUNT {
                        if (map.arguments & (1u16 << index)) != 0 { values.push(Value::from_word(frame.arguments[index])); }
                    }
                    for index in 0..RETURN_REGISTER_COUNT {
                        if (map.returns & (1u8 << index)) != 0 { values.push(Value::from_word(frame.returns[index])); }
                    }
                    for index in 0..SPECIAL_REGISTER_COUNT {
                        if (map.special & (1u16 << index)) != 0 { values.push(Value::from_word(frame.special[index])); }
                    }
                }
            }
        }
        for value in values { mark_value_handle(&mut self.heap, value, work); }
    }
}

#[inline]
fn mark_value_handle(heap: &mut Heap, value: Value, work: &mut Vec<u32>) {
    if let Some(handle) = value.heap_handle() { mark_handle(heap, handle, work); }
}

#[inline]
fn mark_handle(heap: &mut Heap, handle: u32, work: &mut Vec<u32>) {
    if let Some(Some(slot)) = heap.slots.get_mut(handle as usize) {
        if !slot.marked { slot.marked = true; work.push(handle); }
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
        if self.program.functions
            .get(function_index as usize)
            .map_or(true, |function| function.arity > 1)
        {
            return Err(RuntimeError::new("Fiber function cannot take more than one parameter"));
        }
        let id = if let Some(id) = self.free_fibers.pop() { id } else { self.fibers.len() as u32 };
        let object = self.heap.alloc(HeapObject::FiberObject { fiber: id });
        let mut fiber = FiberState::new(object, FiberStatus::New);
        fiber.starter = Some(closure);
        if id as usize == self.fibers.len() { self.fibers.push(Some(fiber)); } else { self.fibers[id as usize] = Some(fiber); }
        let _ = function_index;
        Ok(object)
    }

    fn fiber_id_from_value(&self, value: Value) -> Result<u32, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::FiberObject { fiber }) => Ok(*fiber),
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
        self.current_frame_mut().pc = op.next;
        self.fiber_mut(source).status = FiberStatus::Suspended;

        if is_call {
            self.fiber_mut(target).link = Some(FiberLink {
                caller: source,
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

            self.current_fiber = target;
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
            if let Some(frame) = self.fiber_mut(target).frames.last_mut() {
                frame.special[SpecialReg::CallerFiber.index()] = caller_value.word();
                frame.special[SpecialReg::ResumeValue.index()] = argument.word();
            }
            return;
        }

        let suspend_dst = self.fiber_mut(target).suspend_dst.take();
        self.current_fiber = target;
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
        if let Some(frame) = self.fiber_mut(target).frames.last_mut() {
            frame.special[SpecialReg::ResumeValue.index()] = argument.word();
            frame.special[SpecialReg::CallerFiber.index()] = caller_value.word();
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
        self.current_frame_mut().pc = op.next;
        self.fiber_mut(source).status = FiberStatus::Suspended;
        self.fiber_mut(source).suspend_dst = Some(Reg(op.a));

        // transferError() switches to the target without becoming its caller,
        // then immediately aborts that target. Error propagation therefore
        // follows the caller chain the target already had (typically from
        // try()), exactly like Wren's runFiber(..., isCall=false) + error.
        self.current_fiber = target;
        self.fiber_mut(target).status = FiberStatus::Running;
        self.refresh_hot_frame();
        self.abort_current_fiber(op, error);
    }

    fn yield_current_fiber(&mut self, op: &PreparedOp, value: Value, suspend_dst: Option<Reg>) {
        let current = self.current_fiber;
        self.mark_gc_point(op);
        self.current_frame_mut().pc = op.next;

        let link = self.fiber_mut(current).link.take();
        self.fiber_mut(current).status = FiberStatus::Suspended;
        self.fiber_mut(current).suspend_dst = suspend_dst;

        if let Some(link) = link {
            self.current_fiber = link.caller;
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
        self.current_frame_mut().pc = op.next;
        let current = self.current_fiber;
        self.fiber_mut(current).status = FiberStatus::Suspended;
        self.fiber_mut(current).suspend_dst = Some(dst);
        self.result = Value::NULL;
        self.run_state = RunState::Suspended;
    }

    fn abort_current_fiber(&mut self, op: &PreparedOp, error: Value) {
        let current = self.current_fiber;
        if current == 0 {
            self.fail_at(op, format!("fiber aborted: {:?}", error));
            return;
        }
        self.fiber_mut(current).status = FiberStatus::Failed;
        self.fiber_mut(current).error = error;
        let link = self.fiber_mut(current).link.take();
        let Some(link) = link else {
            self.fail(RuntimeError::new(format!("unhandled fiber error: {:?}", error)));
            return;
        };
        self.current_fiber = link.caller;
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

const OPCODE_COUNT: usize = 122;

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
];

#[inline(always)]
fn handler_for(opcode: Opcode) -> Handler {
    // Opcode values are deliberately dense 0..121. Preparation has already
    // validated the byte stream, so dispatch selection is a single indexed
    // load instead of a generated 122-way match.
    unsafe { *HANDLER_TABLE.get_unchecked(opcode as usize) }
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
    let function=vm.current_frame().function;
    let value=vm.constant_word_unchecked(function,op.x);
    vm.set_reg(op.a,value);
    vm.advance(op)
}
unsafe fn h_move_from_arg(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.current_frame().arguments[op.b as usize];vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_arg(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.current_frame_mut().arguments[op.a as usize]=v;vm.advance(op)}
unsafe fn h_move_from_ret(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.current_frame().returns[op.b as usize];vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_ret(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.current_frame_mut().returns[op.a as usize]=v;vm.advance(op)}
unsafe fn h_move_from_special(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.current_frame().special[op.b as usize];vm.set_reg(op.a,v);vm.advance(op)}
unsafe fn h_move_to_special(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.current_frame_mut().special[op.a as usize]=v;vm.advance(op)}

unsafe fn h_jump(vm:&mut Interpreter, ptr:*const PreparedOp){vm.set_pc((*ptr).x)}
unsafe fn h_jump_if_false(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;if Value::from_word(vm.reg(op.a)).is_falsey(){vm.set_pc(op.x)}else{vm.advance(op)}}
unsafe fn h_jump_if_true(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;if !Value::from_word(vm.reg(op.a)).is_falsey(){vm.set_pc(op.x)}else{vm.advance(op)}}
unsafe fn h_switch(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let word=vm.reg(op.a); let v=if Value::from_word(word).is_num(){Value::from_word(word).as_num().unwrap() as i64}else{word_i32(word) as i64};
    let target=vm.current_prepared().switches[op.x as usize].target(v); vm.set_pc(target)
}
unsafe fn h_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.a);vm.return_from_frame(v,None)}
unsafe fn h_return2(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let a=vm.reg(op.a);let b=vm.reg(op.b);vm.return_from_frame(a,Some(b))}
unsafe fn h_move_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.return_from_frame(v,None)}
unsafe fn h_move_jump(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let v=vm.reg(op.b);vm.set_reg(op.a,v);vm.set_pc(op.y)}
unsafe fn h_load_null_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::NULL.word(),None)}
unsafe fn h_load_false_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::FALSE.word(),None)}
unsafe fn h_load_true_return(vm:&mut Interpreter, _ptr:*const PreparedOp){vm.return_from_frame(Value::TRUE.word(),None)}
unsafe fn h_load_i8_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.return_from_frame(i32_word(op.b as i8 as i32),None)}
unsafe fn h_load_i32_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.return_from_frame(i32_word(op.x as i32),None)}
unsafe fn h_load_const_return(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let f=vm.current_frame().function;let v=vm.constant_word_unchecked(f,op.x);vm.return_from_frame(v,None)}

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
    let op=&*ptr;
    let function=(*vm.hot_frame).function;
    let receiver=vm.reg(op.b);
    let receiver_value=Value::from_word(receiver);
    let class=match vm.receiver_class(receiver_value){
        Ok(class)=>class,
        Err(error)=>{vm.fail_at(op,error.message);return;}
    };
    let target=match vm.probe_invoke_cache(function,op.x,class){
        Some(target)=>target,
        None=>match vm.resolve_invoke_miss(function,op.x,class){
            Ok(target)=>target,
            Err(error)=>{vm.fail_at(op,error.message);return;}
        },
    };
    let mut args=[RegisterWord(0);17];
    let count=vm.gather_call_words(op,&mut args);
    vm.invoke_target(op,target,receiver,&args[..count]);
}
unsafe fn h_call_direct(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words(op,&mut words);
    vm.mark_gc_point(op); vm.current_frame_mut().pc=op.next; if let Err(e)=vm.push_frame_direct(op.x,&words[..count],None,Some(Reg(op.a))){vm.fail_at(op,e.message)}
}
unsafe fn h_call_closure(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; let closure=Value::from_word(vm.reg(op.b)); let function=match vm.heap.get(closure){Some(HeapObject::Closure{function_index,..})=>*function_index,_=>{vm.fail_at(op,"CallClosure operand is not a closure");return;}};
    let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words(op,&mut words);
    vm.mark_gc_point(op); vm.current_frame_mut().pc=op.next; if let Err(e)=vm.push_frame(function,None,&words[..count],Some(closure),Some(Reg(op.a))){vm.fail_at(op,e.message)}
}
unsafe fn h_call_primitive(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; vm.mark_gc_point(op); let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words(op,&mut words); vm.stage_runtime_call_args(&words[..count]); let mut args=[Value::NULL;17]; for i in 0..count{args[i]=Value::from_word(words[i]);}
    match vm.call_host_primitive(PrimitiveId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
}
unsafe fn h_call_foreign(vm:&mut Interpreter, ptr:*const PreparedOp){
    let op=&*ptr; vm.mark_gc_point(op); let mut words=[RegisterWord(0);17]; let count=vm.gather_call_words(op,&mut words); vm.stage_runtime_call_args(&words[..count]); let mut args=[Value::NULL;17]; for i in 0..count{args[i]=Value::from_word(words[i]);}
    match vm.call_host_foreign(ForeignFnId(op.x),&args[..count]){Ok(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},Err(e)=>vm.fail_at(op,e.message)}
}

unsafe fn h_create_class(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let slot=op.x;if op.c!=0{let super_value=Value::from_word(vm.reg(op.b));let super_slot=match vm.heap.get(super_value){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"CreateClass superclass is not a class");return;}};vm.classes[slot as usize].superclass=Some(super_slot);}let value=vm.classes[slot as usize].class_value;vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_bind_method(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class_value=Value::from_word(vm.reg(op.a));let slot=match vm.heap.get(class_value){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"BindMethod target is not a class");return;}};let function=match vm.prepared.function_by_id.get(&FuncId(op.y)).copied(){Some(v)=>v,None=>{vm.fail_at(op,"BindMethod function does not exist");return;}};let table=if op.b!=0{&mut vm.classes[slot as usize].static_methods}else{&mut vm.classes[slot as usize].instance_methods};match table.get_mut(op.x as usize){Some(entry)=>{*entry=Some(CallTarget::Bytecode(function));vm.advance(op)},None=>vm.fail_at(op,"BindMethod symbol out of range")}}
unsafe fn h_bind_foreign_method(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class_value=Value::from_word(vm.reg(op.a));let slot=match vm.heap.get(class_value){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"BindForeignMethod target is not a class");return;}};let table=if op.b!=0{&mut vm.classes[slot as usize].static_methods}else{&mut vm.classes[slot as usize].instance_methods};match table.get_mut(op.x as usize){Some(entry)=>{*entry=Some(CallTarget::Foreign(ForeignFnId(op.y)));vm.advance(op)},None=>vm.fail_at(op,"BindForeignMethod symbol out of range")}}
unsafe fn h_alloc_instance(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let count=vm.classes[op.x as usize].instance_field_count as usize;let value=vm.heap.alloc(HeapObject::Instance{class_slot:op.x,fields:vec![Value::NULL;count].into_boxed_slice()});vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_alloc_foreign(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let size=word_i32(vm.reg(op.b));if size<0{vm.fail_at(op,"negative foreign allocation size");return;}let value=vm.heap.alloc(HeapObject::Foreign{class_slot:op.x,bytes:vec![0;size as usize]});vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_load_field(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let obj=Value::from_word(vm.reg(op.b));let value=match vm.heap.get(obj){Some(HeapObject::Instance{fields,..})=>fields.get(op.x as usize).copied(),_=>None};match value{Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"LoadField slot/object invalid")}}
unsafe fn h_store_field(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let obj=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));match vm.heap.get_mut(obj){Some(HeapObject::Instance{fields,..})=>if let Some(slot)=fields.get_mut(op.x as usize){*slot=value;vm.advance(op)}else{vm.fail_at(op,"StoreField slot out of range")},_=>vm.fail_at(op,"StoreField object is not an instance")}}
unsafe fn h_load_static_field(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let class=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(class){Some(HeapObject::ClassObject{class_slot})=>*class_slot,_=>{vm.fail_at(op,"LoadStaticField target is not class");return;}};match vm.classes[slot as usize].static_fields.get(op.x as usize).copied(){Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"static field slot out of range")}}
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
    vm.current_frame_mut().pc=op.next;
    vm.modules.get_unchecked_mut(slot as usize).initializing=true;

    // No return destination is needed: the module object was written above.
    // Returning from ModuleBody marks the module initialized and resumes here.
    if let Err(error)=vm.push_frame_direct(body.unwrap_unchecked(),&[],None,None){
        vm.modules.get_unchecked_mut(slot as usize).initializing=false;
        vm.fail_at(op,error.message);
    }
}
unsafe fn h_import_variable(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let module=Value::from_word(vm.reg(op.b));let slot=match vm.heap.get(module){Some(HeapObject::ModuleObject{module_slot})=>*module_slot,_=>{vm.fail_at(op,"ImportVariable operand is not module");return;}};let name=StringId(op.x);let index=vm.modules[slot as usize].exports.get(&name).copied();match index.and_then(|i|vm.modules[slot as usize].variables.get(i as usize).copied()){Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"imported variable name is not bound")}}
unsafe fn h_load_module_var(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;match vm.modules[op.x as usize].variables.get(op.y as usize).copied(){Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"module variable slot out of range")}}
unsafe fn h_store_module_var(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let value=Value::from_word(vm.reg(op.a));if let Some(slot)=vm.modules[op.x as usize].variables.get_mut(op.y as usize){*slot=value;vm.advance(op)}else{vm.fail_at(op,"module variable slot out of range")}}

unsafe fn h_alloc_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let value=vm.heap.alloc(HeapObject::Cell(vm.reg(op.b)));vm.set_reg(op.a,value.word());vm.advance(op)}
unsafe fn h_load_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cell=Value::from_word(vm.reg(op.b));match vm.heap.get(cell){Some(HeapObject::Cell(v))=>{vm.set_reg(op.a,*v);vm.advance(op)},_=>vm.fail_at(op,"LoadCell operand is not cell")}}
unsafe fn h_store_cell(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cell=Value::from_word(vm.reg(op.a));let value=vm.reg(op.b);match vm.heap.get_mut(cell){Some(HeapObject::Cell(v))=>{*v=value;vm.advance(op)},_=>vm.fail_at(op,"StoreCell operand is not cell")}}
unsafe fn h_make_closure(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let count=((op.b as u16)|((op.c as u16)<<8)) as usize;let current=vm.current_frame().function as usize;let start=op.args_offset as usize;let mut captures=Vec::with_capacity(count);for index in 0..count{let reg=vm.prepared.functions[current].call_args[start+index];captures.push(vm.reg(reg.0));}let value=vm.heap.alloc(HeapObject::Closure{function_index:op.x,captures:captures.into_boxed_slice()});vm.set_reg(op.a,value.word());vm.advance(op)}

#[inline]
fn normalize_index(index: i32, len: usize) -> Option<usize> {
    let index = if index < 0 { len as i64 + index as i64 } else { index as i64 };
    if index >= 0 && (index as usize) < len { Some(index as usize) } else { None }
}

unsafe fn h_alloc_list(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::List(Vec::new()));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_alloc_list_cap(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let cap=word_i32(vm.reg(op.b));if cap<0{vm.fail_at(op,"negative list capacity");return;}vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::List(Vec::with_capacity(cap as usize)));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_list_len(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));match vm.heap.get(list){Some(HeapObject::List(values))=>{vm.set_reg(op.a,i32_word(values.len() as i32));vm.advance(op)},_=>vm.fail_at(op,"ListLen operand is not list")}}
unsafe fn h_list_get(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));let index=word_i32(vm.reg(op.c));let value=match vm.heap.get(list){Some(HeapObject::List(values))=>normalize_index(index,values.len()).and_then(|i|values.get(i).copied()),_=>None};match value{Some(v)=>{vm.set_reg(op.a,v.word());vm.advance(op)},None=>vm.fail_at(op,"list index out of bounds")}}
unsafe fn h_list_set(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let index=word_i32(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));match vm.heap.get_mut(list){Some(HeapObject::List(values))=>if let Some(i)=normalize_index(index,values.len()){values[i]=value;vm.advance(op)}else{vm.fail_at(op,"list index out of bounds")},_=>vm.fail_at(op,"ListSet operand is not list")}}
unsafe fn h_list_append(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let value=Value::from_word(vm.reg(op.b));vm.mark_gc_point(op);vm.maybe_collect();match vm.heap.get_mut(list){Some(HeapObject::List(values))=>{values.push(value);vm.advance(op)},_=>vm.fail_at(op,"ListAppend operand is not list")}}
unsafe fn h_list_insert(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.a));let index=word_i32(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));vm.mark_gc_point(op);vm.maybe_collect();match vm.heap.get_mut(list){Some(HeapObject::List(values))=>{let mut i=index as i64;if i<0{i=values.len() as i64+i+1;}if i<0||i>values.len() as i64{vm.fail_at(op,"list insertion index out of bounds")}else{values.insert(i as usize,value);vm.advance(op)}},_=>vm.fail_at(op,"ListInsert operand is not list")}}
unsafe fn h_list_remove_at(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let list=Value::from_word(vm.reg(op.b));let index=word_i32(vm.reg(op.c));match vm.heap.get_mut(list){Some(HeapObject::List(values))=>if let Some(i)=normalize_index(index,values.len()){let value=values.remove(i);vm.set_reg(op.a,value.word());vm.advance(op)}else{vm.fail_at(op,"list removal index out of bounds")},_=>vm.fail_at(op,"ListRemoveAt operand is not list")}}

unsafe fn h_alloc_map(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;vm.mark_gc_point(op);vm.maybe_collect();let v=vm.heap.alloc(HeapObject::Map(WrenMap::default()));vm.set_reg(op.a,v.word());vm.advance(op)}
unsafe fn h_map_len(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map=Value::from_word(vm.reg(op.b));match vm.heap.get(map){Some(HeapObject::Map(map))=>{vm.set_reg(op.a,i32_word(map.entries.len() as i32));vm.advance(op)},_=>vm.fail_at(op,"MapLen operand is not map")}}
unsafe fn h_map_get(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map=Value::from_word(vm.reg(op.b));let key_value=Value::from_word(vm.reg(op.c));let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};match vm.heap.get(map){Some(HeapObject::Map(map))=>{let value=map.entries.get(&key).copied().unwrap_or(Value::NULL);vm.set_reg(op.a,value.word());vm.advance(op)},_=>vm.fail_at(op,"MapGet operand is not map")}}
unsafe fn h_map_set(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map_value=Value::from_word(vm.reg(op.a));let key_value=Value::from_word(vm.reg(op.b));let value=Value::from_word(vm.reg(op.c));let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};vm.mark_gc_point(op);vm.maybe_collect();match vm.heap.get_mut(map_value){Some(HeapObject::Map(map))=>{map.entries.insert(key,value);vm.advance(op)},_=>vm.fail_at(op,"MapSet operand is not map")}}
unsafe fn h_map_remove(vm:&mut Interpreter, ptr:*const PreparedOp){let op=&*ptr;let map_value=Value::from_word(vm.reg(op.b));let key_value=Value::from_word(vm.reg(op.c));let key=match vm.value_to_map_key(key_value){Ok(k)=>k,Err(e)=>{vm.fail_at(op,e.message);return;}};match vm.heap.get_mut(map_value){Some(HeapObject::Map(map))=>{let v=map.entries.remove(&key).unwrap_or(Value::NULL);vm.set_reg(op.a,v.word());vm.advance(op)},_=>vm.fail_at(op,"MapRemove operand is not map")}}

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

unsafe fn h_write_barrier(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;vm.advance(op)}
unsafe fn h_read_barrier(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let value=vm.reg(op.b);vm.set_reg(op.a,value);vm.advance(op)}
unsafe fn h_profile_block(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let f=vm.current_frame().function as usize;if let Some(counter)=vm.profile_counters[f].get_mut(op.x as usize){*counter=counter.saturating_add(1);}vm.advance(op)}
unsafe fn h_profile_numeric(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let f=vm.current_frame().function as usize;if let Some(counter)=vm.profile_counters[f].get_mut(op.x as usize){*counter=counter.saturating_add(1);}vm.advance(op)}
unsafe fn h_breakpoint(vm:&mut Interpreter,ptr:*const PreparedOp){let op=&*ptr;let(f,pc)=vm.current_location(op);match vm.host.breakpoint(f,pc){Ok(())=>vm.advance(op),Err(e)=>vm.fail_at(op,e.message)}}

#[inline(always)]
unsafe fn fused_branch(vm:&mut Interpreter,op:&PreparedOp,condition:bool){let take=if op.flags!=0{condition}else{!condition};if take{vm.set_pc(op.y)}else{vm.set_pc(op.next)}}
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

    pub fn heap_live_objects(&self) -> usize { self.heap.live }

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
            Some(HeapObject::String(string)) => String::from_utf8_lossy(string).into_owned(),
            Some(HeapObject::List(values)) => format!("<list:{}>", values.len()),
            Some(HeapObject::Map(map)) => format!("<map:{}>", map.entries.len()),
            Some(HeapObject::Range { from, to, inclusive }) => format!("{}{}{}", from, if *inclusive { ".." } else { "..." }, to),
            Some(HeapObject::Closure { function_index, .. }) => format!("<closure {:?}>", self.prepared.functions[*function_index as usize].id),
            Some(HeapObject::ClassObject { class_slot }) => self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("<class>").to_owned(),
            Some(HeapObject::Instance { class_slot, .. }) => format!("<{} instance>", self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("class")),
            Some(HeapObject::Foreign { class_slot, .. }) => format!("<foreign {}>", self.program.metadata.string(self.classes[*class_slot as usize].name).unwrap_or("class")),
            Some(HeapObject::FiberObject { fiber }) => format!("<fiber:{fiber}>") ,
            Some(HeapObject::ModuleObject { module_slot }) => format!("<module:{}>", self.program.metadata.string(self.modules[*module_slot as usize].name).unwrap_or("module")),
            Some(HeapObject::Cell(_)) => "<cell>".into(),
            None => "<invalid value>".into(),
        }
    }
}

// =============================================================================
// Tests that require only hand-built bytecode.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wren_bytecode::{
        Assembler, BytecodeFunction, SafepointEntry, SafepointKind,
    };
    use crate::wren_ir::{FunctionKind, ModuleInfo};

    fn simple_program(chunk: crate::wren_bytecode::BytecodeChunk, gprs: u16) -> BytecodeProgram {
        let mut program = BytecodeProgram::new();
        program.metadata.modules.push(ModuleInfo { id: ModuleId(0), name: StringId(0), variable_count: 0 });
        program.metadata.strings.push("main".to_owned());
        program.push_function(BytecodeFunction {
            id: FuncId(0), module: ModuleId(0), kind: FunctionKind::ModuleBody,
            name: StringId(0), arity: 0, capture_count: 0,
            frame: FrameLayout { general_register_count: gprs, parameters: vec![], captures: vec![], return_kind: RegisterKind::Value },
            chunk,
        }).unwrap();
        program.rebuild_indexes().unwrap();
        program
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
