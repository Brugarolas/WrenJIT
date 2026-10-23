//! Wren IR (WIR) — mid/high-level SSA IR for a register interpreter and JIT.
//!
//! This IR is intentionally Wren-specific. It keeps source-language semantics
//! explicit (dynamic method invocation, fixed instance fields, fibers, modules,
//! closures) while also exposing optimization-only operations (guards, unboxed
//! numeric operations, deoptimization points, OSR markers).
//!
//! Core design rules:
//! - Data values use SSA `ValueId`s.
//! - Memory/control effects use a separate SSA `EffectId` namespace and never
//!   consume VM registers.
//! - Basic blocks use block parameters instead of Phi instructions.
//! - Every CFG edge carries both data arguments and the current effect token.
//! - Wren method dispatch is represented by `Invoke`; specialized calls are
//!   introduced only after devirtualization/speculation.
//! - Wren instance fields are fixed slots, not hidden-class/shape properties.
//! - Captured mutable variables are represented through explicit cells.
//! - `Num` is the Wren semantic number type. `Int32` and `F64` are internal
//!   optimization representations, not source-visible Wren types.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]

use hashbrown::HashMap;
use std::fmt;
use smallvec::SmallVec;

macro_rules! id_newtype {
    ($name:ident, $repr:ty) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name(pub $repr);

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
    };
}

id_newtype!(ValueId, u32);
id_newtype!(EffectId, u32);
id_newtype!(BlockId, u32);
id_newtype!(InstId, u32);
id_newtype!(FuncId, u32);
id_newtype!(ModuleId, u32);
id_newtype!(ClassId, u32);
id_newtype!(SiteId, u32);
id_newtype!(DeoptId, u32);
id_newtype!(ResumeId, u32);
id_newtype!(StringId, u32);
id_newtype!(MethodSymbol, u32);
id_newtype!(ForeignFnId, u32);
id_newtype!(PrimitiveId, u32);
id_newtype!(ModuleSlot, u32);
id_newtype!(FieldSlot, u16);
id_newtype!(StaticFieldSlot, u16);

/// A small, canonicalized set of observed/possible runtime classes.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct ClassSet {
    classes: SmallVec<[ClassId; 2]>,
}

impl ClassSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn singleton(class: ClassId) -> Self {
        Self {
            classes: SmallVec::from_slice(&[class]),
        }
    }

    pub fn from_iter(classes: impl IntoIterator<Item = ClassId>) -> Self {
        let mut classes: SmallVec<[ClassId; 2]> = classes.into_iter().collect();
        classes.sort_unstable();
        classes.dedup();
        Self { classes }
    }

    pub fn as_slice(&self) -> &[ClassId] {
        &self.classes
    }

    pub fn is_unknown(&self) -> bool {
        self.classes.is_empty()
    }

    pub fn contains(&self, class: ClassId) -> bool {
        self.classes.binary_search(&class).is_ok()
    }
}

impl fmt::Debug for ClassSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.classes.is_empty() {
            return write!(f, "Classes<?>");
        }

        write!(f, "Classes{{")?;
        for (index, class) in self.classes.iter().enumerate() {
            if index != 0 {
                write!(f, ",")?;
            }
            write!(f, "{:?}", class)?;
        }
        write!(f, "}}")
    }
}

/// Wren semantic types plus a few internal optimization representations.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Bottom,
    Top,

    // Wren-visible scalar types.
    Null,
    Bool,
    Num,
    String,

    // Wren-visible heap/runtime types.
    List(Option<Box<Type>>),
    Map,
    Range,
    Class(ClassSet),
    Instance(ClassSet),
    Closure(Option<FuncId>),
    Fiber,
    Foreign(Option<ClassId>),
    Module,

    // Internal location/reference type used for captured mutable variables.
    Cell(Box<Type>),

    // Internal optimization representations. These are never Wren-visible.
    Int32,
    F64,
    RawPtr,

    // Keep unions small and normalize them in optimization passes.
    Union(Vec<Type>),
}

impl Type {
    pub fn is_internal_representation(&self) -> bool {
        matches!(self, Type::Int32 | Type::F64 | Type::RawPtr)
    }

    pub fn is_reference_like(&self) -> bool {
        matches!(
            self,
            Type::String
                | Type::List(_)
                | Type::Map
                | Type::Range
                | Type::Class(_)
                | Type::Instance(_)
                | Type::Closure(_)
                | Type::Fiber
                | Type::Foreign(_)
                | Type::Module
                | Type::Cell(_)
        )
    }

    pub fn accepts(&self, actual: &Type) -> bool {
        use Type::*;

        if self == actual || matches!(self, Top) || matches!(actual, Bottom) {
            return true;
        }

        match (self, actual) {
            (Union(expected), actual) => expected.iter().any(|ty| ty.accepts(actual)),
            (expected, Union(actuals)) => actuals.iter().all(|ty| expected.accepts(ty)),
            (Instance(expected), Instance(actual)) | (Class(expected), Class(actual)) => {
                expected.is_unknown()
                    || (!actual.is_unknown()
                        && actual
                            .as_slice()
                            .iter()
                            .all(|class| expected.contains(*class)))
            }
            (Foreign(None), Foreign(_)) => true,
            (List(None), List(_)) => true,
            (List(Some(expected)), List(Some(actual))) => expected.accepts(actual),
            _ => false,
        }
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Type::*;
        match self {
            Bottom => write!(f, "⊥"),
            Top => write!(f, "⊤"),
            Null => write!(f, "Null"),
            Bool => write!(f, "Bool"),
            Num => write!(f, "Num"),
            String => write!(f, "String"),
            List(None) => write!(f, "List<?>"),
            List(Some(ty)) => write!(f, "List<{ty:?}>") ,
            Map => write!(f, "Map"),
            Range => write!(f, "Range"),
            Class(classes) => write!(f, "Class<{classes:?}>") ,
            Instance(classes) => write!(f, "Instance<{classes:?}>") ,
            Closure(Some(function)) => write!(f, "Closure<{function:?}>") ,
            Closure(None) => write!(f, "Closure<?>"),
            Fiber => write!(f, "Fiber"),
            Foreign(Some(class)) => write!(f, "Foreign<{class:?}>") ,
            Foreign(None) => write!(f, "Foreign<?>"),
            Module => write!(f, "Module"),
            Cell(ty) => write!(f, "Cell<{ty:?}>") ,
            Int32 => write!(f, "i32"),
            F64 => write!(f, "f64"),
            RawPtr => write!(f, "ptr"),
            Union(types) => {
                write!(f, "Union(")?;
                for (index, ty) in types.iter().enumerate() {
                    if index != 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{ty:?}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Null,
    Bool(bool),
    /// A Wren source-level number. Wren numbers are semantically f64 values.
    Num(f64),
    String(StringId),
    /// Internal constant introduced by integer-specialization passes.
    Int32(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MethodKind {
    Getter,
    Setter,
    Method,
    SubscriptGetter,
    SubscriptSetter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodSignature {
    pub name: StringId,
    pub kind: MethodKind,
    /// Number of explicit arguments. The receiver is not included.
    pub arity: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionKind {
    ModuleBody,
    Method,
    StaticMethod,
    Constructor,
    Closure,
    PrimitiveStub,
    ForeignStub,
}

/// Engine-provided core globals that are implicitly visible in every Wren module.
///
/// These are semantic identities, not runtime class slots. The interpreter owns
/// the concrete values and installs them into each module's reserved builtin
/// bindings before executing its body.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinId {
    Object = 0,
    Class = 1,
    Bool = 2,
    Null = 3,
    Num = 4,
    String = 5,
    List = 6,
    Map = 7,
    Range = 8,
    Fn = 9,
    Fiber = 10,
    System = 11,
    Date = 12,
    Random = 13,
    Promise = 14,
}

impl BuiltinId {
    pub const ALL: [Self; 15] = [
        Self::Object, Self::Class, Self::Bool, Self::Null, Self::Num,
        Self::String, Self::List, Self::Map, Self::Range, Self::Fn,
        Self::Fiber, Self::System, Self::Date, Self::Random, Self::Promise,
    ];
    pub const COUNT: usize = Self::ALL.len();
    #[inline] pub const fn name(self) -> &'static str {
        match self {
            Self::Object => "Object", Self::Class => "Class", Self::Bool => "Bool",
            Self::Null => "Null", Self::Num => "Num", Self::String => "String",
            Self::List => "List", Self::Map => "Map", Self::Range => "Range",
            Self::Fn => "Fn", Self::Fiber => "Fiber", Self::System => "System",
            Self::Date => "Date",
            Self::Random => "Random",
            Self::Promise => "Promise",
        }
    }
    #[inline] pub const fn index(self) -> usize { self as usize }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuiltinBinding {
    pub slot: ModuleSlot,
    pub builtin: BuiltinId,
}

#[derive(Clone, Debug)]
pub struct Signature {
    pub name: StringId,
    /// IR-level call parameters. Method-like functions include the receiver as
    /// parameter 0; `Invoke` keeps the receiver separate and lowering prepends it.
    pub params: Vec<Type>,
    pub result: Type,
}

#[derive(Clone, Debug)]
pub struct ModuleInfo {
    pub id: ModuleId,
    pub name: StringId,
    pub variable_count: u32,
    /// Slots preinitialized by the VM with engine-provided core class objects.
    pub builtin_bindings: Vec<BuiltinBinding>,
}

#[derive(Clone, Debug)]
pub struct ClassInfo {
    pub id: ClassId,
    pub module: ModuleId,
    pub name: StringId,
    /// Statically known superclass, when the superclass expression resolves to a
    /// class in this Program. Runtime class linking remains authoritative.
    pub superclass: Option<ClassId>,
    /// Instance fields declared directly by this class.
    ///
    /// FieldSlot operands in WIR/bytecode are class-local until runtime class
    /// linking. The interpreter relocates every method/closure owned by the class
    /// by the inherited field count exactly once, after the actual superclass is
    /// known. Hot execution therefore sees absolute fixed-layout field offsets.
    pub own_instance_field_count: u16,
    /// Static storage slots owned by this class. Static fields are not folded
    /// into the inherited instance layout.
    pub static_field_count: u16,
    pub is_foreign: bool,
}

#[derive(Clone, Debug)]
pub struct ForeignFunctionInfo {
    pub id: ForeignFnId,
    pub module: StringId,
    pub class: StringId,
    pub signature: MethodSymbol,
    pub is_static: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    pub file: Option<StringId>,
    pub line: u32,
    pub column: u32,
    pub length: u32,
}

/// Opaque resume location. The bytecode lowerer can later associate this with a
/// concrete bytecode PC without making the IR depend on a bytecode format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResumePoint {
    pub id: ResumeId,
}

#[derive(Clone, Debug)]
pub struct SnapshotSlot {
    /// Logical interpreter-visible slot. It is intentionally not a physical VM register.
    pub slot: u32,
    pub value: ValueId,
}

#[derive(Clone, Debug)]
pub struct InlinedFrameState {
    pub function: FuncId,
    pub resume: ResumePoint,
    pub slots: Vec<SnapshotSlot>,
}

#[derive(Clone, Debug)]
pub struct DeoptSnapshot {
    pub resume: ResumePoint,
    pub slots: Vec<SnapshotSlot>,
    pub inlined_frames: Vec<InlinedFrameState>,
}

impl DeoptSnapshot {
    pub fn new(resume: ResumePoint) -> Self {
        Self {
            resume,
            slots: Vec::new(),
            inlined_frames: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeoptPoint {
    pub id: DeoptId,
    pub snapshot: DeoptSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BailoutReason {
    TypeMismatch,
    ClassMismatch,
    BoundsCheck,
    NotInt32,
    NotF64,
    NullValue,
    MegamorphicCall,
    InvalidSpeculation,
    Explicit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Read,
    Write,
    Alloc,
    Call,
    Guard,
    Module,
    Class,
    Fiber,
    Foreign,
    Safepoint,
    Barrier,
    Control,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectBehavior {
    None,
    /// Consume the current effect and produce a new effect token.
    Chain(EffectKind),
    /// Consume the current effect without producing a successor token.
    Sink(EffectKind),
}

#[derive(Clone, Debug)]
pub struct EffectValue {
    pub id: EffectId,
    pub def: EffectDef,
}

#[derive(Clone, Copy, Debug)]
pub enum EffectDef {
    Entry,
    BlockParam(BlockId),
    InstResult(InstId, EffectKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntUnaryOp {
    Neg,
    BitNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntBinOp {
    Add,
    Sub,
    Mul,
    DivS,
    RemS,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    ShrU,
    Sar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F64UnaryOp {
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F64BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallConv {
    Wren,
    Fast,
    Foreign,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub target: BlockId,
    pub args: Vec<ValueId>,
    pub effect: EffectId,
}

#[derive(Clone, Debug)]
pub struct SwitchCase {
    pub value: i64,
    pub edge: Edge,
}

/// Mid/high-level Wren operations.
#[derive(Clone, Debug)]
pub enum Inst {
    // ----- Control flow -----
    Jump {
        edge: Edge,
    },
    Branch {
        /// Wren truthiness is applied by lowering; this value is not required to be Bool.
        cond: ValueId,
        then_edge: Edge,
        else_edge: Edge,
    },
    Switch {
        tag: ValueId,
        default_edge: Edge,
        cases: Vec<SwitchCase>,
    },
    Return(ValueId),
    Unreachable,

    // ----- Bookkeeping / tiering -----
    Safepoint,
    OSRPoint,

    // ----- Constants / value movement -----
    Const(Constant),
    Move(ValueId),

    // ----- Speculation / deoptimization -----
    GuardType {
        value: ValueId,
        expected: Type,
        deopt: DeoptId,
    },
    GuardClass {
        value: ValueId,
        expected: ClassSet,
        deopt: DeoptId,
    },
    GuardBounds {
        index: ValueId,
        length: ValueId,
        deopt: DeoptId,
    },
    GuardNotNull {
        value: ValueId,
        deopt: DeoptId,
    },
    GuardInt32 {
        value: ValueId,
        deopt: DeoptId,
    },
    GuardF64 {
        value: ValueId,
        deopt: DeoptId,
    },
    Bailout {
        reason: BailoutReason,
        deopt: DeoptId,
    },

    // ----- Wren dispatch / calls -----
    /// Generic Wren method dispatch. This is the semantic operation produced by
    /// the frontend for ordinary method calls and operators.
    Invoke {
        site: SiteId,
        receiver: ValueId,
        method: MethodSymbol,
        args: Vec<ValueId>,
    },
    /// `super.method(...)`; lookup starts at the superclass of `home_class`.
    InvokeSuper {
        site: SiteId,
        home_class: ClassId,
        receiver: ValueId,
        method: MethodSymbol,
        args: Vec<ValueId>,
    },
    /// Known internal target after devirtualization or for compiler-generated calls.
    CallDirect {
        conv: CallConv,
        callee: FuncId,
        args: Vec<ValueId>,
    },
    /// Known closure value after proving the callee is a closure.
    CallClosure {
        conv: CallConv,
        closure: ValueId,
        args: Vec<ValueId>,
    },
    /// Call a core VM primitive. The primitive table is runtime-owned.
    CallPrimitive {
        primitive: PrimitiveId,
        args: Vec<ValueId>,
    },
    /// Direct host/foreign call after foreign binding resolution.
    CallForeign {
        foreign: ForeignFnId,
        args: Vec<ValueId>,
    },

    // ----- Classes / instances -----
    /// Creates the runtime class object for a Wren class declaration.
    CreateClass {
        class: ClassId,
        superclass: Option<ValueId>,
    },
    /// Binds a compiled Wren method to a class object.
    BindMethod {
        class: ValueId,
        method: MethodSymbol,
        function: FuncId,
        is_static: bool,
    },
    /// Binds a host/foreign method to a class object.
    BindForeignMethod {
        class: ValueId,
        method: MethodSymbol,
        foreign: ForeignFnId,
        is_static: bool,
    },
    AllocInstance {
        class: ClassId,
    },
    AllocForeign {
        class: ClassId,
        size: ValueId,
    },
    /// Loads an instance field. `slot` is local to Function::field_owner in WIR
    /// and is relocated to an absolute fixed-layout slot by runtime class linking.
    LoadField {
        object: ValueId,
        slot: FieldSlot,
    },
    /// Stores an instance field; slot relocation follows LoadField semantics.
    StoreField {
        object: ValueId,
        slot: FieldSlot,
        value: ValueId,
    },
    LoadStaticField {
        class: ValueId,
        slot: StaticFieldSlot,
    },
    StoreStaticField {
        class: ValueId,
        slot: StaticFieldSlot,
        value: ValueId,
    },
    IsInstance {
        value: ValueId,
        class: ValueId,
    },

    // ----- Modules -----
    ImportModule {
        module_name: StringId,
    },
    ImportVariable {
        module: ValueId,
        variable_name: StringId,
    },
    LoadModuleVar {
        module: ModuleId,
        slot: ModuleSlot,
    },
    StoreModuleVar {
        module: ModuleId,
        slot: ModuleSlot,
        value: ValueId,
    },

    // ----- Closures / captured mutable state -----
    /// Allocates a shared mutable cell. Captured mutable locals are rewritten to
    /// cell accesses before closure construction.
    AllocCell {
        initial: ValueId,
    },
    LoadCell {
        cell: ValueId,
    },
    StoreCell {
        cell: ValueId,
        value: ValueId,
    },
    /// Captures are ordered exactly as the callee's `Function::captures` table.
    MakeClosure {
        function: FuncId,
        captures: Vec<ValueId>,
    },

    // ----- Lists -----
    AllocList {
        capacity: Option<ValueId>,
    },
    ListLen {
        list: ValueId,
    },
    ListGet {
        list: ValueId,
        index: ValueId,
    },
    ListSet {
        list: ValueId,
        index: ValueId,
        value: ValueId,
    },
    ListAppend {
        list: ValueId,
        value: ValueId,
    },
    ListInsert {
        list: ValueId,
        index: ValueId,
        value: ValueId,
    },
    ListRemoveAt {
        list: ValueId,
        index: ValueId,
    },

    // ----- Maps -----
    AllocMap,
    MapLen {
        map: ValueId,
    },
    MapGet {
        map: ValueId,
        key: ValueId,
    },
    MapSet {
        map: ValueId,
        key: ValueId,
        value: ValueId,
    },
    MapRemove {
        map: ValueId,
        key: ValueId,
    },

    // ----- Strings -----
    StringLen {
        string: ValueId,
    },
    StringConcat {
        left: ValueId,
        right: ValueId,
    },
    StringSlice {
        string: ValueId,
        start: ValueId,
        count: ValueId,
    },
    StringCodePointAt {
        string: ValueId,
        index: ValueId,
    },

    // ----- Ranges -----
    MakeRange {
        from: ValueId,
        to: ValueId,
        inclusive: bool,
    },

    // ----- Specialized numeric representation operations -----
    IntUnary {
        op: IntUnaryOp,
        value: ValueId,
    },
    IntBin {
        op: IntBinOp,
        left: ValueId,
        right: ValueId,
    },
    F64Unary {
        op: F64UnaryOp,
        value: ValueId,
    },
    F64Bin {
        op: F64BinOp,
        left: ValueId,
        right: ValueId,
    },
    CmpInt {
        kind: CmpKind,
        left: ValueId,
        right: ValueId,
    },
    CmpF64 {
        kind: CmpKind,
        left: ValueId,
        right: ValueId,
    },
    RefEq {
        left: ValueId,
        right: ValueId,
    },
    NumToInt32 {
        value: ValueId,
    },
    NumToF64 {
        value: ValueId,
    },
    Int32ToNum {
        value: ValueId,
    },
    F64ToNum {
        value: ValueId,
    },

    // ----- Async / promises -----
    /// Suspends the current async execution if `promise` is pending. A fulfilled
    /// promise produces its value immediately; rejection propagates through the
    /// current async task.
    Await {
        promise: ValueId,
    },

    // ----- Fibers -----
    MakeFiber {
        closure: ValueId,
    },
    FiberCall {
        fiber: ValueId,
        argument: ValueId,
    },
    FiberTry {
        fiber: ValueId,
        argument: ValueId,
    },
    FiberTransfer {
        fiber: ValueId,
        argument: ValueId,
    },
    FiberTransferError {
        fiber: ValueId,
        error: ValueId,
    },
    FiberYield {
        value: ValueId,
    },
    FiberSuspend,
    FiberAbort {
        error: ValueId,
    },

    // ----- GC lowering hooks -----
    /// Optional explicit barrier pseudo-op. Normal stores are still marked as
    /// writes; a GC lowering pass may insert this when required by the collector.
    WriteBarrier {
        object: ValueId,
        value: ValueId,
    },
    /// Optional explicit read barrier pseudo-op for collectors that require it.
    /// It returns the forwarded/usable reference and therefore has one result.
    ReadBarrier {
        object: ValueId,
    },
}

impl Inst {
    /// Conservative language-exception classification. Guards/deopts, memory
    /// operations and calls may enter a protected handler; total scalar SSA and
    /// ordinary control flow cannot. This is deliberately independent of purity
    /// (integer division, for example, is pure but may fail).
    pub fn may_throw(&self) -> bool {
        !matches!(self,
            Self::Jump { .. } | Self::Branch { .. } | Self::Switch { .. }
            | Self::Return(_) | Self::Unreachable | Self::Const(_) | Self::Move(_)
            | Self::IntUnary { .. }
            | Self::IntBin { op: IntBinOp::Add | IntBinOp::Sub | IntBinOp::Mul
                | IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor
                | IntBinOp::Shl | IntBinOp::ShrU | IntBinOp::Sar, .. }
            | Self::F64Unary { .. } | Self::F64Bin { .. }
            | Self::CmpInt { .. } | Self::CmpF64 { .. } | Self::RefEq { .. }
            | Self::Int32ToNum { .. } | Self::F64ToNum { .. }
            | Self::WriteBarrier { .. } | Self::ReadBarrier { .. }
            | Self::Safepoint | Self::OSRPoint)
    }

    pub fn effect_behavior(&self) -> EffectBehavior {
        use EffectBehavior as B;
        use EffectKind as E;
        use Inst::*;

        match self {
            Return(_) | Unreachable => B::Sink(E::Control),
            Bailout { .. } => B::Sink(E::Guard),
            FiberAbort { .. } => B::Sink(E::Fiber),

            Safepoint | OSRPoint => B::Chain(E::Safepoint),

            GuardType { .. }
            | GuardClass { .. }
            | GuardBounds { .. }
            | GuardNotNull { .. }
            | GuardInt32 { .. }
            | GuardF64 { .. } => B::Chain(E::Guard),

            Invoke { .. }
            | InvokeSuper { .. }
            | CallDirect { .. }
            | CallClosure { .. }
            | CallPrimitive { .. } => B::Chain(E::Call),

            CallForeign { .. } => B::Chain(E::Foreign),

            CreateClass { .. } => B::Chain(E::Alloc),
            BindMethod { .. } | BindForeignMethod { .. } => B::Chain(E::Class),
            AllocInstance { .. } | AllocForeign { .. } => B::Chain(E::Alloc),
            LoadField { .. } | LoadStaticField { .. } => B::Chain(E::Read),
            StoreField { .. } | StoreStaticField { .. } => B::Chain(E::Write),

            ImportModule { .. }
            | ImportVariable { .. }
            | LoadModuleVar { .. }
            | StoreModuleVar { .. } => B::Chain(E::Module),

            AllocCell { .. } | MakeClosure { .. } => B::Chain(E::Alloc),
            LoadCell { .. } => B::Chain(E::Read),
            StoreCell { .. } => B::Chain(E::Write),

            AllocList { .. }
            | AllocMap
            | StringConcat { .. }
            | StringSlice { .. }
            | MakeRange { .. } => B::Chain(E::Alloc),
            ListLen { .. }
            | ListGet { .. }
            | MapLen { .. }
            | MapGet { .. }
            | StringCodePointAt { .. } => B::Chain(E::Read),
            ListSet { .. }
            | ListAppend { .. }
            | ListInsert { .. }
            | ListRemoveAt { .. }
            | MapSet { .. }
            | MapRemove { .. } => B::Chain(E::Write),

            Await { .. } => B::Chain(E::Fiber),
            MakeFiber { .. } => B::Chain(E::Alloc),
            FiberCall { .. }
            | FiberTry { .. }
            | FiberTransfer { .. }
            | FiberTransferError { .. }
            | FiberYield { .. }
            | FiberSuspend => B::Chain(E::Fiber),

            WriteBarrier { .. } | ReadBarrier { .. } => B::Chain(E::Barrier),

            Jump { .. }
            | Branch { .. }
            | Switch { .. }
            | Const(_)
            | Move(_)
            | IsInstance { .. }
            | StringLen { .. }
            | IntUnary { .. }
            | IntBin { .. }
            | F64Unary { .. }
            | F64Bin { .. }
            | CmpInt { .. }
            | CmpF64 { .. }
            | RefEq { .. }
            | NumToInt32 { .. }
            | NumToF64 { .. }
            | Int32ToNum { .. }
            | F64ToNum { .. } => B::None,
        }
    }

    pub fn effect_kind(&self) -> Option<EffectKind> {
        match self.effect_behavior() {
            EffectBehavior::None => None,
            EffectBehavior::Chain(kind) | EffectBehavior::Sink(kind) => Some(kind),
        }
    }

    pub fn result_count(&self) -> usize {
        use Inst::*;

        match self {
            Jump { .. }
            | Branch { .. }
            | Switch { .. }
            | Return(_)
            | Unreachable
            | Safepoint
            | OSRPoint
            | GuardBounds { .. }
            | Bailout { .. }
            | BindMethod { .. }
            | BindForeignMethod { .. }
            | StoreField { .. }
            | StoreStaticField { .. }
            | StoreModuleVar { .. }
            | StoreCell { .. }
            | ListSet { .. }
            | ListAppend { .. }
            | ListInsert { .. }
            | MapSet { .. }
            | FiberAbort { .. }
            | WriteBarrier { .. } => 0,

            Const(_)
            | Move(_)
            | GuardType { .. }
            | GuardClass { .. }
            | GuardNotNull { .. }
            | GuardInt32 { .. }
            | GuardF64 { .. }
            | Invoke { .. }
            | InvokeSuper { .. }
            | CallDirect { .. }
            | CallClosure { .. }
            | CallPrimitive { .. }
            | CallForeign { .. }
            | CreateClass { .. }
            | AllocInstance { .. }
            | AllocForeign { .. }
            | LoadField { .. }
            | LoadStaticField { .. }
            | IsInstance { .. }
            | ImportModule { .. }
            | ImportVariable { .. }
            | LoadModuleVar { .. }
            | AllocCell { .. }
            | LoadCell { .. }
            | MakeClosure { .. }
            | AllocList { .. }
            | ListLen { .. }
            | ListGet { .. }
            | ListRemoveAt { .. }
            | AllocMap
            | MapLen { .. }
            | MapGet { .. }
            | MapRemove { .. }
            | StringLen { .. }
            | StringConcat { .. }
            | StringSlice { .. }
            | StringCodePointAt { .. }
            | MakeRange { .. }
            | IntUnary { .. }
            | IntBin { .. }
            | F64Unary { .. }
            | F64Bin { .. }
            | CmpInt { .. }
            | CmpF64 { .. }
            | RefEq { .. }
            | NumToInt32 { .. }
            | NumToF64 { .. }
            | Int32ToNum { .. }
            | F64ToNum { .. }
            | Await { .. }
            | MakeFiber { .. }
            | FiberCall { .. }
            | FiberTry { .. }
            | FiberTransfer { .. }
            | FiberTransferError { .. }
            | FiberYield { .. }
            | FiberSuspend
            | ReadBarrier { .. } => 1,
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Inst::Jump { .. }
                | Inst::Branch { .. }
                | Inst::Switch { .. }
                | Inst::Return(_)
                | Inst::Bailout { .. }
                | Inst::Unreachable
                | Inst::FiberAbort { .. }
        )
    }

    pub fn edges(&self) -> SmallVec<[&Edge; 2]> {
        match self {
            Inst::Jump { edge } => SmallVec::from_slice(&[edge]),
            Inst::Branch {
                then_edge,
                else_edge,
                ..
            } => SmallVec::from_slice(&[then_edge, else_edge]),
            Inst::Switch {
                default_edge,
                cases,
                ..
            } => {
                let mut edges = SmallVec::<[&Edge; 2]>::with_capacity(cases.len() + 1);
                edges.push(default_edge);
                edges.extend(cases.iter().map(|case| &case.edge));
                edges
            }
            _ => SmallVec::new(),
        }
    }

    pub fn operands(&self) -> SmallVec<[ValueId; 4]> {
        use Inst::*;

        let mut out = SmallVec::<[ValueId; 4]>::new();
        match self {
            Jump { edge } => out.extend_from_slice(&edge.args),
            Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                out.push(*cond);
                out.extend_from_slice(&then_edge.args);
                out.extend_from_slice(&else_edge.args);
            }
            Switch {
                tag,
                default_edge,
                cases,
            } => {
                out.push(*tag);
                out.extend_from_slice(&default_edge.args);
                for case in cases {
                    out.extend_from_slice(&case.edge.args);
                }
            }
            Return(value) => out.push(*value),
            Unreachable | Safepoint | OSRPoint | Const(_) | FiberSuspend | AllocMap => {}
            Move(value) => out.push(*value),
            GuardType { value, .. }
            | GuardClass { value, .. }
            | GuardNotNull { value, .. }
            | GuardInt32 { value, .. }
            | GuardF64 { value, .. }
            | IntUnary { value, .. }
            | F64Unary { value, .. }
            | NumToInt32 { value }
            | NumToF64 { value }
            | Int32ToNum { value }
            | F64ToNum { value }
            | Await { promise: value }
            | FiberYield { value }
            | ReadBarrier { object: value } => out.push(*value),
            GuardBounds { index, length, .. } => {
                out.push(*index);
                out.push(*length);
            }
            Bailout { .. } => {}
            Invoke {
                receiver, args, ..
            }
            | InvokeSuper {
                receiver, args, ..
            } => {
                out.push(*receiver);
                out.extend_from_slice(args);
            }
            CallDirect { args, .. }
            | CallPrimitive { args, .. }
            | CallForeign { args, .. } => out.extend_from_slice(args),
            CallClosure { closure, args, .. } => {
                out.push(*closure);
                out.extend_from_slice(args);
            }
            CreateClass { superclass, .. } => {
                if let Some(superclass) = superclass {
                    out.push(*superclass);
                }
            }
            BindMethod { class, .. } | BindForeignMethod { class, .. } => out.push(*class),
            AllocInstance { .. } => {}
            AllocForeign { size, .. } => out.push(*size),
            LoadField { object, .. } => out.push(*object),
            StoreField { object, value, .. } => {
                out.push(*object);
                out.push(*value);
            }
            LoadStaticField { class, .. } => out.push(*class),
            StoreStaticField { class, value, .. } => {
                out.push(*class);
                out.push(*value);
            }
            IsInstance { value, class } => {
                out.push(*value);
                out.push(*class);
            }
            ImportModule { .. } => {}
            ImportVariable { module, .. } => out.push(*module),
            LoadModuleVar { .. } => {}
            StoreModuleVar { value, .. } => out.push(*value),
            AllocCell { initial } => out.push(*initial),
            LoadCell { cell } => out.push(*cell),
            StoreCell { cell, value } => {
                out.push(*cell);
                out.push(*value);
            }
            MakeClosure { captures, .. } => out.extend_from_slice(captures),
            AllocList { capacity } => {
                if let Some(capacity) = capacity {
                    out.push(*capacity);
                }
            }
            ListLen { list } => out.push(*list),
            ListGet { list, index } | ListRemoveAt { list, index } => {
                out.push(*list);
                out.push(*index);
            }
            ListSet { list, index, value } | ListInsert { list, index, value } => {
                out.push(*list);
                out.push(*index);
                out.push(*value);
            }
            ListAppend { list, value } => {
                out.push(*list);
                out.push(*value);
            }
            MapLen { map } => out.push(*map),
            MapGet { map, key } | MapRemove { map, key } => {
                out.push(*map);
                out.push(*key);
            }
            MapSet { map, key, value } => {
                out.push(*map);
                out.push(*key);
                out.push(*value);
            }
            StringLen { string } => out.push(*string),
            StringConcat { left, right }
            | IntBin { left, right, .. }
            | F64Bin { left, right, .. }
            | CmpInt { left, right, .. }
            | CmpF64 { left, right, .. }
            | RefEq { left, right } => {
                out.push(*left);
                out.push(*right);
            }
            StringSlice {
                string,
                start,
                count,
            } => {
                out.push(*string);
                out.push(*start);
                out.push(*count);
            }
            StringCodePointAt { string, index } => {
                out.push(*string);
                out.push(*index);
            }
            MakeRange { from, to, .. } => {
                out.push(*from);
                out.push(*to);
            }
            MakeFiber { closure } => out.push(*closure),
            FiberCall { fiber, argument }
            | FiberTry { fiber, argument }
            | FiberTransfer { fiber, argument } => {
                out.push(*fiber);
                out.push(*argument);
            }
            FiberTransferError { fiber, error } => {
                out.push(*fiber);
                out.push(*error);
            }
            FiberAbort { error } => out.push(*error),
            WriteBarrier { object, value } => {
                out.push(*object);
                out.push(*value);
            }
        }
        out
    }

    /// Visits every data SSA operand in-place.
    ///
    /// This is the mutation counterpart to `operands()` and is intentionally
    /// centralized here so optimizer passes do not need to duplicate the full
    /// instruction-shape match every time they perform RAUW/copy propagation.
    pub fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut ValueId)) {
        use Inst::*;

        match self {
            Jump { edge } => {
                for value in &mut edge.args {
                    f(value);
                }
            }
            Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                f(cond);
                for value in &mut then_edge.args {
                    f(value);
                }
                for value in &mut else_edge.args {
                    f(value);
                }
            }
            Switch {
                tag,
                default_edge,
                cases,
            } => {
                f(tag);
                for value in &mut default_edge.args {
                    f(value);
                }
                for case in cases {
                    for value in &mut case.edge.args {
                        f(value);
                    }
                }
            }
            Return(value) => f(value),
            Unreachable | Safepoint | OSRPoint | Const(_) | FiberSuspend | AllocMap => {}
            Move(value) => f(value),
            GuardType { value, .. }
            | GuardClass { value, .. }
            | GuardNotNull { value, .. }
            | GuardInt32 { value, .. }
            | GuardF64 { value, .. }
            | IntUnary { value, .. }
            | F64Unary { value, .. }
            | NumToInt32 { value }
            | NumToF64 { value }
            | Int32ToNum { value }
            | F64ToNum { value }
            | Await { promise: value }
            | FiberYield { value }
            | ReadBarrier { object: value } => f(value),
            GuardBounds { index, length, .. } => {
                f(index);
                f(length);
            }
            Bailout { .. } => {}
            Invoke {
                receiver, args, ..
            }
            | InvokeSuper {
                receiver, args, ..
            } => {
                f(receiver);
                for value in args {
                    f(value);
                }
            }
            CallDirect { args, .. }
            | CallPrimitive { args, .. }
            | CallForeign { args, .. } => {
                for value in args {
                    f(value);
                }
            }
            CallClosure { closure, args, .. } => {
                f(closure);
                for value in args {
                    f(value);
                }
            }
            CreateClass { superclass, .. } => {
                if let Some(superclass) = superclass {
                    f(superclass);
                }
            }
            BindMethod { class, .. } | BindForeignMethod { class, .. } => f(class),
            AllocInstance { .. } => {}
            AllocForeign { size, .. } => f(size),
            LoadField { object, .. } => f(object),
            StoreField { object, value, .. } => {
                f(object);
                f(value);
            }
            LoadStaticField { class, .. } => f(class),
            StoreStaticField { class, value, .. } => {
                f(class);
                f(value);
            }
            IsInstance { value, class } => {
                f(value);
                f(class);
            }
            ImportModule { .. } => {}
            ImportVariable { module, .. } => f(module),
            LoadModuleVar { .. } => {}
            StoreModuleVar { value, .. } => f(value),
            AllocCell { initial } => f(initial),
            LoadCell { cell } => f(cell),
            StoreCell { cell, value } => {
                f(cell);
                f(value);
            }
            MakeClosure { captures, .. } => {
                for value in captures {
                    f(value);
                }
            }
            AllocList { capacity } => {
                if let Some(capacity) = capacity {
                    f(capacity);
                }
            }
            ListLen { list } => f(list),
            ListGet { list, index } | ListRemoveAt { list, index } => {
                f(list);
                f(index);
            }
            ListSet { list, index, value } | ListInsert { list, index, value } => {
                f(list);
                f(index);
                f(value);
            }
            ListAppend { list, value } => {
                f(list);
                f(value);
            }
            MapLen { map } => f(map),
            MapGet { map, key } | MapRemove { map, key } => {
                f(map);
                f(key);
            }
            MapSet { map, key, value } => {
                f(map);
                f(key);
                f(value);
            }
            StringLen { string } => f(string),
            StringConcat { left, right }
            | IntBin { left, right, .. }
            | F64Bin { left, right, .. }
            | CmpInt { left, right, .. }
            | CmpF64 { left, right, .. }
            | RefEq { left, right } => {
                f(left);
                f(right);
            }
            StringSlice {
                string,
                start,
                count,
            } => {
                f(string);
                f(start);
                f(count);
            }
            StringCodePointAt { string, index } => {
                f(string);
                f(index);
            }
            MakeRange { from, to, .. } => {
                f(from);
                f(to);
            }
            MakeFiber { closure } => f(closure),
            FiberCall { fiber, argument }
            | FiberTry { fiber, argument }
            | FiberTransfer { fiber, argument } => {
                f(fiber);
                f(argument);
            }
            FiberTransferError { fiber, error } => {
                f(fiber);
                f(error);
            }
            FiberAbort { error } => f(error),
            WriteBarrier { object, value } => {
                f(object);
                f(value);
            }
        }
    }

    /// Replaces all occurrences of one SSA data operand in this instruction.
    #[inline]
    pub fn replace_operand(&mut self, from: ValueId, to: ValueId) -> usize {
        if from == to {
            return 0;
        }

        let mut replaced = 0usize;
        self.for_each_operand_mut(|operand| {
            if *operand == from {
                *operand = to;
                replaced += 1;
            }
        });
        replaced
    }

}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: InstId,
    pub kind: Inst,
    pub results: Vec<ValueId>,
    pub effect_in: Option<EffectId>,
    pub effect_out: Option<EffectId>,
    pub source: Option<SourceSpan>,
}

#[derive(Clone, Debug)]
pub struct Value {
    pub id: ValueId,
    pub ty: Type,
    pub def: ValueDef,
}

#[derive(Clone, Copy, Debug)]
pub enum ValueDef {
    InstResult(InstId, u8),
    BlockParam(BlockId, u16),
    CaptureParam(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    ByValue,
    /// The captured value is a `Type::Cell` and mutations go through the cell.
    ByCell,
}

#[derive(Clone, Debug)]
pub struct CaptureParam {
    pub value: ValueId,
    pub ty: Type,
    pub mode: CaptureMode,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub id: BlockId,
    pub params: Vec<ValueId>,
    pub effect: EffectId,
    pub insts: Vec<InstId>,
    pub sealed: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CallSiteProfile {
    pub receiver_classes: Vec<(ClassId, u64)>,
    pub targets: Vec<(FuncId, u64)>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NumericProfile {
    pub int32_hits: u64,
    pub f64_hits: u64,
    pub other_hits: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ProfileData {
    pub block_counters: HashMap<BlockId, u64>,
    pub call_sites: HashMap<SiteId, CallSiteProfile>,
    pub numeric_sites: HashMap<InstId, NumericProfile>,
}

/// Zero-cost exception region used by structured `try`/`catch` and deferred cleanup.
///
/// `protected_blocks` are ordinary SSA blocks whose throwing runtime operations
/// transfer control to `handler`. `error` is a block parameter of `handler`; the
/// bytecode backend assigns it a physical register and the interpreter writes the
/// thrown Wren value there on handler entry. `depth` resolves nested overlapping
/// regions without putting enter/leave opcodes on the hot path.
#[derive(Clone, Debug)]
pub struct ExceptionRegion {
    pub protected_blocks: Vec<BlockId>,
    pub handler: BlockId,
    pub error: ValueId,
    pub depth: u16,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub id: FuncId,
    pub module: ModuleId,
    pub kind: FunctionKind,
    /// WrenJIT extension: async functions return a Promise when called and may
    /// contain `Await` instructions. The ordinary FunctionKind remains unchanged
    /// so method/receiver ABI rules stay identical.
    pub is_async: bool,
    /// Class whose instance-field layout this function was compiled against.
    ///
    /// This is set for methods and for closures lexically nested inside methods.
    /// Their FieldSlot operands are local to this declaring class and are
    /// relocated to absolute offsets when the runtime class is linked.
    pub field_owner: Option<ClassId>,
    pub signature: Signature,
    pub entry: BlockId,
    pub blocks: Vec<Block>,
    pub nodes: Vec<Node>,
    pub values: Vec<Value>,
    pub effects: Vec<EffectValue>,
    pub captures: Vec<CaptureParam>,
    pub deopts: Vec<DeoptPoint>,
    /// Structured exception metadata. It is side-table-only: normal execution
    /// pays no bytecode dispatch cost merely for being inside a try/defer region.
    pub exception_regions: Vec<ExceptionRegion>,
    pub profiles: ProfileData,
}

impl Function {
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id.0 as usize).filter(|block| block.id == id)
    }

    pub fn node(&self, id: InstId) -> Option<&Node> {
        self.nodes.get(id.0 as usize).filter(|node| node.id == id)
    }

    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.0 as usize).filter(|value| value.id == id)
    }

    pub fn effect(&self, id: EffectId) -> Option<&EffectValue> {
        self.effects.get(id.0 as usize).filter(|effect| effect.id == id)
    }
}

#[derive(Default)]
pub struct Program {
    next_func: u32,
    next_module: u32,
    next_class: u32,
    next_site: u32,
    next_foreign: u32,
    next_primitive: u32,
    next_resume: u32,

    pub functions: Vec<Function>,
    function_index: HashMap<FuncId, usize>,
    pub modules: Vec<ModuleInfo>,
    module_index: HashMap<ModuleId, usize>,
    pub classes: Vec<ClassInfo>,
    class_index: HashMap<ClassId, usize>,
    pub foreign_functions: Vec<ForeignFunctionInfo>,

    strings: Vec<String>,
    string_index: HashMap<String, StringId>,
    methods: Vec<MethodSignature>,
    method_index: HashMap<MethodSignature, MethodSymbol>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_string(&mut self, value: impl AsRef<str>) -> StringId {
        let value = value.as_ref();
        if let Some(id) = self.string_index.get(value) {
            return *id;
        }

        let id = StringId(self.strings.len() as u32);
        let owned = value.to_owned();
        self.strings.push(owned.clone());
        self.string_index.insert(owned, id);
        id
    }

    pub fn string(&self, id: StringId) -> Option<&str> {
        self.strings.get(id.0 as usize).map(String::as_str)
    }

    pub fn intern_method(
        &mut self,
        name: impl AsRef<str>,
        kind: MethodKind,
        arity: u8,
    ) -> MethodSymbol {
        let name = self.intern_string(name);
        let signature = MethodSignature { name, kind, arity };
        if let Some(symbol) = self.method_index.get(&signature) {
            return *symbol;
        }

        let symbol = MethodSymbol(self.methods.len() as u32);
        self.methods.push(signature);
        self.method_index.insert(signature, symbol);
        symbol
    }

    pub fn method(&self, symbol: MethodSymbol) -> Option<MethodSignature> {
        self.methods.get(symbol.0 as usize).copied()
    }

    pub fn reserve_function_id(&mut self) -> FuncId {
        let id = FuncId(self.next_func);
        self.next_func = self.next_func.checked_add(1).expect("function id overflow");
        id
    }

    pub fn push_function(&mut self, function: Function) -> FuncId {
        let id = function.id;
        assert!(
            !self.function_index.contains_key(&id),
            "function id {id:?} is already defined"
        );
        let index = self.functions.len();
        self.functions.push(function);
        self.function_index.insert(id, index);
        id
    }

    pub fn function(&self, id: FuncId) -> Option<&Function> {
        self.function_index
            .get(&id)
            .and_then(|index| self.functions.get(*index))
    }

    pub fn add_module(&mut self, name: impl AsRef<str>, variable_count: u32) -> ModuleId {
        let id = ModuleId(self.next_module);
        self.next_module = self.next_module.checked_add(1).expect("module id overflow");
        let info = ModuleInfo {
            id,
            name: self.intern_string(name),
            variable_count,
            builtin_bindings: Vec::new(),
        };
        let index = self.modules.len();
        self.modules.push(info);
        self.module_index.insert(id, index);
        id
    }

    pub fn module(&self, id: ModuleId) -> Option<&ModuleInfo> {
        self.module_index
            .get(&id)
            .and_then(|index| self.modules.get(*index))
    }

    pub fn set_module_builtin_bindings(&mut self, id: ModuleId, bindings: Vec<BuiltinBinding>) {
        let index = *self.module_index.get(&id).unwrap_or_else(|| panic!("unknown module {id:?}"));
        let module = &mut self.modules[index];
        for binding in &bindings {
            assert!(binding.slot.0 < module.variable_count, "builtin module slot {:?} out of range for {:?}", binding.slot, id);
        }
        module.builtin_bindings = bindings;
    }

    pub fn add_class(
        &mut self,
        module: ModuleId,
        name: impl AsRef<str>,
        superclass: Option<ClassId>,
        own_instance_field_count: u16,
        static_field_count: u16,
        is_foreign: bool,
    ) -> ClassId {
        let id = ClassId(self.next_class);
        self.next_class = self.next_class.checked_add(1).expect("class id overflow");
        let info = ClassInfo {
            id,
            module,
            name: self.intern_string(name),
            superclass,
            own_instance_field_count,
            static_field_count,
            is_foreign,
        };
        let index = self.classes.len();
        self.classes.push(info);
        self.class_index.insert(id, index);
        id
    }

    pub fn class(&self, id: ClassId) -> Option<&ClassInfo> {
        self.class_index
            .get(&id)
            .and_then(|index| self.classes.get(*index))
    }

    pub fn fresh_site(&mut self) -> SiteId {
        let id = SiteId(self.next_site);
        self.next_site = self.next_site.checked_add(1).expect("site id overflow");
        id
    }

    pub fn fresh_resume_point(&mut self) -> ResumePoint {
        let id = ResumeId(self.next_resume);
        self.next_resume = self
            .next_resume
            .checked_add(1)
            .expect("resume id overflow");
        ResumePoint { id }
    }

    pub fn fresh_primitive(&mut self) -> PrimitiveId {
        let id = PrimitiveId(self.next_primitive);
        self.next_primitive = self
            .next_primitive
            .checked_add(1)
            .expect("primitive id overflow");
        id
    }

    pub fn add_foreign_function(
        &mut self,
        module: impl AsRef<str>,
        class: impl AsRef<str>,
        signature: MethodSymbol,
        is_static: bool,
    ) -> ForeignFnId {
        let id = ForeignFnId(self.next_foreign);
        self.next_foreign = self
            .next_foreign
            .checked_add(1)
            .expect("foreign function id overflow");
        let module = self.intern_string(module);
        let class = self.intern_string(class);
        self.foreign_functions.push(ForeignFunctionInfo {
            id,
            module,
            class,
            signature,
            is_static,
        });
        id
    }
}

pub struct Builder<'p> {
    program: &'p mut Program,
    function: Function,
    current_block: BlockId,
    current_effect: EffectId,
    current_source: Option<SourceSpan>,
}

impl<'p> Builder<'p> {
    pub fn new(
        program: &'p mut Program,
        module: ModuleId,
        name: impl AsRef<str>,
        kind: FunctionKind,
        params: Vec<Type>,
        result: Type,
    ) -> Self {
        assert!(program.module(module).is_some(), "unknown module {module:?}");

        let function_id = program.reserve_function_id();
        let name = program.intern_string(name);
        let entry = BlockId(0);
        let entry_effect = EffectId(0);

        let mut function = Function {
            id: function_id,
            module,
            kind,
            is_async: false,
            field_owner: None,
            signature: Signature {
                name,
                params: params.clone(),
                result,
            },
            entry,
            blocks: Vec::new(),
            nodes: Vec::new(),
            values: Vec::new(),
            effects: vec![EffectValue {
                id: entry_effect,
                def: EffectDef::Entry,
            }],
            captures: Vec::new(),
            deopts: Vec::new(),
            exception_regions: Vec::new(),
            profiles: ProfileData::default(),
        };

        let mut block_params = Vec::with_capacity(params.len());
        for (index, ty) in params.into_iter().enumerate() {
            let value = ValueId(function.values.len() as u32);
            function.values.push(Value {
                id: value,
                ty,
                def: ValueDef::BlockParam(entry, index as u16),
            });
            block_params.push(value);
        }

        function.blocks.push(Block {
            id: entry,
            params: block_params,
            effect: entry_effect,
            insts: Vec::new(),
            sealed: false,
        });

        Self {
            program,
            function,
            current_block: entry,
            current_effect: entry_effect,
            current_source: None,
        }
    }

    /// Marks this function as using instance-field slots local to `class`.
    ///
    /// Runtime class linking converts those local slots to absolute fixed-layout
    /// offsets exactly once, after the actual superclass is known.
    pub fn set_field_owner(&mut self, class: ClassId) {
        self.function.field_owner = Some(class);
    }

    pub fn function_id(&self) -> FuncId {
        self.function.id
    }

    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    pub fn current_effect(&self) -> EffectId {
        self.current_effect
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.function.blocks.len()
    }

    #[inline]
    pub fn block_param(&self, block: BlockId, index: usize) -> ValueId {
        self.function
            .block(block)
            .and_then(|block| block.params.get(index))
            .copied()
            .expect("unknown block parameter")
    }

    /// Register a zero-cost structured exception region.
    ///
    /// The handler itself must not be protected by this same region. Nested
    /// outer regions may protect the handler normally.
    pub fn add_exception_region(
        &mut self,
        mut protected_blocks: Vec<BlockId>,
        handler: BlockId,
        error: ValueId,
        depth: u16,
    ) {
        protected_blocks.sort_unstable();
        protected_blocks.dedup();
        assert!(self.function.block(handler).is_some(), "unknown exception handler block");
        assert!(
            !protected_blocks.contains(&handler),
            "exception handler cannot protect itself"
        );
        let error_value = self.function.value(error).expect("unknown exception value");
        assert!(
            matches!(error_value.def, ValueDef::BlockParam(block, _) if block == handler),
            "exception value must be a handler block parameter"
        );
        self.function.exception_regions.push(ExceptionRegion {
            protected_blocks,
            handler,
            error,
            depth,
        });
    }

    pub fn set_source(&mut self, source: Option<SourceSpan>) {
        self.current_source = source;
    }

    pub fn param(&self, index: usize) -> ValueId {
        self.function.blocks[self.function.entry.0 as usize].params[index]
    }

    pub fn capture(&self, index: usize) -> ValueId {
        self.function.captures[index].value
    }

    pub fn add_capture(&mut self, ty: Type, mode: CaptureMode) -> ValueId {
        assert!(
            self.function.nodes.is_empty(),
            "captures must be declared before emitting instructions"
        );
        if mode == CaptureMode::ByCell {
            assert!(
                matches!(&ty, Type::Cell(_)),
                "ByCell captures must have Type::Cell"
            );
        }

        let value = ValueId(self.function.values.len() as u32);
        let index = self.function.captures.len() as u16;
        self.function.values.push(Value {
            id: value,
            ty: ty.clone(),
            def: ValueDef::CaptureParam(index),
        });
        self.function.captures.push(CaptureParam {
            value,
            ty,
            mode,
        });
        value
    }

    pub fn new_block(&mut self, param_types: &[Type]) -> BlockId {
        let block = BlockId(self.function.blocks.len() as u32);
        let effect = self.new_effect(EffectDef::BlockParam(block));
        let mut params = Vec::with_capacity(param_types.len());

        for (index, ty) in param_types.iter().cloned().enumerate() {
            let value = ValueId(self.function.values.len() as u32);
            self.function.values.push(Value {
                id: value,
                ty,
                def: ValueDef::BlockParam(block, index as u16),
            });
            params.push(value);
        }

        self.function.blocks.push(Block {
            id: block,
            params,
            effect,
            insts: Vec::new(),
            sealed: false,
        });
        block
    }

    pub fn add_block_param(&mut self, block: BlockId, ty: Type) -> ValueId {
        let block_ref = self.function.block(block).expect("unknown block");
        assert!(!block_ref.sealed, "cannot add parameters to sealed block {block:?}");
        let index = block_ref.params.len();
        assert!(index <= u16::MAX as usize, "too many block parameters");

        let value = ValueId(self.function.values.len() as u32);
        self.function.values.push(Value {
            id: value,
            ty,
            def: ValueDef::BlockParam(block, index as u16),
        });
        self.function.blocks[block.0 as usize].params.push(value);
        value
    }

    pub fn seal_block(&mut self, block: BlockId) {
        self.function.blocks[block.0 as usize].sealed = true;
    }

    pub fn position_at(&mut self, block: BlockId) {
        let block_ref = self.function.block(block).expect("unknown block");
        self.current_block = block;
        self.current_effect = block_ref
            .insts
            .iter()
            .rev()
            .filter_map(|inst| self.function.node(*inst))
            .find_map(|node| node.effect_out)
            .unwrap_or(block_ref.effect);
    }

    pub fn block_terminated(&self, block: BlockId) -> bool {
        self.function
            .block(block)
            .and_then(|block| block.insts.last())
            .and_then(|inst| self.function.node(*inst))
            .map(|node| node.kind.is_terminator())
            .unwrap_or(false)
    }

    pub fn emit0(&mut self, kind: Inst) -> InstId {
        let (inst, results) = self.emit(kind, &[]);
        debug_assert!(results.is_empty());
        inst
    }

    pub fn emit1(&mut self, kind: Inst, ty: Type) -> ValueId {
        let (_, results) = self.emit(kind, &[ty]);
        results[0]
    }

    pub fn emit(&mut self, kind: Inst, result_types: &[Type]) -> (InstId, Vec<ValueId>) {
        assert!(
            !self.block_terminated(self.current_block),
            "cannot emit after terminator in {:?}",
            self.current_block
        );
        assert!(
            result_types.len() <= u8::MAX as usize,
            "an instruction cannot produce more than 255 SSA values"
        );
        assert_eq!(
            result_types.len(),
            kind.result_count(),
            "wrong result count for instruction {kind:?}"
        );

        let id = InstId(self.function.nodes.len() as u32);
        let effect_behavior = kind.effect_behavior();
        let (effect_in, effect_out) = match effect_behavior {
            EffectBehavior::None => (None, None),
            EffectBehavior::Chain(kind) => (
                Some(self.current_effect),
                Some(self.new_effect(EffectDef::InstResult(id, kind))),
            ),
            EffectBehavior::Sink(_) => (Some(self.current_effect), None),
        };

        let mut results = Vec::with_capacity(result_types.len());
        for (index, ty) in result_types.iter().cloned().enumerate() {
            let value = ValueId(self.function.values.len() as u32);
            self.function.values.push(Value {
                id: value,
                ty,
                def: ValueDef::InstResult(id, index as u8),
            });
            results.push(value);
        }

        self.function.nodes.push(Node {
            id,
            kind,
            results: results.clone(),
            effect_in,
            effect_out,
            source: self.current_source,
        });
        self.function.blocks[self.current_block.0 as usize]
            .insts
            .push(id);

        if let Some(effect) = effect_out {
            self.current_effect = effect;
        }

        (id, results)
    }

    pub fn edge(&self, target: BlockId, args: impl Into<Vec<ValueId>>) -> Edge {
        let args = args.into();
        let target_block = self.function.block(target).expect("unknown edge target");
        assert_eq!(
            args.len(),
            target_block.params.len(),
            "wrong argument count for edge to {target:?}"
        );
        for (index, (&arg, &param)) in args.iter().zip(&target_block.params).enumerate() {
            let arg_ty = &self.function.value(arg).expect("unknown edge argument").ty;
            let param_ty = &self
                .function
                .value(param)
                .expect("broken target block parameter")
                .ty;
            assert!(
                param_ty.accepts(arg_ty),
                "edge argument {index} to {target:?} has type {arg_ty:?}, expected {param_ty:?}"
            );
        }

        Edge {
            target,
            args,
            effect: self.current_effect,
        }
    }

    pub fn jump(&mut self, target: BlockId, args: impl Into<Vec<ValueId>>) {
        let edge = self.edge(target, args);
        self.emit0(Inst::Jump { edge });
    }

    pub fn branch(
        &mut self,
        cond: ValueId,
        then_block: BlockId,
        then_args: impl Into<Vec<ValueId>>,
        else_block: BlockId,
        else_args: impl Into<Vec<ValueId>>,
    ) {
        let then_edge = self.edge(then_block, then_args);
        let else_edge = self.edge(else_block, else_args);
        self.emit0(Inst::Branch {
            cond,
            then_edge,
            else_edge,
        });
    }

    pub fn switch(
        &mut self,
        tag: ValueId,
        default_block: BlockId,
        default_args: impl Into<Vec<ValueId>>,
        cases: impl IntoIterator<Item = (i64, BlockId, Vec<ValueId>)>,
    ) {
        let default_edge = self.edge(default_block, default_args);
        let cases = cases
            .into_iter()
            .map(|(value, target, args)| SwitchCase {
                value,
                edge: self.edge(target, args),
            })
            .collect();
        self.emit0(Inst::Switch {
            tag,
            default_edge,
            cases,
        });
    }

    pub fn ret(&mut self, value: ValueId) {
        let actual = &self.function.value(value).expect("unknown return value").ty;
        assert!(
            self.function.signature.result.accepts(actual),
            "return type {actual:?} is not accepted by {:?}",
            self.function.signature.result
        );
        self.emit0(Inst::Return(value));
    }

    pub fn deopt_point(&mut self, snapshot: DeoptSnapshot) -> DeoptId {
        let id = DeoptId(self.function.deopts.len() as u32);
        self.function.deopts.push(DeoptPoint { id, snapshot });
        id
    }

    pub fn fresh_site(&mut self) -> SiteId {
        self.program.fresh_site()
    }

    pub fn const_null(&mut self) -> ValueId {
        self.emit1(Inst::Const(Constant::Null), Type::Null)
    }

    pub fn const_bool(&mut self, value: bool) -> ValueId {
        self.emit1(Inst::Const(Constant::Bool(value)), Type::Bool)
    }

    pub fn const_num(&mut self, value: f64) -> ValueId {
        self.emit1(Inst::Const(Constant::Num(value)), Type::Num)
    }

    pub fn const_i32(&mut self, value: i32) -> ValueId {
        self.emit1(Inst::Const(Constant::Int32(value)), Type::Int32)
    }

    pub fn const_string(&mut self, value: impl AsRef<str>) -> ValueId {
        let string = self.program.intern_string(value);
        self.emit1(Inst::Const(Constant::String(string)), Type::String)
    }

    pub fn intern_method(&mut self, name: &str, kind: MethodKind, arity: u8) -> MethodSymbol {
        self.program.intern_method(name, kind, arity)
    }

    pub fn invoke(
        &mut self,
        receiver: ValueId,
        method: MethodSymbol,
        args: Vec<ValueId>,
    ) -> ValueId {
        let site = self.program.fresh_site();
        self.emit1(
            Inst::Invoke {
                site,
                receiver,
                method,
                args,
            },
            Type::Top,
        )
    }

    pub fn invoke_super(
        &mut self,
        home_class: ClassId,
        receiver: ValueId,
        method: MethodSymbol,
        args: Vec<ValueId>,
    ) -> ValueId {
        let site = self.program.fresh_site();
        self.emit1(
            Inst::InvokeSuper {
                site,
                home_class,
                receiver,
                method,
                args,
            },
            Type::Top,
        )
    }

    pub fn alloc_instance(&mut self, class: ClassId) -> ValueId {
        self.emit1(
            Inst::AllocInstance { class },
            Type::Instance(ClassSet::singleton(class)),
        )
    }

    pub fn load_field(&mut self, object: ValueId, slot: FieldSlot, ty: Type) -> ValueId {
        self.emit1(Inst::LoadField { object, slot }, ty)
    }

    pub fn store_field(&mut self, object: ValueId, slot: FieldSlot, value: ValueId) {
        self.emit0(Inst::StoreField {
            object,
            slot,
            value,
        });
    }

    pub fn alloc_cell(&mut self, initial: ValueId, inner_type: Type) -> ValueId {
        self.emit1(
            Inst::AllocCell { initial },
            Type::Cell(Box::new(inner_type)),
        )
    }

    pub fn load_cell(&mut self, cell: ValueId, ty: Type) -> ValueId {
        self.emit1(Inst::LoadCell { cell }, ty)
    }

    pub fn store_cell(&mut self, cell: ValueId, value: ValueId) {
        self.emit0(Inst::StoreCell { cell, value });
    }

    pub fn make_closure(&mut self, function: FuncId, captures: Vec<ValueId>) -> ValueId {
        self.emit1(
            Inst::MakeClosure { function, captures },
            Type::Closure(Some(function)),
        )
    }

    pub fn guard_class(
        &mut self,
        value: ValueId,
        expected: ClassSet,
        refined_type: Type,
        deopt: DeoptId,
    ) -> ValueId {
        self.emit1(
            Inst::GuardClass {
                value,
                expected,
                deopt,
            },
            refined_type,
        )
    }

    pub fn guard_i32(&mut self, value: ValueId, deopt: DeoptId) -> ValueId {
        self.emit1(Inst::GuardInt32 { value, deopt }, Type::Int32)
    }

    pub fn guard_f64(&mut self, value: ValueId, deopt: DeoptId) -> ValueId {
        self.emit1(Inst::GuardF64 { value, deopt }, Type::F64)
    }

    /// Marks this function as an async entrypoint. Calls to it produce a Promise
    /// and execute the body on the VM async scheduler.
    pub fn set_async(&mut self, is_async: bool) {
        self.function.is_async = is_async;
    }

    pub fn finish(self) -> FuncId {
        for (index, node) in self.function.nodes.iter().enumerate() {
            debug_assert_eq!(node.id.0 as usize, index);
        }
        for (index, value) in self.function.values.iter().enumerate() {
            debug_assert_eq!(value.id.0 as usize, index);
        }
        for (index, effect) in self.function.effects.iter().enumerate() {
            debug_assert_eq!(effect.id.0 as usize, index);
        }

        let id = self.function.id;
        let function = self.function;
        self.program.push_function(function);
        id
    }

    fn new_effect(&mut self, def: EffectDef) -> EffectId {
        let id = EffectId(self.function.effects.len() as u32);
        self.function.effects.push(EffectValue { id, def });
        id
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    InvalidEntryBlock(BlockId),
    InvalidBlockId { index: usize, id: BlockId },
    InvalidNodeId { index: usize, id: InstId },
    InvalidValueId { index: usize, id: ValueId },
    InvalidEffectId { index: usize, id: EffectId },
    InvalidInstructionReference { block: BlockId, inst: InstId },
    InstructionListedMultipleTimes(InstId),
    InstructionNotInBlock(InstId),
    InstructionAfterTerminator { block: BlockId, inst: InstId },
    MissingTerminator(BlockId),
    InvalidValueReference { inst: InstId, value: ValueId },
    InvalidEffectReference { inst: InstId, effect: EffectId },
    InvalidEdgeTarget { inst: InstId, target: BlockId },
    EdgeArgumentCount {
        inst: InstId,
        target: BlockId,
        expected: usize,
        actual: usize,
    },
    EdgeArgumentType {
        inst: InstId,
        target: BlockId,
        index: usize,
        expected: Type,
        actual: Type,
    },
    InvalidEdgeEffect { inst: InstId, effect: EffectId },
    BrokenEffectChain {
        block: BlockId,
        inst: InstId,
        expected: EffectId,
        actual: Option<EffectId>,
    },
    UnexpectedEffectOutput { inst: InstId },
    MissingEffectOutput { inst: InstId },
    BrokenValueDefinition(ValueId),
    BrokenEffectDefinition(EffectId),
    InvalidDeoptId { inst: InstId, deopt: DeoptId },
    InvalidExceptionHandler { handler: BlockId },
    InvalidExceptionBlock { block: BlockId },
    InvalidExceptionValue { handler: BlockId, value: ValueId },
    ExceptionHandlerSelfProtected { handler: BlockId },
    ResultCount { inst: InstId, expected: usize, actual: usize },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for VerifyError {}

pub fn verify_function(function: &Function) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();

    if function.block(function.entry).is_none() {
        errors.push(VerifyError::InvalidEntryBlock(function.entry));
    }

    for (index, block) in function.blocks.iter().enumerate() {
        if block.id.0 as usize != index {
            errors.push(VerifyError::InvalidBlockId {
                index,
                id: block.id,
            });
        }
    }
    for (index, node) in function.nodes.iter().enumerate() {
        if node.id.0 as usize != index {
            errors.push(VerifyError::InvalidNodeId { index, id: node.id });
        }
    }
    for (index, value) in function.values.iter().enumerate() {
        if value.id.0 as usize != index {
            errors.push(VerifyError::InvalidValueId {
                index,
                id: value.id,
            });
        }
    }
    for (index, effect) in function.effects.iter().enumerate() {
        if effect.id.0 as usize != index {
            errors.push(VerifyError::InvalidEffectId {
                index,
                id: effect.id,
            });
        }
    }

    for region in &function.exception_regions {
        if function.block(region.handler).is_none() {
            errors.push(VerifyError::InvalidExceptionHandler {
                handler: region.handler,
            });
            continue;
        }
        if region.protected_blocks.contains(&region.handler) {
            errors.push(VerifyError::ExceptionHandlerSelfProtected {
                handler: region.handler,
            });
        }
        for &block in &region.protected_blocks {
            if function.block(block).is_none() {
                errors.push(VerifyError::InvalidExceptionBlock { block });
            }
        }
        let valid_error = function.value(region.error).is_some_and(|value| {
            matches!(value.def, ValueDef::BlockParam(block, _) if block == region.handler)
        });
        if !valid_error {
            errors.push(VerifyError::InvalidExceptionValue {
                handler: region.handler,
                value: region.error,
            });
        }
    }

    let mut owner: Vec<Option<BlockId>> = vec![None; function.nodes.len()];

    for block in &function.blocks {
        let Some(block_effect) = function.effect(block.effect) else {
            errors.push(VerifyError::InvalidEffectReference {
                inst: InstId(u32::MAX),
                effect: block.effect,
            });
            continue;
        };

        let valid_block_effect = if block.id == function.entry {
            matches!(block_effect.def, EffectDef::Entry)
        } else {
            matches!(block_effect.def, EffectDef::BlockParam(id) if id == block.id)
        };
        if !valid_block_effect {
            errors.push(VerifyError::BrokenEffectDefinition(block.effect));
        }

        let mut current_effect = block.effect;
        let mut terminated = false;

        for &inst_id in &block.insts {
            let Some(node) = function.node(inst_id) else {
                errors.push(VerifyError::InvalidInstructionReference {
                    block: block.id,
                    inst: inst_id,
                });
                continue;
            };

            if owner[inst_id.0 as usize].replace(block.id).is_some() {
                errors.push(VerifyError::InstructionListedMultipleTimes(inst_id));
            }

            if terminated {
                errors.push(VerifyError::InstructionAfterTerminator {
                    block: block.id,
                    inst: inst_id,
                });
            }

            for operand in node.kind.operands() {
                if function.value(operand).is_none() {
                    errors.push(VerifyError::InvalidValueReference {
                        inst: inst_id,
                        value: operand,
                    });
                }
            }

            let expected_results = node.kind.result_count();
            if node.results.len() != expected_results {
                errors.push(VerifyError::ResultCount {
                    inst: inst_id,
                    expected: expected_results,
                    actual: node.results.len(),
                });
            }

            match node.kind.effect_behavior() {
                EffectBehavior::Chain(kind) => {
                    if node.effect_in != Some(current_effect) {
                        errors.push(VerifyError::BrokenEffectChain {
                            block: block.id,
                            inst: inst_id,
                            expected: current_effect,
                            actual: node.effect_in,
                        });
                    }

                    let Some(effect_out) = node.effect_out else {
                        errors.push(VerifyError::MissingEffectOutput { inst: inst_id });
                        continue;
                    };
                    let Some(effect_value) = function.effect(effect_out) else {
                        errors.push(VerifyError::InvalidEffectReference {
                            inst: inst_id,
                            effect: effect_out,
                        });
                        continue;
                    };
                    if !matches!(
                        effect_value.def,
                        EffectDef::InstResult(def_inst, def_kind)
                            if def_inst == inst_id && def_kind == kind
                    ) {
                        errors.push(VerifyError::BrokenEffectDefinition(effect_out));
                    }
                    current_effect = effect_out;
                }
                EffectBehavior::Sink(_) => {
                    if node.effect_in != Some(current_effect) {
                        errors.push(VerifyError::BrokenEffectChain {
                            block: block.id,
                            inst: inst_id,
                            expected: current_effect,
                            actual: node.effect_in,
                        });
                    }
                    if node.effect_out.is_some() {
                        errors.push(VerifyError::UnexpectedEffectOutput { inst: inst_id });
                    }
                }
                EffectBehavior::None => {
                    if node.effect_in.is_some() || node.effect_out.is_some() {
                        errors.push(VerifyError::UnexpectedEffectOutput { inst: inst_id });
                    }
                }
            }

            for edge in node.kind.edges() {
                verify_edge(function, node.id, edge, current_effect, &mut errors);
            }

            verify_deopt_reference(function, node, &mut errors);

            if node.kind.is_terminator() {
                terminated = true;
            }
        }

        if !terminated {
            errors.push(VerifyError::MissingTerminator(block.id));
        }
    }

    // `nodes`, `values`, and `effects` are append-only optimizer arenas.
    // A node not listed by any block is a detached/dead node left behind by an
    // optimization. Live instruction order and liveness are defined exclusively
    // by `Block::insts`. Duplicate live ownership is still rejected above.
    //
    // Keeping stable IDs avoids global renumbering after every DCE/peephole edit.
    let _ = owner;

    for value in &function.values {
        let valid = match value.def {
            ValueDef::InstResult(inst, result_index) => function
                .node(inst)
                .and_then(|node| node.results.get(result_index as usize))
                .is_some_and(|id| *id == value.id),
            ValueDef::BlockParam(block, param_index) => function
                .block(block)
                .and_then(|block| block.params.get(param_index as usize))
                .is_some_and(|id| *id == value.id),
            ValueDef::CaptureParam(capture_index) => function
                .captures
                .get(capture_index as usize)
                .is_some_and(|capture| capture.value == value.id),
        };

        if !valid {
            errors.push(VerifyError::BrokenValueDefinition(value.id));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn verify_edge(
    function: &Function,
    inst: InstId,
    edge: &Edge,
    current_effect: EffectId,
    errors: &mut Vec<VerifyError>,
) {
    let Some(target) = function.block(edge.target) else {
        errors.push(VerifyError::InvalidEdgeTarget {
            inst,
            target: edge.target,
        });
        return;
    };

    if edge.effect != current_effect {
        errors.push(VerifyError::InvalidEdgeEffect {
            inst,
            effect: edge.effect,
        });
    }

    if edge.args.len() != target.params.len() {
        errors.push(VerifyError::EdgeArgumentCount {
            inst,
            target: edge.target,
            expected: target.params.len(),
            actual: edge.args.len(),
        });
        return;
    }

    for (index, (&arg, &param)) in edge.args.iter().zip(&target.params).enumerate() {
        let Some(arg_value) = function.value(arg) else {
            continue;
        };
        let Some(param_value) = function.value(param) else {
            continue;
        };

        if !param_value.ty.accepts(&arg_value.ty) {
            errors.push(VerifyError::EdgeArgumentType {
                inst,
                target: edge.target,
                index,
                expected: param_value.ty.clone(),
                actual: arg_value.ty.clone(),
            });
        }
    }
}

fn verify_deopt_reference(function: &Function, node: &Node, errors: &mut Vec<VerifyError>) {
    let deopt = match &node.kind {
        Inst::GuardType { deopt, .. }
        | Inst::GuardClass { deopt, .. }
        | Inst::GuardBounds { deopt, .. }
        | Inst::GuardNotNull { deopt, .. }
        | Inst::GuardInt32 { deopt, .. }
        | Inst::GuardF64 { deopt, .. }
        | Inst::Bailout { deopt, .. } => Some(*deopt),
        _ => None,
    };

    if let Some(deopt) = deopt {
        let valid = function
            .deopts
            .get(deopt.0 as usize)
            .is_some_and(|point| point.id == deopt);
        if !valid {
            errors.push(VerifyError::InvalidDeoptId {
                inst: node.id,
                deopt,
            });
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "func {:?} {:?}{}({:?}) -> {:?}",
            self.id, self.kind, if self.is_async { " async" } else { "" }, self.signature.params, self.signature.result
        )?;

        if !self.captures.is_empty() {
            writeln!(f, "  captures:")?;
            for (index, capture) in self.captures.iter().enumerate() {
                writeln!(
                    f,
                    "    c{} = {:?} : {:?} ({:?})",
                    index, capture.value, capture.ty, capture.mode
                )?;
            }
        }

        for block in &self.blocks {
            write!(f, "{:?}(", block.id)?;
            for (index, param) in block.params.iter().enumerate() {
                if index != 0 {
                    write!(f, ", ")?;
                }
                let ty = self
                    .value(*param)
                    .map(|value| value.ty.clone())
                    .unwrap_or(Type::Top);
                write!(f, "%{:?}: {:?}", param, ty)?;
            }
            writeln!(f, ") [effect {:?}]:", block.effect)?;

            for inst in &block.insts {
                let node = &self.nodes[inst.0 as usize];
                if !node.results.is_empty() {
                    for (index, result) in node.results.iter().enumerate() {
                        if index != 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "%{:?}", result)?;
                    }
                    write!(f, " = ")?;
                } else {
                    write!(f, "  ")?;
                }

                write!(f, "{:?}", node.kind)?;
                match (node.effect_in, node.effect_out) {
                    (Some(effect_in), Some(effect_out)) => {
                        write!(f, "  [E {:?} -> {:?}]", effect_in, effect_out)?;
                    }
                    (Some(effect_in), None) => {
                        write!(f, "  [E {:?} -> sink]", effect_in)?;
                    }
                    (None, Some(effect_out)) => {
                        write!(f, "  [E root -> {:?}]", effect_out)?;
                    }
                    (None, None) => {}
                }
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effects_are_threaded_across_cfg_edges() {
        let mut program = Program::new();
        let module = program.add_module("main", 0);
        let class = program.add_class(module, "Box", None, 1, 0, false);

        let function = {
            let mut builder = Builder::new(
                &mut program,
                module,
                "test",
                FunctionKind::ModuleBody,
                vec![],
                Type::Null,
            );

            let cond = builder.const_bool(true);
            let object = builder.alloc_instance(class);
            let one = builder.const_num(1.0);
            builder.store_field(object, FieldSlot(0), one);

            let then_block = builder.new_block(&[Type::Num]);
            let else_block = builder.new_block(&[Type::Num]);
            let join = builder.new_block(&[Type::Num]);

            builder.branch(
                cond,
                then_block,
                vec![one],
                else_block,
                vec![one],
            );

            builder.position_at(then_block);
            let loaded_then = builder.load_field(object, FieldSlot(0), Type::Num);
            builder.jump(join, vec![loaded_then]);

            builder.position_at(else_block);
            let loaded_else = builder.load_field(object, FieldSlot(0), Type::Num);
            builder.jump(join, vec![loaded_else]);

            builder.position_at(join);
            let null = builder.const_null();
            builder.ret(null);

            builder.finish()
        };

        let function = program.function(function).unwrap();
        if let Err(errors) = verify_function(function) {
            panic!("IR verification failed: {errors:#?}\n{function}");
        }
    }

    #[test]
    fn closure_capture_cells_are_explicit() {
        let mut program = Program::new();
        let module = program.add_module("main", 0);

        let inner_id = {
            let mut inner = Builder::new(
                &mut program,
                module,
                "inner",
                FunctionKind::Closure,
                vec![],
                Type::Num,
            );
            let cell = inner.add_capture(
                Type::Cell(Box::new(Type::Num)),
                CaptureMode::ByCell,
            );
            let value = inner.load_cell(cell, Type::Num);
            inner.ret(value);
            inner.finish()
        };

        let outer_id = {
            let mut outer = Builder::new(
                &mut program,
                module,
                "outer",
                FunctionKind::ModuleBody,
                vec![],
                Type::Closure(Some(inner_id)),
            );
            let initial = outer.const_num(42.0);
            let cell = outer.alloc_cell(initial, Type::Num);
            let closure = outer.make_closure(inner_id, vec![cell]);
            outer.ret(closure);
            outer.finish()
        };

        verify_function(program.function(inner_id).unwrap()).unwrap();
        verify_function(program.function(outer_id).unwrap()).unwrap();
    }
}
