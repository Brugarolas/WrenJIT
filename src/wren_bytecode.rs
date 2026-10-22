//! Wren register bytecode — executable format for the baseline VM.
//!
//! This module is the lowering target of `wren_ir.rs`. It deliberately keeps
//! Wren semantics visible where the interpreter benefits from them (method
//! dispatch, fibers, classes, modules) while giving hot primitive operations a
//! compact one-word encoding.
//!
//! Architectural register model:
//! - 256 frame-local general registers (`R0..R255`).
//! - 16 call-transfer argument registers (`ARG0..ARG15`).
//! - 2 call-transfer return registers (`RET0..RET1`).
//! - A small special-value register bank (`THIS`, current closure/module/fiber,
//!   error/resume values, and compiler/runtime scratch registers).
//! - Interpreter machine-state registers are described separately and are not
//!   directly addressable by bytecode.
//!
//! Encoding model:
//! - Code is a stream of little-endian logical 32-bit `Word`s.
//! - The first word of every instruction is `[opcode, A, B, C]`, one byte each.
//! - Hot instructions are exactly one word.
//! - Wide IDs/immediates use following extension words.
//! - Calls and closure construction append packed GPR lists (4 register IDs per
//!   word), so argument setup is performed inside one interpreter dispatch.
//! - Relative branches are measured in words from the end of the branch.
//!
//! General registers are raw 64-bit runtime slots. A slot can contain a tagged
//! Wren value or an internal unboxed representation (`i32`, `f64`, raw pointer)
//! as dictated by the opcode and lowering metadata. Precise GC uses per-PC stack
//! maps, not conservative inspection of all 256 registers.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::identity_op)]
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::wren_ir::{
    BailoutReason, CallConv, ClassId, ClassInfo, ClassSet, CmpKind, Constant, DeoptId,
    F64BinOp, F64UnaryOp, ForeignFnId, ForeignFunctionInfo, FuncId,
    FunctionKind, IntBinOp, IntUnaryOp, MethodSignature, MethodSymbol, ModuleId,
    ModuleInfo, ModuleSlot, PrimitiveId, ResumeId, SiteId, SourceSpan,
    StringId, Type,
};

// =============================================================================
// Architectural registers
// =============================================================================

pub const GENERAL_REGISTER_COUNT: usize = 256;
pub const ARGUMENT_REGISTER_COUNT: usize = 16;
pub const RETURN_REGISTER_COUNT: usize = 2;
pub const MAX_WREN_ARITY: usize = 16;
pub const MAX_DIRECT_FRAME_VALUES: usize = MAX_WREN_ARITY + 1; // THIS + 16 args.
pub const MAX_CLOSURE_CAPTURES: usize = u16::MAX as usize;

/// Frame-local general-purpose register.
///
/// All 256 byte values are valid register numbers, including `R255`; no GPR
/// value is reserved as a sentinel.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Reg(pub u8);

impl Reg {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R{}", self.0)
    }
}

pub const R0: Reg = Reg(0);
pub const R1: Reg = Reg(1);
pub const R2: Reg = Reg(2);
pub const R3: Reg = Reg(3);
pub const R4: Reg = Reg(4);
pub const R5: Reg = Reg(5);
pub const R6: Reg = Reg(6);
pub const R7: Reg = Reg(7);
pub const R8: Reg = Reg(8);
pub const R9: Reg = Reg(9);
pub const R10: Reg = Reg(10);
pub const R11: Reg = Reg(11);
pub const R12: Reg = Reg(12);
pub const R13: Reg = Reg(13);
pub const R14: Reg = Reg(14);
pub const R15: Reg = Reg(15);
pub const R16: Reg = Reg(16);
pub const R17: Reg = Reg(17);
pub const R18: Reg = Reg(18);
pub const R19: Reg = Reg(19);
pub const R20: Reg = Reg(20);
pub const R21: Reg = Reg(21);
pub const R22: Reg = Reg(22);
pub const R23: Reg = Reg(23);
pub const R24: Reg = Reg(24);
pub const R25: Reg = Reg(25);
pub const R26: Reg = Reg(26);
pub const R27: Reg = Reg(27);
pub const R28: Reg = Reg(28);
pub const R29: Reg = Reg(29);
pub const R30: Reg = Reg(30);
pub const R31: Reg = Reg(31);
pub const R32: Reg = Reg(32);
pub const R33: Reg = Reg(33);
pub const R34: Reg = Reg(34);
pub const R35: Reg = Reg(35);
pub const R36: Reg = Reg(36);
pub const R37: Reg = Reg(37);
pub const R38: Reg = Reg(38);
pub const R39: Reg = Reg(39);
pub const R40: Reg = Reg(40);
pub const R41: Reg = Reg(41);
pub const R42: Reg = Reg(42);
pub const R43: Reg = Reg(43);
pub const R44: Reg = Reg(44);
pub const R45: Reg = Reg(45);
pub const R46: Reg = Reg(46);
pub const R47: Reg = Reg(47);
pub const R48: Reg = Reg(48);
pub const R49: Reg = Reg(49);
pub const R50: Reg = Reg(50);
pub const R51: Reg = Reg(51);
pub const R52: Reg = Reg(52);
pub const R53: Reg = Reg(53);
pub const R54: Reg = Reg(54);
pub const R55: Reg = Reg(55);
pub const R56: Reg = Reg(56);
pub const R57: Reg = Reg(57);
pub const R58: Reg = Reg(58);
pub const R59: Reg = Reg(59);
pub const R60: Reg = Reg(60);
pub const R61: Reg = Reg(61);
pub const R62: Reg = Reg(62);
pub const R63: Reg = Reg(63);
pub const R64: Reg = Reg(64);
pub const R65: Reg = Reg(65);
pub const R66: Reg = Reg(66);
pub const R67: Reg = Reg(67);
pub const R68: Reg = Reg(68);
pub const R69: Reg = Reg(69);
pub const R70: Reg = Reg(70);
pub const R71: Reg = Reg(71);
pub const R72: Reg = Reg(72);
pub const R73: Reg = Reg(73);
pub const R74: Reg = Reg(74);
pub const R75: Reg = Reg(75);
pub const R76: Reg = Reg(76);
pub const R77: Reg = Reg(77);
pub const R78: Reg = Reg(78);
pub const R79: Reg = Reg(79);
pub const R80: Reg = Reg(80);
pub const R81: Reg = Reg(81);
pub const R82: Reg = Reg(82);
pub const R83: Reg = Reg(83);
pub const R84: Reg = Reg(84);
pub const R85: Reg = Reg(85);
pub const R86: Reg = Reg(86);
pub const R87: Reg = Reg(87);
pub const R88: Reg = Reg(88);
pub const R89: Reg = Reg(89);
pub const R90: Reg = Reg(90);
pub const R91: Reg = Reg(91);
pub const R92: Reg = Reg(92);
pub const R93: Reg = Reg(93);
pub const R94: Reg = Reg(94);
pub const R95: Reg = Reg(95);
pub const R96: Reg = Reg(96);
pub const R97: Reg = Reg(97);
pub const R98: Reg = Reg(98);
pub const R99: Reg = Reg(99);
pub const R100: Reg = Reg(100);
pub const R101: Reg = Reg(101);
pub const R102: Reg = Reg(102);
pub const R103: Reg = Reg(103);
pub const R104: Reg = Reg(104);
pub const R105: Reg = Reg(105);
pub const R106: Reg = Reg(106);
pub const R107: Reg = Reg(107);
pub const R108: Reg = Reg(108);
pub const R109: Reg = Reg(109);
pub const R110: Reg = Reg(110);
pub const R111: Reg = Reg(111);
pub const R112: Reg = Reg(112);
pub const R113: Reg = Reg(113);
pub const R114: Reg = Reg(114);
pub const R115: Reg = Reg(115);
pub const R116: Reg = Reg(116);
pub const R117: Reg = Reg(117);
pub const R118: Reg = Reg(118);
pub const R119: Reg = Reg(119);
pub const R120: Reg = Reg(120);
pub const R121: Reg = Reg(121);
pub const R122: Reg = Reg(122);
pub const R123: Reg = Reg(123);
pub const R124: Reg = Reg(124);
pub const R125: Reg = Reg(125);
pub const R126: Reg = Reg(126);
pub const R127: Reg = Reg(127);
pub const R128: Reg = Reg(128);
pub const R129: Reg = Reg(129);
pub const R130: Reg = Reg(130);
pub const R131: Reg = Reg(131);
pub const R132: Reg = Reg(132);
pub const R133: Reg = Reg(133);
pub const R134: Reg = Reg(134);
pub const R135: Reg = Reg(135);
pub const R136: Reg = Reg(136);
pub const R137: Reg = Reg(137);
pub const R138: Reg = Reg(138);
pub const R139: Reg = Reg(139);
pub const R140: Reg = Reg(140);
pub const R141: Reg = Reg(141);
pub const R142: Reg = Reg(142);
pub const R143: Reg = Reg(143);
pub const R144: Reg = Reg(144);
pub const R145: Reg = Reg(145);
pub const R146: Reg = Reg(146);
pub const R147: Reg = Reg(147);
pub const R148: Reg = Reg(148);
pub const R149: Reg = Reg(149);
pub const R150: Reg = Reg(150);
pub const R151: Reg = Reg(151);
pub const R152: Reg = Reg(152);
pub const R153: Reg = Reg(153);
pub const R154: Reg = Reg(154);
pub const R155: Reg = Reg(155);
pub const R156: Reg = Reg(156);
pub const R157: Reg = Reg(157);
pub const R158: Reg = Reg(158);
pub const R159: Reg = Reg(159);
pub const R160: Reg = Reg(160);
pub const R161: Reg = Reg(161);
pub const R162: Reg = Reg(162);
pub const R163: Reg = Reg(163);
pub const R164: Reg = Reg(164);
pub const R165: Reg = Reg(165);
pub const R166: Reg = Reg(166);
pub const R167: Reg = Reg(167);
pub const R168: Reg = Reg(168);
pub const R169: Reg = Reg(169);
pub const R170: Reg = Reg(170);
pub const R171: Reg = Reg(171);
pub const R172: Reg = Reg(172);
pub const R173: Reg = Reg(173);
pub const R174: Reg = Reg(174);
pub const R175: Reg = Reg(175);
pub const R176: Reg = Reg(176);
pub const R177: Reg = Reg(177);
pub const R178: Reg = Reg(178);
pub const R179: Reg = Reg(179);
pub const R180: Reg = Reg(180);
pub const R181: Reg = Reg(181);
pub const R182: Reg = Reg(182);
pub const R183: Reg = Reg(183);
pub const R184: Reg = Reg(184);
pub const R185: Reg = Reg(185);
pub const R186: Reg = Reg(186);
pub const R187: Reg = Reg(187);
pub const R188: Reg = Reg(188);
pub const R189: Reg = Reg(189);
pub const R190: Reg = Reg(190);
pub const R191: Reg = Reg(191);
pub const R192: Reg = Reg(192);
pub const R193: Reg = Reg(193);
pub const R194: Reg = Reg(194);
pub const R195: Reg = Reg(195);
pub const R196: Reg = Reg(196);
pub const R197: Reg = Reg(197);
pub const R198: Reg = Reg(198);
pub const R199: Reg = Reg(199);
pub const R200: Reg = Reg(200);
pub const R201: Reg = Reg(201);
pub const R202: Reg = Reg(202);
pub const R203: Reg = Reg(203);
pub const R204: Reg = Reg(204);
pub const R205: Reg = Reg(205);
pub const R206: Reg = Reg(206);
pub const R207: Reg = Reg(207);
pub const R208: Reg = Reg(208);
pub const R209: Reg = Reg(209);
pub const R210: Reg = Reg(210);
pub const R211: Reg = Reg(211);
pub const R212: Reg = Reg(212);
pub const R213: Reg = Reg(213);
pub const R214: Reg = Reg(214);
pub const R215: Reg = Reg(215);
pub const R216: Reg = Reg(216);
pub const R217: Reg = Reg(217);
pub const R218: Reg = Reg(218);
pub const R219: Reg = Reg(219);
pub const R220: Reg = Reg(220);
pub const R221: Reg = Reg(221);
pub const R222: Reg = Reg(222);
pub const R223: Reg = Reg(223);
pub const R224: Reg = Reg(224);
pub const R225: Reg = Reg(225);
pub const R226: Reg = Reg(226);
pub const R227: Reg = Reg(227);
pub const R228: Reg = Reg(228);
pub const R229: Reg = Reg(229);
pub const R230: Reg = Reg(230);
pub const R231: Reg = Reg(231);
pub const R232: Reg = Reg(232);
pub const R233: Reg = Reg(233);
pub const R234: Reg = Reg(234);
pub const R235: Reg = Reg(235);
pub const R236: Reg = Reg(236);
pub const R237: Reg = Reg(237);
pub const R238: Reg = Reg(238);
pub const R239: Reg = Reg(239);
pub const R240: Reg = Reg(240);
pub const R241: Reg = Reg(241);
pub const R242: Reg = Reg(242);
pub const R243: Reg = Reg(243);
pub const R244: Reg = Reg(244);
pub const R245: Reg = Reg(245);
pub const R246: Reg = Reg(246);
pub const R247: Reg = Reg(247);
pub const R248: Reg = Reg(248);
pub const R249: Reg = Reg(249);
pub const R250: Reg = Reg(250);
pub const R251: Reg = Reg(251);
pub const R252: Reg = Reg(252);
pub const R253: Reg = Reg(253);
pub const R254: Reg = Reg(254);
pub const R255: Reg = Reg(255);

pub const GENERAL_REGISTERS: [Reg; GENERAL_REGISTER_COUNT] = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61, R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76, R77, R78, R79, R80, R81, R82, R83, R84, R85, R86, R87, R88, R89, R90, R91, R92, R93, R94, R95, R96, R97, R98, R99, R100, R101, R102, R103, R104, R105, R106, R107, R108, R109, R110, R111, R112, R113, R114, R115, R116, R117, R118, R119, R120, R121, R122, R123, R124, R125, R126, R127, R128, R129, R130, R131, R132, R133, R134, R135, R136, R137, R138, R139, R140, R141, R142, R143, R144, R145, R146, R147, R148, R149, R150, R151, R152, R153, R154, R155, R156, R157, R158, R159, R160, R161, R162, R163, R164, R165, R166, R167, R168, R169, R170, R171, R172, R173, R174, R175, R176, R177, R178, R179, R180, R181, R182, R183, R184, R185, R186, R187, R188, R189, R190, R191, R192, R193, R194, R195, R196, R197, R198, R199, R200, R201, R202, R203, R204, R205, R206, R207, R208, R209, R210, R211, R212, R213, R214, R215, R216, R217, R218, R219, R220, R221, R222, R223, R224, R225, R226, R227, R228, R229, R230, R231, R232, R233, R234, R235, R236, R237, R238, R239, R240, R241, R242, R243, R244, R245, R246, R247, R248, R249, R250, R251, R252, R253, R254, R255];

/// Call-transfer argument register. These are caller-clobbered by the ABI.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ArgReg(pub u8);

impl ArgReg {
    pub const fn new(index: u8) -> Option<Self> {
        if index < ARGUMENT_REGISTER_COUNT as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for ArgReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ARG{}", self.0)
    }
}

pub const ARG0: ArgReg = ArgReg(0);
pub const ARG1: ArgReg = ArgReg(1);
pub const ARG2: ArgReg = ArgReg(2);
pub const ARG3: ArgReg = ArgReg(3);
pub const ARG4: ArgReg = ArgReg(4);
pub const ARG5: ArgReg = ArgReg(5);
pub const ARG6: ArgReg = ArgReg(6);
pub const ARG7: ArgReg = ArgReg(7);
pub const ARG8: ArgReg = ArgReg(8);
pub const ARG9: ArgReg = ArgReg(9);
pub const ARG10: ArgReg = ArgReg(10);
pub const ARG11: ArgReg = ArgReg(11);
pub const ARG12: ArgReg = ArgReg(12);
pub const ARG13: ArgReg = ArgReg(13);
pub const ARG14: ArgReg = ArgReg(14);
pub const ARG15: ArgReg = ArgReg(15);

pub const ARGUMENT_REGISTERS: [ArgReg; ARGUMENT_REGISTER_COUNT] = [ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11, ARG12, ARG13, ARG14, ARG15];

/// Call-transfer return register.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ReturnReg(pub u8);

impl ReturnReg {
    pub const fn new(index: u8) -> Option<Self> {
        if index < RETURN_REGISTER_COUNT as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for ReturnReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RET{}", self.0)
    }
}

pub const RET0: ReturnReg = ReturnReg(0);
pub const RET1: ReturnReg = ReturnReg(1);
pub const RETURN_REGISTERS: [ReturnReg; RETURN_REGISTER_COUNT] = [RET0, RET1];

/// Bytecode-visible special value registers.
///
/// `Scratch0..Scratch3` are reserved for lowering/runtime micro-sequences such
/// as cycle-breaking parallel copies. User register allocation never assigns
/// SSA values to them permanently.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    This = 0,
    Closure = 1,
    Module = 2,
    Fiber = 3,
    CallerFiber = 4,
    Error = 5,
    ResumeValue = 6,
    ForeignContext = 7,
    Scratch0 = 8,
    Scratch1 = 9,
    Scratch2 = 10,
    Scratch3 = 11,
}

pub const SPECIAL_REGISTER_COUNT: usize = 12;
pub const SPECIAL_REGISTERS: [SpecialReg; SPECIAL_REGISTER_COUNT] = [
    SpecialReg::This,
    SpecialReg::Closure,
    SpecialReg::Module,
    SpecialReg::Fiber,
    SpecialReg::CallerFiber,
    SpecialReg::Error,
    SpecialReg::ResumeValue,
    SpecialReg::ForeignContext,
    SpecialReg::Scratch0,
    SpecialReg::Scratch1,
    SpecialReg::Scratch2,
    SpecialReg::Scratch3,
];

impl SpecialReg {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::This),
            1 => Some(Self::Closure),
            2 => Some(Self::Module),
            3 => Some(Self::Fiber),
            4 => Some(Self::CallerFiber),
            5 => Some(Self::Error),
            6 => Some(Self::ResumeValue),
            7 => Some(Self::ForeignContext),
            8 => Some(Self::Scratch0),
            9 => Some(Self::Scratch1),
            10 => Some(Self::Scratch2),
            11 => Some(Self::Scratch3),
            _ => None,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn is_scratch(self) -> bool {
        matches!(
            self,
            Self::Scratch0 | Self::Scratch1 | Self::Scratch2 | Self::Scratch3
        )
    }
}

/// Interpreter implementation registers. These are architectural documentation
/// for the VM implementation, not operands in the bytecode stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineReg {
    Pc,
    Frame,
    StackTop,
    CodeBase,
    ConstantBase,
    InlineCacheBase,
    SafepointBase,
    CurrentFunction,
    CurrentModule,
    CurrentFiber,
    DispatchTable,
}

/// Raw storage used by all value/register banks. The runtime decides how tagged
/// values, unboxed numbers and pointers are represented in these 64 bits.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegisterWord(pub u64);

/// Concrete architectural register file. Frames normally own the 256 GPRs;
/// transfer/special banks may physically live on the fiber/VM depending on the
/// interpreter implementation, but their semantics are exactly those exposed
/// here.
#[derive(Clone)]
pub struct RegisterFile {
    pub general: [RegisterWord; GENERAL_REGISTER_COUNT],
    pub arguments: [RegisterWord; ARGUMENT_REGISTER_COUNT],
    pub returns: [RegisterWord; RETURN_REGISTER_COUNT],
    pub special: [RegisterWord; SPECIAL_REGISTER_COUNT],
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self {
            general: [RegisterWord::default(); GENERAL_REGISTER_COUNT],
            arguments: [RegisterWord::default(); ARGUMENT_REGISTER_COUNT],
            returns: [RegisterWord::default(); RETURN_REGISTER_COUNT],
            special: [RegisterWord::default(); SPECIAL_REGISTER_COUNT],
        }
    }
}

impl RegisterFile {
    #[inline]
    pub fn reg(&self, reg: Reg) -> RegisterWord {
        self.general[reg.index()]
    }

    #[inline]
    pub fn set_reg(&mut self, reg: Reg, value: RegisterWord) {
        self.general[reg.index()] = value;
    }

    #[inline]
    pub fn arg(&self, reg: ArgReg) -> RegisterWord {
        self.arguments[reg.index()]
    }

    #[inline]
    pub fn set_arg(&mut self, reg: ArgReg, value: RegisterWord) {
        self.arguments[reg.index()] = value;
    }

    #[inline]
    pub fn ret(&self, reg: ReturnReg) -> RegisterWord {
        self.returns[reg.index()]
    }

    #[inline]
    pub fn set_ret(&mut self, reg: ReturnReg, value: RegisterWord) {
        self.returns[reg.index()] = value;
    }

    #[inline]
    pub fn special(&self, reg: SpecialReg) -> RegisterWord {
        self.special[reg.index()]
    }

    #[inline]
    pub fn set_special(&mut self, reg: SpecialReg, value: RegisterWord) {
        self.special[reg.index()] = value;
    }
}


// =============================================================================
// Call-transfer ABI
// =============================================================================

/// Method-like functions receive their implicit receiver through `SpecialReg::This`.
/// Their IR signatures include that receiver as parameter 0, while the physical
/// ABI keeps it out of ARG0..ARG15 so all 16 Wren explicit arguments remain
/// available.
#[inline]
pub const fn function_kind_uses_this(kind: FunctionKind) -> bool {
    matches!(
        kind,
        FunctionKind::Method | FunctionKind::StaticMethod | FunctionKind::Constructor
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallAbiError {
    TooManyArguments(usize),
    MissingReceiver,
}

impl fmt::Display for CallAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyArguments(count) => {
                write!(f, "call supplies {count} explicit arguments; maximum is 16")
            }
            Self::MissingReceiver => write!(f, "method-like direct call has no receiver"),
        }
    }
}

impl std::error::Error for CallAbiError {}

/// Initialize a callee frame for a normal dynamic Wren invocation.
///
/// The receiver becomes `THIS`; explicit arguments become ARG0..ARG15. The
/// caller frame is untouched.
pub fn prepare_invoke_frame(
    caller: &RegisterFile,
    callee: &mut RegisterFile,
    receiver: Reg,
    args: &[Reg],
) -> Result<(), CallAbiError> {
    if args.len() > MAX_WREN_ARITY {
        return Err(CallAbiError::TooManyArguments(args.len()));
    }

    callee.set_special(SpecialReg::This, caller.reg(receiver));
    for (index, source) in args.iter().copied().enumerate() {
        callee.set_arg(ArgReg(index as u8), caller.reg(source));
    }
    Ok(())
}

/// Initialize a known direct callee from the packed complete IR call frame.
///
/// For method/static-method/constructor targets, `sources[0]` is the receiver
/// and the remainder are explicit arguments. Other function kinds receive every
/// source through ARG0..ARG15.
pub fn prepare_direct_frame(
    caller: &RegisterFile,
    callee: &mut RegisterFile,
    callee_kind: FunctionKind,
    sources: &[Reg],
) -> Result<(), CallAbiError> {
    let args = if function_kind_uses_this(callee_kind) {
        let (&receiver, args) = sources.split_first().ok_or(CallAbiError::MissingReceiver)?;
        callee.set_special(SpecialReg::This, caller.reg(receiver));
        args
    } else {
        sources
    };

    if args.len() > MAX_WREN_ARITY {
        return Err(CallAbiError::TooManyArguments(args.len()));
    }

    for (index, source) in args.iter().copied().enumerate() {
        callee.set_arg(ArgReg(index as u8), caller.reg(source));
    }
    Ok(())
}

/// Initialize a closure call. The closure object is held in the callee's
/// `CLOSURE` special register and explicit parameters use ARG0..ARG15.
pub fn prepare_closure_frame(
    caller: &RegisterFile,
    callee: &mut RegisterFile,
    closure: Reg,
    args: &[Reg],
) -> Result<(), CallAbiError> {
    if args.len() > MAX_WREN_ARITY {
        return Err(CallAbiError::TooManyArguments(args.len()));
    }

    callee.set_special(SpecialReg::Closure, caller.reg(closure));
    for (index, source) in args.iter().copied().enumerate() {
        callee.set_arg(ArgReg(index as u8), caller.reg(source));
    }
    Ok(())
}

/// Apply frame-entry bindings without executing bytecode prologue moves.
/// This is the normal path used when a bytecode frame is created.
pub fn initialize_frame_bindings(
    frame: &mut RegisterFile,
    layout: &FrameLayout,
    closure_captures: &[RegisterWord],
) -> Result<(), String> {
    layout.validate()?;

    for binding in &layout.parameters {
        let value = match binding.source {
            IncomingSource::This => frame.special(SpecialReg::This),
            IncomingSource::Argument(arg) => frame.arg(arg),
        };
        frame.set_reg(binding.destination, value);
    }

    for binding in &layout.captures {
        let value = closure_captures
            .get(binding.capture_index as usize)
            .copied()
            .ok_or_else(|| {
                format!(
                    "capture binding {} is outside closure capture array of length {}",
                    binding.capture_index,
                    closure_captures.len()
                )
            })?;
        frame.set_reg(binding.destination, value);
    }

    Ok(())
}

// =============================================================================
// Bytecode IDs and physical value kinds
// =============================================================================

macro_rules! bc_id {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);
    };
}

bc_id!(BytecodePc);
bc_id!(ConstIndex);
bc_id!(SwitchTableId);
bc_id!(IcSlot);
bc_id!(TypeGuardId);
bc_id!(ClassGuardId);
bc_id!(DeoptIndex);
bc_id!(SafepointIndex);
bc_id!(CounterId);

impl BytecodePc {
    pub const fn word_index(self) -> usize {
        self.0 as usize
    }
}

/// Physical representation currently stored in a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegisterKind {
    /// Normal tagged Wren value.
    Value,
    /// Internal signed 32-bit integer in the low 32 bits.
    Int32,
    /// Internal raw IEEE-754 `f64` bits.
    F64,
    /// Non-GC raw pointer/handle.
    RawPtr,
}

/// Where an incoming function parameter comes from before it is copied into a
/// frame-local GPR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IncomingSource {
    This,
    Argument(ArgReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParameterBinding {
    pub source: IncomingSource,
    pub destination: Reg,
    pub kind: RegisterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CaptureBinding {
    pub capture_index: u16,
    pub destination: Reg,
    pub kind: RegisterKind,
}

#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// Number of GPRs that are semantically part of the frame. `256` is valid.
    pub general_register_count: u16,
    pub parameters: Vec<ParameterBinding>,
    pub captures: Vec<CaptureBinding>,
    pub return_kind: RegisterKind,
}

impl FrameLayout {
    pub fn validate(&self) -> Result<(), String> {
        if self.general_register_count as usize > GENERAL_REGISTER_COUNT {
            return Err(format!(
                "frame declares {} GPRs, maximum is {}",
                self.general_register_count, GENERAL_REGISTER_COUNT
            ));
        }

        let mut destinations = [false; GENERAL_REGISTER_COUNT];

        for binding in &self.parameters {
            if binding.destination.index() >= self.general_register_count as usize {
                return Err(format!(
                    "parameter destination {:?} is outside {}-register frame",
                    binding.destination, self.general_register_count
                ));
            }
            if let IncomingSource::Argument(arg) = binding.source {
                if arg.index() >= ARGUMENT_REGISTER_COUNT {
                    return Err(format!("invalid incoming argument register {:?}", arg));
                }
            }
            if destinations[binding.destination.index()] {
                return Err(format!(
                    "multiple frame-entry bindings target {:?}",
                    binding.destination
                ));
            }
            destinations[binding.destination.index()] = true;
        }

        for binding in &self.captures {
            if binding.destination.index() >= self.general_register_count as usize {
                return Err(format!(
                    "capture destination {:?} is outside {}-register frame",
                    binding.destination, self.general_register_count
                ));
            }
            if destinations[binding.destination.index()] {
                return Err(format!(
                    "multiple frame-entry bindings target {:?}",
                    binding.destination
                ));
            }
            destinations[binding.destination.index()] = true;
        }

        Ok(())
    }
}

// =============================================================================
// 32-bit instruction words
// =============================================================================

/// Logical 32-bit bytecode word. Serialization should explicitly choose little
/// endian; the in-memory decoder only uses integer shifts and is host-endian safe.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Word(pub u32);

impl fmt::Debug for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08x}", self.0)
    }
}

impl Word {
    #[inline]
    pub const fn abc(opcode: Opcode, a: u8, b: u8, c: u8) -> Self {
        Self(
            (opcode as u32)
                | ((a as u32) << 8)
                | ((b as u32) << 16)
                | ((c as u32) << 24),
        )
    }

    #[inline]
    pub const fn abx(opcode: Opcode, a: u8, bx: u16) -> Self {
        Self::abc(opcode, a, bx as u8, (bx >> 8) as u8)
    }

    #[inline]
    pub const fn ax(opcode: Opcode, ax: u32) -> Self {
        Self((opcode as u32) | ((ax & 0x00ff_ffff) << 8))
    }

    #[inline]
    pub fn opcode(self) -> Option<Opcode> {
        Opcode::from_u8(self.0 as u8)
    }

    #[inline]
    pub const fn a(self) -> u8 {
        ((self.0 >> 8) & 0xff) as u8
    }

    #[inline]
    pub const fn b(self) -> u8 {
        ((self.0 >> 16) & 0xff) as u8
    }

    #[inline]
    pub const fn c(self) -> u8 {
        ((self.0 >> 24) & 0xff) as u8
    }

    #[inline]
    pub const fn bx(self) -> u16 {
        self.b() as u16 | ((self.c() as u16) << 8)
    }

    #[inline]
    pub const fn ax_value(self) -> u32 {
        (self.0 >> 8) & 0x00ff_ffff
    }
}

macro_rules! define_opcodes {
    ($( $name:ident = $value:literal ),+ $(,)?) => {
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Opcode {
            $( $name = $value, )+
        }

        impl Opcode {
            pub const fn from_u8(value: u8) -> Option<Self> {
                match value {
                    $( $value => Some(Self::$name), )+
                    _ => None,
                }
            }

            pub const fn name(self) -> &'static str {
                match self {
                    $( Self::$name => stringify!($name), )+
                }
            }
        }
    };
}

define_opcodes! {
    Nop = 0,
    Trap = 1,
    Move = 2,
    Swap = 3,
    LoadNull = 4,
    LoadFalse = 5,
    LoadTrue = 6,
    LoadI8 = 7,
    LoadI32 = 8,
    LoadConst = 9,
    MoveFromArg = 10,
    MoveToArg = 11,
    MoveFromRet = 12,
    MoveToRet = 13,
    MoveFromSpecial = 14,
    MoveToSpecial = 15,

    Jump = 16,
    JumpIfFalse = 17,
    JumpIfTrue = 18,
    Switch = 19,
    Return = 20,
    Return2 = 21,
    Safepoint = 22,
    OsrPoint = 23,

    GuardType = 24,
    GuardClass = 25,
    GuardBounds = 26,
    GuardNotNull = 27,
    GuardInt32 = 28,
    GuardF64 = 29,
    Bailout = 30,

    Invoke = 31,
    InvokeSuper = 32,
    CallDirect = 33,
    CallClosure = 34,
    CallPrimitive = 35,
    CallForeign = 36,

    CreateClass = 37,
    BindMethod = 38,
    BindForeignMethod = 39,
    AllocInstance = 40,
    AllocForeign = 41,
    LoadField = 42,
    StoreField = 43,
    LoadStaticField = 44,
    StoreStaticField = 45,
    IsInstance = 46,

    ImportModule = 47,
    ImportVariable = 48,
    LoadModuleVar = 49,
    StoreModuleVar = 50,

    AllocCell = 51,
    LoadCell = 52,
    StoreCell = 53,
    MakeClosure = 54,

    AllocList = 55,
    AllocListCap = 56,
    ListLen = 57,
    ListGet = 58,
    ListSet = 59,
    ListAppend = 60,
    ListInsert = 61,
    ListRemoveAt = 62,

    AllocMap = 63,
    MapLen = 64,
    MapGet = 65,
    MapSet = 66,
    MapRemove = 67,

    StringLen = 68,
    StringConcat = 69,
    StringSlice = 70,
    StringCodePointAt = 71,
    MakeRange = 72,

    INeg = 73,
    IBitNot = 74,
    IAdd = 75,
    ISub = 76,
    IMul = 77,
    IDivS = 78,
    IRemS = 79,
    IBitAnd = 80,
    IBitOr = 81,
    IBitXor = 82,
    IShl = 83,
    IShrU = 84,
    ISar = 85,

    FNeg = 86,
    FAdd = 87,
    FSub = 88,
    FMul = 89,
    FDiv = 90,
    FRem = 91,

    ICmpEq = 92,
    ICmpNe = 93,
    ICmpLt = 94,
    ICmpLe = 95,
    ICmpGt = 96,
    ICmpGe = 97,
    FCmpEq = 98,
    FCmpNe = 99,
    FCmpLt = 100,
    FCmpLe = 101,
    FCmpGt = 102,
    FCmpGe = 103,
    RefEq = 104,

    NumToInt32 = 105,
    NumToF64 = 106,
    Int32ToNum = 107,
    F64ToNum = 108,

    MakeFiber = 109,
    FiberCall = 110,
    FiberTry = 111,
    FiberTransfer = 112,
    FiberTransferError = 113,
    FiberYield = 114,
    FiberSuspend = 115,
    FiberAbort = 116,

    WriteBarrier = 117,
    ReadBarrier = 118,

    ProfileBlock = 119,
    ProfileNumeric = 120,
    Breakpoint = 121,

    // WrenJIT async extension. A=destination, B=Promise value.
    Await = 122,
}

impl Opcode {
    /// Number of words for fixed-size instructions. Variable-size instructions
    /// return `None`; use `instruction_word_len` with the first word instead.
    pub const fn fixed_word_len(self) -> Option<usize> {
        use Opcode::*;
        match self {
            Invoke | InvokeSuper | CallDirect | CallClosure | CallPrimitive | CallForeign
            | MakeClosure => None,

            LoadI32
            | LoadConst
            | Jump
            | JumpIfFalse
            | JumpIfTrue
            | Switch
            | GuardBounds
            | GuardNotNull
            | GuardInt32
            | GuardF64
            | Bailout
            | CreateClass
            | AllocInstance
            | AllocForeign
            | LoadField
            | StoreField
            | LoadStaticField
            | StoreStaticField
            | ImportModule
            | ImportVariable
            | StringSlice
            | MakeRange
            | ProfileBlock
            | ProfileNumeric => Some(2),

            GuardType
            | GuardClass
            | BindMethod
            | BindForeignMethod
            | LoadModuleVar
            | StoreModuleVar => Some(3),

            _ => Some(1),
        }
    }

    pub const fn is_terminator(self) -> bool {
        matches!(
            self,
            Self::Trap
                | Self::Jump
                | Self::Switch
                | Self::Return
                | Self::Return2
                | Self::Bailout
                | Self::FiberAbort
        )
    }

    /// Operations at which the runtime may allocate, invoke arbitrary code, or
    /// switch fibers. The lowering phase must emit a precise stack map for each.
    pub const fn may_gc(self) -> bool {
        use Opcode::*;
        matches!(
            self,
            Safepoint
                | Invoke
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
                | ImportModule
                | ImportVariable
                | AllocCell
                | MakeClosure
                | AllocList
                | AllocListCap
                | ListAppend
                | ListInsert
                | AllocMap
                | MapSet
                | StringConcat
                | StringSlice
                | MakeRange
                | MakeFiber
                | FiberCall
                | FiberTry
                | FiberTransfer
                | FiberTransferError
                | FiberYield
                | FiberSuspend
                | Await
        )
    }
}

#[inline]
pub const fn packed_register_word_count(count: usize) -> usize {
    (count + 3) / 4
}

/// Returns the total encoded instruction length in 32-bit words.
pub fn instruction_word_len(first: Word) -> Result<usize, DecodeError> {
    let opcode = first
        .opcode()
        .ok_or_else(|| DecodeError::new(None, format!("invalid opcode {}", first.0 as u8)))?;

    if let Some(fixed) = opcode.fixed_word_len() {
        return Ok(fixed);
    }

    use Opcode::*;
    let len = match opcode {
        Invoke | InvokeSuper => 2 + packed_register_word_count(first.c() as usize),
        CallDirect => 2 + packed_register_word_count(first.c() as usize),
        CallClosure => 2 + packed_register_word_count(first.c() as usize),
        CallPrimitive | CallForeign => 2 + packed_register_word_count(first.c() as usize),
        MakeClosure => 2 + packed_register_word_count(first.bx() as usize),
        _ => unreachable!("fixed-size opcode reached variable-size decoder"),
    };
    Ok(len)
}

/// Packs up to four GPR IDs into one extension word.
#[inline]
pub fn pack_four_registers(regs: &[Reg]) -> Word {
    debug_assert!(regs.len() <= 4);
    let mut value = 0u32;
    for (index, reg) in regs.iter().enumerate() {
        value |= (reg.0 as u32) << (index * 8);
    }
    Word(value)
}

pub fn pack_registers(regs: &[Reg]) -> Vec<Word> {
    regs.chunks(4).map(pack_four_registers).collect()
}

pub fn unpack_registers(words: &[Word], count: usize) -> Result<Vec<Reg>, DecodeError> {
    if words.len() < packed_register_word_count(count) {
        return Err(DecodeError::new(
            None,
            "truncated packed register list".to_owned(),
        ));
    }

    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let word = words[index / 4].0;
        let shift = (index % 4) * 8;
        out.push(Reg(((word >> shift) & 0xff) as u8));
    }
    Ok(out)
}

// =============================================================================
// Constant pools and side tables
// =============================================================================

#[derive(Clone, Debug)]
pub struct InlineCacheDescriptor {
    pub site: SiteId,
    pub method: MethodSymbol,
    /// Present only for `InvokeSuper`.
    pub home_class: Option<ClassId>,
}

#[derive(Clone, Debug)]
pub struct TypeGuardDescriptor {
    pub expected: Type,
}

#[derive(Clone, Debug)]
pub struct ClassGuardDescriptor {
    pub expected: ClassSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwitchTarget {
    pub value: i64,
    pub target: BytecodePc,
}

#[derive(Clone, Debug)]
pub struct SwitchTable {
    pub default_target: BytecodePc,
    pub cases: Vec<SwitchTarget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafepointKind {
    ExplicitPoll,
    Call,
    Allocation,
    Module,
    FiberSwitch,
    Foreign,
    RuntimeMutation,
}

/// Precise GC reference bitmap for one safepoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceMap {
    /// 256 bits, one per GPR.
    pub general: [u64; 4],
    /// 16 bits, one per ARG register.
    pub arguments: u16,
    /// Low two bits are RET0 and RET1.
    pub returns: u8,
    /// Low `SPECIAL_REGISTER_COUNT` bits are the special bank.
    pub special: u16,
}

impl Default for ReferenceMap {
    fn default() -> Self {
        Self {
            general: [0; 4],
            arguments: 0,
            returns: 0,
            special: 0,
        }
    }
}

impl ReferenceMap {
    pub fn set_general(&mut self, reg: Reg, is_reference: bool) {
        let word = reg.index() / 64;
        let bit = reg.index() % 64;
        if is_reference {
            self.general[word] |= 1u64 << bit;
        } else {
            self.general[word] &= !(1u64 << bit);
        }
    }

    pub fn general_is_reference(&self, reg: Reg) -> bool {
        let word = reg.index() / 64;
        let bit = reg.index() % 64;
        (self.general[word] & (1u64 << bit)) != 0
    }

    pub fn set_argument(&mut self, reg: ArgReg, is_reference: bool) {
        let mask = 1u16 << reg.index();
        if is_reference {
            self.arguments |= mask;
        } else {
            self.arguments &= !mask;
        }
    }

    pub fn set_return(&mut self, reg: ReturnReg, is_reference: bool) {
        let mask = 1u8 << reg.index();
        if is_reference {
            self.returns |= mask;
        } else {
            self.returns &= !mask;
        }
    }

    pub fn set_special(&mut self, reg: SpecialReg, is_reference: bool) {
        let mask = 1u16 << reg.index();
        if is_reference {
            self.special |= mask;
        } else {
            self.special &= !mask;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SafepointEntry {
    pub pc: BytecodePc,
    pub kind: SafepointKind,
    pub references: ReferenceMap,
}

/// Physical GPR liveness at one bytecode instruction boundary.
///
/// This is separate from `ReferenceMap`: tracing-JIT side exits must restore
/// every value that the baseline interpreter may read after resuming, including
/// unboxed numeric registers, while the GC map only tracks tagged references.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveRegisterMap {
    pub general: [u64; 4],
}

impl LiveRegisterMap {
    #[inline]
    pub fn set_general(&mut self, reg: Reg, live: bool) {
        let word = reg.index() / 64;
        let bit = reg.index() % 64;
        if live {
            self.general[word] |= 1u64 << bit;
        } else {
            self.general[word] &= !(1u64 << bit);
        }
    }

    #[inline]
    pub fn general_is_live(&self, reg: Reg) -> bool {
        let word = reg.index() / 64;
        let bit = reg.index() % 64;
        (self.general[word] & (1u64 << bit)) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRegisterEntry {
    pub pc: BytecodePc,
    pub live: LiveRegisterMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeoptLocation {
    General(Reg),
    Argument(ArgReg),
    Return(ReturnReg),
    Special(SpecialReg),
    Constant(ConstIndex),
    Null,
    True,
    False,
    Int32(i32),
    F64Bits(u64),
    Unavailable,
}

#[derive(Clone, Debug)]
pub struct BytecodeSnapshotSlot {
    pub slot: u32,
    pub location: DeoptLocation,
    pub kind: RegisterKind,
}

#[derive(Clone, Debug)]
pub struct BytecodeInlinedFrameState {
    pub function: FuncId,
    pub resume_pc: BytecodePc,
    pub slots: Vec<BytecodeSnapshotSlot>,
}

#[derive(Clone, Debug)]
pub struct BytecodeDeoptPoint {
    pub id: DeoptId,
    pub resume_pc: BytecodePc,
    pub slots: Vec<BytecodeSnapshotSlot>,
    pub inlined_frames: Vec<BytecodeInlinedFrameState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumeEntry {
    pub id: ResumeId,
    pub pc: BytecodePc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceMapEntry {
    pub start_pc: BytecodePc,
    pub end_pc: BytecodePc,
    pub source: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileCounterKind {
    Block,
    Numeric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileCounterDescriptor {
    pub kind: ProfileCounterKind,
    /// Optional original IR call/numeric site. Block counters normally use None.
    pub site: Option<SiteId>,
}

#[derive(Clone, Debug, Default)]
pub struct BytecodeChunk {
    pub code: Vec<Word>,
    pub constants: Vec<Constant>,
    pub inline_caches: Vec<InlineCacheDescriptor>,
    pub type_guards: Vec<TypeGuardDescriptor>,
    pub class_guards: Vec<ClassGuardDescriptor>,
    pub switch_tables: Vec<SwitchTable>,
    pub deopts: Vec<BytecodeDeoptPoint>,
    pub safepoints: Vec<SafepointEntry>,
    /// Sorted instruction-boundary liveness used by tracing-JIT side exits.
    pub live_registers: Vec<LiveRegisterEntry>,
    pub resume_points: Vec<ResumeEntry>,
    pub source_map: Vec<SourceMapEntry>,
    pub profile_counters: Vec<ProfileCounterDescriptor>,
}

impl BytecodeChunk {
    pub fn pc(&self) -> BytecodePc {
        BytecodePc(self.code.len() as u32)
    }

    pub fn push_constant(&mut self, constant: Constant) -> ConstIndex {
        let id = ConstIndex(self.constants.len() as u32);
        self.constants.push(constant);
        id
    }

    pub fn push_inline_cache(&mut self, descriptor: InlineCacheDescriptor) -> IcSlot {
        let id = IcSlot(self.inline_caches.len() as u32);
        self.inline_caches.push(descriptor);
        id
    }

    pub fn push_type_guard(&mut self, expected: Type) -> TypeGuardId {
        let id = TypeGuardId(self.type_guards.len() as u32);
        self.type_guards.push(TypeGuardDescriptor { expected });
        id
    }

    pub fn push_class_guard(&mut self, expected: ClassSet) -> ClassGuardId {
        let id = ClassGuardId(self.class_guards.len() as u32);
        self.class_guards.push(ClassGuardDescriptor { expected });
        id
    }

    pub fn push_profile_counter(
        &mut self,
        kind: ProfileCounterKind,
        site: Option<SiteId>,
    ) -> CounterId {
        let id = CounterId(self.profile_counters.len() as u32);
        self.profile_counters
            .push(ProfileCounterDescriptor { kind, site });
        id
    }

    pub fn safepoint_at(&self, pc: BytecodePc) -> Option<&SafepointEntry> {
        self.safepoints.iter().find(|entry| entry.pc == pc)
    }

    #[inline]
    pub fn live_registers_at(&self, pc: BytecodePc) -> Option<&LiveRegisterMap> {
        self.live_registers
            .binary_search_by_key(&pc.0, |entry| entry.pc.0)
            .ok()
            .map(|index| &self.live_registers[index].live)
    }

    pub fn deopt_by_id(&self, id: DeoptId) -> Option<&BytecodeDeoptPoint> {
        self.deopts.iter().find(|point| point.id == id)
    }
}

#[derive(Clone, Debug)]
pub struct BytecodeFunction {
    pub id: FuncId,
    pub module: ModuleId,
    pub kind: FunctionKind,
    /// Async functions return a Promise at call sites and execute on the VM task scheduler.
    pub is_async: bool,
    /// Declaring class for class-local instance FieldSlot operands.
    ///
    /// The interpreter relocates LoadField/StoreField operands in the prepared
    /// function once the actual runtime superclass layout is known.
    pub field_owner: Option<ClassId>,
    pub name: StringId,
    /// Number of explicit Wren parameters; the implicit receiver is not counted.
    pub arity: u8,
    /// Number of physical values stored in the closure environment.
    pub capture_count: u16,
    pub frame: FrameLayout,
    pub chunk: BytecodeChunk,
}

#[derive(Default)]
pub struct RuntimeMetadata {
    pub strings: Vec<String>,
    pub methods: Vec<MethodSignature>,
    pub modules: Vec<ModuleInfo>,
    pub classes: Vec<ClassInfo>,
    pub foreign_functions: Vec<ForeignFunctionInfo>,

    module_index: HashMap<ModuleId, usize>,
    class_index: HashMap<ClassId, usize>,
    foreign_index: HashMap<ForeignFnId, usize>,
}

impl RuntimeMetadata {
    pub fn rebuild_indexes(&mut self) -> Result<(), String> {
        self.module_index.clear();
        self.class_index.clear();
        self.foreign_index.clear();

        for (index, module) in self.modules.iter().enumerate() {
            if self.module_index.insert(module.id, index).is_some() {
                return Err(format!("duplicate module metadata {:?}", module.id));
            }
        }
        for (index, class) in self.classes.iter().enumerate() {
            if self.class_index.insert(class.id, index).is_some() {
                return Err(format!("duplicate class metadata {:?}", class.id));
            }
        }
        for (index, foreign) in self.foreign_functions.iter().enumerate() {
            if self.foreign_index.insert(foreign.id, index).is_some() {
                return Err(format!("duplicate foreign metadata {:?}", foreign.id));
            }
        }
        Ok(())
    }

    pub fn string(&self, id: StringId) -> Option<&str> {
        self.strings.get(id.0 as usize).map(String::as_str)
    }

    pub fn method(&self, id: MethodSymbol) -> Option<MethodSignature> {
        self.methods.get(id.0 as usize).copied()
    }

    pub fn module(&self, id: ModuleId) -> Option<&ModuleInfo> {
        self.module_index
            .get(&id)
            .and_then(|&index| self.modules.get(index))
            .or_else(|| self.modules.iter().find(|module| module.id == id))
    }

    pub fn class(&self, id: ClassId) -> Option<&ClassInfo> {
        self.class_index
            .get(&id)
            .and_then(|&index| self.classes.get(index))
            .or_else(|| self.classes.iter().find(|class| class.id == id))
    }

    pub fn foreign_function(&self, id: ForeignFnId) -> Option<&ForeignFunctionInfo> {
        self.foreign_index
            .get(&id)
            .and_then(|&index| self.foreign_functions.get(index))
            .or_else(|| self.foreign_functions.iter().find(|foreign| foreign.id == id))
    }
}

#[derive(Default)]
pub struct BytecodeProgram {
    pub functions: Vec<BytecodeFunction>,
    pub metadata: RuntimeMetadata,
    function_index: HashMap<FuncId, usize>,
}

impl BytecodeProgram {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rebuild_indexes(&mut self) -> Result<(), String> {
        self.function_index.clear();
        for (index, function) in self.functions.iter().enumerate() {
            if self.function_index.insert(function.id, index).is_some() {
                return Err(format!("duplicate bytecode function {:?}", function.id));
            }
        }
        self.metadata.rebuild_indexes()
    }

    pub fn push_function(&mut self, function: BytecodeFunction) -> Result<(), String> {
        if self.function_index.contains_key(&function.id) {
            return Err(format!("duplicate bytecode function {:?}", function.id));
        }
        let index = self.functions.len();
        self.function_index.insert(function.id, index);
        self.functions.push(function);
        Ok(())
    }

    pub fn function(&self, id: FuncId) -> Option<&BytecodeFunction> {
        self.function_index
            .get(&id)
            .and_then(|&index| self.functions.get(index))
            .or_else(|| self.functions.iter().find(|function| function.id == id))
    }
}

// =============================================================================
// Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct DecodeError {
    pub pc: Option<BytecodePc>,
    pub message: String,
}

impl DecodeError {
    pub fn new(pc: Option<BytecodePc>, message: String) -> Self {
        Self { pc, message }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pc {
            Some(pc) => write!(f, "bytecode decode error at word {}: {}", pc.0, self.message),
            None => write!(f, "bytecode decode error: {}", self.message),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Clone, Copy)]
pub struct DecodedInstruction<'a> {
    pub pc: BytecodePc,
    pub first: Word,
    pub opcode: Opcode,
    pub words: &'a [Word],
}

impl<'a> DecodedInstruction<'a> {
    pub fn len_words(self) -> usize {
        self.words.len()
    }

    pub fn next_pc(self) -> BytecodePc {
        BytecodePc(self.pc.0 + self.words.len() as u32)
    }

    pub fn extension(self, index: usize) -> Option<u32> {
        self.words.get(index + 1).map(|word| word.0)
    }

    pub fn call_argument_count(self) -> Option<usize> {
        match self.opcode {
            Opcode::Invoke
            | Opcode::InvokeSuper
            | Opcode::CallDirect
            | Opcode::CallClosure
            | Opcode::CallPrimitive
            | Opcode::CallForeign => Some(self.first.c() as usize),
            _ => None,
        }
    }

    pub fn packed_registers(self, count: usize, prefix_words: usize) -> Result<Vec<Reg>, DecodeError> {
        if self.words.len() < prefix_words {
            return Err(DecodeError::new(
                Some(self.pc),
                "instruction is shorter than packed-register prefix".to_owned(),
            ));
        }
        unpack_registers(&self.words[prefix_words..], count)
            .map_err(|mut error| {
                error.pc = Some(self.pc);
                error
            })
    }

    pub fn call_arguments(self) -> Result<Vec<Reg>, DecodeError> {
        let count = self.call_argument_count().ok_or_else(|| {
            DecodeError::new(Some(self.pc), "instruction is not a call".to_owned())
        })?;
        self.packed_registers(count, 2)
    }

    pub fn closure_captures(self) -> Result<Vec<Reg>, DecodeError> {
        if self.opcode != Opcode::MakeClosure {
            return Err(DecodeError::new(
                Some(self.pc),
                "instruction is not MakeClosure".to_owned(),
            ));
        }
        self.packed_registers(self.first.bx() as usize, 2)
    }
}

pub struct Decoder<'a> {
    code: &'a [Word],
    cursor: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(code: &'a [Word]) -> Self {
        Self { code, cursor: 0 }
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = Result<DecodedInstruction<'a>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.code.len() {
            return None;
        }

        let pc = BytecodePc(self.cursor as u32);
        let first = self.code[self.cursor];
        let opcode = match first.opcode() {
            Some(opcode) => opcode,
            None => {
                self.cursor = self.code.len();
                return Some(Err(DecodeError::new(
                    Some(pc),
                    format!("invalid opcode {}", first.0 as u8),
                )));
            }
        };

        let len = match instruction_word_len(first) {
            Ok(len) => len,
            Err(mut error) => {
                error.pc = Some(pc);
                self.cursor = self.code.len();
                return Some(Err(error));
            }
        };

        let end = match self.cursor.checked_add(len) {
            Some(end) if end <= self.code.len() => end,
            _ => {
                self.cursor = self.code.len();
                return Some(Err(DecodeError::new(
                    Some(pc),
                    format!("truncated {} instruction", opcode.name()),
                )));
            }
        };

        let words = &self.code[self.cursor..end];
        self.cursor = end;
        Some(Ok(DecodedInstruction {
            pc,
            first,
            opcode,
            words,
        }))
    }
}

// =============================================================================
// Assembler
// =============================================================================

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Clone, Copy, Debug)]
struct BranchFixup {
    extension_word: usize,
    instruction_end: usize,
    target: Label,
}

#[derive(Clone, Debug)]
struct PendingSwitchCase {
    value: i64,
    target: Label,
}

#[derive(Clone, Debug)]
struct PendingSwitch {
    table: SwitchTableId,
    default_target: Label,
    cases: Vec<PendingSwitchCase>,
}

#[derive(Debug)]
pub enum AssembleError {
    UnboundLabel(Label),
    BranchOutOfRange { from: usize, to: usize },
    InvalidCallArity(usize),
    TooManyClosureCaptures(usize),
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundLabel(label) => write!(f, "unbound bytecode label {:?}", label),
            Self::BranchOutOfRange { from, to } => {
                write!(f, "branch from word {from} to word {to} does not fit i32")
            }
            Self::InvalidCallArity(arity) => {
                write!(f, "invalid call argument count {arity}")
            }
            Self::TooManyClosureCaptures(count) => {
                write!(f, "closure has {count} captures; bytecode limit is 65535")
            }
        }
    }
}

impl std::error::Error for AssembleError {}

#[derive(Default)]
pub struct Assembler {
    pub chunk: BytecodeChunk,
    labels: Vec<Option<usize>>,
    branch_fixups: Vec<BranchFixup>,
    pending_switches: Vec<PendingSwitch>,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pc(&self) -> BytecodePc {
        self.chunk.pc()
    }

    pub fn label(&mut self) -> Label {
        let label = Label(self.labels.len() as u32);
        self.labels.push(None);
        label
    }

    pub fn bind(&mut self, label: Label) {
        let slot = self
            .labels
            .get_mut(label.0 as usize)
            .unwrap_or_else(|| panic!("invalid label {:?}", label));
        assert!(slot.is_none(), "label {:?} bound twice", label);
        *slot = Some(self.chunk.code.len());
    }

    #[inline]
    pub fn emit_word(&mut self, word: Word) -> BytecodePc {
        let pc = self.pc();
        self.chunk.code.push(word);
        pc
    }

    pub fn emit0(&mut self, opcode: Opcode) -> BytecodePc {
        self.emit_word(Word::abc(opcode, 0, 0, 0))
    }

    pub fn emit_a(&mut self, opcode: Opcode, a: u8) -> BytecodePc {
        self.emit_word(Word::abc(opcode, a, 0, 0))
    }

    pub fn emit_ab(&mut self, opcode: Opcode, a: u8, b: u8) -> BytecodePc {
        self.emit_word(Word::abc(opcode, a, b, 0))
    }

    pub fn emit_abc(&mut self, opcode: Opcode, a: u8, b: u8, c: u8) -> BytecodePc {
        self.emit_word(Word::abc(opcode, a, b, c))
    }

    pub fn emit_x32(&mut self, opcode: Opcode, a: u8, b: u8, c: u8, x: u32) -> BytecodePc {
        let pc = self.emit_abc(opcode, a, b, c);
        self.chunk.code.push(Word(x));
        pc
    }

    pub fn emit_x32_y32(
        &mut self,
        opcode: Opcode,
        a: u8,
        b: u8,
        c: u8,
        x: u32,
        y: u32,
    ) -> BytecodePc {
        let pc = self.emit_abc(opcode, a, b, c);
        self.chunk.code.push(Word(x));
        self.chunk.code.push(Word(y));
        pc
    }

    pub fn emit_move(&mut self, dst: Reg, src: Reg) -> BytecodePc {
        self.emit_ab(Opcode::Move, dst.0, src.0)
    }

    pub fn emit_load_const(&mut self, dst: Reg, constant: ConstIndex) -> BytecodePc {
        self.emit_x32(Opcode::LoadConst, dst.0, 0, 0, constant.0)
    }

    pub fn emit_jump(&mut self, target: Label) -> BytecodePc {
        let pc = self.emit_x32(Opcode::Jump, 0, 0, 0, 0);
        self.branch_fixups.push(BranchFixup {
            extension_word: pc.word_index() + 1,
            instruction_end: pc.word_index() + 2,
            target,
        });
        pc
    }

    pub fn emit_jump_if_false(&mut self, condition: Reg, target: Label) -> BytecodePc {
        let pc = self.emit_x32(Opcode::JumpIfFalse, condition.0, 0, 0, 0);
        self.branch_fixups.push(BranchFixup {
            extension_word: pc.word_index() + 1,
            instruction_end: pc.word_index() + 2,
            target,
        });
        pc
    }

    pub fn emit_jump_if_true(&mut self, condition: Reg, target: Label) -> BytecodePc {
        let pc = self.emit_x32(Opcode::JumpIfTrue, condition.0, 0, 0, 0);
        self.branch_fixups.push(BranchFixup {
            extension_word: pc.word_index() + 1,
            instruction_end: pc.word_index() + 2,
            target,
        });
        pc
    }

    pub fn emit_switch(
        &mut self,
        tag: Reg,
        default_target: Label,
        cases: impl IntoIterator<Item = (i64, Label)>,
    ) -> BytecodePc {
        let table = SwitchTableId(self.chunk.switch_tables.len() as u32);
        self.chunk.switch_tables.push(SwitchTable {
            default_target: BytecodePc(0),
            cases: Vec::new(),
        });
        let pc = self.emit_x32(Opcode::Switch, tag.0, 0, 0, table.0);
        self.pending_switches.push(PendingSwitch {
            table,
            default_target,
            cases: cases
                .into_iter()
                .map(|(value, target)| PendingSwitchCase { value, target })
                .collect(),
        });
        pc
    }

    fn validate_call_count(opcode: Opcode, count: usize) -> Result<(), AssembleError> {
        let valid = match opcode {
            Opcode::Invoke | Opcode::InvokeSuper | Opcode::CallClosure => count <= MAX_WREN_ARITY,
            Opcode::CallDirect | Opcode::CallPrimitive | Opcode::CallForeign => {
                count <= MAX_DIRECT_FRAME_VALUES
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(AssembleError::InvalidCallArity(count))
        }
    }

    fn emit_packed_call(
        &mut self,
        opcode: Opcode,
        a: u8,
        b: u8,
        args: &[Reg],
        extension: u32,
    ) -> Result<BytecodePc, AssembleError> {
        Self::validate_call_count(opcode, args.len())?;
        let pc = self.emit_abc(opcode, a, b, args.len() as u8);
        self.chunk.code.push(Word(extension));
        self.chunk.code.extend(pack_registers(args));
        Ok(pc)
    }

    pub fn emit_invoke(
        &mut self,
        dst: Reg,
        receiver: Reg,
        args: &[Reg],
        cache: IcSlot,
    ) -> Result<BytecodePc, AssembleError> {
        self.emit_packed_call(Opcode::Invoke, dst.0, receiver.0, args, cache.0)
    }

    pub fn emit_invoke_super(
        &mut self,
        dst: Reg,
        receiver: Reg,
        args: &[Reg],
        cache: IcSlot,
    ) -> Result<BytecodePc, AssembleError> {
        self.emit_packed_call(Opcode::InvokeSuper, dst.0, receiver.0, args, cache.0)
    }

    /// `args` is the complete IR call frame. For method-like target functions,
    /// element 0 becomes `THIS` and the remaining elements become ARG0..ARG15.
    /// For closures/module bodies/stubs, all elements become ARG registers.
    pub fn emit_call_direct(
        &mut self,
        dst: Reg,
        conv: CallConv,
        args: &[Reg],
        callee: FuncId,
    ) -> Result<BytecodePc, AssembleError> {
        self.emit_packed_call(Opcode::CallDirect, dst.0, encode_call_conv(conv), args, callee.0)
    }

    pub fn emit_call_closure(
        &mut self,
        dst: Reg,
        closure: Reg,
        conv: CallConv,
        args: &[Reg],
    ) -> Result<BytecodePc, AssembleError> {
        let pc = self.emit_packed_call(Opcode::CallClosure, dst.0, closure.0, args, encode_call_conv(conv) as u32)?;
        Ok(pc)
    }

    pub fn emit_call_primitive(
        &mut self,
        dst: Reg,
        args: &[Reg],
        primitive: PrimitiveId,
    ) -> Result<BytecodePc, AssembleError> {
        self.emit_packed_call(Opcode::CallPrimitive, dst.0, 0, args, primitive.0)
    }

    pub fn emit_call_foreign(
        &mut self,
        dst: Reg,
        args: &[Reg],
        foreign: ForeignFnId,
    ) -> Result<BytecodePc, AssembleError> {
        self.emit_packed_call(Opcode::CallForeign, dst.0, 0, args, foreign.0)
    }

    pub fn emit_make_closure(
        &mut self,
        dst: Reg,
        function: FuncId,
        captures: &[Reg],
    ) -> Result<BytecodePc, AssembleError> {
        if captures.len() > MAX_CLOSURE_CAPTURES {
            return Err(AssembleError::TooManyClosureCaptures(captures.len()));
        }
        let pc = self.emit_word(Word::abx(
            Opcode::MakeClosure,
            dst.0,
            captures.len() as u16,
        ));
        self.chunk.code.push(Word(function.0));
        self.chunk.code.extend(pack_registers(captures));
        Ok(pc)
    }

    pub fn add_safepoint(
        &mut self,
        pc: BytecodePc,
        kind: SafepointKind,
        references: ReferenceMap,
    ) -> SafepointIndex {
        let index = SafepointIndex(self.chunk.safepoints.len() as u32);
        self.chunk.safepoints.push(SafepointEntry {
            pc,
            kind,
            references,
        });
        index
    }

    pub fn finish(mut self) -> Result<BytecodeChunk, AssembleError> {
        for fixup in &self.branch_fixups {
            let target = self
                .labels
                .get(fixup.target.0 as usize)
                .and_then(|slot| *slot)
                .ok_or(AssembleError::UnboundLabel(fixup.target))?;
            let delta = target as i64 - fixup.instruction_end as i64;
            let delta = i32::try_from(delta).map_err(|_| AssembleError::BranchOutOfRange {
                from: fixup.instruction_end,
                to: target,
            })?;
            self.chunk.code[fixup.extension_word] = Word(delta as u32);
        }

        for pending in &self.pending_switches {
            let default_target = self
                .labels
                .get(pending.default_target.0 as usize)
                .and_then(|slot| *slot)
                .ok_or(AssembleError::UnboundLabel(pending.default_target))?;

            let table = self
                .chunk
                .switch_tables
                .get_mut(pending.table.0 as usize)
                .expect("pending switch table must exist");
            table.default_target = BytecodePc(default_target as u32);
            table.cases.clear();
            for case in &pending.cases {
                let target = self
                    .labels
                    .get(case.target.0 as usize)
                    .and_then(|slot| *slot)
                    .ok_or(AssembleError::UnboundLabel(case.target))?;
                table.cases.push(SwitchTarget {
                    value: case.value,
                    target: BytecodePc(target as u32),
                });
            }
            table.cases.sort_by_key(|case| case.value);
        }

        self.chunk.safepoints.sort_by_key(|entry| entry.pc);
        self.chunk.live_registers.sort_by_key(|entry| entry.pc);
        self.chunk.resume_points.sort_by_key(|entry| entry.pc);
        self.chunk.source_map.sort_by_key(|entry| entry.start_pc);
        Ok(self.chunk)
    }
}

#[inline]
pub const fn opcode_for_int_unary(op: IntUnaryOp) -> Opcode {
    match op {
        IntUnaryOp::Neg => Opcode::INeg,
        IntUnaryOp::BitNot => Opcode::IBitNot,
    }
}

#[inline]
pub const fn opcode_for_int_bin(op: IntBinOp) -> Opcode {
    match op {
        IntBinOp::Add => Opcode::IAdd,
        IntBinOp::Sub => Opcode::ISub,
        IntBinOp::Mul => Opcode::IMul,
        IntBinOp::DivS => Opcode::IDivS,
        IntBinOp::RemS => Opcode::IRemS,
        IntBinOp::BitAnd => Opcode::IBitAnd,
        IntBinOp::BitOr => Opcode::IBitOr,
        IntBinOp::BitXor => Opcode::IBitXor,
        IntBinOp::Shl => Opcode::IShl,
        IntBinOp::ShrU => Opcode::IShrU,
        IntBinOp::Sar => Opcode::ISar,
    }
}

#[inline]
pub const fn opcode_for_f64_unary(op: F64UnaryOp) -> Opcode {
    match op {
        F64UnaryOp::Neg => Opcode::FNeg,
    }
}

#[inline]
pub const fn opcode_for_f64_bin(op: F64BinOp) -> Opcode {
    match op {
        F64BinOp::Add => Opcode::FAdd,
        F64BinOp::Sub => Opcode::FSub,
        F64BinOp::Mul => Opcode::FMul,
        F64BinOp::Div => Opcode::FDiv,
        F64BinOp::Rem => Opcode::FRem,
    }
}

#[inline]
pub const fn opcode_for_int_cmp(kind: CmpKind) -> Opcode {
    match kind {
        CmpKind::Eq => Opcode::ICmpEq,
        CmpKind::Ne => Opcode::ICmpNe,
        CmpKind::Lt => Opcode::ICmpLt,
        CmpKind::Le => Opcode::ICmpLe,
        CmpKind::Gt => Opcode::ICmpGt,
        CmpKind::Ge => Opcode::ICmpGe,
    }
}

#[inline]
pub const fn opcode_for_f64_cmp(kind: CmpKind) -> Opcode {
    match kind {
        CmpKind::Eq => Opcode::FCmpEq,
        CmpKind::Ne => Opcode::FCmpNe,
        CmpKind::Lt => Opcode::FCmpLt,
        CmpKind::Le => Opcode::FCmpLe,
        CmpKind::Gt => Opcode::FCmpGt,
        CmpKind::Ge => Opcode::FCmpGe,
    }
}

#[inline]
pub const fn encode_call_conv(conv: CallConv) -> u8 {
    match conv {
        CallConv::Wren => 0,
        CallConv::Fast => 1,
        CallConv::Foreign => 2,
    }
}

#[inline]
pub const fn decode_call_conv(value: u8) -> Option<CallConv> {
    match value {
        0 => Some(CallConv::Wren),
        1 => Some(CallConv::Fast),
        2 => Some(CallConv::Foreign),
        _ => None,
    }
}

#[inline]
pub const fn encode_bailout_reason(reason: BailoutReason) -> u8 {
    match reason {
        BailoutReason::TypeMismatch => 0,
        BailoutReason::ClassMismatch => 1,
        BailoutReason::BoundsCheck => 2,
        BailoutReason::NotInt32 => 3,
        BailoutReason::NotF64 => 4,
        BailoutReason::NullValue => 5,
        BailoutReason::MegamorphicCall => 6,
        BailoutReason::InvalidSpeculation => 7,
        BailoutReason::Explicit => 8,
    }
}

#[inline]
pub const fn decode_bailout_reason(value: u8) -> Option<BailoutReason> {
    match value {
        0 => Some(BailoutReason::TypeMismatch),
        1 => Some(BailoutReason::ClassMismatch),
        2 => Some(BailoutReason::BoundsCheck),
        3 => Some(BailoutReason::NotInt32),
        4 => Some(BailoutReason::NotF64),
        5 => Some(BailoutReason::NullValue),
        6 => Some(BailoutReason::MegamorphicCall),
        7 => Some(BailoutReason::InvalidSpeculation),
        8 => Some(BailoutReason::Explicit),
        _ => None,
    }
}

// =============================================================================
// Verification
// =============================================================================

#[derive(Clone, Debug)]
pub struct VerificationError {
    pub pc: Option<BytecodePc>,
    pub message: String,
}

impl VerificationError {
    fn at(pc: BytecodePc, message: impl Into<String>) -> Self {
        Self {
            pc: Some(pc),
            message: message.into(),
        }
    }

    fn function(message: impl Into<String>) -> Self {
        Self {
            pc: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pc {
            Some(pc) => write!(f, "bytecode verification error at word {}: {}", pc.0, self.message),
            None => write!(f, "bytecode verification error: {}", self.message),
        }
    }
}

impl std::error::Error for VerificationError {}

pub struct BytecodeVerifier;

impl BytecodeVerifier {
    fn verify_class_set(program: &BytecodeProgram, set: &ClassSet) -> Result<(), VerificationError> {
        for &class in set.as_slice() {
            if program.metadata.class(class).is_none() {
                return Err(VerificationError::function(format!(
                    "type metadata references unknown class {:?}",
                    class
                )));
            }
        }
        Ok(())
    }

    fn verify_type(program: &BytecodeProgram, ty: &Type) -> Result<(), VerificationError> {
        match ty {
            Type::List(Some(element)) | Type::Cell(element) => Self::verify_type(program, element),
            Type::Class(classes) | Type::Instance(classes) => Self::verify_class_set(program, classes),
            Type::Closure(Some(function)) => {
                if program.function(*function).is_none() {
                    Err(VerificationError::function(format!(
                        "type metadata references unknown function {:?}",
                        function
                    )))
                } else {
                    Ok(())
                }
            }
            Type::Foreign(Some(class)) => {
                if program.metadata.class(*class).is_none() {
                    Err(VerificationError::function(format!(
                        "foreign type references unknown class {:?}",
                        class
                    )))
                } else {
                    Ok(())
                }
            }
            Type::Union(types) => {
                for member in types {
                    Self::verify_type(program, member)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn verify_metadata(program: &BytecodeProgram) -> Result<(), VerificationError> {
        for (index, method) in program.metadata.methods.iter().enumerate() {
            if program.metadata.string(method.name).is_none() {
                return Err(VerificationError::function(format!(
                    "method symbol {} references unknown string {:?}",
                    index, method.name
                )));
            }
            if method.arity as usize > MAX_WREN_ARITY {
                return Err(VerificationError::function(format!(
                    "method symbol {} has arity {}, maximum is {}",
                    index, method.arity, MAX_WREN_ARITY
                )));
            }
        }

        let mut module_ids = HashSet::new();
        for module in &program.metadata.modules {
            if !module_ids.insert(module.id) {
                return Err(VerificationError::function(format!(
                    "duplicate module metadata {:?}",
                    module.id
                )));
            }
            if program.metadata.string(module.name).is_none() {
                return Err(VerificationError::function(format!(
                    "module {:?} references unknown name {:?}",
                    module.id, module.name
                )));
            }
        }

        let mut class_ids = HashSet::new();
        for class in &program.metadata.classes {
            if !class_ids.insert(class.id) {
                return Err(VerificationError::function(format!(
                    "duplicate class metadata {:?}",
                    class.id
                )));
            }
            if program.metadata.module(class.module).is_none() {
                return Err(VerificationError::function(format!(
                    "class {:?} references unknown module {:?}",
                    class.id, class.module
                )));
            }
            if program.metadata.string(class.name).is_none() {
                return Err(VerificationError::function(format!(
                    "class {:?} references unknown name {:?}",
                    class.id, class.name
                )));
            }
            if let Some(superclass) = class.superclass {
                if program.metadata.class(superclass).is_none() {
                    return Err(VerificationError::function(format!(
                        "class {:?} references unknown superclass {:?}",
                        class.id, superclass
                    )));
                }
            }
        }

        let mut foreign_ids = HashSet::new();
        for foreign in &program.metadata.foreign_functions {
            if !foreign_ids.insert(foreign.id) {
                return Err(VerificationError::function(format!(
                    "duplicate foreign-function metadata {:?}",
                    foreign.id
                )));
            }
            if program.metadata.string(foreign.module).is_none()
                || program.metadata.string(foreign.class).is_none()
            {
                return Err(VerificationError::function(format!(
                    "foreign function {:?} references an unknown module/class name",
                    foreign.id
                )));
            }
            if program.metadata.method(foreign.signature).is_none() {
                return Err(VerificationError::function(format!(
                    "foreign function {:?} references unknown method {:?}",
                    foreign.id, foreign.signature
                )));
            }
        }

        Ok(())
    }

    /// Verify individual functions plus cross-function direct-call and closure
    /// invariants that cannot be checked from one chunk in isolation.
    pub fn verify_program(program: &BytecodeProgram) -> Result<(), VerificationError> {
        Self::verify_metadata(program)?;

        let mut function_ids = HashSet::new();
        for function in &program.functions {
            if !function_ids.insert(function.id) {
                return Err(VerificationError::function(format!(
                    "duplicate bytecode function {:?}",
                    function.id
                )));
            }
            if program.metadata.module(function.module).is_none() {
                return Err(VerificationError::function(format!(
                    "function {:?} references unknown module {:?}",
                    function.id, function.module
                )));
            }
            if program.metadata.string(function.name).is_none() {
                return Err(VerificationError::function(format!(
                    "function {:?} references unknown name {:?}",
                    function.id, function.name
                )));
            }
            if let Some(owner) = function.field_owner {
                if program.metadata.class(owner).is_none() {
                    return Err(VerificationError::function(format!(
                        "function {:?} references unknown field-owner class {:?}",
                        function.id, owner
                    )));
                }
            }
            Self::verify(function)?;

            if function.field_owner.is_none() {
                for decoded in Decoder::new(&function.chunk.code) {
                    let decoded = decoded.map_err(|error| VerificationError {
                        pc: error.pc,
                        message: error.message,
                    })?;
                    if matches!(decoded.opcode, Opcode::LoadField | Opcode::StoreField) {
                        return Err(VerificationError::function(format!(
                            "function {:?} contains instance field access but has no field-owner class",
                            function.id
                        )));
                    }
                }
            }

            for constant in &function.chunk.constants {
                if let Constant::String(string) = constant {
                    if program.metadata.string(*string).is_none() {
                        return Err(VerificationError::function(format!(
                            "function {:?} constant pool references unknown string {:?}",
                            function.id, string
                        )));
                    }
                }
            }
            for guard in &function.chunk.type_guards {
                Self::verify_type(program, &guard.expected)?;
            }
            for guard in &function.chunk.class_guards {
                Self::verify_class_set(program, &guard.expected)?;
            }
            for cache in &function.chunk.inline_caches {
                if program.metadata.method(cache.method).is_none() {
                    return Err(VerificationError::function(format!(
                        "function {:?} inline cache references unknown method {:?}",
                        function.id, cache.method
                    )));
                }
                if let Some(home_class) = cache.home_class {
                    if program.metadata.class(home_class).is_none() {
                        return Err(VerificationError::function(format!(
                            "function {:?} inline cache references unknown home class {:?}",
                            function.id, home_class
                        )));
                    }
                }
            }
        }

        let mut boundaries_by_function: HashMap<FuncId, HashSet<BytecodePc>> = HashMap::new();
        for function in &program.functions {
            let mut boundaries = HashSet::new();
            boundaries.insert(BytecodePc(function.chunk.code.len() as u32));
            for decoded in Decoder::new(&function.chunk.code) {
                let decoded = decoded.map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })?;
                boundaries.insert(decoded.pc);
            }
            boundaries_by_function.insert(function.id, boundaries);
        }

        for function in &program.functions {
            for point in &function.chunk.deopts {
                for frame in &point.inlined_frames {
                    let boundaries = boundaries_by_function.get(&frame.function).ok_or_else(|| {
                        VerificationError::function(format!(
                            "deopt references unknown inlined function {:?}",
                            frame.function
                        ))
                    })?;
                    if !boundaries.contains(&frame.resume_pc) {
                        return Err(VerificationError::function(format!(
                            "inlined frame {:?} resumes at non-boundary PC {}",
                            frame.function, frame.resume_pc.0
                        )));
                    }
                }
            }

            for decoded in Decoder::new(&function.chunk.code) {
                let inst = decoded.map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })?;

                match inst.opcode {
                    Opcode::CallDirect => {
                        let callee_id = FuncId(inst.extension(0).unwrap());
                        let callee = program.function(callee_id).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("direct call references unknown function {:?}", callee_id),
                            )
                        })?;
                        let expected = callee.arity as usize
                            + if function_kind_uses_this(callee.kind) { 1 } else { 0 };
                        let actual = inst.first.c() as usize;
                        if actual != expected {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "direct call to {:?} supplies {} frame values, expected {}",
                                    callee_id, actual, expected
                                ),
                            ));
                        }
                    }
                    Opcode::MakeClosure => {
                        let target_id = FuncId(inst.extension(0).unwrap());
                        let target = program.function(target_id).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("closure references unknown function {:?}", target_id),
                            )
                        })?;
                        let actual = inst.first.bx() as usize;
                        let expected = target.capture_count as usize;
                        if actual != expected {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "closure for {:?} captures {} values, expected {}",
                                    target_id, actual, expected
                                ),
                            ));
                        }
                    }
                    Opcode::Invoke | Opcode::InvokeSuper => {
                        let cache = &function.chunk.inline_caches[inst.extension(0).unwrap() as usize];
                        let method = program.metadata.method(cache.method).expect("cache method verified");
                        if method.arity != inst.first.c() {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "invoke arity {} does not match method {:?} arity {}",
                                    inst.first.c(), cache.method, method.arity
                                ),
                            ));
                        }
                    }
                    Opcode::CallForeign => {
                        let foreign_id = ForeignFnId(inst.extension(0).unwrap());
                        let foreign = program.metadata.foreign_function(foreign_id).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("call references unknown foreign function {:?}", foreign_id),
                            )
                        })?;
                        let signature = program
                            .metadata
                            .method(foreign.signature)
                            .expect("foreign method signature verified");
                        let expected = signature.arity as usize + 1; // receiver/class in slot 0
                        let actual = inst.first.c() as usize;
                        if actual != expected {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "foreign call {:?} supplies {} frame values, expected {}",
                                    foreign_id, actual, expected
                                ),
                            ));
                        }
                    }
                    Opcode::CreateClass | Opcode::AllocInstance | Opcode::AllocForeign => {
                        let class_id = ClassId(inst.extension(0).unwrap());
                        let class = program.metadata.class(class_id).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("instruction references unknown class {:?}", class_id),
                            )
                        })?;
                        if inst.opcode == Opcode::AllocForeign && !class.is_foreign {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!("AllocForeign targets non-foreign class {:?}", class_id),
                            ));
                        }
                        if inst.opcode == Opcode::AllocInstance && class.is_foreign {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!("AllocInstance targets foreign class {:?}", class_id),
                            ));
                        }
                    }
                    Opcode::BindMethod => {
                        let method = MethodSymbol(inst.extension(0).unwrap());
                        let target = FuncId(inst.extension(1).unwrap());
                        if program.metadata.method(method).is_none() {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!("BindMethod references unknown method {:?}", method),
                            ));
                        }
                        if program.function(target).is_none() {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!("BindMethod references unknown function {:?}", target),
                            ));
                        }
                    }
                    Opcode::BindForeignMethod => {
                        let method = MethodSymbol(inst.extension(0).unwrap());
                        let foreign = ForeignFnId(inst.extension(1).unwrap());
                        let signature = program.metadata.method(method).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("BindForeignMethod references unknown method {:?}", method),
                            )
                        })?;
                        let info = program.metadata.foreign_function(foreign).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("BindForeignMethod references unknown foreign {:?}", foreign),
                            )
                        })?;
                        if info.signature != method || info.is_static != (inst.first.b() != 0) {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "BindForeignMethod metadata mismatch for {:?} (arity {})",
                                    foreign, signature.arity
                                ),
                            ));
                        }
                    }
                    Opcode::ImportModule | Opcode::ImportVariable => {
                        let string = StringId(inst.extension(0).unwrap());
                        if program.metadata.string(string).is_none() {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!("import references unknown string {:?}", string),
                            ));
                        }
                    }
                    Opcode::LoadModuleVar | Opcode::StoreModuleVar => {
                        let module_id = ModuleId(inst.extension(0).unwrap());
                        let slot = ModuleSlot(inst.extension(1).unwrap());
                        let module = program.metadata.module(module_id).ok_or_else(|| {
                            VerificationError::at(
                                inst.pc,
                                format!("module access references unknown module {:?}", module_id),
                            )
                        })?;
                        if slot.0 >= module.variable_count {
                            return Err(VerificationError::at(
                                inst.pc,
                                format!(
                                    "module slot {} is outside module {:?} variable count {}",
                                    slot.0, module_id, module.variable_count
                                ),
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    pub fn verify(function: &BytecodeFunction) -> Result<(), VerificationError> {
        function
            .frame
            .validate()
            .map_err(VerificationError::function)?;

        if function.arity as usize > MAX_WREN_ARITY {
            return Err(VerificationError::function(format!(
                "function arity {} exceeds Wren limit {}",
                function.arity, MAX_WREN_ARITY
            )));
        }

        for binding in &function.frame.parameters {
            match binding.source {
                IncomingSource::This if !function_kind_uses_this(function.kind) => {
                    return Err(VerificationError::function(
                        "non-method function has an incoming THIS binding",
                    ));
                }
                IncomingSource::Argument(arg) if arg.index() >= function.arity as usize => {
                    return Err(VerificationError::function(format!(
                        "incoming {:?} is outside function arity {}",
                        arg, function.arity
                    )));
                }
                _ => {}
            }
        }

        for binding in &function.frame.captures {
            if binding.capture_index >= function.capture_count {
                return Err(VerificationError::function(format!(
                    "capture binding {} is outside capture count {}",
                    binding.capture_index, function.capture_count
                )));
            }
        }

        let chunk = &function.chunk;
        let mut instructions = Vec::new();
        let mut boundaries = HashSet::new();
        boundaries.insert(BytecodePc(chunk.code.len() as u32));

        for decoded in Decoder::new(&chunk.code) {
            let decoded = decoded.map_err(|error| VerificationError {
                pc: error.pc,
                message: error.message,
            })?;
            boundaries.insert(decoded.pc);
            instructions.push(decoded);
        }

        for inst in &instructions {
            Self::verify_instruction(function, *inst, &boundaries)?;
        }

        for table in &chunk.switch_tables {
            if !boundaries.contains(&table.default_target) {
                return Err(VerificationError::function(format!(
                    "switch default target {} is not an instruction boundary",
                    table.default_target.0
                )));
            }
            let mut previous = None;
            for case in &table.cases {
                if !boundaries.contains(&case.target) {
                    return Err(VerificationError::function(format!(
                        "switch target {} is not an instruction boundary",
                        case.target.0
                    )));
                }
                if let Some(prev) = previous {
                    if prev >= case.value {
                        return Err(VerificationError::function(
                            "switch cases must be strictly sorted and unique",
                        ));
                    }
                }
                previous = Some(case.value);
            }
        }

        Self::verify_side_tables(function, &boundaries)?;
        Ok(())
    }

    fn verify_deopt_location(
        function: &BytecodeFunction,
        location: DeoptLocation,
    ) -> Result<(), VerificationError> {
        match location {
            DeoptLocation::General(reg) => {
                if reg.index() >= function.frame.general_register_count as usize {
                    return Err(VerificationError::function(format!(
                        "deopt location {:?} is outside frame register count {}",
                        reg, function.frame.general_register_count
                    )));
                }
            }
            DeoptLocation::Argument(arg) => {
                if arg.index() >= ARGUMENT_REGISTER_COUNT {
                    return Err(VerificationError::function(format!(
                        "invalid deopt argument register {:?}",
                        arg
                    )));
                }
            }
            DeoptLocation::Return(ret) => {
                if ret.index() >= RETURN_REGISTER_COUNT {
                    return Err(VerificationError::function(format!(
                        "invalid deopt return register {:?}",
                        ret
                    )));
                }
            }
            DeoptLocation::Special(reg) => {
                if reg.index() >= SPECIAL_REGISTER_COUNT {
                    return Err(VerificationError::function(
                        "invalid deopt special register",
                    ));
                }
            }
            DeoptLocation::Constant(index) => {
                if index.0 as usize >= function.chunk.constants.len() {
                    return Err(VerificationError::function(format!(
                        "deopt constant {:?} is out of bounds",
                        index
                    )));
                }
            }
            DeoptLocation::Null
            | DeoptLocation::True
            | DeoptLocation::False
            | DeoptLocation::Int32(_)
            | DeoptLocation::F64Bits(_)
            | DeoptLocation::Unavailable => {}
        }
        Ok(())
    }

    fn verify_side_tables(
        function: &BytecodeFunction,
        boundaries: &HashSet<BytecodePc>,
    ) -> Result<(), VerificationError> {
        let chunk = &function.chunk;
        let mut last_safepoint = None;
        for entry in &chunk.safepoints {
            if !boundaries.contains(&entry.pc) {
                return Err(VerificationError::function(format!(
                    "safepoint PC {} is not an instruction boundary",
                    entry.pc.0
                )));
            }
            if let Some(previous) = last_safepoint {
                if previous >= entry.pc.0 {
                    return Err(VerificationError::function(
                        "safepoints must be strictly sorted and unique",
                    ));
                }
            }
            last_safepoint = Some(entry.pc.0);

            // Bits for GPRs outside the declared frame must be clear.
            for reg_index in function.frame.general_register_count as usize..GENERAL_REGISTER_COUNT {
                let reg = Reg(reg_index as u8);
                if entry.references.general_is_reference(reg) {
                    return Err(VerificationError::function(format!(
                        "safepoint {} marks {:?} live outside the frame",
                        entry.pc.0, reg
                    )));
                }
            }
            let valid_special_mask = if SPECIAL_REGISTER_COUNT == 16 {
                u16::MAX
            } else {
                (1u16 << SPECIAL_REGISTER_COUNT) - 1
            };
            if entry.references.special & !valid_special_mask != 0 {
                return Err(VerificationError::function(format!(
                    "safepoint {} sets invalid special-register GC bits",
                    entry.pc.0
                )));
            }
            if entry.references.returns & !0b11 != 0 {
                return Err(VerificationError::function(format!(
                    "safepoint {} sets invalid return-register GC bits",
                    entry.pc.0
                )));
            }
        }

        let mut last_live_pc = None;
        for entry in &chunk.live_registers {
            if !boundaries.contains(&entry.pc) {
                return Err(VerificationError::function(format!(
                    "live-register PC {} is not an instruction boundary",
                    entry.pc.0
                )));
            }
            if let Some(previous) = last_live_pc {
                if previous >= entry.pc.0 {
                    return Err(VerificationError::function(
                        "live-register entries must be strictly sorted and unique",
                    ));
                }
            }
            last_live_pc = Some(entry.pc.0);

            for reg_index in function.frame.general_register_count as usize..GENERAL_REGISTER_COUNT {
                let reg = Reg(reg_index as u8);
                if entry.live.general_is_live(reg) {
                    return Err(VerificationError::function(format!(
                        "live-register entry {} marks {:?} live outside the frame",
                        entry.pc.0, reg
                    )));
                }
            }
        }

        let mut last_resume_pc = None;
        for entry in &chunk.resume_points {
            if !boundaries.contains(&entry.pc) {
                return Err(VerificationError::function(format!(
                    "resume PC {} is not an instruction boundary",
                    entry.pc.0
                )));
            }
            if let Some(previous) = last_resume_pc {
                if previous > entry.pc.0 {
                    return Err(VerificationError::function(
                        "resume points must be sorted by PC",
                    ));
                }
            }
            last_resume_pc = Some(entry.pc.0);
        }

        let mut deopt_ids = HashSet::new();
        for point in &chunk.deopts {
            if !deopt_ids.insert(point.id) {
                return Err(VerificationError::function(format!(
                    "duplicate deopt ID {:?}",
                    point.id
                )));
            }
            if !boundaries.contains(&point.resume_pc) {
                return Err(VerificationError::function(format!(
                    "deopt {:?} resumes at non-boundary PC {}",
                    point.id, point.resume_pc.0
                )));
            }
            for slot in &point.slots {
                Self::verify_deopt_location(function, slot.location)?;
            }
            for frame in &point.inlined_frames {
                // Resume PC belongs to `frame.function`; `verify_program` checks it.
                for slot in &frame.slots {
                    Self::verify_deopt_location(function, slot.location)?;
                }
            }
        }

        for source in &chunk.source_map {
            if source.start_pc.0 > source.end_pc.0 || source.end_pc.0 > chunk.code.len() as u32 {
                return Err(VerificationError::function("invalid source-map PC range"));
            }
        }

        Ok(())
    }

    fn verify_reg(
        function: &BytecodeFunction,
        pc: BytecodePc,
        reg: Reg,
    ) -> Result<(), VerificationError> {
        if reg.index() >= function.frame.general_register_count as usize {
            Err(VerificationError::at(
                pc,
                format!(
                    "register {:?} is outside frame register count {}",
                    reg, function.frame.general_register_count
                ),
            ))
        } else {
            Ok(())
        }
    }

    fn relative_target(inst: DecodedInstruction<'_>) -> Result<BytecodePc, VerificationError> {
        let raw = inst.extension(0).ok_or_else(|| {
            VerificationError::at(inst.pc, "missing relative branch displacement")
        })?;
        let delta = raw as i32 as i64;
        let target = inst.next_pc().0 as i64 + delta;
        if target < 0 || target > u32::MAX as i64 {
            return Err(VerificationError::at(inst.pc, "relative branch target overflow"));
        }
        Ok(BytecodePc(target as u32))
    }

    fn require_deopt(chunk: &BytecodeChunk, pc: BytecodePc, raw: u32) -> Result<(), VerificationError> {
        let id = DeoptId(raw);
        if chunk.deopt_by_id(id).is_none() {
            Err(VerificationError::at(pc, format!("unknown deopt point {:?}", id)))
        } else {
            Ok(())
        }
    }

    fn verify_instruction(
        function: &BytecodeFunction,
        inst: DecodedInstruction<'_>,
        boundaries: &HashSet<BytecodePc>,
    ) -> Result<(), VerificationError> {
        use Opcode::*;
        let chunk = &function.chunk;
        let a = Reg(inst.first.a());
        let b = Reg(inst.first.b());
        let c = Reg(inst.first.c());

        if inst.opcode.may_gc() && chunk.safepoint_at(inst.pc).is_none() {
            return Err(VerificationError::at(
                inst.pc,
                format!("{} may GC but has no stack map", inst.opcode.name()),
            ));
        }

        match inst.opcode {
            Nop | Trap | Safepoint | OsrPoint | Breakpoint => {}

            Move | Swap => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            LoadNull | LoadFalse | LoadTrue | LoadI8 | LoadI32 | LoadConst => {
                Self::verify_reg(function, inst.pc, a)?;
                if inst.opcode == LoadConst {
                    let index = inst.extension(0).unwrap() as usize;
                    if index >= chunk.constants.len() {
                        return Err(VerificationError::at(inst.pc, "constant index out of bounds"));
                    }
                }
            }
            MoveFromArg => {
                Self::verify_reg(function, inst.pc, a)?;
                if ArgReg::new(inst.first.b()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid argument register"));
                }
            }
            MoveToArg => {
                if ArgReg::new(inst.first.a()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid argument register"));
                }
                Self::verify_reg(function, inst.pc, b)?;
            }
            MoveFromRet => {
                Self::verify_reg(function, inst.pc, a)?;
                if ReturnReg::new(inst.first.b()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid return register"));
                }
            }
            MoveToRet => {
                if ReturnReg::new(inst.first.a()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid return register"));
                }
                Self::verify_reg(function, inst.pc, b)?;
            }
            MoveFromSpecial => {
                Self::verify_reg(function, inst.pc, a)?;
                if SpecialReg::from_u8(inst.first.b()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid special register"));
                }
            }
            MoveToSpecial => {
                if SpecialReg::from_u8(inst.first.a()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid special register"));
                }
                Self::verify_reg(function, inst.pc, b)?;
            }

            Jump | JumpIfFalse | JumpIfTrue => {
                if matches!(inst.opcode, JumpIfFalse | JumpIfTrue) {
                    Self::verify_reg(function, inst.pc, a)?;
                }
                let target = Self::relative_target(inst)?;
                if !boundaries.contains(&target) {
                    return Err(VerificationError::at(
                        inst.pc,
                        format!("branch target {} is not an instruction boundary", target.0),
                    ));
                }
            }
            Switch => {
                Self::verify_reg(function, inst.pc, a)?;
                let index = inst.extension(0).unwrap() as usize;
                if index >= chunk.switch_tables.len() {
                    return Err(VerificationError::at(inst.pc, "switch table index out of bounds"));
                }
            }
            Return => Self::verify_reg(function, inst.pc, a)?,
            Return2 => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }

            GuardType => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                let guard = inst.extension(0).unwrap() as usize;
                if guard >= chunk.type_guards.len() {
                    return Err(VerificationError::at(inst.pc, "type guard index out of bounds"));
                }
                Self::require_deopt(chunk, inst.pc, inst.extension(1).unwrap())?;
            }
            GuardClass => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                let guard = inst.extension(0).unwrap() as usize;
                if guard >= chunk.class_guards.len() {
                    return Err(VerificationError::at(inst.pc, "class guard index out of bounds"));
                }
                Self::require_deopt(chunk, inst.pc, inst.extension(1).unwrap())?;
            }
            GuardBounds => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::require_deopt(chunk, inst.pc, inst.extension(0).unwrap())?;
            }
            GuardNotNull | GuardInt32 | GuardF64 => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::require_deopt(chunk, inst.pc, inst.extension(0).unwrap())?;
            }
            Bailout => {
                if decode_bailout_reason(inst.first.a()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid bailout reason"));
                }
                Self::require_deopt(chunk, inst.pc, inst.extension(0).unwrap())?;
            }

            Invoke | InvokeSuper => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                let argc = inst.first.c() as usize;
                if argc > MAX_WREN_ARITY {
                    return Err(VerificationError::at(inst.pc, "invoke arity exceeds 16"));
                }
                for reg in inst.call_arguments().map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })? {
                    Self::verify_reg(function, inst.pc, reg)?;
                }
                let cache = inst.extension(0).unwrap() as usize;
                let descriptor = chunk.inline_caches.get(cache).ok_or_else(|| {
                    VerificationError::at(inst.pc, "inline cache index out of bounds")
                })?;
                if inst.opcode == InvokeSuper && descriptor.home_class.is_none() {
                    return Err(VerificationError::at(
                        inst.pc,
                        "InvokeSuper cache descriptor has no home class",
                    ));
                }
                if inst.opcode == Invoke && descriptor.home_class.is_some() {
                    return Err(VerificationError::at(
                        inst.pc,
                        "Invoke cache descriptor unexpectedly has a home class",
                    ));
                }
            }
            CallDirect => {
                Self::verify_reg(function, inst.pc, a)?;
                if decode_call_conv(inst.first.b()).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid direct-call convention"));
                }
                let argc = inst.first.c() as usize;
                if argc > MAX_DIRECT_FRAME_VALUES {
                    return Err(VerificationError::at(inst.pc, "direct-call frame has more than 17 values"));
                }
                for reg in inst.call_arguments().map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })? {
                    Self::verify_reg(function, inst.pc, reg)?;
                }
            }
            CallClosure => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                if decode_call_conv(inst.extension(0).unwrap() as u8).is_none() {
                    return Err(VerificationError::at(inst.pc, "invalid closure-call convention"));
                }
                let argc = inst.first.c() as usize;
                if argc > MAX_WREN_ARITY {
                    return Err(VerificationError::at(inst.pc, "closure-call arity exceeds 16"));
                }
                for reg in inst.call_arguments().map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })? {
                    Self::verify_reg(function, inst.pc, reg)?;
                }
            }
            CallPrimitive | CallForeign => {
                Self::verify_reg(function, inst.pc, a)?;
                let argc = inst.first.c() as usize;
                if argc > MAX_DIRECT_FRAME_VALUES {
                    return Err(VerificationError::at(inst.pc, "runtime call frame has more than 17 values"));
                }
                for reg in inst.call_arguments().map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })? {
                    Self::verify_reg(function, inst.pc, reg)?;
                }
            }

            CreateClass => {
                Self::verify_reg(function, inst.pc, a)?;
                match inst.first.c() {
                    0 => {}
                    1 => Self::verify_reg(function, inst.pc, b)?,
                    _ => {
                        return Err(VerificationError::at(
                            inst.pc,
                            "CreateClass has_super flag must be 0 or 1",
                        ));
                    }
                }
            }
            BindMethod | BindForeignMethod => {
                Self::verify_reg(function, inst.pc, a)?;
                if inst.first.b() > 1 {
                    return Err(VerificationError::at(inst.pc, "is_static flag must be 0 or 1"));
                }
            }
            AllocInstance => Self::verify_reg(function, inst.pc, a)?,
            AllocForeign => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            LoadField | LoadStaticField => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            StoreField | StoreStaticField => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            IsInstance => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }

            ImportModule => Self::verify_reg(function, inst.pc, a)?,
            ImportVariable => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            LoadModuleVar => Self::verify_reg(function, inst.pc, a)?,
            StoreModuleVar => Self::verify_reg(function, inst.pc, a)?,

            AllocCell | LoadCell => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            StoreCell => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            MakeClosure => {
                Self::verify_reg(function, inst.pc, a)?;
                for reg in inst.closure_captures().map_err(|error| VerificationError {
                    pc: error.pc,
                    message: error.message,
                })? {
                    Self::verify_reg(function, inst.pc, reg)?;
                }
            }

            AllocList | AllocMap => Self::verify_reg(function, inst.pc, a)?,
            AllocListCap => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            ListLen | MapLen | StringLen => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            ListGet | ListRemoveAt | MapGet | MapRemove | StringCodePointAt => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }
            ListSet | ListInsert | MapSet => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }
            ListAppend => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            StringConcat => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }
            StringSlice => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
                let count_raw = inst.extension(0).unwrap();
                if count_raw > u8::MAX as u32 {
                    return Err(VerificationError::at(
                        inst.pc,
                        "StringSlice count register does not fit u8",
                    ));
                }
                Self::verify_reg(function, inst.pc, Reg(count_raw as u8))?;
            }
            MakeRange => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
                if inst.extension(0).unwrap() > 1 {
                    return Err(VerificationError::at(
                        inst.pc,
                        "MakeRange inclusive flag must be 0 or 1",
                    ));
                }
            }

            INeg | IBitNot | FNeg | NumToInt32 | NumToF64 | Int32ToNum | F64ToNum => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            IAdd | ISub | IMul | IDivS | IRemS | IBitAnd | IBitOr | IBitXor | IShl
            | IShrU | ISar | FAdd | FSub | FMul | FDiv | FRem | ICmpEq | ICmpNe
            | ICmpLt | ICmpLe | ICmpGt | ICmpGe | FCmpEq | FCmpNe | FCmpLt
            | FCmpLe | FCmpGt | FCmpGe | RefEq => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }

            Await => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                if !function.is_async && function.kind != FunctionKind::ModuleBody {
                    return Err(VerificationError::at(
                        inst.pc,
                        "Await is only valid in async functions or module bodies",
                    ));
                }
            }
            MakeFiber => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            FiberCall | FiberTry | FiberTransfer | FiberTransferError => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
                Self::verify_reg(function, inst.pc, c)?;
            }
            FiberYield => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            FiberSuspend => Self::verify_reg(function, inst.pc, a)?,
            FiberAbort => Self::verify_reg(function, inst.pc, a)?,

            WriteBarrier => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }
            ReadBarrier => {
                Self::verify_reg(function, inst.pc, a)?;
                Self::verify_reg(function, inst.pc, b)?;
            }

            ProfileBlock => {
                let index = inst.extension(0).unwrap() as usize;
                let counter = chunk.profile_counters.get(index).ok_or_else(|| {
                    VerificationError::at(inst.pc, "profile counter index out of bounds")
                })?;
                if counter.kind != ProfileCounterKind::Block {
                    return Err(VerificationError::at(
                        inst.pc,
                        "ProfileBlock references a non-block counter",
                    ));
                }
            }
            ProfileNumeric => {
                Self::verify_reg(function, inst.pc, a)?;
                let index = inst.extension(0).unwrap() as usize;
                let counter = chunk.profile_counters.get(index).ok_or_else(|| {
                    VerificationError::at(inst.pc, "profile counter index out of bounds")
                })?;
                if counter.kind != ProfileCounterKind::Numeric {
                    return Err(VerificationError::at(
                        inst.pc,
                        "ProfileNumeric references a non-numeric counter",
                    ));
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// Instruction semantics and operand conventions
// =============================================================================

/// Compact documentation usable by disassemblers/debuggers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandRole {
    None,
    General,
    Argument,
    Return,
    Special,
    Flag,
    SmallImmediate,
}

/// Returns the primary ABC-byte roles. Extension words are opcode-specific and
/// documented by `extension_description`.
pub const fn operand_roles(opcode: Opcode) -> [OperandRole; 3] {
    use OperandRole::*;
    use Opcode::*;
    match opcode {
        Move | Swap => [General, General, None],
        LoadNull | LoadFalse | LoadTrue | LoadI8 | LoadI32 | LoadConst => [General, SmallImmediate, None],
        MoveFromArg => [General, Argument, None],
        MoveToArg => [Argument, General, None],
        MoveFromRet => [General, OperandRole::Return, None],
        MoveToRet => [OperandRole::Return, General, None],
        MoveFromSpecial => [General, Special, None],
        MoveToSpecial => [Special, General, None],
        Jump => [None, None, None],
        JumpIfFalse | JumpIfTrue | Switch => [General, None, None],
        Opcode::Return => [General, None, None],
        Return2 => [General, General, None],
        GuardType | GuardClass | GuardNotNull | GuardInt32 | GuardF64 => [General, General, None],
        GuardBounds => [General, General, None],
        Bailout => [SmallImmediate, None, None],
        Invoke | InvokeSuper => [General, General, SmallImmediate],
        CallDirect => [General, Flag, SmallImmediate],
        CallClosure => [General, General, SmallImmediate],
        CallPrimitive | CallForeign => [General, None, SmallImmediate],
        CreateClass => [General, General, Flag],
        BindMethod | BindForeignMethod => [General, Flag, None],
        AllocInstance => [General, None, None],
        AllocForeign => [General, General, None],
        LoadField | LoadStaticField => [General, General, None],
        StoreField | StoreStaticField => [General, General, None],
        IsInstance => [General, General, General],
        ImportModule => [General, None, None],
        ImportVariable => [General, General, None],
        LoadModuleVar | StoreModuleVar => [General, None, None],
        AllocCell | LoadCell => [General, General, None],
        StoreCell => [General, General, None],
        MakeClosure => [General, SmallImmediate, SmallImmediate],
        AllocList | AllocMap => [General, None, None],
        AllocListCap | ListLen | MapLen | StringLen => [General, General, None],
        ListGet | ListRemoveAt | MapGet | MapRemove | StringCodePointAt => [General, General, General],
        ListSet | ListInsert | MapSet => [General, General, General],
        ListAppend => [General, General, None],
        StringConcat | StringSlice | MakeRange => [General, General, General],
        INeg | IBitNot | FNeg | NumToInt32 | NumToF64 | Int32ToNum | F64ToNum => [General, General, None],
        IAdd | ISub | IMul | IDivS | IRemS | IBitAnd | IBitOr | IBitXor | IShl
        | IShrU | ISar | FAdd | FSub | FMul | FDiv | FRem | ICmpEq | ICmpNe
        | ICmpLt | ICmpLe | ICmpGt | ICmpGe | FCmpEq | FCmpNe | FCmpLt
        | FCmpLe | FCmpGt | FCmpGe | RefEq => [General, General, General],
        Await => [General, General, None],
        MakeFiber => [General, General, None],
        FiberCall | FiberTry | FiberTransfer | FiberTransferError => [General, General, General],
        FiberYield => [General, General, None],
        FiberSuspend | FiberAbort => [General, None, None],
        WriteBarrier | ReadBarrier => [General, General, None],
        ProfileNumeric => [General, None, None],
        Nop | Trap | Safepoint | OsrPoint | ProfileBlock | Breakpoint => [None, None, None],
    }
}

pub const fn extension_description(opcode: Opcode) -> &'static str {
    use Opcode::*;
    match opcode {
        LoadI32 => "x0=i32 bits",
        LoadConst => "x0=ConstIndex",
        Jump | JumpIfFalse | JumpIfTrue => "x0=i32 relative word displacement",
        Switch => "x0=SwitchTableId",
        GuardType => "x0=TypeGuardId, x1=DeoptId",
        GuardClass => "x0=ClassGuardId, x1=DeoptId",
        GuardBounds | GuardNotNull | GuardInt32 | GuardF64 => "x0=DeoptId",
        Bailout => "x0=DeoptId",
        Invoke | InvokeSuper => "x0=IcSlot, then packed explicit argument GPRs",
        CallDirect => "x0=FuncId, then packed complete call-frame GPRs",
        CallClosure => "x0=CallConv, then packed argument GPRs",
        CallPrimitive => "x0=PrimitiveId, then packed call-frame GPRs",
        CallForeign => "x0=ForeignFnId, then packed call-frame GPRs",
        CreateClass | AllocInstance | AllocForeign => "x0=ClassId",
        BindMethod => "x0=MethodSymbol, x1=FuncId",
        BindForeignMethod => "x0=MethodSymbol, x1=ForeignFnId",
        LoadField | StoreField => "x0=class-local FieldSlot (relocated to absolute slot by runtime linker)",
        LoadStaticField | StoreStaticField => "x0=StaticFieldSlot",
        ImportModule => "x0=StringId(module name)",
        ImportVariable => "x0=StringId(variable name)",
        LoadModuleVar | StoreModuleVar => "x0=ModuleId, x1=ModuleSlot",
        StringSlice => "x0=count GPR index",
        MakeRange => "x0=inclusive flag (0/1)",
        MakeClosure => "x0=FuncId, then packed capture GPRs; capture count is Bx",
        ProfileBlock | ProfileNumeric => "x0=CounterId",
        _ => "",
    }
}

// =============================================================================
// Disassembler
// =============================================================================

pub fn disassemble(function: &BytecodeFunction) -> Result<String, DecodeError> {
    let mut out = String::new();
    use std::fmt::Write as _;

    writeln!(
        &mut out,
        "func {:?} module={:?} kind={:?} async={} arity={} captures={} gprs={}",
        function.id,
        function.module,
        function.kind,
        function.is_async,
        function.arity,
        function.capture_count,
        function.frame.general_register_count
    )
    .unwrap();

    for inst in Decoder::new(&function.chunk.code) {
        let inst = inst?;
        write!(&mut out, "{:04}  {:<20}", inst.pc.0, inst.opcode.name()).unwrap();
        write!(
            &mut out,
            " A={:3} B={:3} C={:3}",
            inst.first.a(),
            inst.first.b(),
            inst.first.c()
        )
        .unwrap();

        if inst.words.len() > 1 {
            write!(&mut out, "  [").unwrap();
            for (index, word) in inst.words[1..].iter().enumerate() {
                if index != 0 {
                    write!(&mut out, ", ").unwrap();
                }
                write!(&mut out, "0x{:08x}", word.0).unwrap();
            }
            write!(&mut out, "]").unwrap();
        }
        writeln!(&mut out).unwrap();
    }

    Ok(out)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_256_general_registers_are_distinct() {
        assert_eq!(GENERAL_REGISTERS.len(), 256);
        for (index, reg) in GENERAL_REGISTERS.iter().enumerate() {
            assert_eq!(reg.0 as usize, index);
        }
    }

    #[test]
    fn packed_register_roundtrip() {
        let regs = [R0, R1, R2, R3, R4, R17, R128, R255, R33];
        let packed = pack_registers(&regs);
        let unpacked = unpack_registers(&packed, regs.len()).unwrap();
        assert_eq!(unpacked, regs);
    }

    #[test]
    fn variable_call_length_is_decodable() {
        let mut assembler = Assembler::new();
        let cache = assembler.chunk.push_inline_cache(InlineCacheDescriptor {
            site: SiteId(1),
            method: MethodSymbol(2),
            home_class: None,
        });
        assembler
            .emit_invoke(R9, R1, &[R2, R3, R4, R5, R6], cache)
            .unwrap();
        let chunk = assembler.finish().unwrap();
        let decoded = Decoder::new(&chunk.code).next().unwrap().unwrap();
        assert_eq!(decoded.opcode, Opcode::Invoke);
        assert_eq!(decoded.len_words(), 4); // header + IC + two packed words
        assert_eq!(decoded.call_arguments().unwrap(), vec![R2, R3, R4, R5, R6]);
    }

    #[test]
    fn assembler_patches_relative_branch() {
        let mut assembler = Assembler::new();
        let target = assembler.label();
        assembler.emit_jump(target);
        assembler.emit0(Opcode::Nop);
        assembler.bind(target);
        assembler.emit_a(Opcode::Return, R0.0);
        let chunk = assembler.finish().unwrap();

        let mut decoder = Decoder::new(&chunk.code);
        let jump = decoder.next().unwrap().unwrap();
        let displacement = jump.extension(0).unwrap() as i32;
        assert_eq!(jump.next_pc().0 as i32 + displacement, 3);
    }
}
