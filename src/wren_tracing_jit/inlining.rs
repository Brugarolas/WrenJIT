//! Bounded, replayable leaf inlining. No callee frame is needed on a side exit:
//! every check resumes at the caller's original call, before any callee effects.
//! Invoke may guard up to four observed classes sharing the same target. Reads
//! and scalar computations are replayable; stores, allocations and nested Wren
//! calls still require the synchronized fallback.
use super::*;
use crate::wren_bytecode::{IncomingSource, function_kind_uses_this};

impl TraceRecorder<'_> {
    pub(super) fn inline_call<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::wren_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        let (target, class) = if op.opcode == Opcode::Invoke {
            let receiver = runtime.trace_read_slot(FrameSlot::General(Reg(op.b)))
                .map_err(|_| RecordingAbort::UnsupportedCall)?;
            let (target, class) = runtime.trace_inline_target(op.location.function, op.x, receiver)
                .ok_or(RecordingAbort::UnsupportedCall)?;
            (target, Some(class))
        } else { (FuncId(op.x), None) };
        let callee = program.function(target).ok_or(RecordingAbort::UnsupportedCall)?;
        if callee.is_async || callee.capture_count != 0 || !callee.chunk.exception_handlers.is_empty()
            || callee.frame.return_kind != RegisterKind::Value || target == op.location.function
            || builder.trace().stats.inlined_calls >= 8 {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let uses_this = function_kind_uses_this(callee.kind);
        let receiver_reg = if class.is_some() { Some(Reg(op.b)) }
            else if uses_this { op.args.first().copied() } else { None };
        let arguments = if class.is_none() && uses_this { op.args.get(1..).ok_or(RecordingAbort::UnsupportedCall)? } else { &op.args };
        if arguments.len() != callee.arity as usize { return Err(RecordingAbort::UnsupportedCall); }
        let mut args = Vec::new();
        for reg in arguments { args.push(self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged)?); }
        let receiver = receiver_reg.map(|reg| self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged)).transpose()?;
        let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
        builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
        if let Some(classes) = &class {
            builder.emit1(TraceInst::GuardClass { value: receiver.ok_or(RecordingAbort::UnsupportedCall)?, classes: classes.clone(), exit }, TraceType::Any, TraceRep::Tagged);
        }
        let mut registers = vec![None; 256];
        for binding in &callee.frame.parameters {
            if binding.kind != RegisterKind::Value { return Err(RecordingAbort::UnsupportedCall); }
            registers[binding.destination.0 as usize] = match binding.source {
                IncomingSource::This => receiver,
                IncomingSource::Argument(arg) => args.get(arg.0 as usize).copied(),
            };
        }
        let mut pc = BytecodePc(0);
        for count in 1..=32 {
            let inst = decode_at(program, TraceLocation::new(target, pc)).map_err(|_| RecordingAbort::UnsupportedCall)?;
            let get = |reg: u8| registers[reg as usize].ok_or(RecordingAbort::UnsupportedCall);
            let scalar = |reg: u8, rep: TraceRep| {
                let v = get(reg)?;
                if builder.trace().value(v).is_some_and(|v| v.rep == rep) { Ok(v) } else { Err(RecordingAbort::UnsupportedCall) }
            };
            use Opcode::*;
            let value = match inst.opcode {
                Nop | OsrPoint => { pc = inst.next_pc; continue; }
                Move => get(inst.b)?,
                LoadNull => builder.constant(TraceConstant::Null),
                LoadFalse | LoadTrue => builder.constant(TraceConstant::Bool(inst.opcode == LoadTrue)),
                LoadI8 => builder.constant(TraceConstant::Int32(inst.b as i8 as i32)),
                LoadI32 => builder.constant(TraceConstant::Int32(inst.x as i32)),
                LoadConst => {
                    let c = match callee.chunk.constants.get(inst.x as usize) {
                        Some(Constant::Null) => TraceConstant::Null,
                        Some(Constant::Bool(v)) => TraceConstant::Bool(*v),
                        Some(Constant::Num(v)) => TraceConstant::NumBits(v.to_bits()),
                        Some(Constant::Int32(v)) => TraceConstant::Int32(*v),
                        _ => return Err(RecordingAbort::UnsupportedCall),
                    };
                    builder.constant(c)
                }
                MoveFromArg => *args.get(inst.b as usize).ok_or(RecordingAbort::UnsupportedCall)?,
                MoveFromSpecial if inst.b == 0 => receiver.ok_or(RecordingAbort::UnsupportedCall)?,
                LoadField => {
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    builder.emit1(TraceInst::HeapLoadField { object, slot, exit }, TraceType::Any, TraceRep::Tagged)
                }
                LoadCell => {
                    let cell = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::HeapLoadCell { cell, exit }, TraceType::Any, TraceRep::Tagged)
                }
                ListLen | MapLen | StringLen => {
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    let kind = match inst.opcode {
                        ListLen => TraceInst::HeapListLen { list: object, exit },
                        MapLen => TraceInst::HeapMapLen { map: object, exit },
                        _ => TraceInst::HeapStringLen { string: object, exit },
                    };
                    builder.emit1(kind, TraceType::Int32, TraceRep::Int32)
                }
                ListGet => {
                    let list = scalar(inst.b, TraceRep::Tagged)?;
                    let index = scalar(inst.c, TraceRep::Int32)?;
                    builder.emit1(TraceInst::HeapListGet { list, index, exit }, TraceType::Any, TraceRep::Tagged)
                }
                GuardBounds => {
                    let index = scalar(inst.a, TraceRep::Int32)?;
                    let length = scalar(inst.b, TraceRep::Int32)?;
                    builder.emit0(TraceInst::GuardBounds { index, length, exit });
                    pc = inst.next_pc;
                    continue;
                }
                NumToF64 | GuardF64 => {
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardF64 { value, exit }, TraceType::F64, TraceRep::F64)
                }
                NumToInt32 | GuardInt32 => {
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardInt32 { value, exit }, TraceType::Int32, TraceRep::Int32)
                }
                Int32ToNum => {
                    let value = scalar(inst.b, TraceRep::Int32)?;
                    builder.emit1(TraceInst::BoxInt32 { value }, TraceType::Num, TraceRep::Tagged)
                }
                F64ToNum => {
                    let value = scalar(inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::BoxF64 { value }, TraceType::Num, TraceRep::Tagged)
                }
                INeg | IBitNot => {
                    let value = scalar(inst.b, TraceRep::Int32)?;
                    let unary = if inst.opcode == INeg { IntUnaryOp::Neg } else { IntUnaryOp::BitNot };
                    builder.emit1(TraceInst::IntUnary { op: unary, value }, TraceType::Int32, TraceRep::Int32)
                }
                IAdd | ISub | IMul | IBitAnd | IBitOr | IBitXor | IShl | IShrU | ISar => {
                    let (lhs, rhs) = (scalar(inst.b, TraceRep::Int32)?, scalar(inst.c, TraceRep::Int32)?);
                    let bin = match inst.opcode {
                        IAdd => IntBinOp::Add, ISub => IntBinOp::Sub, IMul => IntBinOp::Mul,
                        IBitAnd => IntBinOp::BitAnd, IBitOr => IntBinOp::BitOr, IBitXor => IntBinOp::BitXor,
                        IShl => IntBinOp::Shl, IShrU => IntBinOp::ShrU, _ => IntBinOp::Sar,
                    };
                    builder.emit1(TraceInst::IntBinary { op: bin, lhs, rhs }, TraceType::Int32, TraceRep::Int32)
                }
                FNeg => {
                    let value = scalar(inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value }, TraceType::F64, TraceRep::F64)
                }
                ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe
                | FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe => {
                    let integer = matches!(inst.opcode, ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe);
                    let rep = if integer { TraceRep::Int32 } else { TraceRep::F64 };
                    let (lhs, rhs) = (scalar(inst.b, rep)?, scalar(inst.c, rep)?);
                    let cmp = match inst.opcode {
                        ICmpEq | FCmpEq => CmpKind::Eq, ICmpNe | FCmpNe => CmpKind::Ne,
                        ICmpLt | FCmpLt => CmpKind::Lt, ICmpLe | FCmpLe => CmpKind::Le,
                        ICmpGt | FCmpGt => CmpKind::Gt, _ => CmpKind::Ge,
                    };
                    let kind = if integer { TraceInst::IntCompare { op: cmp, lhs, rhs } }
                        else { TraceInst::F64Compare { op: cmp, lhs, rhs } };
                    let condition = builder.emit1(kind, TraceType::Bool, TraceRep::Bool);
                    let if_true = builder.constant(TraceConstant::Bool(true));
                    let if_false = builder.constant(TraceConstant::Bool(false));
                    builder.emit1(TraceInst::Select { condition, if_true, if_false }, TraceType::Bool, TraceRep::Tagged)
                }
                FAdd | FSub | FMul | FDiv | FRem => {
                    let (lhs, rhs) = (scalar(inst.b, TraceRep::F64)?, scalar(inst.c, TraceRep::F64)?);
                    let bin = match inst.opcode { FAdd => F64BinOp::Add, FSub => F64BinOp::Sub, FMul => F64BinOp::Mul, FDiv => F64BinOp::Div, _ => F64BinOp::Rem };
                    builder.emit1(TraceInst::F64Binary { op: bin, lhs, rhs }, TraceType::F64, TraceRep::F64)
                }
                Invoke if runtime.trace_standard_numeric_method(target, inst.x) => {
                    let cache = callee.chunk.inline_caches.get(inst.x as usize).ok_or(RecordingAbort::UnsupportedCall)?;
                    let signature = program.metadata.method(cache.method).ok_or(RecordingAbort::UnsupportedCall)?;
                    if signature.kind != MethodKind::Method || signature.arity != 1 || inst.args.len() != 1 { return Err(RecordingAbort::UnsupportedCall); }
                    let bin = match program.metadata.string(signature.name) {
                        Some("+") => F64BinOp::Add, Some("-") => F64BinOp::Sub,
                        Some("*") => F64BinOp::Mul, Some("/") => F64BinOp::Div,
                        Some("%") => F64BinOp::Rem, _ => return Err(RecordingAbort::UnsupportedCall),
                    };
                    let (left, right) = (scalar(inst.b, TraceRep::Tagged)?, scalar(inst.args[0].0, TraceRep::Tagged)?);
                    let lhs = builder.emit1(TraceInst::GuardF64 { value: left, exit }, TraceType::F64, TraceRep::F64);
                    let rhs = builder.emit1(TraceInst::GuardF64 { value: right, exit }, TraceType::F64, TraceRep::F64);
                    let result = builder.emit1(TraceInst::F64Binary { op: bin, lhs, rhs }, TraceType::F64, TraceRep::F64);
                    builder.emit1(TraceInst::BoxF64 { value: result }, TraceType::Num, TraceRep::Tagged)
                }
                Return => {
                    let result = scalar(inst.a, TraceRep::Tagged)?;
                    self.write_gpr(builder, op.a, result);
                    builder.write_slot(FrameSlot::Return(ReturnReg(0)), result);
                    builder.trace_mut().stats.inlined_calls += 1;
                    builder.trace_mut().stats.devirtualized_calls += class.is_some() as usize;
                    return Ok(ExpectedStep::Inlined { next: TraceLocation::new(op.location.function, op.next_pc), callee: target, steps: count });
                }
                _ => return Err(RecordingAbort::UnsupportedCall),
            };
            registers[inst.a as usize] = Some(value);
            pc = inst.next_pc;
        }
        Err(RecordingAbort::UnsupportedCall)
    }
}
