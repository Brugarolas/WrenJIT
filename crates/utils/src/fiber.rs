//! Stackful fibers with native context switching.
//!
//! Supported backends mirror the useful subset of the old NanoFiber design:
//! - x86_64 System V (Linux/macOS): hand-written context switch;
//! - AArch64 AAPCS64 (Linux/macOS): hand-written context switch;
//! - x86_64 Windows: Win32 Fiber API.
//!
//! A `Fiber` is deliberately `!Send`/`!Sync`. A suspended native Rust stack may
//! contain thread-affine state or `!Send` guards, so silently migrating one via
//! work stealing would be unsound. WrenJIT's VM-level `FiberState` is the right
//! representation for migratable N:M tasks; this module remains useful for
//! native stackful coroutines and archaeology-compatible experiments.

use std::any::Any;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

#[cfg(all(
    not(windows),
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
compile_error!("nano_runtime::fiber supports Unix x86_64/AArch64 only");

#[cfg(all(windows, not(target_arch = "x86_64")))]
compile_error!("nano_runtime::fiber currently supports Windows x86_64 only");

const MIN_STACK_SIZE: usize = 16 * 1024;
const STACK_ALIGN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberRunState {
    New,
    Running,
    Suspended,
    Completed,
    Panicked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiberError {
    StackTooSmall { requested: usize, minimum: usize },
    AllocationFailed,
    Finished,
    NotInsideFiber,
    Platform(&'static str),
}

impl fmt::Display for FiberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StackTooSmall { requested, minimum } => {
                write!(f, "fiber stack is too small: {requested} bytes (minimum {minimum})")
            }
            Self::AllocationFailed => f.write_str("fiber stack allocation failed"),
            Self::Finished => f.write_str("cannot resume a completed/panicked fiber"),
            Self::NotInsideFiber => f.write_str("operation requires a running nano fiber"),
            Self::Platform(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for FiberError {}

/// String-keyed, type-checked fiber-local storage.
///
/// The C++ original stored raw `void*` values. Here ownership belongs to the
/// Fiber and a key can only be read back as the type that was stored.
#[derive(Default)]
pub struct FiberLocalStorage {
    values: HashMap<String, Box<dyn Any>>,
}

impl FiberLocalStorage {
    pub fn set<T: 'static>(&mut self, key: impl Into<String>, value: T) {
        self.values.insert(key.into(), Box::new(value));
    }

    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.values.get(key)?.downcast_ref::<T>()
    }

    pub fn get_mut<T: 'static>(&mut self, key: &str) -> Option<&mut T> {
        self.values.get_mut(key)?.downcast_mut::<T>()
    }

    pub fn remove<T: 'static>(&mut self, key: &str) -> Option<T> {
        self.values.remove(key)?.downcast::<T>().ok().map(|boxed| *boxed)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

#[cfg(not(windows))]
mod unix_context {
    use super::*;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    #[cfg(target_os = "linux")]
    const MAP_ANON: i32 = 0x20;
    #[cfg(target_os = "macos")]
    const MAP_ANON: i32 = 0x1000;

    unsafe extern "C" {
        fn getpagesize() -> i32;
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: isize,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, length: usize, prot: i32) -> i32;
        fn munmap(addr: *mut c_void, length: usize) -> i32;
    }

    /// Native stack backed by mmap with a no-access guard page at its low end.
    /// Both supported Unix architectures grow stacks downward, so ordinary
    /// overflow faults at the guard page instead of corrupting the heap.
    pub struct Stack {
        mapping: NonNull<u8>,
        mapping_size: usize,
        usable_size: usize,
    }

    impl Stack {
        pub fn new(size: usize) -> Result<Self, FiberError> {
            let page = unsafe { getpagesize() };
            if page <= 0 {
                return Err(FiberError::Platform("getpagesize failed"));
            }
            let page = page as usize;
            let usable_size = size
                .checked_add(page - 1)
                .ok_or(FiberError::AllocationFailed)?
                / page
                * page;
            let mapping_size = usable_size
                .checked_add(page)
                .ok_or(FiberError::AllocationFailed)?;

            let raw = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    mapping_size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            };
            if raw as isize == -1 {
                return Err(FiberError::AllocationFailed);
            }
            let mapping = NonNull::new(raw.cast::<u8>()).ok_or(FiberError::AllocationFailed)?;

            if unsafe { mprotect(raw, page, PROT_NONE) } != 0 {
                unsafe { let _ = munmap(raw, mapping_size); }
                return Err(FiberError::Platform("mprotect guard page failed"));
            }

            Ok(Self { mapping, mapping_size, usable_size })
        }

        #[inline(always)]
        pub fn top(&self) -> *mut u8 {
            // One guard page precedes the usable region. Top is simply the end
            // of the complete mapping and remains page/16-byte aligned.
            unsafe { self.mapping.as_ptr().add(self.mapping_size) }
        }

        #[allow(dead_code)]
        pub fn usable_size(&self) -> usize {
            self.usable_size
        }
    }

    impl Drop for Stack {
        fn drop(&mut self) {
            unsafe { let _ = munmap(self.mapping.as_ptr().cast(), self.mapping_size); }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[repr(C)]
    #[derive(Default)]
    pub struct Context {
        rbx: u64,
        rbp: u64,
        r12: u64,
        r13: u64,
        r14: u64,
        r15: u64,
        rsp: u64,
        rip: u64,
        arg0: u64,
        mxcsr: u32,
        x87_cw: u16,
        _pad: u16,
    }

    #[cfg(target_arch = "aarch64")]
    #[repr(C, align(16))]
    pub struct Context {
        x19: u64,
        x20: u64,
        x21: u64,
        x22: u64,
        x23: u64,
        x24: u64,
        x25: u64,
        x26: u64,
        x27: u64,
        x28: u64,
        x29: u64,
        x30: u64,
        sp: u64,
        arg0: u64,
        q8: u128,
        q9: u128,
        q10: u128,
        q11: u128,
        q12: u128,
        q13: u128,
        q14: u128,
        q15: u128,
        fpcr: u64,
        fpsr: u64,
    }

    #[cfg(target_arch = "aarch64")]
    impl Default for Context {
        fn default() -> Self {
            // All-zero is a valid dormant context before initialization.
            unsafe { std::mem::zeroed() }
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    core::arch::global_asm!(r#"
        .text
        .att_syntax
        .p2align 4
        .globl nano_switch_fiber
        .type nano_switch_fiber,@function
    nano_switch_fiber:
        movq %rbx,  0(%rdi)
        movq %rbp,  8(%rdi)
        movq %r12, 16(%rdi)
        movq %r13, 24(%rdi)
        movq %r14, 32(%rdi)
        movq %r15, 40(%rdi)
        movq (%rsp), %rax
        movq %rax, 56(%rdi)
        leaq 8(%rsp), %rax
        movq %rax, 48(%rdi)
        stmxcsr 72(%rdi)
        fnstcw 76(%rdi)

        movq %rsi, %rax
        movq  0(%rax), %rbx
        movq  8(%rax), %rbp
        movq 16(%rax), %r12
        movq 24(%rax), %r13
        movq 32(%rax), %r14
        movq 40(%rax), %r15
        ldmxcsr 72(%rax)
        fldcw 76(%rax)
        movq 64(%rax), %rdi
        movq 48(%rax), %rsp
        jmp *56(%rax)
    "#);

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    core::arch::global_asm!(r#"
        .text
        .att_syntax
        .p2align 4
        .globl _nano_switch_fiber
    _nano_switch_fiber:
        movq %rbx,  0(%rdi)
        movq %rbp,  8(%rdi)
        movq %r12, 16(%rdi)
        movq %r13, 24(%rdi)
        movq %r14, 32(%rdi)
        movq %r15, 40(%rdi)
        movq (%rsp), %rax
        movq %rax, 56(%rdi)
        leaq 8(%rsp), %rax
        movq %rax, 48(%rdi)
        stmxcsr 72(%rdi)
        fnstcw 76(%rdi)

        movq %rsi, %rax
        movq  0(%rax), %rbx
        movq  8(%rax), %rbp
        movq 16(%rax), %r12
        movq 24(%rax), %r13
        movq 32(%rax), %r14
        movq 40(%rax), %r15
        ldmxcsr 72(%rax)
        fldcw 76(%rax)
        movq 64(%rax), %rdi
        movq 48(%rax), %rsp
        jmp *56(%rax)
    "#);

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    core::arch::global_asm!(r#"
        .text
        .att_syntax
        .p2align 4
        .globl nano_switch_fiber
        .type nano_switch_fiber,%function
    nano_switch_fiber:
        stp x19, x20, [x0, #0]
        stp x21, x22, [x0, #16]
        stp x23, x24, [x0, #32]
        stp x25, x26, [x0, #48]
        stp x27, x28, [x0, #64]
        stp x29, x30, [x0, #80]
        mov x10, sp
        str x10, [x0, #96]
        stp q8,  q9,  [x0, #112]
        stp q10, q11, [x0, #144]
        stp q12, q13, [x0, #176]
        stp q14, q15, [x0, #208]
        mrs x10, fpcr
        str x10, [x0, #240]
        mrs x10, fpsr
        str x10, [x0, #248]

        mov x9, x1
        ldp x19, x20, [x9, #0]
        ldp x21, x22, [x9, #16]
        ldp x23, x24, [x9, #32]
        ldp x25, x26, [x9, #48]
        ldp x27, x28, [x9, #64]
        ldp x29, x30, [x9, #80]
        ldr x10, [x9, #96]
        mov sp, x10
        ldp q8,  q9,  [x9, #112]
        ldp q10, q11, [x9, #144]
        ldp q12, q13, [x9, #176]
        ldp q14, q15, [x9, #208]
        ldr x10, [x9, #240]
        msr fpcr, x10
        ldr x10, [x9, #248]
        msr fpsr, x10
        ldr x0, [x9, #104]
        ret
    "#);

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    core::arch::global_asm!(r#"
        .text
        .att_syntax
        .p2align 4
        .globl _nano_switch_fiber
    _nano_switch_fiber:
        stp x19, x20, [x0, #0]
        stp x21, x22, [x0, #16]
        stp x23, x24, [x0, #32]
        stp x25, x26, [x0, #48]
        stp x27, x28, [x0, #64]
        stp x29, x30, [x0, #80]
        mov x10, sp
        str x10, [x0, #96]
        stp q8,  q9,  [x0, #112]
        stp q10, q11, [x0, #144]
        stp q12, q13, [x0, #176]
        stp q14, q15, [x0, #208]
        mrs x10, fpcr
        str x10, [x0, #240]
        mrs x10, fpsr
        str x10, [x0, #248]

        mov x9, x1
        ldp x19, x20, [x9, #0]
        ldp x21, x22, [x9, #16]
        ldp x23, x24, [x9, #32]
        ldp x25, x26, [x9, #48]
        ldp x27, x28, [x9, #64]
        ldp x29, x30, [x9, #80]
        ldr x10, [x9, #96]
        mov sp, x10
        ldp q8,  q9,  [x9, #112]
        ldp q10, q11, [x9, #144]
        ldp q12, q13, [x9, #176]
        ldp q14, q15, [x9, #208]
        ldr x10, [x9, #240]
        msr fpcr, x10
        ldr x10, [x9, #248]
        msr fpsr, x10
        ldr x0, [x9, #104]
        ret
    "#);

    unsafe extern "C" {
        fn nano_switch_fiber(from: *mut Context, to: *const Context);
    }

    #[inline(never)]
    extern "C" fn return_trap() -> ! {
        std::process::abort();
    }

    pub unsafe fn initialize(
        context: &mut Context,
        stack: &Stack,
        entry: extern "C" fn(*mut super::FiberCore) -> !,
        arg: *mut super::FiberCore,
    ) {
        *context = Context::default();

        #[cfg(target_arch = "x86_64")]
        {
            let top = stack.top() as usize;
            let rsp = (top - core::mem::size_of::<usize>()) as *mut usize;
            // SAFETY: rsp is inside the allocated stack and aligned as a SysV
            // function entry (RSP % 16 == 8). The fake return address catches an
            // impossible accidental return from the trampoline.
            unsafe { rsp.write(return_trap as usize) };
            context.rsp = rsp as u64;
            context.rip = entry as usize as u64;
            context.arg0 = arg as usize as u64;
            unsafe {
                core::arch::asm!(
                    "stmxcsr [{mx}]",
                    "fnstcw [{cw}]",
                    mx = in(reg) &mut context.mxcsr,
                    cw = in(reg) &mut context.x87_cw,
                    options(nostack, preserves_flags),
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            context.sp = stack.top() as usize as u64;
            context.x30 = entry as usize as u64;
            context.arg0 = arg as usize as u64;
            unsafe {
                core::arch::asm!(
                    "mrs {fpcr}, fpcr",
                    "mrs {fpsr}, fpsr",
                    fpcr = out(reg) context.fpcr,
                    fpsr = out(reg) context.fpsr,
                    options(nostack, preserves_flags),
                );
            }
        }
    }

    #[inline(always)]
    pub unsafe fn switch(from: &mut Context, to: &Context) {
        // SAFETY: caller guarantees both contexts belong to live stacks.
        unsafe { nano_switch_fiber(from, to) };
    }
}

#[cfg(windows)]
mod windows_context {
    use super::*;
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::ptr;

    type FiberHandle = *mut c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ConvertThreadToFiberEx(parameter: *mut c_void, flags: u32) -> FiberHandle;
        fn ConvertFiberToThread() -> i32;
        fn CreateFiberEx(
            commit_size: usize,
            reserve_size: usize,
            flags: u32,
            start: extern "system" fn(*mut c_void),
            parameter: *mut c_void,
        ) -> FiberHandle;
        fn DeleteFiber(fiber: FiberHandle);
        fn SwitchToFiber(fiber: FiberHandle);
        fn IsThreadAFiber() -> i32;
    }

    #[derive(Default)]
    struct ThreadRoot {
        handle: FiberHandle,
        converted_by_us: bool,
    }

    impl Drop for ThreadRoot {
        fn drop(&mut self) {
            if self.converted_by_us {
                // SAFETY: this TLS object belongs to the converted thread.
                unsafe { let _ = ConvertFiberToThread(); }
            }
        }
    }

    thread_local! {
        static ROOT: RefCell<ThreadRoot> = RefCell::new(ThreadRoot::default());
    }

    #[inline(always)]
    unsafe fn get_current_fiber() -> FiberHandle {
        let out: usize;
        // Win64 TEB.NtTib.FiberData == GS:[0x20], which is exactly what the
        // GetCurrentFiber() SDK macro expands to on x86_64.
        unsafe {
            core::arch::asm!("mov {}, gs:[0x20]", out(reg) out, options(nostack, preserves_flags));
        }
        out as FiberHandle
    }

    fn ensure_root() -> Result<FiberHandle, FiberError> {
        ROOT.with(|slot| {
            let mut root = slot.borrow_mut();
            if !root.handle.is_null() {
                return Ok(root.handle);
            }

            // SAFETY: Win32 API, current thread only.
            let already_fiber = unsafe { IsThreadAFiber() != 0 };
            if already_fiber {
                root.handle = unsafe { get_current_fiber() };
                root.converted_by_us = false;
            } else {
                root.handle = unsafe { ConvertThreadToFiberEx(ptr::null_mut(), 0) };
                if root.handle.is_null() {
                    return Err(FiberError::Platform("ConvertThreadToFiberEx failed"));
                }
                root.converted_by_us = true;
            }
            Ok(root.handle)
        })
    }

    extern "system" fn trampoline(arg: *mut c_void) {
        // SAFETY: CreateFiberEx received a stable Box<FiberCore> pointer.
        unsafe { super::fiber_entry(arg.cast::<super::FiberCore>()) }
    }

    pub struct PlatformFiber {
        pub handle: FiberHandle,
        pub caller: FiberHandle,
    }

    impl Default for PlatformFiber {
        fn default() -> Self {
            Self { handle: ptr::null_mut(), caller: ptr::null_mut() }
        }
    }

    impl PlatformFiber {
        pub fn create(stack_size: usize, core: *mut super::FiberCore) -> Result<Self, FiberError> {
            let _ = ensure_root()?;
            // Let Windows choose commit granularity while respecting requested reserve.
            let handle = unsafe { CreateFiberEx(0, stack_size, 0, trampoline, core.cast()) };
            if handle.is_null() {
                return Err(FiberError::Platform("CreateFiberEx failed"));
            }
            Ok(Self { handle, caller: ptr::null_mut() })
        }

        pub unsafe fn resume(&mut self) -> Result<(), FiberError> {
            let _ = ensure_root()?;
            self.caller = unsafe { get_current_fiber() };
            unsafe { SwitchToFiber(self.handle) };
            Ok(())
        }

        pub unsafe fn yield_to_caller(&mut self) {
            debug_assert!(!self.caller.is_null());
            unsafe { SwitchToFiber(self.caller) };
        }
    }

    impl Drop for PlatformFiber {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { DeleteFiber(self.handle) };
                self.handle = ptr::null_mut();
            }
        }
    }
}

thread_local! {
    static CURRENT_FIBER: Cell<*mut FiberCore> = const { Cell::new(std::ptr::null_mut()) };
}

pub(crate) struct FiberCore {
    state: FiberRunState,
    entry: Option<Box<dyn FnOnce()>>,
    panic: Option<Box<dyn Any + Send>>,
    locals: FiberLocalStorage,

    #[cfg(not(windows))]
    context: unix_context::Context,
    #[cfg(not(windows))]
    caller_context: unix_context::Context,
    #[cfg(not(windows))]
    stack: unix_context::Stack,

    #[cfg(windows)]
    platform: windows_context::PlatformFiber,
}

/// Native stackful fiber.
///
/// Deliberately not `Send` or `Sync`; see module documentation.
pub struct Fiber {
    core: Box<FiberCore>,
    _thread_affine: PhantomData<Rc<()>>,
}

impl Fiber {
    pub fn new<F>(stack_size: usize, entry: F) -> Result<Self, FiberError>
    where
        F: FnOnce() + 'static,
    {
        if stack_size < MIN_STACK_SIZE {
            return Err(FiberError::StackTooSmall {
                requested: stack_size,
                minimum: MIN_STACK_SIZE,
            });
        }

        #[cfg(not(windows))]
        {
            let stack = unix_context::Stack::new(stack_size)?;
            let mut core = Box::new(FiberCore {
                state: FiberRunState::New,
                entry: Some(Box::new(entry)),
                panic: None,
                locals: FiberLocalStorage::default(),
                context: unix_context::Context::default(),
                caller_context: unix_context::Context::default(),
                stack,
            });
            let ptr = &mut *core as *mut FiberCore;
            unsafe {
                unix_context::initialize(&mut core.context, &core.stack, fiber_entry, ptr);
            }
            Ok(Self { core, _thread_affine: PhantomData })
        }

        #[cfg(windows)]
        {
            let mut core = Box::new(FiberCore {
                state: FiberRunState::New,
                entry: Some(Box::new(entry)),
                panic: None,
                locals: FiberLocalStorage::default(),
                platform: windows_context::PlatformFiber::default(),
            });
            let ptr = &mut *core as *mut FiberCore;
            core.platform = windows_context::PlatformFiber::create(stack_size, ptr)?;
            Ok(Self { core, _thread_affine: PhantomData })
        }
    }

    #[inline(always)]
    pub fn state(&self) -> FiberRunState {
        self.core.state
    }

    #[inline(always)]
    pub fn is_finished(&self) -> bool {
        matches!(self.core.state, FiberRunState::Completed | FiberRunState::Panicked)
    }

    pub fn locals(&self) -> &FiberLocalStorage {
        &self.core.locals
    }

    pub fn locals_mut(&mut self) -> &mut FiberLocalStorage {
        &mut self.core.locals
    }

    /// Resume until the fiber yields, completes, or panics.
    pub fn resume(&mut self) -> Result<FiberRunState, FiberError> {
        if self.is_finished() {
            return Err(FiberError::Finished);
        }

        self.core.state = FiberRunState::Running;
        let core_ptr = &mut *self.core as *mut FiberCore;
        let previous = CURRENT_FIBER.with(|slot| {
            let old = slot.get();
            slot.set(core_ptr);
            old
        });

        #[cfg(not(windows))]
        unsafe {
            unix_context::switch(&mut self.core.caller_context, &self.core.context);
        }

        #[cfg(windows)]
        let platform_result = unsafe { self.core.platform.resume() };

        CURRENT_FIBER.with(|slot| slot.set(previous));

        #[cfg(windows)]
        platform_result?;

        Ok(self.core.state)
    }

    /// Resume and propagate a panic from inside the fiber on the caller stack.
    pub fn resume_or_panic(&mut self) -> Result<FiberRunState, FiberError> {
        let state = self.resume()?;
        if state == FiberRunState::Panicked {
            if let Some(payload) = self.core.panic.take() {
                std::panic::resume_unwind(payload);
            }
        }
        Ok(state)
    }

    /// Cooperatively suspend the currently running nano fiber.
    pub fn yield_now() -> Result<(), FiberError> {
        CURRENT_FIBER.with(|slot| {
            let ptr = slot.get();
            if ptr.is_null() {
                return Err(FiberError::NotInsideFiber);
            }

            // SAFETY: CURRENT_FIBER is installed only while this Fiber is running
            // on the current thread. Fiber is !Send, so there is no cross-thread
            // concurrent mutation.
            let core = unsafe { &mut *ptr };
            core.state = FiberRunState::Suspended;

            #[cfg(not(windows))]
            unsafe {
                unix_context::switch(&mut core.context, &core.caller_context);
            }

            #[cfg(windows)]
            unsafe {
                core.platform.yield_to_caller();
            }

            core.state = FiberRunState::Running;
            Ok(())
        })
    }

    /// Access the current fiber's local storage without leaking references past
    /// the callback. This fixes the old untyped `void*` FLS design.
    pub fn with_current_locals<R>(
        f: impl FnOnce(&mut FiberLocalStorage) -> R,
    ) -> Result<R, FiberError> {
        CURRENT_FIBER.with(|slot| {
            let ptr = slot.get();
            if ptr.is_null() {
                return Err(FiberError::NotInsideFiber);
            }
            let core = unsafe { &mut *ptr };
            Ok(f(&mut core.locals))
        })
    }

    pub fn take_panic(&mut self) -> Option<Box<dyn Any + Send>> {
        self.core.panic.take()
    }
}

#[cfg(not(windows))]
extern "C" fn fiber_entry(core: *mut FiberCore) -> ! {
    // SAFETY: initialized context receives the stable Box<FiberCore> pointer.
    unsafe { run_fiber_entry(core) }
}

#[cfg(windows)]
unsafe fn fiber_entry(core: *mut FiberCore) -> ! {
    unsafe { run_fiber_entry(core) }
}

unsafe fn run_fiber_entry(core: *mut FiberCore) -> ! {
    let core = unsafe { &mut *core };
    core.state = FiberRunState::Running;

    let entry = core.entry.take().expect("fiber entry already consumed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(entry));
    match result {
        Ok(()) => core.state = FiberRunState::Completed,
        Err(payload) => {
            core.panic = Some(payload);
            core.state = FiberRunState::Panicked;
        }
    }

    #[cfg(not(windows))]
    unsafe {
        unix_context::switch(&mut core.context, &core.caller_context);
    }

    #[cfg(windows)]
    unsafe {
        core.platform.yield_to_caller();
    }

    // Completed fibers are never legally resumed.
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::{Fiber, FiberRunState};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn yield_and_resume() {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let out = trace.clone();
        let mut fiber = Fiber::new(64 * 1024, move || {
            out.borrow_mut().push(1);
            Fiber::yield_now().unwrap();
            out.borrow_mut().push(3);
        }).unwrap();

        assert_eq!(fiber.resume().unwrap(), FiberRunState::Suspended);
        trace.borrow_mut().push(2);
        assert_eq!(fiber.resume().unwrap(), FiberRunState::Completed);
        assert_eq!(&*trace.borrow(), &[1, 2, 3]);
    }

    #[test]
    fn fiber_local_storage_survives_yield() {
        let mut fiber = Fiber::new(64 * 1024, || {
            Fiber::with_current_locals(|locals| locals.set("answer", 41u32)).unwrap();
            Fiber::yield_now().unwrap();
            Fiber::with_current_locals(|locals| {
                *locals.get_mut::<u32>("answer").unwrap() += 1;
            }).unwrap();
        }).unwrap();

        assert_eq!(fiber.resume().unwrap(), FiberRunState::Suspended);
        assert_eq!(fiber.locals().get::<u32>("answer"), Some(&41));
        assert_eq!(fiber.resume().unwrap(), FiberRunState::Completed);
        assert_eq!(fiber.locals().get::<u32>("answer"), Some(&42));
    }

    #[test]
    fn panic_is_captured() {
        let mut fiber = Fiber::new(64 * 1024, || panic!("boom")).unwrap();
        assert_eq!(fiber.resume().unwrap(), FiberRunState::Panicked);
        assert!(fiber.take_panic().is_some());
    }
}
