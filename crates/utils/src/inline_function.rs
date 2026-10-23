//! A small-buffer optimized type-erased callable.
//!
//! The old C++ implementation had the right high-level idea (keep small
//! callables inline), but mixed object lifetime, heap metadata and move logic
//! in ways that made large callables and non-trivial moves unsafe. This version
//! keeps the same spirit while using Rust's ownership model.

use core::marker::PhantomData;
use core::mem::{align_of, size_of, MaybeUninit};
use core::ptr;

/// Alignment offered by the inline buffer.
///
/// 16 is enough for ordinary closures and keeps the whole object cache-friendly.
/// Types requiring stricter alignment automatically fall back to `Box`.
pub const INLINE_ALIGN: usize = 64;

/// Trait used internally to adapt ordinary Rust closures to tupled arguments.
///
/// `Function<(A, B), R>` can therefore store a closure written naturally as
/// `|a: A, b: B| -> R { ... }`.
pub trait Invoke<Args> {
    type Output;
    fn invoke(&mut self, args: Args) -> Self::Output;
}

impl<F, R> Invoke<()> for F
where
    F: FnMut() -> R,
{
    type Output = R;
    #[inline(always)]
    fn invoke(&mut self, (): ()) -> R {
        self()
    }
}

macro_rules! impl_invoke_tuple {
    ($(($($T:ident),+)),+ $(,)?) => {
        $(
            impl<F, R, $($T),+> Invoke<($($T,)+)> for F
            where
                F: FnMut($($T),+) -> R,
            {
                type Output = R;
                #[allow(non_snake_case)]
                #[inline(always)]
                fn invoke(&mut self, args: ($($T,)+)) -> R {
                    let ($($T,)+) = args;
                    self($($T),+)
                }
            }
        )+
    };
}

impl_invoke_tuple! {
    (A0),
    (A0, A1),
    (A0, A1, A2),
    (A0, A1, A2, A3),
    (A0, A1, A2, A3, A4),
    (A0, A1, A2, A3, A4, A5),
    (A0, A1, A2, A3, A4, A5, A6),
    (A0, A1, A2, A3, A4, A5, A6, A7),
}

#[repr(C, align(64))]
struct InlineStorage<const N: usize> {
    bytes: [MaybeUninit<u8>; N],
}

impl<const N: usize> InlineStorage<N> {
    #[inline(always)]
    const fn new() -> Self {
        Self { bytes: [MaybeUninit::uninit(); N] }
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr().cast::<u8>()
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr().cast::<u8>()
    }
}

struct VTable<Args, R> {
    call: unsafe fn(*mut u8, Args) -> R,
    drop_inline: unsafe fn(*mut u8),
    drop_heap: unsafe fn(*mut u8),
}

unsafe fn call_impl<F, Args, R>(ptr: *mut u8, args: Args) -> R
where
    F: Invoke<Args, Output = R>,
{
    // SAFETY: constructor installs this vtable only for storage containing F.
    let f = unsafe { &mut *ptr.cast::<F>() };
    f.invoke(args)
}

unsafe fn drop_inline_impl<F>(ptr: *mut u8) {
    // SAFETY: ptr points at an initialized inline F and is dropped exactly once.
    unsafe { ptr::drop_in_place(ptr.cast::<F>()) };
}

unsafe fn drop_heap_impl<F>(ptr: *mut u8) {
    // SAFETY: ptr came from Box::into_raw(Box<F>) and is consumed exactly once.
    unsafe { drop(Box::from_raw(ptr.cast::<F>())) };
}

/// Type-erased callable with small-buffer optimization.
///
/// The default inline capacity is 64 bytes, deliberately tiny compared with the
/// old 4 KiB/4 MiB experiments. Large or over-aligned closures transparently use
/// one `Box` allocation.
///
/// # Threading
/// Stored callables must be `Send`; consequently a `Function` may be moved to a
/// scheduler worker thread. It is intentionally not `Sync` because invocation
/// requires `&mut self` (`FnMut` semantics).
pub struct Function<Args, R, const INLINE: usize = 64> {
    inline: InlineStorage<INLINE>,
    heap: *mut u8,
    is_inline: bool,
    vtable: VTable<Args, R>,
    _marker: PhantomData<fn(Args) -> R>,
}

impl<Args, R, const INLINE: usize> Function<Args, R, INLINE> {
    /// Construct a new type-erased callable.
    ///
    /// `Unpin` is required for inline storage because moving `Function` moves the
    /// inline bytes too. A deliberately self-referential `!Unpin` callable falls
    /// back neither silently nor unsafely; wrap it in a `Box` yourself if needed.
    pub fn new<F>(f: F) -> Self
    where
        F: Invoke<Args, Output = R> + Send + Unpin + 'static,
    {
        let vtable = VTable {
            call: call_impl::<F, Args, R>,
            drop_inline: drop_inline_impl::<F>,
            drop_heap: drop_heap_impl::<F>,
        };

        let fits_inline = size_of::<F>() <= INLINE && align_of::<F>() <= INLINE_ALIGN;
        let mut out = Self {
            inline: InlineStorage::new(),
            heap: ptr::null_mut(),
            is_inline: fits_inline,
            vtable,
            _marker: PhantomData,
        };

        if fits_inline {
            // SAFETY: size/alignment were checked above and the destination is
            // currently uninitialized.
            unsafe { out.inline.as_mut_ptr().cast::<F>().write(f) };
        } else {
            out.heap = Box::into_raw(Box::new(f)).cast::<u8>();
        }
        out
    }

    #[inline(always)]
    fn data_mut_ptr(&mut self) -> *mut u8 {
        if self.is_inline {
            self.inline.as_mut_ptr()
        } else {
            debug_assert!(!self.heap.is_null());
            self.heap
        }
    }

    #[inline(always)]
    fn data_ptr(&self) -> *const u8 {
        if self.is_inline {
            self.inline.as_ptr()
        } else {
            debug_assert!(!self.heap.is_null());
            self.heap
        }
    }

    /// Invoke the callable with tupled arguments.
    #[inline(always)]
    pub fn call(&mut self, args: Args) -> R {
        let ptr = self.data_mut_ptr();
        // SAFETY: vtable and payload are created together in `new`.
        unsafe { (self.vtable.call)(ptr, args) }
    }

    /// Whether the callable lives entirely inside this object.
    #[inline(always)]
    pub const fn is_inline(&self) -> bool {
        self.is_inline
    }

    #[inline(always)]
    pub const fn inline_capacity(&self) -> usize {
        INLINE
    }

    /// Address of the erased callable, useful for diagnostics/benchmarks.
    #[inline(always)]
    pub fn data_address(&self) -> usize {
        self.data_ptr() as usize
    }
}

// SAFETY: `new` only accepts F: Send. The erased payload is either moved as
// ordinary Unpin bytes or through its Box pointer. Invocation still needs &mut.
unsafe impl<Args, R, const INLINE: usize> Send for Function<Args, R, INLINE> {}

impl<Args, R, const INLINE: usize> Drop for Function<Args, R, INLINE> {
    fn drop(&mut self) {
        if self.is_inline {
            let ptr = self.inline.as_mut_ptr();
            // SAFETY: inline payload is initialized and owned by self.
            unsafe { (self.vtable.drop_inline)(ptr) };
        } else if !self.heap.is_null() {
            let ptr = self.heap;
            self.heap = ptr::null_mut();
            // SAFETY: heap payload came from Box::into_raw in `new`.
            unsafe { (self.vtable.drop_heap)(ptr) };
        }
    }
}

impl<R, const INLINE: usize> Function<(), R, INLINE> {
    #[inline(always)]
    pub fn call0(&mut self) -> R {
        self.call(())
    }
}

macro_rules! impl_call_helpers {
    ($(($name:ident, ($($T:ident : $v:ident),+))),+ $(,)?) => {
        $(
            impl<R, $($T),+, const INLINE: usize> Function<($($T,)+), R, INLINE> {
                #[inline(always)]
                pub fn $name(&mut self, $($v: $T),+) -> R {
                    self.call(($($v,)+))
                }
            }
        )+
    };
}

impl_call_helpers! {
    (call1, (A0: a0)),
    (call2, (A0: a0, A1: a1)),
    (call3, (A0: a0, A1: a1, A2: a2)),
    (call4, (A0: a0, A1: a1, A2: a2, A3: a3)),
}

#[cfg(test)]
mod tests {
    use super::Function;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    #[test]
    fn small_closure_is_inline() {
        let n = 7u32;
        let mut f = Function::<(u32,), u32>::new(move |x| x + n);
        assert!(f.is_inline());
        assert_eq!(f.call1(5), 12);
    }

    #[test]
    fn large_closure_uses_heap() {
        let payload = [7u8; 256];
        let mut f = Function::<(), usize, 32>::new(move || payload.len());
        assert!(!f.is_inline());
        assert_eq!(f.call0(), 256);
    }

    #[test]
    fn destructor_runs_exactly_once() {
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }

        let c = Arc::new(AtomicUsize::new(0));
        {
            let d = DropCounter(c.clone());
            let _f = Function::<(), ()>::new(move || { let _ = &d; });
        }
        assert_eq!(c.load(Ordering::SeqCst), 1);
    }
}
