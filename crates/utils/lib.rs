//! Modernized Rust descendants of three small runtime experiments:
//! stackful fibers, a microtask/timer event loop, and a small-buffer
//! type-erased callable.
//!
//! The modules are intentionally independent. WrenJIT can reuse none, one,
//! or all of them without pulling in a third-party async runtime.

pub mod event_loop;
pub mod fiber;
pub mod function;

pub use event_loop::{EventLoop, EventLoopError, Vow, VowState};
pub use fiber::{Fiber, FiberError, FiberRunState};
pub use function::Function;
