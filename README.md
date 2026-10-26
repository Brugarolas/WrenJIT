[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/brugarolas)

# WrenJIT

**WrenJIT** is a high-performance, Wren-derived language runtime built around a register VM, a multi-stage optimizing compiler, a tracing JIT backed by Cranelift, precise moving garbage collection, and an extended asynchronous runtime.

WrenJIT began as an attempt to make Wren dramatically faster. It eventually grew beyond that goal: the runtime now contains a register-based direct-threaded interpreter with super-instructions, its own two SSA IR, register bytecode, optimizing interpreter, tracing optimizing JIT, moving GC integration, generational Sticky Immix garbage collector, structured exceptions, `defer`, `async`/`await`, cancellable Promises, microtasks, timers, and a static module graph.

WrenJIT should therefore be understood as **Wren-derived, not as a promise of drop-in compatibility with every official Wren embedding or implementation detail, although compatibility is near or 100% already**.

WrenJIT should work in AMD64 and ARM64, in Linux, Windows, and MacOS: but only AMD64 Windows and ARM64 MacOS has been tested.

---

## Performance comparison

The final release should publish results for three engines:

- **Wren** — official upstream Wren.
- **WrenLift** — comparison engine; this README intentionally makes no implementation or feature claims about it.
- **WrenJIT** — this project.

All portable benchmark scripts under `benchmarks/portable/` intentionally stay inside the common Wren language surface so the same source can be executed by all three engines.

### Absolute times

Lower is better. Fill these cells with **median wall-clock time for the measured region**, using the methodology below.

| Benchmark | Wren | WrenLift | WrenJIT |
|---|---:|---:|---:|
| `mandelbrot` *(+ others)* | 61410 ms | 10218 m | 2296ms |
| `classes` | 1050 ms | 1211 ms | 43ms |
| `gc` | 890 ms | 206 ms | 126ms |

### Relative speedups

Higher is better. Compute speedup as `comparison_time / WrenJIT_time`.

| Benchmark | WrenJIT vs Wren | WrenJIT vs WrenLift |
|---|---:|---:|
| `mandelbrot` *(+ others)* | 26,75x | 4,45x |
| `classes` | 24,42x | 28,16x |
| `gc` | 6,825x | 1,635x |

### Benchmark methodology

For published numbers:

1. Build every engine in its fastest supported release configuration.
2. Use the same machine, OS, CPU governor/power mode, and terminal environment.
3. Avoid comparing a debug build from one engine with a release build from another.
4. Run each benchmark at least **7 times**; report the median.
5. Keep the warm-up embedded in each benchmark. WrenJIT is a JIT and steady-state results without warm-up are not meaningful.
6. Record CPU model, OS, compiler/toolchain, engine revision, and command line.
7. Verify checksums before accepting timing results.
8. Treat process-startup/compile latency as a **separate benchmark**. The in-script `System.clock` measurements intentionally focus on execution throughput.
9. Report regressions as well as wins. A benchmark table is more valuable if it is reproducible than if every row is flattering.

# Project Status

> **Project status:** WrenJIT is entering its final feature-freeze and documentation/compatibility audit. After that audit, this engine will be preserved as the final Wren-derived release. Further language development will continue separately as **Napalm**, which is free to abandon Wren syntax and compatibility entirely.

---

# Language extensions beyond official Wren

Official Wren already has classes, closures, modules and lightweight cooperative Fibers. WrenJIT keeps those concepts, but adds a substantial language/runtime surface of its own.

## `async` methods and closures

WrenJIT adds `async` functions at the method/closure level. Calling an async function returns a Promise instead of recursively entering a second interpreter dispatch loop.

```wren
class Service {
  static async answer() {
    return 42
  }
}

var work = async {|value|
  return value * 2
}
```

Async execution is represented by scheduler-owned VM execution state using the same register/frame machinery as Fibers.

## `await`

`await` suspends the current logical Wren execution until a Promise settles. It is valid inside async code and at module scope.

```wren
var value = await Service.answer()
```

A pending Promise does **not** block the OS thread. The current execution is marked suspended and the scheduler runs another ready continuation or the event loop.

## Promises

WrenJIT has first-class Promises with four terminal/state categories:

```text
Pending -> Fulfilled
        -> Rejected
        -> Cancelled
```

Public operations include:

```wren
Promise.resolve(value)
Promise.reject(error)

promise.cancel()
promise.isPending
promise.isFulfilled
promise.isRejected
promise.isCancelled
```

Returning a Promise from async code adopts the inner Promise rather than producing a nested Promise.

Promise settlement does not recursively execute awaiters. Continuations are queued through the microtask mechanism, avoiding re-entrant execution during `resolve`, `reject`, or `cancel`.

## Cancellable asynchronous operations

A Promise is also the handle for its asynchronous producer. Cancellation is therefore uniform instead of requiring a second timer/task handle type.

`cancel()` returns `true` when cancellation was accepted and `false` when the operation was already terminal or too far into execution to cancel safely.

WrenJIT does not asynchronously tear down arbitrary running Wren code. Cancellation is cooperative and producer-specific.

## Timer/event-loop API

WrenJIT exposes timer primitives through the static `Timer` class:

```wren
await Timer.delay(100)

var value = await Timer.delay(100) {
  return 42
}

var micro = Timer.microtask {
  return doWork()
}

var interval = Timer.every(1000) {
  System.print("tick")
}
interval.cancel()
```

All Timer operations return Promises.

The runtime uses a microtask queue and deadline heap rather than a fixed polling sleep. Idle execution can park until the next deadline. `Timer.every()` deliberately prevents overlapping ticks; a slow callback finishes before the interval is re-armed, and missed periods are skipped instead of creating a catch-up storm.

## Structured exceptions

Official Wren handles runtime errors through Fibers and intentionally has no language-level exception handling. WrenJIT adds conventional structured exceptions:

```wren
try {
  dangerousOperation()
} catch (error) {
  System.print(error)
} finally {
  cleanup()
}

throw value
```

`throw` can throw an arbitrary Wren value. Runtime errors may also enter a `catch` path.

The implementation uses **zero-cost protected bytecode ranges**: normal execution does not execute `EnterTry` / `LeaveTry` opcodes. Handler metadata is consulted only when an exception actually occurs.

## `defer` and `errdefer`

Lexical cleanup is built into the language:

```wren
defer {
  alwaysRun()
}

errdefer (error) {
  rollback(error)
}
```

- `defer` runs in LIFO order on scope exit.
- `errdefer` runs only while unwinding an error.
- `return`, `break`, and `continue` execute required pending cleanup first.
- `finally` runs on both normal and exceptional exit.
- control transfers from inside cleanup bodies are rejected when they would make cleanup semantics ambiguous.

## Extended module resolver

Wren source syntax remains familiar, but WrenJIT owns a real filesystem/package resolver and compiles the reachable module graph before bytecode generation.

```wren
import "./local"
import "../shared.wren" for Thing
import "@lib/http/client" for Client
import "@std/io" for File
```

Resolution supports:

- importer-relative paths;
- optional `.wren` suffix;
- `<path>/index.wren` fallback;
- `@lib/...` package roots;
- `@std/...` standard-library roots;
- canonical physical-path deduplication;
- stable logical module identities;
- cyclic module initialization;
- project-root discovery;
- environment-configurable package/std roots;
- traversal/symlink escape rejection for package namespaces.

Because imports are statically named string literals, WrenJIT discovers and lowers the complete reachable graph before execution. Filesystem/package lookup therefore does not sit in the interpreter/JIT hot path.

---

# Performance architecture

WrenJIT's performance does not come from a single optimization. The engine is designed as a pipeline where each layer exposes information to the next one while keeping the baseline interpreter fast enough to be useful before native code is generated.

## 1. Fast frontend and whole-program preparation

- The lexer is implemented with **Logos**.
- Source locations are byte-accurate through the frontend and lowerer.
- Parser recovery reports multiple syntax errors in one pass instead of stopping at the first typo.
- **Ariadne** diagnostics are emitted only on cold error paths.
- The static module graph is resolved before VM execution, eliminating filesystem/package resolution from runtime import hot paths.
- Modules are canonicalized so multiple import spellings of one file share one runtime module object.

These are mostly compile-time engineering choices, but they keep runtime metadata compact and predictable.

## 2. Wren IR: SSA for a dynamic language

The high-level/mid-level IR (WIR) is SSA-based and keeps Wren semantics explicit long enough for safe specialization.

Key properties:

- separate `ValueId` SSA for data and `EffectId` SSA for memory/control effects;
- block parameters instead of explicit Phi instructions;
- explicit effect tokens on CFG edges;
- generic dynamic `Invoke` until later devirtualization/speculation;
- fixed instance-field slots;
- explicit cells for captured mutable variables;
- immutable captures passed directly by value;
- internal `Int32`, `F64`, and raw-pointer representations that are not exposed as Wren source types;
- explicit guards, OSR/resume metadata, and deoptimization snapshots;
- structured exception regions represented as metadata, not normal-path control opcodes.

### Baseline HIR optimization pipeline

The cold-start-oriented optimizer performs bounded passes rather than running expensive analyses to a huge fixed point:

- constant folding;
- local value numbering / common-subexpression elimination;
- local algebraic and peephole simplification;
- copy propagation;
- dead-code elimination;
- dead/redundant memory forwarding where legal;
- CFG cleanup and branch simplification;
- guard cleanup;
- partial escape analysis and scalar replacement.

### Partial escape analysis / SROA

WrenJIT implements flow-sensitive partial escape analysis over WIR.

Supported virtualizable allocations include instances, cells, lists, ranges, closures, and maps where their state can be represented field-by-field.

The analysis can:

- remove allocations that never escape;
- sink allocations to only the control-flow paths that actually escape;
- replace field/cell/list accesses with SSA values;
- forward stores to later loads across blocks and loops;
- reconnect scalarized fields across merges using new block parameters;
- materialize virtual objects immediately before an escaping use;
- preserve objects that must remain materialized for deoptimization/exception correctness.

This is not merely a local "stack allocation" pass: non-escaping objects can disappear completely.

### Aggressive HIR mode

A throughput-oriented pipeline is available for hot/AOT-style use and benchmarks. It additionally pays for global analyses such as dominator-based CSE and repeats cleanup/SROA to a bounded fixed point.

Independent WIR functions can also be optimized concurrently, with one optimizer instance per worker.

## 3. Register bytecode instead of a stack machine

WrenJIT lowers WIR to a compact register bytecode designed around interpreter throughput.

### Architectural register model

- **256 frame-local GPRs** (`R0..R255`);
- **16 argument-transfer registers**;
- **2 return-transfer registers**;
- a small special-register bank for `this`, closure, module, Fiber, error/resume values and scratch state.

General registers are raw 64-bit slots and can contain either NaN-boxed Wren values or internal unboxed representations.

### Encoding

- logical bytecode words are 32 bits;
- the first word uses compact `[opcode, A, B, C]` encoding;
- hot instructions remain one word;
- wide IDs/immediates use extension words only when required;
- call/closure register lists are packed four GPR IDs per word;
- relative branches are measured from the end of the branch instruction.

### Bytecode compiler optimizations

The bytecode compiler performs work specifically to reduce interpreter overhead:

- dead pure SSA nodes are removed before allocation;
- global hole-aware register allocation reuses all 256 GPRs aggressively;
- move, guard and block-parameter affinities bias allocation toward coalescing;
- block parameters become edge parallel copies, never runtime Phi instructions;
- parallel-copy cycles use a special scratch register instead of reserving a GPR globally;
- fallthrough layout removes unnecessary branches;
- critical-edge copies can be emitted in shared out-of-line trampolines;
- constants, guard descriptors and IC descriptors are deduplicated;
- frame-entry parameter/capture moves are encoded in `FrameLayout`, eliminating a bytecode prologue;
- precise physical-liveness GC maps are emitted at every GC-capable PC;
- bytecode carries resume/deopt/source/exception metadata in side tables rather than burdening common opcodes.

The current architecture intentionally has no spill opcodes: functions whose live values truly exceed all 256 GPRs fail compilation rather than imposing spill overhead on every ordinary function.

## 4. Prepared token-threaded interpreter

Bytecode is decoded once when an `Interpreter` is created. Each prepared instruction stores a direct Rust handler pointer and compact decoded operands.

The hot loop therefore avoids:

- a giant opcode `match`/switch on every dispatch;
- variable-width bytecode decoding on every iteration;
- repeated FuncId/ClassId/ModuleId resolution;
- repeated switch-table preparation;
- repeated frame-layout decoding.

### Hot interpreter state

Current frame, registers, function and PC are cached in hot raw pointers/indices and refreshed at frame/Fiber/runtime boundaries. Normal GPR access is effectively a pointer add/load/store.

Frames use a contiguous register stack rather than allocating a fresh register vector for every call. Per-frame argument/return/special banks are retained at the Fiber high-water mark and reused.

### Superinstructions

During preparation, common safe instruction pairs are fused when the second instruction is not a control-flow/deopt/resume target. Existing combinations include:

- compare + conditional branch;
- `Move` + `Move`;
- `Move` + `Return`;
- `Move` + `Jump`;
- constant/load + `Return` variants;
- field/static/module/cell load + `Return` variants;
- redundant explicit write-barrier removal after mutation handlers that already perform the StickyImmix barrier.

This reduces dispatch count without changing the persistent bytecode format.

### Adaptive switches

Prepared switch tables choose among:

- linear search for very small case counts;
- dense jump tables for compact ranges;
- sorted/binary lookup for sparse larger switches.

### NaN-boxed values

Source-visible values use a 64-bit NaN-box representation. Numbers remain ordinary IEEE-754 doubles; special tags represent null, booleans and heap handles. NaNs are canonicalized so they cannot collide with tag space.

Internal numeric bytecodes may use raw unboxed `i32`/`f64` register representations and box only at semantic boundaries.

## 5. Quickening and inline caches

The baseline interpreter is adaptive even before the tracing JIT takes over.

### Numeric quickening

Repeated generic numeric invokes can be rewritten to direct numeric handlers for:

- arithmetic;
- comparisons;
- bitwise operations;
- shifts;
- unary numeric operations.

This removes repeated generic Wren method lookup on stable numeric sites.

### Method dispatch caches

Generic method invokes use compact polymorphic inline caches (PICs):

- up to four receiver-class/target pairs are kept inline;
- probes are manually unrolled;
- sites fall back to a megamorphic path instead of endlessly patching/depatching;
- method-table epochs invalidate speculative assumptions when required.

Dense `MethodSymbol`-indexed class method tables avoid hashing on cache misses.

### Map shapes and Map PICs

String-key Maps can use a shape/slot representation and their own four-entry PIC. A hit can resolve shape + slot without rehashing the key. Maps can fall back to the general dictionary representation when shape assumptions no longer fit.

This makes repeated object-like Map access substantially cheaper while preserving Wren Map semantics.

## 6. Precise moving StickyImmix GC

The current runtime integrates a **StickyImmix** collector with precise roots and generational barriers.

Important design choices:

- per-PC reference maps identify exactly which VM registers contain managed references;
- VM roots, Fiber state, globals, async state and event-loop payloads participate in root discovery;
- a stable handle table lets a compact NaN-boxed `Value` survive object movement without rewriting every Wren value;
- moving object addresses are reloaded after safepoints;
- object descriptors precisely trace Lists, Maps, Cells, closures, instances, Fibers/async state and other managed object layouts;
- object-remembering write barriers support generational collection;
- native/JIT stores can elide the barrier when the stored value is statically known not to be a heap reference;
- external Rust/native buffers are accounted separately so large backing stores can contribute to GC pressure;
- root providers expose VM/JIT stack-map roots to the collector;
- GC workers can process tracing/copy work in parallel according to collector configuration;
- `System.gc()` requests an explicit full collection.

Current collector tuning is configurable through environment variables such as heap size, worker count and external-memory threshold in the GC integration.

## 7. Tracing JIT with Cranelift

Hot execution can leave the baseline interpreter and enter native Cranelift code.

The JIT is deliberately split into four layers:

1. the interpreter exposes a small `TraceRuntime` interface;
2. the recorder observes original, **unfused** bytecode and builds Trace SSA;
3. Trace IR optimizations specialize the recorded path;
4. Cranelift lowers optimized traces to native code.

### Native loop state stays native

A compiled trace receives one compact activation structure containing pointers to architectural register banks and side-exit storage.

Loop values such as induction variables and accumulators can remain in native registers for the entire native loop. Wren architectural registers do **not** need to be written back on every iteration.

### Sparse side exits

Each guard/exit knows exactly which architectural values must be reconstructed. Cold side-exit blocks materialize only that sparse snapshot before returning to Rust/baseline execution.

### Root, entry and side traces

The tier supports hot loop traces and hot acyclic entry traces. Repeated side exits may grow child side traces. Eligible edges can be **stitched native-to-native**, avoiding a manager/interpreter round-trip after the child is installed.

Side-trace depth is bounded to prevent uncontrolled trace-tree growth.

### Bounded Wren-call inlining

The recorder can inline bounded Wren calls when it can prove the call target and safely reproduce the call/return path. Method-table epoch guards protect inlined dynamic dispatch assumptions.

### Trace IR optimizations

The current Trace optimizer includes a broad set of loop/speculation passes, including:

- canonicalization and copy cleanup;
- box/guard cancellation;
- loop representation promotion (`f64`, signed `i32`, unsigned `u32` loop phis);
- sparse conditional constant propagation (SCCP);
- induction-variable simplification;
- class/type propagation;
- dominator-tree GVN/CSE;
- loop-invariant code motion (LICM);
- partial redundancy elimination (PRE);
- strength reduction;
- range analysis;
- load/store PRE;
- loop-invariant guard hoisting;
- loop unswitching;
- dead box/copy cleanup.

The optimizer is careful not to move assumptions across full runtime synchronization boundaries where heap/register state may have changed.

### Direct native heap fast paths

Selected hot heap operations lower directly to Cranelift instead of invoking the Rust opcode handler, including fast paths for operations such as:

- instance-field load/store;
- captured-cell load/store;
- List length/get/set;
- Map length;
- String length;
- read/write barrier interactions where legal.

Generated code uses a small JIT-visible heap sidecar rather than depending on the private Rust `HeapObject` enum layout.

### Runtime leaf calls

Some uncommon non-GC operations can use synchronized **leaf** runtime helpers. These commit the required architectural state but preserve unrelated SSA/native values. A small per-trace leaf budget prevents mixed traces from turning into native-to-Rust trampoline loops.

Full GC/frame-changing runtime calls remain hard synchronization epochs.

### Cranelift configuration and verification

The native tier supports Cranelift speed optimization, optional verifier checks, IR/VCode diagnostics, trace verification, configurable hotness thresholds, watchdog exits and detailed JIT statistics.

## 8. Async scheduler and event loop

Async functions reuse the VM's explicit frame/register state rather than native OS stacks.

Current scheduling is cooperative and keeps:

- a FIFO queue of ready async executions;
- suspended awaiters attached to Promises;
- a dedicated microtask queue;
- a deadline heap for timers;
- Promise forwarding/adoption edges;
- cancellation producers for events/tasks/promises.

When the VM has no runnable work, it can park until the next timer deadline instead of polling at fixed intervals.

Promise settlement and cancellation enqueue continuation work as microtasks instead of recursively resuming Fibers from inside settlement code.

This subsystem is intentionally the foundation that the later Napalm runtime can evolve into a larger N:M scheduler without changing Promise/`await` semantics again.

## 9. Structured errors without normal-path tax

WrenJIT's `try`/`catch` implementation uses protected ranges and handler metadata. There are no normal-path enter/leave-try dispatches.

- catch blocks are cold handler entries;
- exact error state is materialized at the catch safepoint;
- call frames retain enough call-site information to unwind across Wren calls;
- hot code inside a protected region remains traceable/JIT-able;
- an actual exceptional edge returns to the baseline interpreter when required;
- escape/SROA conservatively preserves objects whose exceptional state must remain observable.

`defer`, `errdefer`, `finally`, `return`, `break`, and `continue` are lowered so required cleanup runs exactly once in lexical LIFO order.

## 10. Verification and differential testing

The engine contains verifier/test infrastructure for:

- WIR structural invariants;
- bytecode instruction boundaries and register ranges;
- precise GC maps;
- deoptimization/resume metadata;
- exception handler ranges;
- trace SSA and side-exit state;
- interpreter/JIT differential execution;
- original Wren checksum compatibility for core benchmark kernels.

For the existing `jit.wren`, the test suite already checks known Wren-compatible results for `integratePi`, `xorshift` and Mandelbrot with the JIT both disabled and enabled.

---

# Tests included with this release-document package

## Portable cross-engine performance tests

`benchmarks/portable/` contains:

- `bench_fields.wren`
- `bench_dispatch_pic.wren`
- `bench_closures.wren`
- `bench_recursion.wren`
- `bench_list.wren`
- `bench_map.wren`
- `bench_strings.wren`
- `bench_fibers.wren`
- `bench_inheritance.wren`
- `bench_mixed_runtime.wren`

Every measured workload prints:

```text
BENCH|benchmark_name|seconds|checksum
```

The checksum is part of the benchmark contract. Never publish a timing from an engine that produces a different checksum.

## WrenJIT-only feature tests

`tests/wrenjit/` contains focused regression scripts for language/runtime features that official Wren does not parse or expose:

- `test_async_promises.wren` — async methods/closures, top-level await and Promise adoption;
- `test_timer_event_loop.wren` — microtask ordering, delay callbacks and non-overlapping/cancellable intervals;
- `test_promise_cancel.wren` — terminal/idempotent cancellation and cancelled-await behavior;
- `test_defer_exceptions.wren` — `defer`, `errdefer`, `throw`, `catch`, `finally`, return cleanup;
- `modules/main.wren` — extensionless relative resolution, aliases and `index.wren` resolution.

The project's existing GC stress, algorithm/JIT, and hot-function tests should remain part of the final release suite instead of being duplicated here.

---

# Features deliberately not claimed by this final WrenJIT release

The following belong to the next language/runtime phase and should **not** be advertised as WrenJIT features until they actually land and pass the final audit:

- true parallel Wren/Napalm execution through isolates/tasks;
- the planned web-service DSL;
- the TypeScript-like Napalm frontend;
- arbitrary shared mutable state between parallel isolates;
- any compatibility claim not verified by the final source-by-source documentation audit.

---

# Project lineage

```text
official Wren
     |
     v
  WrenJIT
  - new compiler/runtime architecture
  - optimizing register interpreter
  - StickyImmix GC
  - Cranelift tracing JIT
  - language/runtime extensions
     |
     | final feature freeze + release
     v
Napalm (Reloaded)
  - new syntax
  - no Wren compatibility constraint
  - future parallelism / web DSL / further language work
```

The original implementation of Napalm is retained separately as `napalm_old` for historical/reference purposes.

---

# Upstream Wren

WrenJIT exists because Wren provided an unusually small, elegant and understandable language/runtime to experiment on. Official Wren documentation is available at <https://wren.io/>.

In particular, upstream Wren's concurrency model is based on cooperative Fibers, and upstream runtime error handling deliberately uses Fiber abort/`try()` instead of language-level exceptions. WrenJIT preserves Fibers while adding a second structured async/error layer on top.

Before publishing WrenJIT, keep the upstream copyright/license notices required by the Wren code or assets actually carried by the repository.

