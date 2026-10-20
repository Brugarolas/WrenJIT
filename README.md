# WrenJIT

Run performance workloads with an optimized build:

```powershell
cargo run --release -- scripts/jit.wren
```

`cargo run -- scripts/jit.wren` builds an unoptimized debug interpreter and
Cranelift compiler. Use that configuration for debugging, not comparisons with
the bundled optimized Wren executable.

To exclude Cargo's build time, build once and run the executables directly:

```powershell
cargo build --release
.\target\release\WrenJIT.exe scripts/jit.wren
.\scripts\wren_cli.exe scripts/jit.wren
```

Run each several times without other builds running. The script's `elapsed`
line measures execution (including trace recording and compilation). Use an
external stopwatch if you also want process startup and source compilation.

Inspect JIT activity with:

```powershell
$env:WREN_JIT_STATS = "1"
.\target\release\WrenJIT.exe scripts/jit.wren
Remove-Item Env:WREN_JIT_STATS
```

`WREN_JIT_EVENTS=1` also logs recording/compilation events. `WREN_NO_JIT=1`
selects this project's interpreter for a comparison without native traces.

The tracing tier currently specializes numeric operations on local registers.
Unsupported operations, including unsigned bitwise methods and module variable
accesses, fall back to the interpreter.

The performance fixes cover three cases:

- Exhausting the native iteration budget yields to the interpreter without
  discarding the trace. Long loops can reenter native code.
- Repeated short exits temporarily cool a trace down. A later invocation can
  still benefit from native execution.
- Recording an inner loop stops if execution takes an enclosing loop's
  backedge. This transient outcome backs off and retries; it does not install
  the enclosing path as an inner trace or permanently blacklist the header.

Bitwise primitives in both interpreter paths now use unsigned 32-bit semantics,
as in [original Wren](https://github.com/wren-lang/wren/blob/main/src/vm/wren_core.c).

Run the JIT regressions, including numerical comparisons with the interpreter
and reference benchmark checksums:

```powershell
cargo test wren_interpreter::
cargo test wren_traceir::
cargo test wren_tracingjit::
cargo test --release wren_interpreter::
```

## StickyImmix memory management

The VM uses the local MMTk 0.33 binding in `src/wren_gc.rs`, with StickyImmix
selected explicitly. NaN-boxed values contain stable handles; their canonical
atomic slots are updated when Immix defragments. Interpreter and native traces
share this representation, so moving GC does not change the JIT value ABI.

Objects are promoted by nursery collections. Mutations of lists, maps,
instances, cells and fiber stacks remember mature owners before storing young
references. The binding claims MMTk's object log bit once per collection epoch;
it does not use the probable-write hook that can enqueue duplicate owners.
Native-resource tracking is also generational: minor collections inspect young
owners only, while full collections reclaim unreachable mature owners.
`System.gc()` requests a full collection. Automatic heap pressure can request a
nursery collection; native-buffer growth requests a full collection because
small GC owners can retain large external buffers.
Explicit binding collections wait for completion, retrying when an already
scheduled nursery collection subsumes a full-heap request.

Compiler-generated reference maps also cover HIR instruction boundaries.
Interpreter backedges poll the GC request flag, including after a native trace
yields, without adding a runtime helper to each native loop iteration. Keep the
native loop/dispatcher budget finite when other mutators need bounded GC
cooperation. Hand-assembled bytecode must provide reference maps or explicit
`Safepoint` instructions at its polling boundaries.

Configuration is process-wide and must be identical for every VM:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WREN_GC_HEAP_MB` | `256` | Fixed managed heap size in MiB; minimum 8. |
| `WREN_GC_WORKERS` | `2` | Positive number of GC workers. |
| `WREN_GC_EXTERNAL_MB` | `64` | External-memory growth between GCs that triggers a full collection; 0 disables this trigger, not accounting. |
| `WREN_GC_STATS` | unset | Print collection counts, pause/rendezvous time, allocation, copying and barrier counters. |

Invalid numeric configuration fails explicitly. Heap size does not cap native
buffers, handle-table storage, JIT code or compiler metadata. Statistics are
process-wide, including the querying mutator's unflushed allocation/barrier
counters; other running mutators flush theirs at collection or detach.

### Embedding and unsafe contracts

- `Value` is not a root. Native callbacks retaining a value across allocations
  must hold `HostContext::root(value)` and use its `value()` afterward. Use
  `Interpreter::add_root` / `set_root` / `release_root` for embedding roots.
- Never retain a payload pointer or Rust reference into a managed object across
  an allocation, collection or safepoint. Reload the current address from its
  root/handle. Constructor scratch roots protect outgoing references while
  allocating their owner.
- A mutator is thread-affine. Threads waiting on I/O or another mutator must
  enter `Mutator::with_native` rather than block while Running. Its unsafe
  callback must not access managed memory, mutate root providers or call GC
  APIs. Idle mutators on the same OS thread park together, including during
  attachment of another VM.
- Custom descriptors must enumerate every strong edge using stable atomic
  slots. Payloads must be byte-relocatable; use stable external storage for
  address-sensitive resources. All mutating threads must participate in the
  safepoint protocol, and writes must use the binding's barriers.
- Weak handles clear only when their referent is actually reclaimed; a nursery
  collection may retain an unreachable mature referent. Deferred finalizers
  run only through `Mutator::run_finalizers`, once, with a rooted argument.
  Resurrection preserves native resources until actual reclamation. Cleanup
  callbacks run during GC and must not panic, allocate, resurrect or inspect
  children.

Run the GC regressions, including old-to-young edges, large objects, finalizer
resurrection, constructor roots, suspended fibers and multi-mutator rendezvous:

```powershell
cargo test sticky_
$env:WREN_GC_HEAP_MB = "8"
$env:WREN_GC_EXTERNAL_MB = "1"
$env:MMTK_IMMIX_ALWAYS_DEFRAG = "true"
$env:MMTK_IMMIX_DEFRAG_EVERY_BLOCK = "true"
cargo test --release sticky_ -- --test-threads=1
Remove-Item Env:MMTK_IMMIX_ALWAYS_DEFRAG, Env:MMTK_IMMIX_DEFRAG_EVERY_BLOCK
```

For an allocation-heavy performance workload with an old-to-young reference
ring (expected checksum `125000250256`):

```powershell
cargo build --release
$env:WREN_GC_STATS = "1"
.\target\release\WrenJIT.exe scripts/gc.wren
Remove-Item Env:WREN_GC_STATS, Env:WREN_GC_HEAP_MB, Env:WREN_GC_EXTERNAL_MB
```

Time repeated runs without other builds or tests running. Defragment-every-block
is a correctness stress setting, not a performance configuration.
