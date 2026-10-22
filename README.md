# WrenJIT

Source compilation uses `wren_pipeline::compile_program(&mut hir)` after AST
lowering: HIR -> optimization passes -> register bytecode. The CLI and source
execution tests share this entry point. HIR is optimized in place, and the
existing IR and bytecode verification remains enabled by default.

Add default passes in `wren_ir_optimizing::optimization_passes()`. For a custom
configuration, reuse the optimizer builder:

```rust,ignore
let passes = wren_ir_optimizing::optimization_passes()
    .with_function_pass(MyFunctionPass)
    .with_pass(my_node_pass);
let bytecode = wren_pipeline::CompilationPipeline::new(passes)
    .compile(&mut hir)?;
```

Pass `wren_ir_optimizer::Optimize::new()` to select an empty HIR optimization
stage. Bytecode lowering has its own options, configurable through
`with_bytecode_options(...)`; `compile_with_stats(...)` reports backend statistics
after HIR optimization. `wren_bytecodecompiler` remains the direct backend API
for callers that already prepared their HIR.

### Random numbers and collection getters

`Random` is available as a core global (no import required):

```wren
var random = Random.new()
System.print(random.int(3, 10)) // Integer in [3, 10).
System.print(random.int(4))     // Integer in [0, 4).
```

Each instance has independent SplitMix64 state, seeded from the standard
library's randomized hash seed. This is a non-cryptographic generator.
Integer sampling uses rejection to avoid modulo bias. Bounds must be finite,
exact integers between -9007199254740991 and 9007199254740991, and the upper
bound must be greater than the lower bound. Random operations execute in the
interpreter; they are not native JIT intrinsics.

Getters use no parentheses: `"string".count`, `(1..10).count`, `(1..10).min`,
`1.23.sin`, and `[1, 2, 3].isEmpty`. Range `count` counts unit-step iteration
in either direction, respecting inclusive/exclusive endpoints. `isEmpty` is
also available on strings, maps, and map-key views.

### List and string concatenation

`list.remove(value)` removes and returns the first matching element, or returns
`null` without modifying the list if no match exists. Matching dispatches the
element's `==` operator, including user-defined equality; strings compare by
contents. `removeAt(index)` remains the index-based alternative.

`List + List` returns a new list containing the left elements followed by the
right elements. Neither operand is modified; nested objects remain shared
(a shallow copy). Empty operands and concatenation of a list with itself are
supported. `String + String` concatenates strings, including Unicode text.
Both operators require the right operand to have the same type as the receiver.
These heap-allocating operations currently execute in the interpreter, not as
direct native JIT instructions.

```wren
var combined = ["a", "b", "c"] + ["d", "e", "f"]
System.print(combined) // [a, b, c, d, e, f]
System.print("hello" + " world") // hello world
```

### Map removal and clearing

Iterating a map yields entry objects with read-only `key` and `value` getters.
`map.keys` returns a live iterable view of its keys, with a `count` getter.
Iteration order is unspecified; do not structurally modify the map during
iteration. Entries retain the key/value pair captured when they were produced,
and may be retained after the map is cleared.

```wren
for (entry in birds) {
  System.print("The state bird of %(entry.key) is %(entry.value)")
}
for (state in birds.keys) {
  System.print("The state bird of %(state) is " + birds[state])
}
```

`map.remove(key)` removes the key and returns its stored value, or `null` if the
key was absent. A key whose stored value is `null` is still removed.
`map.clear()` removes all entries in place and returns `null`. All aliases see
the emptied map, which can immediately be reused.

```wren
var capitals = {"Georgia": "Atlanta"}
System.print(capitals.remove("Georgia")) // Atlanta
capitals.clear()
System.print(capitals.count) // 0
```

### Nested list and map subscripts

Comma-separated subscripts traverse nested lists and maps in order. List
indices start at zero:

```wren
var matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
System.print(matrix[0, 1]) // 2, like matrix[0][1]
matrix[1, 2] = 60
System.print(matrix[-1, -1]) // 9
System.print(matrix[1, 0..2]) // [4, 5, 60]
```

Each intermediate value must be a List or Map. List indices must be finite
integers within that list's bounds; negative indices count from its end,
including for ragged lists. Invalid indices such as `matrix[3, 5]` raise a
runtime error. Maps use their normal key semantics at every step:

```wren
var map_list = [{}, {}]
map_list[1, "source"] = [10, 20, 30]
System.print(map_list[1, "source", 0]) // 10

var data = {"rows": map_list}
data["rows", 1, "source", -1] = 99
System.print(data["rows", 1, "source"]) // [10, 20, 99]
```

A missing final map key returns `null` when read and is inserted on assignment.
Missing or non-container intermediate values raise a runtime error; nested
containers are not created automatically. Use `var` only to declare the root
variable, then assign its elements without `var`.

Additional indices support deeper nesting (`cube[x, y, z]`). Reads accept up to
16 indices, and assignments up to 15, leaving one argument for the assigned
value. Assignments update the innermost container in place and return that value.
A read's final index on a List can also be a range to slice it. Classes
can still define their own multi-argument subscript getters and setters.

### Ranges and list slices

Numeric `from..to` ranges include the endpoint; `from...to` ranges exclude it.
Both ascending and descending ranges support `for` iteration and the `from`,
`to`, `min`, `max`, and `isInclusive` getters.

Use a range as a list subscript to create a new, shallow-copy list:

```wren
var trees = ["cedar", "birch", "oak", "willow"]
trees[1..2]    // [birch, oak]
trees[1..-1]   // [birch, oak, willow]
trees[0...2]   // [cedar, birch]
trees[3..1]    // [willow, oak, birch]
trees[0...0]   // []
```

Negative bounds count from the end. Slice bounds must be finite integers and
within bounds; an exclusive endpoint may be just beyond the selected elements.
`list[0..-1]` and `list[0...list.count]` copy even an empty list. Slices do not
modify the original list, but nested objects remain shared. Subscript assignment
accepts integer indices only, not ranges.

### HIR escape analysis and scalar replacement

`wren_ir_escape.rs` implements partial escape analysis (Graal-style) plus full
SROA over HIR. `ScalarReplacement` runs at the end of the default
`InterpreterOptimizationBundle`; if it changes anything, simplify/copy/DCE run
once more. `EscapeAnalysis::analyze(&function)` exposes the per-allocation
classification (`NoEscape`, `PartialEscape`, `GlobalEscape`) without rewriting.

- Candidates: `AllocInstance`, `AllocCell`, `AllocList` (constant capacity),
  `MakeRange` (numeric bounds), `MakeClosure` and `AllocMap`.
- Field, cell and list accesses (`LoadField`/`StoreField`, `LoadCell`/`StoreCell`,
  `ListLen`, constant-index `ListGet`/`ListSet`, `ListAppend`) become SSA values
  (block parameters built with Braun-style lazy SSA construction). Write
  barriers on virtual objects are dropped, and `RefEq` folds to a constant.
- `NoEscape` allocations disappear entirely. On `PartialEscape` paths the object
  is materialized (allocation plus stores of its current virtual state) only
  right before the first escaping use, or at a merge with an already
  materialized path. Reads before that point are still forwarded from SSA.
- Nested virtual objects are resolved over several rounds (up to 4).
- Allocations referenced by deopt snapshots or inlined frames are pinned.
  Out-of-range virtual list accesses and other error-preserving cases also
  force materialization, so runtime errors are not lost.

There is no HIR inliner yet. Any `Invoke` with the object as receiver or
argument (including `Fn.new { ... }` and list/range methods) is a real escape,
so today's main wins are forwarded field reads in constructors, dead
allocations, and branches where the object escapes on only one path.
`ScalarReplacement::stats` reports the removed, sunk and forwarded allocations.

Trace compilation has the same split: `wren_trace_ir_optimizer.rs` owns the
extensible, verified pass runner, while `wren_trace_ir_optimizing.rs` owns the
default list and transforms. The recorder runs this list for root and side
traces before Cranelift lowering when `optimize_trace_ir` is enabled.

The default sequence preserves numeric representation specialization, then runs
GVN, LICM, PRE and another GVN cleanup. GVN reuses pure scalar expressions;
LICM moves safe invariant computations into the native preheader. PRE uses the
loop's entry/backedge join: `e(phi(initial, next))` becomes
`phi(e(initial), e(next))` when `e(next)` is already computed in the body. It is
loop-edge PRE for this linear Trace IR, not arbitrary-CFG PRE. Loads, runtime
effects, checked operations and trapping integer division are excluded from
code motion; runtime calls also reset GVN availability.

```rust,ignore
use wren_trace_ir_optimizer::Optimize;
use wren_trace_ir_optimizing::{GlobalValueNumbering, LoopInvariantCodeMotion};

Optimize::new()
    .with_pass(GlobalValueNumbering)
    .with_pass(LoopInvariantCodeMotion)
    .optimize_trace(&mut trace)?;
```

Add default passes in `wren_trace_ir_optimizing::optimization_passes()`, or append
custom implementations of `TraceOptimization` with `.with_pass(...)`.
`TraceStats` exposes `gvn_eliminated`, `licm_hoisted` and `pre_eliminated`.
The old `wren_trace_ir::TraceOptimizer` API remains available for compatibility.

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

The tracing tier specializes numeric operations on local registers, including
arithmetic, comparisons, unsigned bitwise methods, and floating remainder
(`%` and the typed `FRem` bytecode).

Numeric getters `abs`, `floor`, `ceil`, and `sqrt` lower to native Cranelift
operations. `round`, `sin`, `cos`, `tan`, `log`, and `exp` use scalar-only native
helpers with the same Rust `f64` semantics as the interpreter. Remainder also
uses a scalar helper, preserving signed zero, NaN, infinity, and division-by-zero
behavior without approximating it as `a - trunc(a / b) * b`. These helpers
neither reenter the interpreter nor allocate or access the GC heap.
`isNan`, `isInfinity`, and `isInteger` lower to native comparisons and selection;
infinities and NaN are not integers. Num specialization requires the host to
confirm the standard builtin implementation and guards receivers as numbers.
All these operations work with the default `no_runtime_calls_in_native_trace`
restriction; pure scalar math helpers are not VM runtime calls.

Heap operations such as range creation and list slicing still require the
interpreter. The optional synchronized runtime bridge can execute other
bytecodes from a trace when `no_runtime_calls_in_native_trace` is false, but it
is not direct native lowering and remains disabled by default.

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
