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
