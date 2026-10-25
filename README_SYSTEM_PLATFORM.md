# WrenJIT System platform helpers

Adds two WrenJIT-specific zero-argument methods to `System`:

```wren
System.os()
System.arch()
```

They return the Rust target constants from `std::env::consts` as Wren strings.

Typical values include:

- OS: `linux`, `windows`, `macos`, `freebsd`, `android`, `ios`, ...
- architecture: `x86_64`, `aarch64`, `x86`, `arm`, `riscv64`, ...

The values describe the platform WrenJIT was compiled for, which is the correct
behavior for cross-compiled binaries.

No parser, IR, bytecode, JIT, GC, or FFI changes are required.
