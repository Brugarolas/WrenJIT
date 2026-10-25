# WrenJIT native FFI

WrenJIT includes a small typed C FFI built on `libloading` + `libffi`.
It is separate from Wren's existing embedding/`foreign` mechanism:

- **`foreign` methods / `Host::call_foreign`** let the application embedding WrenJIT provide native methods.
- **`FFI`** lets WrenJIT code open a shared library and call an exported C ABI symbol directly.

## Basic API

```wren
var libc = FFI.open("libc.so.6")
var strlen = libc.function("strlen", "usize", ["cstring"])

System.print(strlen.call("hello")) // 5
```

`library.function(symbol, resultType, argumentTypes)` resolves the symbol once and returns a callable native-function object. Prepared libffi CIFs are cached by signature **per OS thread**, so functions sharing a signature reuse the same call-interface description without moving libffi's non-`Send` descriptors between scheduler workers.

A native function exposes `call(...)` overloads from 0 through 16 arguments, matching WrenJIT's existing maximum method arity.

## Core objects

### `FFI`

```wren
FFI.open(path)       // -> native library
FFI.null             // -> null (explicit spelling for pointer arguments)
FFI.lastError        // -> errno/GetLastError snapshot from the last FFI call
```

`FFI.lastError` is captured immediately after `ffi_call`, before Wren allocation or formatting can overwrite the OS error state. As with C itself, inspect it only when the called function's contract says an error occurred; successful C functions are not required to clear an old errno/last-error value.

### Native library

```wren
library.name
library.function(symbol, resultType, argumentTypes)
```

Libraries remain mapped until the interpreter is destroyed. WrenJIT deliberately does not expose `close()` in v1: bound function pointers must never dangle, and MMTk worker threads must not accidentally execute library unload/finalizer code.

### Native function

```wren
function.name
function.call(...)
```

### Native pointer

A non-null `pointer` result becomes an opaque FFI pointer object. A native null pointer becomes ordinary Wren `null`.

```wren
pointer.isNull          // normally false; native NULL is normalized to Wren null
pointer.offset(bytes)   // address arithmetic only; does not dereference memory
pointer.readCString()   // copies at most 16 MiB, stopping at NUL
pointer.toString        // hexadecimal address
```

Pointers can be passed back to arguments declared as `pointer`. WrenJIT never exposes a raw address as `Num`, because a 64-bit pointer cannot in general be represented exactly by IEEE-754 `f64`.

## Supported types

Canonical spellings are shown first; several C aliases are accepted.

| WrenJIT FFI type | C / ABI meaning | Wren value |
| --- | --- | --- |
| `void` | no result | `null` |
| `bool` | 1-byte C boolean | `Bool` |
| `i8`, `u8` | fixed 8-bit integer | `Num` |
| `i16`, `u16` | fixed 16-bit integer | `Num` |
| `i32`, `u32` | fixed 32-bit integer | `Num` |
| `i64`, `u64` | fixed 64-bit integer | `Num`, only when exact |
| `isize`, `usize` | pointer-sized integer | `Num`, only when exact |
| `char` | target C `char` | `Num` |
| `unsigned char` | C `unsigned char` | `Num` |
| `short`, `unsigned short` | C short | `Num` |
| `int`, `unsigned int` | C int | `Num` |
| `long`, `unsigned long` | C long | `Num`, only when exact |
| `long long`, `unsigned long long` | C long long | `Num`, only when exact |
| `f32` / `float` | C float | `Num` |
| `f64` / `double` | C double | `Num` |
| `pointer` | `void*` | opaque pointer or `null` |
| `cstring` | NUL-terminated `char*` | copied Wren String or `null` |

Recognized aliases also include common names such as `int32_t`, `uint64_t`, `size_t`, `ssize_t`, `intptr_t`, `uintptr_t`, `c_int`, `c_long`, `ptr`, `void*`, `char*`, and `const char*`.

`void` is valid only as a return type.

## Integer precision

Wren numbers are IEEE-754 doubles. WrenJIT therefore refuses to silently corrupt wide integer values:

- integer arguments must be finite, integral, in range, and exactly representable by the source Wren `Num`;
- integer results are converted to `Num` only if conversion is exact;
- otherwise the call reports a runtime error.

This still permits large exactly representable values (for example powers of two above `2^53`), while rejecting arbitrary 64-bit values whose low bits would be lost. Napalm can later grow first-class `i64/u64` source values without changing the native signature model.

## C strings

For a `cstring` argument, WrenJIT creates an owned `CString` before entering native code. Therefore:

- moving/collecting the Wren String cannot invalidate the pointer during the call;
- embedded NUL bytes are rejected;
- the pointer is valid only for the duration of that call unless the C API explicitly copies it.

If a C function returns a pointer *into* one of its `cstring` arguments (for example a `strchr`-style API), declare the result as `cstring` when you want an immediate copy. Declaring such an interior result as `pointer` and retaining it after the call would outlive WrenJIT's temporary CString and is therefore invalid.

For a `cstring` result, WrenJIT immediately copies the bytes into a Wren String. The native pointer is not retained.

If the native API returns an allocated string that the caller owns, declare it as `pointer`, call `readCString()`, and then pass the pointer to the API's deallocator.

## Example: caller-owned native memory

```wren
var libc = FFI.open("libc.so.6")
var malloc = libc.function("malloc", "pointer", ["usize"])
var free = libc.function("free", "void", ["pointer"])

var p = malloc.call(4096)
if (p == null) Fiber.abort("malloc failed")

System.print(p.toString)
free.call(p)
```

## Safety boundary

The FFI validates Wren values against the **declared** signature. It cannot prove that the declaration matches the actual C symbol. A wrong native signature can still corrupt memory or crash the process; that is inherent to a low-level C FFI.

Likewise, opaque pointer lifetime is owned by the native API. `offset()` performs integer address arithmetic only, but passing an invalid/stale pointer to C or calling `readCString()` on one is unsafe.

## Deliberately not in v1

- callbacks from C into Wren;
- variadic functions;
- structs/unions passed by value;
- direct Wren heap pointers;
- automatic header parsing;
- C++ name mangling / methods;
- asynchronous/blocking-native-call offload;
- manual library unload.

Callbacks deserve a separate design because a callback may arrive on an arbitrary OS thread and therefore must coordinate with WrenJIT's GC safepoints, isolate ownership and future N:M scheduler. Struct support should similarly be added with explicit ABI layout objects rather than guessed layouts.

## Dependencies

```toml
libffi = "5.2.0"
libloading = "0.9.0"
```

`libffi` uses its bundled libffi build by default. If a deployment deliberately uses a system libffi, enable the crate's `system` feature instead.
