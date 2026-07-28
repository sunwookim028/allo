<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Integrating HLS IPs that use `hls::stream` interfaces

This document explains a change that lets `allo.IPModule` integrate a
hand-written HLS C++ block ("IP") whose **interface uses `hls::stream<T>`
ports**, not just arrays and scalars. It is written for someone new to the Allo
compiler and to compilers in general, so it starts with background and builds up
to the specific edits.

If you only want the "what changed" list, jump to [The five edits](#the-five-edits).

---

## 1. Background you need first

### 1.1 What an `IPModule` is

An IP is a `.cpp` file you wrote by hand (or got from a vendor) containing one
HLS function, e.g.:

```cpp
void vadd(int A[32], int B[32], int C[32]) { ... }
```

`allo.IPModule(top="vadd", impl="vadd.cpp")` lets an Allo design *call* that
function. Allo does **not** read the function body. It only:

1. **parses the signature** to learn the argument types,
2. **emits a call** to it in the generated code, and
3. **stitches the IP source back in** with `#include` at the end.

So the IP is a black box wired in by name — the same idea as an `extern`
declaration in C, or a foreign-function binding (`ctypes`).

### 1.2 A 60-second MLIR primer

Allo compiles your Python down through **MLIR**, an intermediate representation.
A few facts are enough to read the rest of this doc:

- Everything is an **operation** named `dialect.opname`. `func.func` is the
  "define a function" op; `func.call` calls one; `allo.stream_construct` creates
  a FIFO. `func`, `allo`, `memref`, `affine` are *dialects* (namespaced groups of
  ops).
- `%x` is a **value** (data flowing through the function). `@name` is a
  **symbol** (a global name, e.g. a function). So `call @vadd(%0)` means "call
  the function named `vadd`, passing value `%0`".
- **Types**: `memref<32xi32>` is a buffer of 32 ints (Allo's array type);
  `!allo.stream<i32, 4>` is a FIFO of `i32` with depth 4 (the leading `!` means
  "a type defined by a dialect").
- A **`func.func private @vadd(...)` with no body** is a *declaration* — "this
  function exists somewhere, here is how to call it". This is exactly how the IP
  appears inside MLIR.

### 1.3 The pipeline a call goes through

For a dataflow (`@df.region`) design, an IP call passes through these stages, in
order:

```
Python  ──parse──►  IPModule.args     (backend/ip.py)
        ──build──►  func.func private + func.call in MLIR   (ir/builder.py)
        ──hoist──►  streams moved onto kernel interfaces     (dataflow.py)
        ──emit───►  HLS C++ text                             (mlir/.../EmitVivadoHLS.cpp)
        ──splice─►  #include the IP .cpp into kernel.cpp      (backend/hls.py)
```

The key discovery behind this change: **the emit and splice stages already
handle streams correctly.** A hand-written
`func.func private @my_ip(memref<8xi32>, !allo.stream<i32, 4>)` plus a `call`
emits exactly the C++ we want with **zero backend changes**. So the whole task
was getting the *frontend* (parse → build → hoist) to produce that IR.

### 1.4 Why streams were the hard part

- **Parsing:** `hls::stream<int>&` has a namespace (`::`), a template (`<>`), and
  a reference (`&`). The old type regex could not match it.
- **Direction is ambiguous:** in C++ you read a stream with `.read()` and write
  it with `.write()`, but **both are declared the same way**: `hls::stream<T>&`.
  So the tool cannot tell, from the signature alone, whether the IP *consumes* or
  *produces* a given stream. The user must say.
- **The hoisting pass rejected the call:** `move_stream_to_interface()` figures
  out each stream's direction by looking at *how it is used*. It understood
  `StreamPut`/`StreamGet` but hit a hard `raise` on anything else — including a
  `func.call` to an IP.

### 1.5 What "hoisting" means (the stage most people haven't seen)

In your Python you declare a stream once at region scope and use it inside
kernels, as if it were shared. But MLIR/HLS functions can only touch what is
*passed into them*. So a pass called `move_stream_to_interface()` rewrites:

```mlir
// before: each kernel has its own local copy of the stream
func.func @producer() { %s = allo.stream_construct {name="fifo"} ; put(%s, ...) }
func.func @consumer() { %s = allo.stream_construct {name="fifo"} ; get(%s) }
```

into:

```mlir
// after: one stream in `top`, passed into each kernel as an argument
func.func @producer(%s: !allo.stream<i32,4>) { put(%s, ...) }
func.func @consumer(%s: !allo.stream<i32,4>) { get(%s) }
func.func @top() { %s = allo.stream_construct {name="fifo"} ; producer(%s); consumer(%s); }
```

"Hoisting" = lifting the declaration out of the kernel body and onto the kernel's
**interface** (argument list), then creating one real stream in `top` and
threading it to everyone. This is what turns three disconnected copies into one
connected FIFO. An IP call has to survive this pass so its stream operand gets
retargeted to the shared stream too.

---

## 2. The design decisions

- **Direction is declared with `input_idx` / `output_idx` on `IPModule`.** These
  list which argument positions the IP *reads* (input) vs *writes* (output). This
  mirrors the existing AIE `ExternalModule` API, so it is not a new concept in the
  codebase. For `vadd_stream(A, B, C)` where the IP reads A, B and writes C:
  `input_idx=[0, 1], output_idx=[2]`.
- **Kernel-scope only.** The IP is called from inside a `@df.kernel` body. (An
  earlier experiment showed calling it directly in the `@df.region` body crashes a
  different pass, `_build_top`; supporting that is a separate, larger change.)
- **HLS targets only** *(as of this change)*. A stream IP works for
  `vitis_hls`/`vivado_hls` (csyn and beyond). It cannot run on the CPU targets
  (`llvm`, `simulator`) or in `csim` mode, because those link the IP as a shared
  library through raw pointers and have no way to represent a FIFO. We reject
  those paths with a clear error instead of failing confusingly later.
  **Since then**, `target="simulator"` *is* supported, through a stream shim
  that makes the IP drive Allo's ring buffers directly — see
  [`IP_STREAM_SIM_SHIM.md`](./IP_STREAM_SIM_SHIM.md). The plain `llvm` target
  and `csim` remain fenced off, because they run the kernels sequentially and a
  blocking stream read could never be satisfied.

---

## 3. The five edits

### Edit 1 — Parser recognizes `hls::stream` — `allo/backend/ip.py`

*(This edit was made first, before the others.)*

`parse_cpp_function()` returns a list of `(type, shape)` tuples describing each
argument. The shape slot encodes the kind: `()` scalar, `None` pointer, a tuple
of dims for an array. A new sentinel **`STREAM`** (an instance of `_StreamShape`)
marks a stream, and for a stream the *type* slot holds the full stream type
string as written, e.g. `('hls::stream< int8_t >', STREAM)`.

Why a sentinel in the shape slot: the 2-tuple shape is unpacked positionally in
several places and is also shared with the AIE `ExternalModule`. Keeping the
2-tuple intact and marking streams with a distinct shape value means existing
code keeps working, and any shape-dispatching code (`len(shape)`) fails loudly on
a stream rather than silently doing the wrong thing.

Supporting pieces: new regexes (`_TYPE_TOKEN`, `_STREAM_TOKEN`) that match a
namespaced template whole; the comma-splitter now tracks `<`/`>` depth (so a
template argument's internal comma does not split a parameter); and inline
`/* ... */` comments are stripped first (the HLS emitter annotates stream
parameters with `/* v0[2] */`, which would otherwise look like array dims).

**Known limitation:** the template regex allows one level of `<>` nesting, so
`hls::stream<int>` and `hls::stream<ap_int<8>>` parse, but a double-nested
element like `hls::stream<vector<ap_int<8>,4>>` does not.

### Edit 2 — `IPModule` accepts `input_idx` / `output_idx` — `allo/backend/ip.py`

`IPModule.__init__` gained optional `input_idx=None, output_idx=None`
parameters, stored on the object. They declare per-argument direction — required
for stream ports since the C++ signature cannot express it. Default `None`
preserves the historic behavior for array/scalar IPs. This matches
`ExternalModule`, and the IR builder already reads `obj.input_idx`.

Also added a small helper `IPModule.has_stream_args` (True if any argument is a
stream) used by the fencing in Edit 4.

### Edit 3 — Builder types the stream call — `allo/ir/builder.py`

In `build_Call`, the `IPModule`/`ExternalModule` branch (~line 3183) now has a
case for `shape is STREAM`. For a stream argument it:

1. **Clones the `stream_construct` op into the calling kernel**, exactly as
   `put`/`get` do. A stream referenced by name resolves to the construct op that
   was created where the stream was declared — possibly in another function.
   Cloning makes a local copy so the call is a use of a construct *inside this
   kernel* (the hoisting pass later deduplicates by the stream's `name`).
2. **Adopts the operand's own stream type** (`!allo.stream<T, depth>`) as the
   declaration's argument type, rather than building a `memref`. The operand
   carries both the element type and the FIFO depth; the C++ signature has
   neither (it never states a depth).
3. **Skips the alloc/copy dance.** Streams pass by reference, so there is nothing
   to copy in or out.
4. **Records direction on the call.** It builds a `stream_dirs` string — one
   character per operand, `i`/`o`/`_` — from `input_idx`/`output_idx`, and
   attaches it as a string attribute on the `func.call`. This is how the
   direction the *user* declared in Python reaches the hoisting pass, which only
   sees MLIR. (If a stream arg is in neither list, the builder raises a clear
   error telling the user to declare it.)

The result, for `vadd_stream(sA, sB, sC)`:

```mlir
func.func private @vadd_stream(!allo.stream<i32, 4>, !allo.stream<i32, 4>, !allo.stream<i32, 4>)
...
call @vadd_stream(%2, %0, %1) {stream_dirs = "iio"} : (...) -> ()
```

### Edit 4 — Fence the CPU / simulator paths — `allo/backend/ip.py`, `allo/passes.py`

*(Partly superseded: the dataflow simulator now runs stream IPs through a shim;
see [`IP_STREAM_SIM_SHIM.md`](./IP_STREAM_SIM_SHIM.md). The fences described
below still stand for the plain `llvm` target and for `csim`.)*

A stream IP cannot run on the CPU. The two wrapper generators
(`generate_nanobind_wrapper`, `generate_mlir_c_wrapper`) and the shared-library
compile call them, so a guard at the top of each raises a clear
`NotImplementedError` if the IP has stream args. `call_ext_libs_in_ptr` (used by
both the LLVM JIT and the OMP dataflow simulator) also checks up front, so the
error surfaces early with a helpful message rather than as a cryptic g++ or JIT
failure. `csim` mode reaches these guards transitively (it re-parses the
generated `kernel.cpp` through the nanobind path).

### Edit 5 — Hoisting pass accepts the IP call — `allo/dataflow.py`

This is the linchpin. In `move_stream_to_interface()`, the loop that classifies a
stream's direction by walking its uses previously understood only
put/get/empty/full and hit `raise ValueError("Stream is not used correctly")` on
anything else. A new branch handles a `func.CallOp`:

```python
elif isinstance(use.owner, func_d.CallOp):
    dirs = use.owner.attributes["stream_dirs"].value   # written by Edit 3
    direction = "in" if dirs[use.operand_number] == "i" else "out"
```

`use.operand_number` is the stream's position among the call's operands, so we
read the matching character from the `stream_dirs` string Edit 3 attached.
Everything after classification is already generic: the pass appends the stream
to the kernel's signature and rewrites every use (including this call's operand)
to the new argument, and `_build_top` threads one shared stream through the whole
graph. No further changes were needed.

**Constraint (inherited, not introduced):** direction is a single value per
stream per kernel. A given kernel either reads or writes a given stream —
separate in/out streams are fine, a bidirectional port on one stream is not
expressible. This is exactly how put/get already behave.

---

## 4. What the finished pipeline produces

For a region with feeder kernels, a wrapper kernel that calls the IP, and a drain
kernel, the emitted HLS C++ is:

```cpp
void ip_wrap_0(
  hls::stream< int32_t >& v8,
  hls::stream< int32_t >& v9,
  hls::stream< int32_t >& v10
) {
  vadd_stream(v10, v8, v9);
}
...
void top(int32_t *v20, int32_t *v21, int32_t *v22) {
  #pragma HLS dataflow
  hls::stream< int32_t > v33;   // one real FIFO per stream, declared in top
  hls::stream< int32_t > v34;
  hls::stream< int32_t > v35;
  feedA_0(...); feedB_0(...); ip_wrap_0(v34, v35, v33); drain_0(...);
}
```

and `vadd_stream.cpp` is copied into the project, `#include`d in `kernel.cpp`,
and `add_files`'d into `run.tcl`.

---

## 5. How to use it

```python
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

vadd_stream = allo.IPModule(
    top="vadd_stream",
    impl="vadd_stream.cpp",
    input_idx=[0, 1],   # the IP reads streams A and B
    output_idx=[2],     # the IP writes stream C
)

@df.region()
def top(A: int32[32], B: int32[32], C: int32[32]):
    sA: Stream[int32, 4]
    sB: Stream[int32, 4]
    sC: Stream[int32, 4]

    @df.kernel(mapping=[1], args=[A])
    def feedA(a: int32[32]):
        for i in range(32): sA.put(a[i])

    @df.kernel(mapping=[1], args=[B])
    def feedB(b: int32[32]):
        for i in range(32): sB.put(b[i])

    @df.kernel(mapping=[1])
    def ip_wrap():
        vadd_stream(sA, sB, sC)      # the IP call, kernel scope

    @df.kernel(mapping=[1], args=[C])
    def drain(c: int32[32]):
        for i in range(32): c[i] = sC.get()

mod = df.build(top, target="vitis_hls", mode="csyn", project="out.prj")
```

See `tests/ip_integration/test_stream_ip.py` and
`tests/ip_integration/vadd_stream.cpp` for the runnable version.

---

## 6. Verification performed

- **Parser:** returns `('hls::stream< int32_t >', STREAM)` for stream args and
  unchanged tuples for memref/pointer/scalar/`ap_int` (regression).
- **IR (pre-hoist):** the private decl carries `!allo.stream` types and the call
  carries `stream_dirs`.
- **IR (post-hoist):** `df.customize` no longer aborts; the IP-calling kernel
  gets the streams on its interface (`stypes` updated) and the graph is wired
  through one shared construct per stream in `top`.
- **Codegen:** emitted C++ has `hls::stream< int32_t >&` on the wrapper signature
  and the `vadd_stream(...)` call; the IP `.cpp` is copied, `#include`d, and
  `add_files`'d.
- **Fencing:** `target="simulator"` (and the LLVM path) raise a clear
  `NotImplementedError`. *(The simulator fence was later lifted — see
  [`IP_STREAM_SIM_SHIM.md`](./IP_STREAM_SIM_SHIM.md); the LLVM one stands.)*
- **Regressions:** existing `tests/ip_integration/test_external.py` (7 passed, 1
  skipped) and `tests/dataflow/test_df_unit.py` pass.

End-to-end csynth requires `vitis_hls` on PATH (the `test_stream_ip.py::
test_stream_ip_csynth` case is skipped automatically when it is not available).

---

## 7. Files changed

| File | Edit |
|---|---|
| `allo/backend/ip.py` | Parser (Edit 1); `input_idx`/`output_idx` + `has_stream_args` (Edit 2); CPU fences (Edit 4) |
| `allo/ir/builder.py` | Type the stream call, attach `stream_dirs` (Edit 3) |
| `allo/dataflow.py` | Direction classifier accepts `func.call` (Edit 5) |
| `allo/passes.py` | Fence `call_ext_libs_in_ptr` (Edit 4) |
| `tests/ip_integration/vadd_stream.cpp` | Example stream IP |
| `tests/ip_integration/test_stream_ip.py` | Tests |
