<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Allo DSL Reference

A concise usage reference for writing **Allo** kernels and schedules. Allo is a
Python-embedded, MLIR-based accelerator design language. A kernel describes
*what* to compute; a schedule describes *how* it maps to hardware; a backend
(`cpu` / `vitis`) runs or synthesizes it. This file covers syntax and APIs only —
not compiler internals. Authoritative source: `docs/` (frontend, scheduling,
simulation, typing_rules). Worked examples: `allo/library/transformer/`,
`allo/library/systolic/`, `tests/`.

---

## 1. Imports & namespaces

```python
import allo
from allo.lang import (
    kernel, consteval, KernelOptions, Template, constexpr,
    Stream, Stateful, grid, range,
    bool, i8, i16, i32, i64, u1, u8, u32, f16, f32, f64, bf16, index,
    apint, apfloat, Module,
)
from allo.operators import arith, math, linalg   # explicit operator calls
```

- Types, `kernel`, `consteval`, `KernelOptions`, `Template`, `constexpr`,
  `Stream`, `grid`, `range`, `Stateful`, `Module` come from `allo.lang`.
- Top-level `allo` namespace also provides `allo.grid`, `allo.range`, `allo.max`,
  `allo.min`, and the spatial built-ins `allo.get_wid(axis)` / `allo.get_nw(axis)`.

---

## 2. Kernel definition

A kernel is a plain Python function decorated with `@kernel`. **Every parameter
must be type-annotated.** Returning a value requires an **explicit return
annotation**.

```python
@kernel
def saxpy(a: f32, x: f32[16], y: f32[16], out: f32[16]):   # no return -> no annotation
    for i in range(16):
        out[i] = a * x[i] + y[i]

@kernel
def scalar_add(x: i32, y: i32) -> i32:                      # return needs annotation
    return x + y
```

- No-return kernels omit the annotation or use `-> None`.
- **Multiple returns** use a tuple annotation: `-> (i32, f32)`; unpack at the call
  site `lhs, rhs = split_pair(x, y)`.
- **`return` placement is restricted**: only at the top level of the body or in a
  *first-level* `if`/`else` branch. Returns inside loops or nested `if` are rejected.

### Nested kernels (local helpers)

Declared at the top level of an enclosing kernel body, exactly one `@kernel`
decorator, called like any kernel. The supported way to wire producer/consumer
stages and PE arrays inside one top kernel.

```python
@kernel
def outer(x: i32, out: i32[1]):
    @kernel
    def add_one(v: i32) -> i32:
        return v + 1
    out[0] = add_one(x)
```

- Nested defs are **not** allowed inside `if`/`for`/`grid`/`while` bodies.
- **Recursion is rejected** (direct or indirect).
- **Captures are compile-time only**: a nested kernel may capture `constexpr`
  values, types/aliases, other kernels, `consteval` fns, operators, modules. It
  may **not** capture runtime values (outer params, local scalars, loop indices,
  buffers) — pass those explicitly as arguments.

---

## 3. Types & annotations

| Category          | Types                                        |
| ----------------- | -------------------------------------------- |
| Signed integers   | `i2`–`i16`, `i32`, `i64`, `i128`, `i256`     |
| Unsigned integers | `u1`–`u16`, `u32`, `u64`, `u128`, `u256`     |
| Floating point    | `f16`, `f32`, `f64`, `bf16`                  |
| Special           | `index`, `bool` (alias of `u1`), `constexpr` |

- `index` is the preferred type for loop indices and dynamic indices.
- Custom widths: `apint(width, signed=False)` (unsigned default),
  `apfloat(exp_width, sig_width)`.

```python
u17 = apint(17)
i23 = apint(23, signed=True)
```

### Shaped annotations

Written `dtype[shape]`. Shapes are compile-time integer expressions (literals,
visible constants, template params, unary `+`/`-`, binary `+ - * //`).

```python
@kernel
def reshape_like(inp: i32[M * N], out: i32[M, N]):
    for i, j in allo.grid(M, N):
        out[i, j] = inp[i * N + j]
```

- **Rank-0** shaped value: `dtype[()]` (or quoted `"dtype[]"`), indexed with `()`.
- With `KernelOptions(enable_tensor=True)`, the same syntax describes MLIR tensors
  (immutable, can be returned directly); the default is mutable buffers (memrefs).

---

## 4. Variables & scope

- **Annotated assignment declares a variable**: `base: i32 = x`.
- **Scalars must be initialized at declaration.** A runtime local can also be
  introduced by assigning an existing runtime value: `v = x`.
- **Shaped locals may be declared without an initializer** — allocates a local
  buffer: `buf: i32[4]`.
- **`constexpr` for compile-time variables** — must be annotated, initialized at
  declaration, never reassigned: `N: constexpr = 4`.
- **List initializers** for shaped values when every element is a compile-time
  `int`/`float`, with matching shape: `table: i32[2, 2] = [[1, 2], [3, 4]]`.
- **NumPy array initializers**: a captured NumPy array (a module global or
  closure variable) initializes a shaped local, equivalent to spelling out the
  list literal. The array's shape must match the annotation and its dtype must be
  integer or floating-point; elements are coerced to the declared element type.
  ```python
  W = np.array([[1, 2], [3, 4]], dtype=np.int32)   # captured from outer scope

  @kernel
  def top(out: i32[2, 2]):
      buf: i32[2, 2] = W       # constant buffer, baked into the module
  ```
- **Block scope**: variables declared inside `if`/`for`/`grid`/`while` are local to
  that block; declare before the block to use afterward. A name cannot be
  redeclared in the same scope; later assignments are cast to the original type.

---

## 5. Loops

`range` and `allo.range` both support 1/2/3-arg forms. **Label loops with
`name=`** so the schedule API can select them.

```python
for i in range(10, name="i"): ...
for i in range(10, 20): ...
for i in range(0, 20, 2): ...
```

- Loop bounds may be runtime values; non-`constexpr` steps must be positive.
- **`allo.grid`** is shorthand for a multidimensional loop (≥2 dims; target is a
  matching tuple). Dims may be ints or `(start, stop)` / `(start, stop, step)`
  tuples. Name the whole op with `allo.grid(..., name="ij")`.

```python
for i, j in allo.grid(32, 32):
    for k in range(32):
        c[i, j] += a[i, k] * b[k, j]
```

- **`grid` does not support loop-carried scalar accumulation across iterations** —
  use nested `range` loops for that.
- **`while`** supports runtime conditions and loop-carried scalar updates.
- **Not supported**: `break`, `continue`, `for...else`, `while...else`.

---

## 6. Conditionals

```python
if x == 0:
    result = 1
elif y > x:
    result = 2
else:
    result = 3
```

- Conditions use comparisons, `and`, `or`, `not`. **No multi-way comparison**
  (`a < b < c`) — write `a < b and b < c`.
- **Ternary** `x if cond else y` lowers to a select; at least one branch must be a
  runtime value for type inference.
- A `constexpr` condition is evaluated at compile time; only the taken branch is
  emitted.

### `match` / `case`

```python
match sel:          # sel must be a runtime integer (int or index)
    case 0:
        acc = 10
    case 1:
        acc = acc + 5
    case _:         # optional wildcard -> the default arm
        acc = 0
out[0] = acc
```

- Lowers to `scf.index_switch` and emits a C++ `switch` (one `case` per arm,
  each with an implicit `break` — **no fall-through**). The subject is
  index-cast before the switch.
- **Only integer-literal patterns** (`case 0:`, `case -1:`, or a `constexpr`
  that folds to an int) and the **wildcard** `case _:` are supported.
- The wildcard is optional; with no `case _:` the default arm is empty and an
  unmatched subject falls through to the code after the `match`.
- Scalars reassigned inside arms propagate out (phi): a value is threaded as a
  switch result, and an arm that does not redefine it keeps the incoming value.
- **Not supported:** OR-patterns (`case 0 | 1:`), guards (`case x if ...:`),
  capture/class/sequence/mapping patterns, and `return` inside an arm.

---

## 7. Operators & typing styles

| Category   | Operators                                     |
| ---------- | --------------------------------------------- |
| Arithmetic | `+ - * / // % **`                             |
| Unary      | `+x  -x  ~x  not x`                           |
| Comparison | `== != < <= > >=`                             |
| Boolean    | `and  or`                                     |
| Bitwise    | `& \| ^ << >>`                                |
| Assignment | `=  += -= *= /= //= %= **= &= \|= ^= <<= >>=` |

- `min`/`max` are built-ins (also `allo.max`/`allo.min`).
- Inside a kernel you may only call **Allo kernels, Allo operators, and
  `consteval` functions**. `print`/`len` are evaluated at compile time on
  compile-time args.
- **Typing style** (default `"hls"`): set via
  `@kernel(options=KernelOptions(typing_style="cpp"))`.
  - `hls`: hardware bit-growth — integer `+ - *` widen to preserve full
    intermediate precision and lower as **balanced trees** (e.g. `i32 + i32 -> i33`,
    `i32 * i32 -> i64`).
  - `cpp`: C++-style pairwise promotion to a common type (e.g. `i32 + i32 -> i32`).
  - See `docs/typing_rules.md` for full tables. Widen explicitly before
    shifts/mixed-width ops if you need a specific width.

---

## 8. Indexing & bit manipulation

- Shaped values use tuple indexing; index count must match rank: `dst[i, j]`.
  Rank-0 indexed with `()`.
- **Bit access on integer scalars**: `x[k]` reads/writes bit `k`; `x[lo:hi]`
  reads/writes the half-open range `[lo, hi)`. **Slice width must be a
  compile-time constant**; the offset may be dynamic.

```python
for p in range(4):
    out[p] = packed[p * 8 : p * 8 + 8]      # dynamic offset, constant width 8
    word[p * 8 : p * 8 + 8] = lanes[p]       # bit-range write
```

- **Not supported**: Python buffer slices (`A[0:4]`), partial subviews of a
  higher-rank buffer (`A[i]` for rank-2), `...`, and tensor methods like `.T` /
  `.copy()`.

---

## 9. Operator namespaces (`allo.operators`)

Python operators cover scalar and shaped elementwise math. Explicit operator
calls are useful when an op needs an output accumulator.

```python
from allo.operators import arith, math, linalg

arith.add(x, y, acc=out)                 # buffer mode: explicit acc=
```

- **`math`**: `exp exp2 log log2 abs pow sqrt rsqrt sin cos tan sinh cosh tanh
  floor ceil erf` (scalar or shaped).
- **`linalg`**: `matmul`, `dot`. Buffer mode passes `acc=`; tensor mode
  (`enable_tensor=True`) returns the value directly.

```python
@kernel(options=KernelOptions(enable_tensor=True))
def dense(a: f32[2, 3], b: f32[3, 4]) -> f32[2, 4]:
    return linalg.matmul(a, b)

@kernel
def buffer_matmul(a: f32[2, 3], b: f32[3, 4], out: f32[2, 4]):
    linalg.matmul(a, b, acc=out)
```

---

## 10. Streams (local FIFO channels)

`Stream[payload]` declares a FIFO. Payload is a scalar dtype or a shaped buffer.
Optional second bracket group declares an **array** of streams. Default depth `2`;
override with `Stream[i32, 8]`.

```python
fifo: Stream[i32]                # single rank-0 stream, depth 2
fifo.put(x)
v = fifo.get()

arr: Stream[i32][2, 2]           # 2x2 array of streams
arr[0, 1].put(x)
out[0] = arr[0, 1].get()

blk: Stream[i32[4, 4]]           # block payload: transfers a whole 4x4 buffer
```

- Streams are **declaration-only**: no initializer, **cannot** be return values.
  They are passed **explicitly to nested kernels** to connect stages.
- A stream array must be indexed with exactly one scalar index per dimension
  before `get()`/`put()`. Stream refs are not assignable.
- Vitis emission: scalar payload → `hls::stream<T>`; shaped payload →
  scalarized to `hls::stream<T>`
- Allo cannot auto generate testbench/host code if the top-level kernel has stream args,
  but the kernel code is still valid.

```python
@kernel
def nested_stream(x: i32, out: i32[1]):
    fifo: Stream[i32]
    @kernel
    def producer(v: i32, stream: Stream[i32]):
        stream.put(v + 1)
    @kernel
    def consumer(stream: Stream[i32], dst: i32[1]):
        dst[0] = stream.get()
    producer(x, fifo)
    consumer(fifo, out)
```

---

## 11. Stateful variables (C `static` semantics)

`Stateful[T]` marks a local declaration as persistent across kernel invocations
(backed by a module-level global). `T` is a scalar dtype or shaped type.
Declaration-only — not a param or return type.

```python
@kernel
def counter() -> i32:
    count: Stateful[i32] = 0
    count = count + 1
    return count            # returns 1, 2, 3, ... on successive calls
```

---

## 12. Spatial mapping (PE arrays)

`@kernel(mapping=[...])` replicates a kernel across a grid of worker instances
(PEs). The kernel is *invoked once*; the compiler specializes one hardware
function per worker. Inside the body, `allo.get_wid(axis)` is this worker's index
and `allo.get_nw(axis)` is the worker count along `axis`. Workers communicate
through **stream arrays**.

```python
@kernel
def systolic(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    fifo_A: Stream[f32][P0, P1]
    fifo_B: Stream[f32][P0, P1]

    @kernel(mapping=[P0, P1])
    def pe(A: f32[M, K], B: f32[K, N], C: f32[M, N],
           fifo_A: Stream[f32][P0, P1], fifo_B: Stream[f32][P0, P1]):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        # per-PE behavior selected by (i, j); idle corner PEs are pruned
        ...

    pe(A, B, C, fifo_A, fifo_B)
```

- Mapping dims accept constant `int`s and template variables (so array size is a
  specialization knob); mapping variables must bind to integers.
- See `tests/test_systolic_gemm.py` for a full output-stationary GEMM PE array.
  The schedule primitive `outline(..., mapping=...)` produces the same form from an
  existing loop.

---

## 13. Compile-time features

- **Global Python `int`/`float`** are visible as compile-time constants.
- **`@consteval`** marks a Python helper run during compilation; callable from
  kernels.

```python
@consteval
def factor():
    return 3
```

- **`@consteval(lazy=True)`** is a *lazy* variant: the helper is written in Allo
  kernel syntax (not plain Python) and lowered into the IR, then evaluated at
  compile time and folded to a constant before codegen — it never reaches the
  hardware. Use it when the computation needs kernel semantics (typed arithmetic,
  bit ops, loops, local arrays) but every call argument is a compile-time
  constant.

```python
@consteval(lazy=True)
def reverse_bits(data: i32, bit_range: i32) -> i32:
    mask = (1 << bit_range) - 1
    rev: i32 = 0
    for i in range(0, bit_range):
        i_32: i32 = i
        if data & (1 << i_32):
            rev |= 1 << (bit_range - 1 - i_32)
    return (data & ~mask) | rev
```

  Every call site must pass compile-time-constant arguments; each call is then
  replaced by the computed constant and the function is deleted. A non-constant
  argument, or a body that does not reduce to a constant (e.g. a `while` loop or
  a data-dependent trip count), is a compile error.

- **Templates** parameterize over compile-time types/values. A templated kernel is
  not concrete until specialized with `kernel[...]`.

```python
T = Template("T")
N = Template("N")

@kernel(T, N)
def fill_template(x: T, out: T[N]):
    for i in range(N):
        out[i] = x

fill_i32_4 = fill_template[i32, 4]      # specialize before compile/run
```

- Type templates work in scalar annotations and as the head of shaped
  annotations; integer templates in shape exprs, loop bounds, and `mapping`.
- A global alias `T = i32` is concrete immediately (callers can't specialize); a
  `Template("T")` is a delayed binding the caller must supply.

---

## 14. `KernelOptions`

Passed as `@kernel(options=KernelOptions(...))`:

| Option               | Effect                                              |
| -------------------- | --------------------------------------------------- |
| `typing_style="hls"` | (default) HLS bit-growth; `"cpp"` for C++ promotion |
| `enable_tensor=True` | Shaped annotations are MLIR tensors, not buffers    |
| `fast_math=True`     | Allow float add/sub reassociation (n-ary lowering)  |

---

## 15. Scheduling API

`kernel.schedule()` compiles the kernel and returns a `Schedule` decoupling
algorithm from hardware mapping. Select targets by name, apply primitives, then
inspect `s.payload` or hand off with `s.export(...)`. A **templated kernel must be
specialized first** (`gemm[i32, 32].schedule()`).

```python
s = top.schedule()
i = s.loop("i")
outer, inner = s.split(i, factor=4)
s.pipeline(inner, ii=1).apply()
```

### Selecting targets

| Alias                                 | Result                            |
| ------------------------------------- | --------------------------------- |
| `s.loop(name, *, under=, path=)`      | one `LoopRef`                     |
| `s.loops(*names, under=, path=)`      | tuple of `LoopRef` (all if empty) |
| `s.op(name, *, under=, kind=, path=)` | one `OpRef`                       |
| `s.buffer(name, *, under=, path=)`    | one `BufferRef`                   |

Loop names come from the frontend `name=`. Low-level `s.query.loop/op/buffer(...)`
returns a `RefSelection` with `.one()` / `.first()` / `.all()` / `.names(*names)`.
`under=` scopes the lookup under another op.

### Tagging primitives (deferred — chain then `.apply()`)

| Primitive                                                    | Effect                                                          |
| ------------------------------------------------------------ | --------------------------------------------------------------- |
| `s.pipeline(targets=None, *, ii=1)`                          | Pipeline a loop with initiation interval `ii`.                  |
| `s.dataflow(targets=None)`                                   | `#pragma HLS dataflow` — run top-level stmts concurrently.      |
| `s.unroll(targets, *, factor=0, tag_only=False)`             | Unroll (`factor=0` = full). Physical unroll unless `tag_only`.  |
| `s.partition(targets, *, dim=0, kind=, factor=0)`            | Bank a buffer. `kind` ∈ `s.Complete`/`s.Block`/`s.Cyclic`.      |
| `s.bind_storage(targets, *, impl=s.BRAM, mem_type=s.RAM_2P)` | `#pragma HLS bind_storage` — pin a buffer to a memory resource. |
| `s.cse/dce/licm/canonicalize(targets=None)`                  | Generic MLIR cleanup passes.                                    |

`partition`: `dim=0` = all dims; `s.Complete` needs `factor=0`, `s.Block`/`s.Cyclic`
need `factor>0`.

`bind_storage`: `impl` ∈ `s.BRAM`/`s.URAM`/`s.LUTRAM`/`s.SRL`/`s.AUTO`/…; `mem_type` ∈
`s.RAM_1P`/`s.RAM_2P`/`s.RAM_S2P`/`s.RAM_T2P`/`s.ROM_*`/…. Vitis-only hint (CPU ignores it).

### Structural primitives (apply immediately; return live refs — old refs go stale)

| Primitive                                       | Returns                                                        |
| ----------------------------------------------- | -------------------------------------------------------------- |
| `s.affine(targets=None)`                        | `[LoopRef]` (scf → affine)                                     |
| `s.split(target, *, factor=1)`                  | `(outer, inner)`                                               |
| `s.reorder(targets)`                            | reordered `LoopRef`s (≥2 affine)                               |
| `s.tile(targets, *, factors=1)`                 | `(tile_loops, point_loops)`                                    |
| `s.flatten(targets)`                            | one `LoopRef` (≥2 perfectly-nested loops)                      |
| `s.compute_at(target, axis)`                    | `LoopRef` (fuse producer into consumer loop)                   |
| `s.buffer_at(target, axis)`                     | `BufferRef` (`{base}.local`)                                   |
| `s.reuse_at(target, axis, *, ring=False)`       | `BufferRef` (`{base}.reuse`, line/window buffer)               |
| `s.outline(target, *, func_name, mapping=None)` | `(kernel_ref, call_ref)`. `mapping` int/seq → spatial PE form. |

After a structural transform invalidates a ref, recover it via the returned ref,
re-select by name, or `s.live(ref)`. Reusing a consumed ref raises
`ConsumedHandleError`.

### Composition

```python
gemm_s = gemm.schedule()
gemm_s.tile(gemm_s.loops("i", "j"), factors=[4, 4])

top_s = top.schedule()          # top invokes gemm
top_s.compose(gemm_s)           # replay gemm's schedule onto top's private copy
```

- `s.compose(*callees, id=None)` — replay each direct callee's *entire* schedule
  onto the specialized copy `"{primary}.{callee}"`. Variadic = compose each in
  turn. Transitive. `id=` selects a specific repeat copy.

### Streaming

```python
s.compose(stage_a, stage_b, stage_c)
s.streamline("stage_a", "stage_b")               # DRAM boundary -> on-chip FIFO
s.streamline("stage_b", "stage_c", lanes=4, depth=8)
s.dataflow()                                     # run stages concurrently
```

- `s.streamline(producer, consumer, *, producer_ids=None, consumer_ids=None,
  lanes=1, depth=2)` converts a DRAM hand-off between composed stages into a
  stream. One→one = FIFO; one→many = `tee` (skip/residual); many→one = `merge`
  (each producer fills a disjoint contiguous block). `lanes` = parallel FIFOs
  (contiguous dim must divide by it); `depth` must cover fork/join latency skew or
  the dataflow **deadlocks** (the warning names the depth to set).

### Apply & export

```python
s.apply() # mannually apply pending tags; auto-applied by export
code   = s.export("vitis").hls_code
report = s.export("vitis", part=PART, project_path=proj).synth()
s.export("cpu")(A, B, C)        # functional run
```

- **`s.export()` will mutate the original module in `Kernel`**.
- Reading `s.payload` / `s.snapshot` while `dirty` auto-applies pending tags.
- `s.export(backend, **kwargs)` (`"cpu"`/`"vitis"`) applies transforms, binds the
  module back to the kernel, returns a backend object. Schedules built from a raw
  module/string/file (`Schedule.from_module/from_string/from_file`) support every
  transform but **cannot be exported**.
- Debug: `s.dump_tree()`, `s.dump_transform_script()`, `s.debug_dump()`.

---

## 16. Simulation & Building

An Allo kernel is a callable. A **direct call runs the CPU backend** with default
options; mixing NumPy and kernel calls is free.

```python
A = np.arange(N, dtype=np.float32); C = np.zeros(N, dtype=np.float32)
vec_add(A, B, C)                            # CPU backend; C updated in place
```

### CPU backend (functional validation)

```python
backend = s.export("cpu", opt_level=3)      # or default opt_level=2
backend(A, B, C)                            # == backend.run(A, B, C)
```

**CPU backend may deadlock**. Its functionality is not fully verified; it serves
as a convenient reference for expected behavior when Vitis is not available.
**Use Vitis C-simulation for reliable CPU validation** (no need to build an full project).


### Vitis backend (HLS codegen / csim / synth / emu)

```python
mod = s.export("vitis", device="u55c") # or specify part="..."
ret = mod(A, B, C)              # csim (Python-native): == mod.csim(A, B, C)
mod.synth()         # C-to-RTL synthesis;

# generate sample input for emulation
A = np.arange(N, dtype=np.float32); B = np.arange(N, dtype=np.float32)
C = 1024 # a scalar input (e.g. a control flag) is also accepted

# set top-level interfaces
mod.set_axi(0, bundle="gmem0")
mod.set_axi(1, bundle="gmem1")
mod.set_axilite(2)
mod.set_axilite(-1) # return

# generate Vitis project
mod.scaffold_project("/path/to/proj", A, B, C) # A, B, C packed into binary files
# then go to shell to run emulation/implementation
```

```bash
cd /path/to/proj
make target
```

- See [ENVIRONMENT.md](ENVIRONMENT.md) for available make targets and env setup.
- Prefer to do C-simulation with Python API (convenient enough),
  but run cosimulation and hardware implementation in shell with `make`.
- Target by `part="<full-part>"` **or** `device="<shorthand>"` (e.g. `pynqz2`,
  `u280`, `zcu102`) — not both. Constructor knobs: `freq_mhz=300.0`,
  `flow="vitis"|"vivado"`.
- `hw_emu`/`hw` go through `v++`/XRT and need `export PLATFORM=/path/<shell>.xpfm`;
  `b.precheck(mode)` validates a buildable project without the long link step.
- Interface pragmas: `b.set_axi(idx, ...)` (m_axi buffer), `b.set_axis(idx, ...)`
  (axis stream), `b.set_axilite(idx, ...)` (`-1` = return value).
- All run/csim/synth accept `exist_ok=True` (default; `False` forces rebuild).

#### Synthesis report parsing

Synthesis and report inspection are **decoupled**: synthesize once, then parse
`mod.synth_report` (the path to `csynth.xml`) as often as you like — no re-synth.

```python
from allo.backend.vitis import parse_report

mod = s.export("vitis", device="u280", project_path="prj")
mod.scaffold_project()              # writes prj/ (or pass mod.synth() to build)

if sys.argv[1] == "synth":
    mod.synth()                     # slow: invokes Vitis HLS once
elif sys.argv[1] == "report":
    r = parse_report(mod.synth_report)   # fast: just parses the XML
    print(r)                      # readable overall + per-module summary
```

`parse_report(path)` accepts the `csynth.xml` file or its directory and returns
a `SynthReport`. **Fixed-schema fields use attribute access** (typed,
discoverable); **open-ended collections are a dict / list**:

```python
r.version                  # Vitis tool version, e.g. "2023.2"
r.part                     # FPGA part number; r.product_family, r.top
r.fmax                     # achievable clock (MHz) = 1000 / estimated period
r.timing.estimated_clock_ns, r.timing.target_clock_ns
r.latency.worst_cycles     # total cycles (None if data-dependent / "undef")
r.latency.worst_time       # equivalent wall-clock time, e.g. "61.730 us"
r.latency.interval_min     # initiation interval (II); .pipeline_type, ...
r.resources.lut            # whole-design usage: .lut/.ff/.dsp/.bram/.uram
r.available.lut            # device capacity (same fields)
r.utilization["lut"]       # % of device per resource

for itf in r.interfaces:   # grouped per bundle (m_axi/s_axi/axis/ap_ctrl)
    itf.name, itf.protocol, itf.data_bits

m = r.modules["top"]       # per-module breakdown (incl. the top module)
m.resources.dsp, m.latency.pipeline_ii, m.timing.fmax_mhz
```

`BRAM` counts are in 18K-block units (matching the report). Loop-level detail is
not parsed — module granularity only.

#### Vitis-specific environment variables
- `XILINX_VITIS` points to the Vitis install path;
  no need to source the full Vitis setup scripts before running Allo code.
- `VIVADO_IMPL_JOBS` (default `4`) controls parallelism in the Vivado implementation
- `ALLO_ENABLE_VITIS_APFLOAT=1` (default "0") to force `ap_float` Vitis codegen support;
  `ap_float` is required for `bf16/tf32` support but needs Vitis 2023.1+ (inclusive) to build.
  If unset, Allo auto-detects Vitis version and enables `ap_float` when supported.

### Calling convention

- **Buffer args**: NumPy arrays, validated against shape/dtype, written back
  in place — prefer **in-place output buffers** (portable across CPU/Vitis).
- **Scalar args**: Python numbers. **Scalar returns supported** on both backends.
- **Vitis top kernels reject shaped return values** — pass shaped outputs as
  buffer args.
- `Stream` values are internal only — never top-level Python call args; use NumPy
  buffers at the boundary.
- CPU sim runs local streams through a dataflow simulator; contiguous
  stream-connected nested calls run concurrently (OpenMP sections). Bounded FIFOs:
  `put` blocks when full, `get` when empty — an imbalanced pair can deadlock.

---

## 17. Library module pattern (`allo.lang.Module`)

Reusable designs in `allo/library/` follow a consistent shape: a `_make(...)`
factory builds the kernel **and** its schedule, wrapped in a `Module` subclass.

```python
from allo.lang import Module

def _make(Tin, Tacc, Tout, S, D, L=16, ...):
    @kernel
    def top(x: Tin[S, D], g: Tin[D], y: Tout[S, D]):
        ...                            # algorithm
    s = top.schedule()
    s.partition(s.buffer("buf"), dim=2, kind=s.Cyclic, factor=L)
    s.unroll("rl")
    s.pipeline(s.flatten(("rct", "rs")), ii=ii)
    return top, s

class RMSNorm(Module):
    def __init__(self, Tin, Tacc, Tout, S, D, L=16, ...):
        # validate args with raise, build a stable name
        top, s = _make(Tin, Tacc, Tout, S, D, L, ...)
        super().__init__(f"RMSNorm_S{S}_D{D}_L{L}", top, s)
```

`Module(name, module, schedule)` exposes `.module` (the `Kernel`), `.schedule`,
and is itself callable (`__call__` forwards to the kernel).

---

## 18. Common restrictions (the frontend rejects these early)

- All kernel parameters require annotations; returning a value requires a return
  annotation.
- `return` not allowed inside loops or nested `if`.
- No `break`, `continue`, loop `else`, arbitrary Python calls, attribute
  assignment, chained assignment (`a = b = c`), or multi-way comparison.
- `constexpr` must be annotated, initialized once, never reassigned.
- Nested kernels cannot capture runtime values from an outer scope.
- `Stream` / `Stateful` can be declared in a body but not be top-level params or
  return values (`Stream` is passed to nested kernels explicitly).
- No recursion (direct or indirect).
- Bit-slice width must be compile-time constant.
- No Python buffer slices, partial subviews, `...` shapes, or tensor methods
  (`.T`, `.copy()`).
- The Allo frontend **omits C-style int promotion in `hls` typing** — arithmetic
  stays in operand width (with HLS bit-growth) and truncates on store; widen
  explicitly (e.g. to `i32`) before shifts/mixed-width ops.
