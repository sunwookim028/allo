<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Frontend Syntax

The Allo frontend is a restricted Python-embedded DSL (eDSL). It reuses Python
syntax for readability, but only the constructs described here are part of the
kernel language. Kernels are written as plain Python functions and import their
building blocks from `allo`:

```python
import allo
from allo.lang import bool, f32, i32, u1, u32, kernel, Stream
```

Most type names, `kernel`, `consteval`, `KernelOptions`, `Template`, `constexpr`,
`Stream`, `grid`, and `range` are re-exported from `allo.lang`. A few markers
that are not in `allo.lang` — `Stateful` and the explicit `APInt` class — come
from `allo.lang.core`. The spatial built-ins `allo.get_wid` / `allo.get_nw` and
helpers such as `allo.grid`, `allo.range`, and `allo.max` are available on the
top-level `allo` namespace.

## Kernel Definition

Allo kernels are Python functions decorated with `@kernel`. Every parameter must
have a type annotation.

```python
from allo.lang import f32, kernel

@kernel
def saxpy(a: f32, x: f32[16], y: f32[16], out: f32[16]):
    for i in range(16):
        out[i] = a * x[i] + y[i]
```

Scalar annotations use type names directly. Shaped annotations use
`dtype[shape]`:

```python
@kernel
def scalar_add(x: i32, y: i32) -> i32:
    return x + y

@kernel
def vector_add(x: i32[16], y: i32[16]) -> i32[16]:
    out: i32[16] = 0
    for i in range(16):
        out[i] = x[i] + y[i]
    return out
```

Without `from __future__ import annotations`, Python evaluates annotations
before Allo sees them. In that mode, scalar annotations such as `x: i32` still
work, but shaped annotations must be quoted, for example `x: "i32[16]"`. With
postponed annotations, prefer importing Allo types into the file's scope and
using bare names such as `u32[4]`.

Functions with no return value can omit the return annotation or use `-> None`.
Returning a value requires an **explicit return annotation**.

```python
@kernel
def fill(out: i32[4]):
    for i in range(4):
        out[i] = i

@kernel
def no_result(out: i32[4]) -> None:
    return
```

Multiple return values are written as tuple annotations.

```python
@kernel
def split_pair(x: i32, y: f32) -> (i32, f32):
    return x + 1, y + 1.0

@kernel
def caller(x: i32, y: f32, out: f32[1]):
    lhs, rhs = split_pair(x, y)
    out[0] = rhs + lhs
```

Return placement is **intentionally restricted**. A return may appear at the top
level of the kernel body or in a first-level `if`/`else` branch. Returns inside
loops and nested `if` statements are rejected.

```python
@kernel
def choose(cond: bool, x: i32, y: i32) -> i32:
    if cond:
        return x
    return y
```

Kernels can define nested kernels as local helpers. A nested kernel must be
declared at the top level of the enclosing kernel body, use exactly one
`@kernel` decorator, and can be called like any other kernel.

```python
@kernel
def outer(x: i32, out: i32[1]):
    @kernel
    def add_one(v: i32) -> i32:
        return v + 1

    out[0] = add_one(x)
```

Nested kernel definitions are not allowed inside `if`, `for`, `grid`, or `while`
bodies. Recursive kernel calls are rejected, including indirect recursion across
multiple kernels.

## Types and Annotations

The scalar types are:

|     Category      |                         Types                         |
| :---------------: | :---------------------------------------------------: |
|  Signed integers  | `i2` through `i16`, plus `i32`, `i64`, `i128`, `i256` |
| Unsigned integers | `u1` through `u16`, plus `u32`, `u64`, `u128`, `u256` |
|  Floating point   |              `f16`, `f32`, `f64`, `bf16`              |
|      Special      |             `index`, `bool`, `constexpr`              |

`bool` is an alias for `u1`. `index` is the preferred type for loop indices and
values used as dynamic indices.

Use `apint(width, signed=False)` for custom integer widths beyond the predefined
aliases. Unsigned is the default; pass `signed=True` for signed integers.
`apfloat(exp_width, sig_width)` constructs the supported floats.

```python
from allo.lang import apint, kernel

u17 = apint(17)
i23 = apint(23, signed=True)

@kernel
def custom_width(x: u17, y: i23, out: u17[1]):
    out[0] = x + y
```

Shaped values are written as `dtype[shape]`. With postponed annotations, a
rank-0 shaped value is written as `dtype[()]` because Python does not allow an
empty subscript. The quoted spelling `"dtype[]"` is also accepted.

```python
@kernel
def shapes(a: f32[8], b: i32[4, 4], acc: f32[()]):
    acc[()] = a[0] + b[0, 0]
```

Shape expressions are compile-time integer expressions. They may use integer
literals, visible constants, template parameters, unary `+`/`-`, and the binary
operators `+`, `-`, `*`, and `//`.

```python
M = 4
N = 8

@kernel
def reshape_like(inp: i32[M * N], out: i32[M, N]):
    for i, j in allo.grid(M, N):
        out[i, j] = inp[i * N + j]
```

By default, shaped annotations describe mutable buffers. With
`KernelOptions(enable_tensor=True)`, the same syntax describes MLIR tensors.

```python
from allo.lang import KernelOptions

@kernel(options=KernelOptions(enable_tensor=True))
def tensor_add(x: f32[4], y: f32[4]) -> f32[4]:
    return x + y
```

### Streams

`Stream` describes local FIFO channels. The payload type can be a scalar dtype
or a shaped buffer payload. The optional second bracket group describes an array
of streams; omitting it creates a single rank-0 stream. The default stream depth
is `2`; write `Stream[i32, 8]` to override it.

```python
from allo.lang import Stream, i32

@kernel
def scalar_stream(x: i32, out: i32[1]):
    fifo: Stream[i32]
    fifo.put(x)
    out[0] = fifo.get()

@kernel
def stream_array(x: i32, out: i32[1]):
    fifo: Stream[i32][2, 2]
    fifo[0, 1].put(x)
    out[0] = fifo[0, 1].get()
```

A stream with a shaped payload transfers a whole block. In Vitis HLS emission,
scalar payloads map to `hls::stream<T>` and shaped payloads map to
`hls::stream_of_blocks<T[...], depth>`.

```python
@kernel
def block_stream(out: i32[1]):
    fifo: Stream[i32[4, 4]]
    buf: i32[4, 4]
    buf[0, 0] = 7
    fifo.put(buf)
    recv = fifo.get()
    out[0] = recv[0, 0]
```

Streams can be passed explicitly to nested kernels. This is the supported way to
connect producer and consumer stages inside one top-level kernel.

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

Streams must be declared without initializers. A stream array must be indexed
with exactly one scalar index per stream dimension before `get()` or `put()`.
Stream references are not assignable; use `put(value)` to write and `get()` to
read. Stream values are not valid kernel return values.

### Stateful Variables

`Stateful[T]` marks a local declaration as persistent across kernel invocations,
matching C `static` semantics. The backing storage is a module-level global, so
the value survives between calls. `T` may be a scalar dtype or a shaped buffer
type; `Stateful` is declaration-only and cannot be a parameter or return type.

```python
from allo.lang import Stateful

@kernel
def counter() -> i32:
    count: Stateful[i32] = 0
    count = count + 1
    return count


# counter() returns 1, 2, 3, ... on successive calls.
```

A stateful array persists element writes the same way:

```python
@kernel
def running_sum(x: f32) -> f32:
    acc: Stateful[f32[1]] = 0.0
    acc[0] = acc[0] + x
    return acc[0]
```

A stateful scalar mutated inside a loop is deliberately kept out of the
loop-carried (SSA) machinery; it loads and stores its global on each access.

## Variables and Scope

Annotated assignments declare variables.

```python
@kernel
def declarations(x: i32, out: i32[4]):
    base: i32 = x
    tmp: i32[4] = 0
    for i in range(4):
        tmp[i] = base + i
        out[i] = tmp[i]
```

Shaped locals may be declared without an initializer. This allocates a local
buffer (or an empty tensor in tensor mode).

```python
@kernel
def local_buffer(out: i32[4]):
    buf: i32[4]
    for i in range(4):
        buf[i] = i
        out[i] = buf[i]
```

Scalar variables must be initialized when declared. A runtime local can also be
introduced by assigning an existing runtime value.

```python
@kernel
def inferred_local(cond: bool, x: i32, y: i32, out: i32[1]):
    v = x
    if cond:
        v = y
    else:
        v = x + y
    out[0] = v
```

Compile-time variables must be declared with `constexpr`. They are evaluated
during compilation and cannot be reassigned.

```python
from allo.lang import constexpr

@kernel
def constexpr_bound(out: i32[4]):
    N: constexpr = 4
    for i in range(N):
        out[i] = i
```

List initializers are supported for shaped values when every element is a
compile-time `int` or `float`. The list shape must match the annotation.

```python
@kernel
def constants(out: i32[2, 2]):
    scale: constexpr = 3
    table: i32[2, 2] = [[1, scale], [scale + 1, scale + 2]]
    for i, j in allo.grid(2, 2):
        out[i, j] = table[i, j]
```

Allo uses block scope. Variables declared inside an `if`, `for`, `grid`, or
`while` body are local to that block. Declare a variable before the block if it
must be used afterward. A name cannot be redeclared in the same scope; later
assignments are cast back to the variable's original type.

Nested kernels follow the same scoping model, but their captures are deliberately
limited. They may capture compile-time symbols from the enclosing scope:
`constexpr` values, concrete types, type aliases, other kernels, `consteval`
functions, Allo operators, and modules. They may **not** capture runtime values
such as enclosing kernel parameters, local scalar variables, loop indices, or
buffers. Pass runtime values explicitly as nested-kernel arguments.

```python
@kernel
def captures(x: i32, out: i32[1]):
    offset: constexpr = 2
    T: constexpr = i32

    @kernel
    def add_offset(v: T) -> T:
        return v + offset

    out[0] = add_offset(x)
```

## Loops

Both Python `range` and `allo.range` are supported, with the same one-, two-, or
three-argument forms. Use Allo's `range`/`grid` with the optional `name=`
keyword to label a loop so the [scheduling](scheduling.md) API can select it.

```python
from allo.lang import range

@kernel
def ranges(out: i32[20]):
    for i in range(10, name="i"):
        out[i] = i
    for i in range(10, 20):
        out[i] = i
    for i in range(0, 20, 2):
        out[i] = i * 2
```

Loop bounds may depend on runtime values. Loop steps must be positive if they are
not `constexpr`.

```python
@kernel
def variable_bounds(a: i32[10], out: i32[10]):
    for i in range(10):
        for j in range(a[i], 10, a[i]):
            out[j] += i
```

`allo.grid` is a shorthand for a multidimensional loop. It requires at least two
dimensions, and the loop target must be a tuple with the same number of
variables.

```python
@kernel
def matmul(a: f32[32, 32], b: f32[32, 32]) -> f32[32, 32]:
    c: f32[32, 32] = 0.0
    for i, j in allo.grid(32, 32):
        for k in range(32):
            c[i, j] += a[i, k] * b[k, j]
    return c
```

Grid dimensions may also be written as `(start, stop)` or `(start, stop, step)`
tuples. A whole `grid` can be named with `allo.grid(..., name="ij")`, which
labels the loop-like operation (not the individual axes).

```python
@kernel
def strided_grid(out: i32[8, 8]):
    for i, j in allo.grid((0, 8, 2), (1, 8, 2)):
        out[i, j] = i + j
```

`grid` does not support non-trivial loop-carried scalar dependencies. Use nested
`range` loops when the body must update a scalar accumulator across iterations.

`while` loops are supported for runtime conditions and may update loop-carried
scalar values.

```python
@kernel
def count(out: i32[1]):
    i: i32 = 0
    acc: i32 = 0
    while i < 4:
        acc += i
        i += 1
    out[0] = acc
```

`break`, `continue`, `for ... else`, and `while ... else` are not supported.

## Conditionals

Runtime `if`/`elif`/`else` statements lower to structured control flow. Variables
declared outside the conditional can be assigned in either branch and used
afterward.

```python
@kernel
def classify(x: i32, y: i32) -> i32:
    result: i32 = 0
    if x == 0:
        result = 1
    elif y > x:
        result = 2
    else:
        result = 3
    return result
```

Conditions may use comparison operators, `and`, `or`, and `not`. Ternary
expressions lower to a select when the condition is runtime; at least one branch
must be a runtime value so the result type can be inferred.

```python
@kernel
def select(cond: bool, x: i32, y: i32) -> i32:
    return x if cond else y
```

If a condition is a `constexpr`, the frontend evaluates it during compilation and
emits only the selected branch.

## Expressions and Operators

The frontend supports the following Python operators.

|  Category  |                                    Operators                                     |
| :--------: | :------------------------------------------------------------------------------: |
| Arithmetic |                       `+`, `-`, `*`, `/`, `//`, `%`, `**`                        |
|   Unary    |                            `+x`, `-x`, `~x`, `not x`                             |
| Comparison |                         `==`, `!=`, `<`, `<=`, `>`, `>=`                         |
|  Boolean   |                                   `and`, `or`                                    |
|  Bitwise   |                            `&`, `\|`, `^`, `<<`, `>>`                            |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `\|=`, `^=`, `<<=`, `>>=` |

Multi-way comparisons such as `a < b < c` are not supported; write them with
`and`.

```python
@kernel
def comparisons(a: i32, b: i32, c: i32) -> bool:
    return a < b and b < c
```

The default `typing_style` is `"hls"`, which uses HLS-oriented integer promotion
(an addition may widen internally and then cast back to the destination type).
`KernelOptions(typing_style="cpp")` selects C++-style promotion rules. See the
[Typing Rules](typing_rules.md) reference for the full tables.

```python
@kernel(options=KernelOptions(typing_style="cpp"))
def cpp_style(x: u32, y: i32, out: u32[1]):
    out[0] = x + y
```

`min` and `max` are supported as built-ins and lower to Allo arithmetic
operators.

```python
@kernel
def clamp(x: i32, lo: i32, hi: i32) -> i32:
    return min(max(x, lo), hi)
```

Only Allo kernels, Allo operators, and `consteval` functions may be called from
inside a kernel. The static built-ins `print` and `len` are evaluated during
compilation when their arguments are compile-time values.

## Indexing and Memory Access

Shaped values use tuple-style indexing. The number of indices must match the
rank.

```python
@kernel
def copy_2d(src: f32[4, 4], dst: f32[4, 4]):
    for i, j in allo.grid(4, 4):
        dst[i, j] = src[i, j]
```

Rank-0 shaped values are indexed with `()`.

```python
from allo.operators import linalg

@kernel(options=KernelOptions(enable_tensor=True))
def dot_scalar(a: f32[4], b: f32[4]) -> f32:
    return linalg.dot(a, b)[()]
```

### Bit Manipulation

Integer scalar values support single-bit and bit-range access with subscript
syntax. `x[k]` reads bit `k`; `x[lo:hi]` reads the half-open bit range
`[lo, hi)`. The same forms on the left-hand side write bits. The **width** of a
slice must be a compile-time constant, but the offset may be dynamic.

```python
@kernel
def bit_ops(x: u32, out: u1[1]):
    out[0] = x[3]            # single bit

@kernel
def unpack(packed: u32, out: u8[4]):
    for p in range(4):
        out[p] = packed[p * 8 : p * 8 + 8]   # dynamic offset, constant width=8

@kernel
def pack(lanes: u8[4]) -> u32:
    word: u32 = 0
    for p in range(4):
        word[p * 8 : p * 8 + 8] = lanes[p]   # bit-range write
    return word
```

A bit slice may be applied to a scalar loaded from a buffer element, e.g.
`packed[i][p * 8 : p * 8 + 8]`. A slice whose width is not a compile-time
constant (`x[lo:hi]` with runtime `lo` and `hi`) is rejected.

Python slice indices on buffers such as `A[0:4]`, partial subviews such as `A[i]`
for a rank-2 buffer, and `...` are not part of the frontend.

## Operator Namespaces

Python operators cover scalar arithmetic and shaped elementwise expressions.
Explicit operator calls are useful when an operation needs an output accumulator.
The operator modules live under `allo.operators`.

```python
from allo.operators import arith, linalg, math

@kernel
def memref_elementwise(x: f32[4], y: f32[4], out: f32[4]):
    arith.add(x, y, acc=out)
```

Math operators include `exp`, `exp2`, `log`, `log2`, `abs`, `pow`, `sqrt`,
`rsqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, and
`erf`. They work on scalar and shaped values.

```python
@kernel
def sigmoid(x: f32[8], out: f32[8]):
    for i in range(8):
        out[i] = 1.0 / (1.0 + math.exp(-x[i]))
```

Linalg operators include `matmul` and `dot`. They support both buffer mode and
tensor mode. In buffer mode, pass an explicit `acc=` output. In tensor mode, the
same operation can return a tensor value directly.

```python
@kernel(options=KernelOptions(enable_tensor=True))
def dense(a: f32[2, 3], b: f32[3, 4]) -> f32[2, 4]:
    return linalg.matmul(a, b)

@kernel
def buffer_matmul(a: f32[2, 3], b: f32[3, 4], out: f32[2, 4]):
    linalg.matmul(a, b, acc=out)
```

## Spatial Mapping

A `@kernel(mapping=[...])` describes a grid of worker instances — a spatial
array of processing elements (PEs). The kernel is invoked once, but the compiler
replicates it across the mapping grid and specializes each worker into its own
hardware function. Inside the body, `allo.get_wid(axis)` returns this worker's
index along a mapping axis, and `allo.get_nw(axis)` returns the number of workers
along that axis. Workers communicate through stream arrays.

```python
import allo
from allo.lang import f32, kernel, Stream

M, N, K = 2, 2, 2
P0, P1 = M + 2, N + 2

@kernel
def systolic(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    fifo_A: Stream[f32][P0, P1]
    fifo_B: Stream[f32][P0, P1]

    @kernel(mapping=[P0, P1])
    def pe(
        A: f32[M, K],
        B: f32[K, N],
        C: f32[M, N],
        fifo_A: Stream[f32][P0, P1],
        fifo_B: Stream[f32][P0, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        # ... per-PE behavior selected by (i, j) ...

    pe(A, B, C, fifo_A, fifo_B)
```

A mapping dimension may be a `constexpr` or a template parameter (so the array
size is a specialization knob). The `mapping` argument accepts constant `int`s
and template variables; mapping variables must bind to integers. The
[scheduling](scheduling.md) `outline(..., mapping=...)` primitive produces the
same spatial form from an existing loop nest.

## Compile-Time Features

Global Python `int` and `float` values are visible as compile-time constants.

```python
SCALE = 3

@kernel
def add_scale(x: i32) -> i32:
    return x + SCALE
```

`consteval` marks a Python helper that runs during compilation.

```python
from allo.lang import consteval

@consteval
def factor():
    return 3

@kernel
def use_factor(x: i32) -> i32:
    return x + factor()
```

Templates parameterize kernels over compile-time types and values. A templated
kernel is not concrete until it is specialized with `kernel[...]`.

```python
from allo.lang import Template, f32, i32, kernel

T = Template("T")
N = Template("N")

@kernel(T, N)
def fill_template(x: T, out: T[N]):
    for i in range(N):
        out[i] = x

fill_i32_4 = fill_template[i32, 4]
```

Template bindings must be provided before compilation or execution. Type
templates can be used in scalar annotations and as the head of shaped
annotations. Integer templates can be used in shape expressions, loop bounds, and
the `mapping` list.

Templates differ from ordinary global aliases. A global alias such as `T = i32`
is a concrete type chosen immediately; callers cannot specialize it. A
`Template("T")` is a delayed binding point that the caller must supply.

```python
FixedT = i32

@kernel
def fixed_alias(x: FixedT, out: FixedT[4]):
    for i in range(4):
        out[i] = x

T = Template("T")

@kernel(T)
def delayed_type(x: T, out: T[4]):
    for i in range(4):
        out[i] = x

delayed_i32 = delayed_type[i32]
delayed_f32 = delayed_type[f32]
```

## Diagnostics

The frontend reports compilation errors in a clang-like style. Diagnostics
include the source file, line, column, message, the relevant source line, and a
caret span pointing at the AST node that triggered the error.

For example, an undefined name in:

```python
def broken(x):
    return x + y
```

is rendered as:

```text
broken.py:11:16: error: Name 'y' is not defined
11 |     return x + y
   |                ^
```

The same format is used for missing annotations, unsupported control flow, return
type mismatches, illegal captures, and invalid operator calls. When an error
occurs while compiling a called or nested kernel, the message is wrapped with
call context so the caller/callee relationship is visible.

Source locations are based on Python source inspection. They are reliable for
kernels defined in normal `.py` files. In a REPL, notebook, `python -c`, or other
dynamically generated context, Python may not expose stable source lines; Allo
still reports the error, but file names and line numbers can be missing or
inaccurate. For compiler debugging, set `ALLO_SHOW_COMPILER_TRACEBACK=1` to keep
the full Python traceback instead of the shortened user diagnostic.

## Common Restrictions

The frontend intentionally rejects unsupported Python early and reports the
source location. The most important restrictions are:

- All kernel parameters require explicit annotations.
- Returning a value requires an explicit return annotation.
- `return` is not supported inside loops or nested `if` statements.
- `break`, `continue`, loop `else` blocks, arbitrary Python calls, attribute
  assignment, chained assignment such as `a = b = c`, and multi-way comparisons
  are not supported.
- `constexpr` variables must be explicitly annotated, initialized at declaration,
  and never reassigned.
- Runtime values from an outer kernel scope cannot be captured by nested kernels.
- `Stream` and `Stateful` may be declared in a kernel body but cannot be kernel
  parameters or return values. (`Stream` is passed explicitly to nested kernels.)
- Recursive kernel calls, including indirect recursion through nested kernels,
  are not supported.
- A bit slice must have a compile-time-constant width.
- Python buffer slices, partial tensor subviews, dynamic `...` shapes, and tensor
  methods such as `.T` and `.copy()` are not supported.
