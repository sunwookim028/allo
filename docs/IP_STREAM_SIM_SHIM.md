<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Running an `hls::stream` IP under the CPU dataflow simulator

This document explains the change that lets a hand-written HLS C++ block ("IP")
whose interface uses `hls::stream<T>` ports run under Allo's **CPU dataflow
simulator** (`df.build(top, target="simulator")`), not only on the FPGA path.

It is a companion to [`IP_STREAM_INTEGRATION.md`](./IP_STREAM_INTEGRATION.md),
which describes how such an IP is integrated for `vitis_hls`/`vivado_hls`. Read
that one first if you want the vocabulary; this one is self-contained enough to
follow on its own. It assumes no knowledge of MLIR or compilers — every piece of
jargon is explained the first time it appears.

**Before this change**

```python
mod = df.build(top, target="simulator")
# NotImplementedError: IP 'vadd_stream' has hls::stream<T> arguments, which are
# only supported for the vitis_hls/vivado_hls targets...
```

**After**

```python
mod = df.build(top, target="simulator")
mod(A, B, C)          # runs on the CPU; C == A + B
```

---

## 1. Background: the three things you need to know

### 1.1 What Allo's dataflow simulator is

Allo lets you describe an accelerator as a **dataflow region**: a set of
concurrent kernels connected by FIFO channels.

```python
@df.region()
def top(A: int32[32], B: int32[32], C: int32[32]):
    sA: Stream[int32, 4]        # a FIFO channel of int32, depth 4

    @df.kernel(mapping=[1], args=[A])
    def feedA(a: int32[32]):
        for i in range(32):
            sA.put(a[i])        # blocking write
```

`target="simulator"` compiles this to run **on your CPU** so you can check the
numerics without waiting for hardware synthesis. It does that by turning each
`@df.kernel` into its own **OpenMP thread**. (OpenMP is the standard C/C++
threading runtime; "each kernel becomes an `omp.section` inside an
`omp.parallel`" simply means each kernel body gets its own thread that runs
concurrently with its peers.) The kernels really do run at the same time, so a
kernel that blocks waiting on a FIFO gets unblocked by a peer that is running
right now. This matters a lot below.

### 1.2 What a `Stream` becomes on the CPU

There is no hardware FIFO on a CPU, so the simulator builds one in memory: a
**ring buffer** (a fixed array plus two indices that wrap around).

For `Stream[int32, 4]` the simulator allocates this object (this is MLIR type
syntax; `memref<5xi32>` means "a 5-element array of 32-bit integers", and
`memref<i32>` means "a single 32-bit integer in memory"):

```
!allo.struct< memref<5xi32>, memref<i32>, memref<i32> >
//              data           head          tail
//            5 = depth + 1   read index   write index
```

* **`data`** — the storage. Its length is `depth + 1`, not `depth`. The extra
  slot is what makes the two states unambiguous: `head == tail` means *empty*,
  and `(tail + 1) % cap == head` means *full*. Without the spare slot those two
  conditions would be the same test.
* **`head`** — the index the *consumer* reads from. Only the consumer advances
  it.
* **`tail`** — the index the *producer* writes to. Only the producer advances
  it.

Both start at 0. Because each index has exactly one writer, no lock is needed;
what *is* needed is careful ordering, which section 3 covers in detail.

Every `sA.put(x)` / `sA.get()` in your Allo code is rewritten into a fixed
sequence of loads, stores and memory fences over this ring buffer. That rewrite
lives in `allo/backend/simulator.py`, function `_process_function_streams`.

### 1.3 What an `IPModule` is

`allo.IPModule` wraps a hand-written C++ function so you can call it from an
Allo kernel:

```python
vadd_stream = allo.IPModule(
    top="vadd_stream", impl="vadd_stream.cpp",
    link_hls=False, input_idx=[0, 1], output_idx=[2],
)
```

Allo parses the C++ signature, and **compiles the source itself** — that last
part is the hinge this whole design turns on.

For CPU targets, Allo generates a small C++ **wrapper** around the IP,
`g++`-compiles wrapper + IP into a shared library (`.so`), and hands that `.so`
to the JIT (just-in-time compiler) that runs the Allo module. The JIT then calls
the wrapper like any other function. That machinery is
`IPModule.generate_mlir_c_wrapper` and `IPModule.compile_shared_lib` in
`allo/backend/ip.py`.

---

## 2. Why the CPU path could not already do this

On the **FPGA path** it just works, and it is worth being precise about why: the
IP's `A.read()` and Allo's `sA.get()` both compile down to *Vitis's own*
`hls::stream` primitive. Vitis owns the representation on both sides, so the two
halves meet in the middle automatically.

On the **CPU** there is no shared primitive:

| | Allo's kernels | the IP |
|---|---|---|
| written in | Python → MLIR | C++ |
| compiled by | LLVM JIT, in-process | `g++`, into a `.so` |
| a stream is | the ring buffer of §1.2 | whatever `hls_stream.h` says |

Two different FIFO implementations in two different compilation units. A
`put` from an Allo kernel would land in the ring buffer; a `read()` in the IP
would look in a Vitis `hls::stream` object that nobody ever fills. So stream IPs
were fenced off on the CPU with a clear `NotImplementedError` rather than left
to fail confusingly inside `g++`.

---

## 3. The idea: a one-sided stream shim

Allo compiles the IP from source. Therefore **Allo controls which
`hls_stream.h` the IP sees.**

So: ship a *shim* `hls::stream<T>` whose `read()` and `write()` **are** the ring
buffer handshake, put its directory first on the include path, and hand it
Allo's ring buffers. The IP's body is not modified in any way — `A.read()` in
the unmodified IP now performs exactly the operation an Allo kernel's
`sA.get()` performs, on exactly the same buffer.

```
  ┌───────────┐   put     ┌──────────────────────┐   read()   ┌──────────────┐
  │ feedA     │──────────►│  ring buffer (sA)    │◄───────────│ vadd_stream  │
  │ (Allo,    │           │  data / head / tail  │            │ (IP, g++,    │
  │  JIT'd)   │           │  in shared memory    │            │  shim header)│
  └───────────┘           └──────────────────────┘            └──────────────┘
     thread 1                  one buffer, no copies              thread 3
```

Four properties of this design are worth stating explicitly, because each one is
a decision that could have gone otherwise:

* **One-sided.** Only the IP side is adapted. Allo's kernels, its MLIR lowering,
  and the ring-buffer format are untouched — which means the shim cannot
  possibly regress the existing simulator.
* **Zero-copy.** The shim does not own a FIFO and does not marshal data between
  two FIFOs. It holds *pointers into Allo's buffer*. There is exactly one queue,
  so there is no "who has the real data" question and no extra latency.
* **Simulator-only.** The shim is used for `target="simulator"` and nothing
  else. The plain `llvm` target runs the kernels **sequentially**, one call
  after another; an IP that blocks waiting for a FIFO that a not-yet-run kernel
  will fill would hang forever. That path keeps its `NotImplementedError`. The
  same reasoning applies to Vitis `csim`.
* **The IP must be a concurrent process.** For the same reason, the IP has to be
  called from inside its own `@df.kernel`, so the simulator gives it its own
  thread. This is checked and reported (§6), not left to deadlock.

---

## 4. The protocol, line by line

This is the part that must be exactly right. A ring buffer shared by two threads
is a classic place for a data race that shows up as "works 999 times, corrupts
once". The shim is therefore a **literal translation** of what
`allo/backend/simulator.py` emits, not an independent reimplementation.

File: `allo/backend/ip_sim/allo_fifo.h`.

```c
template <typename T> struct AlloFifo {
  T *data;        // cap slots
  int32_t cap;    // depth + 1
  int32_t *head;  // read index  — only the consumer advances it
  int32_t *tail;  // write index — only the producer advances it
};
```

### 4.1 `put` (what the IP's `write()` calls)

Line numbers are `allo/backend/simulator.py` as of this change.

| step | shim (`allo_fifo_put`) | simulator.py | why |
|---|---|---|---|
| 1 | `__atomic_thread_fence(SEQ_CST)` | `openmp_d.FlushOp` — line 870 | Before reading `head` we must see the consumer's latest value, not one cached in this thread's registers/cache. |
| 2 | load `tail` | `memref_d.LoadOp` — line 871 | We are the only producer, so nobody else can move `tail`. |
| 3 | `next = (tail + 1) % cap` | `AddIOp` + `RemUIOp` — lines 875–882 | The slot we would publish. |
| 4 | `while (head == next) backoff();` | `scf.while`, condition at lines 915–920, re-fence each iteration at line 903 | `head == next` is the FULL test. The fence *inside* the loop is essential: without it the compiler is free to hoist the load of `head` out of the loop and spin forever on a stale value. |
| 5 | `data[tail] = value;` | `memref_d.StoreOp` — lines 989–994 | **Before** the publish. |
| 6 | `__atomic_store_n(tail, next, SEQ_CST)` | `omp.critical` + store, lines 996–999, later converted to `omp.atomic.write` by `convert_critical_write_to_atomic_write` (line 1783, called at line 1871) | This single store is what makes the element visible. Atomic so the consumer can never observe a half-written index. |
| 7 | `__atomic_thread_fence(SEQ_CST)` | `openmp_d.FlushOp` — line 1000 | Trailing fence, mirroring the emitted code. |

**The ordering rule, stated plainly:** step 5 must happen before step 6. The
consumer decides "there is an element" purely by looking at `tail`. If `tail`
became visible first, the consumer could read a slot that has not been written
yet — it would read stale garbage, and no test would reliably catch it.

### 4.2 `get` (what the IP's `read()` calls)

The exact mirror image:

| step | shim (`allo_fifo_get`) | simulator.py | why |
|---|---|---|---|
| 1 | fence | line 869 (shared prologue) | See the producer's latest `tail`. |
| 2 | load `head`, compute `next` | lines 885–895 | We are the only consumer. |
| 3 | `while (head == tail) backoff();` | condition at lines 1002–1007, fence at 903 | `head == tail` is the EMPTY test. |
| 4 | `value = data[head];` | `memref_d.LoadOp` — lines 1062–1065 | **Before** the publish. |
| 5 | `__atomic_store_n(head, next, SEQ_CST)` | `omp.critical` + store — lines 1088–1090 | Releases the slot. |

**The mirrored ordering rule:** step 4 must happen before step 5. The producer
decides "that slot is free" purely by looking at `head`. If `head` were
published first, the producer could overwrite the slot while we are still
reading it.

The MLIR `get` path emits no trailing `omp.flush`; the shim performs one anyway,
purely for symmetry with `put`. A sequentially-consistent store already implies
a full fence, so this is not an extra ordering constraint, just an explicit one.

### 4.3 Why sequential consistency, and why `usleep`

* **`__ATOMIC_SEQ_CST`** is the strongest and simplest ordering C++ offers: every
  thread observes all seq-cst operations in one consistent global order. It is
  what `omp.flush` + `omp.atomic.write` give on the MLIR side, so using it keeps
  the two sides literally equivalent. A weaker acquire/release pairing would be
  sufficient in theory and faster, but this is a *simulator* — matching the
  reference exactly is worth far more than the cycles.
* **`usleep(1)` in the spin loop** mirrors the `usleep(1)` the simulator injects
  (line 913). Its job is to stop a blocked kernel from burning a core at 100%
  and starving the very peer it is waiting for — which, with more kernels than
  cores, is the difference between "slow" and "hangs".
* **`#pragma omp taskyield`** (line 908 in the emitted code) is present in the
  shim but compiled only when `ALLO_IP_SIM_OPENMP=1`. It is a *scheduling hint*
  with no bearing on correctness. It is off by default because `g++` would link
  the IP against GNU's `libgomp` while the JIT'd Allo code uses LLVM's `libomp`,
  and hosting two OpenMP runtimes in one process is unsafe. `usleep(1)` does the
  actual yielding.

### 4.4 The shim header itself

`allo/backend/ip_sim/hls_stream.h` is a thin `namespace hls` wrapper over those
operations:

```c++
template <typename T, int DEPTH = 0> class stream {
public:
  explicit stream(AlloFifo<T> *fifo) : fifo_(*fifo), owned_(nullptr) {}  // bound to Allo's buffer
  T    read()                  { return allo_fifo_get(fifo_); }
  void write(const T &value)   { allo_fifo_put(fifo_, value); }
  bool read_nb(T &value)       { return allo_fifo_try_get(fifo_, &value); }
  bool write_nb(const T &value){ return allo_fifo_try_put(fifo_, value); }
  bool empty() const;  bool full() const;  std::size_t size() const;
  // plus the free operators `is >> value` and `os << value`
};
```

It covers the operations an IP performs on a *port*; it is not a full
reimplementation of Vitis's class. Two details:

* It is non-copyable, matching Vitis — and also because copying a FIFO *view*
  would silently duplicate a queue's state.
* A stream the IP declares **internally** (one Allo never passed in, so no Allo
  buffer exists) gets its own heap ring buffer of
  `ALLO_SIM_LOCAL_STREAM_DEPTH` slots, so such an IP still compiles and runs.

---

## 5. The ABI: how Allo's FIFO reaches C++

"ABI" (application binary interface) is just: what the arguments physically look
like when one compiled function calls another. This section is what lets the
generated wrapper be written mechanically instead of guessed at.

### 5.1 What the JIT actually passes

After lowering, an Allo stream argument is a pointer to the
`!allo.struct<memref<5xi32>, memref<i32>, memref<i32>>` from §1.2. The struct is
**not** passed as a struct pointer: MLIR unpacks each `memref` into plain
scalars, a so-called *memref descriptor*.

* `memref<Nxi32>` → `(allocated_ptr, aligned_ptr, offset, size, stride)`, with
  element `i` living at `aligned_ptr[offset + i]`.
* `memref<i32>` (rank 0) → `(allocated_ptr, aligned_ptr, offset)`, value at
  `aligned_ptr[offset]`.

Note `cap` is simply the data memref's `size` field.

### 5.2 Why the wrapper does not hand-match that layout

Rather than reconstructing five scalars per memref by hand — fragile, and it
would silently rot if MLIR's layout ever changed — the wrapper uses the **same
mechanism the existing memref-IP path already uses**: it declares each argument
as an *unranked* memref (`memref<*xi32>`) on the MLIR side, which crosses into
C++ as a `(int64_t rank, void *descriptor)` pair, and then hands that pair to
`DynamicMemRefType` from `mlir/ExecutionEngine/CRunnerUtils.h`. That class is
MLIR's own descriptor reader; it knows the layout so we don't have to.

Each stream therefore becomes **three** unranked-memref arguments (data, head,
tail). A three-stream IP has nine MLIR operands, as seen below.

### 5.3 Before and after, in MLIR

Before the rewrite — the IP is declared with Allo stream types, which no CPU ABI
can express:

```mlir
func.func private @vadd_stream(!allo.stream<i32, 4>, !allo.stream<i32, 4>, !allo.stream<i32, 4>)

func.func @ip_wrap_0(%arg0: !allo.stream<i32, 4>, ...) attributes {df.kernel, ...} {
  call @vadd_stream(%arg2, %arg0, %arg1) {stream_dirs = "iio"} : (...) -> ()
```

After — the declaration and the call site now speak the wrapper's ABI:

```mlir
func.func private @pyvadd_stream_1785273214135020487(
    memref<*xi32>, memref<*xi32>, memref<*xi32>,   // stream 0: data, head, tail
    memref<*xi32>, memref<*xi32>, memref<*xi32>,   // stream 1
    memref<*xi32>, memref<*xi32>, memref<*xi32>)   // stream 2

func.func @ip_wrap_0(%arg0: memref<!allo.struct<memref<5xi32>, memref<i32>, memref<i32>>>, ...) {
  call @pyvadd_stream_1785273214135020487(%cast, %cast_0, ..., %cast_7) : (...) -> ()
```

Note the kernel's arguments are now FIFO structs — the simulator retyped them —
and each `%cast` is one field of one struct, cast to an unranked memref.

### 5.4 The generated wrapper

For each stream port the wrapper emits (abridged, three ports in the real file):

```c++
extern "C" __attribute__((visibility("default")))
void pyvadd_stream_1785273214135020487(
    int64_t s0_data_rank, void *s0_data_ptr, /* head, tail, then s1, s2 ... */) {
  UnrankedMemRefType<int32_t> s0_data_u = {s0_data_rank, s0_data_ptr};
  DynamicMemRefType<int32_t>  s0_data(s0_data_u);
  /* ... same for s0_head, s0_tail ... */
  assert(s0_data.rank == 1 && "Allo FIFO storage must be a 1-D memref");
  assert(s0_data.strides[0] == 1 && "Allo FIFO storage must be contiguous");

  AlloFifo<int32_t> s0_fifo;
  s0_fifo.data = s0_data.data + s0_data.offset;
  s0_fifo.cap  = (int32_t)s0_data.sizes[0];      // = depth + 1
  s0_fifo.head = s0_head.data + s0_head.offset;
  s0_fifo.tail = s0_tail.data + s0_tail.offset;
  hls::stream<int32_t> s0(&s0_fifo);
  /* ... */
  vadd_stream(s0, s1, s2);
}
```

Array and scalar ports are emitted exactly as `generate_mlir_c_wrapper` emits
them, so an IP may freely mix array, scalar and stream ports.

---

## 6. The changes, file by file

### Edit 1 — the shim headers — `allo/backend/ip_sim/` *(new)*

`allo_fifo.h` (the protocol of §4) and `hls_stream.h` (the `namespace hls`
wrapper of §4.4). New directory, so nothing existing can be affected by it.

`IP_SIM_INCLUDE_DIR` in `allo/backend/ip.py` points at it, and
`compile_shared_lib` puts it **first** with `-I`, so the IP's
`#include <hls_stream.h>` resolves to the shim even when `link_hls=True` also
put Vitis's include directory on the list.

### Edit 2 — the wrapper generator — `allo/backend/ip.py`

* `generate_stream_sim_wrapper()` — emits the wrapper of §5.4. Named alongside
  the existing `generate_mlir_c_wrapper()` and deliberately built the same way
  (`UnrankedMemRefType` → `DynamicMemRefType`) so there is one descriptor-reading
  idiom in the codebase, not two.
* `stream_element_type()` / `split_template_args()` — pull `T` out of
  `hls::stream<T>` / `hls::stream<T, DEPTH>` as written in C++. The optional
  `DEPTH` argument is dropped: on the CPU the depth that exists is the one the
  Allo `Stream` declaration allocated.
* `stream_arg_indices` — positions of the stream ports, in order.
* `compile_shared_lib(stream_sim=False)` — the `stream_sim=True` flavour adds
  the shim include directory (first), `-Wno-unknown-pragmas` (the IP keeps its
  `#pragma HLS ...` lines, which are hardware directives that `g++` neither
  knows nor needs), and, under `ALLO_IP_SIM_OPENMP=1`, `-fopenmp`.
* `_reject_stream_on_cpu()` — message updated: the simulator is now supported;
  the plain `llvm` target and `csim` still are not, and the message now says
  *why* (they call the IP once, sequentially).

### Edit 3 — symbol visibility — `allo/backend/ip.py`

`compile_shared_lib` now compiles with **`-fvisibility=hidden`**, and both
wrapper generators mark the entry point
`__attribute__((visibility("default")))` (the `_EXPORT_ATTR` constant).

This one is worth spelling out, because it was a real, silent deadlock found
during verification, not a precaution.

The wrapper entry point carries a per-instance hash (`pyvadd_stream_<hash>`), so
those never collide. But the **IP's own top function keeps the name the user
wrote** — `vadd_stream` — and it was exported from every `.so` built from it:

```
$ nm -D --defined-only libpyvadd_stream_<hashA>.so
0000000000001283 T pyvadd_stream_<hashA>
000000000000121c T vadd_stream          ← global
```

When two IPModules built from *different* sources but sharing a top name end up
in one process (two tests in one pytest run — exactly what
`test_stream_ip_sim.py` does), the dynamic linker resolves a global symbol to
the definition it loaded **first**. So the second wrapper called the *first*
IP's body. In the test suite that meant an IP compiled for 256 elements ran a
body that consumed 32 and returned; the feeders then blocked forever on a full
FIFO and the whole run hung, with no error message anywhere.

Hiding everything but the entry point makes each wrapper's call bind inside its
own `.so`:

```
$ nm -D --defined-only libpyvadd_stream_<hashA>.so
0000000000001203 T pyvadd_stream_<hashA>
```

This is applied to both wrapper flavours because the hazard is identical for
memref IPs; it changes no behaviour other than removing the interposition.

### Edit 4 — let the simulator through the CPU fence — `allo/passes.py`

`call_ext_libs_in_ptr` rewrites each IP call into a call through unranked-memref
pointers. A stream port has no such representation, so this function used to
reject any stream IP outright.

It now takes `allow_stream_ip=False`. The simulator passes `True`, which makes
this pass **skip stream IPs entirely** (they are excluded from `lib_map` and
their declaration is left standing) and leave them to `backend/simulator.py`,
which runs later and knows what ring buffer each stream became. The plain `llvm`
target still passes `False` and still raises — with a message that now explains
the sequential-execution reason.

One consequential detail: because a stream IP's *declaration* is now
deliberately left in the module, the loop below it can encounter an external
function, which has no body. The guard became
`isinstance(op, func_d.FuncOp) and not op.is_external`.

### Edit 5 — declare the wrapper — `allo/backend/simulator.py`

`declare_stream_ip_wrappers()` replaces
`func.func private @vadd_stream(!allo.stream<...>, ...)` with
`func.func private @pyvadd_stream_<hash>(memref<*xi32>, ...)` — the §5.3
"after". `_plan_stream_ip_wrapper()` computes both the per-IP-argument plan
(`stream` / `memref` / `scalar`) and the flattened MLIR operand list.

Only the *declaration* changes here. The call sites cannot be rewritten yet: at
this point the streams are still `!allo.stream`, and the ring buffers do not
exist.

### Edit 6 — rewrite the call sites — `allo/backend/simulator.py`

`_lower_stream_ip_calls()` runs from inside `_process_function_streams`, at the
exact point where each kernel's stream arguments have just been retyped to FIFO
structs and `arg_stream_table` (which block argument corresponds to which
stream) is known. Placement is the whole trick: a moment earlier there is no
ring buffer to point at, a moment later the mapping is gone.

For each operand of the call it emits, per stream: an `affine.load` of the FIFO
struct, three `allo.struct_get`s (data / head / tail), and a `memref.cast` of
each to an unranked memref — the same cast `call_ext_libs_in_ptr` uses for array
arguments. Array operands are cast the same way; scalars pass through untouched.

It also enforces the constraints, with messages that say what to do:

* the call must be inside a `@df.kernel` (checked via the `df.kernel` attribute)
  — otherwise the IP does not get its own thread and would deadlock;
* each stream operand must be a stream passed into that kernel;
* the stream's element type must match the IP's declared element type — the IP
  writes Allo's buffer *in place*, so a mismatch is not convertible, it is a
  type error (raised as `TypeError`);
* the stream's elements must be scalars.

`_check_no_unlowered_stream_ip_calls()` then sweeps the module for any call to a
stream IP the rewrite did not reach, and fails loudly. Without it, an unhandled
shape (say, a call in a helper function) would reach the LLVM lowering as a call
to a symbol that no longer exists — a confusing crash far from the cause.

### Edit 7 — wire it into the build — `allo/backend/simulator.py`

* `build_dataflow_simulator(module, top_func_name, ext_libs=None)` — takes the
  IP list, builds the `stream_ips` map and calls Edit 5, then threads the plans
  through `_process_function_streams` so Edit 6 can run at the right moment.
* `_process_function_streams` also learns to **skip** stream IPs when collecting
  "PE calls": an IP is an opaque external function, not a processing element, and
  it has no body to walk into.
* `LLVMOMPModule.__init__` passes `allow_stream_ip=True` to `call_ext_libs_in_ptr`,
  passes `ext_libs` to `build_dataflow_simulator`, and compiles each stream IP
  with `compile_shared_lib(stream_sim=True)` (others unchanged), adding the
  resulting `.so` to `shared_libs` for the JIT.

---

## 7. How to use it

**The IP** (`vadd_stream.cpp`) — ordinary HLS, unmodified:

```c++
#include <hls_stream.h>
#include <stdint.h>

extern "C" {
void vadd_stream(hls::stream<int32_t> &A, hls::stream<int32_t> &B,
                 hls::stream<int32_t> &C) {
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II = 1
    int32_t a = A.read();
    int32_t b = B.read();
    C.write(a + b);
  }
}
}
```

**The Allo program** — the IP gets its own kernel, between feeders and a drain:

```python
import allo, numpy as np
import allo.dataflow as df
from allo.ir.types import int32, Stream

N = 32
vadd_stream = allo.IPModule(
    top="vadd_stream", impl="vadd_stream.cpp",
    link_hls=False,                 # the shim replaces Vitis's header,
    input_idx=[0, 1], output_idx=[2],   # so vitis_hls need not be installed
)

@df.region()
def top(A: int32[N], B: int32[N], C: int32[N]):
    sA: Stream[int32, 4]
    sB: Stream[int32, 4]
    sC: Stream[int32, 4]

    @df.kernel(mapping=[1], args=[A])
    def feedA(a: int32[N]):
        for i in range(N):
            sA.put(a[i])

    @df.kernel(mapping=[1], args=[B])
    def feedB(b: int32[N]):
        for i in range(N):
            sB.put(b[i])

    @df.kernel(mapping=[1])          # its own kernel => its own thread
    def ip_wrap():
        vadd_stream(sA, sB, sC)

    @df.kernel(mapping=[1], args=[C])
    def drain(c: int32[N]):
        for i in range(N):
            c[i] = sC.get()

mod = df.build(top, target="simulator")
a = np.random.randint(-1000, 1000, N).astype(np.int32)
b = np.random.randint(-1000, 1000, N).astype(np.int32)
c = np.zeros(N, dtype=np.int32)
mod(a, b, c)
np.testing.assert_array_equal(c, a + b)
```

Run it with at least as many OpenMP threads as there are kernels:

```bash
export OMP_NUM_THREADS=8
python your_script.py
```

**If a kernel count exceeds `OMP_NUM_THREADS`,** two kernels share a thread and
run one after the other — and a blocking IP will hang. This is a property of the
simulator, not of the shim, but stream IPs make it much easier to hit.

### Things that are (deliberately) rejected

| you write | you get |
|---|---|
| `df.build(top, target="llvm")` with a stream IP | `NotImplementedError`: sequential execution can never satisfy a blocking read |
| the IP called outside a `@df.kernel` | `NotImplementedError`: "Wrap the call in its own @df.kernel" |
| `hls::stream<int8_t>` port fed by `Stream[int32, 4]` | `TypeError`: "The element types must match" |
| a stream of arrays | `NotImplementedError`: the shim supports streams of scalars |
| `vadd_stream.generate_mlir_c_wrapper()` (memref path) | `NotImplementedError`, unchanged |

---

## 8. Verification

Environment used (`zhang-21`):

```bash
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8
export PATH="$LLVM_BUILD_DIR/bin:$PATH"
export OMP_NUM_THREADS=8
```

### New tests — `tests/ip_integration/test_stream_ip_sim.py`

```bash
conda run -n allo python -m pytest tests/ip_integration/test_stream_ip_sim.py -q
# 5 passed in 3.56s
```

| test | what it pins down |
|---|---|
| `test_stream_ip_sim` | the end-to-end numeric result — this is the case that raised `NotImplementedError` before |
| `test_stream_ip_sim_repeated_runs` | a run leaves every FIFO with `head == tail`, so the module is reusable |
| `test_stream_ip_sim_backpressure` | 256 elements through depth-2 FIFOs: both spin paths hit thousands of times |
| `test_stream_ip_sim_element_type_mismatch` | the type check fires |
| `test_stream_ip_sim_wrapper_shape` | the generated wrapper includes the shim before the IP and binds all three fields per port |

### Concurrency soak

Deliberately tiny FIFOs and long runs, each configuration in its own process:

```bash
# N=2048 depth=1 x20 iterations   -> SOAK OK
# N=4096 depth=2 x10 iterations   -> SOAK OK
# N=512  depth=8 x40 iterations   -> SOAK OK
```

`depth=1` (a two-slot ring) is the most contended case possible: nearly every
`put` blocks on FULL and nearly every `get` blocks on EMPTY. All values matched
`a + b` exactly on every iteration.

This soak is what surfaced the symbol-interposition deadlock of Edit 3 — it
reproduced only when several IP `.so`s coexisted in one process.

### Regressions

```bash
conda run -n allo python -m pytest tests/ip_integration/test_external.py \
                                   tests/ip_integration/test_stream_ip.py -q
# 10 passed, 2 skipped in 93.59s      (2 skipped: vitis_hls not on PATH)

conda run -n allo python tests/dataflow/test_df_unit.py
# Dataflow Simulator Passed! / Dataflow Simulator Passed! /
# Dataflow Simulator (Arithmetic) Passed!

conda run -n allo python tests/dataflow/test_region_stateful.py
# exit 0
```

`test_stream_ip.py`'s rejection test was rewritten (it previously asserted the
simulator refuses stream IPs, which is precisely what changed) into
`test_stream_ip_sequential_cpu_paths_rejected`, which asserts the *remaining*
fences — `generate_mlir_c_wrapper`, `generate_nanobind_wrapper`, and
`call_ext_libs_in_ptr` with `allow_stream_ip=False` — still raise.

The FPGA codegen tests are unchanged and still pass; the csynth tests skip
because `vitis_hls` is not on this shell's PATH.

### Lint

`black` and `pylint --rcfile=./scripts/lint/pylintrc` report nothing on the new
or changed code. (`allo/backend/simulator.py` has pre-existing `black` and
`pylint` findings elsewhere in the file, all outside the changed hunks; they were
left alone rather than reformatted into this diff.)

---

## 9. Limitations and where to look next

* **Scalar element types only.** `_c_type_to_mlir` maps the plain C scalar types
  Allo already knows (`allo/utils.py: c2allo_type`). An HLS type such as
  `ap_int<8>` has no CPU representation here and is refused.
* **Streams of arrays are not supported** (`stream_type.rank != 1` is rejected).
  Allo itself can build such FIFOs; the shim's element type would have to become
  an array view.
* **One kernel per IP call.** The IP must sit in its own `@df.kernel`. Calling
  it directly in the `@df.region` body remains unsupported, as on the FPGA path.
* **Depth comes from Allo.** A `hls::stream<T, DEPTH>` port's `DEPTH` is ignored;
  the depth that exists on the CPU is the one the `Stream[...]` declaration
  allocated. On the FPGA path Vitis would honour the port's depth, so a design
  that depends on a deeper IP-side buffer can behave differently between the two
  targets.
* **`ALLO_IP_SIM_OPENMP=1`** compiles the `taskyield` hint into the shim. It is
  off by default (two OpenMP runtimes in one process); turn it on only when
  investigating scheduling behaviour.

### File map

| file | role |
|---|---|
| `allo/backend/ip_sim/allo_fifo.h` | the ring-buffer protocol in C (§4) |
| `allo/backend/ip_sim/hls_stream.h` | the shim `hls::stream<T>` (§4.4) |
| `allo/backend/ip.py` | wrapper generation, include path, `-fvisibility=hidden` (Edits 2–3) |
| `allo/passes.py` | `call_ext_libs_in_ptr(..., allow_stream_ip)` (Edit 4) |
| `allo/backend/simulator.py` | declaration swap, call-site rewrite, build wiring (Edits 5–7) |
| `tests/ip_integration/test_stream_ip_sim.py` | the simulator-path tests (§8) |
| `docs/IP_STREAM_INTEGRATION.md` | the FPGA-path counterpart |
