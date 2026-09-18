# Allo shortcomings — surfaced while building L2 TPU (FlashAttention)

Notes from the Track B step 2 effort to add VPU + hardware loop + on-device
transpose to `levels/L2/tpu.py`. Each item is a concrete obstacle that
required either a workaround in user code or a patch to the Allo library.

Priority annotations below were folded in from the former root `STATE.md`
(now removed; project state is judged from git/GitHub, and the living
fork-vs-upstream feature map is the pinned fork issue
https://github.com/sunwookim028/allo/issues/13). Related feature-gap tracking
lives as fork issues and is
not restated here: combinational wires (fork issue #9), HLS dependence pragma
(fork issue #10), shared mutable memory across kernels (fork issue #11; relates
to items 1-2 below), streams as top-level inputs (fork issue #12), and the
nested sub-region Stream compile-time-constant shape constraint (fork issue #4;
relates to item 3).

## 1. Region-scope `@ Stateful` lowering is incomplete on `main`

- Declaring `int32[N] @ Stateful = 0` at `@df.region` body scope (so the
  buffer is shared across every `@df.kernel` in the region) is the
  natural way to express a Gemmini-style decoder + driver split with
  shared spad / acc / imem.
- On `allo/main`, this either crashes with
  `AttributeError: 'ASTContext' object has no attribute 'global_op_cache'`
  or trips an MLIR `Assertion 'value' failed` (null Value) when a kernel
  reads-and-writes a region-scope Stateful inside a loop or branch.
- Update (2026-07-15): the `AttributeError: 'ASTContext' object has no
  attribute 'global_op_cache'` copy-crash landed on `main` via upstream
  PR #577 (global_op_cache copy fix). The broader region-scope stateful
  propagation block (`stateful_var_map` / `stateful_counter` through
  `ASTContext.copy()`, per-function `global_op_cache` reset, anchoring
  `memref.get_global` at each function's entry block) remains fork-local
  on `main`; it did not need its own branch (the former
  `feature/region-scope-stateful` branch, commit `5c4d1b5`, is deleted).
- Net effect: the crash fix is upstream; the sharing-scratchpad/acc-across-
  decoder-and-driver-kernels feature itself is still fork-local, tracked in
  fork issue #7.

## 2. `@ Stateful` cannot be declared inside `@df.kernel` bodies

- Putting `pc: int32[1] @ Stateful = 0` inside a `@df.kernel` triggers
  `AttributeError: 'ASTContext' object has no attribute 'global_op_cache'`.
- Forces all PC / loop-counter / preload-latch state to be hoisted to
  region scope, even when conceptually private to a single kernel.
- For an L2 decoder this means 8 single-element `int32[1] @ Stateful`
  arrays at region scope (`pc`, `halted`, `iter_count`, `iter_idx`,
  `loop_start_pc`, `iter_stride_a/b/d`) just to get persistence.
- **Priority: High** — currently forces 8 single-element region-scope arrays
  in the L2 decoder.

## 3. Simulator drops nested-call stream lowering

- `_process_function_streams` in `allo/backend/simulator.py` only
  scans `func.body.blocks[0].operations` for `func.call` ops. If a
  PE call (e.g. `mxu_fp32`) is nested inside an `affine.for` /
  `affine.if` (the natural shape of a decoder + driver split), the
  callee is never recursively processed. Its `allo.stream_put / get`
  ops survive into LLVM lowering, where `convert-func-to-llvm` fails
  with:

      cannot be converted to LLVM IR: missing
      `LLVMTranslationDialectInterface` registration for dialect for op:
      func.func

- The fix is mechanical (deep-scan `func_d.CallOp` via
  `recursive_collect_ops` in addition to the top-level scan) but it has
  to be patched into the library — there is no user-side workaround
  short of inlining the sub-region.
- Symptom is opaque: error points at the *callee* `func.func`, not at
  the top-level for-loop that hides the call.

## 4. `math.exp` / `math.log` are not recognized by the AST builder

- Inside `@df.kernel` bodies, `math.exp(x)` raises `KeyError: 'exp'`.
- Must use `allo.exp(x)` (and friends) instead.
- Not documented as a constraint; the failure mode (KeyError on a
  Python-builtin-ish name) does not point at the workaround.
- **Priority: Low** — `allo.exp` works.

## 5. Variable shadowing between region params and kernel-local names

- Declaring a local `d_addr: int32 = cmd[3]` inside `@df.kernel def
  compute_driver` raises `AssertionError: Invalid assignment to
  d_addr, type mismatch` because the enclosing `@df.region def
  tpu(..., d_addr: int32[1], ...)` parameter leaks into the kernel
  scope. The compiler treats the local `int32` write as an attempted
  rebinding of the region parameter (a `int32[1]`).
- Workaround: rename every kernel-local that happens to share a name
  with a region parameter (`d_addr → cmd_d`, etc.).
- The error message names the variable but not the shadowing, so this
  takes a while to diagnose.
- **Priority: Medium** — silent/misdirected error.

## 6. Local `int32` decls inside `elif` branches don't dominate uses

- Pattern that fails: declaring a fresh local inside an `elif` branch
  and then referencing it in another branch's calc.
- For example, `eff_d: int32 = rs1_lo + d_off` inside
  `elif funct7 == FUNCT7_PRELOAD:` and `new_idx: int32 = iter_idx[0] + 1`
  inside `elif funct7 == FUNCT7_LOOP_END:` produced MLIR that didn't
  verify (cryptic dominance / null-Value errors during lowering).
- Workaround: hoist every such temporary out of the if/elif chain so it
  is declared in a block that dominates all uses. This bloats the
  decoder body.
- **Priority: Medium** — bloats the decoder.

## 7. No bitwise `&` operator support in Allo expression DSL

- For decoding instruction flag bits we wanted `(iflags & 2) >> 1`.
- Allo rejects `&` inside `@df.kernel` bodies; we end up emulating bit
  extraction with arithmetic:

      a_stride_on: int32 = (iflags // 2) - ((iflags // 4) * 2)
      b_stride_on: int32 = (iflags // 4) - ((iflags // 8) * 2)
      d_stride_on: int32 = (iflags // 8) - ((iflags // 16) * 2)

- Works, but verbose and obscures intent (the reader has to recognize
  the division-pair as a single-bit extraction).
- **Priority: Low** — arithmetic emulation works.

## 8. Single-MXU-call rule (Allo region instantiation)

- `mxu(...)` cannot appear in two different `if`/`elif` branches even
  if they are mutually exclusive at runtime. Allo instantiates
  sub-regions at build time regardless of conditions, so two branch
  callsites become two independent instances and silently break.
- Each level keeps `mxu(...)` in exactly one combined branch (`OP_MM
  | OP_MMT` for L1; `COMPUTE_PRELOADED | COMPUTE_ACCUMULATED` for L2).
- This forces unnatural code structure — the natural reading is "if
  preloaded, do mxu with these args; if accumulated, do mxu and add" —
  but the compiler needs us to flatten them.
- **Priority: Medium** — forces unnatural code structure.

## 9. Sim cache invalidation misses imported helpers

- `.cache/llvm_sim/` is keyed on the level's `tpu.py` and top-level
  `tpu_config.py` only. Editing an imported helper (e.g.
  `levels/_common/mxu_fp32.py` or `kernels/*.py`) does not invalidate
  the cache.
- Stale cache will hide compilation failures: the build appears to
  succeed (cache hit), runtime then misbehaves.
- Has bitten this project at least twice during the session — commit
  messages claim "validated" because the cached object loaded fine,
  but a clean rebuild fails.
- **Priority: High** (allo-tpu side) — repeated stale-cache "validated".

## 10. Error messages point at lowered MLIR, not source

- Most failure modes surface as MLIR / LLVM errors at line numbers in a
  generated module that the user never sees. Examples:
  - `loc("-":1892:3): error: cannot be converted to LLVM IR ...`
  - `Assertion 'value' failed`
  - `Failure while creating the ExecutionEngine`
- Mapping these back to the offending Python construct requires
  dumping `s.module` and counting lines — there is no source-position
  attribution back to the original `tpu.py`.
- The MLIR Context cannot be re-instantiated in the same Python
  process without crashing
  (`LLVM ERROR: Option 'fast' already exists!`),
  so debugging via "build twice and diff" doesn't work.
- **Priority: Low.**

---

# Allo shortcomings — surfaced while building an instruction-programmable TPU (2026-09)

A second pass, from building `examples/accelerator/tinytpu_vitis/` on `main`:
an int8 instruction-programmable tiled-GEMM accelerator taken through the
Vitis dataflow path to **RTL co-simulation**, and compared against a data-type-
and mesh-matched Gemmini (`COMPARISON.md`).

The findings below were previously scattered across `examples/accelerator/*/`
markdown and **none of them had reached `notes/`**, which is why they are
consolidated here. They are ordered by what they would cost an Allo user, not
by when they were found.

## 11. The dataflow simulator deadlocks when processes outnumber OMP threads — **FIXED**

The simulator appears to give each `df.kernel` instance an OMP thread and to
block that thread on an empty/full stream. With fewer threads than processes, a
blocked process can hold a thread its own producer needed, and the region
wedges **silently** -- no message, no indication of which process is blocked on
which channel.

Measured on a 22-process region (`T*T + 6`) at 16x16x16 with stream depth 16:

| `OMP_NUM_THREADS` | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| | hang | hang | pass | pass |

The threshold is exactly the process count. With 32 threads the design runs at
**depth 4**, and a shape that had never passed at *any* depth passed at depth 8.

- Deep FIFOs mask it, by letting producers finish before anyone must block, so
  the symptom presents as "required stream depth grows with the program" -- a
  plausible-looking *design* problem. This cost multiple sessions.
- `CLAUDE.md` currently advises `OMP_NUM_THREADS=8`, which is fine for the small
  regions in `tests/dataflow` but is **not a safe default**. The rule is
  `OMP_NUM_THREADS >= number of kernel instances`.
- `examples/accelerator/tinytpu_vitis/kpn_model.py` is a ~140-line model of a
  channel graph that reports which processes are blocked on which channels and
  at what occupancy. It found this in one run. That report is cheap.
- **FIXED 2026-09-17** in `allo/backend/simulator.py`
  `_inject_omp_parallel_sections`: the OpenMP team is now sized to the section
  count (`num_threads = len(pe_call_define_ops)`) instead of defaulting to the
  core count. Our 22-process design now runs every shape exactly at
  `OMP_NUM_THREADS=8`, the value that used to hang; the golden tests and the
  upstream dataflow suite still pass.
- Credit: independently found and fixed by `chhzh123` on the SPMW branch
  (`a03edb85`, 2026-09-05) from the other direction -- "56 at 8x8 FEATHER on a
  48-core host". Two unrelated projects hitting the same wall is the argument
  for it being upstreamed rather than carried.
- **Diagnosis, tier 0 -- DONE 2026-09-18** (`7bc6d413`). `LLVMOMPModule.__call__`
  now arms a watchdog around the blocking `execution_engine.invoke`, default ON
  at 600 s (`ALLO_SIM_TIMEOUT=<sec>`, `=0` to silence). On a real deadlock it
  prints the top function, the kernel-instance count, `OMP_NUM_THREADS`, the
  pid, the likely causes, and an explicit note that the process is NOT being
  killed and Ctrl-C will not work (the simulator is inside a blocking C call),
  with the `kill -9` line. It repeats with geometric backoff.
  - The watchdog is **one reused thread parked on a `Condition`**, not a
    per-call `threading.Timer`: the Timer version was measured at **+170 us per
    call**, a fifth of a small region's runtime and inside the window
    `tests/dataflow/mesh_perf.py` measures throughput over -- a watchdog that
    perturbs what it watches. The reused thread costs **+12.6 us**.
  - It is advisory: nothing is killed, nothing is raised, and a healthy run that
    trips the timeout still returns a correct result. There is a test asserting
    exactly that, which is what makes defaulting it ON defensible.
  - `tests/dataflow/test_sim_timeout.py`, 4 tests, 15 s, cannot hang the suite.
- **Still open: tier 1, the per-channel report.** The watchdog says *that* the
  region is stuck, not *who* is stuck on *which* channel. The natural hook now
  exists: `fc08bb6b` collapsed three byte-identical spin-wait sites into one
  `_build_spin_wait_loop`, so instrumenting the generated spin -- beside the
  `usleep(1)` it already contains -- is a one-line change rather than three.
  The remaining cost is a runtime shared library to receive the callback, and
  its risk is linkage (see the `LLVM_BUILD_DIR` / GLIBC pitfall).
  `examples/accelerator/tinytpu_vitis/kpn_model.py` shows the report format;
  what does not transfer is its mechanism -- it is a single-threaded
  cooperative scheduler that can observe "a full sweep advanced nobody", and
  the real simulator's processes are opaque JIT'd code on OpenMP threads.

## 12. Bit-slices lower to *signed* `ap_int<N>`, silently, and the simulator disagrees

`w[54:61]` on an unsigned value emits:

```cpp
ap_int<7> v268;  v268 = w02(60, 54);
int32_t nr = v268;                     // 64 -> 0b1000000 -> -64
```

so any field whose top bit is set reads back **negative**. A loop bounded by it
runs zero times.

- Cost: an ISA row-count field of 64 silently loaded nothing and the design
  produced zeros -- **251 of 256 outputs wrong**. It failed exactly at the
  sign-bit boundary (63 fine, 64 not).
- **The dataflow simulator treats the slice as unsigned and passed the same
  program.** This is a genuine simulator/RTL divergence, and it is the reason
  this bug survived every functional check that had been passing.
- Workaround: an N-bit field safely carries `0 .. 2^(N-1) - 1`; budget one
  spare bit per field and assert it at the assembler.
- **Priority: High.** Either lower unsigned slices to `ap_uint<N>`, or make the
  simulator model the sign so the two agree.

## 13. ~~No program-controlled DMA~~ -- **largely RETRACTED**; the gap is convenience, not capability

`wrap_io=True` copies each argument into a local buffer before the region runs,
sized to the **declared** array rather than to what the program touches.
`wrap_io=False` drops the copy but every access then pays bus latency.

Measured, same design, one build each (cycles, cosim):

| config | marginal | fixed | 4x4x4 | 16x16x16 |
|---|---|---|---|---|
| `wrap_io=True` | 18.1 cyc/instr | 1102 | 2.12x | 1.94x |
| `wrap_io=False` | **39.8** | **481** | **1.21x** | 2.25x |
| Gemmini | **10.8** | **483** | 1.00x | 1.00x |

### The retraction (2026-09-18)

The conclusion drawn from that table -- "Gemmini has a third option and Allo
does not expose it" -- **was wrong**, and the table itself is confounded.

`wrap_io=False` was measured with a **strided** access pattern and no local
buffer. Probing which patterns Vitis actually bursts, with `wrap_io=False` and
flat arguments:

```
lA[(off + r) * MAXDIM + e], e over meta_for(T)      <- what our dma_ld does
  [HLS 214-115] Multiple burst reads of length 4 and bit width 8

for i in range(n): buf[i] = lA[off + i]   (n RUNTIME)
  [HLS 214-115] Multiple burst reads of VARIABLE LENGTH and bit width 8
```

**A contiguous copy with a runtime length from a runtime offset infers a real
variable-length AXI burst, at II=1.** That *is* `mvin`, and Allo expresses it
today. The 39.8 cycles/instruction in the table above is the cost of 4-byte
bursts, not of `m_axi`; it condemns the access pattern, not the configuration.

What remains true, and what is actually left:

- `wrap_io=True` genuinely is not a DMA. `wrap_data_movement`
  (`allo/ir/transform.py:450`) takes its extent from
  `shape = MemRefType(arg.type).shape` -- the **static type** -- with no offset
  and no length anywhere in the generated function. It is a whole-argument
  hoist run once at region entry, and every declared word is a startup cycle
  whether the program touches it or not.
- So the two options are "hoist everything" or "issue your own bursts", and the
  second one works. What Allo lacks is only the *convenience* of an
  `allo.dma(buf, ptr, offset, length)` intrinsic that makes the burst idiom
  obvious rather than something you discover by reading HLS burst messages.
- **Priority: Medium** (an ergonomics and documentation item), down from High.
  The performance work it was blocking is ours, not Allo's.

### The general lesson

The measurement that produced the wrong conclusion was real and repeatable; it
was the *attribution* that was wrong. Two configurations were compared while a
third variable -- the access pattern -- differed between them, and the result
was charged to the configuration. Before charging a cost to the toolchain,
check that the thing being measured is the thing named.

## 14. `wrap_io=False` rejects multi-dimensional arguments to nested kernels

> Top-level multi-dimensional arrays are linearized to 1D pointers ... which
> cannot be passed to nested functions expecting multi-dimensional arrays

- The message is good and names the fix. Flat `int8[M*N]` arguments with manual
  `row * stride + col` addressing work, and are arguably the honest shape for
  DRAM anyway.
- **Priority: Low** (documentation), but it interacts with #13: taking the
  low-fixed-cost option forces flat arguments.

## 15. Vitis `csim` executes dataflow processes in declaration order

A consumer declared before its producer reads an empty stream:

```
ERROR [HLS SIM]: an hls::stream is read while empty
```

- Kernel declaration order in a `@df.region()` is therefore **load-bearing** for
  `csim` (not for RTL, where processes are concurrent). Nothing documents this.
- Cost here: `dma_st` was declared third and consumed what `accu`, declared
  last, produced. Reordering fixed it with no hardware change.
- **Priority: Medium.** A note in the dataflow docs would be enough.

## 16. `cosim` is not wired into `df.build`

`df.build(target="vitis_hls", mode=...)` handles `csim` and `csyn`; every other
mode routes to the `XDEVICE` Makefile flow, and the emitted `host.cpp` is an
OpenCL/XRT host, which is not what `cosim_design` wants.

- For an *instruction-programmable* design this matters more than it looks: the
  loop trip counts are data, so `csynth` can only report a worst-case bound
  derived from the ISA's field widths. Measured, same design, two builds:

  | build | csynth | cosim | ratio |
  |---|---|---|---|
  | 16x16 array, 262 instances | **2.259e+08** | **1,176** | **~192,000x** |
  | 4x4 array, before narrowing a row-count field | 91,407 | 4,133 | 22x |

  The five-order-of-magnitude case is the headline; the 22x is the one that was
  *fixed*, by giving the row count its own 7-bit field instead of a 12-bit one
  (no datapath change). Cosim is the only number comparable to a real
  accelerator's cycle count.
- **The general rule, which cost two projects time independently:** a model and
  a measurement that disagree are usually answering different questions, and the
  question the model is answering is often about the *design space* rather than
  the *program*. csynth was not broken either time -- it was correctly bounding
  the machine the encoding permitted. The mirror-image case is a model that
  predicts a *schedule* being read as a prediction about the emitter you
  actually shipped.
- `examples/accelerator/tinytpu_vitis/cosim.py` is a working driver: it
  generates a plain C++ testbench from the same program and reference the
  simulator uses, patches `m_axi` depths (cosim requires them; Allo emits none),
  and drives `vitis_hls`. It is ~180 lines and could be folded into the backend.
- **Priority: Medium.**

## 17. Frontend constraints worth documenting

Each cost real time; none is a bug exactly, but none is discoverable:

- **Stream-array subscripts must be compile-time.** A runtime index fails with
  "Fail to resolve the expression as symbolic expression" -- correct, but it
  does not mention streams. Use `allo.meta_for`.
- **Nested `meta_for` over a 2-D stream array fails** where a single `meta_for`
  over a 1-D one is fine. Forces flat `[T*T]` stream arrays indexed `i*T + j`.
- **Names bound inside `meta_if` are not visible after it** ("Unsupported Name
  `a`"). Declare before, assign inside.
- **Runtime loop bounds *do* work** in a `df.kernel`, including with stream ops
  in the body. This is the capability that makes a workload-independent design
  possible at all -- one build, shape as data -- and it is undocumented.
- **`df.build` is `customize(func)` + `s.build(...)`**, so the schedule
  primitives (`s.partition`, `MockBuffer`) are reachable on the Vitis path.
  Also undocumented, and load-bearing: partitioning the feeders and the
  accumulator took the top-level interval from 168 to 74 cycles.
- **Priority: Low individually, Medium as a "dataflow gotchas" page.**

## 18. Catapult lowers `try_get`/`try_put` to *blocking* reads with `success` hard-coded true

`mlir/lib/Translation/EmitCatapultHLS.cpp` emits `ch.read(v)` for
`StreamTryGetOp` and `ch.write(v)` for `StreamTryPutOp`, then emits
`bool success = true;` unconditionally. The comment says why: `nb_read` /
`nb_write` inside a spin-while loop segfaults Catapult's go compile (LOOP-19),
and a blocking op "always succeeds → spin-while exits in 1 iteration →
bounded".

The workaround is defensible; what is not is that it is silent, and that the
comment understates it. "Scheduling semantics differ only at runtime" is true
for the `while not S.try_put(x): pass` idiom, which is what the bring-up
designs used. It is false for the reason non-blocking ops exist:

- Any design that **branches on failure** -- try this channel, else try the
  next; poll a control channel and do other work if empty; arbitrate between
  requesters -- has its else-branch turned into dead code, because `success` is
  a compile-time `true`. The RTL is silently a different design from the one
  written: a decoupled, backpressured graph becomes lock-step blocking.
- It is not diagnosed. No warning, no `#error`, nothing in the emitted C++
  marking the substitution. The first sign is a Catapult schedule that does not
  match the intended architecture, or a deadlock in a design the simulator runs
  fine.
- The three backends now disagree on the same frontend op, which is the deeper
  problem: **Vivado** emits honest `.read_nb(` / `.write_nb(`
  (`EmitVivadoHLS.cpp`); **Catapult** emits blocking + `true`; **TAPA** emits
  nothing and hard-fails the build (see #19 below, and
  `tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb`). A frontend
  primitive whose meaning changes per target is a correctness trap, not a
  portability inconvenience.
- Related Catapult deviation, same file: `empty()` is emitted as
  `!ch.available(1)` because `ac_channel` has no `.empty()` in the
  synthesizable subset (EDG CIN-59). That one is a faithful translation.
- Documented today only in `notes/ASIC_HLS_EXPLORATION.md` (as a backend note,
  not as a correctness risk) and `docs/source/backends/nonblocking_streams.rst`.
- **Priority: Medium** as it stands, **High** for anyone building arbitration
  on the Catapult path. The cheap fix is to refuse: raise on
  `StreamTryGetOp`/`StreamTryPutOp` for `target="catapult"` unless an explicit
  opt-in attribute says the blocking substitution is acceptable. Failing to
  build beats silently building the wrong circuit.

## 19. `try_get`/`try_put` on the TAPA target fail to emit, with a generic message

`EmitTapaHLS.cpp`'s visitor dispatches only `StreamConstructOp` /
`StreamGetOp` / `StreamPutOp`. The base hooks `emitStreamTryGet`,
`emitStreamTryPut`, `emitStreamEmpty`, `emitStreamFull` in
`mlir/include/allo/Translation/EmitBaseHLS.h` are empty bodies and are never
reached: the op falls through to `visitUnhandledOp`, and `emitBlock` reports
`"can't be correctly emitted"`, which surfaces as
`RuntimeError: Failed to emit HLS code. ... Common issues: nested functions
with multi-dimensional arrays when wrap_io=False.`

Failing is the right call -- this is strictly better than #18. But the message
names a cause that has nothing to do with the actual one, so the user is sent
looking at `wrap_io` instead of at an unsupported op. TAPA has `try_read` /
`try_write`, so the gap is implementable, not fundamental.

- `tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb` asserted
  `.try_read(` / `.try_write(` and so had been failing outright. Marked
  `xfail(strict=True, raises=RuntimeError)` with the mechanism named, so the
  gap is recorded and the test turns red the moment the codegen lands.
- History worth noting: `3723e817` deleted this test as dead, and merge
  `cdac5e68` resurrected it. A test can come back from the dead in a merge
  without anyone noticing it is red.
- **Priority: Low** for the codegen, **Medium** for the error message --
  `emitError` should name the op it could not emit.

## A failure mode worth naming: an unused capability measures as a worthless one

Two independent instances, one from this project and one from the MiniTPU
project, and the trap is sharper than "profile before optimising":

- MiniTPU split a controller FSM to remove a serialisation bound and measured
  **719 cycles before, 719 after** -- because their emitter issues eight pushes
  then eight pops, so the two halves never hold work at the same time.
- We scaled the array from 4x4 to 16x16 and measured **1.47x for 16x the PEs**
  -- because the design is fixed-cost bound and the data path cannot feed it.

In both cases the measurement is correct and the obvious reading of it is
wrong. "Splitting the controller does not help" and "a bigger array does not
pay" are what the numbers literally say, and both conclusions are false: the
capability was *unused*, not *worthless*. Nothing in the measurement
distinguishes those two, which is what makes it dangerous -- a null result
normally retires a hypothesis, and here it silently retires the wrong one.

Practical rule: before changing a mechanism, confirm the workload can present
the mechanism with work it could exploit. If it cannot, fix the schedule or the
feed first, or the experiment will tell you the mechanism is useless.

## Theories tested and disproved

Recorded so nobody re-runs them. Each looked plausible and each cost a cycle of
investigation:

| theory | test | result |
|---|---|---|
| PE `put` order closes a cycle through the drainer's read order | swap the two puts | minimum depth unchanged |
| a cycle in the process graph deadlocks | two repros, one with real traffic on every edge | **both ran** -- cyclic regions are fine |
| the `vld` burst length is the cause | chunk it | still hung |
| the sequencer's control broadcast is the cause | rewrite as a forwarding chain | exact, depth unchanged |

The actual cause was #11, in the simulator, not in any of these.

## 20. The emitter can generate a local whose name collides with a parameter

A kernel body that produces enough SSA temporaries can emit a local with the
same name as one of the function's own parameters, giving C++ that does not
compile:

```cpp
void mover_0(int32_t v0[8], int8_t v1[256], hls::stream< int32_t >& v2) {
  ...
  int8_t v2;          // shadows the stream parameter
  v2 = v50;
  ...                 // later use of v2 as a stream:
}
// ERROR: [HLS 207-3746] subscripted value is not an array, pointer, or vector
```

- Found while probing m_axi burst behaviour (item 13): a `@df.kernel` taking
  three arguments, the third a `Stream`, with a `meta_for` body creating
  several temporaries. The parameter list is numbered `v0, v1, v2` and the
  body's temporaries restart into the same namespace.
- The failure is late and the message is unhelpful: it surfaces from the C++
  front end as a subscript error on a name the user never wrote, with no
  indication that a collision happened. Nothing in Allo warns.
- Workaround: change the kernel's arity or restructure the body so the counters
  do not meet -- which is to say, guess.
- **Priority: Medium.** It is silent at the Allo level, and the diagnostic
  points nowhere near the cause.

## 21. No `#pragma HLS dependence` primitive, so a false dependence cannot be asserted away

Vitis takes `#pragma HLS dependence variable=x inter false` for exactly the case
where the scheduler cannot prove two accesses are independent but the author
can. **Allo emits no dependence pragmas and has no primitive for one** -- the
only pragmas it generates are the `m_axi` / `s_axilite` interface lines in
`allo/backend/vitis.py:410`, plus per-array `bind_storage` / `array_partition`.

- Cost, measured: an accumulator doing `ar[f1+r] = ar[f1+r] + v` with `r`
  carried schedules at `Final II = 3` in BRAM (store/load distance 1) and II=2
  fully partitioned into registers. One pragma line would have said the reads
  and writes never alias.
- Without it the recurrence has to be engineered away in *hardware*: a
  write-behind rotation (hold the last two rows in registers, write `ar` two
  iterations late, answer reads in that window from a bypass mux) takes the
  memory off the carried path and reaches II=1 -- at **13.7x the flip-flops in
  that unit** (1,270 -> 17,450) for a 2.3% end-to-end gain.
- So the missing primitive is not cosmetic: it is the difference between a
  one-line assertion and a hardware redesign with a real area price.
- **Priority: Medium-High.** It is the standard HLS escape hatch for II
  problems and Allo cannot reach it. A `s.dependence(...)` primitive alongside
  the existing `s.partition(...)` is the natural shape.

## 22. The SystemC fork's `Wire` is semantically incomplete -- and wrong in RTL, not just in simulation

Recorded here because it closes a question this project spent real time on: whether
the SystemC path (`choonsik1/allo:SystemC-emitter`) offers the **non-handshaked
fixed-latency edge** that a no-interlock machine needs and Allo's `Stream` cannot
express. It does not.

An earlier investigation reported `pe_wire` as *"synthesises clean and fails
csim"* and concluded the SystemC thread model could not represent a wire --
i.e. that the design was good and the simulator was lying. **That is backwards.**
Simulating Catapult's own `pe_wire` netlist under xsim:

```
Stream[int32,2]  boundary   PASS   (all 18 producer/consumer pacings)
Channel[vld_rdy] boundary   PASS   (all 18)
Wire[int32]      boundary   FAIL   8/8 wrong, at all 18
```

Measured `0 0 0 2 2 10 10 28` against golden `2 10 28 60 110 182 280 408`. **The
csim failure was a true positive.**

Mechanism, from the netlist rather than the model: `acc_0` has no input
handshake at all -- its only input is bare data -- and its loop counter advances
on its *consumer's* ready. `mul_0` latches the wire on its *producers'* valid.
Nothing couples the two counters, `mul` takes ~3 cycles per product and `acc`
one per step, so `acc` runs the entire loop before `mul` produces anything.

- **Positive control:** holding `acc_0` in reset 3-4 cycles longer and stepping
  it once per product makes the *identical* wire RTL produce the exact golden
  result. The wiring and arithmetic are right; only the lockstep is missing, and
  the correct window is **2 cycles wide** (delays 2 and 5 both fail).
- The construct names an edge without specifying when either end samples it. Its
  correctness depends on a global cycle discipline that nothing in the emitter
  establishes, checks, or documents. The one design previously cited as evidence
  that wires work was already flagged as luck; this makes luck the general case.
- **A wire design CAN be verified** -- xsim on the Catapult netlist against a
  numpy golden, with four injected faults all going red, including a one-cycle
  latency change. But it needs hand-built lockstep per design and is not
  checkable by the type system.
- Making wires sound needs the emitter to give cycle-locked kernels a shared
  advance -- one enable driving every stage's counter, which is what a VLIW
  delay line is. `notes/archive/SYSTEMC_COMB_MODE.md` scopes that as the
  `SC_METHOD` "comb" mode in five phases, and its own top risk is whether such a
  model simulates as well as it synthesises.

**Consequence for the roadmap:** the SystemC path does not currently supply a
usable non-handshaked edge, so the MiniTPU-class direction is blocked further
back than "Allo won't hand out two memory ports". It needs a scheduling model
first.

### Two incidental findings

- **A valid/ready protocol violation in the emitted memory port.** `feed_0`
  asserts `v0_req_vld` from its reset state but books the acknowledgement two
  FSM states later, while `AlloMem` consumes the request one cycle after reset
  -- so the full `pe_wire` top deadlocks from reset at any reset length. It
  affects `pe_stream` and `pe_channel` equally, **including the design the
  branch reports as Xcelium-cosim bit-exact**, which suggests these netlists
  were never simulated standalone.
- **No SystemC library or MatchLib on this host** (`ace-01`): no `libsystemc*`,
  `systemc.h`, `connections.h` or `mc_connections.h` anywhere. csim cannot run
  here for any design. SystemC 2.3.x + NVlabs MatchLib would be a few hours and
  no licence.
