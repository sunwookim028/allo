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
- **Still open, and still worth doing:** the *diagnosis* remains absent. The
  symptom was a silent hang with no indication of which process was blocked on
  which channel, and that is what cost the sessions -- not the deadlock itself.
  `examples/accelerator/tinytpu_vitis/kpn_model.py` shows the report is ~30
  lines of bookkeeping.

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

## 13. No program-controlled DMA: bulk copy or unbuffered `m_axi`, nothing between

`wrap_io=True` copies each argument into a local buffer before the region runs,
sized to the **declared** array rather than to what the program touches.
`wrap_io=False` drops the copy but every access then pays bus latency.

Measured, same design, one build each (cycles, cosim):

| config | marginal | fixed | 4x4x4 | 16x16x16 |
|---|---|---|---|---|
| `wrap_io=True` | 18.1 cyc/instr | 1102 | 2.12x | 1.94x |
| `wrap_io=False` | **39.8** | **481** | **1.21x** | 2.25x |
| Gemmini | **10.8** | **483** | 1.00x | 1.00x |

- `wrap_io=False` reaches Gemmini's *fixed* cost (481 vs 483) but doubles the
  marginal cost. Crossover is ~29 instructions.
- Gemmini has the third option and Allo does not expose it: a bursted DMA the
  program controls (`mvin` moves exactly the tiles named), which is low on
  *both* terms. **This is the single largest structural gap** to a
  Gemmini-class design, worth ~1.9x of a measured 1.8x total.
- **Priority: High** for any accelerator work.

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
