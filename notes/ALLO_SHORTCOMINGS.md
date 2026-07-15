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
