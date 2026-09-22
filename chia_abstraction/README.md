# CHIA at the abstraction level: a loop over Allo itself

Two **dispositions** of one loop over one **workload spec**:

| disposition | what the agent edits | the question |
| --- | --- | --- |
| `using` | the design and its program generator, with today's Allo abstractions (`microarch_isa.py`, `isa_dsl.py`) | can an agent find a design point over the abstractions as they stand? |
| `maintaining` | **Allo itself** — schedule primitives, the IR builder, passes, the C++ emitters | can an agent extend the abstractions when the abstraction is what blocks the design? |

They share the workload, the gate ladder, the design cases, the PPA feedback and
the objective, so their success rates are comparable. `maintaining` is expected
to be the harder one; **how much harder is the result this directory exists to
measure**, which is why every iteration records which rung of the ladder it
reached rather than only whether it won.

The workload is an input (`workloads.py`), not a constant. Pointing the loop at
a new workload is a new entry there plus a design case that can execute it.

## The frozen/editable boundary

Enforced mechanically, in the same five ways the design-level loop is, plus one
that is specific to editing a compiler.

1. **A path allowlist over the candidate's git diff** (`patch_policy.py`).
   `FROZEN_GLOBS` is checked *first*, so widening an allowlist by mistake
   cannot open a frozen path. Frozen in both dispositions: every test
   (`tests/**`, `tests/limits/**`), every gate, every golden and reference
   model, the design files a candidate is scored on, this harness, and the
   build system (`CMakeLists.txt`, `*.cmake` — because `ninja` runs what they
   generate and the build is the one stage that must be able to write).
   `EDITABLE["maintaining"]` is `allo/**/*.py`, `mlir/lib/**`,
   `mlir/include/allo/**` and `docs/source/**`; `EDITABLE["using"]` is the two
   design files. A `using` candidate may not touch Allo and a `maintaining`
   candidate may not touch the design — a change that needs both is two
   experiments.
   Both the `diff --git` headers and the `---`/`+++` pairs are read, and they
   must agree: a patch whose header names an allowed file and whose body names
   a test is refused.
2. **Every evaluation is assembled from git.** A slot worktree is
   `git checkout -f <ref>` + `git clean -xffd`, then the allowlisted patch is
   applied. Afterwards `git status --porcelain --untracked-files=all` must name
   *exactly* the patch's declared files, and each path in `FROZEN_CHECK` is
   compared **byte for byte** with `git show <ref>:<path>` — re-checked after
   every stage. The design's own evaluator (`cosim.py`, `bench_isa.py`,
   `stress_isa.py`, `isa_ref.py`, `kpn_model.py`) is additionally pinned
   byte-identical to main @ `MAIN_BASE`, so this loop measures the design
   exactly as main does.
3. **An added-line policy** (`patch_policy.PY_DENY` / `CPP_DENY`). Narrow and
   checked against the tree: `allo/` uses none of `sys.modules`, `builtins`,
   `importlib.reload`, frame walking or `pytest` today, and
   `mlir/lib/Translation` opens no files, so refusing them in *added* lines
   refuses nothing Allo does. `os.system` and `shell=True` do occur in
   `allo/backend/hls.py`; only new occurrences are refused.
4. **A bubblewrap sandbox.** Filesystem read-only, only the work directory and
   a private `/tmp` writable, own PID namespace. The build is the exception and
   is sandboxed to `mlir/build` alone. Bind **order** is the reverse of the
   design loop's: the work directory sits *inside* the slot here, so the slot's
   read-only bind must come first or every gate dies with `EROFS`.
5. **Nonce-vouched verdicts** (`abs_gate_runner.py`). A fresh nonce per run,
   read on stdin before anything imports the candidate; the verdict is the
   check's **return value**, and `CHIA-GATE <check> OK <nonce>` is printed only
   then. The nonce never reaches a log or a verdict.
6. **The ordering that is specific to this loop.** `chia_agent/gate_runner.py`
   freezes `numpy` *and* `allo` before importing the candidate. Here **`allo`
   is the candidate**, so it cannot be frozen, and a snapshot taken after
   `import allo` would already contain whatever the candidate did to numpy.
   `abs_gate_runner.py` therefore reuses that module's `_Frozen` / `_snapshot`
   / `_changed` primitive — imported, not copied — with `WATCHED` narrowed to
   `numpy` and `builtins`, and applies the freeze **while `allo` is still
   unimported**.

**What this gives up, stated plainly.** Allo's own test suite uses Allo to
check Allo, so a candidate that weakened a check inside `allo/` could in
principle pass gate (b). The answer is that the design cases' correctness comes
from `mode="csim"` — the emitted C++ compiled by g++ and **run**, compared to a
numpy golden computed with numpy frozen — and the TinyTPU-isa cycle numbers come
from Vitis's own `cosim_*.log` files. Neither executes Allo. It is still a
policy, not a proof; every accepted diff gets read by a person.

## The gate ladder, cheapest first

CAKE's rule: filter through the cheap static gates before spending hardware
time. The measured cost of each rung on this host, on an unmodified tree:

| # | gate | what it checks | cost |
| --- | --- | --- | --- |
| 0 | `policy` | path allowlist, added lines, and the primitive rule | milliseconds |
| 1 | `assemble` | slot reset, patch applied, 22 frozen paths byte-identical | ~1 s |
| 2 | `gate:build` | `ninja -C mlir/build -j48`; the compiler's diagnostic is the feedback | ~30 s warm |
| 3 | `gate:import` | `import allo` + HLS backends; `allo.__file__` inside the slot | ~1.5 s |
| 4 | `gate:tests` | Allo's own suites, per test name, against the **measured** baseline | ~120 s (fast tier) |
| 5 | `gate:design` | TinyTPU-isa `bench_isa` + `stress_isa`; every design case bit-exact under csim; `tests/limits/` verdicts unchanged | ~30 s + ~15 s + ~45 s |
| 6 | `gate:resources` | every csynth resource inside budget on every case | free (same csynth) |
| 7 | `ppa` | RTL cosim cycles per shape; csynth latency/interval/area/clock per case | ~150 s + ~60 s |

`gate:tests` compares against **main's measured baseline, not zero**: main does
not pass its own suite. The baseline is recorded by `evaluate_abs.py
--record-baseline` from *inside* the sandbox, because a hand-recorded one does
not reproduce there — see the note in `gate_tests`.

The static gate at rung 0 that is specific to a compiler loop: **a new
`Schedule` method must raise `AlloValueError` and carry a docstring**, or the
candidate is refused before anything is built. That is CAKE's "types checked at
construction" as a gate rather than as advice, and the tree contains both the
pattern (`s.dependence`: five raises, three tests) and the counter-example
(`align_value`: no validation, no test).

## The objective is multi-term

**Cycles are the score. Resources are a constraint.** Every csynth resource
(BRAM_18K, DSP, FF, LUT, URAM) must stay within `max(1.10 × baseline,
baseline + floor)` on every design case, and the estimated clock must still
meet 3.33 ns; outside that the candidate is rejected at `gate:resources`
whatever it did to cycles.

Chosen over a weighted score for four reasons (`objective.py`): no exchange
rate between a BRAM and a cycle has to be invented without place-and-route; a
constraint is a gate and gates compose; "fewer cycles at no more than +10% of
any resource, on every design case" is a claim a reader can check; and a
weighted sum over design cases would let a big win on one pay for a regression
on another.

Nothing is aggregated. Results are reported **per design case**. A candidate
that gains cycles somewhere and leaves the budget or loses cycles elsewhere is
classified `trade`, never `win`, and is not kept. The design-level loop's one
verified win — 8.6% fewer cycles at 2.3× the BRAM — is classified `trade` by
this module, and the LLM-free suite asserts that (phase `h`).

## The design cases

| case | machine | correctness | PPA |
| --- | --- | --- | --- |
| `tinytpu_isa` | instruction-programmable int8 TPU, 4×4 weight-stationary | `bench_isa` + `stress_isa`, then per-shape cosim `mismatches = 0` | **RTL C/RTL cosim cycles** at the workload's shapes, + csynth area/clock |
| `systolic_1d` | the *same workload*, a 1-D streaming systolic chain, fixed function | csim bit-exact vs numpy | none — Vitis refuses to synthesise it (`HLS 200-779`: several kernel instances read one top-level array in a dataflow region; register item "shared memory" / issue #27) |
| `blocks_stream` | producer/consumer over `Stream[int32[M,N], d]` — the emitter's `hls::vector` path | csim bit-exact | csynth latency / interval / area |
| `mlp_layered` | 3-layer float32 MLP with `s.pipeline`, `s.unroll`, `s.partition` | csim within 1e-3 | csynth latency / interval / area |

Why these three second cases: `systolic_1d` is the same workload on a different
machine, so "reusable across design cases" means something for a GEMM
abstraction; `blocks_stream` is an emitter path neither other case touches;
`mlp_layered` is floating point and layered, so an abstraction that pays off
there too is not a GEMM trick. All three are defined in `design_cases.py`,
which is frozen, with module-level designs and a fixed RNG seed — module level
because `@df.region`'s type resolution uses the defining module's globals, so a
builder-local type alias resolves to nothing.

## What is reused, and what is rewritten

Reused from `examples/accelerator/tinytpu_vitis/chia_agent/`, by import rather
than by copy:

- `gate_runner._Frozen` / `_snapshot` / `_changed` — the module-freezing
  primitive. It is the one security-critical piece in either harness and there
  should be exactly one of it.
- `evaluate.parse_synth`, `evaluate.check_memory_model`, `SIMTIME_SLACK`,
  `TARGET_NS`, `MAIN_BASE` — the csynth parse, the memory-model checks and the
  main pin.
- `bench_isa.py`, `stress_isa.py`, `cosim.py`, `param_check.py`,
  `isa_ref.py` — the TinyTPU-isa design case's own frozen gates, run as they
  are.
- `preflight.py` (the billing gate), `spend.py` (spend from opencode's DB),
  `llm.py` (the 40-minute-MCP-timeout LLM wrapper), `fake_model.py` (the
  scripted model the LLM-free suite points opencode at).

Rewritten, and why:

- `abs_gate_runner.py` — the freeze **ordering** must change, because `allo` is
  the candidate.
- `patch_policy.py` — the subject is a diff over a compiler, not the content of
  two spec files; and it carries the primitive rule, which has no analogue in
  the design loop.
- `evaluate_abs.py` — a slot worktree with a C++ rebuild, not a composed tree.
- `abs_loop.py` / `abs_tool.py` — two dispositions, a workload input,
  per-iteration rung instrumentation, and a `build_allo` tool (a compiler
  diagnostic is the cheapest feedback a C++ candidate can get).
- `suite_runner.py`, `limits_runner.py`, `design_cases.py`, `objective.py`,
  `workloads.py`, `prompt.py` — new; there is nothing at the design level that
  runs Allo's own suites, the limitation corpus, several design cases, or a
  two-term objective.

## Files

| file | role |
| --- | --- |
| `workloads.py` | frozen: the workload-level specs, and the shapes each is scored on |
| `patch_policy.py` | frozen: the path allowlist, the added-line policy, the primitive rule, the non-blocking hints |
| `abs_gate_runner.py` | frozen: runs one check and vouches for its verdict with a nonce, freezing numpy/builtins before `allo` loads |
| `suite_runner.py` | frozen: Allo's own suites, per file with a timeout, per test name against a recorded baseline |
| `limits_runner.py` | frozen: `tests/limits/` verdicts against a recorded baseline; a `REPRODUCES -> FIXED` is reported as a repair |
| `design_cases.py` | frozen: the second/third/fourth design cases, their numpy goldens, csim and csynth |
| `objective.py` | frozen: cycles scored, resources constrained, `win`/`trade`/`neutral`/`worse` |
| `evaluate_abs.py` | frozen: the slot, the ladder, one JSON verdict |
| `prompt.py` | frozen: the two briefs, built from the workload spec |
| `abs_tool.py` | the MCP surface: read, edit, `check_policy`, `build_allo`, `run_gates`, `score` |
| `abs_loop.py` | one search, either disposition, with per-iteration rung instrumentation |
| `test_abs_harness.py` | the LLM-free end-to-end test; run it before spending |
| `baseline/` | the measured baselines: `suite_fast.json`, `limits.json`, `ppa.json` |

## Running it

```bash
conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

# 1. record the baselines (no patch), from inside the sandbox. ~8 min.
python chia_abstraction/evaluate_abs.py --out .chia_scratch/baseline --record-baseline

# 2. the LLM-free test suite. $0.
python chia_abstraction/test_abs_harness.py --phases d,f,h,i    # static, seconds
python chia_abstraction/test_abs_harness.py                     # everything

# 3. a search. conda activate chia_env; ray start --head ...
set -a; source chia.env; set +a
export CHIA_TOTAL_CAP_USD=228.54       # this track's cumulative ceiling
python chia_abstraction/preflight_check.py --budget-usd 15   # or preflight.py
python chia_abstraction/abs_loop.py --disposition maintaining \
    --workload gemm_int8_16 --iterations 2 --budget-usd 15 \
    --angle "loop directives cannot attach to a while loop" \
    --log-dir chia_runs/abs-$(date +%Y%m%d-%H%M%S)
```

`CHIA_TOTAL_CAP_USD` is a **cumulative** ceiling on CHIA2026 spend, not a
per-run one. The split between the two tracks is recorded in
`../examples/accelerator/tinytpu_vitis/chia_agent/allocation.json`; the
pre-flight gate cannot tell the tracks apart, so each track honours the split by
setting its own ceiling.
