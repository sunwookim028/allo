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


## The aborted first launch: two harness defects a real agent found, for $4.18

The first launch of both pilots produced no gradeable candidate, and is kept
as evidence rather than deleted. Two defects, both of which made the agent's
feedback loop inert, and **both invisible to every test in this directory**:

1. `abs_tool._evaluate` launched `evaluate_abs.py` with `sys.executable` --
   the CHIA environment (py3.10, no numpy, no allo), where it cannot import.
   **Every** candidate failed at stage `evaluator` and no harness feedback ever
   reached the agent. True since `design_cases.py` gained module-level imports;
   the held-out run hit it too, and the `json.loads(dict)` crash hid it.
2. `list_probes` / `declare_probe` imported `probes.py`, which needs numpy and
   allo. Both raised in that environment, so **the agent could not declare a
   probe call** -- Pilot A said exactly that in its final message -- the probe
   rung was skipped, and `expressive` was unreachable.

**This is the no-caller lesson happening to this harness.** Phase `k` proved
the probe rung from the EXPERIMENTER's environment and never exercised it
through the path the AGENT uses, so a capability that nothing exercised the way
a real user does could not be seen to be broken. Repairs: `probe_meta.py`
imports nothing and `probes.py` reads its ground truth from it, so the two
cannot drift; **phase `l` runs the tool surface inside the agent's environment**
and asserts the evaluator is launched with the allo interpreter.

**$4.18 for two defects a real agent found that this directory's own testing
could not.** That is the price, and it is worth stating at that price.

### What the aborted Pilot A proposed, and its grade: kind (4), ALREADY EXPRESSIBLE

Recorded as an observation about what an agent reaches for, not as a pilot
result -- it never received a single harness verdict.

It chose **memory write ports** -- the property seeded measurement M3 points
straight at, which the pre-registration said in advance would be reported as
*choosing the most salient seeded measurement, not discovery*. It diagnosed
kind (1) CANNOT EXPRESS and proposed `s.bind_storage(target, impl, type)`.

**The diagnosis is wrong and the capability already exists.** On the pilot tree:
`allo/memory.py` defines `Memory(resource=..., storage_type=..., latency=...,
depth=...)`, used as a type annotation (`b: float32[64,64] @
Memory(resource="BRAM", storage_type="RAM_2P")`), with
`VALID_RESOURCE = {BRAM, URAM, LUTRAM, SRL, AUTO}` and storage types spanning
the port structures -- `RAM_1P`, `RAM_2P`, `RAM_T2P`, `RAM_1WNR`; and both
`EmitVivadoHLS.cpp:2737` and `EmitTapaHLS.cpp:2179` already emit
`#pragma HLS bind_storage variable=... type=... impl=...` from it. There is no
`Schedule.bind_storage`, so the agent's SPELLING was new; the CAPABILITY was
not. Grade: **(4) already expressible**, misdiagnosed as (1).

So the pre-registration was right twice over: A chose the most salient seeded
measurement, **and** it chose the thing already expressible.

### And what that makes the real gap -- which is what Pilot B is grading

The port property is already DECLARABLE. What is missing is the **legality
rule**: nothing checks a design's access pattern against its declaration. A
design may declare `RAM_1P` and still emit two write statements, and nothing
refuses it. That is exactly the ground truth here -- `rbA` emitted with two
write statements while its module was named `_1R1W`, a declaration the
implementation contradicted, with no checker anywhere.

That is this directory's own *cannot express* versus *cannot refuse*
distinction landing on the pilot built around it. **Pilot B is NOT retargeted**;
its prompt is unchanged mid-run and its grade carries the information:

- a way to DECLARE port counts is **kind (4)**, the same as A -- and two
  independent arms landing on an existing surface is itself a finding about
  what agents reach for;
- a CHECKER that refuses a design whose writes contradict its declaration is
  **kind (1) or (2)**, and is the architectural abstraction that was asked for;
- and it is recorded separately whether the agent DISCOVERS that the
  declaration already exists and says "the gap is enforcement, not expression".
  Diagnosis is the half this project's own engineers got wrong twice.

The probe rung already grades exactly this and needs no change: `expressive`
requires REFUSING `ports_dual` and ACCEPTING a bit-exact `ports_banked` -- a
checker, not a declaration.

## RESULT -- Pilot B, iteration 1: an architectural legality rule that discriminates

Run `chia_runs/pilotB-20260922-180708`, ref `03e426f9`, $1.39, 36 turns,
1207 s for the turn and 1895 s including the harness. **Accepted; rung
`improved`; objective `expressive`; `newly_expressible:
['ports:write-port-declaration']`.**

It proposed `s.memory_ports(target, write_ports)` -- **a legality rule, not a
declaration**. It counts statically visible `affine.store`/`memref.store` to
the buffer in the innermost loop and refuses when
`stores > write_ports x banks`, **before touching the IR**, then sets a
`memory_ports` attribute.

### The probe rung, against both measured ground truths

| site | required | measured |
| --- | --- | --- |
| `ports_dual` (1 RAM, 2 write statements) | refused | **refused** |
| `ports_banked` (2 instances, 1 write each) | accepted, bit-exact under csim | **accepted, `banked_confirmed: true`** |

It declared both calls itself (`s.memory_ports("buf", 1)` at each site) through
`declare_probe`. Every other rung passed: 294/294 tests with no regressions,
`bench_isa` ALL EXACT, `stress_isa` 492/492, all design cases bit-exact, 20
`tests/limits` verdicts unchanged, every resource at ratio 1.000, +0 cycles.

### Did it discover the declaration already existed? **No.**

Its diagnosis, verbatim: *"The machine's physical limitation ... could not be
expressed in the language (1). Because the language couldn't state the write
port capacity of an array, it accepted loops that mapped multiple concurrent
writes to the same bank"*. That first half is **factually wrong** on its own
tree: `Memory(resource=, storage_type="RAM_1P")` states exactly that, and both
emitters emit `bind_storage` from it. Checked against the transcript: B's
reasoning text never mentions `bind_storage`, `storage_type` or `Memory(` --
the matches in its session are files it read, not conclusions it drew. So it
reached the right ARTEFACT (a checker) through a half-wrong DIAGNOSIS: the gap
was **enforcement only**, kind (2), not kind (1)+(2) as it claimed.

### Is `banks x write_ports >= stores` sound enough to ship? **No, and it errs in both directions.**

The agent named one hole itself, unprompted: writes aliasing to the same bank
(`buf[2*i]` and `buf[2*i+2]` under cyclic 2) would be *"erroneously
accept[ed]"*. Two more, **measured** here by applying its own primitive:

| case | its verdict | truth |
| --- | --- | --- |
| cyclic factor 2 | accepted | correct |
| **`Partition.Block` factor 2** | **ACCEPTED** | **wrong -- a false accept.** Both stores land in bank 0; the block-partitioned RTL emits **2 write statements per bank instance**, measured by csynth tonight. Its code does `banks *= factor` for `Block` and `Cyclic` alike |
| **2 stores, loop NOT pipelined** | **refused** | **wrong -- a false refuse.** Sequential stores need one port; the rule never consults `pipeline_ii` |

So it is a heuristic that both admits unsynthesisable designs and rejects legal
ones, and the false accept is reachable through a standard primitive with
ordinary arguments. **That gap is worth more than the primitive**: the correct
rule needs the layout map (the IR already carries the cyclic partition as
`(d0) -> (d0 mod 2, d0 floordiv 2)`) composed with each store's affine index,
and the loop's initiation interval -- both present in the IR, neither used.

### Against the pre-registration

Predicted for B: separates the ground truths and is confirmed, **30%**;
refuses both, 45%; accepts both, 10%; crash or build failure, 15%. **The 30%
branch happened.** Predicted architectural 90% -- yes, it describes the
machine's memories. Predicted states-its-premise 60% -- **yes, unprompted and
specifically.**

**Grade: (1)/(2) architectural legality rule, `expressive`, sound only as a
heuristic.** It is structurally what `s.dependence` is -- refuse the
provably-false, accept the unprovable, label the residue -- arrived at without
being shown that shape for this property.

## PRE-REGISTRATION -- Pilots A and B, written 2026-09-22 17:19 UTC, BEFORE either ran

Both run on this branch at the commit that adds this section, on ONE tree for
their duration, with the ref recorded in each run's `summary.json`. Not rebased
onto `origin/main`: main now carries `dev/SESSION_REPORT.md`, whose lines
86-94 list five architectural targets including Pilot B's -- rebasing would
hand Pilot A its answer key. Verified absent from this tree.

Common: `google-vertex/gemini-3.1-pro-preview`, 1 worker each, at most 3
iterations, 1 debug attempt, **$30 cap each** (per-run cap sums only the run's
own sessions). Pilots run concurrently in slots 0 and 1. The limitations
register and the design history are removed from the agent's reference map.
Exactly what each prompt seeded is `prompt.SEEDED`, reproduced in each run's
`summary.json`.

Grading, per pilot: the four-way scale (same-abstraction / equal power at a
different level / instance solved without an abstraction / no solution, with
the rung where it stopped); **is the abstraction architectural** (does it
describe the machine, or the tooling -- a gate-passing tooling abstraction is
graded `tooling`); and **did it state its unchecked premise**. Pilot B is also
graded mechanically by the probe rung against the measured RTL ground truth.

### Pilot A (open arm: measurements only) -- predictions

- **Which property it names.** Memory write ports 50%, address-term/AGU
  structure 25%, the `nr` row-count field 10%, something else or tooling 15%.
- **A confound, stated now:** measurement M3 -- seeded because the brief asked
  for it -- is the rbA table with "two always-blocks that WRITE" and the DC
  refusal. It is the most concrete, actionable row in the seed and it points
  straight at Pilot B's property. **If A names memory ports, that is weak
  evidence of identification**; it will be reported as "chose the most
  salient seeded measurement", not as discovery. A choice of a property NOT
  suggested by M3 is the informative outcome.
- **How far it gets.** Builds and passes the cheap gates: 50%. Reaches
  `expressive`: under 10%.
- **Architectural:** yes, 70% (it is instructed). **States the premise:** 70%.

### Pilot B (directed: write-port property + RTL ground truth) -- predictions

- Proposes a write-port declaration of some form: 90%.
- **Probe outcome.** The IR encodes a cyclic partition as an affine LAYOUT
  MAP on the memref type, `(d0) -> (d0 mod 2, d0 floordiv 2)`; the stores
  carry affine indices `i*2` and `i*2+1`. A check that counts stores per buffer
  without composing them with the layout map refuses BOTH probes. So:
  refuses both 45%; separates them correctly and the banked design is
  confirmed bit-exact (`expressive`) 30%; accepts both 10%; build failure or
  crash at the probe 15%.
- **Architectural:** yes, 90%. **States the premise:** 60% -- a port count
  checked against the access pattern is closer to a fact than a promise, and
  an agent may reasonably say so.

### What would change the conclusion

If B passes and A does not, the directed-target arm works and the open arm
does not: agents can implement an architectural legality rule when told the
property, not find one. If A names a property not suggested by M3 and builds
it, that is the result the owner asked for. If both fail at the same rung, the
rung is the finding.

## The held-out rediscovery run, 2026-09-22

Given only `symptom.md` — never told the primitive exists, working from
`a4151ca0`, where it never has — the agent produced a 162-line candidate
across five files in one 24.8-minute turn.

### A finding in its own right: two retractions of one result

The held-out result has now been retracted TWICE -- first for a leak through
the docs (`limitations.rst:41`, missed because the leak detector failed open),
then for a leak through the PROMPT. Both came from this track rather than from
a reviewer. The general rule: **a held-out experiment must scan everything the
agent reads, and the prompt is the thing most easily forgotten, because the
experimenter wrote it.** `abs_loop --heldout` now assembles the full prompt and
refuses on any hit; it finds five in the current maintaining prompt, which is
the proof the repair was needed.

### SECOND CORRECTION (2026-09-22, evening): the system prompt described the answer

**The first retraction below is incomplete, and its surviving claim does not
survive either.** The held-out loop assembles its system prompt from THIS
tree's `chia_abstraction/prompt.py` — never from the redacted graft, whose own
redacted `prompt.py` was therefore cosmetic. At the time of the run
(`ecdf6e96`) the maintaining brief said, in so many words:

- `s.dependence` (`allo/customize.py:833-943`) **is the pattern**;
- it has **"five explicit `AlloValueError` raises"**;
- it sets **"a `dependence` `ArrayAttr` of `DictAttr` appended to (not
  overwriting) the loop's attributes"**;
- it has **"one branch in `EmitVivadoHLS.cpp:emitLoopDirectives`
  (2603-2647)"**;
- **"Other emitters that cannot honour it must REJECT it, not ignore it"**;
- `s.dependence` is a claim that, **"if false, produce[s] wrong RTL while every
  software simulation passes."**

Every feature credited to the agent below is in that list: the name, the
attribute and its shape, the append, the emitter location, the rejecters, and
the promise-not-fact docstring. So:

- **The attribution paragraph below is WRONG** where it says the diagnosis and
  the rejecters are the agent's. Both were in its prompt.
- **"On a filed, still-open defect its version is better than the tree's" is
  withdrawn as a claim about the agent.** It remains true that the candidate
  has rejecters and `bbea2af0` does not; it is not evidence of judgement, it
  is compliance with an instruction.
- **"It noticed the claim is a promise" is withdrawn.** It was told.
- **The differences table is not evidence of reconstruction.** It is where an
  implementation deviated from a specification it was given.

**What is left, stated at its true strength:** given a prose specification of
a primitive in its system prompt, the agent wrote a 162-line, five-place
implementation that builds, passes every gate, and aborts the compiler on first
call because its Python and C++ halves disagree on an IR type. That is an
*implementation-from-specification* result, graded **near miss**, and it says
nothing about whether an agent can find an abstraction.

**Why the leak detector did not catch it:** it scanned the TREE. The channel
that carried the most information — the prompt — was never scanned. A detector
that proves it looked is not enough if it looked in the wrong place. Fixed:
`abs_loop --heldout` now assembles the full system and task prompt, runs it
through the same `LEAK_RE`, and **refuses to start** on any hit. A held-out
re-run also needs a frame whose pattern exemplar is not the held-out answer;
the current frame uses `s.dependence` as the exemplar and would be refused.

### RETRACTED AS A REDISCOVERY RESULT. Read this before anything below.

**The held-out ref leaked the answer, and the leak was in the agent's read
path.** `docs/source/developer/limitations.rst` line **41** — inside the
lines 1-150 the agent actually read — says:

> combinational wires (fork issue #9), **HLS dependence pragma (fork issue
> #10)**, ...

and the same file says at :204 "No schedule primitive and no emitter path for
``#pragma HLS dependence``", at :1171 "21. No ``#pragma HLS dependence``
primitive, so a false dependence cannot be asserted away", and at :1178
"Vitis takes ``#pragma HLS dependence variable=x inter false`` for exactly the
case". `microarch_isa.py` carries the pragma verbatim in a comment.
**14 leaks in total across 5 files.**

**Why the leak check said zero: it failed open.** `scan_leaks` used
`git grep -E LEAK_RE`, and `git grep -E` is POSIX ERE — it rejects `(?:...)`
and exited **128**. The helper was called with `check=False`, so the failure
was read as "no matches". *The detector reported a clean tree because it had
crashed.* This is the same disease as the tool server that never bound and
returned anyway, and as `usage` reporting $0.00: **a check that fails open.**
I built one while cataloguing them. It now uses Python's `re` over `git show`,
raises on any git failure, and returns `file:line:text`.

Why the redaction did not cover these files: `graft` deliberately leaves the
design's own evaluator and the docs at the base commit, so only the *grafted*
harness files were redacted. That decision was right for the evaluator and
wrong for the docs, and the failing-open detector is what stopped the mistake
being seen.

**What does NOT survive:** the word *rediscovery*. The agent was shown a line
naming the pragma as a known gap.

**What does survive, and is weaker but real:** the agent was pointed at a
*gap* — "no schedule primitive and no emitter path for `#pragma HLS
dependence`" — not at a design, and turned it into a correct, building,
validated, five-place implementation that passes every gate, including
rejecters in three emitters that the human commit did not write. That is an
**implementation** result, not a rediscovery result. The gate numbers below,
the differences table, and every harness finding stand unchanged.

**Before any rediscovery claim is made, the experiment must be re-run** with
the docs redacted at the held-out ref and the fixed detector.

**Four-way outcome, as an implementation result: NEAR MISS.** It passes every
rung of the ladder and it **aborts the compiler the first time anything calls
it.** See "G3" below. The four-way grade is *not* `same-abstraction`.

What it produced:

- `Schedule.dependence(target, axis, dep_type="inter", direction="RAW",
  distance=0, true_false="false")` — the same name and essentially the same
  signature as the fork's real `s.dependence`;
- four `AlloValueError` validations against closed sets, before any IR is
  touched;
- a `dependence` `ArrayAttr` of `DictAttr` on the loop, **appended to rather
  than overwritten** — step 2 of the pattern, exactly;
- an emitter branch in `EmitVivadoHLS.cpp` (+81);
- **rejecters in `EmitCatapultHLS.cpp`, `EmitTapaHLS.cpp` and
  `EmitIntelHLS.cpp`** (+4 each), e.g. `emitError(op, "Catapult HLS does not
  support the dependence directive")`.

### The attribution, which must not be collapsed

**The prompt supplied the five-place pattern, told it to reject rather than
ignore, and told it to state what is a promise — so the shape is guided. What
is the agent's is the diagnosis, that a dependence assertion is the right
abstraction for this symptom, and the decision to add the rejecters.**

An unqualified "an agent rediscovered `s.dependence`" would be the most
damaging overstatement available here, and it is one sentence away.

### On a filed, still-open defect its version is better than the tree's

The original human commit (`bbea2af0`) did **not** write the rejecters.
Issues **#23, #24 and #31** exist precisely because `EmitCatapultHLS.cpp`,
`EmitTapaHLS.cpp` and `EmitIntelHLS.cpp` ignore a directive instead of
rejecting it — all three are "ignored instead of rejected". The agent's
version closes that shape of defect for this directive; the tree's does not.

### It noticed the claim is a promise

From its own docstring, unprompted as to content:

> *"If what it expresses is a PROMISE rather than a fact (e.g. false
> dependence claim), a false claim would produce wrong RTL while every
> software simulation stayed exact."*

That is the thing the original commit's own register entry had to admit
afterwards.

### Differences from the tree's version — evidence it reconstructed rather than recalled

A reader deciding whether this was memorisation will look for exactly these:

| | the tree (`bbea2af0`) | the agent |
| --- | --- | --- |
| argument order | `(axis, target)` | `(target, axis)` |
| the claim | `dependent=False` (bool) | `true_false="false"` (string) |
| distance | `>= 1`, and only with `dependent=True` | `>= 0` allowed |
| `dep_class` | `array` / `pointer` | absent |
| declared-inside-the-loop check | present | absent |
| function arguments | `arg_index` path | absent |

A recall would not differ in those ways.

### Provenance of the surviving claim

From opencode's own database, session `ses_f3674bacdffeBJ...`: **36 reads**,
all of `allo/customize.py`, the five `Emit*HLS.cpp` emitters,
`mlir/include/allo/Translation/EmitIntelHLS.h`, `allo/backend/vitis.py`, and
`docs/source/developer/limitations.rst` lines 1-150. **It never opened
`microarch_isa.py`**, so it did not see the verbatim pragma in that file's
comment; what it saw was the `limitations.rst:41` line naming the gap. That is
why the surviving claim is *implementation*, and why it is not *nothing*.

## G3: the decisive test, and it failed

**We removed the step that would have exposed the defect, then built a ladder
that could not compensate.** The five-place pattern's step 4 is the test, and
the agent was explicitly told it could not write one — tests are frozen. The
thing that would have caught this bug is exactly that step. This is an
indictment of the experimental design, not of the agent, and the next version
of the experiment turns on it.

**One structural fact made the work simultaneously unrewardable and
undetectably broken.** Those look like two problems — an objective that
undervalues new abstractions, and a gate ladder with a hole — and they are one
problem wearing two hats. It generalises past this harness: **any gate ladder
that exercises a compiler only through existing designs cannot see a new
capability, in either direction. It cannot see that it is good and it cannot
see that it is broken.**

The gate cannot exercise a primitive that nothing calls, so `heldout.py`'s G3
step applies the primitive **plus a harness-authored call site in the
candidate's own spelling** — forced by its reversed signature to

    s.dependence("ar", "accu_0:c", dep_type="inter", true_false="false")

and measures. Result: **the compiler aborts.**

    BuiltinAttributes.cpp:365: mlir::IntegerAttr::getInt():
      Assertion `(getType().isIndex() || getType().isSignlessInteger())
                 && "must be signless integer"' failed.
    #10 VhlsModuleEmitter::emitLoopDirectives(Operation*)

The cause is one line. The agent's Python builds the `distance` field as an
**unsigned** integer:

    i32 = IntegerType.get_unsigned(32)
    "distance": IntegerAttr.get(i32, distance),

and its own C++ then reads it with `IntegerAttr::getInt()`, which requires a
**signless** integer. (The tree's version uses `IntegerType.get_signless(64)`.)
The Python side and the emitter side disagree about the IR type, and nothing
noticed, because **nothing called it**.

**This is the strongest argument in the directory for the harness-authored
call-site rung.** Without G3 the candidate looked correct: it built in 28.3 s,
passed 291/291 tests with zero regressions, was bit-exact on three design
cases, and left every resource at ratio 1.000. The defect is reachable only
through a call site, and the agent cannot write one. So the objective could
not REWARD it and the gates could not CATCH it, for the same reason.

It also makes the five-place pattern's point concretely: a primitive and its
emitter must agree, and the thing that proves they agree is the test — step 4,
the one the agent was explicitly told it could not write.

## A limitation of THIS OBJECTIVE, not of the candidate

**A new abstraction, by construction, has no callers.** The objective scores
cycles at existing call sites, so it can only reward an extension that pays
off where the designs already are — which is systematically the opposite of
the extensions this project most wants, the ones whose value is
generalisability across design cases.

This candidate is the measured proof: correct, building, validated, bit-exact
everywhere, every resource flat, and **`neutral`**. `s.dependence` itself would
have scored `neutral` the day it was written.

`expressive` was meant to cover this and does not: it fires only on
`newly_expressible` (a design case that did not synthesise now does) or
`limits_fixed` (a `tests/limits` verdict moving REPRODUCES -> FIXED). A brand
new primitive with no callers triggers neither.

The guard this collides with — *"you cannot make a design use your primitive;
it has to pay off where the designs already are"* — exists to stop an agent
rewriting a design to flatter its own work, and that is still right. The
resolution is a **harness-authored call site**: derived by introspection from
the candidate's own `Schedule` method signature, applied to a frozen design,
and scored as one unit with the primitive. The agent never writes it, so the
guard holds, and a new abstraction can still show payoff. That is what the
`heldout.py` G3 step does by hand today; it belongs in the ladder as a rung.

**Not implemented tonight.** Recorded here as a known limitation of the
objective so that a `neutral` on a new primitive is read correctly.

## Is the cosim deterministic? A cycle-identity count, not an assertion

Measured 2026-09-22, no-patch controls of the full ladder at one ref:

**8 successful cosims, 8 identical cycle counts, 0 differing counts, 1 run
that yielded no number at all.**

Every successful run gave `tinytpu_isa` 172 / 686 and `blocks_stream`
interval 89 / latency 206, matching the recorded baseline and each other.
The single failure was `4x4x4 cycles=None no TB line` **while 16x16x16
returned 686 in the same run** — a MISSING number, never a different one.

A nondeterministic simulation gives different numbers. A flaky pipeline gives
no number. The evidence says the second. So:

**The simulation is deterministic; the PIPELINE can fail to produce a number.
A single `ppa:cosim` rejection is therefore not evidence and must be retried
rather than believed.**

Our documentation says the cosim is deterministic and uses that to justify
single measurements. That remains true of the simulation while being
incomplete about the pipeline. Logs are now copied out of the work directory
**before** any rejection, because the first occurrence destroyed its own
evidence and would otherwise have been a shrug.

## Name the disease: a check that fails open

Four instances in one night, in four different instruments:

| instrument | what it reported | what was true |
| --- | --- | --- |
| CHIA's tool server | the LLM call returned normally | it never bound to a port; the agent had no tools |
| `response.usage` | `$0.00` for the turn | the turn cost `$4.46` |
| a manifest of an export | self-consistent | the export was truncated |
| **this repo's own leak detector** | **`"leaks": []`** | **14 leaks; `git grep -E` had exited 128** |

The common shape: **a negative result from an instrument is only evidence if
the instrument can be shown to have run.** Every check here that can return
"nothing found" should be able to fail loudly and should be able to prove it
looked. `scan_leaks` now raises on any git failure and returns
`file:line:text` rather than a bare list length, and
`test_abs_harness.py` asserts it finds a leak that is known to be present —
a detector that cannot be shown to detect is not a detector.

## Two findings worth keeping, beyond the harness

**The design track's one verified win classifies as `trade`, not `win`, and is
not kept.** Its real numbers -- 8.6% fewer cycles at 16x16x16 for BRAM18K
42 -> 98, +20% LUT, +40% FF -- fail this objective's resource constraint on
`tinytpu_isa`. That is a second opinion on the decision the project has been
circling, derived from first principles rather than from the first opinion, and
nothing was tuned to produce it (`test_abs_harness.py` phase `h` asserts it).
Whether it is ultimately the right call is a separate question; that it is
independent is the point.

The reasoning for **cycles scored, resources constrained** rather than a
weighted sum is in `objective.py` and is worth repeating here in its honest
form: **a weighted score needs a price for a BRAM in cycles, and without
place-and-route there is no defensible number.** A made-up one silently decides
every trade-off in the search. A constraint needs no exchange rate, composes
with the other gates, is checkable by a reader ("fewer cycles at no more than
+10% of any resource, on every design case"), and cannot be gamed by letting a
big win on one design case pay for a regression on another.

**Three measurement failures, all the same disease.** Each was found by
measuring rather than accepting:

1. `main` has **seven** pre-existing test failures in the fast tier, not the
   three that were reported to me. An inherited count would have shown four
   phantom regressions on every candidate.
2. A hand-recorded baseline showed **eleven regressions on an unmodified
   tree** -- every one an artifact of the sandbox's read-only checkout, not of
   any candidate. The fix is to record every baseline from inside the same
   sandboxed path a candidate takes, not to special-case the eleven.
3. CHIA's tool server defaults to port 8000, another track held it, the server
   never bound, **and the LLM call still returned**. The agent spent two
   iterations with no tools and no error. Silent, plausible, and it consumes
   budget. General rule, now enforced: **a harness that cannot prove its tools
   are reachable has no business starting a paid run.**

## Two things about money that anyone building on this harness must know

**1. `response.usage` reports $0.00 for calls that cost real money.** Measured
here: `[iter1] $0.00, ? turns, 1488s` for a turn that cost **$4.46** in
opencode's database. Any budget logic, cap, or cost report that reads `usage`
is blind to exactly the long, expensive calls it most needs to see. **Read the
database, never `usage`.** `spend.spent_since` reads the database and is live:
measured moving $0.62 over 45 seconds with a call in flight.

**2. A per-run cap must sum only that run's own sessions.**
`chia_agent/spend.py:spent_since(t0)` sums every opencode message on the
ACCOUNT since `t0`. Its docstring says over-counting is "the safe direction"
— **that sentence is true for one track at a time and false for two, and it
should be corrected where it sits.** With two tracks concurrent it does not
merely over-count, it stops the WRONG RUN: this pilot's counter read **$15.82
against a $15 cap, of which $4.46 was this run** and the rest was the
co-design track's two workers. `Budget` here now sums only sessions this loop
has seen return; the CUMULATIVE cap stays account-wide, which is correct,
because the account is shared. The heavier-spending track is the more exposed
one — to a premature stop, not to an overrun.

## Running two tracks on one host

Measured tonight, all three the hard way:

1. **Never `ray stop`.** It matches Ray processes by name across the whole
   host, so it kills every track's raylet, not just the one it is run from.
   It cost the co-design track a paid run. This harness's teardown is
   `ray.shutdown()`, which disconnects this driver only; there is no `ray stop`
   anywhere in it.
2. **Name your own cluster -- and know that naming it does not protect it from
   a `kill`.** On 2026-09-22 this track's Ray head was removed mid-session by
   the coordinating agent, which harvested processes in `chia_env` older than
   ~50 minutes, filtered by environment name and age rather than by owner, and
   read this track's head as an orphan of a finished search. It cost one
   launch at $0. It was an operator error, not a hazard of the environment,
   and it is the same host-global mistake as `ray stop`: matching by name
   rather than by owner. Kill only by the PID of a process whose owner is
   confirmed. Separately,  `ray.init(address="auto")` reads the host's
   newest GCS address file and will attach to another track's cluster. Start a
   head on a distinct port and pass `CHIA_RAY_ADDRESS`.
3. **Name your own tool ports.** CHIA's tool server actor reads
   `CHIA_TOOL_BASE_PORT` from the **worker** environment and defaults to 8000,
   which another track already held. It never bound, and an LLM call with no
   tools still returns — the agent spent two iterations making no edit. The
   loop now sets the port range in `runtime_env` and **probes its own tool
   surface before any model call**, refusing to search if it does not answer.

Also worker-environment, not driver-environment: `PYTHONPATH` (so the modules
reused from `chia_agent/` are importable when unpickled) and `PATH` (so
`opencode` is found). All four go in `runtime_env["env_vars"]`.
