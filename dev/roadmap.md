# Roadmap to the end state

The end state, as the owner stated it on 2026-09-24:

> one branch; every tool integrated; Julian's `allo-asic`, choonsik1's EVA and
> the SystemC backend all integrated; repo layout upstream-consistent; headline
> numbers for the example designs and flows — including the ACT track —
> reproducible push-button; CHIA experiments reproducible from this repository
> alone; licensing consistent with upstream practice.

This file tracks what closes each of those and what it is waiting on. It is the
durable record: anything not written here is lost when a session ends.

**On the ETAs.** They are given in *agent-sessions* (one focused agent, working
to a written brief) plus wall-clock where an external tool dominates. They
assume no usage limit interrupts — an assumption that failed roughly six times
on 2026-09-22/23, each costing a partial re-run. Treat them as ordering
information, not as promises.

---

## A. One branch

**Done.** `main` plus three deliberate exceptions, each with a stated reason:

| branch | why it exists |
| --- | --- |
| `big-shapes` | pending the `QD=16` re-measure — **in flight** |
| `upstream-omp-team-size` | an upstream contribution candidate, kept separate on purpose |
| `wip-snapshot-20260922` | holds a prompt-audit finding that exists nowhere else |

Nine branches were merged or archived as annotated tags; nine more that were
fully merged were deleted. Feature branches created by the agents below are
expected to land and disappear.

**Remaining:** `big-shapes` lands or is archived when its re-measure returns.
`wip-snapshot-20260922` is deletable once the docs restyle lands its finding in
the CHIA page. `upstream-omp-team-size` stays until it is contributed upstream.

**ETA:** both closable within the two agent-sessions already running.

## B. Integration

| component | state | ETA |
| --- | --- | --- |
| SystemC emitter + EVA | agent running: merge `SystemC-emitter`, then evaluate 10 divergent commits from `systemc-ip-integration` | 1–2 sessions; the C++ may need a rebuild to verify |
| Julian's `allo-asic` | requested from the synthesis session, which has the checkout and a standing budget | 1 session on their side |
| ACT | already in-tree; moves in §C | with the layout move |
| CHIA loop | already in-tree; see §E | — |

**The one real unknown is the EVA merge**: two candidate branches diverge by 10
commits each in the emitter and the IR builder, so some cherry-picks will be
left for the author to confirm. That is the designed outcome, not a failure.

**A month of divergence in `mlir/`** means the emitter may not be verifiable
without a rebuild. If so, the honest report is what could and could not be
checked.

## C. Layout, upstream-consistent

Target and reasoning in `dev/repo_layout.md`. Ordered by dependency:

1. **Split `ip/compose.py`** — `Unit`, `Channel`, `Memory`, `Architecture` are
   generic and go to `allo/`; `ip/tinytpu.py` stays with the design. *This is
   first because it is what makes §B's manifest emitter possible.*
2. **`examples/accelerator/tinytpu_vitis/` → `examples/tinytpu/`** — one
   rename, references updated, gates re-run. **Done 2026-09-24.**
3. **`allo/backend/asic/`** — both entry points: AAAH as the Allo-facing mode,
   the flat flow as the control mode, kept permanently. **Done** (`fdb262cc`):
   the four tools live in `allo/backend/asic/tools/`, each taking the design as
   a required argument; reports stay with the design. The ADK directories were
   verified **configuration only** -- seven files, 28 KB, no library data, and
   every blob that ever existed under that path across the imported history is
   one of those seven, so the payload was stripped from history rather than
   deleted in a later commit.
4. **`examples/systemc_rtlsim/` split by kind** — **Done 2026-09-24.** Neither
   SystemC directory was a design directory. The Allo designs (including
   `pe_split.py`) are now `examples/systemc/`; the testbenches, shims and run
   scripts are `tests/systemc/` (with the cross-check under
   `tests/systemc/rtlsim/`); the logs and archived emitter output are
   `dev/records/systemc/`.
5. `chia_runs/` leaves the repository root.

**Blocked until the agents working inside `examples/tinytpu/` finish** — the re-measure, the T8 re-export and the `ip-gap` verification all
write there, and renaming under them would destroy their work. This is the one
genuinely sequential dependency in the whole plan.

**ETA:** 2 agent-sessions once that tree is quiet.

**Known defect, fixed by the move:** `construct-commercial.py` looked for its
nodes and ADK at `examples/accelerator/{nodes,adks}/`, which do not exist. The
vendored flow now lives at `allo/backend/asic/{nodes,adks}/`, and the script's
three `dirname` calls from `asic_synthesis/` reach the repository root only from
`examples/tinytpu/` — one level shallower — so it runs from a clean checkout.

## D. Push-button headline numbers

A headline claim counts only when **a gate enforces it**, not when a page states
it. **Six of eight are enforced**, up from four. The headline here said six
before the workloads and end-to-end gates existed, which its own table did not
support -- it listed four. The two that are still not enforced are named rather
than counted: **CHIA**, whose harness exists and whose repair is unfinished,
and the **ASIC run**, which has a preflight and needs a Design Compiler
licence. Neither is an agent-session away.

*Known-failing tests, so nobody chases them.* `pytest tests/` does not collect
on this host: 25 errors, all `tests/dataflow/aie/*`, no `aie` module. With that
directory ignored: **800 passed, 64 skipped, 2 xfailed, 7 failed** — and those
same seven fail on a clean `main`, so they are pre-existing and unrelated to
anything recent: `test_hierachical_mesh::test_2x2`, three in
`ip_integration/test_external.py`, `test_builder::test_minmax_cast`, and two in
`test_stateful.py`.

| flow | claim | gate | state |
| --- | --- | --- | --- |
| design | reproduces the published row | `reproduce.sh` | **enforced** |
| ACT | every encodable mapping verified, 12/12 | `act_compile.py --gate` | **enforced** |
| numbers | every published area figure traces to a report | `check_numbers.py` | **enforced** |
| RTL | cycles bit-exact | inside `reproduce.sh` | **enforced** |
| CHIA | the loop runs, guards hold, $0 | `test_harness.py` | exists; **repair unfinished** |
| workloads | which models are **verified**, which only **executed** | `workloads/gate.py` | **enforced**, 2026-09-24 |
| ASIC | area + timing floor, one flow both sides | preflight + sequence | preflight **landed**; a run still needs a licence |
| **end-to-end** | **PyTorch → mapping → cycles → area** | `e2e_gate.sh` + `check_pairing.py` | **local tier enforced**; area tier needs a licence |

**The end-to-end gate is not one command, and must not be written as though it
were.** Every error in this work has lived in the seams *between* flows — a
configuration claimed but not run, a row measured on a different design, an
area figure beside cycles from other RTL — and no single-flow gate can see any
of them. `examples/tinytpu/e2e_gate.sh` runs the local tier in ~36 s (model →
specs → mapping → verified cycles, then the join, then the area/report trace)
and *prints* the remote tier rather than claiming it, because area needs a DC
licence.

**The join is the part that matters**, and it is enforced with or without a
licence. `check_pairing.py` admits a cycle/area pair only when both sides cite
the same committed export (verified by the manifest md5 the run recorded) or
both declare all four of `T`, `MAXDIM`, `QD`, `DMA_WORDS` and they agree. A
missing key is *cannot pair*, never *matches*.

Applying it produced two findings on the first run:

- **No model-level cycle count can be paired with any committed area.** The
  workload numbers are `QD=16`; every committed TinyTPU export predates
  `63ee6ec7` and records no `QD`. The end-to-end claim reaches cycles and
  stops, and `pairings.json` records that as a refusal with its reason.
- **`T8_MAXDIM64`'s area is orphaned.** It was synthesised on 2026-09-22 from
  the export at `7a3c2a17` (manifest md5 `6e9b2596`); `f0ee3223` re-exported
  that variant against QD=16 main on 2026-09-24, so the RTL in the tree is not
  the RTL that area describes. One DC run on the current export clears it.

Both gates test their failure paths: `tests/act/test_gates_negative.py`, 23
constructed refusals, each asserting the gate says *why*.

The ASIC row is **not** push-button and must not be written as though it were:
`preflight.py` checks `dc_shell`, the pinned mflowgen, sv2v, the vendored nodes,
the ADK definition, the variant's RTL and lists, and the fetched `stdcells.db`
against its recorded md5 — and then prints the sequence. A run still needs a DC
licence and ~70 minutes of a specific machine. Its value is failing in seconds
rather than an hour in.

**ETA:** both landed 2026-09-24. What remains is not agent work: one DC run
on the current `T8_MAXDIM64` export, and a re-export of `T4_MAXDIM64_shipped`
at a recorded `QD` plus one DC run on it, after which the model-to-area
refusal above becomes an admissible pairing and the gate will say so itself.

## E. CHIA reproducible from this repository alone

**Blocked on the harness repair**, which was killed twice by usage limits. Four
known blockers, from `dev/`:

1. `MAIN_BASE` in `evaluate.py` goes stale — wrong three times in two days.
   Derive it, or make staleness fail loudly.
2. `isa_dsl.py` fails its own spec policy. **Do not widen the dunder allowlist
   unilaterally** — it is a security guard on agent-authored code; this needs a
   human decision.
3. `check_invariants` is pinned to `MAXDIM=16`.
4. `evaluate.FROZEN` needs the ISA spec artefacts.

The bar is not that the harness's own tests pass — it is that **the loop runs
end to end on current `main`**, or that every step short of a paid call is
verified with the unverified step named.

**ETA:** 1–2 sessions. Blocker 2 needs an answer first.

---

## Standing hazards

Recorded because each cost real work today.

- **Never run a tree-wide git operation in a checkout another agent is writing
  to.** This destroyed an agent's work once and nearly a second time.
- **Remove a worktree when its agent finishes, not when the disk fills.** Each
  costs 0.6–1.9 GB; five live agents accrue ~2 GB.
- **`/tmp` is 15 GB and shared.** When it fills, the harness cannot create its
  output-capture file, so *every* command fails including the escape hatch, and
  the only way out is a terminal outside the tool.
- **The configuration trap**: the worktree default is not `MAXDIM=16`, and
  `TPU_QD` now defaults to 16. State the variables beside every number. This
  produced two published mistakes and two near-misses in two days.
- **The `allo` conda env's editable install pointed at a removed worktree.**
  Repointed to `/home/sk3463/allo` on 2026-09-24. It broke silently: `import
  allo` still worked from inside a checkout and failed everywhere else, so it
  surfaced only when an agent ran a docs build from its own worktree. Removing
  a worktree is not free if anything outside git references it.
- **Test a check's failure path, not only its pass path.** A check that has
  only ever said ok proves nothing. The ASIC preflight's library-checksum
  branch was confirmed by copying a real `stdcells.db`, appending one null
  byte, and pointing the preflight at it: it reported the mismatch, named both
  digests, said the areas were not comparable with the committed set, and
  **refused to print the run sequence**. That is discrimination; a passing run
  alone would not have shown it. The same tool has one branch still untested --
  what it does when committed snapshots disagree about the library -- for the
  good reason that there is no second library to test it with, and that is
  recorded rather than glossed.
- **Subagents do not reliably have `ListAgents`.** An agent told to announce
  its path claim to its peers could not see them, and guessing at names
  returned "no agent reachable". Pass peer agent IDs explicitly in a brief
  rather than instructing an agent to look them up — and when an agent cannot
  reach a peer it should say so and let the dispatcher relay, not guess.
- **The session scratchpad is shared between concurrent agents.** Two agents
  writing `reproduce.sh` output to the same scratchpad path truncated one
  another's log mid-cosim; the verdict survived only because the summary block
  happened to be contiguous at its own offset. An agent that reads a truncated
  log sees a run that did not finish, or worse a run that appears to have
  finished differently. Give every agent a distinct path, and prefer a
  worktree-local file to a shared scratch directory.
- **Never resolve a repository root by counting levels.** Search upward for a
  marker, or take the path as an argument. Hit three times in two days: a
  construct script counting three `dirname`s landed on `examples/` rather than
  the root and could not find its node library from a clean checkout (and was
  *announced fixed without being run* -- a different check was run and taken as
  coverage); the TinyTPU rename found ~30 `ROOT`/`REPO` values derived by
  counting, **three already wrong** and made correct only by accident of the
  move; and the first fix was itself a re-count that happened to suit the new
  layout. A relative path that encodes tree shape is a latent break in any
  repository being reorganised, and this one is mid-reorganisation.
- **A licence-free check cannot report a licensed branch as passing.** The
  ASIC preflight's ADK-checksum branch reports as *correctly missing* without a
  licence, which is not the same as passing. Say which branches a run could not
  reach rather than reporting the run as green.
- **Read a gate's output, never its exit code**, and confirm what ran is what is
  being claimed. Four instruments were caught reporting success without having
  run; three further checks ran against the wrong object and passed.
