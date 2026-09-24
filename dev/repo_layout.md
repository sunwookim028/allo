# Repository layout: designs, tools, core

Decision 2026-09-24: the fork will host several accelerator designs — TinyTPU,
MiniTPU, EVA — and several flows — Vitis HLS, the ASIC/PD flow, the SystemC
emitter, the ACT mapping compiler, the CHIA agentic loop. The layout must match
how upstream separates **example designs**, **tools**, and **core IR/compiler
passes**, and push-button scripts and docs must assume it.

## What is wrong today

`examples/feather/` is three plain Python files: a design expressed in Allo.
`examples/accelerator/tinytpu_vitis/` was a whole project — `ip/`, `act/`,
`asic_synthesis/`, `chia_agent/`, an RTL exporter, reports and a reproduce
script. Two problems follow:

1. **TinyTPU sits a level deeper than every other example** for no reason a
   reader can infer, so `examples/` no longer lists the designs. *(Fixed
   2026-09-24 by step 2 of the sequencing below.)*
2. **The flows live inside one design's directory.** The ASIC flow, the ACT
   mapper and the CHIA loop are not properties of TinyTPU, but a second design
   cannot reach them without importing through `examples/accelerator/
   tinytpu_vitis/`. This is the binding problem: MiniTPU and EVA are arriving,
   and the flows would have to be copied or cross-imported.

Also, `chia_runs/` sits in the repository root with nothing tracked in it —
scratch output in a place that implies it is part of the project.

## Target layout

Revised 2026-09-24 after reading upstream's actual conventions. The first draft
proposed a `tools/` root; upstream has no such thing, and inventing a root for
work that has a natural home is how a fork drifts from the project it tracks.

Upstream's conventions, as they actually are:

- **`allo/`** is the compiler package. `allo/backend/` is *one module per
  target* (`catapult.py`, `hls.py`, `llvm.py`, `tapa.py`, `vitis.py`, `xls.py`,
  `aie/`). Alongside it sit non-backend subpackages — **`allo/autoscheduler/`**,
  `allo/frontend/`, `allo/harness/`, `allo/ir/`, `allo/library/`,
  `allo/primitives/` — so a search or analysis component does *not* have to be
  a backend to live in `allo/`.
- **`examples/`** holds designs.
- **`scripts/`** holds setup and infrastructure, and already contains
  `act-test-recipe.sh`.

So:

```
allo/
  backend/systemc.py      the SystemC emitter, beside catapult.py
                          (with mlir/lib/Translation/EmitSystemC.cpp)
  backend/asic.py         the AAAH-facing half: emits the architectural
                          manifest that PD planning consumes. DOES NOT EXIST
                          YET -- see 'two PD flows' below
  act/                    the mapping compiler: spec schema, validation,
                          mapping search, the tiered judge
                          -- precedent is allo/autoscheduler/, not backend/

chia/                     the agentic loop, at the root

backend/asic/             the PD flow, vendored: mflowgen nodes, ADK
                          definitions, and tools/ -- stubs, preflight,
                          extractor, number checker. Landed 2026-09-24;
                          `scripts/asic/` in the first draft, revised on the
                          flow maintainer's judgement -- see 'Vendoring'.
                          A design's construct graph stays with the design,
                          because it names that design's top module.

examples/
  feather/                (exists)
  tinytpu/                <- examples/accelerator/tinytpu_vitis/
    ip/                   its units
    act/                  its ISA binding and workload corpus
    asic/reports/         its synthesis results
    reproduce.sh
  minitpu/  eva/  systemc/
```

**Why each landed where it did.**

**The SystemC emitter is a backend**, unambiguously: it emits for a target, and
`catapult.py` — the flow it feeds — is already a sibling. No argument needed.

**ACT is not a backend and is still part of the compiler.** It does not emit for
a target; it searches a mapping space and grades the result, which is what
`allo/autoscheduler/` does. That precedent is the whole justification for
`allo/act/` rather than `allo/backend/act.py`.

**CHIA is at the root because it is not Allo.** It drives the compiler, the
flows and the gates; it spends real money; it has its own guards and sandbox.
Putting it inside `allo/` would imply that importing Allo imports an agent
harness. It is the one component that genuinely earns a new root.

**There are two PD flows, not one, and they sit on opposite sides of the
question.** This was established by reading the flow's own node library rather
than inferring from its name.

- **The flat flow** — the one every number we have came from — consumes exactly
  a directory of Verilog plus `sv2v_manifest.f`, entered at the design
  collector. Nothing Allo-level. **Infrastructure: `scripts/asic/`.**
- **AAAH** ("Allo-Aided ASIC Harness") consumes Allo-level architecture, and
  **not through files**. Its compilation node loads an Allo design and *calls*
  `build(project, target, mode, configs)` with
  `configs["asic_manifest"] = {...}` — Allo is asked to **emit an architectural
  manifest as part of compilation**. Downstream planners then read
  `semantic_id`, `macro_class_id`, `members`, `pe_instances`, `rtl_ports` and
  `rtl_module` maps, `stream_bundles` with per-stream width, direction and
  endpoints, `pid` with x/y, `anchor_kernel`, `control_pins`, `protocol`. Macro
  planning selects RTL-equivalence classes by *semantic identity* and validates
  ports by ordinal position, because equivalent generated modules use different
  port names; physical-intent placement lays macros on a grid from PID row and
  column deltas, weighting edges by **pre-HLS stream bit width**. None of that
  survives in emitted Verilog. **This is a backend: `allo/backend/asic.py`,
  beside `catapult.py`.**

Do not force one answer onto both. The flat flow is infrastructure and the
AAAH-facing half is a backend, and they can coexist.

**The blocking gap: our Allo does not emit that manifest.** `asic_manifest`
appears nowhere in our `allo/`. The interface is satisfied by a different
lineage of Allo, which is why the TinyTPU path was routed *around* the
compilation node in the first place. So adopting AAAH is not a directory move —
**it needs the manifest emitter implemented here**, and until it exists those
nodes cannot run against our designs at all.

We are unusually well placed to write it. `ip/compose.py` already holds an
`Architecture` of units, channels and memories, with per-channel widths and
explicit endpoints — which is most of what the manifest asks for. The emitter
is largely a serialisation of a structure we already build, not a new analysis.

**And there is a ceiling worth stating before anyone promises too much.** Even
with a perfect manifest, DC at the flattening effort these runs use dissolves
the boundaries again unless synthesis is told to preserve them — and changing
that is exactly what would make new runs non-comparable with the four we have.
**The manifest buys planning and attribution, not automatically a
hierarchy-preserving netlist.**

**The flat flow goes under `scripts/` because it is infrastructure, not
compilation.** It consumes RTL that Allo already emitted and runs external EDA
tools; nothing imports it. `scripts/` is where upstream keeps exactly this kind
of thing, and `act-test-recipe.sh` shows the fork has used it that way already.

**Results stay with the design, the flow does not.** `examples/tinytpu/asic/
reports/` holds that design's numbers; `scripts/asic/` holds the machinery every
design shares. This resolves the tension that made `tools/` tempting: the thing
that must be shared and the thing that must not are different things.

**Two corrections to the first draft, both from the same test** — *does this act
on designs, or is it a design?*

- `examples/systemc_rtlsim/` is **not a design**. It is a cross-check harness
  (`mgc_shim.v`, `ref_xsim/`, `run_mulacc.sh`, `REPRO.sh`) validating SystemC
  output against RTL simulation. *First draft said: move it beside
  `examples/systemc/`. That was wrong for the same reason — `examples/systemc/`
  was **also** not a design directory.* Applying the test to both (2026-09-24):
  the Allo designs stay in `examples/systemc/`, the harness and testbenches
  become `tests/systemc/` (cross-check under `tests/systemc/rtlsim/`), and the
  logs, verdicts and archived emitter output become `dev/records/systemc/`. The
  one design that was buried in the harness directory, `pe_split.py`, moved up
  to `examples/systemc/`.
- `rtl_export/` was a bad name for something that is not a separate tool.
  `export_rtl.py` packages a configuration's Verilog *for the ASIC handoff* —
  its own docstring says so. It is a stage of the PD flow:
  `scripts/asic/export.py`.

## The honest caveat about "generic"

Not all of today's tooling is design-independent, and pretending otherwise
would produce a `tools/` that only one design can call. The ACT mapper knows
TinyTPU's ISA; the ASIC flow's construct graph names its top module; the CHIA
loop's gates run TinyTPU's stress suite. **The split must be made by moving
what is genuinely generic and leaving a design-specific shim behind**, not by
moving directories wholesale and adding parameters until it compiles.

The test for each piece: *could MiniTPU call this without editing it?* If the
answer needs a caveat, the piece is not ready to move, and saying so is the
correct outcome for that piece.

## Sequencing

This move renames paths that push-button scripts, docs and another machine's
flow all depend on, so it must not run concurrently with work in flight.

1. **Wait** for the cycle-row doc update and the incoming ASIC-flow commits.
2. **Move designs first** — `examples/accelerator/tinytpu_vitis/` →
   `examples/tinytpu/`. One rename, references updated, gates re-run. Low risk,
   and it makes `examples/` list the designs again. **Landed 2026-09-24**:
   `examples/accelerator/` is gone and `examples/` lists the designs again.
3. **Then extract tools**, one at a time, each with the "could MiniTPU call
   this?" test applied and recorded. `tools/asic/` first, because a second
   machine is already committing into it and the sooner its home is stable the
   fewer redirects it takes.
4. **Then add the new designs** — MiniTPU, EVA — against the settled layout
   rather than into a moving one.
5. `chia_runs/` leaves the repository root.

## Docs that follow from it

Per `dev/docs_style.md`, each tool gets a page with a **Quick start** before any
explanation: the ASIC flow (preflight, the documented sequence, what is
committed and what stays on scratch) — still owed; the quick start currently
lives in `examples/tinytpu/asic_synthesis/README.md` — `tools/chia` (how to run it, what it costs, what the
guards are — the existing page is about what it *found*), `tools/act`, and the
SystemC emitter, which is currently only reachable through the Catapult page.
Each design gets a page that says what it is, how to run it, and what it
reproduces — with history moved to the results page.

## Open input needed

- **The SystemC-emitter repository's location**, to merge EVA. Its owner is
  reported to be content with the merge; the URL and the intended history
  treatment (merge with history, or import as a subtree) are not yet known here.
- Whether MiniTPU is imported or referenced. It lives in another engineer's
  tree today and is read-only from this side.

## Vendoring: the nodes yes, the ADK no

Settled 2026-09-24 on the flow maintainer's judgement, which is better than the
blanket "vendor nothing" this file previously carried. They are different cases
and treating them as one was the error.

**Vendor the 36 mflowgen nodes.** They *are* the flow. Without them our
construct scripts describe a graph whose steps live in a third-party GitHub
repository, at an unpinned commit, on an account we do not control. A preflight
that merely points at it is adequate for a colleague on the same machine and
inadequate as the reproduction record for a published number: if that repository
moves, is rewritten or disappears, **every area figure in our docs becomes
unreproducible, and the failure is silent until someone tries.** It is 1.8 MB of
Apache-2.0 code we are entitled to redistribute.

**Do not vendor the ADK.** It is 7 MB of third-party PDK whose licence we have
not checked, the ADK node downloads it at run time anyway, and the specific view
it contains is unusable by DC. Instead, **have the preflight verify the fetched
library by the `stdcells.db` md5 that the settings snapshot now records.** That
is reproducibility without redistribution, and it is strictly better than
either extreme.

If the no-vendoring plan is ever restored for the nodes, the preflight must
**pin and verify** the upstream commit rather than check for presence. An
unpinned pointer is the thing that bites.

**Verified 2026-09-24, not assumed.** `allo/backend/asic/adks/` holds seven
files and 28 KB: `freepdk-45nm/` with `configure.yml`, `adk-overlay.tcl`,
`vcs-compile.args`, `vcs-bagl.args` and a README, and `skywater-130nm/` with
`configure.yml` and a README. No library data — no `.db`, `.lib`, `.lef`,
`.tf`, no `pkgs/`, no view. And not only in the tree: every blob that ever
existed under that path across the whole imported history is one of those seven
files, so the payload was removed from the history as `PROVENANCE.md` claims,
not merely deleted in a later commit. The decision stands as recorded.

## Two results from building the extractor

Both are the kind of thing that only appears when a number is generated rather
than transcribed.

**The standard-cell library is now confirmed identical across all seven runs**,
`stdcells.db` md5 `f5560259` — previously assumed. And the clock port is
recorded as the SDC actually constrained it, read out of `design.sdc` rather
than from a parameter claiming it: `clock` for the comparison design, `ap_clk`
for ours.

**Worst slack must be the minimum across path groups, not the first one
reported.** Taking whichever appeared first would have given **+1.10 ns instead
of +0.20 ns** for the T=8 design. Our published figures happen to be the correct
minimum, so nothing is wrong — but the naive extraction would have been wrong by
0.9 ns and looked entirely plausible. The extractor records every group's slack
alongside the minimum so the choice is visible rather than implicit.

## Known defect to fix during the move

`asic_synthesis/construct-commercial.py` resolves its node library and ADK at
`examples/accelerator/{nodes,adks}/`, which do not exist in this repository, so
it **cannot run from a clean checkout today**. Wherever the vendored nodes
land, that path has to match. **Resolved 2026-09-24.** The vendored flow landed
at `allo/backend/asic/{nodes,adks}/` and the construct script was pointed at
it — but by three `dirname` calls from `asic_synthesis/`, which under
`examples/accelerator/tinytpu_vitis/` reached `examples/`, not the repository
root, so it still could not find the nodes. **A re-count was then tried and was
also wrong** — the same defect class twice in a row — and the fix that holds
is `a34e1346`: **search upward for `allo/backend/asic/nodes`** rather than
count directories at all. It works at any depth and survived the design rename
with no edit. Note also that the construct graph and `make_stubs.py` were
committed by a different session than the one that produced the reports;
attribute them accordingly when moving.

**The rule, because this has now cost three fixes in two days.** Never derive a
repository root by counting `..` or `parents[n]`. Search upward for a marker,
or take the path as an argument. The TinyTPU rename turned up roughly thirty
`ROOT`/`REPO` values derived by counting, **three of them already wrong** and
made correct only by accident of the move — a count is a latent break in any
tree being reorganised, and this one is mid-reorganisation.

**The same defect, second instance, fixed 2026-09-24.** `reduce_asic.py`
generates its own construct graph, and that template hardcoded
`~/allo-asic` — a path that exists on the synthesis host and nowhere else, so
the committed `asic_reduce/reduce_8_2/construct-reduce.py` could not run from a
checkout at all. It now finds the flow the way `construct-commercial.py` does,
by searching upward, with `ALLO_ASIC_FLOW` for the case the generated file is
copied outside the tree — which is exactly what the `/scratch` builds on
zhang-21 are. Both the template and the committed generated file were changed
together and are byte-identical.

Both graphs were **built**, not merely read: against a stub `mflowgen`, each
resolves its four nodes and the `freepdk-45nm` ADK and returns a graph. A copy
of `construct-reduce.py` placed outside any checkout fails with its own message
and builds once `ALLO_ASIC_FLOW` is set. The tools follow the same rule: their
`REPO` — which only feeds the `--docs` default and printed paths — is found by
the same upward search.

## The tools folded in, 2026-09-24

`check_numbers.py`, `extract_results.py`, `make_stubs.py` and `preflight.py`
moved from `examples/tinytpu/asic_synthesis/tools/` to
`allo/backend/asic/tools/`. **Results stayed**: `asic_synthesis/reports/` is
TinyTPU's, and so is `construct-commercial.py`, which names its top module.

Each tool now takes the design on the command line instead of deriving it from
its own location — `--reports` for the number checker and the extractor,
`--design` for the preflight — which is the same "could MiniTPU call this
without editing it?" test applied one piece at a time. The extractor's
`--check` reproduces all eight committed `results.json` byte-for-byte from the
new home, which is the evidence that the move changed nothing about what the
numbers are.

The preflight came with them and is **not push-button**, so nothing here should
describe it as such: it needs a DC licence and roughly 70 minutes of a specific
machine. What it removes is the hour spent discovering that.

**Which of its branches a licence-free run reached, named rather than glossed.**
On this machine it exercised the node library, the ADK definition, the
variant's RTL and file lists, and the agreement of `stdcells_db_md5` across the
committed snapshots, and correctly reported `dc_shell`, mflowgen and sv2v as
absent. It did **not** exercise the `--build` branch that hashes a fetched
`stdcells.db` and compares it with `f5560259`: without a licence there is no
build tree to hash, so that branch reports as *correctly missing*, which is not
passing. It needs one run on the licensed machine, which the synthesis session
has offered.

**Done, in a separate tool, and the premise needed correcting.** The note here
previously said `results.json` now carries `QD` per run. **It does not** — no
committed `results.json` has a `QD` field, and only one export's
`MANIFEST.json` (`T8_MAXDIM64`, added at `f0ee3223`) records one at all. What
`results.json` does carry is `T`, `MAXDIM`, `TPU_DMA_WIDEN`, the export
directory, the manifest and its md5, and the emitting `allo` commit.

The refusal was built on that real ground instead, as
`allo/backend/asic/tools/check_pairing.py` rather than as an addition to
`check_numbers.py` — the reasoning above was right that associating a figure
with the cycle counts *near it in prose* is not decidable. So nothing is
inferred from prose: `pairings.json` **declares** each pair, and the checker
admits it only on the same committed export (verified by md5) or on four
configuration keys present and equal on both sides. Missing is `cannot pair`.

The first run found that no model-level cycle count is pairable with any
committed area (the exports predate `QD` being recorded) and that
`T8_MAXDIM64`'s area is orphaned by its own re-export.
