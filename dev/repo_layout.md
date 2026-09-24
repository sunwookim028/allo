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
  one design that was buried in the harness directory, `dot_product_four_links.py`, moved up
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

## Audited 2026-09-24: what the reorganisation did NOT catch

Checked file by file against the target above, after the moves landed. The
three that did land are verified done: the design rename, the SystemC split
(designs in `examples/systemc/`, harness in `tests/systemc/`, logs in
`dev/records/systemc/`), and the four PD tools into
`allo/backend/asic/tools/`. `chia_runs/` is gone from the root.

What has not moved, worst first:

1. **CHIA is still inside a design, and it is in two places.**
   `examples/tinytpu/chia_agent/` is the loop the doc allots the root `chia/`,
   and `chia_abstraction/` at the root is a second CHIA code body. Neither is
   at `chia/`. `chia_abstraction/evidence*` and `baseline*` are records;
   `chia_abstraction/test_abs_harness.py` and
   `examples/tinytpu/chia_agent/test_{harness,codesign}.py` are tests.
2. **ACT's generic core is an unjustified new root.** ***Done 2026-09-24,
   second pass, at the owner's direction*** ("maybe `act/` can be under
   `allo/`"). `act/` -> `allo/act/`, on the `allo/autoscheduler/` precedent.

   **The price, measured and recorded rather than absorbed.** `allo/act/` is a
   subpackage of `allo`, so importing it runs `allo/__init__.py`, which imports
   the compiled MLIR bindings. `act/` at the root did not. Two consequences,
   both real:

   * **`pytest tests/act` no longer runs in a worktree with no build.** It
     errors at collection instead. `dev/toolchains.rst` used to offer this as
     the thing you could still exercise in an unbuilt tree; it now says the
     property is gone rather than simply not mentioning it.
   * **`act_compile.py --gate` went from ~1.3 s to 4.2 s**, all of it the
     `import allo`. **This matters for CHIA**, where that gate runs once per
     candidate: a three-second constant on every candidate of every sweep.

   Four pages claimed the old property and now state the new truth explicitly:
   `CLAUDE.md`, `dev/toolchains.rst`, `docs/source/extensions/act.rst` and
   `docs/source/designs/workload_suite.rst`. A reader who relied on it needs to
   be told it is gone, not to stop seeing it mentioned.

   **A root named `act` was also three tools' marker for "this is a checkout".**
   `tests/act/test_gates_negative.py`, `examples/tinytpu/workloads/gate.py` and
   `examples/tinytpu/e2e_gate.sh` each searched upward for a directory named
   `act` beside `allo/`. All three now look for `examples/tinytpu`. The first
   two failed loudly and were caught the same hour; `e2e_gate.sh` is not in the
   required gate set, so it failed only when someone ran it, with an error
   message naming the *other* marker in its condition. **A marker directory is
   a reference like any other, and it does not show up in a grep for imports.**

   The
   TinyTPU binding (`examples/tinytpu/act_*.py`, `act/corpus/`) correctly
   stays with the design, but `examples/tinytpu/act/{judge,spec,legality,
   submission,calibrate,cycles,correctness,measure,variants,baseline}.py` is
   the validation layer and the tiered judge, which is not design-specific.
   **That half has NOT moved** -- see "What the second pass could not do".
3. **`examples/tinytpu/` is mostly not a design.** Of 33 top-level files about
   six are. Harnesses and gates (`bench_isa.py`, `stress_isa.py`, `cosim.py`,
   `isa_ref.py`, `kpn_model.py`, `mutate.py`, `mutate_actions.py`,
   `e2e_gate.sh`, `reproduce_codesign.sh`, `workloads/gate.py`,
   `act/rules_test.py`, `act/rtl_hang.py`) belong in `tests/`; measurement
   sweeps (`csynth_sweep.py`, `latency_grid.py`, `parity_sweep.py`,
   `reduce_csynth.py`, `reduce_latency_probe.py`, `impact/`) are tooling; and
   `export_rtl.py`, `export_gemmini_rtl.py`, `saif_capture.py` and
   `reduce_asic.py` are PD-flow stages the doc already names
   (`allo/backend/asic/`, since that is where the flow actually landed -- the
   doc's `scripts/asic/export.py` is now stale).

   **PREREQUISITE for the harness half, recorded so it is not rediscovered.**
   The gates cannot move by moving, because the *design* imports them:
   `cosim.py` imports `stress_isa` for the `TPU_TB=stress` testbench, and
   `gen_isa.py` imports it "only for its random-program generator". Moving
   `stress_isa.py` to `tests/` makes two design entry points import from
   `tests/`, which is worse than leaving it. **The prerequisite is extracting
   those two pieces -- the random-program generator and the stress testbench --
   out of the gate and into modules the design can own.** Only then is the
   remainder a move. (`isa_dsl.py` and `act_compile.py` import
   `bench_isa.SHAPES`; that one is free, being a redirect to the `shapes.py`
   that already exists.) It is a refactor with a real risk of changing
   behaviour, and it must not ride along on a placement pass.
4. **Generated output and raw dumps sit in `examples/`.** ***`rtl_handoff/`,
   `gemmini_rtl/` and `csynth_reports/` done 2026-09-24, second pass; the
   `impact/probe_shared/` and `examples/eva/generated/` halves remain.***
   `examples/tinytpu/rtl_handoff/` and `gemmini_rtl/` were exporter output;
   `csynth_reports/` is 21 raw Vitis `.rpt` files of the same class as
   `dev/records/tinytpu/logs/csynth_isa_*.rpt`; `impact/probe_shared/` is half
   of an experiment whose other half is already in
   `dev/records/tinytpu/impact-results/`. `examples/eva/generated/` is emitted
   SystemC, which `examples/systemc/README.md` already rules on ("output, not
   source"); three of its files are byte-identical to the archived copies in
   `dev/records/systemc/`.
5. **`agents/`, `devtools/` and `playground/` have no slot at all.** ***Done
   2026-09-24 for the first two; `playground/` deliberately left.*** `agents/`
   is two live Allo designs plus a dated working note (its designs ->
   `examples/`, the note -> `dev/`); `devtools/` is twelve standalone
   introspection scripts whose own README says nothing imports them, which is
   the `scripts/` case; `playground/` is two scripts nothing references.
6. **A Quick start is inside a design.**
   `examples/tinytpu/asic_synthesis/README.md` is 313 lines of flow quick
   start and results -- the page this file has already said is owed.
   `asic_synthesis/reports/` is also at the path the target spells
   `examples/tinytpu/asic/reports/`.
7. **Two working notes are on the published site.**
   `docs/source/extensions/chia_results.rst` §"The Planned Experiments" is a
   budget allocation for the next sessions ("$300 is authorised ... about $500
   remains"), and `docs/source/designs/minitpu.rst` is a distillation keyed to
   another engineer's home directory on this host. `design_space.rst` is a
   weaker case of the same.
8. **The published site sends readers into `dev/`.** `dev/toolchains.rst` is
   cited as the environment reference by seven pages; the directive figures on
   `backends/catapult.rst` live only in
   `dev/systemc/noc/FINDINGS_wire_channel.md`;
   `designs/gemmini_comparison.rst` rests a methodology claim on
   `dev/paper_outline.md`; and `examples/systemc/README.md` sends readers to
   three `dev/systemc/*.md` files as the backend guide. Either those move on
   to the site or the pages stop depending on them.
   `dev/systemc/dataflow_links_examples.py` is two runnable designs in `dev/`.

Two things that look wrong and are not: a design's construct graph stays with
the design (it names that design's top module), and `act/mapspace.py` versus
`chia_agent/mapspace.py` are deliberately separate -- `docs/source/extensions/
codesign.rst` says the co-design loop reuses ACT's interface and none of its
code.


## Executed 2026-09-24, second pass

Against the owner's statement of the principle -- *"not mixing designs and
tools and notes and docs"*, *"'reduce' is a design concept, 'asic' is a
tooling concept, should not mix"*, *"examples/tinytpu has a lot of different
types of artifacts; examples/systemc is just not a good naming convention
there, should be design"* -- so: `examples/` holds designs and is named after
them, tools live in `allo/`, dated measurements in `dev/records/`, reference
pages in `docs/source/`, and a file's name says which it is.

| from | to | why |
| --- | --- | --- |
| `act/` | `allo/act/` | the mapper core is part of the compiler, not a design and not a tool acting on designs |
| `examples/systemc/*.py`, `demos/` | `tests/systemc/` | emitter demonstrations, not designs |
| `examples/systemc/README.md` | merged into `tests/systemc/README.md` | one directory, one README |
| `agents/{eva_blocks,pe_alu,eva_pe_router_split}.py` | `examples/eva/` | EVA design sources, and EVA already had a home |
| `agents/INTERCONNECT.md` | `dev/interconnect_reference.md` | a reference document |
| `agents/README.md` | `dev/records/agent_interconnect_2026-08-15.md` | a dated working note |
| `devtools/` | `scripts/devtools/` | standalone scripts acting on the compiler; nothing imports them |
| `examples/tinytpu/csynth_reports/` | `dev/records/tinytpu/csynth_reports/` | 21 raw Vitis `.rpt` dumps, the class `dev/records/tinytpu/logs/` already holds |
| `examples/tinytpu/rtl_handoff/` | `dev/records/tinytpu/rtl_handoff/` | 13 MB of generated Verilog: a record of what was synthesised, not a design |
| `examples/tinytpu/gemmini_rtl/` | `dev/records/tinytpu/gemmini_rtl/` | the same for the opponent's RTL |

`tests/systemc/test_emit.py` is new: the claims the moved programs make about
the emitted SystemC, collected by pytest. `tiled_systolic.py` was asserting
`AlloMem<`/`AlloMemW<` counts against an emitter that now produces
`AlloMemPins<` three times and neither of the old two, and nothing caught it
because nothing ran it.

### The `examples/systemc/` judgement, and a deliberate divergence from upstream

**Upstream's `examples/` is `aie`, `feather`, `machsuite`, `polybench`,
`torch`, `transformer_hls.py`.** `aie` is a *backend* name, so
`examples/systemc/` had upstream precedent and a fork that tracks upstream
should not drop such a precedent silently. **Recorded here as a deliberate
divergence, with its reason**: the owner asked for design names under
`examples/`, and the stronger objection applies anyway -- the files were not
designs. Five of the seven asserted on the emitted SystemC *text*, which is a
claim about the emitter; `tiled_systolic.py`'s own header calls itself
`tests/dataflow/test_tiled_systolic.py` "run through `target='systemc'`";
`demos/` is "the smallest possible program for one language feature each". So
the question "rename it after which design?" has no answer, and they went to
the harness that already acts on them. `examples/aie/` is untouched: it is
upstream's, it holds designs, and this fork has no standing to rename it.

### What the second pass could not do, and what blocks it

Four findings from the audit above are **not** executed, each for a reason
found by trying.

1. **The harnesses and gates cannot leave `examples/tinytpu/` by moving.**
   Audit item 3 says `bench_isa.py`, `stress_isa.py`, `cosim.py`, `mutate*.py`
   and the rest belong in `tests/`. They cannot go while the *design* imports
   them: `cosim.py` imports `stress_isa` for the `TPU_TB=stress` testbench and
   `gen_isa.py` imports it "only for its random-program generator", so moving
   `stress_isa.py` to `tests/` makes two design entry points import from
   `tests/`. (`isa_dsl.py` and `act_compile.py` import `bench_isa.SHAPES`,
   which is only a redirect to the existing `shapes.py` -- those two are free.)
   Separating them needs the random-program generator and the stress testbench
   extracted from the gate, which is a refactor, not a move. **Blocked on that
   extraction.**

2. **The PD-flow stages cannot fold into `allo/backend/asic/`.** Audit item 3
   names `export_rtl.py`, `export_gemmini_rtl.py`, `saif_capture.py` and
   `reduce_asic.py`. Applying this file's own test -- *could MiniTPU call this
   without editing it?* -- all four fail it as they stand: `export_rtl.py`
   hardcodes `TOP = "tinytpu_isa"` and a `DEST` naming TinyTPU's export
   directory, `saif_capture.py` names `sim/verilog/tinytpu_isa.tcl`, and
   `reduce_asic.py` is about the adder tree specifically. Moving them wholesale is exactly what
   the "honest caveat about generic" section forbids. **Blocked on the
   design-generic half being separated from the shim**, which is where
   `reduce_asic.py`'s launch/collect and `export_rtl.py`'s `compile_order`
   would go.

3. **`rtl_handoff/` and `gemmini_rtl/` are generated. Moved, with the
   interface change made deliberately.** ***Done 2026-09-24, second pass.***
   `preflight.py` used to resolve `<--design>/rtl_handoff/<variant>` and
   `<--design>/gemmini_rtl/<variant>`: the exports' location and their two
   possible names were both tree shape encoded in a tool. It takes
   **`--exports`** now -- the same flag, with the same meaning, that
   `check_pairing.py` already required -- so the two sibling tools agree
   instead of one of them guessing.

   **The integrity property was the constraint, and it holds.** `check_pairing.py`
   admits an area figure only against an export whose `sv2v_manifest.f` md5
   matches what the run recorded. That manifest is a list of bare filenames --
   its own header says "paths are relative to THIS FILE's directory" -- so it
   carries no repository path and **its md5 is invariant under the move**: the
   digest over all 18 committed manifests is `17501fdf` before and after.
   `check_pairing.py`'s full output is byte-identical across the move, and
   `check_numbers.py` still prints `AREA NUMBERS OK`.

   The historical absolute paths inside `reports/*/settings.json`,
   `results.json` and `RUN.rpt` are **not** rewritten. They record where a run
   actually read its RTL, on a machine that is not this one; editing them to
   match today's tree would be falsifying the evidence.

4. **CHIA is still in two places, and neither is `chia/`. Deferred on purpose,
   with the destination settled and the trigger named.** The owner has
   endorsed a root-level `chia/` ("only `chia/` might deserve a similar root
   level position"), so **where** it goes is not the open question. **When**
   is: it is the component that spends real money, and **no required gate
   exercises it**. Moving a money-spending harness with nothing watching is how
   a `CHIA_TOTAL_CAP_USD` ceiling turns into a silent failure rather than a
   refusal.

   **The trigger: move it once the $0 `chia_agent/test_harness.py` gate is in
   the required set.** Then a move is covered by a run that costs nothing, and
   the three-way split (`chia_abstraction/`'s code, its `evidence*`/`baseline*`
   records, and `test_abs_harness.py`) can be made under cover.

   **Do not start before that gate lands.** As of 2026-09-24 the loop is dead
   on `main` -- every candidate fails at stage `model` because
   `workloads/run.py`'s import closure is not in `evaluate.FROZEN` -- and a
   repair is in flight in `examples/tinytpu/chia_agent/` and
   `examples/tinytpu/workloads/`. Reorganising a component mid-repair costs
   both sides more than waiting does.

**`playground/` was left in place deliberately.** `int8_gemm.py` and
`mat_vec_test.py` are scratch scripts nothing references, and one of them
defines `test_single_systolic()` with `MODE = "csyn"` -- moving it under
`tests/` would have pytest collect a function that needs Vitis. The honest
answer is that it should probably not be in the repository at all, which is a
deletion question for the owner and not a move to make unasked.
