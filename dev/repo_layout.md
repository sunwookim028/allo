# Repository layout: designs, tools, core

Decision 2026-09-24: the fork will host several accelerator designs — TinyTPU,
MiniTPU, EVA — and several flows — Vitis HLS, the ASIC/PD flow, the SystemC
emitter, the ACT mapping compiler, the CHIA agentic loop. The layout must match
how upstream separates **example designs**, **tools**, and **core IR/compiler
passes**, and push-button scripts and docs must assume it.

## What is wrong today

`examples/feather/` is three plain Python files: a design expressed in Allo.
`examples/accelerator/tinytpu_vitis/` is a whole project — `ip/`, `act/`,
`asic_synthesis/`, `chia_agent/`, an RTL exporter, reports and a reproduce
script. Two problems follow:

1. **TinyTPU sits a level deeper than every other example** for no reason a
   reader can infer, so `examples/` no longer lists the designs.
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

scripts/asic/             the PD flow: construct graph, RTL export, stubs,
                          preflight, extractor, number checkers

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
  output against RTL simulation. It belongs with the emitter's own example
  material, as `examples/systemc/`, matching where the upstream-of-this-work
  repository already keeps it.
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
   and it makes `examples/` list the designs again.
3. **Then extract tools**, one at a time, each with the "could MiniTPU call
   this?" test applied and recorded. `tools/asic/` first, because a second
   machine is already committing into it and the sooner its home is stable the
   fewer redirects it takes.
4. **Then add the new designs** — MiniTPU, EVA — against the settled layout
   rather than into a moving one.
5. `chia_runs/` leaves the repository root.

## Docs that follow from it

Per `dev/docs_style.md`, each tool gets a page with a **Quick start** before any
explanation: `tools/asic` (preflight, the documented sequence, what is committed
and what stays on scratch), `tools/chia` (how to run it, what it costs, what the
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

## Known defect to fix during the move

`asic_synthesis/construct-commercial.py` resolves its node library and ADK at
`examples/accelerator/{nodes,adks}/`, which do not exist in this repository, so
it **cannot run from a clean checkout today**. Wherever the vendored nodes
land, that path has to match. Note also that the construct graph and
`make_stubs.py` were committed by a different session than the one that
produced the reports; attribute them accordingly when moving.
