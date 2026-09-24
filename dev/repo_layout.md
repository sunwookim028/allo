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

```
examples/                 one directory per design, each self-describing
  feather/                (exists)
  tinytpu/                <- examples/accelerator/tinytpu_vitis/
  minitpu/                <- to come
  eva/                    <- from the SystemC-emitter repository
  systemc_rtlsim/         (exists)

tools/                    flows, usable by any design
  act/                    mapping compiler: spec -> program, and its judge
  chia/                   the agentic loop, its guards and its harness
  asic/                   PD flow: construct graph, RTL lists, stubs,
                          preflight, extractor, checkers
  rtl_export/             design -> flat RTL + manifests

allo/                     core IR, passes, schedule primitives (unchanged)
docs/source/              published pages
dev/                      working notes, not published
```

**A design directory owns:** its units, its ISA, its programs, its own
reproduce script, and its own results. **A tool directory owns:** everything
that would otherwise be copied into the second design.

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
