# Provenance of this flow

This directory is Julian Bushlow's mflowgen ASIC flow, vendored from
[`jbushlow/allo-asic`](https://github.com/jbushlow/allo-asic) at commit
`e903e36` on 2026-09-24. It is Apache-2.0; `LICENSE` is his, unmodified. The
import kept its history, so `git log allo/backend/asic` credits its author
across 103 commits.

It lives here rather than in `scripts/` because the flow is a backend for an
ASIC target, next to `allo/backend/catapult.py`, which is likewise a module that
orchestrates an external EDA tool. Two entry points share it: the **flat flow**,
which consumes emitted Verilog plus a file list, and **AAAH**, the `allo-*`
nodes, which consume Allo-level architecture — unit boundaries, semantic
equivalence classes, stream bundles with widths and directions, PE grid
coordinates — through an `asic_manifest` config passed into Allo's own `build()`.

## What was deliberately not imported

**The FreePDK45/NanGate ADK payload** (`adks/freepdk-45nm/pkgs`, `view-tiny`):
17 files, 7 MB, whose headers carry Nangate copyright and "provided pursuant to
a License Agreement" language. Redistributing them is not ours to do, so they
were removed from the imported history as well as from the tree. Nothing is
lost: the ADK node fetches the standard view at run time, that view is the only
one DC can use — `view-tiny` has no compiled `stdcells.db` and no
`rtk-tech.tf`, so DC emits a GTECH netlist and fails the node's own
postcondition — and `tools/preflight.py` verifies what was fetched against the
`stdcells.db` md5 recorded in every run's settings snapshot.

**Designs that need an Allo fork we are not on**, and `allo-rebuild/`:
`allo-test`, `allo-mininpu-v2`, `allo-scaling-eval`, `allo-eva-*`,
`AlloSystemC-channel`, `TestNPU-*`, and `tutorial-vvadd` (whose RTL was never in
the repo — it is a Drive download). They remain in Julian's repo.

Kept: `designs/GcdUnit`, the 16-second known-good smoke run, and
`designs/template-design`, the skeleton. GcdUnit's RTL is adapted from the
mflowgen tutorials.

## Known defects in the code as imported

Both were hit and diagnosed during the TinyTPU/Gemmini area runs; neither is
fixed here, so that the vendored code still matches its upstream:

1. **The ADK node is not idempotent against a repository checkout.**
   `adks/freepdk-45nm/configure.yml` sets `sandbox: False` and runs
   `mv {adk_view}/adk.tcl {adk_view}/adk-base.tcl`, so with a view that lives in
   the tree a second run renames the overlay onto its own base and `adk.tcl`
   sources itself — DC then dies with "too many nested evaluations". With
   `view-standard` the view is unpacked into the build directory and the tree is
   untouched, which is the configuration we use.
2. **`sv2v-design-collector` rejects a Vitis top module.** Its top-module check
   does not match a `module` declaration carrying a `(* CORE_GENERATION_INFO *)`
   attribute, which is what Vitis HLS emits. The TinyTPU designs set
   `normalize_rtl: False` and pass Verilog-2001 through unchanged, which avoids
   it.

## Tests

The node library's own tests pass **per node directory**, 182 of them. They
cannot be collected as one suite from the repository root: several share a
basename and several assume the working directory is their own node.

`nodes/allo-asic-compilation` used to be the one failing directory, because it
imports `allo` and expects an `asic_manifest` emitter this fork did not have.
`allo/backend/asic_manifest.py` is now that emitter
(`docs/source/backends/asic_manifest.rst`), and the directory's other 17 tests
pass with `ALLO_HOME` set to the checkout. What still fails there is
`test_catapult_manifest.py`, and only its second half: it also loads
`$ALLO_HOME/scripts/extract_catapult_pe_manifest.py`, the post-HLS *Catapult
RTL* extractor, which is a parse of generated Verilog rather than an
architectural fact and is not written here. The import it opens with,
`from allo.dataflow import _build_manifest_top_arguments`, now resolves.

```bash
cd allo/backend/asic/nodes/<node> && python -m pytest -q .
```
