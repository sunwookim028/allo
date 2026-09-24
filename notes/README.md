# notes/

Working documentation for this checkout. Four live documents; everything else is either
next to the code it describes, or in `archive/`.

| File | What it holds |
|---|---|
| **[STATE.md](STATE.md)** | Where things stand — remotes, branches, current work, known gaps, ranked next steps. **Start here.** |
| **[SIMULATOR.md](SIMULATOR.md)** | The JIT dataflow simulator: architecture, the link abstraction, the timing model and why the per-PE clock breaks on a mesh, cycle-accuracy routes, LightningSim status. |
| **[BACKEND.md](BACKEND.md)** | The SystemC/Catapult backend: landed changes, and the `comb` emission mode that was designed but never implemented. |
| **[ALLO_GOTCHAS.md](ALLO_GOTCHAS.md)** | Things that cost a day. Silent wrong answers first, then build traps, environment, and measurement traps. |

## Elsewhere

Documentation that lives next to its code, deliberately:

- `mlir/lib/Translation/EmitSystemC.md` — the layered emitter walkthrough
- `docs/SYSTEMC_BACKEND.md` — user-facing backend docs
- `docs/DATAFLOW_LINKS.md` — link types, with a runnable companion
- `examples/systemc/VERDICTS.md` — per-example SystemC outcomes

Two things live **outside this repo**:

- `/home/zsm9/simulator_profiling` — the JIT-simulator profiling harness, its results, and
  the LightningSim stage runs (`README.md`, `RESULTS.md`, `RUN.md`, `lightningsim_*/`).
  Moved out of the repo on 2026-08-15; `SIMULATOR.md` cites it throughout.
- `/home/zsm9/final_noc` — the NoC evaluation: designs, harnesses, synthesis scripts and
  the Allo-vs-reference comparison.

- `papers/` — reference PDFs cited by `SIMULATOR.md`
- `archive/` — superseded or inherited; see below

## archive/

Nothing here is current. Two kinds of thing:

**Inherited from the `sup` fork** (`sunwookim028/allo`, hence the directory name
`allo_sup`). These describe a *different project* (`allo-tpu`) and a branch topology that
does not exist in this checkout, and were not re-verified — at least one claim in them is
now wrong. Kept for provenance only: `STATE.sup.md`, `BRANCHES.sup.md`,
`ALLO_SHORTCOMINGS.md`, `ALLO_LESSONS.md`, `PITFALLS_DATAFLOW_REGION.md`,
`CATAPULT_QUICKSTART.md`, `HIERARCHY_DESIGN.md`.

**Our own sources, superseded by the merge above**: `claude_simulator.md` (keeps the dated
session log and the full per-paper analysis, neither reproduced in `SIMULATOR.md`),
`simulator_concept.md`, `simulator_shared_clock.md`, `simulator_cycle_model.md`,
`simulator_lightningsim_plan.md`, `BACKEND_CHANGES.md`, `SYSTEMC_COMB_MODE.md`,
`claude_evaluation.md` (the evaluation-shell charter, executed by the `final_noc` work),
`emitter_snapshot_2026-07-30/` (byte-identical to commit `0f7a116`), and
`simulator_org*.py` (byte-identical to commits `174e3b4` / `180cbbf`), and
`simulator_nb_nondet.py` — a pre-timing-layer snapshot of the simulator. It is the
only version that can RUN A MESH, because it predates the read barrier that makes the
current simulator's per-PE clocks diverge. Non-deterministic by construction; see its
header for the swap-in procedure.
