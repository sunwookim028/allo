# Retired notes

Files here are kept, not deleted: several carry measurement data that may be
cited later, and moving them keeps that data greppable and recoverable. Nothing
in this directory is maintained — treat every claim as dated. Live notes are one
level up in `notes/`.

Two files here are **sources, not retired notes** -- see the second table.

The five below were retired **2026-09-17**, in the notes audit that also
corrected `MAINTENANCE_CHECKLIST.md`, `PITFALLS_DATAFLOW_REGION.md`,
`DATAFLOW_SEMANTICS.md`, `ASIC_HLS_EXPLORATION.md`, and
`docs/source/backends/nonblocking_streams.rst`.

| File | Retired | Why |
| --- | --- | --- |
| `HIERARCHY_DESIGN.md` | 2026-09-17 | Historical design record, so declared by its own 2026-07-15 status update. Items 1-2 landed upstream via PR #577; live tracking and the actual pitch drafts moved to fork issue #7. Its item-4 file references (`_build_top` bare-scalar emission, `s_axilite` in `postprocess_hls_code`) describe code that was reverted and no longer exists. |
| `CATAPULT.md` | 2026-09-17 | 533-line Catapult synthesis findings for one design (`top_decoupled_2x1`). The decision it fed is closed and recorded in `notes/ASIC_HLS_EXPLORATION.md` ("not pursuing further"); the reusable methodology (block synthesis, FIFO depth inference, the error cookbook) is duplicated in the still-live `notes/CATAPULT_QUICKSTART.md`. |
| `HLS_SYNTH_REPORT.md` | 2026-09-17 | Finished measurement record (U280, Vitis HLS 2023.2, 2026-03/04). Its headline numbers now live in `docs/source/backends/nonblocking_streams.rst`; its fp16 fixes (§6.3) are upstream as of `upstream/main`; and its claim that `EmitTapaHLS.cpp` carries the non-blocking ops is false — that support was removed. |
| `ALLO_LE§ONS.md` | 2026-09-17 | Prescribes a two-repo workflow (`~/projects/allo` for compiler work, `~/projects/allo-tpu` for application work, "never edit both repos in the same session") against a repo that does not exist. Accelerator work now happens inside this tree (`examples/accelerator/tinytpu*`), so the central rule is moot. Its still-true parts — the one-process-per-MLIR-dump rule and the compiler-internal-error list — were extracted into `notes/PITFALLS_DATAFLOW_REGION.md` before the move. |
| `PROGRAMMABLE_DATAFLOW.md` | 2026-09-17 | A 15-line framing draft that breaks off mid-sentence ("What is the textbook-level wisdom? -") and was never continued. The direction it sketches was executed instead as the instruction-programmable TinyTPU work, whose findings are in `notes/ALLO_SHORTCOMINGS.md` §11-17. The weakest of the five calls: restoring it is one `git mv`. |

## Sources kept for provenance, added 2026-09-18

These two are not superseded notes. They are the raw research the live notes
were distilled from, kept here so the provenance of every number stays
reachable without adding 72 KB to `notes/`.

| File | Distilled into | Read it for |
| --- | --- | --- |
| `minitpu_scaling.md` | `notes/MINITPU_REFERENCE.md` | The 8-stage TinyTPU scaling plan, dropped from the live note as plan rather than evidence. **Read with care:** it describes `~/core/npu`, which is a *different machine* from the live `~/core/minitpu` -- different issue slots, a 6-bit rather than 7-bit DELAY field, and a banked VMEM where the live design has one flat dual-port URAM array with undefined cross-port collisions. Its scaling stages rest on that banked VMEM and on `mxu_adapter.sv`, which is dead code. The live note tabulates the differences. |
| `tpu_design_space.md` | `notes/DESIGN_SPACE.md` | The full DSE survey with its per-axis prose. The live note keeps every cited number, the seven monotone axes and the conclusion; this keeps the reasoning around them. |

Both were written against trees that do not travel (`~/core/npu` is now empty),
so nothing in them can be re-derived from source on another host. That is why
they are committed rather than dropped.

## Things these files still get right

- `CATAPULT.md` §3-4 are the only per-module Catapult latency/area numbers for
  `top_decoupled_2x1` anywhere in the repo.
- `HLS_SYNTH_REPORT.md` §1-3 are the only blocking-vs-non-blocking LUT/FF
  breakdown at module granularity.
- `HIERARCHY_DESIGN.md` §5 (the `IsolatedFromAbove` diagnosis) and §8 (Alt A/B/C)
  are still the fullest statement of the region-as-module argument; fork issue #7
  summarizes but does not replace them.
