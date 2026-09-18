# Retired notes

Files here are kept, not deleted: several carry measurement data that may be
cited later, and moving them keeps that data greppable and recoverable. Nothing
in this directory is maintained — treat every claim as dated. Live notes are one
level up in `notes/`.

Two files here are **sources, not retired notes** -- see the second table.

The three below were retired **2026-09-17**, in the notes audit that also
corrected `MAINTENANCE_CHECKLIST.md`, `PITFALLS_DATAFLOW_REGION.md`,
`DATAFLOW_SEMANTICS.md`, `ASIC_HLS_EXPLORATION.md`, and
`docs/source/backends/nonblocking_streams.rst`. They were re-audited
**2026-09-18**; each is here because it holds data that exists nowhere else in
the working tree.

| File | Retired | Why it is still here |
| --- | --- | --- |
| `HIERARCHY_DESIGN.md` | 2026-09-17 | Historical design record, so declared by its own 2026-07-15 status update; items 1-2 landed upstream via PR #577 and its item-4 file references (`_build_top` bare-scalar emission, `s_axilite` in `postprocess_hls_code`) describe reverted code. Kept for §5 and §8: fork issue #7 carries the four faces, §5's root cause and Alt A, but **not** Alt B, Alt C or the recommendation between them. Cited live from `notes/PITFALLS_DATAFLOW_REGION.md`. The copy on `choonsik1/SystemC-emitter` is an older variant, not this text. |
| `CATAPULT.md` | 2026-09-17 | 533-line Catapult synthesis findings for one design (`top_decoupled_2x1`). §5-6 are duplicated in the live `notes/CATAPULT_QUICKSTART.md` (except CRD-413, the `1.000000f` literal fix and CIN-319, all three of which survive as code: `EmitVivadoHLS.h:102` / `EmitCatapultHLS.cpp:143`, `Utils.cpp:63`, and the `static` at `EmitCatapultHLS.cpp:308`). §0 and §3-4 are the reason to keep it — see below. The "not pursuing further" decision it fed was **reopened 2026-09-18** (`notes/ASIC_HLS_EXPLORATION.md`), so these numbers are live reference again. |
| `HLS_SYNTH_REPORT.md` | 2026-09-17 | Finished measurement record (U280, Vitis HLS 2023.2, 2026-03/04). Its fp16 fixes (§6.3) are upstream as of `upstream/main`, and its claim that `EmitTapaHLS.cpp` carries the non-blocking ops is false — that support was removed. Kept for §1-3; the claim that its headline numbers moved into `docs/source/backends/nonblocking_streams.rst` is only half true (see below). |

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

## Deleted 2026-09-18 — recoverable, pointers kept

Two of the five 2026-09-17 retirements held no data of their own. They were
deleted rather than stored, on the same pattern as `SYSTEMC_COMB_MODE.md`
(`notes/ALLO_SHORTCOMINGS.md` #22). Both blobs were verified byte-identical by
`md5sum` before removal:

- `ALLO_LESSONS.md` — `git show f1c3aad1^:notes/ALLO_LESSONS.md`
  (also `61dc8fa8:notes/archive/ALLO_LESSONS.md` on `choonsik1/SystemC-emitter`;
  md5 `c2aebca04d73b80db19150cf38c36cf6`). Prescribed a two-repo workflow against
  `~/projects/allo-tpu`; `~/projects` does not exist and accelerator work is
  in-tree under `examples/accelerator/tinytpu*`. Everything in it that is still
  true is already live: the one-process-per-MLIR-dump rule and the
  compiler-internal-error list in `notes/PITFALLS_DATAFLOW_REGION.md`, the stale
  `.cache/llvm_sim/` rule as `notes/ALLO_SHORTCOMINGS.md` #9.
- `PROGRAMMABLE_DATAFLOW.md` — `git show 4ee5bdde:notes/PROGRAMMABLE_DATAFLOW.md`
  (also `68cf2b32:notes/PROGRAMMABLE_DATAFLOW.md` on
  `choonsik1/fix/nb-stream-scalar`; md5 `4933289d3c46dc81edc5fb66fc5db971`).
  15 lines of framing, truncated mid-sentence ("What is the textbook-level
  wisdom? -"), no measurement in it. Its design-space bullets are superseded at a
  far higher evidence level by `notes/DESIGN_SPACE.md` §4-6.

## Things these files still get right

Each verified 2026-09-18 by grepping the whole tree for the numbers.

- `CATAPULT.md` §3-4 are the only per-loop latency and per-module/per-FIFO area
  numbers for `top_decoupled_2x1` anywhere in the repo.
  `notes/ASIC_HLS_EXPLORATION.md` carries only the tile totals (295 / 67 / 657 /
  298 cycles; CT=14991, CT=14991, MT=16180) — not the loop breakdown, the
  8-FIFO area table, or the compute/storage/communication split. §0 (structural
  diagram, execution timeline, concurrency analysis) is likewise unique.
- `HLS_SYNTH_REPORT.md` §1-3 are the only blocking-vs-non-blocking LUT/FF
  breakdown at module granularity, and the only place the decoupled-mesh area
  numbers (5355/5300 for 1 MT + 1 CT, 7361/6724 for 2×1) appear at all.
  **The nb-streams doc is not a substitute:** it repeats LUT 1417 / 1457 but
  gives FF as 248 / 260 where this file measured 1325 / 1369. The two disagree
  and the doc labels its table "estimated"; this file is the primary record.
- `HIERARCHY_DESIGN.md` §5 (the `IsolatedFromAbove` diagnosis) and §8 (Alt A/B/C)
  are still the fullest statement of the region-as-module argument; fork issue #7
  summarizes but does not replace them.
