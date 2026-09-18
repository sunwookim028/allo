# ASIC HLS Backend Exploration

## Summary

Explored two commercial ASIC HLS backends (Catapult HLS, Tapa) as part of the
mesh accelerator research. Decision: not pursuing further. Vitis HLS remains the
primary backend. CIRCT is the recommended long-term direction for ASIC targets.

## Catapult HLS (Siemens EDA)

### What was built
- The Catapult backend itself (`mlir/lib/Translation/EmitCatapultHLS.cpp`,
  `allo/backend/catapult.py`) came from upstream PR #543 (Feb 2026). What this
  fork added on top is non-blocking stream support for it (commit of
  2026-04-14), plus the synthesis bring-up below. The emitter is 683 lines
  today.
- Key overrides vs Vivado:
  - F32 → `ac_ieee_float<binary32>` (nangate-45nm requires this, not plain `float`)
  - `ac_channel<T>` for streams. Note the `try_*` ops deliberately emit
    *blocking* `read()` / `write()`: `nb_read` / `nb_write` inside a spin-while
    loop segfaults Catapult's go compile (LOOP-19). `empty()` is emitted as
    `!ch.available(1)`.
  - `static` prefix on local channel declarations (fixes HIER-6)
  - Block synthesis mode to allow channels to cross hierarchical boundaries

### Synthesis results (Catapult 2024.2, nangate-45nm_beh, 500 MHz / 2ns)
Design: `top_decoupled_2x1` (1 MT + 2 CTs, M=N=K=2, 16 elements)
- CT0/CT1 latency: 295 cycles each; MT: 67 cycles; Total sequential: 657 cycles
- Throughput: 298 cycles (CT-bound)
- Area scores (Catapult internal): CT0=14991, CT1=14991, MT=16180

### Key issues resolved during development
- HIER-10: Local channels → solved by block synthesis
- HIER-47/ASSERT-1: FIFO_DEPTH=0 → solved by block synthesis
- CIN-291: float→fixed-point conversion → use `ac_ieee_float<binary32>`
- CRD-415/CRD-413: double literal / assignment issues → emit `0.000000f`
- HIER-6: Non-static local channel → add `static` prefix

### Why not continuing
- Market niche (automotive/defense, Siemens-adjacent shops)
- Not a standard research community reference tool
- Cadence Stratus is stronger competitor for ASIC research citations
- CIRCT (MLIR-native, Google/Intel-backed) has better long-term trajectory

## Tapa HLS (non-blocking stream additions)

### What was added
- `emitStreamTryGet`, `emitStreamTryPut`, `emitStreamEmpty`, `emitStreamFull` overrides
  in `EmitTapaHLS.cpp`: maps to `.try_read()`, `.try_write()` (Tapa API)
- One test (`test_nb_ops_tapa_codegen`), since removed as well

### Why removed
- Tapa is not used in our mesh research flow
- NB stream semantics are fully validated via Vitis HLS (primary target)
- Keeping dead codepath creates maintenance burden in EmitTapaHLS.cpp

## Reference
- Full Catapult synthesis findings: `notes/archive/CATAPULT.md` (retired
  2026-09-17; the earlier pointer to commit `01f25e2` is dead — no such
  revision exists in this repo).
- `notes/archive/HLS_SYNTH_REPORT.md` has the Vitis HLS numbers for comparison.
- Tool setup and error cookbook for anyone who does re-run Catapult:
  `notes/CATAPULT_QUICKSTART.md`; `ppa` mode is documented in
  `notes/ppa_analysis.md` and still lives in `allo/backend/catapult.py`.

## Status of the removal (verified 2026-09-17)
- `mlir/lib/Translation/EmitTapaHLS.cpp` has no `try_write` / `try_read`
  emission and its visitor dispatches only construct/get/put. The base-class
  hooks in `EmitBaseHLS.h` are empty, but they are never reached: the op falls
  through to `visitUnhandledOp` and `emitBlock` reports "can't be correctly
  emitted", so `build(target="tapa")` raises `RuntimeError` rather than
  silently producing nothing. (Corrected 2026-09-18; the earlier "not
  diagnosed — it simply produces nothing" reading was wrong. The message it
  does print blames `wrap_io`, which is unrelated — see
  `notes/ALLO_SHORTCOMINGS.md` #19.)
  `tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb` asserted
  `.try_read(` / `.try_write(` and was therefore failing outright; it is now
  `xfail(strict=True, raises=RuntimeError)`.
