# ASIC HLS Backend Exploration

## Summary

Explored two commercial ASIC HLS backends (Catapult HLS, Tapa) as part of the
mesh accelerator research. Vitis HLS is the primary backend and has the only
end-to-end measured results (`examples/accelerator/tinytpu_vitis`).

**The 2026-09-17 decision recorded here -- "not pursuing Catapult further,
CIRCT is the recommended long-term direction" -- was reopened on 2026-09-18 and
no longer describes the plan.** Both halves of it moved:

* **Catapult is being pursued again**, as a bounded spike on `zhang-21` (the
  host that has the tool). The reason is not the synthesis numbers below; it is
  that the SystemC fork carries a `Wire` -- a non-handshaked edge -- and a
  MiniTPU-class VLIW delay line needs one. That `Wire` is currently **wrong in
  RTL, not merely in csim** (`ALLO_SHORTCOMINGS.md` #22): simulating Catapult's
  own `pe_wire` netlist under xsim fails 8/8 at all 18 producer/consumer
  pacings, while `Stream` and `Channel` pass all 18. The failure is diagnosed
  rather than mysterious -- holding `acc_0` in reset 3-4 cycles longer and
  stepping it once per product makes the *identical* RTL produce exact golden
  output, so only the lockstep is missing and the correct window is 2 cycles
  wide. The spike's first deliverable is that test passing under Catapult's own
  scheduler, not a TPU. A second, independent reason to want the tool: it gives
  **ASIC PPA**, and the Gemmini comparison is cycles-only today.
  Note that no SystemC library or MatchLib exists on `ace-01`, so csim cannot
  run here for any design -- which is itself part of why the move is worth
  making.
* **The CIRCT path is not currently reproducible in this checkout.** Its clone
  (`externals/circt`, 2.3 GB with its build tree) was deleted on 2026-09-18 in
  the pre-migration cleanup. It was untracked, not a submodule, and referenced
  by nothing. The pin survives only because the generated RTL stamps it:
  **CIRCT `af5369d`**. The generator lives on `chia-codesign`
  (`examples/accelerator/tinytpu/microarch.py`), not here, and the artifacts
  worth keeping -- the per-unit modules, `gen_ip.tcl`, and `manifest.json` --
  are committed there under `examples/accelerator/tinytpu/rtlgen/`.
  `manifest.json` is a per-module scheduling model (determinacy class, latency,
  per-port bank/factor/latency/width) and so is directly relevant to #22's
  conclusion that the SystemC path lacks one. It is a partial answer: the four
  `counted_static` units carry latencies, while the top and both DMA units are
  `indeterminate` with none -- the data-dependent units a delay line actually
  has to schedule against.

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

### Why not continuing (the 2026-09-17 reasoning; see Summary -- reopened)
- Market niche (automotive/defense, Siemens-adjacent shops)
- Not a standard research community reference tool
- Cadence Stratus is stronger competitor for ASIC research citations
- CIRCT (MLIR-native, Google/Intel-backed) has better long-term trajectory

## The C++ path can express nothing Vitis cannot (2026-09-18)

**Scope, and it is the whole point of this section:** this is about the **C++
emitter path** (`EmitCatapultHLS.cpp` -> `ac_channel`). It is **not** about the
SystemC path, which is a different fork (`choonsik1/allo:SystemC-emitter`) with
a different type set -- `Stream` / `Channel` / `Wire` over MatchLib
Connections -- and which is the actual reason Catapult is being pursued (see
Summary, and `ALLO_SHORTCOMINGS.md` #22: the SystemC `Wire`, the one
non-handshaked edge in play, is wrong in RTL rather than merely in csim, with
the apparatus committed at `examples/systemc_rtlsim/`). Nothing below carries
over to that path.

### The measurement

`CatapultModuleEmitter` is a subclass of the Vivado emitter --
`class CatapultModuleEmitter : public allo::hls::VhlsModuleEmitter`
(`EmitCatapultHLS.cpp:109`). Counted 2026-09-18:

| | |
|---|---|
| `EmitCatapultHLS.cpp` | 683 lines |
| `EmitVivadoHLS.cpp` | 3428 lines (Catapult is 20% of it) |
| member functions `VhlsModuleEmitter` declares | 57 |
| of those, overridden by Catapult | **14** (12 out-of-line, 2 inline in the class body) |

The 14: `emitModule`, `emitFunction`, `emitFunctionDirectives`, `emitValue`,
`emitArrayDecl`, `emitArrayDirectives`, `emitLoopDirectives`,
`emitStreamConstruct`, `emitStreamTryGet`, `emitStreamTryPut`,
`emitStreamEmpty`, `emitStreamFull`, `emitStatefulGlobalElementType`,
`emitFloatArrayElement`.

Every one substitutes *syntax* at a point where Vivado already emits something:
type spellings (`ac_ieee_float<binary32>`, `ac_int<W,S>`), `ac_channel` for
`hls::stream`, directive comments for pragmas, `static` on local channels, an
`f` suffix on float literals. None adds a construct; the other 43 emitters are
inherited verbatim. So whatever the frontend cannot say, both backends fail to
say identically -- and where they differ, Catapult says *less* (#18:
`try_get`/`try_put` degrade to blocking with `success` hard-coded `true`).

### Consequence: a three-way classification

Anything flagged as a limitation on this path is one of:

- **(A)** genuinely Allo's own -- the frontend/IR cannot express it;
- **(B)** *apparently* Vitis's, but (A) underneath -- Vitis has the construct,
  Allo has no way to reach it;
- **(C)** genuinely the tool's.

**On the C++ path, (B) is nearly always really (A)**, because the emitter is a
thin syntax layer: switching Vitis -> Catapult changes spelling, not
expressiveness. A (B) diagnosis there should be treated as a claim to check,
not a reason to change backend. Six flagged items:

| item | class |
|---|---|
| shared multi-ported memory (two ports of one array to two kernels) | **(A)**, and the decision-relevant one -- below |
| #21 no `#pragma HLS dependence`, so a false dependence cannot be asserted away | **(A)** -- Vitis has the pragma; Allo emits only `m_axi`/`s_axilite`/`bind_storage`/`array_partition` |
| #13 "no program-controlled DMA" | **(B) -> neither** -- retracted: a contiguous runtime-length copy does infer a variable-length AXI burst. The cost had been charged to the configuration when the access pattern was the variable that differed |
| #18 non-blocking `try_get`/`try_put` | **(C) + (A)** -- the LOOP-19 `go compile` segfault on `nb_read` is Catapult's; emitting `success = true` silently instead of refusing is ours |
| native `float` unsynthesizable (CIN-291) | **(C)** -- `nangate-45nm_beh` genuinely lacks it, and the `ac_ieee_float` override is the entire fix |
| #22 a non-handshaked fixed-latency edge | **(C)** *on this path* -- neither `hls::stream` nor `ac_channel` has one. That is precisely why the SystemC path exists, and why the spike is aimed there and not here |

### The decision-relevant instance: shared multi-ported memory is (A), and blocked

`AlloMemPins` **is** an unarbitrated 1R1W dual-port RAM, and it synthesizes.
Allo simply will not hand out both ports to two kernels. So the shared-scratchpad
gap is (A): no backend switch resolves it.

Verified 2026-09-18, with the provenance stated because it matters: `AlloMemPins`
is **not in this checkout's working tree**. It lives on the
`choonsik1/SystemC-emitter` fork, which is fetched in this clone
(`remotes/choonsik1/SystemC-emitter`), in
`mlir/lib/Translation/EmitSystemC.cpp`. Read there, the module has separate read
and write pin bundles (`radr`/`re`/`q` and `wadr`/`d`/`we`) over one `T mem[SIZE]`,
with both accesses serviced in a single `wait()`-delimited cycle and both ready
lines tied high; its own comment says "single-cycle, unarbitrated, no bank
conflicts". It is instantiated on both sides of the boundary, the in-design case
being described there as "a replicated, multi-client array", and it is emitted
for synthesis (it carries a CIN-233 reset-write workaround for exactly that
case). `#22`'s incidental valid/ready finding names the older `AlloMem`, which
is a different module.

Two caveats a reader on zhang-21 should keep: this is a **SystemC-path** artifact,
so what it demonstrates is that the *hardware* is unobjectionable, not that the
C++ path emits it; and "Allo refuses to hand out both ports" is the frontend/IR
restriction, which is the (A) part and is common to both paths -- #22 already
states it in passing ("blocked further back than 'Allo won't hand out two
memory ports'"), but does not source the counter-evidence, which is what this
entry adds. The
multi-client claim is read from source and comments, not from a run -- no
SystemC library or MatchLib exists on `ace-01` (Summary), so nothing here was
csim'd or synthesized.

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
