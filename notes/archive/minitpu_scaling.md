# MiniTPU → TinyTPU scaling report

Sources are cited inline as `path:line` or `path` where line numbers aren't
meaningful (generated tables, prose docs). Everything under
`/home/sk3463/core/npu` was read-only; nothing there was modified, built, or
synthesized.

---

## 1. MiniTPU's actual architecture

**Systolic array.** `N = 16` (`src/pkg/minitpu_config_pkg.sv:8`), so a 16x16
PE grid, weight-stationary (`src/core/mxu/systolic_array.sv`,
`mxu_pe.sv`). Datapath is BF16 in (`DATA_WIDTH = 16`,
`minitpu_config_pkg.sv:11`), each PE holds one stationary BF16 weight and a
sign+8+15 = 24-bit accumulator (`MXU_ACC_FRAC_W = 15`, `MXU_ACC_W = 1+8+15 = 24`,
`src/core/vpu/vpu_pkg.sv:125-126`) — i.e. an implicit-fp24 partial sum, not a
plain fixed-point integer accumulator. `psum_in` is tied to zero only at row 0
(`systolic_array.sv` gen_north block), so the array accumulates exactly `DIM =
16` deep in hardware; anything deeper is a BF16 `vadd` in the VPU
(`docs/OPERATOR_SUPPORT.md` §1, "Anything deeper is `vadd` on BF16 VREGs").
Command timings (from `docs/isa_latency.json` `matrix` block and
`docs/ISA_AND_INTERFACES.md`): `vmatload` (load stationary weights) occupies
the controller 18 cycles and holds VREG read port A; `vmatpush` (stream one
activation row) occupies it 5 cycles and its *first* result appears 82 cycles
later; `vmatpop` occupies it 7 cycles. Push and pop share one controller and
cannot overlap, so draining N result rows currently costs ~13 cycles/row of
controller time, not 6 (`docs/SCHEDULING.md` §2). Output leaves the array in
push order through a **32-entry-per-lane FIFO** (`MXU_OUTPUT_FIFO_DEPTH = 32`,
`docs/isa_latency.json` resources block; 4 entries per push, 4 consumed per
pop, so 8 pushes fill it) — overflow drops results silently, there is no
replay (`docs/isa_latency.json` note, `src/core/vpu/fifo.sv`).

**VMEM (on-chip scratchpad).** 2 ping-pong slots × 8192 rows × 32 B/row = 512
KiB total (`docs/VMEM.md` §1). A row is one BF16 word × 16 lanes = 256 bits =
32 B — also exactly one DM (device-memory) word and one AXI beat. Banking is
**4 banks per lane** (`NUM_SUBLANES = 4`, `src/core/vpu/vpu_pkg.sv:47`),
`bank = row mod 4` (a bit-slice, not a divider, because `NUM_SUBLANES` is
constrained to a power of two — `vpu_pkg.sv:93-97`), giving `BANK_DEPTH = 8192
/ 4 = 2048`. A `vld`/`vst` gathers up to 4 rows (one full VREG) per access
using a `sublane_stride`/`sublane_mask` pair that is entirely an **assembly-time
immediate**, so bank-conflict cost (`gcd(stride, 4)` cycles: 1 if stride is odd,
2 if ≡2 mod 4, 4 if ≡0 mod 4) is knowable at compile time with zero runtime
arbitration (`docs/VMEM.md` §3, `docs/VMEM_LAYOUT_AND_BANKING.md` §2). The
architectural insight documented at length in `VMEM_LAYOUT_AND_BANKING.md` is
that `bank = row mod NUM_SUBLANES` is chosen *specifically* to make the native
`(sublane, lane)` tiling — the layout every real kernel uses — conflict-free,
mirroring how Pallas/Mosaic treats the TPU's own `(8,128)` tiling as native
and inserts explicit relayouts for anything else.

**Register / accumulator structure.** VREG file: 32 registers × 64 BF16
elements (`vreg_entries: 32`, `vreg_elements: 64` in `docs/isa_latency.json`
`profile`; `NUM_VREGS = 32` in `src/core/vpu/vpu_pkg.sv:53`). One VREG = 64
elements = `NUM_LANES(16) x NUM_SUBLANES(4)` = 128 bytes. There are exactly
**two asynchronous VREG read ports** (A and B) and **one logical write port**
(replicated by sublane for physical placement) — binary ALU is the only
consumer of both A and B; SFU, reduction, X-store, and matrix streaming all
compete for port A only (`docs/ISA_AND_INTERFACES.md` "The architectural VREG…"
paragraph, `docs/VLIW_SLOT_AUDIT.md`). There is no separate "accumulator
register file" distinct from VREG — MXU results land in ordinary VREGs via
`vmatpop`, and multi-tile accumulation (deeper than 16) is software `vadd` on
VREGs, which is exactly the cost TinyTPU's own GEMM lowering already pays with
its `vadd` accumulation over 4x4 tiles (`isa.py` `gemm`'s `new_acc` block) —
this is a structural parallel worth exploiting directly.

**DDR / DMEM interface.** DMEM is 2 GiB kernel-visible of a 16 GiB aperture,
persistent across launches, first-fit allocator with coalescing free, 32-byte
alignment (`docs/OPERATOR_SUPPORT.md` §1). The physical chain
(`docs/DDR_BW_UTIL.md`): DDR4 SODIMM @ 2133 MT/s×64b = 17.1 GB/s peak → MIG AXI
port 256b @ ~266.7 MHz = 8.53 GB/s → `minitpu_0/m00_axi` 256b @ 200 MHz = 6.4
GB/s → internal DM word is actually **512 bits** geared 2:1 down to the 256-bit
AXI width (`src/pkg/minitpu_config_pkg.sv:26` `DM_DATA_WIDTH = 256` — note the
doc's "512b internal" claim vs the package's `DM_DATA_WIDTH = 256` should be
read as: the doc describes a *pre-BF16-native* padded internal word; the
current `minitpu_config_pkg.sv` already shows `DM_DATA_WIDTH = 256`, i.e. BF16
lever 1 in `DDR_BW_UTIL.md` may already be applied or the doc predates this
package snapshot — **flagged as unverified**, worth reconciling against
`src/minitpu.sv:37,39` before quoting the "3.2 GB/s useful" figure as current).
DMA descriptors carry `base + row*stride` from two SREGs, asynchronous
issue/completion via `flush_slot`, and **no hardware arbiter or interlock**
between the DMA (D-slot) port and the compute (X-slot) port into VMEM — a
`SIM_ONLY` assertion is the only guard (`docs/VMEM.md` §5).

**VLIW slot structure and issue rules.** One 128-bit bundle per cycle, two
mutually exclusive physical formats (`src/core/sequencer/sequencer_pkg.sv`
header comment):
- **Compute format** (bit 127 = 0): E slot 19 bits (one ALU/SFU/reduction op),
  M slot 7 bits (one matrix command), X slot 40 bits (one VMEM load/store,
  encoding row + 7-bit sublane_stride + 4-bit sublane_mask + 2-bit AGU level +
  3-bit AGU shift + 13-bit SREG offset — see `docs/ISA_AND_INTERFACES.md`
  "architectural VREG" section and `docs/VMEM.md` §4), S slot 27 bits (scalar
  move/add/shift/MAC across 4 SREGs), F slot 27 bits (loop/control/delay/
  halt/swap), plus a 6-bit delay field.
- **DMA format** (bit 127 = 1): D slot 36 bits (one `vmemld`/`vmemst`
  descriptor, exclusive — cannot share a bundle with any compute slot), 6-bit
  delay, 85 bits reserved/unused.

Issue rule: **no hazard interlocks at all**. The sequencer only *holds the
fetch head* for the encoded `delay` field, matrix-controller busy
(`matrix_busy_o`), and halt-drain; everything else — VREG RAW, single
write-port collisions, WAW, SREG RAW, VMEM slot ownership — is a **static
scheduling responsibility carried entirely by the assembler**
(`board_package/asm.py`'s `schedule()`), never checked at runtime
(`docs/AGENTS.md` "The one thing that matters most"; `docs/SCHEDULING.md` §1).
A `saddi` decodes to no VREG reads/writes and MXU occupancy isn't in the RAW
model either — those are compiler-only obligations
(`AGENTS.md` "`_decode` models VREG hazards only").

**Instruction count and categories.** From `docs/OPERATOR_SUPPORT.md` §1
(sourced from `docs/isa_latency.json` and `sequencer_pkg.sv`), 6 slot
categories (E, X, M, S, D, F) covering roughly:
- E (10 ops): `vadd vsub vmul vmov vgelu vexp vrecip vrsqrt vredsum vredmax`
- X (2 ops): `vld vst`
- M (3 ops): `vmatload vmatpush vmatpop`
- S (1 mnemonic family, several forms): scalar move/add/shift/MAC
- D (2 ops): `vmemld vmemst`
- F (5 forms): loop begin/end, delay, `flush_slot`, `vmemswap`, halt

So on the order of **~25 distinct opcodes across 6 physical issue slots** —
structurally the same "6 units" shape as TinyTPU (`dma_load, dma_store, vload,
vstore, vpu, mxu` in `tinytpu/microarch.py`), but roughly 2.5x the opcode
count, plus the entirely new axis of *co-issue* (multiple slots active in one
128-bit bundle) that TinyTPU's strictly-sequential fetch-decode-dispatch loop
(`tinytpu/microarch.py` `tinytpu` top, one opcode dispatched per iteration)
does not have at all.

---

## 2. Per-instruction latency table (`docs/isa_latency.json`)

**Structure.** The file is the declared SSOT for the scheduling contract
(comment block at the top). Top-level keys:
- `profile`: `{num_lanes: 16, num_sublanes: 4, vreg_entries: 32,
  vreg_elements: 64}` plus a note that `W` is the sequencer's future
  single-write-port reservation offset and the earliest legal consumer is
  `L = W + 1`.
- `operations`: an array of records, each `{slot, ops[], vreg_reads,
  vreg_writes, w, rtl_param?, issue, consumer_override?}` — one record per
  *group* of opcodes that share a schedule (e.g. `vadd/vsub/vmul` share one
  record because they share the ALU pipeline).
- `rtl_params`: the six `WB_W_*` localparams these `w` values are generated
  from/checked against (`WB_W_ALU=5, WB_W_SFU=7, WB_W_REDUCE=15, WB_W_VLD=6,
  WB_W_MPOP_FIRST=3, WB_W_MPOP_LAST=6`).
- `matrix`: separate `occupancy` (controller-busy cycles per M command) vs.
  `result_latency` (cycles from `vmatpush` to first result) — explicitly
  flagged as "measured, not derived" via `tb_matrix_full_vreg` and a
  `MATRIX_BUSY` probe, and explicitly two *different* facts that are easy to
  conflate.
- `resources`: hardware *capacities* a schedule can silently exceed —
  `mxu_output_fifo.depth=32`, `loop_buffer.capacity_bundles=24` (`LB_CAP`),
  `delay_field.width_bits=6` (`DELAY_W`) — each cross-checked against
  `board_package/asm.py` constants by `tools/gen_isa_doc.py --check` and
  against RTL localparams by `tb_isa_conformance.sv`.

**Representative values** (condensed from the generated table in
`docs/ISA_AND_INTERFACES.md` / `docs/isa_latency.json`):

| Op(s) | Slot | W (writeback offset) | Earliest consumer L | Issue |
|---|---|---:|---:|---|
| `vadd`/`vsub`/`vmul` | E | 5 | 6 | II=1, one E op/bundle |
| `vmov` | E | 5 | 6 | II=1 |
| `vgelu`/`vexp`/`vrecip`/`vrsqrt` | E (SFU) | 7 | 8 | II=1 |
| `vredsum`/`vredmax` | E (reduce) | 15 | 16 | II=1; measured source-valid=13, W=15 adds 2 writeback-boundary cycles |
| `vld` | X | 6 (conflict-free) | 7 | II=1 conflict-free; **L = W + degree** under a bank conflict, degree ∈ {1,2,4} |
| `vst` | X | — | — | II=1; contends for port A |
| `vmatload` | M | — | — | 18-cycle command, holds port A |
| `vmatpush` | M | — | — | 5-cycle command; **first result at +82** |
| `vmatpop` | M | 3..6 | 7 | 7-cycle command; reserves write port for 4 masked beats |
| scalar move/add/shift/MAC | S | — | — | no dynamic RAW scoreboard at all |
| `vmemld`/`vmemst` | D | — | fence | async; `flush_slot` blocks on completion |
| loop/control/delay/halt/swap | F | — | — | halt drains all pending VREG producers |

Capacities: MXU output FIFO 32 entries/lane (4 per push, 4 per pop → 8 pushes
fill it); loop buffer 24 bundles (`LB_CAP`); delay field 6 bits (max 63-cycle
single gap, relevant because the 82-cycle push→result gap needs a filler
bundle).

This is directly analogous to TinyTPro's `tpu.latency(unit, ii=, depth=)`
declarations in `tinytpu/microarch.py` (e.g. `tpu.latency(mxu, ii=1,
depth=20)`) — MiniTPU's table is what those declarations look like once they
are *measured from silicon* rather than *asserted pre-synthesis*, and it adds
two things TinyTPU's model has no room for yet: (a) a **conditional** latency
(`vld`'s L = W + degree depending on an assembly-time-known bank conflict),
and (b) a **occupancy vs. result-latency** split for one instruction
(`vmatpush`), i.e. a unit that stays busy for a short window but produces its
result much later — a queue/pipeline-depth semantics, not a simple
depth-cycle model.

---

## 3. Staged scaling plan for TinyTPU

Ordered by discovery value per unit of implementation risk. Each stage names
concrete `isa.py`/`microarch.py` edits in the existing idiom.

### Stage 0 (prerequisite, ~0 new lines, do first): verify current cost model gaps
Not a code change — a gap-check. TinyTPU's `microarch.py` schedule declares
fixed `tpu.latency(unit, ii=1, depth=N)` per unit and treats every unit as a
simple fixed-latency pipeline. MiniTPU's `isa_latency.json` shows the real
machine needs at least two extra latency shapes (conditional latency,
occupancy≠result-latency) that TinyTPU's `UnitLatency`/`tpu.latency` API
(`allo/exp/dsa/core.py:438` `UnitLatency` class, `core.py:938` `ISA.latency`)
may not yet express. **Read `UnitLatency`'s fields before Stage 5** — if it
only stores `(ii, depth)`, the VLIW/multi-issue stage below cannot honestly
model MXU push/pop the way MiniTPU does, and that's worth knowing before
committing effort to the array-size stages.

### Stage 1 — Array 4x4 → 8x8 (not straight to 16x16)
**Discovery value: high. Risk: low-medium.**

`isa.py`: change `MATMUL_TILE = 4` to `8` (`isa.py` near `matmul`/`gemm`
definitions); the `view(bram, w, (1,4,4))` shapes in `matmul`'s `@I.access`
follow `MATMUL_TILE` automatically if parameterized (currently hardcoded
literal `4` in three places — `isa.py`'s `matmul` access region and the two
`view(...)` calls in `gemm`'s expansion — so this is a find/replace of a
handful of `4`s, ~10 lines touched).

`microarch.py`: `SYS_DIM = 4` → `8`; `mxu`'s `Xt`/`Wt` local buffers scale from
4x4 to 8x8 (16→64 registers when `Complete`-partitioned); the fully-unrolled
`k` loop in the dot product goes from a 4-deep adder tree to an 8-deep one,
multiplied across `i*j = 64` output cells instead of 16 — this is the part
that will *not* scale linearly in LUTs/DSPs under the current
fully-partition/fully-unroll schedule (`mxu_s.partition(..., kind=Complete)` +
`mxu_s.unroll("k")` in `microarch.py`). Expect the co-design loop's own
Vitis-HLS synthesis pass to show DSP/LUT growth close to O(SYS_DIM²) or worse,
which is itself a useful discovery for the agent: **the current MXU
microarchitecture is not a systolic array in the streaming sense at all — it
is a fully-unrolled combinational tile multiply** (all of X and W staged into
registers, then `i,j,k` triple loop with `k` fully unrolled and `j` pipelined,
`microarch.py`'s `mxu_s` schedule). MiniTPU's actual MXU
(`src/core/mxu/systolic_array.sv`, `mxu_pe.sv`) is a **weight-stationary,
temporally-streamed** array: each of the 256 PEs holds one persistent BF16
weight register and only ever sees one activation value and one partial-sum
value per cycle, propagated systolically west→east and north→south. Going to
8x8 or 16x16 in TinyTPU's current style (Stage 1 literally) will demonstrate
*why* MiniTPU doesn't scale that way — this is exactly the kind of "rediscover
the tradeoff" result the co-design loop should be able to produce, provided
someone reads the synthesis report afterward rather than just the cycle count.

New degrees of freedom opened: array size as a knob; unroll-vs-pipeline
tradeoff on the `k` reduction (agent can try `mxu_s.pipeline("k")` instead of
`unroll("k")` and observe the II/depth/area tradeoff directly); tile size vs.
BRAM port count (bigger tiles need more BRAM read ports if not fully
partitioned).

What could break: HLS synthesis timeout or resource blowup at 8x8+ if fully
unrolled; `VEC_LANES = 8` no longer evenly divides a bigger matmul tile
cleanly for the existing `vadd`-based accumulation path in `gemm` (today two
8-lane `vadd`s cover a 4x4=16-element tile exactly; an 8x8=64-element tile
needs 8 such vadds, which is mechanical but must be generated, not hand
written four times).

### Stage 2 — True systolic MXU: weight-stationary streaming, decoupled push/pop
**Discovery value: very high. Risk: medium-high.** This is the stage that
actually reproduces MiniTPU's MXU shape rather than just its size, and is
arguably higher-value than Stage 1's raw size bump.

`isa.py`: split `matmul` into two or three instructions mirroring
`vmatload`/`vmatpush`/`vmatpop` — e.g. `mxu_load(w)` (stage weights, no
output), `mxu_push(x)` (stream one activation row, produces one output row
some cycles later), `mxu_pop(z)` (drain one result row). This is a genuine ISA
redesign, not a parameter change: the current `matmul` is atomic
(access-then-compute in one instruction); the new form needs the systolic
array's internal state (stationary weights) to persist *across* instructions,
which the current `ISA`/`Instruction` model in `allo/exp/dsa/core.py` may or
may not support cleanly — check whether `@tpu.unit` state can be a
`tpu.entry`-local persistent array shared across multiple `@tpu.instruction`
bindings (TinyTPU's `bram`/`vreg` are already declared inside the top and
passed by reference to every unit, so a persistent weight register array
`wreg: f32[SYS_DIM, SYS_DIM]` declared the same way, mutated by `mxu_load` and
read by `mxu_push`, is plausible — but this needs the `tpu.bind(...,
trips=...)` model checked against multi-cycle result latency, which is new
territory).

`microarch.py`: replace the single `mxu` unit (~15 lines) with three units
(~60-80 lines) modeling load/push/pop separately, each with its own
`tpu.latency(unit, ii=1, depth=N)`, plus a small output FIFO array (even a
depth-4 or depth-8 shift register would demonstrate the concept without
needing MiniTPU's full 32-entry-per-lane depth) to decouple push-issue-time
from pop-issue-time.

New degrees of freedom: occupancy vs. result-latency as two separate
schedulable quantities (directly modeling MiniTPU's `matrix.occupancy` vs
`matrix.result_latency` split); output-FIFO depth as a tunable resource with a
measurable throughput/area tradeoff (this is literally MiniTPU's own
best-documented lever, `docs/SCHEDULING.md` §3.3, FIFO16→FIFO32 measured at
1.30x); push/pop interleaving as a schedule-level optimization the agent can
discover for itself, the same way `docs/SCHEDULING.md` §3.1 documents a human
discovering it (883→584 cycles/block from "push as early as the FIFO allows,
pop only when you have to").

What could break: correctness under the new sequential fetch-decode-dispatch
top if `mxu_push`/`mxu_pop` aren't correctly serialized relative to
`bram`/`vreg` accesses (the top's comment in `microarch.py` already flags this
class of hazard: "consecutive instructions carry dependencies through
bram/vreg, so it must not be pipelined"); getting the HLS-emitted FIFO to
actually behave as a real hardware FIFO under Vitis's dataflow/pipelining
directives rather than being optimized away.

### Stage 3 — fp32 → bf16
**Discovery value: medium-high. Risk: low.** Confirmed feasible: Allo's
frontend has a native `bf16` type (`allo/lang/core.py:277`,
`bf16 = APFloat(8, 7)`, importable as `from allo.lang.core import bf16`
alongside the `f32`/`i32` already imported in both `isa.py` and
`microarch.py`).

`isa.py`: swap `from allo.lang.core import f32` for `bf16` (or import both, if
DRAM stays f32 and only `bram`/`vreg` go bf16 — MiniTPU itself keeps DMEM
layout-agnostic and only the on-chip VMEM/VREG path is BF16-native,
`docs/OPERATOR_SUPPORT.md`'s "BF16 is the only VREG type"); change the three
`tpu.global_`/`tpu.scalar`/`tpu.vector` dtype arguments. ~5-10 lines.

`microarch.py`: change every `f32[...]` type annotation on unit signatures and
locals (`dmem`, `bram`, `vreg`, `Xt`, `Wt`, `acc`) to `bf16` where MiniTPU
keeps VMEM/VREG bf16, decide whether to keep the MXU accumulator wider (fp32
or a custom fp24-equivalent, matching MiniTPU's `MXU_ACC_W = 24`-bit
sign+8+15 accumulator) — this is the single most interesting *numerics*
decision in this stage and should be exposed as a real knob (accumulator
width vs. weight/activation width), not silently done in whatever type Allo
defaults to. ~15-20 lines.

New degrees of freedom: mixed-precision accumulation (bf16 in, wider
accumulate out) as an explicit, measurable choice — this is exactly MiniTPU's
own still-open Tier 3 item ("MXU accumulation across weight loads... built,
proven at 1.46x better accuracy... reverted at 8.6 points of LUT",
`docs/OPERATOR_SUPPORT.md` §4), so an agent that can freely try both and see
the LUT/accuracy tradeoff for itself is directly reproducing a real,
documented, costly decision MiniTPU's authors already made and *reverted*.
Also opens DDR-bandwidth-per-byte tradeoffs once paired with Stage 6.

What could break: Vitis HLS's bf16 support may be weaker/slower to synthesize
than fp32 arithmetic (fewer built-in library ops); numerical accuracy of
existing `verify.py`/`oracle.py` reference checks, which presumably assume
fp32 throughout — needs an bf16-aware tolerance, not a straight `rtol=0`
comparison (echoing MiniTPU's own hard-won lesson in
`docs/SCHEDULING.md` §6 point 3 about degenerate test operands hiding real
bugs).

### Stage 4 — Single scratchpad → banked VMEM (2-4 banks, MiniTPU-style)
**Discovery value: high. Risk: medium.**

`isa.py`: TinyTPU's `bram = tpu.scalar("bram", slots=BRAM_SIZE, dtype=f32)` is
a single flat address space with no banking concept in the `ISA` framework
itself (`allo/exp/dsa/core.py`'s `scalar()`/`vector()` builders don't expose
banks — confirmed by reading `core.py:804-836`). Banking in TinyTPU's idiom
has to be introduced as an **address-decomposition convention** at the
`view`/`contiguous` access-pattern level (e.g. redefine `vload`'s access
region so `bram_addr` decomposes into `bank = addr mod NUM_BANKS`, `bank_addr
= addr div NUM_BANKS`, mirroring `docs/VMEM.md` §2's bit-slice) rather than as
a first-class ISA feature — this itself is worth flagging to the agent
building this stage: **the underlying DSA framework has no banked-memory
primitive**, so this stage is necessarily hand-rolled address arithmetic in
the access regions, not a builtin. ~20-30 lines in `isa.py` (new helper for
bank/bank_addr decomposition, applied in `vload`/`vstore`/`dma_load`/
`dma_store`'s access regions).

`microarch.py`: change `bram: f32[BRAM_SIZE]` to `bram: f32[NUM_BANKS,
BANK_DEPTH]` and partition it with `mxu_s.partition(..., dim=1, kind=Cyclic,
factor=NUM_BANKS)` or, more directly, declare `NUM_BANKS` separate arrays the
way `vreg`'s slot dimension is already `Complete`-partitioned
(`top_s.partition(top_s.buffer("vreg"), dim=1, kind=top_s.Complete)` in the
existing schedule) — this is the cheapest correct route since TinyTPU already
demonstrates the "complete-partition one dimension to get independent ports"
pattern for `vreg`; doing the same for a `bram` bank dimension (not fully
complete — 4 or 8-way, not 8192-way) is the natural extension. ~15-25 lines.

New degrees of freedom: bank-conflict cost as an assembly-time-knowable
quantity (once addresses are static or affine, mirroring MiniTPU's
`docs/VMEM.md` §3 conditional-latency table) — this gives the co-design agent
its first *conditional* latency to reason about, rather than every unit being
a flat `ii=1` pipeline; number-of-banks vs. area vs. conflict-rate as a
three-way tradeoff the agent can sweep.

What could break: `dma_load`/`dma_store`'s current linear `contiguous(bram,
d, n)` access pattern (`isa.py`) assumes a flat address space; introducing
banks means DMA either needs bank-aware striding too (mirroring MiniTPU's
"DMA never uses the affine per-cycle AGU, it walks its own base/stride
autonomously" split, `docs/VMEM.md` §4) or must be restricted to
bank-boundary-aligned transfers only, which is a real design decision that
should be surfaced rather than silently hard-coded.

### Stage 5 — Scalar dispatch → 2-3 slot VLIW (M + one of {E, X})
**Discovery value: very high, but this is the biggest single jump. Risk:
high.**

`isa.py`: no change needed at the semantics level if instructions stay
independent operations — VLIW-ness is a *microarchitecture and encoding*
concern, not an ISA-semantics one (this mirrors MiniTPU exactly: `isa.py`
would stay describing individual ops; only how they're packed into a fetched
word changes).

`microarch.py`: the current `tinytpu` top (`microarch.py`, the `@tpu.entry`
function) is a single `for pc in arange(n_instr)` loop reading one
`[opcode, a0, a1, a2]` 4-word record per iteration and dispatching via one big
if/elif chain — the docstring explicitly says this loop "must not be
pipelined or made a dataflow region" because consecutive instructions carry
dependencies through `bram`/`vreg`. A real VLIW step means: (a) widening the
instruction record to carry 2+ independent op-fields per fetched word
(mirroring MiniTPU's 128-bit bundle with E/M/X/S/F sub-fields,
`sequencer_pkg.sv` header), (b) decoding each sub-field independently, (c)
dispatching to 2+ units **in the same cycle** rather than one per iteration,
which requires the units genuinely being independent HLS blocks called
concurrently rather than the current strictly-sequential dispatch — this is a
much larger restructuring than any earlier stage, likely 80-150 lines across
both files, and needs the same kind of static-hazard reasoning MiniTPU's
`asm.py`/`schedule()` does in software (`docs/SCHEDULING.md` §1's whole "every
hardware resource constraint is a software obligation" argument) since
TinyTPU has no separate assembler layer today — `program.py` may already play
that role and is worth reading before scoping this stage precisely.

New degrees of freedom: co-issue rules as a first-class design space (which
pairs of units may share a cycle — MiniTPU's answer is "M exclusive with
everything else that touches port A, D exclusive with all compute slots,"
`docs/ISA_AND_INTERFACES.md`); this is the single richest axis for an
optimizing agent, because it turns "which operations can be scheduled
together" into a search problem with real, measurable cycle-count payoff
(exactly what `docs/SCHEDULING.md` §7's `GemmSpec`/`GemmSchedule` split is
built to explore for MiniTPU) — and it is the discovery TinyTPU's design space
currently cannot produce *at all*, since one-opcode-per-cycle admits no
co-issue question to ask.

What could break: almost everything about the current schedule composition
(`top_s.compose(dl_s, ds_s, vl_s, vs_s, vpu_s, mxu_s)`) assumes each unit call
is a discrete, sequential call in the emitted top; two independent unit calls
issued "in the same cycle" is not a thing Allo's HLS-oriented scheduling
model may express directly (Vitis's own co-issue/ILP within one C++ statement
group is a very different mechanism from a real VLIW decode). This stage may
turn out to need a different backend (Allo's `rtl` export via CIRCT,
mentioned in `export_backend`'s docstring, "Kai's CIRCT RTL generator") rather
than the Vitis-HLS path the co-design loop currently scores against — worth
scoping as a research spike before committing implementation effort, and
flagged here as the highest-uncertainty item in this plan.

### Stage 6 — Accumulator tiles / deeper matmul accumulation
**Discovery value: medium. Risk: low-medium.** Natural follow-on to Stage 2:
once `mxu_push`/`mxu_pop` exist as separate ops, add an internal partial-sum
register array (`SYS_DIM x SYS_DIM` wider-than-input accumulator, e.g. fp32
accumulate over bf16 inputs) that persists across multiple `mxu_push` calls
before a single `mxu_pop`, directly modeling MiniTPU's still-reverted "MXU
accumulation across weight loads" feature (`docs/OPERATOR_SUPPORT.md` §4 Tier
3). ~30-40 lines: a persistent accumulator buffer plus an `mxu_accumulate`
flag/instruction variant. Opens the exact LUT-vs-accuracy tradeoff MiniTPU
measured and reverted (8.6 points of LUT for 1.46x accuracy) — since TinyTPU's
synthesis loop already measures LUTs, this is close to a direct
re-derivation opportunity.

### Stage 7 — Transformer ops: softmax, layernorm, GELU
**Discovery value: medium (necessary for realistic benchmarks, but low
architectural novelty on its own). Risk: low.**

`isa.py`: add `vexp`, `vrecip`, `vrsqrt`, `vredsum`, `vredmax` as new
`@tpu.instruction`s on `vreg` (mirrors MiniTPU's SFU/XLU op set almost
exactly — MiniTPU's own `vgelu`/`vexp`/`vrecip`/`vrsqrt`/`vredsum`/`vredmax`
are the literal reference implementation to copy the *semantics* of,
including the documented domain restriction: `docs/AGENTS.md` "The SFU's
domains are partial, and it clamps rather than faults" — **this is worth
copying verbatim as a design lesson**: MiniTPU's `vexp` is defined for
non-positive operands only and silently returns 1.0 outside its domain,
which is exactly the kind of footgun a small ISA should decide deliberately
rather than inherit by accident from a lookup-table or CORDIC
implementation). ~40-60 lines for 4-5 new instructions plus a `layernorm`/
`softmax` `@I.expand` macro analogous to `isa.py`'s existing `gemm` macro.

`microarch.py`: 2-3 new `@tpu.unit`s (an SFU-like nonlinear unit, a
cross-lane reduction unit) plus schedule directives; MiniTPU's reduction is
`docs/isa_latency.json`'s highest-latency E-op (W=15) because it crosses
*all 64 elements* rather than operating lanewise, which is a genuinely
different pipeline shape (a reduction tree, not a per-lane pipe) from
everything else TinyTPU currently has — a good forcing function for the
agent to discover that reductions cost meaningfully more than elementwise
ops. ~60-100 lines including reduction tree scaffolding scaled down from
`src/core/xlu/reduction_tree.sv`'s approach (parameter down from 64 elements
to TinyTPU's native `VEC_LANES = 8` for a first cut).

What could break: reduction across only `VEC_LANES = 8` elements (vs.
MiniTPU's 64) is a much smaller tree, so the interesting cross-lane latency
discovery may be muted until Stage 1 (bigger vectors) lands too — sequence
this stage *after* any vector-width increase, not before, or the "reduction
is expensive" lesson won't show up clearly.

### Stage 8 — KV-cache-friendly access pattern
**Discovery value: medium, but requires most other stages first (attention
needs matmul, softmax, and enough VMEM/banking to be meaningful). Risk:
medium.** Not recommended before Stages 1-4 land. When ready: `isa.py` needs
an append-only ring-buffer address pattern for K/V (a `view`/`contiguous`
variant with a wraparound modulus) and `microarch.py` needs a small persistent
KV-cache buffer inside the top. MiniTPU's own README explicitly flags this as
unbuilt ("Decode keeps no KV cache in either model" — decode throughput
degrades 1.10 → 0.33 tok/s over 128 tokens without one) — so this stage is
directly reproducing a known, *currently unsolved* MiniTPU gap, which makes it
a good target for genuine novel discovery rather than rediscovery, but only
once the prerequisite machinery (attention itself) exists.

---

## 4. MiniTPU design decisions worth deliberately exposing as rediscoverable tradeoffs

1. **Why 16x16, not bigger or smaller.** Not directly justified by a single
   doc line found here (flagged as **unverified** — no doc explicitly states
   "we chose 16 because X"), but the consequence is documented precisely:
   `docs/VMEM_LAYOUT_AND_BANKING.md` ties `NUM_LANES=16` to the whole VMEM row
   width (`16 lanes × 16-bit = 256 bits = one AXI beat`), so array width and
   DRAM burst width are coupled by construction in this design. A scaling
   agent that tries decoupling array width from VMEM row width (e.g. a 32x32
   array reading 16-wide VMEM rows over two cycles) would be rediscovering
   why MiniTPU picked the values it did.
2. **Why BF16.** `src/pkg/minitpu_config_pkg.sv`'s `BUILD_INFO_FORMAT`
   comment states it outright: "There is no FP32 state anywhere in this
   design. The widest numeric format is the MXU PE's private partial sum at
   MXU_ACC_W = 24 bits… fp32's extra 8 bits could only ever hold zeros
   there" — i.e. BF16 was chosen because the accumulator width was decided
   first (24 bits, driven by LUT/DSP budget) and BF16 is exactly what an
   8-bit-exponent format needs, with fp32's extra mantissa bits being pure
   waste given that accumulator. An agent that tries fp16 (10-bit mantissa,
   5-bit exponent) instead of bf16 at the same accumulator width would
   directly rediscover why exponent range, not mantissa precision, was the
   deciding factor for a transformer workload (activations/weights need
   dynamic range more than precision).
3. **Why `bank = row mod 4` and not an XOR-swizzle or bigger bank count.**
   Extensively argued in `docs/VMEM_LAYOUT_AND_BANKING.md`: the banking
   function is chosen to make the *actual* access pattern (stride-1, full
   mask, the native `(sublane,lane)` tiling) conflict-free, and every kernel
   in the whole GPT-2/Qwen pipeline uses exactly that pattern
   (`docs/VMEM_LAYOUT_AND_BANKING.md` §1's audit: 27 `vld`/`vst` sites, 0
   using non-default stride/mask). The doc's explicit verdict: "Do not
   change the banking… An alternative banking… is a bad trade for a machine
   whose entire workload is at stride 1." This is a rich rediscovery target:
   an agent given a banked TinyTPU and a benchmark suite should converge on
   "stride 1, small bank count" rather than reaching for exotic swizzles,
   *if* the benchmark suite's access patterns are realistic (see §5) — if
   the benchmarks are synthetic and stride-varied, the agent will instead
   "discover" that swizzling helps, which would itself be a useful negative
   result showing the benchmark suite doesn't match a real workload.
4. **Why no hazard interlocks.** `README.md`'s framing line captures the
   whole argument in one sentence: "No hazard interlocks: a legal schedule is
   a correctness argument, and the assembler carries it." The tradeoff is
   area (no scoreboard/replay logic — freeing LUTs that are the scarce
   resource at 83-84% utilization, `docs/IMPLEMENTATION_RESULTS.md`) against
   software complexity (the assembler must be exactly correct, and two
   documented silicon bugs — the SREG/matrix-occupancy hazard and the
   `flush_slot` mask mismatch — happened *because* nothing caught them at
   runtime, `docs/AGENTS.md`'s two bulleted incident reports). TinyTPU's
   `microarch.py` already leans this direction implicitly (the top's
   docstring: "consecutive instructions carry dependencies through bram/vreg,
   so it must not be pipelined") — an agent extending TinyTPU toward
   multi-issue will face the identical choice (build a scoreboard in
   hardware, or push correctness entirely into an assembler/scheduler layer)
   and MiniTPU's answer, plus its cost (two real board bugs), is a directly
   transferable data point.
5. **Why grow VMEM by adding slots, not rows.** `docs/VMEM.md` §8's bit-budget
   argument (`COMPUTE_RESERVED_W = 128 - 1 - 19 - 7 - 40 - 27 - 27 - 6 = 1`
   spare bit) is a clean, fully-worked example of "the binding resource is
   the instruction encoding, not the silicon" — doubling VMEM rows per slot
   needs exactly the 1 spare bit available; doubling VREG count needs 5 bits
   that don't exist; adding a slot needs 0 X-slot bits because slot selection
   already lives in the (spacious) DMA descriptor format. This is a directly
   portable lesson for TinyTPU's own 4-word `[opcode,a0,a1,a2]` instruction
   format: any scaling stage above should be evaluated first against "does
   this fit in the existing address-field widths" before touching HLS
   resource usage at all, since encoding width is a hard architectural wall
   that logic synthesis effort cannot buy back.
6. **Why VMEM is a staging buffer, not a cache, at these sizes.**
   `docs/OPERATOR_SUPPORT.md` §5's "Question the answer depends on" framing —
   VMEM barely matters in size as long as kernel boundaries stay one-op-at-a-
   time, but becomes the binding resource the moment kernels fuse (a whole
   transformer layer resident on-chip needs ~7680 of 8192 rows). This is a
   directly reusable framing for deciding *when* TinyTPU's own BRAM_SIZE
   (8192 words today) becomes worth growing: not as an isolated capacity
   question, but as a question about how many ISA instructions get fused per
   host launch.

---

## 5. Benchmarks to score against once scaled

MiniTPU's own operator/kernel inventory is the best source, both because it's
board-validated and because its docs already report *measured* cycle counts
that make excellent regression targets:

1. **Elementwise ops at scale** — `WL1` (32×64 BF16 add, 119 cycles measured,
   `docs/VLIW_SLOT_AUDIT.md`) and `WL2` (32×64 multiply-add, 155 cycles) are
   small, cheap, good smoke-test benchmarks for any vector-width increase
   (Stage 1/3).
2. **Tiled GEMM** — `WL3` (16×16 BF16 GEMM, 283 cycles, exact) and `WL5`
   (16×16 GEMM → GELU fusion, 253 cycles, "three observed M-pop/E-GELU
   co-issues") are the direct analogues of TinyTPU's own `gemm` macro and are
   the right target once Stage 2 (real systolic push/pop) lands — TinyTPU
   should be able to reproduce a comparable per-tile cycle count once its MXU
   is restructured, and the *rate* at which cycles/tile drops as tile size
   grows is itself a useful metric (this is what `docs/SCHEDULING.md` reports
   as "791 cycles, 583 bundles, 0.74 issued/cycle" for a plain GEMM before its
   own optimization stages).
3. **DMA-streamed compute** — `WL4` (32×32 DMA-streamed multiply-add, 929
   cycles) exercises the DMA/compute overlap axis (Stage 4/6 territory);
   good for stress-testing bank-conflict-free tiling once VMEM is banked.
4. **LayerNorm** — `WL6`/the "faithful 768-element LayerNorm" (941 core
   cycles measured in Verilator, 218 encoded bundles,
   `docs/IMPLEMENTATION_RESULTS.md`) is the best available *reduction-heavy*
   benchmark — a strong target once Stage 7 lands, since it specifically
   exercises the highest-latency E-op class (`vredsum`, W=15) plus ping-pong
   accumulation to work around the no-WAW-interlock constraint
   (`docs/VLIW_SLOT_AUDIT.md` "its first in-place version was rejected by the
   production assertion").
5. **Softmax with a padded pitch** — the bank-conflict micro-benchmark in
   `docs/VMEM_LAYOUT_AND_BANKING.md` §3a (natural pitch 32 → 126 cycles,
   padded pitch 33 → 118 cycles, `tools/check_softmax_packed.py`) is small,
   exact, and specifically designed to exercise the bank-conflict cost model
   — an excellent target for validating Stage 4's banking implementation is
   modeled correctly (it's a real 8-cycle, deterministic, well-isolated
   effect to reproduce).
6. **A full transformer block** — the GPT-2 124M "transformer block, 64x768,
   12 heads" accuracy figure (0.68% vs fp32 reference, `README.md`) and the
   Qwen2.5-0.5B decoder-layer figure (3.81%, 32x896) are the right *end-to-end*
   targets once Stages 1-7 are all in place — these are large asks for a
   from-scratch small-DSA rebuild, but even a scaled-down analogue (a single
   4-8 head, 64-128 dim block) run through the co-design loop would be a
   meaningful capstone benchmark, and MiniTPU's own accuracy numbers are the
   calibration point for "did precision loss compound acceptably."
7. **Batch invariance** as a correctness benchmark, not just a performance
   one — MiniTPU's "B=1..16 bit-identical" property
   (`docs/VMEM_LAYOUT_AND_BANKING.md` §6) is a cheap, strong invariant test:
   any TinyTPU scale-up that adds batching should be checked for the same
   property, and MiniTPU's own documented near-miss (an earlier version of
   the same doc stated the invariant without the row-wise-only qualifier,
   which was wrong for attention) is a specific pitfall worth encoding as a
   test rather than an assumption.

---

## Flagged uncertainties (not independently verified in this pass)

- The `docs/DDR_BW_UTIL.md` "3.2 GB/s useful, 512-bit internal DM word"
  narrative appears to be written against an earlier `DM_DATA_WIDTH`; the
  currently-read `src/pkg/minitpu_config_pkg.sv:26` already shows
  `DM_DATA_WIDTH = 256`. Whether "BF16 lever 1" from that doc has already
  landed, or whether the doc and the package have simply drifted, was not
  resolved — don't quote the 3.2 GB/s figure as MiniTPU's *current* number
  without checking `src/minitpu.sv:37,39` and the doc's own date against
  recent commits.
- No document found here explicitly states *why* the array is 16x16 (as
  opposed to, say, 8x8 or 32x32) beyond the consequence that it sets the VMEM
  row width; if a rationale exists it likely lives in commit history or an
  external design doc not covered by this file list.
- Whether Allo's `ISA`/`Instruction` framework (`allo/exp/dsa/core.py`) can
  express cross-instruction persistent hardware state (needed for Stage 2's
  weight-stationary split) was not tested — only inferred from the pattern
  already used for `bram`/`vreg` (top-local arrays passed by reference into
  units). This should be prototyped early in Stage 2, not assumed.
- Whether a true multi-issue VLIW (Stage 5) is reachable on Allo's Vitis-HLS
  backend at all, versus requiring the CIRCT/`rtl` backend mentioned in
  `microarch.py`'s `export_backend`, was not determined — flagged as the
  single highest-uncertainty claim in this report.
