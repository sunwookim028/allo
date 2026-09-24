# SystemC / Catapult backend — changelog and open design

Merged 2026-08-12 from `BACKEND_CHANGES.md` (landed changes) and
`SYSTEMC_COMB_MODE.md` (a proposal that was **never implemented**). Originals in
`archive/`.

Deeper references live next to the code they describe and are **not** duplicated here:
- `mlir/lib/Translation/EmitSystemC.md` — the layered emitter walkthrough
- `docs/SYSTEMC_BACKEND.md` — user-facing backend documentation
- `notes/CATAPULT_QUICKSTART.md` — the run recipe

---

## Part 1 — Landed changes

### Single-shot kernel execution (`EmitSystemC.cpp`)

Kernel `run()` bodies execute **once then idle** (`body; while(1) wait();`) rather than
free-running `while(1){body}`. This fixes `both` (read-modify-write) memory accumulators,
which previously re-accumulated on every pass.

csim gained an `__allo_done` counter (guarded out of `__SYNTHESIS__`): each kernel bumps
it after its single pass, and the memory-output testbench advances the clock until all
kernel instances complete before reading memories — replacing a fixed cycle count.

Net: +4 dataflow examples bit-exact (`tiled_gemm`, `pingpong_gemm`, `hierachical`,
`wrap_movement`); suite 28/28 unchanged; synthesis-neutral.

> Later work replaced the single-shot shape for pipelinable kernels — see
> `EmitSystemC.md` on the reset-action / `while(1)` loop shape. A finite loop still
> emits as a reset action and is therefore **not** pipelined; that is a known gap.

### ap_int shim: csynth regression fix (`EmitSystemC.cpp`)

The `ap_int`/`ap_uint` bit-slice shim was a `struct : ac_int<W>` subclass. Catapult
rejects assignment to an `ac_int`-derived struct (`ac_int.h:2259`, **CIN-15**), which
broke `go compile` on **every** design.

Fix: under `#ifdef __SYNTHESIS__` alias `ap_(u)int` to plain `ac_int<W,S>`; keep the
full-featured subclass — the `x(hi,lo)` bit-range and the >64-bit narrowing — only for
csim under `#else`.

Verified: `mem_port_reverse` (AlloMem) synthesizes end-to-end; pure-stream kernels
schedule cleanly; csim suite 28/28.

**Consequence, still open:** synthesis loses the csim-only bit-range, so packed-stream
designs fail locally at their `(hi,lo)` site until native `ac_int` `.slc` emission lands.
The >64-bit narrowing half was separately solved by `emitNarrowCastSuffix`, which appends
an explicit `.to_int64()`/`.to_uint64()` and works in both csim and synthesis.

> Root cause worth recording: `ac_int` narrows to `ac_int` at any width and refuses only
> to narrow to a **builtin C int**. The whole shim exists because `getSCTypeName` maps
> widths 8/16/32/64 to native `int32_t` etc. Emitting `ac_int` for those widths too
> would delete the shim, `ap_rng`, and the bit-slicing csynth gap together. The
> mechanism already exists as `BIT_FLAG` in `EmitCatapultHLS.cpp:32`; the cost is
> re-validating every measured design, not the edit.

---

## Part 2 — `comb` emission mode (PROPOSED, NOT IMPLEMENTED)

Verified 2026-08-12: no `style=="comb"` or `emitCombKernel` exists in
`mlir/lib/Translation/` or `allo/dataflow.py`. This is a scoped design, kept because the
Step 0 findings below are hard-won and still true.

**Goal.** A second SystemC emission style for router-shaped kernels: emit them as
concurrent RTL (a combinational `SC_METHOD` driving raw handshake wires plus a clocked
`SC_METHOD` holding state) instead of the sequential `SC_THREAD` + `PushNB`/`PopNB`.
Motivated by `SCHD-67` — the fixed-iomode-offset collision that blocks scheduling when a
kernel body issues several handshakes. Purely additive; existing csim/csynth untouched.

**Proven before the work stopped.** A hand prototype (`/tmp/route_proto/route.cpp`, the
`drv` kernel re-modelled this way) csynths clean — 0 `SCHD`, RTL produced, clocked method
at 1 c-step. And Step 0 established the interconnect **must be raw `sc_signal` handshakes,
not Connections ports**: `disable_spawn()` is `#ifdef CONNECTIONS_SIM_ONLY` (gone under
synthesis) and the port's public API is transactions only — the raw `val`/`msg`/`rdy`
wires are not exposed.

### Opt-in and qualification

`@df.kernel(mapping=..., style="comb")` stamps a `df.kernel.style` StringAttr, read in
`emitKernelModule` (`EmitSystemC.cpp:867`). **Opt-in only, no auto-detect** — a
mis-qualified kernel mis-emits silently, so the user asserts eligibility.

A `comb` kernel must be: **non-blocking only** (every stream op is `try_put`/`try_get`;
blocking ops need a thread), a **single time loop** (`[init]; for t in range(N): <per-cycle
logic>` — the outer `for` becomes the clock), **state representable as registers**
(bounded scalars/arrays), and **compile-time link indices** (already guaranteed by
`meta_for`). Violations must be hard errors, not silent mis-emission.

### The transformation (IR → two `SC_METHOD`s)

| Allo IR | comb emission |
|---|---|
| `for t in range(N):` | a `tcnt` **register**; `seq` runs one iteration/clock; no `SC_THREAD`, no `wait()` |
| carried state (`sp[]`, held flit) | `sc_signal` **registers**, updated in the clocked `seq` |
| `if cond: ch[i].try_put(x)` | **`comb`** drives `o_vld[i]=cond`, `o_dat[i]=x` (ALWAYS driven, else CIN-87 latch) |
| `... = ch[i].try_get()` | **`comb`** reads `i_vld[i]`/`i_dat[i]`; drives `i_rdy[i]` from accept logic |
| `if ok: sp[i]+=1` | **`seq`** updates on `(cond && o_rdy[i])` / `(i_vld[i] && accept)` |
| `meta_for` unroll | `#pragma hls_unroll yes` |

```
SC_METHOD(comb); sensitive << <state regs> << <input vld/dat, output rdy>; dont_initialize();
SC_METHOD(seq);  sensitive << clk.pos();   // sync reset in-body
```

### Interconnect

Stream/channel args become **raw handshake port bundles** — producer end
`sc_out<bool> _vld; sc_out<T> _dat; sc_in<bool> _rdy;`, consumer end the mirror (reuse
`getStreamPayloadTypeName`/`linkPayloadUnsigned` for `T`). A comb↔comb link becomes a raw
`sc_signal` bundle at top level instead of `Connections::Combinational` + `AlloFifo`.

Combinational routers are unbuffered/skid-buffered and a comb kernel holds its own skid
register, so **no FIFO on comb↔comb links** initially; a depth>0 `Stream` between comb
kernels should error. **Mixed comb↔thread links are out of scope for v1** — require both
ends comb, or emit a clear error.

### File map

`allo/dataflow.py` (accept `style=`, stamp the attr) · `EmitSystemC.cpp::emitKernelModule`
(branch on style; new `emitCombKernel` — raw ports, state registers, split the `for t`
body into drives and updates) · `EmitSystemC.cpp::emitTopModule` (raw-signal channel
members and wiring, guarded on both ends comb) · new raw-drive lowering helpers parallel
to `emitStreamTryPut`/`emitStreamTryGet`.

### Phases and risks

1. Attr plumbing (no behaviour change) → 2. `emitCombKernel` on `drv3`, diff against the
prototype, csynth until `SCHD` is gone → 3. raw-signal channels and wiring, then **csim**
the comb module → 4. mark the router kernels, csynth the whole router, re-run csim T1–T4
→ 5. hardening (qualify errors, optional raw-signal FIFO, optional auto-detect).

**Highest risk is csim correctness**: synthesizing is necessary but not sufficient, and
the `SC_METHOD` comb + clocked pair must simulate identically to the known-good
`SC_THREAD` result. Then **CIN-87 latches** — every comb output needs an unconditional
default assignment before the conditional logic. Then **state explosion** — many links ×
wide flits as registers (an earlier variant hit 37 GB of architect memory with
dynamic-index memory; keep comb kernels memory-free).

**Done (v1)** = the Channel NoC router emits synthesizable RTL through Catapult with
`style="comb"` kernels, `SCHD` clean, and still passes csim T1–T4.

Repro anchors, if they still exist: `/tmp/route_proto` (works), `/tmp/route_conn` (why
Connections raw access fails), `/tmp/emit_drv3.py` → `drv3_prj` (the failing SC_THREAD case).
