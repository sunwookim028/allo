# SystemC `comb` emission mode — scope

**Goal.** Add a second SystemC emission style for router-shaped kernels: emit them as
concurrent RTL (a combinational `SC_METHOD` driving raw handshake wires + a clocked
`SC_METHOD` holding state) instead of the current sequential `SC_THREAD` + `PushNB`/
`PopNB`. This is the only route that synthesizes when a kernel body issues several
handshakes (the `SCHD-67` fixed-iomode-offset collision). csim/csynth of existing
kernels is untouched — this is purely additive.

**Proven.** Hand-prototype `/tmp/route_proto/route.cpp` (the `drv` kernel re-modelled
this way) csynths clean — 0 `SCHD`, RTL produced, clocked method at 1 c-step. And Step 0
established the interconnect must be **raw `sc_signal` handshakes**, not Connections
ports: `disable_spawn()` is `#ifdef CONNECTIONS_SIM_ONLY` (gone under synthesis) and the
port's public API is transactions only — the raw `val`/`msg`/`rdy` wires aren't exposed.

## 1. Opt-in
- `@df.kernel(mapping=..., style="comb")` → set a `df.kernel.style = "comb"` StringAttr
  on the kernel func (in `allo/dataflow.py`, where `df.kernel` attrs are built).
- Emitter reads it in `emitKernelModule` (EmitSystemC.cpp:867): if `style=="comb"`,
  take the new path; else the existing `SC_THREAD` path (unchanged).
- **Opt-in first, no auto-detect.** A mis-qualified kernel mis-emits silently, so the
  user asserts eligibility. Auto-analysis is a later, separate step.

## 2. Qualification (assert-and-verify, not infer)
A `comb` kernel MUST be:
- **Non-blocking only** — every stream/channel op is `try_put`/`try_get` (no blocking
  `put`/`get`, which need a thread). Reject at emit time if a blocking op is present.
- **Single "time" loop** — the body is `[init state]; for t in range(N): <per-cycle
  logic>`. The outer `for` becomes the clock; there is no other multi-cycle sequencing.
- **State = registers** — carried values across `t` (arrays like `sp[]`, latched
  flit/valid) must be representable as registers; only bounded scalars/arrays.
- **Compile-time link indices** — already guaranteed by `meta_for` unroll.
Emit a hard error (not silent) if any of these is violated.

## 3. The transformation (IR → two `SC_METHOD`s)
Worked example = `drv3`. Mapping (validated in `route_proto`):

| Allo IR | comb emission |
|---|---|
| `for t in range(N):` (outer loop) | a `tcnt` **register**; `seq` runs one iteration/clock; NO `SC_THREAD` loop, NO `wait()` |
| carried state (`sp[]`, held flit) | `sc_signal` **registers**, updated in the clocked `seq` |
| `if cond: ch[i].try_put(x)` | **`comb`** drives `o_vld[i]=cond`, `o_dat[i]=x` (ALWAYS driven, else CIN-87 latch) |
| `... = ch[i].try_get()` | **`comb`** reads `i_vld[i]`/`i_dat[i]`; `i_rdy[i]` driven by accept logic |
| `if ok: sp[i]+=1` (accepted) | **`seq`** updates on `(cond && o_rdy[i])` (put) / `(i_vld[i] && accept)` (get) |
| `meta_for` unroll | `#pragma hls_unroll yes` |

Two processes in the module ctor:
```
SC_METHOD(comb); sensitive << <all state regs> << <all input vld/dat, output rdy>; dont_initialize();
SC_METHOD(seq);  sensitive << clk.pos();   // sync reset handled in-body
```

## 4. Interconnect (the added plumbing)
- **Ports.** A `comb` kernel's stream/channel args become **raw handshake port bundles**:
  producer end `sc_out<bool> _vld; sc_out<T> _dat; sc_in<bool> _rdy;`  consumer end the
  mirror. (New port emission in `emitKernelModule`'s arg loop, alongside the existing
  `Connections::In/Out` path — reuse `getStreamPayloadTypeName`/`linkPayloadUnsigned`
  for `T`.)
- **Channels.** A link between two `comb` kernels becomes a **raw `sc_signal` bundle**
  (`sc_signal<bool> vld; sc_signal<T> dat; sc_signal<bool> rdy;`) at top level, instead
  of `Connections::Combinational` + `AlloFifo`. New branch in the `channels` loop of
  `emitTopModule` (~line 1490) + the wiring loop.
- **Buffering.** Combinational routers (eva) are unbuffered / skid-buffered — a `comb`
  kernel holds its own skid register (part of its state), so a link between two `comb`
  kernels needs NO `AlloFifo`. Decision: **no FIFO on comb↔comb links** initially;
  a depth>0 `Stream` between comb kernels either (a) errors, or (b) later gets a
  raw-signal FIFO. Start with (a).
- **Mixed links (comb↔thread).** OUT OF SCOPE for v1 — require both ends `comb` (or a
  clear error). Bridging a raw bundle to a `Connections` channel is a later item.

## 5. Emitter changes (file/function map)
- `allo/dataflow.py` — accept `style=` on `@df.kernel`, stamp `df.kernel.style` attr.
- `EmitSystemC.cpp::emitKernelModule` — branch on style; new `emitCombKernel(func)`:
  - raw handshake ports (arg loop) + state-register members;
  - walk the body: hoist state to regs, split the `for t` body into `comb` (drives) and
    `seq` (updates), lower `try_put`/`try_get` to raw-signal reads/writes.
- `EmitSystemC.cpp::emitTopModule` — raw-signal channel members + wiring for comb links
  (guard: both ends comb).
- New op lowering helpers for the comb path (raw drive) — parallel to
  `emitStreamTryPut`/`emitStreamTryGet` but writing `_vld`/`_dat`/reading `_rdy`.
- The `#ifndef __SYNTHESIS__ wait()` for-loop logic (EmitVivadoHLS.cpp) is IRRELEVANT to
  comb kernels (no thread) — comb kernels never hit `emitScfFor`'s thread path.

## 6. Phases (each independently validated)
1. **Attr plumbing** — `style="comb"` reaches the emitter; no behaviour change yet.
2. **`emitCombKernel` on drv3** — hand the emitter the drv3 kernel, generate the
   two-`SC_METHOD` module + raw ports; diff against `route_proto`; csynth → `SCHD` gone.
3. **Raw-signal channels + wiring** — top-level bundles for comb↔comb; wire drv→col;
   csynth the whole `drv3` top; then **csim** it (the comb module must SIMULATE too —
   SC_METHOD comb + clocked seq must reproduce the transfer, cross-check vs the
   SC_THREAD csim result `[1,11,..]`).
4. **Router kernels** — mark `in_port`/`out_port`/`drv`/`col` `style="comb"`; csynth →
   RTL for the whole router; re-run the Channel router **csim T1–T4** to confirm no
   functional regression; if a reference (eva RTL) is available, diff.
5. **Hardening** — the qualify errors; a raw-signal FIFO for depth>0 comb links (opt);
   auto-detect qualifying kernels (opt).

## 7. Risks
- **csim correctness of the comb model** — synthesizing is necessary but not
  sufficient; the SC_METHOD comb+clocked pair must also simulate identically. Phase 3
  cross-checks against the known-good SC_THREAD csim. HIGHEST risk.
- **CIN-87 latches** — every comb output must be unconditionally driven; the emitter
  must emit default-assignments for all `vld`/`dat` before the conditional logic.
- **State explosion / large unrolled bodies** — many links × wide flits as registers;
  watch the architect memory (drv2 hit 37GB with dynamic-index memory — keep comb
  kernels memory-free or stage memory separately, per the earlier col_0 finding).
- **Scope creep** — mixed comb↔thread links and buffered comb links are explicitly
  deferred; v1 is comb↔comb, unbuffered, non-blocking, single-time-loop.

## 8. Definition of done (v1)
The Channel NoC router emits synthesizable RTL through Catapult (all partitions,
`SCHD` clean, `rtl.v` produced) with `style="comb"` kernels, AND still passes csim
T1–T4. Everything else (auto-detect, buffered comb links, mixed links) is follow-up.

Repro anchors: `/tmp/route_proto` (works), `/tmp/route_conn` (why Connections raw-access
fails), `/tmp/emit_drv3.py`→`drv3_prj` (the failing SC_THREAD case).
