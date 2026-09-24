# The Allo SystemC / Catapult‑HLS Emitter (`EmitSystemC.cpp`)

This document explains how Allo turns a `@df.region` dataflow design into synthesizable
SystemC for Siemens Catapult HLS. It provides more details than SYSTEMC_BACKEND.md. It is **layered**:

- **Part I — Overview** (everyone): what it is and the shape of what it emits.
- **Part II — Design author's guide** (Allo users): how your design maps to RTL, and the
  pitfalls that bite.
- **Part III — Maintainer's guide** (emitter devs): architecture, every mechanism with line
  references, and the Catapult‑specific workarounds.
- **Part IV — Assessment & improvement backlog.**

File: `mlir/lib/Translation/EmitSystemC.cpp` (~3000 lines). Entry point: `allo::emitSystemC`.
Reached from Python via `df.build(..., target="systemc", mode="csim"|"csyn"|"cosim")`.

---

## Part I — Overview

### What it is

A **textual SystemC emitter**. `SystemCModuleEmitter` subclasses **`CatapultModuleEmitter`**
(which subclasses `VhlsModuleEmitter`, the Vivado‑HLS emitter). It overrides the emit‑handlers
that must produce Connections‑based SystemC and **defers to the base class for everything
generic** (arithmetic, local arrays, functions, ac_int/ac_fixed codegen, Catapult loop
pragmas). It prints SystemC to an `os` stream — there is no intermediate representation of the
emitted code.

### What it emits

One `@df.kernel` → one `SC_MODULE` with a clocked `SC_THREAD run()`. The `@df.region` top →
one wiring `SC_MODULE` that instantiates the kernels and connects them with channels. Plus an
emitted **header preamble** (`device_header`) carrying type shims and the memory/FIFO
component library, and a **testbench** that drives the region.

```
@df.region  ──►  device_header (shims, AlloMem, AlloFifoC, ap_int/ac_int)
                 SC_MODULE(kernel_0) { ports; run(){ reset; body }; }   ◄─ per kernel
                 SC_MODULE(kernel_1) { ... }
                 SC_MODULE(top)      { channels; instances; bindings; }
                 SC_MODULE(tb)       { src/snk threads, memory preload, done poll }
```

### The three build modes

| mode    | what runs                                   | what it proves |
|---------|---------------------------------------------|----------------|
| `csim`  | g++ compiles + runs the SystemC (no `__SYNTHESIS__`) | functional correctness |
| `csyn`  | Catapult synthesizes the design → Verilog   | it synthesizes; area/latency estimate |
| `cosim` | Catapult RTL + SCVerify + Xcelium vs the csim golden | the **RTL** matches the C++ |

**Key fact:** csim and synthesis compile *different code* — the emitter uses 5
`#ifdef __SYNTHESIS__` splits (loop shape, the `ap_int` shim, `wait()` placement, memcpy
guards). csim passing does **not** guarantee cosim passing; run cosim.

---

## Part II — Design author's guide

### The three link primitives — pick deliberately

| Allo type | RTL form | handshake | buffered | non‑blocking | use when |
|-----------|----------|-----------|----------|--------------|----------|
| **`Wire`** | `sc_signal<T>` | none (combinational) | no | no (`try_*` rejected) | cycle‑locked producer/consumer; you own the timing |
| **`Channel`** (valid_ready) | `Connections::Combinational<T>` | full valid/ready | no | `try_get`/`try_put` | flow‑controlled point link, no storage |
| **`Channel`** (valid_only) | raw `_dat` + `_vld` signals | valid only, no ready | no | `try_*` (put always succeeds) | a fast one‑way link where a dropped datum is acceptable |
| **`Stream`** (depth ≥ 1) | `Connections::Fifo` (`AlloFifoC`) | credit/handshake | yes | `try_*`, `empty()`/`full()` | real elastic buffering between kernels |

- A **`Wire` gives zero storage AND zero alignment** — Only use it when you control the timing.
- **`Stream.empty()/full()`** read a synchronous **sideband** the emitter maintains, not the
  raw Connections flag (which is a sim‑only latched‑data flag that lags a cycle).
- A **`Stream` used by exactly one kernel** (that both produces and queries it) becomes a
  **self‑FIFO**: one kernel owns both the enqueue and dequeue ends, wired as a self‑loop. (special case, typically not used)

### Memory‑port arrays

A boundary array argument (`int32[N]`) becomes one of two things depending on how you use it:

- **Sequential scan** (1‑D, every access is identity `a[i]` inside exactly one loop) → a
  **Connections stream port**; `a[i]` reads become `.Pop()`, `b[i]=v` writes become `.Push()`.
- **Random access** (anything else, or fan‑in/out to >1 kernel) → a **memory port**: an
  `AlloMem`/`AlloMemW` behind a req/rsp channel pair. Loads become
  `req.Push(LOAD,addr); rsp.Pop()`, stores `req.Push(STORE,addr,val)`.

### Reset

Default is **async** reset (`async_reset_signal_is`). Set env **`ALLO_SYNC_RESET`** to emit
synchronous reset (`reset_signal_is`) — smaller flops.

### The loop shape (throughput)

Your kernel's outermost `for t in range(NUM_IT)` is treated as the **steady‑state loop**:
one iteration = one hardware step. Under synthesis it is emitted as `while(1)` so Catapult
**pipelines** it (a step every clock). If it stayed a finite `for` before the terminal idle
loop, Catapult would classify the whole body as *reset action* and never pipeline it. 
This transform only fires when it is safe (see Part III §2).

### Pitfalls (these have all caused real bugs)

1. **Consume every result.** A result‑producing op with no uses (`try_put`'s ok flag,
   `atomic_rmw`, …) is **silently deleted** by `MemRefDCE`. Always use the returned value.
2. **`~x` (bitwise NOT) is broken** — silently wrong. Use `(0 - x)` or an explicit mask.
3. **Narrow `UInt(N)` locals read back as signed** — values ≥ 2^(N‑1) flip sign silently.
   Keep one spare bit, or don't rely on the top half.
4. **A read‑modify‑write output array (`C[i] += ...`) forces single‑shot execution.** The
   kernel body runs once (see Part III §5); you can't free‑run/pipeline it, because a
   free‑running body would re‑accumulate every pass.
5. **Bit‑slicing (`x(hi,lo)`) does not synthesize on packed streams under the plain‑ac_int
   fallback** (CIN‑15). It works in csim; it can fail at `csyn`. Watch packed‑flit designs.
6. **int8 payloads fail to synthesize** — `signed char` has no MatchLib marshaller (CRD‑276).
   The emitter forces payloads to `ac_int<W>`, but be aware char‑width elements are special.
7. **`S.put(i)` with a raw loop index fails to build** (index vs i32) — assign the index to a
   typed local first.
8. **csim ≠ cosim.** Because of the `#ifdef __SYNTHESIS__` splits, always run cosim before
   trusting synthesis.

### Running it

```bash
# functional
df.build(design, target="systemc", mode="csim",  project="out/csim")(inputs...)
# synthesis (area/Fmax) — exclude the testbench harness from the top:
ALLO_DESIGN_TOP=<kernel>_0 python csyn_subdir.py <module> <region> <abs-project-dir>
# RTL cosim — uses the full region as top, needs a build SUBDIR (not the source dir)
df.build(design, target="systemc", mode="cosim", project="out/cosim")(inputs...)
```
Catapult In/Out ports degrade to raw signals if you run synthesis in the source dir instead
of a build subdir (SCHD‑30); `csyn_subdir.py` and `mode="cosim"` handle this.

---

## Part III — Maintainer's guide

### Class + setup

`SystemCModuleEmitter : CatapultModuleEmitter` (L243‑261). Constructor sets
`state.acFloatConstCtor = true` and `state.scfWhileWait = true`. Emission order in
`emitModule` (L2306): guard non‑dataflow modules → `flattenHierarchy` (inline non‑leaf
callees) → `device_header` preamble → each kernel module → top module → testbench.

### 1. Argument classification (L431‑534, setup L2333‑2417)

Direction comes from two frontend string attrs, one char per arg:
- `stypes` → `streamDir()` — `'i'`/`'o'` for Stream/Channel/Wire args.
- `arg_dirs` → `argDir()` — `'i'`/`'o'`/`'b'`/0 for memref args.

memref decision tree: `streamArgDir` (pure i/o **and** `isSeqStreamable` **and**
`!forceMemPort`) → stream port; else `memPortArgDir` → memory port; else internal array.

`isSeqStreamable` (L459) is the linchpin and deliberately narrow: rank 1, every use an
identity‑map affine load/store, each access inside **exactly one** loop. Three override sets
computed in `emitModule` refine it: `localStreamArgs` (self‑FIFO: stream used by 1 call),
`forceMemPortArgs` (seq‑streamable but used by >1 call → memory port, since 2 producers on one
Combinational aborts MatchLib), `streamArgAlias` (same channel to multiple arg slots of one
call → alias to avoid SystemC E115).

> **Fragility:** everything rests on frontend `stypes`/`arg_dirs`/`unsigned` string attrs with
> **silent 0‑returns on mismatch** — no validation. The three setup passes each re‑walk all
> funcs independently.

### 2. Loop‑shape transform (`isSteadyStateLoop` L534, `emitAffineFor` L577)

Fires only if **all** hold: enclosing func has `df.kernel`/`dataflow`; loop is outermost;
**induction var is dead** (a body reading `t` changes meaning); constant bounds; **no store to
a memory PORT** (walk stores, chase through `SubView/Cast/ReinterpretCast` to root; a
`BlockArgument` root = a port → disable). Stores to kernel‑local register arrays are fine.

Emission: `#ifdef __SYNTHESIS__ → while(1)` else `for(t=lo;t<hi;t+=step)`; body once; under
csim a trailing `wait()` is injected so the `SC_THREAD` yields and peers run.

> A prior attempt wrapped `while(1)` around the state **declarations** too → cold restart every
> pass, `buf` lost its resource path (`Unknown path '/router_0/run/buf:rsc'`). The correct fix
> leaves declarations above the loop and converts the loop in place.

### 3. Memory‑port access (L654‑1031)

A memory port packs a request word `ac_int<1 + ADDRW + DATAW, false>` (`scAddrW`=ceil(log2 N),
`scDataW`=element width). LOAD: `req.Push((idx)<<1); res = rsp.Pop()` (opcode bit0=0). STORE:
`req.Push((wdata)<<(1+ADDRW) | (idx)<<1 | 1)`. Floats transported as raw bits via `_fbits`.
Index flattening: `emitFlatIndex[Core|Memref]` (row‑major Σ idx·stride). Routing in
`emitAffineLoad/Store` + `emitLoad/Store`; local arrays fall back to the base emitter.

> `AlloMemW` (write‑only port) is float‑lossy (`int64` cast, L2664, deferred). `memPortArgDir`
> comment says only input LOAD is "fully wired" though store paths are emitted.

### 4. Bit manipulation (L840‑986)

`emitBitcast`: int↔fp via `.set_data`/`_fbits`; int↔int via same‑width `memcpy` on a temp
(the `void*` form is a CIN‑71 synthesis abort, so it's `#ifdef`‑guarded). `emitGetBit/SetBit/
GetSlice/SetSlice` each copy `num` into an `ac_int<nw, true> _bs_<rn>` temp first (a native
`int32_t` has no bit‑index/`.slc`/`.set_slc`).

> **Correctness smell:** the `_bs_` temp is hardcoded **signed** (`, true>`) at all four sites
> regardless of operand signedness — masking usually saves it, but slicing high bits of an
> unsigned value is a latent bug.

### 5. `emitKernelModule` (L1040‑1343) — the core

Ports: `sc_in_clk clk; sc_in<bool> rst; sc_out<bool> done;` then per‑arg ports (Stream →
Connections In/Out + optional `_empty`/`_full` sc_in sidebands gated on `streamArgQueried`;
self‑FIFO → `_enq`/`_deq` + a `localFifos` counter; Channel valid_ready → Combinational,
valid_only → `_dat`+`_vld` + generated modulario accessors; Wire → raw sc_in/sc_out; memref →
stream / `_req`(+`_rsp`) / `_req`‑only / internal array).

Reset via `alloResetFn()` (L142): `async_reset_signal_is` default, `reset_signal_is` under
`ALLO_SYNC_RESET`.

`run()` body: `Reset()` each stream port; zero self‑FIFO counters; write `0` to `wireOutPorts`
(CIN‑233: a driven `sc_out` must be set in reset); `done.write(false); wait();` emit
`static const` const‑globals (a 128KB weight array on the 64KB `SC_THREAD` stack segfaults —
`static` moves it to static storage / synthesizes as ROM); `emitBlock(body)`;
`done.write(true)`; `#ifndef __SYNTHESIS__ __allo_done++`; `while(1){wait();}`.

> **Single‑shot** run wrapper (body runs once) is orthogonal to §2: the *body's* `for t` becomes
> `while(1)`; the `run()` wrapper stays single‑shot so RMW memory accumulators are correct.

### 6. Link primitives (L1346‑1855)

- **Wire** (`emitWireGet/Put`): `sc_signal<T>` `.read()`/`.write()`. No non‑blocking form.
- **Channel** protocol‑dispatched by `isValidOnlyChannel`: valid_ready →
  `Connections::Combinational` `.Pop/.Push/.PopNB/.PushNB`; valid_only → raw `_dat`+`_vld` +
  the generated `_put/_try_put/_get/_try_get` methods carrying `#pragma design modulario`
  (they mirror the Connections reference exactly — wait **before** the valid test, sample
  after the edge; this was WRONG in a first attempt).
- **Stream**: `.Pop/.Push` (+ shaped‑stream nesting, whole‑block memport put); `try_*` →
  `.PopNB/.PushNB`; `empty()/full()` read the synchronous sideband/`_cnt`, not `In::Empty()`.

> **CRD‑304 `_nb` hack** in `emitChannelTryGet` + both `emitStreamTryGet` paths: `PopNB` takes
> `Message&`, so a native `int32_t` result won't bind; Pop into a payload‑typed `_nb` temp then
> convert. This once forced designs onto odd flit widths (26‑bit instead of 32). **Three
> near‑identical copies of this block.**

### 7. `emitTopModule` (L1899‑2300)

Ports (clk/rst/done + boundary arrays → In/Out or internal mem). Collect construct/call ops.
Channel members: depth‑0 stream → bare Combinational; depth≥1 → `_in`+`_out` + `AlloFifoC` (or
plain `Connections::Fifo` if never queried); dangling streams skipped (CONNECTIONS‑101/125).
Instances `uN` + per‑kernel `_done` signals. `memInsts`: one physical memory per (call,
memport arg), replicated per grid client. Constructor binds each operand **per the callee's
own emitted port**; `SC_METHOD(_agg_done)` ANDs all kernel `done` → top `done`.

> **Shared‑write replica merge** (L2964): arrays written by multiple replicas are split across
> replicas that wrote disjoint elements and **summed at readout** — fragile; assumes disjoint
> writes + zero‑init.

### 8. `device_header` preamble (L2419‑3007)

Includes (systemc, mc_connections, connections_fifo, ac_int/ac_fixed/ac_channel/ac_std_float,
`half`). Float shims (`_fbits`, `_mem_decode`, `sc_trace`), a `Wrapped<ac_ieee_float<>>`
marshaller specialization (matchlib ships it only for `ac_std_float`). The **`ap_int`/`ap_uint`
shim** (`ap_sel<W,Big>`): under csim a struct subclassing `ac_int` that adds `(hi,lo)` bit‑range
and >64‑bit narrowing; **under `__SYNTHESIS__` a plain `ac_int` alias** — because a struct
deriving from ac_int trips CIN‑15, so bit‑slicing a packed stream can fail at synthesis. Then
`AlloMem` (req/rsp), `AlloMemW` (store‑only), `AlloFifo` (dead reference/backup), `AlloFifoC`
(active FWFT FIFO, ~49% area, higher Fmax). `ALLO_SYNC_RESET` does a **literal string replace**
of `async_reset_signal_is` (hardcoded length 21). Signedness recovered by `linkPayloadUnsigned`
(scans defining op + users for `unsigned`, because allo builds signless `i22`);
`getStreamPayloadTypeName` forces ≤64‑bit int payloads to `ac_int<W>` (CRD‑276).

### Catapult / Connections / SystemC error‑code workarounds (re‑break risks)

| code | what it is | workaround (line) |
|------|------------|-------------------|
| CIN‑71  | `void*` cast rejected at synth | memcpy/`_fbits` `#ifdef`‑guarded (835, 2444) |
| CIN‑15  | struct assign from non‑struct (ac_int subclass) | plain‑ac_int alias under synth — **disables bit‑slice** (2558) |
| CIN‑233 | driven `sc_out` must be reset | `wireOutPorts` written in reset (1276) |
| CRD‑276 | int8 has no marshaller | force payloads to `ac_int<W>` (88) |
| CRD‑304 | `PopNB` non‑const ref | `_nb` payload temp (1447, 1691) |
| CRD‑413 | implicit >64‑bit narrowing csim‑only | explicit `.to_int64()` (1859) |
| SCHD‑30 | multi‑handshake II=1 / degraded ports | `-IO_MODE super`, build subdir (2698) |
| CONNECTIONS‑101/125 | dangling stream | skip unused streams (1968) |
| E115 | 2 `sc_out` on one signal | duplicate‑arg aliasing (418) |

---

## Part IV — Assessment & improvement backlog

**Strengths.** Clean subclass reuse (only overrides the SystemC‑specific handlers). A coherent
three‑primitive link model that maps to distinct, defensible RTL. Faithful Connections modeling
(the modulario valid_only accessors are exactly right, and the reasoning is documented). The
hard real‑world fixes are in and *explained* (done signal, const‑global ROM, loop shape, wide
words, FWFT FIFO). The code is unusually well‑commented — most hacks cite the exact tool error.

**What I would improve, in priority order:**

1. **Shrink the csim/synthesis divergence.** The `#ifdef __SYNTHESIS__` splits (loop shape,
   `ap_int` shim vs alias, `wait()`, memcpy) mean csim and cosim test different code, and
   correctness rests on "observationally identical" *arguments*. Add a small **standing set of
   designs that diff csim vs cosim head‑to‑head** in CI, and treat any new `#ifdef` split as a
   place that needs one.

2. **Move argument classification into the IR.** `isSeqStreamable`/`forceMemPort`/`streamDir`
   infer stream‑vs‑memory‑vs‑register from usage, with silent 0‑returns and a real bug history
   (unsigned ports, seq‑stream detection). Decide it **once as explicit dialect attributes** in
   the pass/Python layer; the emitter should read intent, not re‑derive it textually.

3. **One coherent signedness model.** `linkPayloadUnsigned` + `fixUnsignedType` + the hardcoded
   `, true>` bit‑op temps + signless `i22` have caused *silent wrong results*. This deserves a
   single source of truth rather than per‑site patches.

4. **Make the loop‑shape transform explicit, not inferred.** `isSteadyStateLoop` is a heuristic
   contract (dead IV + const bounds + no port store). Make it an **opt‑in kernel attribute**
   (or at least emit a note when a kernel *just misses* the guard) so intent is visible and the
   regression surface is bounded. Validate it across the whole dataflow suite before trusting it.

5. **Add golden‑file emitter tests.** Correctness currently rests on the slow csim/csyn/cosim
   sweep. Per‑construct **snapshot tests** (wire, channel valid_ready/valid_only, stream,
   memory port, loop shape, bit‑slice) catch emission regressions in seconds.

6. **De‑duplicate.** Index flattening exists ~4 times; the `_nb` temp block 3 times; the packed
   `reqT` string is rebuilt at 5+ sites; the three `emitModule` setup passes each re‑walk all
   funcs. Extract helpers — these are exactly where a subtle divergence hides.

7. **Isolate the Catapult‑version workarounds.** The emitter encodes a lot of Catapult‑2024.2
   knowledge (CIN‑/CRD‑/SCHD‑). Concentrate them behind named, documented helpers so a tool
   upgrade has one place to re‑verify. Drop the dead `AlloFifo` from every emitted header.

8. **Fix the known latent bugs.** The signed `_bs_` bit‑op temp; the `ALLO_SYNC_RESET`
   hardcoded‑length string replace (brittle to any rename); `AlloMemW` float‑lossiness; the
   shared‑write replica‑sum convention (silently corrupts on non‑disjoint writes).

9. **Fmax is a *design* limit the emitter could eventually help with.** One kernel = one
   `SC_THREAD` = one combinational recurrence (the measured 448 vs 547 MHz router gap). Not a
   workaround target, but a real future feature: emit a multi‑stage kernel as pipeline‑registered
   sub‑threads so a design can break its own recurrence without hand‑splitting into kernels.

**Open TODOs already in the code:** emit ac_int/ac_fixed natively and drop the `ap_int` shim
(L2544); sub‑region hierarchy (L2809); wire up memory‑port `'o'`/`'b'` cleanly (L504);
block‑streams (L1515).

**`@ Stateful` (done 2026‑08‑16).** A stateful variable lowers to a private `memref.global`
(`__stateful_` prefix + `static` attr + initial value). Vitis declares it as a function‑scope
`static`, correct for a function *called repeatedly*. An SC_THREAD is entered once and loops
internally, so `static` would be shared by every instance of the module AND skipped by the RTL
reset. The emitter instead declares it as an **SC_MODULE member** and assigns its initial value
in the **reset action**: per‑instance, reset‑initialised, and scheduled as a register. A member
rather than a `run()` local because an SC_THREAD body lives on a ~64KB coroutine stack, so a
large stateful array would overflow it and segfault — the same trap the const‑array block
documents. Array resets emit one loop nest (the frontend only allows a splat initialiser), not
an assignment per element. Uses need no rewriting — the base `emitGetGlobal` already binds the
SSA result to the global's symbol name.

**Measured: a function‑scope `static` and an SC_MODULE member synthesize IDENTICALLY**
(Catapult 2024.2, 2026‑08‑16). The same design was emitted twice — once with the member +
reset‑action assignment above, once hand‑edited to the Vitis shape
(`static int32_t x[4] = {0,0,0,0};` inside `run()`) — and put through the full flow. csim
correct both ways; csynth clean both ways; and the two `rtl.v` files are **byte‑identical
apart from the generation timestamp** (21 always‑blocks, 1242 lines). Catapult hoists the
static initialiser into the reset action and the registers get a proper async reset
(`always @(posedge clk or negedge rst) … if (~rst) buffer_0_dat <= 32'b0`).

Consequences worth knowing before touching this code:

- The claim that `static` "skips the reset" is **false for the synthesized path**. It is true
  only of the C++ semantics that csim executes.
- `static` also solves the coroutine‑stack overflow by itself (static storage is not the
  thread stack). What actually overflowed was the original *non‑static* local.
- The two forms differ **only in csim**: a `static` is shared by all instances of the class
  and initialises once per program; a member is per‑object and re‑initialises on every reset.
  Neither bites today — each kernel `func` becomes its own SC_MODULE class instantiated once,
  and the testbench asserts reset exactly once — so this is latent, not active. It would
  surface as csim/cosim divergence if one class were ever instantiated N times, or if a design
  reset mid‑simulation.
- So the member form is a **defensive** choice, not a required one. The minimal implementation
  would have been to stop skipping stateful globals in the const‑array block and let
  `emitGlobal` emit `static` (it already keys on the `__stateful_` name), for identical
  hardware. The member buys per‑instance storage and reset re‑initialisation against two
  things nothing currently does.

**Measured: an SC_MODULE MEMBER ARRAY CANNOT BE EXTERNALIZED by a resource directive**
(Catapult 2024.2, 2026‑08‑16). Worth knowing before anyone tries to expose boundary arrays
as `_rsc_*` memory pins the cheap way.

Catapult's own `examples/interfaces/ram_w_handshake` selects a RAM interface per array with
`directive set /mul_matrix/inMat:rsc -MAP_TO_MODULE {…_r}` — no source change at all. That
works because the design is a **C++ function** and `inMat` is an **argument**, so the array
inherently has an outside. Reproducing it on a SystemC member array fails three different
ways depending on how the array is used:

| array shape | outcome |
|---|---|
| read **and** written, live | resource `/top/run/arr:rsc` exists and the directive is accepted, but the component is rejected — `IFSYN-7 Invalid component selection` (the `_r`/`_w` components are single‑direction) |
| persists across reset | `CIN-233 'arr' must be set in reset action, preserving state across reset is not supported` — Catapult classifies a member array as **design state that must be reset**, the opposite of an external memory |
| read‑only, or write‑only | no external provenance or consumer, so it is proven dead and eliminated: `directive set: Unknown path '/top/run/arr:rsc'` — even when filled with real data from a port |

The last row is the crux: a member array has no outside, and there is no way to tell
Catapult it has one. **Only a PORT gives an array an outside**, which is precisely what
`ram_wire` provides (`sc_out<A> addr_read; sc_in<D> data_read;`). So exposing boundary
arrays as memory pins requires emitting ports — the directive shortcut does not exist here.

Useful side finding for when that is done: model the bundle on
`ccs_ramifc_w_handshake_{r,w}` rather than the bare `ram_wire`. Its pins are
`s_re, s_rrdy, s_raddr, s_din` (read) and `s_we, s_wrdy, s_waddr, s_dout` (write) — note
`s_rrdy`/`s_wrdy`, so the memory **can** stall the design. A bare `ram_wire` has no such
line and does impose fixed timing.

**Measured: a `modulario` method CANNOT contain a data‑dependent wait** (Catapult 2024.2,
2026‑08‑17). This is what stops the memory ready lines from being honoured.

The RAM‑pin bundles carry `_rrdy`/`_wrdy`, matching Catapult's own
`ccs_ramifc_w_handshake_{r,w}`. The obvious way to use them is to stall in the accessor:

```cpp
#pragma design modulario <in>
T A_rd(AddrT addr) {
  A_radr.write(addr); A_re.write(true);
  do { wait(); } while (!A_rrdy.read());   // <-- does NOT synthesize
  ...
}
```

That fails outright:

```
Top-down synthesis of C-CORE 'rev_0__A_rd' failed (ASM-2)
CHANOPERWRITE "io_syncw(ccs_ccore_done)" ... (BASIC-25)
```

A `modulario` method is a **fixed‑protocol C‑CORE**: its cycle behaviour must be static, so a
loop whose trip count depends on an input signal cannot be expressed. Not a throughput cost —
a hard build failure. Reverting to the fixed two‑edge read restores `cosim MATCH`, which
confirms the stall loop was the sole cause.

Why Catapult's own component *does* have `s_rrdy`: in the **function flow** the tool generates
the stall logic itself from `inMat[i]`. In the SystemC flow the accessor is ours to write, and
that is precisely what `modulario` forbids.

**Consequence, and it is a real limitation:** the pins are declared and `AlloMemPins` holds
them high, but the design does **not** honour them. Against a memory that is not always ready
it would sample `_q` before the data arrives. Making them real needs the stall *outside* the
`modulario` method, or an interface component that absorbs it as the vendor's does.

**Related trap in the same area:** a driven `sc_out` must be written in the reset action
(CIN‑233), and the emitter does that via `wireOutPorts`. Writing a literal `0` is wrong for a
float payload — `sc_out<ac_ieee_float<binary32>>::write` takes `const T&` and there is no
implicit `int`→`T` conversion, so it fails to compile only for float designs. `wireOutPorts`
therefore carries a per‑port zero expression. (The deleted `AlloMemW` used `_mem_decode<T>(0)`
for the same reason; removing it as dead code re‑introduced the problem until the suite caught
it on `test_producer_consumer`.)

**Not pinned to registers.** `hls_resource … map_to_module="[Register]"` is emitted from
`emitArrayDirectivesPreheader`, which runs off `emitAlloc`; a stateful array comes from
`emitGlobal` and so never gets the pragma. It is functionally correct but may map to a 1R1W
RAM — one access per cycle, and the zero‑area Genus artifact of `RESULTS.md` trap #2. Route
stateful arrays through the partition/pragma path before using `@ Stateful` for a router buffer. Two hooks made
this reuse the base's initializer formatting: `emitGlobalStorageQualifier` (new, suppresses
`static`) and `emitStatefulGlobalElementType` (so the declaration prints Catapult‑native types
rather than `ap_int`). **Not supported:** a region‑scope stateful shared by >1 kernel — shared
mutable state between concurrent modules, rejected with a diagnostic in `emitModule`. Note the
memory‑port path is **not** an escape hatch: it REPLICATES a shared array (one `AlloMem` per
client, writes summed at readout, L388), which reproduces the same silent disagreement. Real
support needs a multi‑client arbitrated memory.
