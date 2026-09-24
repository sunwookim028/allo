# The Allo SystemC / Catapult-HLS Emitter (`EmitSystemC.cpp`)

This document describes the SystemC emitter targeting Catapult HLS. `target="systemc"` turns an Allo `@df.region` dataflow design into synthesizable **SystemC**
that Siemens **Catapult HLS** compiles to RTL.


*See `mlir/lib/Translation/EmitSystemC.md` for more details.*

---

## Background information

**Allo** lets you write hardware as a Python **dataflow region** — a `@df.region` whose body
constructs `Stream`/`Channel`/`Wire` links and calls `@df.kernel`s that talk over those links. The
Allo frontend lowers that to **MLIR** (the compiler IR): each kernel becomes a `func.func`
tagged `df.kernel`, the region top a `func.func` tagged `top`, and the links become Allo
dialect ops (`allo.stream_construct`, `allo.channel_get`, …).

**An HLS emitter** is the compiler stage that walks that MLIR and *prints C++ (or specifically SystemC)* — no RTL yet.
The C++ (or SystemC) is then handed to a High-Level-Synthesis tool (Vitis, or here **Catapult**) which
produces Verilog. For implementing the SystemC emitter, three emitters forming a **subclass chain** are used:

```
VhlsModuleEmitter  (Vivado/Vitis: ap_int, #pragma HLS ...)          EmitVivadoHLS.cpp  (~3400 lines)
   └── CatapultModuleEmitter  (swaps to ac_int, #pragma hls_...)    EmitCatapultHLS.cpp (~640 lines)
          └── SystemCModuleEmitter  (SC_MODULE + Connections links) EmitSystemC.cpp     (~3000 lines)
```

Each layer **overrides only what differs** and inherits the rest. Catapult is a thin subclass
of Vivado that swaps the vendor types (`ap_int` → `ac_int`) and pragmas (`#pragma HLS pipeline`
→ `#pragma hls_pipeline_init_interval`). SystemC subclasses Catapult and replaces the
*structure* (functions → `SC_MODULE`s) and the *links* (arrays/`hls::stream` → MatchLib
Connections), while still reusing the base for arithmetic, control flow, and most of the body.

**The three build modes** — the same emitted SystemC code, exercised three ways:

| mode | what runs | `__SYNTHESIS__` | what it proves |
|---|---|---|---|
| `csim` | g++ compiles + runs the SystemC on the host | not defined | functional correctness (fast) |
| `csyn` | Catapult synthesizes the SystemC → Verilog | defined | it synthesizes; area / Fmax |
| `cosim` | the synthesized RTL runs in Xcelium vs the csim "golden" | defined | RTL is bit-exact with csim |

**Key fact to keep in mind:** csim and synthesis compile *different code*. The emitter is using multiple (currently 5)
`#ifdef __SYNTHESIS__` splits. So a
green csim does **not** prove the RTL is right — run `cosim`.

---

## Design choices

**1. Subclass, don't rewrite.** `SystemCModuleEmitter` inherits from
`CatapultModuleEmitter` so it gets `ac_int`/`ac_fixed` codegen and Catapult loop pragmas for
free, and only overrides the ~26 handlers that must produce SystemC. (If it inherited from
Vivado directly it would emit Xilinx `#pragma HLS ...`, which Catapult ignores.) Every method
definition in `EmitSystemC.cpp` is tagged `// override (base emitter)` or `// new (SystemC-only)`
so you can tell reused-and-tweaked behavior from SystemC-specific behavior at a glance.

**2. Three link primitives, three hardware realizations.** Allo's links are not a
cosmetic label; each maps to genuinely different RTL:

| Allo link | RTL | handshake | buffered | use when |
|---|---|---|---|---|
| `Wire` | `sc_signal<T>` | none (combinational) | no | producer/consumer are cycle-locked |
| `Channel` (valid_ready) | `Connections::Combinational<T>` | full valid/ready | no | flow-controlled point link |
| `Channel` (valid_only) | raw `_dat` + `_vld` signals | valid only (no ready) | no | cheap one-way link; a race may drop a datum |
| `Stream[T, depth]` | `Connections::Fifo` (or `AlloFifoC`, its subclass, when `empty()`/`full()` are queried) | credit/handshake | yes | real elastic buffering between kernels |

`Known limitation:` a `Wire` gives **zero storage and zero alignment** — it should be used carefully;
`valid_only` **drops** a datum if the consumer
is not looking that cycle. Both are deliberate performance points, not bugs — pick them only
when you own the timing.

**3. `ac_int` everywhere, with an `ap_int` shim for the leftovers.** The reused Vivado
body still prints Vitis `ap_int` types and `x(hi,lo)` bit-slice syntax. Rather than override
every such site, the emitted header aliases `ap_int` to a thin **`ac_int` subclass** that adds
the two Vitis affordances (`(hi,lo)` and `>64-bit` narrowing). Under `__SYNTHESIS__` it collapses
to a plain `ac_int` alias. `Known limitation:` a design that bit-slices a *packed >64-bit* value
works in csim but can fail at `csyn` (Catapult rejects subclassing its builtin `ac_int`, CIN-15).

**4. A kernel's repeat loop becomes `while(1)` for synthesis.** A dataflow kernel's
outermost `for t in range(NUM_IT)` just means "run one step NUM_IT times" — `t` itself is
never used. If the emitter keeps it as a finite `for`, Catapult treats the body as one-time
setup code ("reset action") and won't pipeline it. Emitting it
as `while(1)` under `__SYNTHESIS__` instead gives Catapult the shape it *can* pipeline — one
step per clock, same results. The rewrite only fires when it's safe: the loop counter `t` is
unused *and* the kernel doesn't store to a memory port (a forever-loop would corrupt a
`C[i] += …` accumulator). A genuinely counted loop stays bounded. See "How it works" for the
exact guard.

**5. Random-access arrays become memory ports.** A 1-D array scanned strictly in order
(`a[i]` in one loop) is realized as a cheap **stream** (`a[i]` → `.Pop()`). Anything else (2-D,
strided, random, re-read) can't be a FIFO, so it becomes a **memory port**: an addressable
`AlloMem`/`AlloMemW` behind a req/rsp channel, indexed by a flattened row-major address.

---

## Working principle

```
@df.region  ──flatten──►  per-kernel SC_MODULEs  ──►  top wiring SC_MODULE  ──►  device header + tb
 (MLIR)      (inline        (one SC_THREAD each,        (instantiate kernels,      (type shims,
              sub-regions)   ports = links)              connect channels)          AlloMem/AlloFifoC)
```

**Stage 0 — flatten the hierarchy** (`flattenHierarchy`). A SystemC `SC_THREAD` can't
structurally instantiate a sub-region, so before emitting, every call to a *non-leaf* callee
(a sub-region, or a kernel that calls one) is **inlined** into the top, transitively, until the
top is a flat set of channel constructs + leaf-kernel calls. A design that's already flat is a
no-op.

**Stage 1 — each kernel → an `SC_MODULE`** (`emitKernelModule`). For a `@df.kernel`:
- Every argument becomes a **port**, classified from two frontend string attributes: `stypes`
  (stream/channel/wire dir) and `arg_dirs` (memref dir `i`/`o`/`b`). A directional 1-D
  sequentially-scanned array → a `Connections::In/Out` stream port; a random-access array → a
  memory port (`_req`/`_rsp`); a `Wire` → a raw `sc_in/sc_out`.
- A clocked `SC_THREAD run()` is registered with reset (`async_reset_signal_is` by default;
  set env `ALLO_SYNC_RESET` for synchronous reset — smaller flops on the ASIC path).
- `run()` resets its ports, then emits the kernel body. `get`/`put` on the body's links become
  `.Pop()`/`.Push()` (stream), `.PopNB()`/`.PushNB()` (`try_*`), or `sc_signal.read()/write()`
  (wire); `a[i]`/`b[i]=v` on a stream-ified array become `.Pop()`/`.Push()`.

  ```cpp
  // before (MLIR):  affine.load %a[%i] ; arith.addi ; affine.store %b[%i]
  // after (SystemC):
      ac_int<32,true> v = a.Pop();   // a[i]  -> Pop
      ac_int<32,true> r = v + 1;
      b.Push(r);                     // b[i]= -> Push
  ```

**Stage 2 — the region top → a wiring `SC_MODULE`** (`emitTopModule`). It declares a channel
member per link (`Connections::Combinational`, an `AlloFifoC` for buffered streams, an `sc_signal`
for wires), instantiates each kernel, and binds every kernel port to the right channel. Boundary
arrays become either top-level stream ports or internal `AlloMem`/`AlloMemW` memories (decided by
`regArgMemPort`). Each kernel's `done` is AND-ed into a top-level `done`.

**Stage 3 — the device header + testbench** (`emitModule`). Emits the C++ preamble: MatchLib
Connections includes, the float shims, the `ap_int` shim, and the component library — `AlloMem`
(read/write memory), `AlloMemW` (write-only), `AlloFifoC` (the buffered FIFO). Then a
testbench that drives the boundary streams from `input<N>.data`, runs the design, and checks
`output<N>.data`.

---

## The SystemC output

For the `add1` kernel above (`a: int32[8]` in, `b: int32[8]` out), the emitter produces:

```cpp
SC_MODULE(add1) {
  sc_in_clk clk;  sc_in<bool> rst;  sc_out<bool> done;
  Connections::In < ac_int<32, true> > a;     // 1-D, scanned in order -> stream In
  Connections::Out< ac_int<32, true> > b;     //                       -> stream Out
  SC_HAS_PROCESS(add1);
  add1(sc_module_name n) : sc_module(n), done("done"), a("a"), b("b") {
    SC_THREAD(run); sensitive << clk.pos(); async_reset_signal_is(rst, false);
  }
  void run() {
    a.Reset(); b.Reset(); done.write(false); wait();
#ifdef __SYNTHESIS__
    while (1) {                                 // steady-state loop (was `for i`)
#else
    l_steady: for (int i = 0; i < 8; i += 1) {
#endif
      ac_int<32,true> v = a.Pop();
      b.Push(v + 1);
#ifndef __SYNTHESIS__
      wait();                                   // csim: SC_THREAD must yield each step
#endif
    }
    done.write(true);
    while (1) { wait(); }
  }
};
```

The direct MLIR → SystemC mappings:

| MLIR | SystemC |
|---|---|
| `func.func @add1` (`df.kernel`) | `SC_MODULE(add1)` + `SC_THREAD(run)` |
| seq-streamable memref arg, dir `i`/`o` | `Connections::In/Out<ac_int<W,S>>` |
| `affine.load %a[%i]` / `affine.store %b[%i]` | `a.Pop()` / `b.Push(v)` |
| random-access memref arg | `AlloMem` req/rsp memory port |
| outermost `affine.for`, dead counter | `while(1)` (synth) / `for` (csim) |
| `allo.channel_get/put`, `allo.stream_*` | `.Pop()/.Push()`, `.PopNB()/.PushNB()`, `empty()/full()` sidebands |

---

## How to use it

```python
import allo.dataflow as df

# functional simulation (host g++):
mod = df.build(design, target="systemc", mode="csim",  project="out/csim")
mod(inputs...)

# RTL cosim (Catapult synth + Xcelium, checked bit-exact vs the csim golden):
mod = df.build(design, target="systemc", mode="cosim", project="out/cosim")
mod(inputs...)                                   # prints "cosim MATCH" on success
```

For synthesis area/Fmax, synthesize a single kernel as the top (excludes the testbench harness):

```bash
ALLO_DESIGN_TOP=<kernel>_0  python your_csyn_driver.py     # -> rtl.v + cycle.rpt
```

### Selecting the cell library — use `library`, not `device`

This flow targets an **ASIC standard-cell library**, so the config key is `library`:

```python
df.build(design, target="systemc", mode="csyn", project="out/csyn",
         configs={"clock_period": 2.0, "library": "nangate-45nm_beh"})
```

`device` is still accepted as a legacy spelling, but it is the wrong word here and it
caused a real bug: `device` means an FPGA **part** to `vitis_hls` (`u280`) and a **cell
library** to Catapult. A script driving both backends with one `--device` flag therefore
sent `u280` here, which became `solution library add u280` and failed deep inside a csyn
run with `Could not locate library file for library name u280`.

Resolution, designed so such a script is fixed by *adding* one key rather than
restructuring:

| configs | library used |
|---|---|
| `{"library": L}` | `L` — any `device` is the other backend's and is ignored |
| `{"device": L}` where `L` is not an FPGA part | `L` (legacy; custom ASIC libraries keep working) |
| `{"device": <FPGA part>}` with no `library` | **error**, naming the fix |
| neither | `nangate-45nm_beh` |

An FPGA part with no `library` errors rather than silently defaulting: quietly picking a
library the caller never chose would hand back an area number measured against the wrong
technology.

Environment knobs: `ALLO_SYNC_RESET` (synchronous reset), and — for the steady-state loop
transform — the kernel's outermost `for t` must have a **dead counter** and no memory-port store.

`Known limitation:` run `cosim`, not just `csim`, before trusting synthesis — the two paths
differ (loop shape, `ap_int` shim). See `mlir/lib/Translation/EmitSystemC.md` for the full list
of `#ifdef __SYNTHESIS__` divergences.

---

## Tests and Verification

The backend was validated by re-implementing the open-source **RaveNoC** hand-written RTL router
as an Allo design (`rvn_router.py`) and checking equivalence on RaveNoC's *own* captured traffic
(vanilla config: 2 VC, buffer 2, 2×2 mesh, XY routing):

- **Functional (csim):** routing (all 4 routers), a 256-flit wormhole packet, and VC-priority
  QoS under concurrent contention — all deliver losslessly to the RaveNoC-correct ports.
- **RTL (cosim):** the Catapult-synthesized Allo RTL is **bit-exact** with the csim golden on
  both synthetic contention vectors and the real 256-flit packet.

A `csim`-vs-`cosim` regression harness design (to catch `#ifdef`-divergence regressions) is
sketched in `tests/dataflow/COSIM_REGRESSION.md`.

---

## Files

The SystemC backend's dependencies (all required for `target="systemc"` to build and run):

| File | Role |
|---|---|
| `mlir/lib/Translation/EmitSystemC.cpp` | the emitter (this document's subject); companion `EmitSystemC.md` has line-referenced internals |
| `mlir/include/allo/Translation/EmitSystemC.h` | its public declaration (`emitSystemC`) |
| `mlir/include/allo/Dialect/LIMOps.td` | Wire/Channel/Stream ops (get/put/try/empty/full/construct) |
| `mlir/include/allo/Dialect/AlloTypes.td` | `WireType` / `ChannelType` / `StreamType` |
| `mlir/include/allo/Dialect/AlloAttrs.td` | `ChannelProtocol` (valid_ready / valid_only) |
| `mlir/lib/Dialect/AlloOps.cpp` | op verifiers/builders for the above |
| `allo/backend/hls.py` | the `target="systemc"` dispatch + `stypes`/`arg_dirs`/`unsigned` attr stamping |
| `allo/backend/catapult.py` | the `platform=="systemc"` build / csim / csyn / cosim flow |

Plus the dataflow frontend (allo/dataflow.py) that provides the `Wire`/`Channel`/`Stream` Python types and the
`emit_systemc` Python binding.
