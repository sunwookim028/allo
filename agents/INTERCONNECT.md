# Allo Interconnect Reference (agent-facing)

The contract an agent uses to **wire IP blocks together**. Blocks are fixed logic
with a typed port list (see each block's `PORT_SPEC`, e.g. `eva_stripped.py`); the
agent chooses the *connections* — which primitive, which topology, which
directions, what depth — subject to the rules here. A verifier checks the emitted
wiring against this reference + the blocks' `PORT_SPEC`.

Ground truth: `allo/ir/types.py` (types + methods), `allo/backend/hls.py` (backend
guard). Do not invent API outside this table.

## The three link primitives

| primitive | declare | write | read | buffered | latency | backpressure | backends |
|---|---|---|---|---|---|---|---|
| **Stream** (FIFO) | `Stream[T, depth][dims]` | `.put(x)` / `ok=.try_put(x)` | `.get()` / `v,ok=.try_get()` | yes (`depth`, default 2) | ≥1 | yes (blocks / `full()`) | **all** (incl. JIT simulator) |
| **Wire** | `Wire[T][dims]` | `.put(x)` (drive, same cycle) | `.get()` (sample, same cycle) | no | 0 (combinational) | none | **SystemC only** |
| **Channel** (handshake) | `Channel[T, valid_ready]` or `Channel[T, valid_only]` | `.put(x)` / `ok=.try_put(x)` | `.get()` / `v,ok=.try_get()` | no | 0 (rendezvous) | valid_ready: yes / valid_only: no | **SystemC only** |

## Op × primitive compatibility (do NOT emit an illegal cell)

| op | Stream | Wire | Channel |
|---|---|---|---|
| `put` / `get` (blocking) | ✓ | ✓ (same-cycle, never blocks) | ✓ |
| `try_put` / `try_get` (non-blocking) | ✓ | **✗** | ✓ |
| `empty` / `full` | ✓ | ✗ | ✗ |

`Wire` has **no** `try_*`/`empty`/`full` — a wire is always its current value, so
there is nothing to "try." `try_*` is the natural API for `Channel` (the handshake
*is* the poll) and for `Stream` (poll the FIFO).

## Backend availability (hard guard — `hls.py`)

- `Stream`: every backend, **including the JIT simulator** (`target="simulator"`).
- `Wire`, `Channel`: **SystemC backend only** (`target="systemc"`). Emitting them
  for any other target raises `NotImplementedError`.
- On the **simulator**, non-blocking (`try_*`) is only *deterministic* with the
  timing layer active (branch `wire`); otherwise outcomes are scheduler-decided.

## Emission mapping (SystemC)

- `Stream[T, depth>=1]` → `AlloFifo<T,depth>`
- `Stream[T, 0]` / `Channel[T, valid_ready]` → `Connections::Combinational<T>` (`.Push()`/`.Pop()`)
- `Channel[T, valid_only]` → **currently same as valid_ready** (Connections). The
  lighter `sc_signal + valid` lowering is a known TODO — do **not** assume
  `valid_only` drops backpressure yet.
- `Wire[T]` → bare `sc_signal<T>` (`.write()`/`.read()`)

## Rules / guardrails (the verifier enforces these)

1. **A bare `Wire` does not synchronize.** As the *sole* link between two
   independently-paced kernels it transfers garbage (consumer reads before
   producer drives). Use `Wire` only for side-band/payload bundles **gated by a
   companion `Channel`/`Stream` handshake**, or between kernels held in lock-step.
2. **Pure-wire-connected kernels should be fused into one kernel** (latency 0).
   Fusion *is* the lock-step guarantee that makes the wires correct.
3. **Every feedback cycle needs a registered element or priming.** A cycle of
   `Wire`/`Channel` (zero-latency) with no `Stream` (or no prime token) is a
   combinational loop → deadlock / non-convergence. Break it with a `Stream`
   (depth ≥1) or prime a token on one edge.
4. **All ports must be bound.** Every `in` port gets a source, every `out` port a
   sink, types and widths matching `PORT_SPEC`. Dropping a *direction* means
   dropping that port on **both** blocks it connected (and any driver/collector).
5. **Protocol must be legal for the backend** (table above). No `Wire`/`Channel`
   on the simulator; no `try_*` on a `Wire`.

## Choosing a primitive per connection

| the connection needs… | use |
|---|---|
| buffering / decoupled rates / lossless, works on simulator | **`Stream[T, depth]`** |
| lossless handshake, no buffer, SystemC | **`Channel[T, valid_ready]`** |
| zero-latency same-cycle value + a companion sync link | **`Wire[T]`** (+ handshake) |
| two kernels only ever wire-connected | **fuse them** (no link) |
| variable-rate poll (this-cycle-or-skip) | `try_*` on `Stream` or `Channel` |

## How a wiring plugs into a block

For each block instance the agent emits an `@df.kernel` whose loop body is:
`WIRE-IN` (read this instance's input ports from the chosen streams / topology
indices) → **`LOGIC`** (call the block's fixed `*_logic(...)`) → `WIRE-OUT` (write
output ports to the chosen streams), plus a top region declaring the links and the
edge drivers/collectors. `eva_pe_router_split.py` is one filled-in example (mesh;
`Stream` mesh + non-blocking local); the agent produces others (e.g. drop N/S for
east-west-only traffic, or swap the local link to `Channel[valid_ready]` on SystemC).

---

## Task spec / prompt schema (what the agent is *given*)

Structured, **not** one blob of prose. Fields:

```jsonc
{
  "backend": "systemc",                 // gates the primitive set (Stream|Wire|Channel vs Stream-only)
  "blocks": ["router", "pe"],           // + their PORT_SPEC (the fixed contract)
  "topology": {"type": "mesh", "M": 2, "N": 2},   // instances + adjacency (or explicit)
  "traffic": [                          // array-level -> per-connection: what actually flows
    {"flow": "systolic", "dirs": ["W->E"], "active": true},
    {"flow": "router",   "dirs": ["N","S","E","W"], "active": false}  // inactive => drop those directions
  ],
  "per_connection": {                   // requirements that pick the primitive + depth
    "sys_ew": {"bandwidth": "1 word/cycle", "flow_control": "lossless", "latency_max": null},
    "local_eject": {"bandwidth": "<=1/cycle", "flow_control": "handshake"}
  },
  "constraints": {"buffer_budget": "<= 32 words total", "primitives_allowed": ["Stream","Channel","Wire"]},
  "objective": "remove unused directions and minimize buffering while meeting bandwidth",
  "invariants": ["all_ports_bound", "no_unbroken_comb_loop", "feedback_primed", "types_match"],
  "output": "wiring_json"               // agent emits JSON; generator emits @top MLIR (below)
}
```

### Input lowering level (decided): **pre-lowered per-connection**

The prompt lists each connection with its direction / bandwidth / flow-control
already resolved. The agent's job is narrow and verifiable: **pick primitive +
depth per connection, and drop inactive directions.** It does NOT re-derive the
traffic pattern.

### Output representation: JSON → `@top` MLIR (canonical wiring)

The whole interconnect is a slice of Allo MLIR: `*_construct` ops (one per
connection) + `call @kernel(...)` ops (one per instance, operands bind
constructs to ports positionally). Example (`link_types_demo`):

```mlir
func.func @top(%A: memref<2x2xi32>, %B: memref<2x2xi32>) attributes {dataflow} {
  %0 = allo.stream_construct()  {name = "fifo"} : !allo.stream<i32, 4>
  %1 = allo.wire_construct()    {name = "wire"} : !allo.wire<i32>
  %2 = allo.channel_construct() {name = "chan"} : !allo.channel<i32, valid_ready>
  call @producer_0(%A, %0, %1, %2) : (...) -> ()
  call @consumer_0(%B, %0, %1, %2) {last} : (...) -> ()
}
```

Agent emits **structured wiring JSON** (1:1 with the above); a deterministic
generator emits the `@top` MLIR; then lower to systemc/simulator. Mapping:

| JSON | MLIR |
|---|---|
| connection `{name, primitive, dtype, depth\|protocol}` | `%k = allo.<prim>_construct() {name} : !allo.<prim><...>` |
| instance `{kernel, ports: [binding...]}` | `call @kernel(<memrefs + bound constructs>)` |

Why JSON→MLIR (not agent-writes-MLIR / agent-writes-Python): the agent stays in a
form that's trivial to **verify against `PORT_SPEC` + this reference** before any
codegen; the output lands as canonical, fully-typed IR directly consumable by the
lowering passes; and there is no MLIR/Python-API syntax to hallucinate. The
verifier typechecks the generated `@top`: every `call` operand's `!allo.<...>`
type matches the kernel port type, every port bound, no `wire` in an unbroken
cycle, feedback primed.

**Which infos belong in the prompt (and why):**
- **backend** — hard-gates which primitives are legal.
- **blocks + PORT_SPEC** — the fixed contract; every port must be wired.
- **topology / instances** — how many blocks and how they're arranged.
- **traffic pattern** — array-level flows → which per-connection directions are
  *active*; inactive ones are the "remove unused directions" opportunity.
- **per-connection requirements** — the levers that pick the primitive & depth:
  - **bandwidth target** (words/cycle or total) — the supervisor's example;
  - **flow-control need** — lossless(=Stream/Channel[valid_ready]) vs none(=Wire) vs lossy;
  - **latency budget** (optional) — tight latency favors Wire/Channel over deep FIFO.
- **global constraints** — area/buffer budget; allowed primitive subset.
- **objective** — what to optimize (min buffering @ bandwidth / min area by
  dropping directions / min latency). The agent needs an explicit goal.
- **invariants** — the correctness contract the output must satisfy (verifier).
- **output format** — Allo `top` directly, or a structured wiring JSON that a
  deterministic generator turns into Allo (recommended: easier to verify).
