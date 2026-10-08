<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

<img src="tutorials/allo-icon.png" width=128/> Allo for Accelerator Co-Design
==============================================================================

[**Fork docs**](https://sunwookim028.github.io/allo/) | [**Upstream Allo**](https://github.com/cornell-zhang/allo)

This is the working tree of the next Allo: a co-design flow whose users are
**people and AI agents**. Upstream Allo's key idea is composable accelerator
design. This project carries it up the stack. Accelerators are regular: a few
unit kinds, repeated and wired together. Allo states that regularity as
composable abstractions, which are units, channels, instructions composed from
per-unit actions, and IP blocks. From one specification it derives the
compiler's model of the machine, the RTL, and the physical-design inputs, and it
checks that they agree. A change to the machine is answered with measured
feedback from compiler to silicon. Each abstraction is a **legality rule**: it
refuses a program that violates the machine property it states, and records what
it cannot prove as an obligation.

Today one design runs the whole stack. TinyTPU-isa is a TPU-style int8
accelerator written as Allo units. The ACT compiler maps unchanged PyTorch
models onto it, and every machine change is re-checked against PyTorch:

```bash
cd examples/tinytpu && make mlp      # PyTorch MLP -> ACT -> TinyTPU, stage by stage
examples/tinytpu/reproduce.sh        # functional gates + cosim cycle counts (~6 min)
```

The design target is [MiniTPU](#direction), a 16x16 BF16 VLIW TPU that runs
GPT-2 on an FPGA. It is to become a template, and TinyTPU one small instance
of it.


## Direction

**Main claim.** Centring the regularity of accelerator structures lets Allo
accelerate and automate the stack, from compiler through RTL generation to
physical design. The abstractions are the key, and composability (of ISA, RTL
modules and hard IPs) is how regularity pays off: faster co-design feedback,
faster iteration, better end results.

**Extended claims (agents).** Allo gives LLM agents faster and richer feedback.
It guides their exploration without constraining it. It supports data-driven
abstraction discovery. Agents can work in either of two roles: as a hardware
designer using the existing programming model, or as a tool developer improving
the modelling abstractions. The legality-check discovery runs are a case study.

**Design target.**
- **The template.** A MiniTPU-shaped unit library that composes at three levels:
  - *parameters*: shapes, widths, depths;
  - *optional modules*: accumulator file, special-function unit, transpose,
    scalar unit;
  - *swappable engines*: the multiply-accumulate plug-in, and the matrix
    engine (systolic or adder tree).
- **The instances.** MiniTPU is the full instance and TinyTPU the small int8
  instance. DotTree, and later Jalapeño, are instances with an adder-tree
  matrix engine.
- **What counts as a match.** The Allo MiniTPU must match the real RTL
  (`~/core/minitpu`, pinned at `b3ba0a4d`; same owner as this project): its own
  programs give bit-exact results, and its resources are reported beside
  MiniTPU's. The Allo version may add interlocks; that is a recorded deviation.

**Principles.**
- **Co-design needs timing and power across an array of architectures.** A
  flow that reports cycles for one design cannot support co-design decisions.
  Power is still absent on TinyTPU.
- **Two architectures are worth most where they disagree.** TinyTPU (int8, wide
  accumulator file) and MiniTPU (BF16, no accumulator file) are aligned only to
  probe the toolchain, never to make their numbers comparable. Their
  disagreement is itself a finding: an int8 machine that accumulates narrow has
  an overflow problem, not a rounding problem.
- **Generality is judged against named targets: Groq's LPU and OpenAI's
  Jalapeño.** An abstraction counts when it moves one of their mechanisms
  (adder-tree reduction, M down to 1, NUMA slice memory, explicit placement,
  two networks) from impossible to expressible (`docs/source/designs/ip_gaps.rst`).
- **Agent-authored extensions count even when small or rediscovered.** A
  rediscovery reached without being told is a valid result.

**Threads.**

| thread | holds today | next |
| --- | --- | --- |
| Composable regularity: `compose`, units, stream ports, ISA spec, Actions | TinyTPU (8 unit kinds, T=4/8), DotTree | the MiniTPU template; the unit library shared across designs |
| Legality rules and obligations: `s.dependence`, `s.encodable_on`, netlist rules | implemented and tested; one design consumes them | a calendar-shaped port rule; MiniTPU's timing contract |
| Compiler onto a changeable machine: ACT, PyTorch ingest | `make mlp`, bit-exact after each machine change | a MiniTPU target |
| Backends and physical design | Vitis cosim; Design Compiler flat flow; SystemC emit + csim | see D-1 |
| Measurement discipline: cosim, Design Compiler, commit pairing | TinyTPU vs Gemmini under one standard-cell flow | power; one clean-checkout path |
| Agent integration | harness with frozen gates and vouched verdicts; five paid runs | method left open (D-5) |


## Decisions

Dated, newest last. A decision changes only by a new entry that says which
entry it supersedes.

**D-1 (2026-10-01). Backend roles.**
- **Vitis HLS is the *measurement* backend** for the TinyTPU line (cycles,
  and RTL for the ASIC flow) until another backend reproduces those numbers.
  New design features must not depend on Vitis-only semantics.
- **SystemC → Catapult is the near-term *target* backend.** Allo emits SystemC
  and Catapult synthesizes it, on zhang-21. Here, only emission and a functional
  csim stand-in on pinned open-source libraries run; Catapult is the authority.
- **The open HLS stack is the long-term target.** That is Kai Shao's RTLGen
  plus [AMC](https://github.com/cornell-zhang/amc-dialect), emitting
  SystemVerilog for Verilator and the open-source ASIC flow.
- **A backend that cannot honour a directive must refuse it, not drop it.**
- *Supersedes* the 2026-09-09 assessment, which chose Vitis on the grounds that
  its persistent processes avoid RTLGen's per-instruction fill and drain. That
  assessment is in git at `35c38494`, `BACKEND_CHOICE.md`.
- *Reverses if* Catapult cannot reach II=1 on TinyTPU's pipelined loops.

**D-2 (2026-10-01). One home for each kind of record.**
- **Direction, decisions, milestones and their status** live in this README.
  No GitHub milestones.
- **Defects and limitations** stay in the fork's existing GitHub issues.
- **Evidence** lives in `dev/records/`, dated.
- **Repros** live in `tests/limits/`, linked from their issue.
- `docs/source/developer/limitations.rst` is a page of what a user must know,
  and links to the issues rather than tracking them.

**D-3 (2026-10-01). Repository layout, approved for M0.**
- `examples/minitpu/` is the home of the TPU unit library (the template). It is
  built there from U1. TinyTPU's `examples/tinytpu/ip/` stays in place, frozen
  (D-4), and is retired by the TinyTPU-instance track, when the instance imports from the
  template.
- SystemC gets a front door, `allo/backend/systemc.py`.
- New agent-harness code goes in a root-level `agents/`. The existing CHIA
  harness (`examples/tinytpu/chia_agent/`, `chia_abstraction/`) stays where it
  is, frozen like TinyTPU: its guards rebuild trees from git at older commits
  by path, so moving it would break replays. It moves or retires when a method
  replaces it (D-5). Almost all of it is ours; only the agent loop and the
  model client come from `ucb-bar/chia`.
- Designs stay flat under `examples/`, with no intermediate folder such as
  `accelerator/`.
- Fork-only knowledge stays in the Sphinx docs (`docs/source/`).

**D-4 (2026-10-01). TinyTPU becomes an instance; today's TinyTPU is frozen.**
- The current TinyTPU stays as the regression reference: published cycles,
  ASIC area, the Gemmini comparison, and the stress and mutation gates.
- It is retired only when the template instance reproduces it (the TinyTPU-instance track).
- `examples/minitpu/` is rewritten as the template. Its `reference.py`
  (arithmetic) and `program.py` (schedule rules) carry over.

**D-5 (2026-10-01). The agent method is open.** CHIA is the current instance,
not a commitment. Harness work goes into the method-agnostic core first.

**D-6 (2026-10-01). Settle before building.**
- No design work until M0 is on `main`.
- *Amended by D-7:* the expressiveness probes (P) are replaced by the unit
  ladder.

**D-7 (2026-10-02). Build and validate MiniTPU unit by unit.**
- A single-shot port of the full core is not attempted. The milestones climb
  MiniTPU's own separability order (`docs/UNITS.md` §2): arithmetic leaves,
  storage with ports, datapath composites, control, then the core. The real
  units replace the synthetic probes of P.
- **Validated** means bit-exact against the RTL unit, and equal to the unit's
  declared latency, both by a differential harness that drives the Allo unit
  and the Verilator-wrapped `.sv` module with the same stimulus. MiniTPU's own
  `tb/` is a second check. Cycle-by-cycle agreement is checked only once a unit
  has Catapult RTL (the Catapult track).
- MiniTPU is the same owner's design, pinned at `b3ba0a4d`. Changing it (e.g.
  adding interlocks) is an owner's decision, recorded here when made.


**D-8 (2026-10-02). The ladder runs on zhang-21.**
- The unit ladder (U1-U5) and the Catapult track both run on zhang-21, which has
  Catapult, Xcelium and Vivado (with the ZCU104 part). ace-01 stays usable as a
  second host, through the csim stand-in.
- zhang-21 needs two installs, both pinned (`dev/toolchains.rst`):
  - Verilator 5.052, one exact conda-forge build, installed by
    `scripts/verilator-setup.sh` into its own prefix. MiniTPU's README names
    5.051; at its pinned commit its 14-testbench unit suite passes on 5.052
    (ace-01, 2026-10-02).
  - `torch==2.14.0` CPU, for the ACT flow.
- MiniTPU is cloned there by its owner, at the pinned commit.

**D-9 (2026-10-02). The ladder is a probe of the tools.** *Amends D-7.*
- The main product of each unit is what it exposes in each tool: the Allo
  simulator, the programming model and its passes, the SystemC emitter and
  Catapult, RTLGen, AMC. Matching the RTL stays the goal of each unit, but a
  unit is done when every tool has been tried on it and every difference is
  explained and triaged, not only when it matches.
- Each unit fills one row of a matrix: one column per tool, each cell *match*,
  *finding*, *blocked* or *n/a*. A finding is classed as a tool **bug**, a
  **missing abstraction**, a **workaround** (it works only by distorting the
  unit) or a **semantic mismatch** (the tools disagree on what the unit means).
  The matrix is evidence (`dev/records/minitpu/`); defects go to issues (D-2).
- The owner is in the loop at three points per unit: how the unit is expressed
  in Allo, before coding; triage of the filled matrix (fix now, record, or
  research); and any change to the programming model or to passes, which
  becomes a D-n entry before code. Mechanical work does not wait.
- Expect more research and design iterations in this phase than in M0. Work
  is split across parallel agents by tool track; one session integrates and
  is the only writer to `main`. Feature code goes on a branch in a worktree
  and merges after review.
- When the owner is away, the agent decides from these entries, records each
  call as *provisional* with its reason, and keeps going. Changes to the
  programming model or passes stay proposals until reviewed.

**D-10 (2026-10-02, provisional until the owner confirms). A unit's latency
is reported by its backend and checked on its RTL; it is constrained only
where a contract demands.**
- Every backend that produces RTL writes `latency.json` beside the build:
  per unit, `latency` (input-accept edge to output-visible edge, no stall),
  `ii`, the port style, and a `status` with a reason. A number whose status
  is not `scheduled` is not consumed. Catapult and Vitis do this now; RTLGen
  and AMC already report it and get adapters at M2.
- A latency table that ACT or an assembler consumes is built from manifests,
  per (unit, backend, clock), never a constant in the source: the same adder
  is 2 cycles in Catapult, 5 in Vitis and 1 in AMC at 5 ns.
- The harness checks every RTL unit's measured latency and rate against its
  manifest. A MiniTPU-declared latency is reported beside it and decides a
  verdict only when the unit is pinned.
- `latency=L` on a kernel is optional, for a contract outside the tool
  (matching MiniTPU at D-7; cycle-locked composition at U3). Catapult honours
  it on stream ports and refuses a Wire kernel, `L < 1`, or an infeasible L,
  quoting the cause. Vitis meets L by missing the clock, so Allo refuses when
  the estimate exceeds the target. RTLGen and AMC refuse until they grow a
  bound.
- `comb` is a port shape, not `latency=0`: Catapult emits a combinational
  CCORE; RTLGen and AMC refuse it.
- Evidence: `dev/records/minitpu/latency_report_2026-10-02.rst`.

**D-11 (2026-10-02). `Stateful` state persists across calls on every
backend, and one `Stateful` has one kernel.**
- A `Stateful` array or scalar keeps its value from one call of the region
  to the next, on the simulator, in SystemC csim and in any cosim Allo
  drives. Found at U2: the simulator already persists, SystemC csim and
  RTLGen cosim restarted per call (`u2_regfile_2026-10-02.rst`, M1).
- Two kernels sharing one `Stateful` is refused at build, on every backend.
  The simulator ran them unordered with no warning (M2); Vitis refused;
  SystemC refused. A memory with several ports is a different thing, and is
  the subject of the memory-port proposal that follows.
- Verdicts in the unit ladder use single-call traces until the
  implementation lands (branch `core-stateful`).

**D-13 (2026-10-02). A combinational output is declared, never inferred.**
- A unit port written as `Wire[T, comb]` (or `Wire(..., comb=True)`) is a
  same-cycle output: its value is a function of the unit's `Wire` inputs and
  of state loaded before any store in the iteration, with no register on the
  path. SystemC emits it as a combinational process (`SC_METHOD`) over
  signal storage; Catapult then builds a clockless block. Measured at U2: a
  register file with three such read ports is bit-exact against MiniTPU's
  `vpu_regfile.sv` at read latency 0, write visible after 1
  (`dev/records/minitpu/u2_comb_read_2026-10-02.rst`).
- A backend that cannot build it refuses, naming the port: RTLGen and AMC
  always register an output. The emitter refuses a `comb` cone that reads a
  stream, a store of the same iteration, or anything clocked.
- `latency.json` reports such a port as `comb`, not `0`; a `comb` port and
  `latency=` are different things (D-10).
- Explicit rather than inferred: an inferred form would change a port's
  latency silently when its cone changed shape, which D-1 forbids.
- Generated storage is reset (Catapult requires it; MiniTPU's register file
  is not): a recorded deviation, reported as the reset-flop share of
  sequential area, unless an unreset form proves well supported.

**D-14 (2026-10-02). Unreset storage is declared, never inferred.**
- Storage is reset by default. `Stateful(..., reset=False)` declares storage
  whose contents survive reset (only its control is reset), as MiniTPU's
  register file and FIFO contents are.
- SystemC lowers it to a clock-edge write process with no reset action and
  `-RESET_CLEARS_ALL_REGS no` for Catapult. Measured at U2 on `vpu_regfile`:
  0 reset flops, bit-exact, DC area 4,021.9 um^2 against MiniTPU's 4,021.7
  (`dev/records/minitpu/u2_comb_wire_impl_2026-10-02.rst`, F3).
- A backend that cannot leave storage unreset refuses, naming the storage.
  The simulator treats it as ordinary storage; harness verdicts mask its
  pre-write contents as undefined (as for MiniTPU's).

**D-12 (2026-10-02). A memory declares its ports; each port has one owner.**
- `compose.Memory(rows=, ports=(Port(name, kind, latency, visible, count),),
  collision=)` declares an on-chip memory. `kind` is `r`, `w` or `rw`;
  `latency` is the read latency in edges, `0` meaning asynchronous;
  `visible` is the edges until a write is seen on any port; `count` gives
  that many interchangeable ports (as AMC's port type does), default 1.
  A `Memory` without `rows` stays today's boundary array.
- A unit binds ports, not memories (`memories=("vreg.ra", "vreg.rb")`). Each
  port has exactly one owner; this replaces one owner per memory. A port
  that several writers share in the hardware (MiniTPU's OR-muxed VREG write
  port) is owned by one writeback unit fed over channels. Composition checks
  each body against its ports' direction and accesses per iteration.
- `collision=` states what a same-word access on two ports in one cycle
  means: `refuse`, an `obligation` on the composition checked by stress
  cosim, or `undefined` (masked in verdicts). VMEM's compute/DMA collision is
  an obligation (MiniTPU issue #21).
- The per-cycle port calendar is derived from the declared latencies at the
  Actions layer and handed to the assembler and ACT; composition checks
  structure only.
- A declared read latency `L >= 1` lowers to exactly the read pipeline that
  matched MiniTPU on every RTL backend at U2 (the pipe written as data);
  `latency=0` lowers to a combinational read (D-13).
- Every backend honours a port, refuses it naming the port and the cause, or
  lowers it through a stated implementation (replica, registers, SRAM),
  reported in `memory.json` beside `latency.json`; none drops it. Vitis
  refuses any memory with more than one owner for now.
  `allo.memory.Memory(latency=, depth=)` is refused until it becomes the
  one-port shorthand of this declaration.
- Evidence: `dev/records/minitpu/d12_memory_ports_2026-10-02.rst`,
  `u2_regfile_2026-10-02.rst` (G1, H3), `u2_word_array_2026-10-02.rst`.
- *Reverses if* the regfile prototype (three reader units and one writeback
  unit) cannot match MiniTPU per cycle on Catapult, or a two-port VMEM
  cannot reach II=1.

**D-15 (2026-10-04). An engine is declared, and a swap that changes the
accumulate order is a different function.**
- A swappable engine is a record, not a function: operand, accumulator and
  output types with their widths; `mul`/`add`/`pack` bodies; their latency as
  a D-10 `latency=` on those bodies; the accumulate `order` it is exact for;
  the same arithmetic in numpy; and the directives it needs of a schedule. A
  unit binds the record's names as parameters (`MAC_IN`, `MAC_ADD`, ...),
  never a bare function.
- A matrix engine declares its `order` (`sequential` | `tree`). What a
  composition books is the declared latency; what a backend built is
  `latency.json`. Neither is a constant in a body.
- The contract reference of a composite takes the order as an argument. An
  engine swap that keeps the order is verified against the same reference;
  one that changes it is verified against the reference evaluated with the
  new order, and recorded as a different function, as `ip_gaps.rst` does for
  a narrower node type. Measured at U3: systolic and tree differ on 74 of
  8,192 bf16 outputs at DIM 16 on random data, 0 on exact-sum data, 0 at int8.
- MiniTPU's instance admits one order, `sequential` acc24 with one
  `pack_bf16` (P-5); an instance with an adder-tree engine declares `tree`.
- Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §1,
  prototype `examples/minitpu/template/`.

**D-16 (2026-10-04; re-keyed 2026-10-08). The ladder models MiniTPU
`b3ba0a4d` (ISA `v1`) through U5, then re-pins.**
- MiniTPU now has three ISA versions (`minitpu-comp`, master `a9757be`):
  `v1-course` = the frozen course ISA (Lab 2's released tree, timing of
  minitpu `613190d`: result latency 82, four-beat vmatpop), `v1` = master
  **and our pin** (minitpu-comp, 2026-10-08: `05e1bdf`'s one-beat vmatpop and
  85-cycle result latency, `49d895d`'s 8-deep loop stack and `3bcf0b7`'s
  4-bit agu_shift are all ancestors of `b3ba0a4d`; U4 track B measured all
  eight `WB_W_*` equal to the `v1` base). The "82" the ladder measures is
  push->valid at the MXU port (2 + 5*DIM); `v1`'s 85 is counted from the
  vmatpush's issue to the issue of the first vmatpop that finds its result:
  85 = 82 + 4 (the last of the four pushed rows lands at +4) - 1 (the pop
  engine's own register), measured on the pin (U4 track B, §10). `v2` = `docs/ISA_V2.md` (branches, one zero-overhead
  loop, post-increment addressing, semaphores, a fault register, and
  interlocks). E03 (freeze-on-stall; new `en_i` ports and a VMEM landing
  ring) is in flight on branch `e03`.
- U4 and U5 model the control that exists in RTL at `b3ba0a4d`. E03 and v2
  become declared variants afterwards: new ports as declared ports, v2's
  interlocks as optional modules with their own contract. D-7's note stands:
  an interlock in the Allo MiniTPU is a recorded deviation until the pin
  moves to a version that has it.
- The shared truth for ISA timing stays MiniTPU's `docs/isa_latency.json` +
  `isa_slots.json`; Allo's `latency.json`/`memory.json` feed a generator that
  writes a version's deltas there, with `--check` failing on disagreement.
  The schema is agreed when `minitpu-comp`'s `versions` branch lands.
- Arithmetic contracts are unchanged across v1, E03 and v2; open items there
  (vrecip wrap ISA-N03, vmax/vmin NaN ISA-X02) change our references only
  when decided.
- Track E's composition drafts, labelled D-16..D-19 in their record, take the
  next free numbers when adopted.

**D-17 (2026-10-04). A unit is instantiated, and the instantiation binds its
parameters, channels and engines.**
- `compose.Architecture` instantiates a `Unit` under an instance name with a
  binding of the unit's free names: parameters (`DIM`, `N`), channels,
  engines (D-15). One `Unit` object composes any number of times in one
  region; two instances at two bindings is what the Design target's
  "parameters" level means.
- A binding names only free names of the body, and every check `compose`
  already makes (declaration == body, one owner per channel, legality on the
  bound parameter set) runs per instance.
- `@df.unit` gets the same at the front end: `unit[P0, P1](...)` type
  parameters as kernels have, resolved at the instantiation and type-checked
  by the `wiring-type` rule. Until then a `@df.unit` sized by a module global
  is frozen at decoration (C9), and a factory (C11) is the recorded
  workaround for leaves only.
- A parameter in a slice bound (`word[0:MAC_IN_BITS]`) is refused today; a
  bound unit converts by typed assignment until the front end folds
  constants into bounds.
- `compose`'s binding (a renaming of the body's free names) and the front
  end's type parameters are two mechanisms for one idea; the front-end form
  subsumes the renaming when it lands, so the renaming layer is not kept.
- Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §2
  (two PE instances, bf16 and int8 engines, in one region: simulator
  512/512 + 512/512, csim 256/256 + 256/256); `u1_alu` C9/C11.

**D-18 (2026-10-04). A schedule belongs to the function that needs it and
travels with it.**
- A function or engine that needs a directive to meet its declared latency
  carries that directive (`Engine.directives`); every unit that binds it
  applies it, through `Architecture.directives`, without naming the
  function's internals. A region's `schedule()` names only what the region
  adds.
- A function the front end builds as its own `func.func` has nameable loops
  (`leading_zeros19:offset`), so one directive covers every caller; an
  inlined function does not, and a directive on it is refused, not dropped
  (D-1).
- The SystemC emitter carries `pipeline` (Catapult's II pragma), `unroll`
  (`hls_unroll`), `partition` (`hls_resource [Register]`) and `latency=`
  (the I/O cycle constraint); any other directive is refused naming the
  function. A backend that cannot honour a carried directive refuses it the
  same way.
- The front-end form, a schedule attribute on the function itself applied
  when the function is built, follows as separate work.
- Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §3
  (one unroll on the shared `leading_zeros19` covers every PE instance);
  `u1_alu` C10; `u3_track_a` A7/H4; the C-W1 fix.

**D-19 (2026-10-04). An optional module is a declared delta, and an
instance's ISA is the slots its modules bring.**
- An optional module is an `Option`: the units and channels it adds, the
  rebinding of its neighbours' ports when it is present, and the ISA slots
  that exist only with it. An architecture is a base plus options; no unit
  list is built by a Python conditional. This is `generate if` with the ISA
  attached: the form Rocket Chip's and Gemmini's configuration parameters
  take, which MiniTPU's RTL does not yet have (its SFU is a fixed stage).
- Composition is legal iff the netlist rules pass on the result: a module
  left out whose channel stays declared, or added without its rebind, is
  refused at composition, naming the channel.
- The assembler of an instance refuses an instruction whose module is not
  composed in, naming the module; `gen_isa --check` holds the instance's
  spec to its options, so the ISA table is derived from the composition
  rather than written beside it.
- The front end holds every region to the netlist rules, nested kernels
  counted as endpoints (E1, in fix batch 3, is the gap).
- Use cases in the two designs on hand: MiniTPU's SFU (`vpu.sv:88`, one
  `sfu_group` per sublane fed from VREG port A, tag pipe `:157-184`; slots
  `vexp`/`vgelu`/`vrecip`/`vrsqrt`), transpose (`vpu.sv:132`; `vtranspose`),
  reduction tree (`vreduce`/`vlanered`) and `perf_counters`
  (`minitpu_core.sv:137`); TinyTPU's accumulator file (`ip/units/accumulator`,
  channels `c_acc`/`ac2sp`), which MiniTPU lacks. The SFU is the first
  implementation. The RTL-side counterpart (`generate if` in `vpu.sv`, the
  decoder's refusal) is synced with the `minitpu-comp` session's v2 work.
- Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §4
  (H15); prototype `examples/minitpu/template/optional.py`.

**D-20 (2026-10-04). A derived parameter is a property, and every relation it
rests on is a legality condition.**
- A geometry is a frozen record whose derived numbers (`LEVELS`, the tap
  level, the switch span, push->valid, a composite's latency) are properties
  computed from the declared ones and the bound engines' declared latencies;
  none is a field, so none can be typed in beside the number it must equal.
- Each relation a unit's correctness rests on is a `legality` on that unit,
  run at composition, naming the parameter and the consequence; never an
  assertion in a testbench (MiniTPU's own tap relation lived only there,
  `UNITS.md` §5, §8.3).
- A derived latency is a booking (D-10): the harness compares it with the
  manifest's measured value per (unit, backend, clock) and a difference is a
  recorded verdict. The assembler reads the booking from the composed
  instance, never a constant. This is what makes a later re-pin (D-16) safe:
  a changed adder latency moves every dependent number through one relation.
- Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §5;
  prototype `examples/minitpu/template/legality.py` -- twelve derived numbers
  equal Phase 0's measurements (push->valid 12/22/82, span 5/15/75, PE 4,
  `vmatpush` 85, tree 13/9 and 9/5), five wrong declarations refused.

**D-21 (2026-10-08). A short-term working baseline: MiniTPU's RTL obtained
from Allo, mostly as RTL IP; and Allo's MXU integrated into MiniTPU.**
- `examples/minitpu/rtl/` makes one demonstration claim: a MiniTPU is built
  from an Allo design in which most or all units are the real RTL, wrapped as
  IP (PR #48's `RTLModule` where its ready/valid adapters fit; a less general
  shim for the core's AXI top where they do not), driven by an Allo program
  end to end. The whole-core shim is planned first; the hybrid (the real
  `mxu.sv` inside the ladder's Allo units) runs in parallel. The feature may
  be less general than the ladder's units; it is a baseline, not the ladder.
- The inverse angle counts equally: the MXU Allo already models (U3) is
  synthesized by Allo and integrated into MiniTPU's RTL tree in place of
  `mxu.sv`, held to MiniTPU's own testbenches. Its latency differs from the
  shipped MXU's, so it is a declared ISA version in the agreed
  `versions.<name>` seam, not a silent change.
- Neither replaces the ladder (U4, U5) nor moves the pin (D-16).
- Evidence as it lands: `dev/records/minitpu/minitpu_rtl_*.rst`.
- Settled after the probes (owner, 2026-10-08; `minitpu_rtl_plan_2026-10-08.rst`,
  `pr48_probe_2026-10-08.md`, `minitpu_rtl_m0_2026-10-08.rst`):
  (1) the baseline wraps the **core seam** (`minitpu_core.sv`: a ready/valid
  memory pipe, IRAM port, start/done, two CSRs), not the AXI top; the AXI top
  is an optional later milestone. (2) **PR #48 is merged** once `main` is
  clean, and its debts are fixed in one fork follow-up commit (the Verilator
  5.052 `--json-only` port, flag pass-through, a stall knob, docs to Sphinx,
  the Verilator pin); the shim may extend it. (3) The headline claim is the
  whole core's: "MiniTPU RTL obtained from Allo, all units the real RTL"; the
  hybrid (real `mxu.sv` inside Allo units) runs in parallel; Allo's MXU
  inside MiniTPU's tree is queued behind Catapult II=1 and the `versions`
  seam. (4) The gate is **bits**: the drain bit-identical to MiniTPU's own
  testbench digest (`examples/minitpu/rtl/oracle.json`); cycles are reported
  beside the testbench's, a difference is a recorded finding. One looped GEMM
  first, then the `sim_kernel.py` set.
- Re-scoped (owner, 2026-10-08): the wrapped core is not a demo but the
  **substitution spine** for U4/U5. Once `minitpu_core.sv` runs inside an
  Allo region against the testbench-digest oracle, each Allo-modelled unit
  replaces its RTL counterpart inside the same region -- the MXU first, then
  the VPU units, then the control -- with the same oracle on every step,
  until the core is all Allo (U5). M-R3 is therefore the first swap, not the
  AXI top. Its probe target is the tool gap that a transaction-level IP seam
  cannot carry a per-cycle sideband (`vpu_ctrl_t`): the first swap is RTL
  inside the same Verilated model (a file-list replacement), and a
  cycle-level mixed RTL/Allo co-simulation seam is what a later swap needs.
- Status: M-R1 and M-R2 passed on 2026-10-08
  (`minitpu_rtl_m1_2026-10-08.rst`): all 52 oracle launches bit-identical
  to MiniTPU's testbench digests inside the Allo region; with MiniTPU's own
  DDR bridge inside the shim (`minitpu_rtl_m2b_2026-10-08.rst`) the cycle
  counts equal the testbench's on all 52 as well. The baseline lives at
  `examples/minitpu/rtl/` (one design tree with the Allo-modelled core).

**D-22 (2026-10-08). TinyTPU is the toy instance for communication; MiniTPU
is the design driver.**
- TinyTPU-isa (`examples/tinytpu/`, frozen, D-4) exists to communicate the
  Allo co-design programming model and the compiler -- docs, demos, the
  tutorial -- and as the regression reference for the toolchain. It need not
  be a real implementation instance of the MiniTPU family.
- MiniTPU is the practical design driver: the unit ladder (U1-U5), the
  template, the ISA seam with its compiler, and the physical-design numbers
  are MiniTPU's. A "TinyTPU as an instance" of the template is a probe of the
  template's mechanisms (an int8 engine, an accumulator option), informative
  when it reproduces TinyTPU-isa's gates and cycles and not required to.
  D-4's retirement of the frozen copy is therefore optional, not a milestone.
- Naming: "TinyTPU-isa" is the shipped design; "TinyTPU" bare is the family
  name of that toy; both stay. `mvoutrelu.patch` is a demonstration variant
  of the shipped design; the parity baselines are its measurement
  configurations.

**D-23 (2026-10-08). The sequencer's command to the VPU is three
valid-qualified slot commands plus four declared resources.**
- MiniTPU's `vpu_ctrl_t` (an 85-bit combinational struct, valid in the issue
  cycle only, nine valid bits, payload ignored outside each op's valid
  cycle; U4 Phase 0) is declared in Allo as three slot commands -- V
  (vector), X (memory), M (matrix), the RTL's own letters -- each a `Stream`, and four shared
  resources declared as D-12 ports: VREG read ports A/B, port C (a store
  loses to the matrix stream), the write port under the calendar, the VMEM
  compute port. The collisions the RTL resolves by convention become
  declared ownership the composition checks.
- An instance may gate the payload (zero when not valid): a recorded
  deviation from the RTL, not a semantic change.
- Owner's condition: streams must not cost a deadlock or an issue-rate loss.
  With blocking streams every command channel is sized and single-owned
  (`stream_ports.rst`'s obligation), and the pass check is Phase 0's
  sequencer programs issuing cycle for cycle as the RTL does (D-24) in csim
  and on Catapult RTL; a stall the RTL does not have is a finding, and a
  non-blocking form is the fallback.

**D-24 (2026-10-08). U4 timing: cycle-locked first, self-timed compared.**
- The Allo sequencer is first transcribed cycle for cycle (fetch -> issue ->
  VPU command): the write-back claim `W` is derived from the bound units'
  declared latencies plus the write-back stages (D-20) and checked against
  Phase 0's measurement (`W = L + VPU_WB_STAGES` for all seven classes);
  `v1` programs run unchanged. This is for speed: it reuses MiniTPU's
  assembler, images and testbenches as they are.
- A self-timed composition (dataflow issue, `W` re-derived from the
  manifests, the timing published as a `versions.list.<name>` delta) is
  built beside it on the same programs, and the record compares the two --
  cycles, area, what each exposes -- before the template keeps one.
- No interlock at `b3ba0a4d` (D-16); an interlocked sequencer is a D-19
  option aligned with v2 when the pin moves. The host path (`iram_loader`,
  `command_unit`, the bridge) stays with M3: the baseline wraps the core seam
  (D-21).

## Milestones

Each milestone passes on **one acceptance check** and names the tools it uses
as-is, fixes, integrates, and upgrades. Status is recorded here.

**Status:** M0 done (2026-10-02, on `main`). Next: U1.

| | milestone, and its pass check | uses as-is | fixes | integrates | upgrades |
| --- | --- | --- | --- | --- | --- |
| **M0** | **Settled.** Layout as in D-3, stale docs fixed. *Check:* a clean checkout passes `reproduce.sh --no-cosim`, the SystemC emit tests and `pytest tests/act` | Allo core, TinyTPU gates | stale docs | choonsik1's unmerged commits (simulator math lowering, RISC-V-as-IP, EVA) | the layout of D-3; `AGENTS.md` |
| **U1** | **Arithmetic leaves:** `vpu_bf16_add`/`_pipe`, `vpu_bf16_mul`, `mxu_bf16_mul_acc24`, `mxu_acc24_add_pipe`, `vpu_alu`. *Check:* each bit-exact and at its declared latency against the RTL unit | Allo simulator, SystemC csim stand-in, Verilator 5.051, MiniTPU at `b3ba0a4d` | — | — | **the differential unit harness** (new, reused by every unit); bf16/acc24 numerics; declared pipeline latency on a unit |
| **U2** | **Storage with ports:** `vpu_regfile` (3 async reads, 1 sync write), `vpu_word_array` (2 read/write ports), `vpu_fifo`, the output FIFO. *Check:* as U1, plus the port conflicts MiniTPU's assembler refuses | as U1 | SystemC gives two clients of one memory a replica each | — | `compose` declares memory ports; addressed memory shared by several state machines |
| **U3** | **Datapath composites:** `xlu_reduction_tree`, `sfu`, `xlu_transpose`, `mxu_pe` → `mxu_systolic_array` → `mxu`. *Check:* as U1; the MXU against its push/commit/pop contract | as U1 | — | — | `compose` optional modules and swappable engines (MAC plug-in, matrix engine) |
| **U4** | **Control:** sequencer pieces (loop control, scalar address generation, address resolve, fetch/decode), DMA address generation, DMA. *Check:* against `tb_bundle_*` and `docs/isa_latency.json` | as U1 | — | — | the `vpu_ctrl_t` contract as a declared interface; the interlock decision |
| **U5** | **Compute core** (VPU + MXU + sequencer) on MiniTPU's own assembled programs. *Check:* `RTL-MATCH n/n`, Allo simulator and SystemC csim against the RTL | as U1; MiniTPU's `asm.py` | — | — | the simulator scaled to about 280 kernels |
| *track* | **Catapult, per unit:** each landed unit through SystemC → Catapult on zhang-21. *Check:* RTL cycle-equal to MiniTPU's unit in Verilator, with area and timing reported | Catapult 2024.2, Xcelium (zhang-21), Verilator | SystemC refuses or translates `s.dependence`, partitions and pipelining instead of dropping them; memory-port ready pins; `Wire` in RTL | `Wire` combinational mode (`c7402f9f`) | bf16 through Catapult synthesis |
| *track* | **TinyTPU as an instance** (after U3). *Check:* `stress_isa` on the instance; cycles against the frozen reference | Vitis cosim; the stress, mutate and `gen_isa` gates | — | — | ISA spec and `gen_isa` generalised to instances; an int8 MAC plug-in and an accumulator module |
| M1 | **The compiler onto MiniTPU** (after U5). *Check:* `make mlp TARGET=minitpu` bit-exact against PyTorch | ACT core, torch tracing | ACT's TinyTPU-shaped assumptions | MiniTPU's assembler as the emission back end | a VLIW machine model; bf16 workloads |
| M2 | **The open HLS stack.** *Check:* the Catapult track's check, through RTLGen + AMC | Verilator | — | Kai's RTLGen (reconciling `allov2` with this core); AMC | lowering composed regions to AMC |
| M3 | **The full MiniTPU stack.** *Check:* MiniTPU's own board checks on an Allo bitstream, and/or a physical-design report | Vivado, the ASIC flat flow (zhang-21) | the ASIC-manifest path, never run end to end | DMA and host path; board access; the open-source ASIC flow (needs Docker access) | the full VLIW core |


## Status

The table says where each flow runs; the second column is this host (ace-01).

| flow | ace-01 | elsewhere | maturity |
| --- | --- | --- | --- |
| Allo frontend, `compose`, Actions | yes | | one design family; no optional modules or swappable engines yet |
| Allo dataflow simulator | yes | | functional only; one OS thread per kernel; no math dialect yet |
| Vitis HLS csynth + cosim | yes | | every published cycle count; Vitis csim is not used (it hangs on misordered processes) |
| SystemC emit | yes | | TinyTPU and EVA emit; schedule directives are dropped |
| SystemC csim, functional stand-in (`scripts/systemc-csim-setup.sh`, pinned to Catapult 2024.2's library versions) | yes | | TinyTPU and EVA results identical to Catapult's own csim libraries (zhang-21, 2026-10-02); no synthesis, no cycles |
| Catapult csim / csyn / cosim / PPA (the SystemC flow's real target) | no | zhang-21 | EVA verified in RTL cosim (choonsik1). **TinyTPU's SystemC synthesizes through `go extract`**, 0 errors, 332 s (2026-10-02): area score 320,032 (75% registers), slack -0.072 ns at 2.0 ns, and a 10.7M-cycle reset from array-clearing loops. Not yet simulated as RTL. `dev/records/catapult_handoff/zhang21_compare_2026-10-02/` |
| Vivado (incl. Zynq UltraScale+) | yes | | not yet used |
| Verilator 5.051 | yes | | not yet used |
| Design Compiler + mflowgen (FreePDK45) | no | zhang-21 | TinyTPU and Gemmini area, measured |
| ASIC manifest → physical design | no | zhang-21 | written, never run end to end |
| RTLGen, AMC | no | | not integrated |
| ACT | yes | | TinyTPU target only |
| Agent harness | yes ($0 self-test) | GCP for paid runs | five paid runs (smoke + 1-4) |


## Running it

See `CLAUDE.md` for the environment. `LLVM_BUILD_DIR` must be exported by
hand. The gates:

```bash
examples/tinytpu/reproduce.sh --no-cosim         # functional gates, ~1 min, no Vitis
python examples/tinytpu/stress_isa.py            # the correctness gate, ~10 s
python examples/tinytpu/act_compile.py --gate    # every ACT mapping verified
pytest tests/act/ tests/dataflow/test_systemc_backend.py
```


## Upstream Allo

Allo is a Python-embedded, MLIR-based language and compiler for composable
accelerator design: see the [upstream repository](https://github.com/cornell-zhang/allo)
and its [documentation](https://cornell-zhang.github.io/allo). It targets AMD and
Intel FPGAs and AMD Ryzen NPUs. If you use Allo, please cite:

> Hongzheng Chen, Niansong Zhang, Shaojie Xiang, Zhichen Zeng, Mengjia Dai, and Zhiru Zhang, "**Allo: A Programming Model for Composable Accelerator Design**", Proc. ACM Program. Lang. 8, PLDI, Article 171 (June 2024).

Component papers: [Dato](https://arxiv.org/abs/2509.06794) (dataflow
programming model), [ARIES](https://cornell-zhang.github.io/allo/backends/aie.html)
(AIE backend), [formal verification of HLS transformations](https://github.com/cornell-zhang/allo/blob/main/allo/verify.py),
and [FPGA-based LLM inference](https://github.com/cornell-zhang/allo/tree/main/examples).
If you are using a coding agent, import [AGENTS.md](AGENTS.md).
