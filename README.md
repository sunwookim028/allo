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
