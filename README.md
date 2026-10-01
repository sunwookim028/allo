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
  (`~/core/minitpu`, another engineer's design, read-only to us): its own
  programs give bit-exact results, and its resources are reported beside
  MiniTPU's. The Allo version may add interlocks; that is a recorded deviation.

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
- **SystemC → Catapult is the near-term *target* backend.** Emission and a
  behavioural csim run on this host; Catapult itself runs on zhang-21.
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
  built there in M1. TinyTPU's `examples/tinytpu/ip/` stays in place, frozen
  (D-4), and is retired at M2, when the TinyTPU instance imports from the
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
- It is retired only when the template instance reproduces it (milestone M2).
- `examples/minitpu/` is rewritten as the template. Its `reference.py`
  (arithmetic) and `program.py` (schedule rules) carry over.

**D-5 (2026-10-01). The agent method is open.** CHIA is the current instance,
not a commitment. Harness work goes into the method-agnostic core first.

**D-6 (2026-10-01). Settle before building.**
- No design work until M0 is on `main`.
- M1 starts only after the expressiveness probes (P) have been reviewed.


## Milestones

Each milestone passes on **one acceptance check** and names the tools it uses
as-is, fixes, integrates, and upgrades. Status is recorded here. The milestones
after M0 are reviewed again once M0 is done.

| | milestone, and its pass check | uses as-is | fixes | integrates | upgrades |
| --- | --- | --- | --- | --- | --- |
| **M0** | **Settled.** Layout as in D-3, stale docs fixed. *Check:* a clean checkout passes `reproduce.sh --no-cosim`, the SystemC emit tests and `pytest tests/act` | Allo core, TinyTPU gates | stale docs | choonsik1's unmerged commits (simulator math lowering, RISC-V-as-IP, EVA) | the layout of D-3; `AGENTS.md` |
| **P** | **Expressiveness probes.** Small designs with MiniTPU's hard shapes: several state machines over **addressed memories** (VREG, VMEM), shared and contended ports, a two-ported memory, fixed-latency delay lines, a double-buffered commit, a VLIW bundle issued to several slots at once, and a stall interlock. *Check:* a matrix showing each shape expressed, refused, or wrong in the Allo simulator and in SystemC csim | Allo simulator, SystemC emitter | whatever the probes expose | open-source SystemC/MatchLib as a documented toolchain step | — |
| M1 | **MiniTPU compute core as a composed template**, bit-exact against MiniTPU's RTL on programs from MiniTPU's own assembler. *Check:* `RTL-MATCH n/n` on the Allo simulator and on SystemC csim | Allo simulator, SystemC emitter, Verilator 5.051, MiniTPU's `asm.py` and testbench (read-only) | csim's dependence on a `catapult` binary; csim's output arrays starting at zero; the composition limits found by P | a reference harness that runs MiniTPU program images on its Verilator RTL | `compose` gains optional modules and swappable engines; interlocks; bf16/acc24 through csim; the simulator scaled to about 280 kernels |
| M2 | **TinyTPU as an instance of the template.** *Check:* `stress_isa` passes on the instance, and its cosim cycles are reported against the frozen reference | Vitis cosim; the stress, mutate and `gen_isa` gates | — | — | ISA spec and `gen_isa` generalised to instances; Actions on the template |
| M3 | **Template → RTL, matched and measured.** *Check:* MiniTPU core RTL bit-exact in Verilator, plus a QoR table against MiniTPU's ZCU104 build | Catapult (zhang-21), Verilator, Vivado 2023.2 (this host) | SystemC refuses or translates `s.dependence`, partitions and pipelining instead of dropping them; memory-port ready pins; `Wire` in RTL | `Wire` combinational mode (`c7402f9f`); a runbook for zhang-21 | bf16 through Catapult synthesis |
| M4 | **The compiler onto MiniTPU.** *Check:* `make mlp TARGET=minitpu` bit-exact against PyTorch | ACT core, torch tracing | ACT's TinyTPU-shaped assumptions | MiniTPU's assembler as the emission back end | a VLIW machine model; bf16 workloads |
| M5 | **The open HLS stack.** *Check:* M3's check, through RTLGen + AMC | Verilator | — | Kai's RTLGen (reconciling `allov2` with this core); AMC | lowering composed regions to AMC |
| M6 | **The full MiniTPU stack.** *Check:* MiniTPU's own board checks on an Allo bitstream, and/or a physical-design report | Vivado, the ASIC flat flow (zhang-21) | the ASIC-manifest path, never run end to end | DMA and host path; board access; the open-source ASIC flow (needs Docker access) | the full VLIW core |


## Status

The table says where each flow runs; the second column is this host (ace-01).

| flow | ace-01 | elsewhere | maturity |
| --- | --- | --- | --- |
| Allo frontend, `compose`, Actions | yes | | one design family; no optional modules or swappable engines yet |
| Allo dataflow simulator | yes | | functional only; one OS thread per kernel; no math dialect yet |
| Vitis HLS csynth + cosim | yes | | every published cycle count; Vitis csim is not used (it hangs on misordered processes) |
| SystemC emit + behavioural csim | yes, with open-source libraries | | TinyTPU emits and runs; schedule directives are dropped; no cycles |
| Catapult csyn / cosim / PPA | no | zhang-21 | never run on a TPU design |
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
