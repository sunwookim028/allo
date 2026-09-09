# Which backend can carry a Gemmini-competitive instruction-programmable TPU?

Measured on this machine, not reasoned from documentation.

## The one axis the three backends differ on

An accelerator with an ISA needs the decoded instruction to reach every PE, and
a grid of PEs needs operands from somewhere. Both are **fan-out of one array to
many processes**, and that is exactly where the backends part company.

| | chia RTLGen | Vitis HLS dataflow | SystemC/Catapult (choonsik1) |
|---|---|---|---|
| array read by N processes | allowed | **rejected** | memory port (req/rsp), arbitrated |
| array written by N processes | allowed | **rejected** | memory port |
| unit occupancy | a `func.call`: fills and drains **per instruction** | persistent process | persistent `SC_THREAD`, `while(1)` under synthesis |
| non-blocking ops | `stall_prob` cosim | `read_nb`/`write_nb` (upstream has `try_get`/`try_put`) | `try_get`/`try_put`, `empty()`/`full()` |
| link primitives | `Stream` | `hls::stream` | `Wire` / `Channel` (valid_ready or valid_only) / `Stream` |
| cycle counts here | yes (cocotb+Verilator) | yes (`cosim`) | needs Catapult -- **not installed** |

## What Vitis actually said

`repro/vitis_csyn_errors.log`, from `vitis_hls -f run.tcl` on the 4x4x4 grid:

```
ERROR: [HLS 200-779] Non-shared array 'v730' failed dataflow checking:
                     it can only have a single reader and a single writer.
ERROR: [HLS 200-979] Argument 'v731' failed dataflow checking:
                     it can only be written in one process function.
```

`v728/v729/v730` are `A`, `B`, `imem` -- read by all 36 instances -- and `v731`
is `C`, written by 16 of them. So the *same* fan-out that hangs the dataflow
simulator (`repro/README.md`) is rejected outright by Vitis, with a diagnostic
instead of a hang. Two independent tools agreeing on which construct is the
problem is worth more than either one alone.

## Is Vitis a dead end? No -- and its process model is what we want

Read the emitted C++ (`kernel.cpp`, 1410 lines) before concluding anything: it is
`#pragma HLS dataflow` over **36 persistent concurrent `pe_i_j` functions** with
152 `hls::stream` channels. Persistent processes are precisely the property the
chia backend lacks, and lacking it is the largest single cost measured there:

- `FINDINGS_v2.md` G.4 -- K back-to-back `mm`s cost `43.0*K + 4` cycles, where 43
  is one instruction's whole latency. Consecutive instructions overlap by
  **exactly zero**, because a unit is a `func.call` and "'func.call' cannot be
  predicated; the enclosing loop cannot pipeline across it".
- That is **712 of the executor's 1448 cycles at 16x16x16, 37% of the whole
  run**, spent starting an empty pipeline and draining it, once per instruction.

Vitis dataflow processes do not pay that. So the trade is: give up shared
random-access memory, gain persistent units.

**Non-blocking FIFOs are not the fix for the fan-out error.** `read_nb`/
`write_nb` change *blocking behaviour*; `HLS 200-779` is a structural check on
how many processes touch an array. They are, however, the fix for the *other*
failure -- the command-broadcast deadlock in `repro/README.md`, where a blocking
`put` on the command chain closes a cycle through the operand streams.

## The restructure Vitis forces is the architecturally correct one

A real accelerator does not have 36 PEs randomly addressing DRAM; it has
feeders. Giving each array exactly one owning process is both Vitis-legal and
closer to Gemmini:

    sequencer  owns imem  -> per-row broadcast process -> each PE's cmd FIFO
    loader     owns A, B  -> the array's west and north edge FIFOs
    drainer    owns C     <- the array's south and east edge FIFOs
    PE grid    owns only its own accumulator and its neighbour links

The per-row broadcast processes are what break the deadlock: they take no part
in the operand flow, so they cannot be in a cycle with it. This is the same
decoupled access-execute shape already measured on chia -- worth 1.35x there --
with the difference that the scratchpad becomes edge FIFOs rather than a shared
banked memory.

## Could that reach Gemmini?

Arithmetic on measured coefficients, **a projection, not a measurement**. At
16x16x16 the chia executor is 1448 cycles, of which 712 is per-instruction
fill/drain/dispatch and 736 is work. Persistent processes remove the 712. The
stage isolation (G.2) says the bound is then `max(stage)`, not `sum`:

| | cycles | vs Gemmini's 1141 |
|---|---|---|
| chia v2 as measured | 1734 | 1.49x |
| ... persistent units, so max(stage) with dispatch removed | ~800 | **0.70x** |

Two conditions have to hold and neither is verified: the array must still reach
II=1 under Vitis (chia needed a split output buffer and a de-skew register file
to get there, and Vitis schedules differently), and the edge-FIFO feed must
sustain the array without a shared scratchpad. Both are one experiment each.

## Catapult / SystemC

The in-tree `allo/backend/catapult.py` is not a candidate: 408 lines of scalar
C++ codegen with no stream or dataflow handling, and Catapult is not installed
here (`/opt` has xilinx and intel only).

`choonsik1/allo:SystemC-emitter` is a different matter -- 3738 lines of
`EmitSystemC.cpp` plus a 452-line design document, reached as
`df.build(..., target="systemc", mode="csim"|"csyn"|"cosim")`. Three things in
it bear directly on our blockers:

1. **Fan-out is handled, not rejected.** "Random access (anything else, or
   fan-in/out to >1 kernel) -> a memory port: an `AlloMem`/`AlloMemW` behind a
   req/rsp channel pair." That is a structural answer to the exact construct
   Vitis refuses and the simulator hangs on.
2. **Persistent, pipelined processes.** One `@df.kernel` becomes an `SC_MODULE`
   with a clocked `SC_THREAD run()`, and the outermost loop is emitted as
   `while(1)` under synthesis "so Catapult pipelines it (a step every clock)" --
   the property Vitis has and chia lacks, stated as a design goal.
3. **Richer links.** `Wire` (no handshake), `Channel` (valid/ready or
   valid-only), `Stream` (a real FIFO), each with `try_get`/`try_put` and
   `empty()`/`full()`.

It also **independently corroborates our accumulator finding**. Its pitfall list
says: "A read-modify-write output array (`C[i] += ...`) forces single-shot
execution. The kernel body runs once; you can't free-run/pipeline it." That is
the same constraint that forced the split output buffer and the separate
accumulate walk on chia (`FINDINGS_v2.md` F.2) -- so it is a property of
HLS-style accumulation across three backends, not a chia quirk, and it
strengthens the case that what is missing is an *accumulating memory primitive*.

The blocker is practical: `csyn`/`cosim` there need Catapult, which is not
installed. `csim` needs only g++ and a SystemC library, so functional
verification is reachable; cycle counts are not, and their own doc warns
"csim != cosim ... always run cosim before trusting synthesis."

## Recommendation

1. **Vitis, with the feeder/drain restructure.** It is the only path on `main`
   that gives cycle counts on this machine, its process model fixes the largest
   measured deficit, and the restructure it forces is one we want anyway. Two
   experiments settle it: does the array hold II=1 under Vitis, and does the
   edge-FIFO feed keep it fed.
2. Keep chia v2 as the measured baseline -- it is the only design with
   end-to-end RTL numbers today (1734 cycles, 1.49x Gemmini, 16 MACs at II=1).
3. Track the SystemC branch rather than adopt it now: it is the only backend
   that answers the fan-out problem structurally, but it cannot produce a cycle
   count here until Catapult is available. Worth asking what they measure with.
