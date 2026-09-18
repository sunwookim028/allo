# Simulating the Catapult SystemC netlists under xsim

The apparatus behind `notes/ALLO_SHORTCOMINGS.md` #22. The conclusion is in that
note; this is how to re-run it, and it took reverse-engineering the netlists'
internal signal names (e.g. `tb.u_mul.mul_0_run_inst.v8_and_cse`) to build, so
it is kept rather than rebuilt.

## What it tests

Three emitted designs for the same computation -- an element-wise multiply
feeding an accumulator -- differing only in the edge between the two:

| design | edge |
| --- | --- |
| `pe_stream` | `Stream[int32, 2]`, a depth-2 FIFO |
| `pe_channel` | `Channel`, valid/ready handshake |
| `pe_wire` | `Wire`, a non-handshaked combinational edge |

Each is driven at 18 producer/consumer pacings (`STALL_IN` 1..6 x `STALL_OUT`
1..3) plus a baseline, and checked against golden `2 10 28 60 110 182 280 408`.

## The result (`results.txt`, 75 runs)

| design | PASS | FAIL |
| --- | --- | --- |
| `pe_stream` | 19 | 1 -- the injected fault, as intended |
| `pe_channel` | 19 | 1 -- the injected fault, as intended |
| **`pe_wire`** | **2** | **27** |

`pe_wire` fails **8/8 elements wrong at every one of the 18 pacings**, producing
`0 0 0 2 2 10 10 28`. The two passes are not pacings; they are the positive
control below.

## Why it fails, and the positive control that proves it

From the netlist: `acc_0` has no input handshake and advances on its
*consumer's* ready, while `mul_0` latches on its *producers'* valid. Nothing
couples the two counters, and `mul` is ~3x slower per element, so `acc` runs the
whole loop before `mul` produces anything.

`LOCKSTEP` + `ACC_RST_DELAY` holds `acc_0` in reset longer and steps it once per
product. On the **identical** wire RTL:

    ACC_RST_DELAY = 0  1  2  3  4  5  6
                    F  F  F  P  P  F  F

So the wiring and the arithmetic are correct and only the lockstep is missing --
and **the correct window is 2 cycles wide**, which is the number that matters
for anyone trying to schedule against a `Wire`.

## The faults, so the test is known to be able to fail

Four injections, all red: `BREAK_DATA` on each of the three designs, and
`BREAK_WIRE` on top of the *working* `ACC_RST_DELAY=3` configuration. A test
that only ever passes proves nothing; these are why the passes above count.

## Running it

Needs Vivado's xsim on PATH (`/opt/xilinx/Vivado/2023.2/settings64.sh`) and the
netlists, which are **not** in this repo -- they are Catapult output from the
`choonsik1/SystemC-emitter` fork. `REPRO.sh` documents where they came from.
`mgc_shim.v` supplies the Mentor primitives the netlists instantiate.

    ./run.sh          # the three designs across the pacing sweep
    ./run_mulacc.sh   # the lockstep / fault-injection sweep

Both print one `RESULT: PASS|FAIL <tag>` line per run; `results.txt` is that
output, sorted. The 76 xsim work directories are not kept -- they are ~29 MB and
regenerable, and the `RESULT:` lines are the evidence.
