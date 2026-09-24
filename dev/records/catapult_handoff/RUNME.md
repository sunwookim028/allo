# Catapult synthesis handoff — `pc_int32_systemc`

> Other handoffs in this directory: `ppa_mac16/` (power, `mode="ppa"`, not yet
> run), and the completed runs `zhang21_run_2026-09-24/` (this one) and
> `zhang21_power_2026-09-24/` (power by hand).

Emitted on ace-01 (no Catapult, no licence) on 2026-09-24 from `main` @ `a622c9ea`,
by `df.build(top, target="systemc", mode="csyn", project=...)`. Everything Catapult
consumes is committed here; **no `allo` import is needed on the licence host.**

## The design

`@df.region() top(A: int32[8], B: int32[8])` — a `producer` kernel pushes `A[i]` into
`Stream[int32, 4]`, a `consumer` kernel pops and writes `B[i] = fifo.get() + 1`.
So `B == A + 1`. `A = 0..7`, therefore `B = 1..8`.

## Configuration (all of it)

| | |
| --- | --- |
| top module | `top` (`SC_MODULE(top)` in `kernel.cpp`; `directive set -DESIGN_HIERARCHY top`) |
| library | `nangate-45nm_beh` |
| clock | `CLOCK_PERIOD 2.0` ns (500 MHz) |
| directives | `-IO_MODE super`, `-SPECULATE true` (emitted because `platform == "systemc"`) |
| tcl | **use the one the emitter wrote**, `pc_int32_systemc/run.tcl`, unmodified |
| Catapult feature | `CatapultUltra` |

`kernel.cpp` is self-contained: the three SC_MODULEs (`producer_0`, `consumer_0`,
`top`), plus a `tb` module and `sc_main` used only by csim. It contains no absolute
paths; `run.tcl` resolves its source via `[file dir [info script]]`, so the project
may live anywhere.

## Commands, in order

```bash
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home   # the /Mgc_home component matters
export PATH=$MGC_HOME/bin:$PATH
export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu   # nothing pre-sets this
unset LD_PRELOAD

mkdir -p /scratch/$USER/catapult_handoff
cp -r <checkout>/dev/records/catapult_handoff/pc_int32_systemc /scratch/$USER/catapult_handoff/
cd /scratch/$USER/catapult_handoff/pc_int32_systemc

# (optional, ~1 min, $0, no licence needed for the compile itself) behavioural csim
./csim.sh && diff output0.data golden_output0.data && echo "CSIM PASS"

# the run
catapult -shell -f run.tcl 2>&1 | tee catapult.log
```

## Success criterion (written before the run, 2026-09-24)

**PASS** iff all four hold:

1. `catapult.log` contains `LIC-14` (licence checked out) and the process exits 0.
2. `catapult.log` contains no line matching `^Error` — in particular no `SCHD-67`,
   `SCHD-30`, `HIER-47`, `ASSERT-1` or `CIN-`.
3. `Catapult/top.v1/rtl.v` exists and is non-empty, and contains `module top`.
4. `Catapult/top.v1/cycle.rpt` exists and reports a finite Latency and Throughput
   for `top` (not `-` / `unbounded`).

**FAIL** otherwise. Please report, from `Catapult/top.v1/`: the `cycle.rpt` Latency
and Throughput for `top`, and the `rtl.rpt` total area (Catapult score units), plus
the first error line from `catapult.log` if it failed.

**Negative control** (offered by the licence host; confirms the criterion can go red):
re-run with the scheduling directives removed, which the docs predict is the failing
configuration for a multi-handshake body —

```bash
sed '/-IO_MODE super/d; /-SPECULATE true/d' run.tcl > run_nodirectives.tcl
catapult -shell -f run_nodirectives.tcl 2>&1 | tee catapult_nodirectives.log
```

This one is expected to be *allowed* to pass (the design has one handshake per body,
so it may well schedule anyway); it is informative either way and is not part of the
criterion above.

## What was and was not checked on ace-01

- `kernel.cpp` **compiles and simulates correctly** on ace-01 against substitute
  open-source libraries (hlslibs `ac_types` + `matchlib_connections` + `ac_simutils`,
  and Vivado 2023.2's SystemC 2.3.1), producing exactly `1..8`. So a csim failure on
  zhang-21 points at the Siemens headers/library, not at the design.
- Nothing in the Catapult flow itself was run. `go analyze` / `go compile` /
  `go assembly` / `go extract` are entirely unexercised here.
