# `mode="ppa"` handoff — `mac16` (zhang-21)

Emitted on ace-01 (no Catapult, no Xcelium, no licence) on 2026-09-24 from the
`mode="ppa"` fix, by

```python
df = allo.customize(mac16)          # int8[16] . int8[16] -> int32
df.build(target="catapult", mode="ppa", project=...,
         configs={"testbench": "mac16_tb.cpp", "ncsim_root": "/opt/cadence/XCELIUM2403",
                  "clock_period": 5.0, "library": "nangate-45nm_beh"})
```

Everything Catapult consumes is committed here; **no `allo` import is needed on the
licence host.** This is the first end-to-end exercise of the repaired `mode="ppa"`:
before the fix it emitted the `csyn` tcl (stopping at `go extract`) and produced no
power at all. Nothing in this flow has been run anywhere — ace-01 has no Catapult.

## The design and its testbench

| | |
| --- | --- |
| design | `kernel.cpp`, Allo-emitted: `void mac16(int8_t a[16], int8_t b[16], int32_t *out)`, marked `#pragma hls_design top` |
| testbench | `mac16_tb.cpp`, hand-written: `CCS_MAIN` + `CCS_DESIGN(mac16)`, **200 transactions**, pseudo-random int8 operands over the full [-128, 127] range, self-checking |
| library / clock | `nangate-45nm_beh`, `CLOCK_PERIOD 5.0` ns |
| simulator | Xcelium (`/SCVerify/USE_NCSIM true`, `/NCSim/NC_ROOT /opt/cadence/XCELIUM2403`) |
| activity | `/LowPower/SWITCHING_ACTIVITY_TYPE saif` (FSDB would need Verdi) |
| tcl | **use the emitted `run.tcl` unmodified** |

The testbench is the workload: the power number is only as representative as it is.
Its stimulus is deliberately the same shape as the hand run in
`../zhang21_power_2026-09-24/` (248.47 µW), so the two are comparable.

The testbench's reference model was checked on ace-01 against a plain-C++ stand-in for
the kernel (`errors=0`); the Catapult headers themselves are not available there, so
nothing else about it was compiled.

## Commands, in order

```bash
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home     # the /Mgc_home component matters
export PATH=$MGC_HOME/bin:$PATH
export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu   # Catapult + PowerPro
export CDS_LIC_FILE=<the Xcelium licence this host uses>         # needed by the switching sim
unset LD_PRELOAD

mkdir -p /scratch/$USER/ppa_mac16
cp -r <checkout>/dev/records/catapult_handoff/ppa_mac16/* /scratch/$USER/ppa_mac16/
cd /scratch/$USER/ppa_mac16

# NOTE the log goes OUTSIDE the project: Catapult owns ./catapult.log and overwrites it.
catapult -shell -f run.tcl > $HOME/ppa_mac16_stdout.log 2>&1

python3 check_ppa.py            # prints PASS or FAIL + what to report; exit code 0/1
```

Expect ~2 min (the hand run of the equivalent design took 81 s).

## Success criterion (written before the run, 2026-09-24)

`check_ppa.py` **is** the criterion; it reads only files the tools emit. **PASS** iff
all four hold:

1. `Catapult/mac16.v1/power.rpt` exists, and in its `Power Report (uW)` table the
   design-level `Dynamic` row has a Total **greater than zero**. (Zero means the
   switching simulation ran nothing — the exact silent failure this work is about.)
2. The same report's `Switching Activity (Percent Asserted)` row shows **100.00 %**
   flop outputs, i.e. the activity came from the SAIF, not from default toggle rates.
3. `$HOME/ppa_mac16_stdout.log` contains `MAC16 TB errors=0` (the testbench checked
   itself) and `Simulation PASSED` (SCVerify's own comparison).
4. `Catapult/mac16.v1/rtl.v` exists and contains `module mac16`.

**Informative, NOT pass/fail** — please report these whatever the outcome:
the total/dynamic/static µW and the per-instance rows; `cycle.rpt` latency and
throughput (a negative latency is legitimate for a free-running thread and is not a
failure); `rtl.rpt` total area; and, if it failed, the first line of the
`go switching` step in the log.

There is deliberately **no** criterion on the absolute µW matching the hand run: the
Allo-emitted top has a different port interface (arrays plus an output pointer) from
the hand-written `mac.cpp`, so the two are the same workload but not the same netlist.
A figure in the same order of magnitude (tens to hundreds of µW) is expected; a figure
that is exactly zero is a failure by criterion 1.

## What could not be checked here, and where it would break

- **Nothing in the Catapult flow was run.** `go analyze/compile/assembly/extract/
  switching` and `flow run /PowerAnalysis/report_pre_pwropt_Verilog` are unexercised.
- The riskiest step is `go switching`: SCVerify must wrap a DUT marked only with
  `#pragma hls_design top` (the hand run used `CCS_BLOCK()` plus
  `/SCVerify/USE_CCS_BLOCK true`, which Allo does not emit). If SCVerify complains
  that it cannot find the design, re-run with
  `flow package option set /SCVerify/USE_CCS_BLOCK true` added after the
  `USE_NCSIM` line — from Allo that is `configs={"use_ccs_block": True}`.
- If the run fails at the simulator rather than the design, check `CDS_LIC_FILE` and
  that `/opt/cadence/XCELIUM2403` is still the right root; it is emitted from
  `configs["ncsim_root"]`, not compiled into the backend.

## Optional second pass (exercises the parser too)

With the checkout and the `allo` conda env on this host, the same run through Allo
also exercises the report parser and the fail-loud checks, and prints the table and
the activity caveat:

```python
import allo
from allo.ir.types import int8, int32
def mac16(a: int8[16], b: int8[16]) -> int32:
    acc: int32 = 0
    for i in range(16):
        acc += a[i] * b[i]
    return acc
s = allo.customize(mac16)
mod = s.build(target="catapult", mode="ppa", project="ppa_prj",
              configs={"testbench": "<abs path>/mac16_tb.cpp",
                       "ncsim_root": "/opt/cadence/XCELIUM2403",
                       "clock_period": 5.0, "library": "nangate-45nm_beh"})
print(mod())
```

It raises rather than printing zeros if the power report is missing, has zero dynamic
power, or shows 0 % annotation.
