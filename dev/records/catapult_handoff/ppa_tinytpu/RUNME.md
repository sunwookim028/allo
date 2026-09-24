# `mode="ppa"` handoff — TinyTPU-isa (zhang-21)

The first power measurement of the design the fork actually ships. Area and
latency have been measured for a long time; power has not, because
`mode="ppa"` needs "a design SCVerify can drive", i.e. a C++ testbench with
`CCS_MAIN` and `CCS_DESIGN(<top>)`, and TinyTPU had none. This directory is
that testbench plus everything Catapult consumes.

Emitted on ace-01 (no Catapult, no Xcelium, no licence) on 2026-09-24 by

```bash
TPU_MAXDIM=16 python examples/tinytpu/ppa_catapult.py     # writes this directory
```

**No `allo` import is needed on the licence host**; `kernel.cpp`, `run.tcl` and
`tinytpu_tb.cpp` are committed. Nothing in the Catapult flow has been run
anywhere.

## The design, the configuration and the testbench

| | |
| --- | --- |
| design | `kernel.cpp`, Allo-emitted from the `@df.region()` `tinytpu_isa`: `void tinytpu_isa(uint64_t imem[56], int8_t A[256], int8_t B[256], int8_t C[256])`, `#pragma hls_design top`, `#pragma hls_design dataflow`, 85 `ac_channel`s, 8 units, 4x4 PE array |
| configuration | `TPU_T=4`, **`TPU_MAXDIM=16`**, `TPU_QD=16`, derived `SPAD_ROWS=NVR=64`, `NAR=128`, `IMEM_SIZE=56` — i.e. `reproduce.sh`'s pinned build, the one the published 175/265/421/482/674 cycle counts are for |
| testbench | `tinytpu_tb.cpp`, generated: `CCS_MAIN` + 4 `CCS_DESIGN(tinytpu_isa)` calls on one instance, self-checking |
| library / clock | `nangate-45nm_beh`, `CLOCK_PERIOD 5.0` ns — **the same as `../ppa_mac16/`**, so the two figures differ in the design and not in the technology |
| simulator | Xcelium (`/SCVerify/USE_NCSIM true`, `/NCSim/NC_ROOT /opt/cadence/XCELIUM2403`) |
| activity | `/LowPower/SWITCHING_ACTIVITY_TYPE saif` |
| tcl | **use the emitted `run.tcl` unmodified** |

`use_ccs_block` is **not** set, following `ppa_mac16`: Allo marks the top with
`#pragma hls_design top`, which SCVerify wrapped there without it. The
TinyTPU top has the same *shape* — a plain C++ function with array parameters —
so the same should hold; if SCVerify cannot find the design, see
"If it fails" below.

## What the power number will and will not represent

`go switching` simulates this testbench against the pre-power RTL and the
resulting SAIF is the whole basis of the figure, so **the testbench is the
workload**. It is four calls on one RTL instance:

| # | program | operands |
| --- | --- | --- |
| 0 | GEMM 16x16x16 | uniform full-range int8, seed 900 |
| 1 | GEMM 16x16x16 + ReLU | uniform full-range int8, seed 901 |
| 2 | GEMM 16x16x16 | uniform full-range int8, seed 902 |
| 3 | `isa_dsl.vector_program(8)` | uniform full-range int8, seed 950 |

* **16x16x16** is the largest published shape. TinyTPU's cycle count is a fixed
  startup/drain term plus work, so at the smaller shapes the activity is mostly
  program load and pipeline fill; 16x16x16 is the shape whose steady state is
  longest relative to that term.
* **Full-range int8, not the [-4, 4] that `bench_isa.py` and `cosim.py` use.**
  That distribution exists to match Gemmini's `allo_cmp.c` for a like-for-like
  *cycle* comparison, and cycles do not depend on operand values. Power does:
  [-4, 4] holds the top five bits of every operand constant, which would bias
  the number low. `ppa_catapult.py --dist small` regenerates the
  Gemmini-distribution variant if that comparison is ever wanted.
* **Three seeds**, so one unlucky draw does not become the number.
* **`vector_program(8)`**, because a GEMM never exercises `vld`, a second
  accumulator region, `vadd` across three regions, or an `mvout` to a nonzero
  DRAM row — and a unit that never switches contributes only leakage.

**It therefore measures** one shape and one operand distribution at T=4 /
MAXDIM=16, nangate45 behavioural, 5 ns, pre-layout, with activity SAIF-annotated
on flops and primary inputs and propagated to internal nets by PowerPro. **It
does not represent** the shipped `TPU_MAXDIM=64` build, a mix of shapes, a duty
cycle with idle time between kernels, or anything post-place-and-route. Quote it
with that attached; `mode="ppa"` prints the same caveat with every result.

## Self-checking, and how strong the check is

Each call prefills the whole of `C` with random bytes and compares all 256 cells
afterwards, splitting the count:

* `wrong` / `errors=` — cells the program is expected to write. The mask is
  computed by running `isa_ref` twice over two independent prefills and taking
  the union, so a written cell that coincidentally keeps its old value is still
  counted as written. **`errors=0` is the criterion.**
* `clobbered=` — cells it must leave alone. **Reported, not gating**: whole-array
  preservation is a property of SCVerify's memory wrapper on a four-array-port
  design, which nothing has exercised before, whereas `errors=` is the
  arithmetic. Note that at MAXDIM=16 a 16x16x16 GEMM covers *all* of `C`, so
  only case 3 has preserved cells at all (96 written of 256).

864 result cells are checked over the four calls; **707 of them (81.8 %) sit at
a clip boundary**, which is unavoidable — sixteen full-range int8 products
saturate an int8 result unless the operands are within about ±3, which is
exactly why the *cycle* testbench uses [-4, 4]. A saturated cell still carries
the sign of the whole 16-term dot product, so the check is far from vacuous, but
it is more a sign check than a value check, and that is why `stress_isa.py` and
`TPU_TB=stress` cosim remain the correctness gates. This one is the cheap guard
that the power number did not come from a simulation computing nonsense.

## Evidence produced before the handoff, on ace-01

1. **The expectations are the design's, not just numpy's.** `ppa_catapult.py`
   runs all four cases on Allo's simulator and requires the golds back exactly
   before it writes anything (`simulator OK:` x4). Each GEMM gold is also
   cross-checked between `stress_isa.gemm_gold` and `isa_ref.run`.
2. **The testbench compiles**, with `g++ -std=c++11` — the standard `run.tcl`
   sets — against two-line stand-ins for `ac_int.h` and `mc_scverify.h`.
3. **Its self-check can go red.** Run against a stand-in design replaying a
   reference dump: correct -> `errors=0 clobbered=0`, exit 0; one result cell
   flipped -> `errors=1`, exit 1; one preserved cell flipped -> `errors=0
   clobbered=1`, exit 0 (reported, not a failure, as documented above).
   All three are asserted by `ppa_catapult.py` on every regeneration.
4. **`check_ppa.py` can go red.** It was run against eight doctored copies of
   `../ppa_mac16/zhang21_run_2026-09-24/`'s outputs, renamed to this top: PASS
   on the baseline; FAIL on `errors=7`, on a missing summary line, on
   `Simulation FAILED`, on a zeroed design-level Dynamic row, on 0 % annotation,
   on a missing `power.rpt`, and on an `rtl.v` with the wrong top module; and
   still PASS on `clobbered=5`, which must not gate.

One thing found by doing this that is worth knowing: `cosim.py`'s
`static alignas(64) T x[N]` form — copied at first — **does not compile under
`-std=c++11`** (a standard attribute in the middle of the decl-specifiers; g++
rejects it). Vitis gets away with it only because it compiles its testbench with
`-std=gnu++0x`. The Catapult testbench drops `alignas` entirely; the
`align_value(64)` promise it was keeping is made by the *Vitis* `m_axi` path and
has no counterpart here.

## Commands, in order

```bash
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home     # the /Mgc_home component matters
export PATH=$MGC_HOME/bin:$PATH
export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu   # Catapult + PowerPro
export CDS_LIC_FILE=<the Xcelium licence this host uses>         # needed by the switching sim
unset LD_PRELOAD

mkdir -p /scratch/$USER/ppa_tinytpu
cp -r <checkout>/dev/records/catapult_handoff/ppa_tinytpu/* /scratch/$USER/ppa_tinytpu/
cd /scratch/$USER/ppa_tinytpu

# NOTE the log goes OUTSIDE the project: Catapult owns ./catapult.log and overwrites it.
# (If $HOME is slow NFS here, put it on local scratch and pass that path below.)
catapult -shell -f run.tcl > $HOME/ppa_tinytpu_stdout.log 2>&1

python3 check_ppa.py            # prints PASS or FAIL + what to report; exit code 0/1
# or: python3 check_ppa.py Catapult/tinytpu_isa.v1 /path/to/that/log
```

Expect **longer than `ppa_mac16`'s 56 s** — this design is a 4x4 PE array, eight
units and 85 channels against a 16-tap MAC — but not enormously: the switching
simulation is roughly 2,700 design cycles (four calls, ~674 each in the Vitis
cosim). If it has not finished in an hour, say so rather than waiting; a step
that runs for ever is itself the result.

## Success criterion (written before the run, 2026-09-24)

`check_ppa.py` **is** the criterion; it reads only files the tools emit and
imports nothing. **PASS** iff all four hold:

1. `Catapult/tinytpu_isa.v1/power.rpt` exists, and in its `Power Report (uW)`
   table the design-level `Dynamic` row has a Total **greater than zero**. (Zero
   means the switching simulation ran nothing.)
2. The same report's `Switching Activity (Percent Asserted)` row shows
   **100.00 %** flop outputs — the activity came from the SAIF, not from default
   toggle rates.
3. The stdout log contains `TINYTPU TB errors=0 clobbered=<n>` (matched on the
   number, not as a substring) **and** an `Info: scverify_top...: Simulation
   PASSED` line (anchored to that line, not a bare substring that Catapult's
   Info text could also contain).
4. `Catapult/tinytpu_isa.v1/rtl.v` exists and contains `module tinytpu_isa`.

**Informative, NOT pass/fail** — please report these whatever the outcome: the
total/dynamic/static µW; the per-instance rows (up to ten — the per-unit
breakdown is the point of the exercise); `clobbered=`; `cycle.rpt` latency and
throughput (**a negative latency is legitimate** for a free-running thread, and
this design is eight of them); `rtl.rpt` total area; the SAIF size; and, if it
failed, the first error line of the `go switching` step.

There is deliberately **no criterion on the absolute µW**. There is no prior for
TinyTPU — this run *is* the first data point — so any threshold would be
invented rather than derived. `ppa_mac16`'s 272.69 µW is a 16-tap MAC and is not
a scale for this. Only "greater than zero" is checked, because zero is the known
silent failure.

## What could not be checked here, and where it would break

* **Nothing in the Catapult flow was run.** `go analyze/compile/assembly/extract/
  switching` and `flow run /PowerAnalysis/report_pre_pwropt_Verilog` are
  unexercised on this design. It has never been through Catapult at all — its
  ASIC numbers to date come from Vitis RTL through commercial synthesis.
* **The riskiest step is `go switching`**, for the same reason as in
  `ppa_mac16` plus one more: SCVerify must wrap a top with **four array ports**
  and a `#pragma hls_design dataflow` body of eight concurrent processes, not a
  single MAC. If SCVerify complains that it cannot find the design, add
  `flow package option set /SCVerify/USE_CCS_BLOCK true` after the `USE_NCSIM`
  line (from Allo: `configs={"use_ccs_block": True}`).
* **`go compile` / `go assembly` may need scheduling directives.** Nothing here
  sets `-IO_MODE` or `-SPECULATE` — the backend emits those only for the
  `systemc` platform. If the schedule fails with SCHD-67/SCHD-30, that is the
  first thing to try, and it is a backend change, not a hand edit.
* **The C++ design body is never executed** in this flow, so the fact that a
  sequential C++ run of an `ac_channel` dataflow region with feedback would
  deadlock does not apply. `CCS_DESIGN` resolves to the RTL wrapper under
  SCVerify, and `Simulation PASSED` comes from the testbench's own
  `CCS_RETURN`. (This is why `mode="csim"` is *not* part of this handoff.)
* If it fails at the simulator rather than the design, check `CDS_LIC_FILE` and
  that `/opt/cadence/XCELIUM2403` is still the right root; it comes from
  `configs["ncsim_root"]`, not from the backend.

## The shipped `TPU_MAXDIM=64` build

The area numbers the fork publishes are for `TPU_T=4 TPU_MAXDIM=64`, so power
there is the more useful figure — but **run this one first**, because anything
that fails will fail the same way and faster.

Regenerating at 64 is one variable:

```bash
TPU_MAXDIM=64 python examples/tinytpu/ppa_catapult.py -o <dir>
```

It works today on ace-01 (26 s, all four simulator checks pass, testbench
compiles). What changes is **only the array extents**: the emitted `kernel.cpp`
has the same 85 channels, the same eight units and the same 4x4 array; `spad`
and `vr` go from 64 to 1024 rows and the four DRAM ports from 56/256 to
56/4096 words. So the *structure* Catapult schedules is identical and the
switching simulation runs a comparable number of cycles (868 vs 674 for
16x16x16 in the Vitis cosim, because MAXDIM is the DRAM row stride). The
testbench grows to 359 KB of initialisers.

Nothing observed here says 64 is intractable; equally, nothing here measures
it. The honest statement is that the 64 build is one command away and untried,
and that its risk is memory elaboration in Xcelium and PowerPro's per-instance
reporting over 16x larger RAMs, not the design.
