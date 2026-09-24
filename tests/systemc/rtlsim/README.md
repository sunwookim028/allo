# SystemC vs RTL cross-check

The apparatus behind [limitation 22](../../../docs/source/developer/limitations.rst):
a `Wire` boundary that passes SystemC csim but is wrong in Catapult's own netlist,
shown on two independent RTL simulators.

Full write-up, including what each result means:
`docs/source/extensions/catapult_systemc.rst`
(https://sunwookim028.github.io/allo/extensions/catapult_systemc.html).

The design under test is [`tests/systemc/dot_product_four_links.py`](../dot_product_four_links.py)
— one running dot product expressed four ways, differing only at the `mul → acc`
boundary. The recorded results are in
[`dev/records/systemc/rtlsim/`](../../../dev/records/systemc/rtlsim/):
`results.txt` (the verdict matrix) and `ref_xsim/` (two runs of the *same* netlist,
one pass and one fail, kept for diffing).

## Contents

| File | What |
|---|---|
| `REPRO.sh` | the whole matrix: three boundaries, the pacing sweep, the lockstep control, the fault injections (~23 s) |
| `run_mulacc.sh` | one case on Vivado `xsim`; `run.sh` is the `tb_top.v` variant |
| `run_mulacc_xrun.sh` | the Xcelium counterpart, same arguments; `RUNNER=run_mulacc_xrun.sh ./REPRO.sh` |
| `tb_mulacc.v` | the isolated `mul → boundary → acc` testbench; drives both input streams and sinks the result, so every handshake is controlled |
| `tb_top.v` | the four-kernel-PE testbench, preloading `AlloMem` by hierarchical poke |
| `mgc_shim.v` | behavioural stand-in for the one Catapult library cell the archived netlists use |
| `guard_experiment/` | the negative result: `guard.patch` extends the free-running-loop guard to any loop reading a `Wire`, and `rtl_base/` vs `rtl_guard/` are the netlists before and after. `emit.py` re-emits `dot_product_four_links.py`; `run_sc.py` runs it against `$ALLO_ROOT`. |

## Netlists

The `pe_wire` / `pe_stream` / `pe_channel` netlists the runners default to are **not in
the checkout**. They survive in the **`choonsik1/allo`** history at
`0eff4888:agents/noc/rtl/<design>/rtl.v` — not in a `choonsik1/SystemC-emitter`
repository, which does not exist (`SystemC-emitter` is a *branch* of `choonsik1/allo`);
`pe_wire` has the same hash at `779e4350^`. Extract them into a sibling `noc/rtl/`
directory, which is where `run.sh` / `run_mulacc.sh` look (`$S/../noc/rtl`). This was
already the case before this directory moved — the scripts are unchanged.

`guard_experiment/` *does* ship netlists, so those cases run without any extraction:

```bash
cd tests/systemc/rtlsim
RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_wire
RTLDIR=$PWD/guard_experiment/rtl_base  ./run_mulacc_xrun.sh pe_wire
RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_stream -d CONNECTIONS_FIFO
RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_channel
```

`RTLDIR` is honoured by the Xcelium runner only.

**`LOCKSTEP` does not run against these netlists**, so the `guard_experiment/`
commands above are *not* a drop-in replacement for the `REPRO.sh` matrix. Section 3
of `REPRO.sh` (the positive control) and the `BREAK_WIRE` fault both pass
`-d LOCKSTEP`, and `tb_mulacc.v:87` then pokes
`u_mul.mul_0_run_inst.v8_and_cse` — a net that exists only in the original
(`0eff4888`) netlists. Against `rtl_base/` or `rtl_guard/` that hierarchical
reference does not elaborate. `RTLDIR` *does* pass through `REPRO.sh` to the
runner, so pointing the whole matrix at `guard_experiment/` looks like it works —
but sections 3 and the `BREAK_WIRE` fault then die in elaboration and the runner
prints `XRUN FAILED`, which `REPRO.sh`'s `grep -oE 'PASS|FAIL .*'` does not match.
Until 2026-09-24 those rows printed as a **blank line**, indistinguishable from a
pass at a glance; they now print `NO VERDICT`. Treat the shipped netlists as
covering the three boundaries and `BREAK_DATA` only. (Also in
`docs/source/extensions/catapult_systemc.rst`, "Replaying from the committed
netlists".)

## What the matrix is supposed to show

`Wire` failing 8/8 at **all 18 pacings** is the result under test, not a
regression: it is [limitation 22](../../../docs/source/developer/limitations.rst),
and `dev/records/systemc/rtlsim/results.txt` has recorded exactly that since the
first run. The same is true of `LOCKSTEP` passing at `ACC_RST_DELAY` 3 and 4 and
failing at 0, 1, 2, 5 and 6 — the two-cycle window is the positive control, and its
width is the measurement. A run in which `Wire` passed a pacing, or `LOCKSTEP`
passed outside 3..4, would be the surprise.
