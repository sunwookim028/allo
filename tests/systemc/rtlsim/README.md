# SystemC vs RTL cross-check

The apparatus behind [limitation 22](../../../docs/source/developer/limitations.rst):
a `Wire` boundary that passes SystemC csim but is wrong in Catapult's own netlist,
shown on two independent RTL simulators.

Full write-up, including what each result means:
`docs/source/extensions/catapult_systemc.rst`
(https://sunwookim028.github.io/allo/extensions/catapult_systemc.html).

The design under test is [`examples/systemc/pe_split.py`](../../../examples/systemc/pe_split.py)
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
| `guard_experiment/` | the negative result: `guard.patch` extends the free-running-loop guard to any loop reading a `Wire`, and `rtl_base/` vs `rtl_guard/` are the netlists before and after. `emit.py` re-emits `pe_split.py`; `run_sc.py` runs it against `$ALLO_ROOT`. |

## Netlists

The `pe_wire` / `pe_stream` / `pe_channel` netlists the runners default to are **not in
the checkout**. They survive in the `choonsik1/SystemC-emitter` history at
`0eff4888:agents/noc/rtl/<design>/rtl.v`; extract them into a sibling `noc/rtl/`
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
