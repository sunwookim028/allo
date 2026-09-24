# SystemC backend — validation harness

Things that *act on* the SystemC designs: hand-written RTL testbenches, Catapult
synthesis scripts, and the SystemC-vs-RTL cross-check. The designs themselves are in
[`examples/systemc/`](../../examples/systemc/); the logs these produced are in
[`dev/records/systemc/`](../../dev/records/systemc/).

Nothing here is collected by `pytest` — these drive external EDA tools. The
pytest-visible SystemC tests are `tests/dataflow/test_systemc_backend.py` and
`tests/dataflow/test_stateful_systemc.py`.

**Neither Catapult nor Xcelium is installed on this fork's development host**
(`dev/toolchains.rst`), so nothing in this directory can be exercised here. It is kept
runnable for a host that has them.

## Contents

| File | What it does | Needs |
|---|---|---|
| `csyn_subdir.py` | runs Catapult `csyn` on a design from `examples/systemc/`, from a build **subdirectory** — the workaround for CIN-124 / SCHD-30 port degradation | `MGC_HOME` |
| `synth_compute.tcl`, `synth_source.tcl` | Catapult synthesis of the two `stream_boundary` submodule tops | Catapult |
| `cosim_tb_producer_consumer.v` | hand-written RTL cosim testbench for the producer/consumer design (B = A + 1) | Xcelium or VCS |
| `cosim_tb_mem_port_reverse.v` | the same for the memory-port design; **currently deadlocks**, and the header explains why that is a real csim-vs-RTL divergence rather than a testbench bug | Xcelium or VCS |
| `rtlsim/` | the SystemC-vs-RTL cross-check behind limitation 22 — see its own README | Vivado xsim or Xcelium |

## Running

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

# csyn a design by module name; the module is looked up in examples/systemc/
python tests/systemc/csyn_subdir.py pc_channel pc_channel
ALLO_DESIGN_TOP=producer_0 python tests/systemc/csyn_subdir.py pc_channel pc_channel

# Catapult synthesis of a stream-interface submodule top
catapult -shell -f tests/systemc/synth_compute.tcl
```

The two `.tcl` scripts read an archived emitter output,
`dev/records/systemc/generated/stream_boundary.cpp`. They find it by searching
**upward** from the script for the repository root, so neither the cwd nor the depth
of this directory matters — see `dev/roadmap.md` on why nothing here resolves a root
by counting levels. `csyn_subdir.py` and `rtlsim/guard_experiment/emit.py` do the same.
Regenerate the archived `.cpp` with `python examples/systemc/stream_boundary.py`.

The two `cosim_tb_*.v` testbenches are driven by hand; each file's header gives the
exact `df.build(..., mode="csyn")` call that produces the RTL it expects and the
`xrun` invocation that runs it.
