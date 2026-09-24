# EVA on the Allo SystemC backend

End-to-end example of emitting the EVA chip (`eva_sb_syscredit_rtprime`, the
runtime-prime variant) through Allo's SystemC backend, then compiling and running
it as a MatchLib Connections csim.

## Layout
- `eva_sb_syscredit_rtprime.py` — the EVA chip source (copied from
  `pe_core_implementation/Vitis_HLS/bubble_model/final_runs/`).
- `eva_workloads.py` — workload helpers (`load_prog_packets`, ...) used by the cosim.
- `build_eva_systemc.py` — emits the chip via `df.build(..., target="systemc",
  mode="csim")` at M=N=1, writing the project to `generated/`.
- `cosim_eva_systemc.py` — **functional cosim**: emits + drives a real passthrough
  workload and checks `out_e` against a numpy golden (ramp). No JIT simulator needed.
- `generated/` — the emitted SystemC project:
  - `kernel.cpp` — the full design: 22 `SC_MODULE`s, 32 `AlloFifo`s, fp16 `half`
    types, MatchLib Connections, and the auto-generated `sc_main` testbench
    (reads `input*.data`, writes `output*.data`).
  - `csim.sh` — g++ compile (against MatchLib) + run.  `kernel.h`, `Makefile`,
    `run.tcl` — supporting build files.

## Reproduce
```bash
# env (adjust MGC_HOME to your Catapult install)
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home
SCH=<a dir with>: include -> $MGC_HOME/shared/include, lib -> $MGC_HOME/shared/lib/Linux/gcc-10.3.0-64
export SYSTEMC_HOME=$SCH
export ALLO_CXX_EXTRA="-DSC_INCLUDE_DYNAMIC_PROCESSES -Wl,-rpath,$MGC_HOME/lib"
export LD_LIBRARY_PATH=$MGC_HOME/lib:$SCH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=<allo checkout>          # e.g. /home/zsm9/allo_sup

# emit, then compile + run
python build_eva_systemc.py
cd generated && ./csim.sh
```

## Status
- **Emits** cleanly, **compiles** with 0 g++ errors, **runs**.
- **Functional cosim: PASS (bit-exact).** `cosim_eva_systemc.py` drives the 1x1
  passthrough workload and gets `out_e = [1,2,3,4,5,6]` == the ramp golden.

### Notes / gotchas
- `prime_cfg` (runtime prime) is the **first** positional arg (the `node` kernel
  is discovered first); pass `int32[M,N]` all = `PRIME_TOKENS` (6).
- rtprime's runtime-prime credit flow needs a **generous NSTEP margin** (>=160).
  At NSTEP=55 the tokens never drain and every output is zero; at ~215 the ramp
  passes through cleanly. (The compile-time-prime archive variant drained at 55.)
- Scale: this is 1x1 (~22 modules). An 8x8 build is `M*N + 8*(M+N)` = 192 kernel
  instances -> a much larger emit/compile (feasibility unverified); the captured
  8x8 mmm/fft goldens would require it, but a 1x1 numpy-golden cosim already
  proves functional correctness of the systemc emission.
