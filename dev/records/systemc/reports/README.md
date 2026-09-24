# SystemC backend — test & cosim reports

Full logs (not just summaries) from validating the Allo SystemC backend: the
dataflow cosim sweeps, per-design golden checks, EVA emit/compile/cosim, the
`static tb` regression, and the parallel `sim(timing)` reference-crash
investigation. Logs contain MatchLib `CONNECTIONS-101 ... wasn't reset` warnings
(benign: dangling boundary FIFOs) — filter them with
`grep -v CONNECTIONS-101` when skimming.

## cosim_sweeps/  — full `tests/dataflow` sweeps (sim vs systemc, per design)
Chronological; each is a `df.build(target="simulator")`-monkeypatch sweep that
also builds+runs the systemc csim and diffs. Tags: `COSIM_OK` / `COSIM_MISMATCH`
/ `CSIM_RUN_FAIL`.
- `dataflow_cosim_sweep_5_FINAL_after_static_tb.log` — **the authoritative one**:
  18 OK / 2 mismatch (empty/full) / 7 fail (stateful).
- `..._1_initial` … `_4_...` — earlier sweeps across the fix history.
- `slow_systolic_batch_1200s_cap.log` — the big systolic designs at a 1200s cap.

## per_test/  — per-design correctness
- `test_<name>__systemc_vs_numpy_golden.txt` — systemc output vs a numpy golden
  (bypasses the JIT sim). 1D_systolic / tiled_systolic / weight_stationary /
  daisy_chain / systolic_conv — **all correct**.
- `test_smith_waterman_systolic__systemc_vs_golden.log` — the int8 tb-I/O bug
  hunt (values matrices).
- `test_systolic_conv__systemc_vs_numpy_golden.txt` — the "mismatch" that was a
  golden error (correlation vs convolution), systemc is correct.
- `fanin_fix__...`, `stream_alias_fix__...` — cooperative_gemv (fan-in) and
  large_scale_gemm (E115 one-port) fixes, verified COSIM_OK.
- `test_systolic_variants__which_one_mismatched.log` — which of the 5
  `test_systolic` functions was the flagged one (= smith_waterman).
- `slow_systolic_numpy_golden_summary.log` — one-line verdict per slow design.

## eva/  — EVA on SystemC (chip = eva_sb_syscredit_rtprime)
- `eva_rtprime_1x1_passthrough__cosim_nstep215_PASS.log` — **the win**: rtprime
  1x1 passthrough cosim, `out_e == ramp` bit-exact.
- `eva_rtprime_1x1_passthrough__cosim_attempt1_nstep55_FAIL.log` — same workload
  at NSTEP=55 → all zeros (runtime-prime credit flow needs a bigger margin).
- `eva_rtprime_1x1__systemc_emit.log` / `__systemc_compile.log` — emit + g++.
- `eva_8x8__emit_feasibility.log` — 8x8 emits (140k lines, 54s).
- `eva_8x8__compile_feasibility_and_runtime_segv.log` — 8x8 compiles (~20s,
  1.1GB) but the binary segfaults (stack overflow constructing the tb).
- `eva_8x8__run_with_unlimited_stack_OK.log` — proves it's a stack overflow
  (`ulimit -s unlimited` → runs).  `..._run_static_tb_default_stack.log` — the
  `static tb` emitter fix → runs on the default stack.
- `eva_archive_1x1_passthrough__cosim_PASS.log` — earlier proof on the archive
  chip (the methodology).
- `eva_fwd_1x1_mmm_workload__all_zeros_nstep200.log` — the first EVA attempt
  (wrong chip + NSTEP too tight).

## regression/  — did anything break?
- `static_tb_regression__backend28_eva1x1_eva8x8.log` — the `static tb` change:
  **backend suite 28/28**, EVA 1x1 cosim PASS, 8x8 runs on default stack.
- `test_systemc_backend_suite.log` — a `test_systemc_backend.py` run.

## sim_regression/  — NOT the systemc backend
The parallel `sim(timing)` rewrite crashed the JIT *reference* (an MLIR
block-insert assert) for func_index / hierachical_function / tiled_gemm /
region_toparg_aliasing. `simcheck2` shows each `SIM_CRASH` while the systemc
build of the same design runs fine. (Later `sim(timing)` commits fixed it — those
four are COSIM_OK again in sweep 5.)

## emitter_build_logs/  — ninja/g++ rebuilds of EmitSystemC.cpp (dev artifacts)

---
### Bottom line
SystemC backend: **28/28** backend suite; **18/20** functional dataflow designs
cosim bit-exact (the 2 misses are the empty/full latency-insensitive gap); the 7
remaining fails are the deferred stateful compile gap. EVA (rtprime) emits +
compiles + **cosims correctly at 1x1**; 8x8 emits + compiles and runs with the
`static tb` fix.
