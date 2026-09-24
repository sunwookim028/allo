# SystemC backend — measurement records

Dated records of what was measured, kept out of the published Sphinx site. Nothing here
is a design or a test; it is evidence. The designs are in `examples/systemc/`, the
harness that produced these is in `tests/systemc/`.

| Path | What |
|---|---|
| `VERDICTS.md` | per-example verdict for every `tests/dataflow` design run against the SystemC backend, simulator vs `mode="csim"` on identical seeded inputs, with the failures categorized by root cause |
| `reports/` | the full logs behind those verdicts — cosim sweeps, per-design golden checks, EVA emit/compile/cosim, the `static tb` regression, the `sim(timing)` reference-crash investigation. `reports/README.md` is the index. |
| `rtlsim/results.txt` | the verdict matrix from `tests/systemc/rtlsim/REPRO.sh` |
| `rtlsim/ref_xsim/` | two `xsim` runs of the **same** `pe_wire` netlist, one PASS and one FAIL, kept so the divergence can be diffed |
| `generated/` | emitted SystemC kept for reference: what the emitter produced for five of the `examples/systemc/` designs. Output, not source — regenerate by running the matching `.py`. `tests/systemc/synth_*.tcl` read `stream_boundary.cpp` from here. |

These files were written by their runs and are **not** edited afterwards. Read them as
history: `VERDICTS.md` and `reports/README.md` were accurate when the runs happened, and
the paths they mention (`scratchpad/harness.py`, `examples/systemc/reports/`) are the
paths of that time, not of today.

## They no longer match the current emitter (checked 2026-09-24)

Re-running the six designs on `origin/main` reproduces three of the five archived
`.cpp` and not the other two, for reasons that predate this directory:

- `stream_boundary.py` asserts `SC_MODULE(compute_0)` and fails, because
  `df.build(top, target="systemc").hls_code` now returns **Vitis HLS** C++
  (`ap_int.h`, `hls_stream.h`, `extern "C"`) rather than SystemC. The archived
  `generated/stream_boundary.cpp` *is* SystemC, so it is not currently
  regenerable — which also means the two `tests/systemc/synth_*.tcl` depend on
  the archive rather than on a fresh emission.
- `tiled_systolic.py` asserts `AlloMem<` / `AlloMemW<` and fails, because the
  memory boundary became RAM pins: the emitter now writes three `AlloMemPins<`.
  `mem_port_reverse.py` and `mem_port_scatter.py` were updated for that and
  pass; `tiled_systolic.py` was not.

Neither is caused by the move — both scripts are byte-identical to what they
were — and neither is covered by `tests/dataflow/test_systemc_backend.py`, which
is why they went unnoticed. Recorded here rather than fixed, because fixing them
is a change to the designs and to the emitter, not to where files live.


## A static trace that disagrees with the emission finding (2026-09-24)

The finding above is that `df.build(top, target="systemc").hls_code` returns
Vitis HLS C++ rather than SystemC. **Reading the code suggests it should
work**, so whoever picks this up should not assume the dispatch is simply
missing:

- `allo/dataflow.py:858` forwards `target=target` unchanged into
  `s.build(...)`; it special-cases only `aie` and `simulator` before that.
- `allo/customize.py:1532` has `case "systemc": platform = "systemc"`.
- `allo/backend/hls.py:484` has `case "systemc": success =
  allo_d.emit_systemc(self.module, buf)`, and `self.hls_code` is read from that
  same buffer at line 499.

So the three hops that would have to be broken are each present. The
observation and the trace disagree, which means the cause is somewhere
narrower -- a region path that rebuilds with a default target, a cached
`hls_code`, or an emitter that silently falls through and returns success.

Not settled here: this checkout's MLIR bindings predate the SystemC merge, so
`import allo.dataflow` raises before reaching any of it. It needs one run on a
tree with current bindings, which is minutes of work for someone who has one.
