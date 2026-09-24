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

## The emission finding, settled (2026-09-24)

The static trace was right and the observation was wrong. On a tree with current
bindings, `df.build(top, target="systemc").hls_code` returns **SystemC**:
`SC_MODULE(compute_0)`, `Connections`, no `ap_int.h`, no `hls_stream.h`. The same
build with `target="vitis_hls"` returns the Vitis C++ that was seen. All three
hops work.

What was actually broken is the example. `cea8274a` (2026-08-10, a docs wording
pass) changed `examples/systemc/stream_boundary.py` from `target="systemc"` to
`target="vitis_hls"` — both the emit and the csim — while leaving the
`assert "SC_MODULE(compute_0)" in code` that only a SystemC emission can satisfy.
So the script asked for Vitis and then asserted SystemC, and running it produced
exactly the symptom: "`hls_code` returns Vitis HLS C++". Reverted; the assertion
passes again, and `stream_boundary.py` regenerates SystemC rather than depending
on `generated/stream_boundary.cpp`.

It does not reproduce the archive byte-for-byte — 476 normalised lines differ,
all emitter evolution since the archive was taken (float helpers, `ac_channel`
self-FIFOs, the vendor `Connections::Fifo`). `tests/systemc/synth_*.tcl` still
read the archived `.cpp`, which is the honest thing for a record to do.

`tiled_systolic.py` is untouched and still asserts `AlloMem<` / `AlloMemW<`
against an emitter that writes `AlloMemPins<`. That one really is a design
change, as this file already said.

`generated/vstream_boundary.cpp` is the fossil of the same defect: it is Vitis
HLS C++ (`ap_int.h`, `hls_stream.h`) in a directory whose table calls itself
"emitted SystemC", written by `stream_boundary.py` while it was pointed at
`vitis_hls` and named for the `vstream_boundary.prj` that commit invented.
Nothing reads it. Kept, because records are not edited to look tidier than the
runs that made them.

## The EVA reference, and what the A/B pair had to be (2026-09-24)

`reports/zhang21_2026-09-24/` asked for `cosim_eva_systemc.py` at NSTEP=215
emitted with the old and the new emitter. The pair is in
`eva_nstep215_ab_2026-09-24/`, but **the emitter is not the axis**, which took a
rebuild to find out rather than a reading:

- The committed reference was regenerated at `72c70dcb`. Reverting all of
  `mlir/lib/Translation/` and `mlir/include/allo/Translation/` to `72c70dcb`
  (five files, 546 lines) and rebuilding the bindings changes the EVA emission in
  **zero** lines after SSA-name normalisation.
- The signed → unsigned difference comes from `allo/ir/builder.py`: `3de74846`
  and `094ab413` (#612), which attach the `unsigned` `UnitAttr` to
  `GetIntSliceOp`. Removing just that reproduces the reference's types exactly.

A second, independent difference the handoff did not have: the region's MLIR
argument order is now the **declared** order, not a discovery order, so
`prime_cfg` moved from first to last and every `input<k>.data` index moved with
it. `cosim_eva_systemc.py` still passed the old order, which does not raise —
the arity matches, so each array is written to the wrong slot. Fixed in the
script, with an assertion against the module's signature.

Neither side of the pair was run: this host has no Catapult, so there are no
`output*.data`. The `*.data` in `eva_nstep215_ab_2026-09-24/` are ordinary
tracked files — the ignore rule that made `git add -f` necessary lives in
`examples/eva/generated/.gitignore` and does not reach `dev/records/`.
