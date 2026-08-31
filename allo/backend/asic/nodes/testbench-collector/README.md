# User testbench collector

This node packages user-owned simulation collateral without translating it.
VCS consumes the SystemVerilog testbench directly; sv2v is reserved for
synthesizable design RTL.

Simple mode sets `testbench_path` and `testbench_file`. The main file is the
only compiled source, while neighboring files are packaged as collateral. Use
`testbench_manifest` when multiple HDL sources must be compiled in a specific
order. Normal manifest lines are HDL files or directories, `@data path` adds
runtime collateral, `!path` excludes a file or tree, and `#` starts a comment.

The packaged `testbench/testbench.f` uses paths appropriate for a downstream
input named `testbench`. `testbench-contract.json` records the testbench top,
DUT instance, pass/failure markers, timeout, file list, and runtime-argument
file used uniformly by RTL, FFGL, and BAGL simulation nodes.

When the simulation node's `waveform` parameter is enabled, it passes
`+ASIC_DUMP_VCD`. User testbenches must respond by producing a textual VCD at
`outputs/run.vcd`, following this pattern:

```systemverilog
if ($test$plusargs("ASIC_DUMP_VCD")) begin
  $dumpfile("outputs/run.vcd");
  $dumpvars(0, TestbenchTop);
end
```

Do not use `$vcdplusfile` for this output: VCS writes binary VPD data through
the VCD+ tasks even when the filename ends in `.vcd`, and PrimeTime power
analysis requires textual VCD. Legacy upstream `design.args` are retained as
compile arguments in `testbench-compile.args`; runtime arguments are kept
separately in `testbench-runtime.args`.
