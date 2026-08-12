# Allo build node

This node loads a parameterized Allo Python design and calls its configured
entrypoint with:

```python
build(project, target, mode, configs)
```

Only `backend: vitis` is currently supported. Backend selection is handled by
explicit branches in `backend.py`; later backends can add their own target,
configuration, tool checks, artifact discovery, and validation without adding
backend-specific path parameters to the graph. The node currently runs Vitis C
synthesis, validates the enriched ASIC manifest, and publishes the complete
Allo project, debug artifacts, manifests, and synthesis RTL.

Important parameters are:

- `allo_design_file`: absolute path, or a path relative to the design's
  `construct_path`.
- `allo_entrypoint`: design-module callable, default `build`.
- `backend`: currently `vitis`; unsupported values fail explicitly.
- `backend_options`: shell-safe comma-separated `key=value` options interpreted
  by the selected backend branch. The Vitis default is `device=u280`.
- `build_mode`: currently expected to be `csyn`.
- `macro_clock_period`: Vitis HLS and hardened-macro target period in
  nanoseconds. The node converts this to the MHz value required by Allo/Vitis.
- `clock_period`: full-chip target period in nanoseconds. It must be greater
  than or equal to `macro_clock_period`, ensuring macros are built for a clock
  at least as fast as the chip that instantiates them.
- `python_bin`: Python executable, default `python`.
- `allo_setup_script`: shell script sourced before preflight and compilation to
  configure the prebuilt LLVM backend. It defaults to
  `/work/shared/common/allo/setup-llvm-main.sh`.
- `allo_testbench_enabled`: when true, freeze a design-supplied workload after
  HLS succeeds while the same Allo environment and design parameters remain
  active.
- `allo_testbench_workload_factory`: design-module callable returning the
  backend-independent call signature, initial argument arrays, and expected
  output arrays. It defaults to `testbench_workload`.
- `allo_testbench_top_function`: fallback top-function name if the workload
  does not provide one.

The frozen outputs are `workload-manifest.json` and `workload-vectors/`.
Array dtype, shape, element width, and count are recorded in JSON; values are
serialized as one hexadecimal element per line. This means downstream
testbench generation does not receive design-only parameters such as array
size, FIFO depth, or dtype—it consumes the realized workload and HLS outputs.

The success marker is written only after the pre-HLS/final manifests, zero
unmatched joins, debug directory, and synthesized Verilog have all been
validated.

This mflowgen version does not expand node parameters inside assertion text.
Parameterized environment checks therefore run in `preflight.py`, invoked as
the first expanded command. Static mflowgen preconditions only verify that the
node's required scripts were staged. The preflight checks the selected Python
executable, Allo import, design path, build mode, backend selection, and backend
tool availability before `run.sh` starts.

Preflight and compilation run from the same `run.sh` command, so the backend
module and LLVM setup are each loaded only once. Backend selection owns module
selection: the Vitis branch loads `xilinx-2022.1`, and unsupported backends
error. The wrapper disables Bash
`nounset` while sourcing because the shared setup script unsets and subsequently
reads `PYTHONPATH`.

The wrappers also clear inherited `LD_LIBRARY_PATH`, `LIBRARY_PATH`, compiler
include variables, `GCC_EXEC_PREFIX`, and `LD_PRELOAD` before sourcing the
Allo setup. This prevents commercial ASIC tool modules—especially Synopsys's
bundled `libstdc++.so.6`—from overriding the GCC 13 runtime required by Allo's
compiled MLIR extension. `PATH` is retained so a loaded backend such as Vitis
remains discoverable.
