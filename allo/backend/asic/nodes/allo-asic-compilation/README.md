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
- `clock_period`: target clock period in nanoseconds. The node converts this to
  the MHz value required by Allo/Vitis.
- `python_bin`: Python executable, default `python`.
- `allo_setup_script`: shell script sourced before preflight and compilation to
  configure the prebuilt LLVM backend. It defaults to
  `/work/shared/common/allo/setup-llvm-main.sh`.

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
