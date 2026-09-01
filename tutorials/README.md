<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Allo Tutorials

A hands-on, example-driven walkthrough of Allo — a Python-embedded, MLIR-based
accelerator design language. Each notebook pairs explanation with runnable code:
a `@kernel` describes *what* to compute, a `Schedule` describes *how* it maps to
hardware, and a backend (`cpu` / `vitis`) runs or synthesizes it.

| #   | Notebook                                     | Topics                                                                                                                                                                                                                              |
| --- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [`01_language.ipynb`](01_language.ipynb)     | Kernel syntax, data types, variables & scope, loops, conditionals, operators & typing styles, indexing & bit manipulation, streams, stateful variables, templates/`consteval`, spatial mapping, and clang-style diagnostics.        |
| 2   | [`02_scheduling.ipynb`](02_scheduling.ipynb) | `kernel.schedule()`, target selection, tagging vs. structural primitives (pipeline/unroll/partition/split/reorder/tile/flatten/compute_at/buffer_at/reuse_at/outline), kernel composition, streaming (`streamline`), and debugging. |
| 3   | [`03_simulation.ipynb`](03_simulation.ipynb) | Running kernels on the CPU backend and Vitis C-simulation, the calling convention, dataflow stream simulation, run modes, and caching.                                                                                              |
| 4   | [`04_backend.ipynb`](04_backend.ipynb)       | Vitis HLS codegen, target/interface configuration, C-to-RTL synthesis and report parsing, project scaffolding, and the hardware flow.                                                                                               |

Read them in order — later notebooks build on earlier ones.

## Prerequisites

- Build Allo from source and activate its environment (see the top-level
  [`docs/README.md`](../docs/README.md)):

  ```bash
  conda activate allo
  pip install -v -e .        # builds the MLIR/C++ backend and runtime libraries
  ```

## Running the notebooks

Launch Jupyter, open a notebook under `tutorials/`, and run the cells top to
bottom:

```bash
conda activate allo
jupyter lab            # or: jupyter notebook
```

The first code cell just imports NumPy and Allo; each notebook runs against the
`allo` package installed in the active environment.

## Vitis toolchain

Notebooks 3 and 4 include cells that exercise the real Vitis HLS toolchain
(C-simulation and synthesis). Those cells are **gated** on
`is_vitis_available()` and skip gracefully with a printed note when the
toolchain is not present, so the notebooks run end-to-end either way.

- HLS C++ codegen (`s.export("vitis").hls_code`) needs **no** toolchain and is
  the fastest way to inspect what Allo emits.
- C-simulation and synthesis need a Vitis HLS install. Point `XILINX_VITIS` at
  the install path; set it to an invalid path to deliberately skip synthesis and
  save time.
- Hardware emulation / on-board runs (`hw_emu` / `hw`) additionally require a
  platform (`export PLATFORM=/path/to/<shell>.xpfm`) and are only *described* in
  the tutorials, not executed.

## Going deeper

The tutorials are a guided tour; the authoritative references live alongside the
code:

- [`../ALLO.md`](../ALLO.md) — concise DSL reference for writing kernels and schedules.
- [`../docs/frontend.md`](../docs/frontend.md), [`../docs/scheduling.md`](../docs/scheduling.md),
  [`../docs/simulation.md`](../docs/simulation.md), [`../docs/typing_rules.md`](../docs/typing_rules.md).
- [`../tests/`](../tests) — worked examples for every feature.
- [`../allo/library/`](../allo/library) — reusable transformer and systolic-array designs.
