# MLIR ↔ Vitis HLS Static Accumulator Playground

This mini-project explores how to represent a simple scalar accumulator with
static state in both Vitis HLS C++ and MLIR (MemRef dialect), and how to bridge
the two via Allo’s `emit-vivado-hls` translator.

## Motivation

The stock Allo Vitis HLS backend does not currently emit `static` storage for
stateful kernels. By experimenting in this playground we add a minimal patch to
the translator, then verify that both the hand-written HLS kernel and the MLIR
variant behave equivalently.

## Layout

- `hls/` – Vitis HLS reference kernel (`kernel.h`, `kernel.cpp`) plus a small
  C++ testbench and build helper.
- `mlir/` – MemRef-based MLIR IR featuring a `memref.global` that models the
  static accumulator state.
- `scripts/` – Automation utilities, e.g. `verify.py` for running the C++ tests
  and checking the MLIR → HLS translation.
- `docs/` – Notes about the translator patch and any follow-up findings.
  See `docs/translation_static_notes.md` for the minimal translator change.

## Prerequisites

- A working C++17 toolchain (e.g. `g++`) for compiling the reference testbench.
- A built Allo toolchain, specifically `mlir/build/bin/mlir-translate` with the
  Vitis HLS translation registered.

## Verification Workflow

1. Run the HLS-side testbench:
   ```bash
   cd /home/sk3463/allo/sunwoo_playground/mlir_to_hls
   ./scripts/verify.py --run-hls
   ```
2. After the translator patch is applied and rebuilt, confirm the MLIR path:
   ```bash
   cd /home/sk3463/allo/sunwoo_playground/mlir_to_hls
   ./scripts/verify.py --run-mlir
   ```
   Or run the translator directly:
   ```bash
   /home/sk3463/allo/mlir/build/bin/mlir-translate \
     --emit-vivado-hls \
     /home/sk3463/allo/sunwoo_playground/mlir_to_hls/mlir/accumulate.mlir
   ```
3. For complete coverage (C++ run + translation checks) omit flags:
   ```bash
   ./scripts/verify.py
   ```

> The verification script will be added in a later step; until then these
> commands serve as placeholders for the intended workflow.


