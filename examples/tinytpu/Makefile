# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch -> ACT -> TinyTPU, one model at a time.
# The walk-through, with every command's output: README.md
#
#   make mlp                        mlp_small, compiled by ACT, run on the design
#   make mlp MODEL=mlp_deep TOP=5   another model (models.py), five mappings
#   make mlp-compile                stop after ACT; run on isa_ref, no build
#   make mlp-cosim MODEL=mlp_tiny   RTL cycles: one csynth, cosim per layer (Vitis)
#   make mlp-suite                  every model in the suite, one table
#
# The design's parameters are environment variables, e.g. `TPU_T=8 make mlp`.

MODEL ?= mlp_small
TOP   ?= 3
# This host's LLVM build, used only if it exists: elsewhere, export your own.
LLVM_BUILD_DIR  ?= $(wildcard /home/sk3463/llvm-allo-6b09f739/build)
OMP_NUM_THREADS ?= 8
export LLVM_BUILD_DIR OMP_NUM_THREADS

# This shell's python if it can import allo the way the scripts do (with the
# checkout on sys.path; allo need not be pip-installed), else the `allo` env's.
PYTHON ?= $(shell python -c 'import sys; sys.path.insert(0, "../.."); import allo' \
            >/dev/null 2>&1 && echo python \
            || echo conda run --no-capture-output -n allo python)
export PYTHON

.PHONY: mlp mlp-compile mlp-cosim mlp-suite

mlp:
	$(PYTHON) workloads/demo.py $(MODEL) --top $(TOP)

mlp-compile:
	$(PYTHON) workloads/demo.py $(MODEL) --top $(TOP) --no-sim

mlp-cosim:
	$(PYTHON) workloads/run.py --cosim $(MODEL)

mlp-suite:
	$(PYTHON) workloads/run.py
