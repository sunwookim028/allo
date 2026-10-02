#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned Verilator used to run MiniTPU's RTL units as the reference
# for Allo units (README, D-7, D-8): conda-forge verilator 5.052, one exact build,
# in its own conda prefix, so the allo env is untouched.
#
# Usage:
#   scripts/verilator-setup.sh                # install into $PREFIX; print the env to export
#   eval "$(scripts/verilator-setup.sh --env)"
#
# Verilator compiles the C++ it generates with the host compiler, and --timing
# needs C++20 coroutines (g++ >= 10). On RHEL 8 hosts, enable gcc-toolset-13
# (`scl enable gcc-toolset-13 bash`) when running Verilator.

set -euo pipefail

PREFIX="${ALLO_VERILATOR_HOME:-$HOME/.cache/allo/verilator}"
SPEC="conda-forge::verilator==5.052=py312pl5321h9d6c286_0"

print_env() {
    echo "export PATH=$PREFIX/bin:\$PATH"
}

if [ "${1:-}" = "--env" ]; then
    print_env
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$PREFIX/bin/verilator" ]; then
    conda create -q -y -p "$PREFIX" "$SPEC" > /dev/null
fi
"$PREFIX/bin/verilator" --version
print_env
