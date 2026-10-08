#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# M-R1 from a checkout with built bindings: shim up to date, then every oracle.json launch through the region.
# Needs the pinned MiniTPU clone (oracle.py's CLONE), the pinned Verilator and gcc-toolset-13.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
set +eu   # the env script (conda, module) is not written for -eu
source "$root/examples/minitpu/harness/env-zhang21.sh" >/dev/null 2>&1
set -eu
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH        # env-zhang21.sh puts Catapult's g++ 10.3 first
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
export VERILATOR=${VERILATOR:-/work/shared/users/phd/sk3463/tools/verilator/bin/verilator}
export PYTHONPATH=$root OMP_NUM_THREADS=8
"$ALLO_PYTHON" "$here/gen_shim.py" --check
"$ALLO_PYTHON" "$here/run_kernel.py" "${@:---all}"
