#!/usr/bin/env bash
set -eo pipefail

eval "$(micromamba shell hook -s bash)"
micromamba activate allo

source /work/shared/common/allo/setup-llvm19.sh
source /work/shared/common/allo/vitis_2023.2_u280.sh

export MINITPU_OUTPUT_ROOT="${MINITPU_OUTPUT_ROOT:-/work/shared/users/phd/sk3463/miniTPU_outputs}"
mkdir -p "${MINITPU_OUTPUT_ROOT}"
echo "[sunwoo_playground][setup] Output root: ${MINITPU_OUTPUT_ROOT}"

