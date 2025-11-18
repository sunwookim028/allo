#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  echo "[miniTPU_HLS][setup] $*"
}

try_source_vitis() {
  local candidate="$1"
  if [[ -f "${candidate}" ]]; then
    log "Sourcing Vitis environment: ${candidate}"
    # shellcheck disable=SC1090
    source "${candidate}"
    return 0
  fi
  return 1
}

log "Project root: ${PROJECT_ROOT}"

if [[ -n "${XILINX_VITIS:-}" ]]; then
  log "XILINX_VITIS already set to ${XILINX_VITIS}"
else
  DEFAULT_CANDIDATES=(
    "/opt/Xilinx/Vitis/*/settings64.sh"
    "${HOME}/Xilinx/Vitis/*/settings64.sh"
    "/tools/Xilinx/Vitis/*/settings64.sh"
  )
  for pattern in "${DEFAULT_CANDIDATES[@]}"; do
    for path in ${pattern}; do
      if try_source_vitis "${path}"; then
        break 2
      fi
    done
  done
fi

if command -v vitis_hls >/dev/null 2>&1; then
  log "vitis_hls located at $(command -v vitis_hls)"
else
  log "WARNING: vitis_hls not found on PATH. Please install or adjust settings."
fi

log "Environment setup complete."


