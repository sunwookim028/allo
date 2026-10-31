# source env.sh -- what the scripts here expect (the `allo` env sets only LLVM_BUILD_DIR)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate allo
export LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-$HOME/tools/llvm-allo/build}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TMPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scratch/tmp"; mkdir -p "$TMPDIR"
