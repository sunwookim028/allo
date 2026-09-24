# source env.sh -- what the scripts here expect (the `allo` env sets none of it)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate allo
export LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-/home/sk3463/llvm-allo-6b09f739/build}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TMPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scratch/tmp"; mkdir -p "$TMPDIR"
