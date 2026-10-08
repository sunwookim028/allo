# Source (bash) on zhang-21 for the AMC side. Reuses the AMC build recorded in
# dev/records/open_hls/amc_exploration_2026-10-02.rst (amc-dialect fe60c121).
# Run the scripts under `scl enable gcc-toolset-13 --`. TMPDIR must be a LOCAL
# disk (U1: on NFS the second AMCModule call fails in shutil.rmtree).
AMC=/work/shared/users/phd/sk3463/scratch/amc
export PATH=$AMC/env/bin:$AMC/amc-dialect/build/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH
export PYTHONPATH=$AMC/amc-dialect/allo:$AMC/amc-dialect/build/tools/amc/python_packages/amc_core:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TMPDIR=${AMC_LOCAL_TMP:-/tmp/$USER-u4e-amc}; mkdir -p $TMPDIR
unset LLVM_BUILD_DIR
