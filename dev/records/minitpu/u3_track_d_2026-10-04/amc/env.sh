# Source (bash) on zhang-21 for the AMC side. Reuses the AMC build recorded in
# dev/records/open_hls/amc_exploration_2026-10-02.rst (amc-dialect fe60c121).
AMC=/work/shared/users/phd/sk3463/scratch/amc
export PATH=$AMC/env/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH
export PYTHONPATH=$AMC/amc-dialect/allo:$AMC/amc-dialect/build/tools/amc/python_packages/amc_core:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p $AMC/tmp && export TMPDIR=$AMC/tmp   # AMCModule uses tempfile.mkdtemp
