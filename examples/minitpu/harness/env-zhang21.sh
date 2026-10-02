# Source (bash) on zhang-21 before running the harness against every backend.
# Order matters: the allo env's activate script prepends gcc-toolset-13, and
# Catapult's module prepends its own bin/ (with a python of its own), so the
# conda python is pinned by path and Catapult's g++ 10.3 is put first last --
# Catapult's libsystemc needs its libstdc++ (GLIBCXX_3.4.26 at link time).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allo                      # sets LLVM_BUILD_DIR to the shared build
module load catapult-2024                # MGC_HOME + licence variables
export PATH="$MGC_HOME/bin:/opt/cadence/XCELIUM2403/tools.lnx86/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH"
export ALLO_PYTHON="$CONDA_PREFIX/bin/python"
unset LD_PRELOAD
export OMP_NUM_THREADS=8
export SYSTEMC_HOME=$MGC_HOME/shared
export ALLO_CXX_EXTRA="-DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM -L$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/lib"
