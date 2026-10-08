# Source before running these scripts. RTLGEN_ROOT is Kai's clone + env, built per
# dev/records/open_hls/rtlgen_exploration_2026-10-02.rst (zhang-21 default below).
R=${RTLGEN_ROOT:-/work/shared/users/phd/sk3463/scratch/rtlgen}
export PATH=$R/env/bin:$PATH
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/xdgcache XILINX_VITIS=/nonexistent
unset LLVM_BUILD_DIR
export PYTHONPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}
