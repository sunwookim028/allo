# M-R1 env: the harness env, then gcc-toolset-13 first (env-zhang21.sh alone resolves g++ to Catapult's 10.3)
W=/work/shared/users/phd/sk3463/scratch/wt-mr1
source $W/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
export VERILATOR=/work/shared/users/phd/sk3463/tools/verilator/bin/verilator
export PYTHONPATH=$W
export OMP_NUM_THREADS=8
