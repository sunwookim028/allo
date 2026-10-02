source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
module load catapult-2024
export PATH=/opt/cadence/XCELIUM2403/tools.lnx86/bin:$PATH
unset LD_PRELOAD
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
export SYSTEMC_HOME=$MGC_HOME/shared
export ALLO_CXX_EXTRA="-DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM -L$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/lib"
