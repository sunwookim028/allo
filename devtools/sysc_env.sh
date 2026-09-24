# =====================================================================================
# Environment for the SystemC / Catapult / Xcelium flow in THIS checkout.
#
#   source devtools/sysc_env.sh
#
# Everything that needs csim, csynth or RTL cosim needs this. Source it; do not exec it.
# =====================================================================================

# --- conda -----------------------------------------------------------------------------
# Must be an interactive-style activate. `conda run -n allo` is NOT equivalent: the JIT
# simulator fails there with "Unknown function top".
source /home/zsm9/miniconda3/etc/profile.d/conda.sh
conda activate allo

# --- this checkout ---------------------------------------------------------------------
# WITHOUT this the import silently grabs the INSTALLED /home/zsm9/allo instead of the
# working tree, and you debug an emitter you are not editing.
export PYTHONPATH=/home/zsm9/allo_sup${PYTHONPATH:+:$PYTHONPATH}

# --- LLVM (JIT simulator only) ---------------------------------------------------------
# build-rhel8 ONLY. Overriding this with plain build/ aborts on GLIBC_2.33.
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8

# --- Catapult / SystemC ----------------------------------------------------------------
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home
export SYSTEMC_HOME=$MGC_HOME/shared
export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu

# --- Xcelium (RTL cosim) ---------------------------------------------------------------
export NCSim_NC_ROOT=/opt/cadence/XCELIUM2403
export NC_ROOT=$NCSim_NC_ROOT
export CDS_LIC_FILE=5280@en-license-05.coecis.cornell.edu

# --- PATH ------------------------------------------------------------------------------
# conda FIRST. The login profile puts $MGC_HOME/bin ahead of conda, and Catapult's python3
# has no nanobind -- the subprocess that vhls/vitis csim builds spawn then fails, producing
# mass FALSE test failures that look like emitter bugs.
export PATH=/home/zsm9/miniconda3/envs/allo/bin:$MGC_HOME/bin:$NCSim_NC_ROOT/tools.lnx86/bin:$PATH

# --- SystemC compile flags -------------------------------------------------------------
# -rpath so the compiled sim finds the SystemC runtime WITHOUT needing LD_LIBRARY_PATH.
export ALLO_CXX_EXTRA="-DSC_INCLUDE_DYNAMIC_PROCESSES -Wl,-rpath,$MGC_HOME/lib -Wl,-rpath,$(ls -d $MGC_HOME/shared/lib/Linux/gcc-*-64 | head -1)"

# --- runtime ---------------------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
unset LD_PRELOAD                       # a stale LD_PRELOAD breaks the Catapult make

# NOTE ON LD_LIBRARY_PATH — deliberately NOT exported here.
# run_experiment.sh scopes it PER COMMAND: build with $CONDA_PREFIX/lib, run the compiled
# sim with Catapult paths only. Keep that pattern rather than setting one global value;
# a compiled SystemC sim run with conda's lib ahead of Catapult's has been seen to fail
# with "libgmp.so.11: cannot open shared object file".

echo "sysc_env: allo=$(python -c 'import allo,os;print(os.path.dirname(allo.__file__))' 2>/dev/null)"
echo "          MGC_HOME=$MGC_HOME   XCELIUM=$NCSim_NC_ROOT   OMP=$OMP_NUM_THREADS"
