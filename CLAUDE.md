# allo — Coding Agent Notes

## Quick pitfalls

- **Region arg-order reordering** when scalar in early kernel `args=[...]` → fix in `_build_top`; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **OMP segfault at exit** when GC races with OMP threads — set `OMP_NUM_THREADS=N`, run regions in separate processes; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **`LLVM_BUILD_DIR` is NOT set by the conda env** — neither `conda activate allo` nor `conda run` sets it (verified 2026-09-17), and the simulator asserts `LLVM_BUILD_DIR is not set` without it. Export it explicitly; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Scalar `@df.region()` args** — bare `int32` in `args=[...]` is **rejected** (PR #577); use `int32[1]` → `m_axi`. Auto-capture → `s_axilite` redesign pending upstream; details: `notes/PITFALLS_DATAFLOW_REGION.md`

## Environment

```bash
conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # the env does NOT set this

# OMP_NUM_THREADS no longer has to exceed the kernel-instance count: the
# simulator now sets the OpenMP team size to the section count itself.
# Before that fix a PE blocked on a stream spun in its section, so a team
# smaller than the section count never started the sections that would
# unblock it and the region hung SILENTLY. See notes/ALLO_SHORTCOMINGS.md #11.
export OMP_NUM_THREADS=8
```

## Toolchains on this host

Verified 2026-09-18 on this machine. None of this is installed by the repo, and
the `allo` conda env sets none of it; a migration (e.g. to zhang-21) has to
reproduce or re-point every row. The conda env, `LLVM_BUILD_DIR` and
`OMP_NUM_THREADS` are covered above and are not repeated here.

| Tool | Location | On `PATH` by default? |
| --- | --- | --- |
| Vivado 2023.2 — `vivado`, and the `xsim` trio `xvlog`/`xelab`/`xsim` | `/opt/xilinx/Vivado/2023.2/settings64.sh` | **Yes** — `which xvlog` already resolves to `/opt/xilinx/Vivado/2023.2/bin/xvlog`, so the login profile sources `settings64.sh`. Do not assume that on the new host. |
| Vitis HLS 2023.2 | `/opt/xilinx/Vitis_HLS/2023.2` (`settings64.sh` present) | **No.** `which vitis_hls` finds nothing; scripts source `settings64.sh` themselves — `examples/accelerator/tinytpu_vitis/cosim.py` hardcodes the path in its `VITIS` constant. |
| Verilator 5.051 (`devel rev vUNKNOWN-built20260904-2286359`) | `VERILATOR_ROOT` tree at `~/.local/share/verilator`; driver at `~/.local/bin/verilator` | Yes. Leave `VERILATOR_ROOT` **unset** — the driver derives it and warns if an inconsistent one is exported. |
| Chipyard | `~/chipyard/env.sh` | No; `source` it. It `conda activate`s `/home/sk3463/chipyard/.conda-env`, so it **replaces** the `allo` env — source it in a separate shell. |
| Cadence Xcelium | — | **Not installed here.** `/opt/cadence` does not exist and no `xrun` is on `PATH`; `/opt` holds only `xilinx` among EDA vendors. Earlier notes describing `/opt/cadence/XCELIUM2403` and an `unset LD_PRELOAD` workaround do not apply to this host. (`notes/ALLO_SHORTCOMINGS.md` cites Xcelium cosim results from elsewhere.) |

### Vitis binutils vs. glibc `.relr.dyn`

Vitis 2023.2 ships binutils 2.37, which cannot read this system's glibc:
`unknown type [0x13] section '.relr.dyn'`, then `cannot find libm.so.6`. Both
the csim and the cosim link fail without it.

The fix in tree is **not** a `PATH` override — it is a compiler-driver flag.
`examples/accelerator/tinytpu_vitis/cosim.py` sets `LDFLAGS = "-B/usr/bin"` and
splices it into the generated Vitis script, pointing the driver at the system
linker (2.42) while leaving the rest of the Vitis toolchain in place. Same
story in that directory's `RESULTS_ISA.md`. Any new Vitis flow needs the
equivalent.

### Python for cosim vs. Python for `allo`

The `allo` conda env is **python 3.12** (`3.12.13`); the miniconda **base** env
is **python 3.14** (`3.14.6`). This matters because CMake picks them
independently: `/home/sk3463/allo-chia-wt/build/CMakeCache.txt` records
`Python3` as the env's 3.12 but `Python` as base 3.14, and nanobind took its
suffix from the latter — `NB_SUFFIX=.cpython-314-x86_64-linux-gnu.so`. The
chia worktree's bindings are therefore tagged `cpython-314` and the 3.12
interpreter will not import them. Pass an explicit `-DPython_EXECUTABLE=` as
well as `-DPython3_EXECUTABLE=` when configuring.

The python 3.14 venv that carried `cocotb 2.1.0` + `ml_dtypes` for chia cosim
**is gone** — it lived under `/tmp` and nothing matching it survives. The only
cocotb on the host now is **2.0.1** in the `mininpu` conda env (python 3.11);
neither base 3.14 nor the `allo` env has cocotb or `ml_dtypes`. Rebuilding that
venv is a prerequisite for `allo/backend/rtl/sim/` on `chia-codesign`.

## Golden test for dataflow simulator

```bash
# `conda run` does not source the activate scripts, so export the env first.
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
python tests/dataflow/test_df_unit.py
python tests/dataflow/test_region_stateful.py
```

## TinyTPU-isa (the one accelerator design on `main`)

`examples/accelerator/tinytpu_vitis/`. From a clean checkout, one command
builds the checkout's bindings, runs the functional gates, runs cosim, and
checks the published cycle counts (252/383/591/667/919):

```bash
examples/accelerator/tinytpu_vitis/reproduce.sh            # ~6 min; --no-cosim: ~1 min
```

`bench_isa.py` / `cosim.py` (default TB) are the **performance** setup
(Gemmini's [-4, 4] operands) and miss real bugs; `stress_isa.py` and
`TPU_TB=stress python cosim.py` are the **correctness** gates. Run
`stress_isa.py` (~10 s) after any change to `microarch_isa.py`, and
`mutate.py` after any change to the harness. `assemble()` rejects programs that
read `ar`/`vr` before writing them (the arrays are not cleared by hardware).

## Project state

Live state is judged from git/GitHub/notes, not a checked-in status file:
`git branch -vv`, `gh pr list -R cornell-zhang/allo`,
`gh issue list -R sunwookim028/allo`, and `notes/`.

- Living feature map (fork vs upstream): the pinned fork issue
  https://github.com/sunwookim028/allo/issues/13. It is the single living
  picture of what the fork carries vs upstream; project state is otherwise
  judged from git/GitHub, not checked-in `.md` snapshots.
- Fork-local file inventory: fork issue #5
  (https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476).
- `notes/MAINTENANCE_CHECKLIST.md`: the upstream-merge procedure.

## Notes from AGENTS.md

See `AGENTS.md` for build instructions, testing, and code style guidelines.
