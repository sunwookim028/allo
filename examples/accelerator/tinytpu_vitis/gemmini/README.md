# The Gemmini baseline, reproducibly

Every Gemmini number in `../COMPARISON.md` came from a Chipyard tree whose
changes were **never committed anywhere**: four uncommitted diffs across three
nested repositories, plus a benchmark that was untracked even inside its own
submodule. Captured 2026-09-18, before migrating off this host.

## Pins

| repo | commit |
| --- | --- |
| `chipyard` | `e0207441` |
| `generators/gemmini` | `25809f7` |
| `.../software/gemmini-rocc-tests` | `1a1a1c6` |

## What is here

| file | applies to | why it is needed |
| --- | --- | --- |
| `gemmini_CustomConfigs.patch` | `gemmini` | `int8Dim4Config` -- `defaultConfig` with `meshRows/Columns = 4`. The matched baseline itself. |
| `chipyard_RoCCAcceleratorConfigs.patch` | `chipyard` | `Int8Dim4GemminiRocketConfig`, the name `COMPARISON.md` tells you to pass as `make CONFIG=`. Without it the documented repro fails on its first command. |
| `allo_cmp.c` | `gemmini-rocc-tests/bareMetalC/` | the entire cycle benchmark. **Was untracked**, so `git stash` in that submodule would have deleted it. |
| `roccTests_gemmini_h.patch` | same | the `GEMMINI_ACC_SHR` macro. Without it any FP config gives **318 compile errors** and the whole suite fails to build. |
| `roccTests_gemmini_params_h.patch` | same | `DIM` 16 -> 4, `BANK_ROWS`, `ACC_ROWS`, `ACC_READ_FULL_WIDTH`. |
| `roccTests_Makefile.patch` | same | builds `allo_cmp`. |

## Applying

    cd ~/chipyard && git apply <this dir>/chipyard_RoCCAcceleratorConfigs.patch
    cd generators/gemmini && git apply <this dir>/gemmini_CustomConfigs.patch
    cd software/gemmini-rocc-tests
    git apply <this dir>/roccTests_gemmini_h.patch
    git apply <this dir>/roccTests_gemmini_params_h.patch
    git apply <this dir>/roccTests_Makefile.patch
    cp <this dir>/allo_cmp.c bareMetalC/

Then build the RTL (`make CONFIG=Int8Dim4GemminiRocketConfig` under
`sims/verilator`) and run `allo_cmp-baremetal`.

## Things that cost time to learn and are not visible in the sources

* **The stock `matmul` / `matmul_ws` tests print no cycle counts** -- their
  `read_cycles()` calls are commented out upstream. `allo_cmp.c` exists because
  of this; do not expect to get numbers from the shipped benchmarks.
* **One Verilator run takes about 22 minutes** at ~8.6 us/s simulated. Budget
  accordingly; the five-shape sweep is not interactive.
* **Stale binaries silently produce a wrong comparison.** The 54 binaries found
  in `build/` were int8 artifacts from an earlier elaboration and would have
  been run against a differently-configured RTL without any error. Rebuild the
  benchmark whenever the config changes, and check `GEMMINI DIM=` in the boot
  banner -- `gemmini_int8_dim4.log` opens with `GEMMINI DIM=4 elem_t_bytes=1`,
  which is the line that proves which hardware produced the numbers.
* `gemmini_counter.h` exposes 8 hardware counters that were never read. Roughly
  an hour of work if a future comparison wants per-unit attribution rather than
  total cycles.

## Fairness note

Ours is the accelerator alone under Vitis `cosim`; Gemmini's `rdcycle` figure
includes RoCC dispatch from Rocket and its own tiling loop. That favours us
slightly, and `COMPARISON.md` states it rather than correcting for it.
