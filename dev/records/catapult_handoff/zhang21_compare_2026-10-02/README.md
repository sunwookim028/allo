# zhang-21: Catapult csim vs the stand-in, and first csyn of TinyTPU's SystemC emission

Requested by ace01 after M0 (`c3de83f3`). Host zhang-21.ece.cornell.edu, RHEL 8.10,
glibc 2.28, 64 cores. Run 2026-10-02 (UTC) at checkout `c3de83f3` (origin/main,
fast-forwarded from `d7377398`). Tool inventory: `../zhang21_inventory_2026-10-01.md`.

## Summary

- **(a) Catapult's own csim libraries agree exactly with ace01's stand-in.** TinyTPU's 3
  stress cases give `wrong=0/16` with `clobbered_outside` 4057 / 4057 / 4065, then
  `SYSTEMC CSIM OK`. EVA gives `out_e = [1..6]`, `PASS (bit-exact)`. Every number matches ace01's.
- **The compiler must be Catapult's g++ 10.3.** System g++ 8.5 fails to link
  `libsystemc.so` from `gcc-10.3.0-64` (`GLIBCXX_3.4.26`).
- **(b) Catapult 2024.2 synthesizes TinyTPU's `target="systemc"` emission through
  `go extract`, with 0 errors in 332 s.** The result:
  - Area score: 320031.8 post-assignment, 75% of it registers.
  - Timing: slack **-0.072 ns** at the 2.0 ns clock (nangate-45nm_beh), on the sequencer.
  - Reset: **10,747,772 reset cycles** in total, from the array-clearing `for` loops in
    each thread's reset action.

## Environment

`env.sh` (sourced for every run below):

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
module load catapult-2024          # MGC_HOME, MGLS/SALT/CDS licence vars, PREPENDS $MGC_HOME/bin
export PATH=/opt/cadence/XCELIUM2403/tools.lnx86/bin:$PATH   # module points at a non-existent XCELIUM2409
unset LD_PRELOAD
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
export SYSTEMC_HOME=$MGC_HOME/shared
export ALLO_CXX_EXTRA="-DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM -L$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64 -Wl,-rpath,$MGC_HOME/lib"
```

Versions:
- Catapult Ultra Synthesis 2024.2/1130128, from `MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home`.
- SystemC 2.3.3-Accellera, from `$MGC_HOME/shared/lib/Linux/gcc-10.3.0-64`.
- ac_types 4.9.0, ac_simutils 1.6.0, Connections 2.2.0.
- `g++ (Calypto) 10.3.0`.
- Python is the `allo` conda env.

`ALLO_CXX_EXTRA` was used exactly as ace01 proposed; the link needed no change.

## Setup check

`examples/tinytpu/reproduce.sh --no-cosim` rebuilt this checkout's bindings and
passed in 169 s. Its last lines are in `reproduce_no_cosim.tail.txt`:
`ISA OK`, `UNITS OK`, `ALL EXACT`, `STRESS OK: 492/492`, `ACT GATE OK: 12/12`, and
`REPRODUCED (functional only)`.

Gotcha: invoked through the `/home/sk3463/work/allo` symlink, the script stops with
`allo resolves to /work/shared/users/phd/sk3463/allo/allo/__init__.py, not this
checkout (/home/sk3463/work/allo)`. Its `ROOT` is the logical path, while Python
resolves the physical one. Run it from `/work/shared/users/phd/sk3463/allo`
(`cd -P`).

## (a) Catapult csim libraries vs the stand-in

| Run | Command | g++ | Wall | Result |
|---|---|---|---|---|
| a1 | `python examples/tinytpu/systemc_csim.py 3 --project <scratch>/tt.prj` | Catapult 10.3.0 | 51 s | 3 x `wrong=0/16`, clobbered 4057/4057/4065, `SYSTEMC CSIM OK` |
| a1' | same, with `/usr/bin` first on PATH | system 8.5.0 | 36 s | link failure (below) |
| a2 | `python examples/eva/cosim_eva_systemc.py` | Catapult 10.3.0 | 15 s | `[1..6]`, `COSIM RESULT : PASS (bit-exact)` |

`module load catapult-2024` prepends `$MGC_HOME/bin`, so the plain `g++` on PATH is
already Catapult's. a1' is the explicit system-g++ attempt. Verbatim, from `a1_g85.log`:

```
/opt/siemens/catapult/2024.2/Mgc_home/shared/lib/Linux/gcc-10.3.0-64/libsystemc.so: undefined reference to `std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream()@GLIBCXX_3.4.26'
collect2: error: ld returned 1 exit status
```

a1, verbatim:

```
CASE gemm 4x4x4 full seed=101 -> gemm 4x4x4 full seed=101: wrong=0/16 clobbered_outside=4057 7.8s
CASE gemm 4x4x4 full seed=101 flat -> gemm 4x4x4 full seed=101 flat: wrong=0/16 clobbered_outside=4057 7.3s
CASE gemm 4x4x4 corner seed=102 -> gemm 4x4x4 corner seed=102: wrong=0/16 clobbered_outside=4065 7.8s
SYSTEMC CSIM OK
```

a2, verbatim:

```
config: M=N=1  K=6  pc=15  NSTEP=215  PRIME_TOKENS=6
EXPECT ramp    : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SYSTEMC out_e  : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
COSIM RESULT   : PASS (bit-exact)
```

**No difference from ace01's stand-in numbers.**

Side finding: a2 rewrote `examples/eva/generated/{kernel.cpp,kernel.h}` with a
3724+/3735- diff against `c3de83f3`. The diff includes the emitter's new
`AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE AC_RND_CONV` prelude and a `_fbits(ac::bfloat16)`
overload, so **the checked-in generated EVA files are stale relative to the current
emitter**. The files were restored with `git checkout -- examples/eva/generated`; the
diff's SHA256 is in `SHA256SUMS.txt`.

## (b) First Catapult synthesis of TinyTPU's `target="systemc"` emission

`b_csyn.py` does the following:

```python
s = customize(tinytpu_isa); schedule(s)
mod = s.build(target="systemc", mode="csyn", project="<scratch>/tt_csyn.prj")
mod()
```

`build()` only writes the project. `mod()` launches Catapult the way hls.py does, as
`cd <prj>/build; catapult -shell -f <prj>/run.tcl`, from a build subdirectory.
It started at 2026-10-02T02:14:11Z and ended at 02:20:10Z. The build took 0.4 s,
Catapult took 331.8 s, and the process exited 0 with
`Catapult HLS synthesis completed successfully`.

`run.tcl` (generated, copied here) contains: c++11, `-DESIGN_HIERARCHY tinytpu_isa`,
a 2.0 ns `clk`, `-IO_MODE super`, `-SPECULATE true`, `nangate-45nm_beh`, and the
sequence `go analyze`, `go compile`, `go assembly`, `go extract`.

**Where it stops:** nowhere; every stage completes. Per-stage times for solution `tinytpu_isa.v1`:

| stage | s | stage | s |
|---|---|---|---|
| analyze | 13.75 | cluster | 19.18 |
| compile | 43.37 | architect | 8.79 |
| libraries | 0.89 | allocate | 4.62 |
| assembly | 8.00 | schedule | 91.97 |
| loops | 9.93 | dpfsm | 28.80 |
| memories | 10.37 | instance | 26.54 |
| | | extract | 24.93 |

Catapult ran on about one core throughout (92% CPU, 12 threads, 1.87 GB peak).

**Messages:** there were 0 errors and 304 warnings. The first 50 are in
`b_first50_warnings.txt`, and the full log is `b_csyn.log.gz`. By code:

| count | code | what |
|---|---|---|
| 286 | CIN-124 | "SystemC thread ... should be scheduled with 'iomode=fixed' since it writes to sc_signal ...". 210 come from `connections_fifo.h(267)` (vendor FIFO `Seq`: `full`, `head`, `buffer(i).dat`). The rest come from emitted `kernel.cpp` threads `run` writing the plain `sc_signal` `done`, two each per thread (e.g. `kernel.cpp(1055)`). |
| 5 | (none) | "Hierarchical design detected - Design analysis will be done at block level" |
| 4 each | MEM-87/99/100 | `connections_fifo.h(70)` `buffer.init_val` compacted from 64 to 32 or 8 bits |
| 1 | LIB-83 | `nangate-45nm_beh` was built by Library Builder 2025.1/1129627, newer than Catapult 2024.2/1130128 |

hls.py's comment ties CIN-124 to a degradation to `iomode=fixed` followed by SCHD-30
when Catapult's cwd holds the source. Here the run used a build subdir and **no SCHD-30
occurred**: CIN-124 is advisory, and `-IO_MODE super` stood.

**Reports** (`rtl.rpt`, `cycle.rpt`):
- Area score: 272183.2 post-scheduling, 336714.9 post-DP&FSM, and **320031.8**
  post-assignment. Registers are 239398.6 of that (75%).
- Timing: critical path 2.0721 ns, **slack -0.0721 ns** at 2.0 ns, with no clock
  uncertainty. It runs from `u0/sequencer_0:run:inst/reg(live_iv(0))` through
  `while:else#1:else:v143:mux1h#1` to `reg(while:else#1:else:f2(11:0).lpi#1.dfm)`.
- Design total: 1174 real operations, throughput 1, II 0, and **Reset Length 10,747,772**.
  Each thread's reset action contains a `for` loop clearing its arrays, e.g.
  `dma_st_0` 393222 cycles, `accu_0` 262148 and each `pe_i_j` 196611. Those loops,
  summed, are the design's reset latency. That is a property of the emission, not of
  Catapult.
- RTL: `rtl.v` (48,795 lines) and `concat_rtl.v` are not committed; their SHA256 is in
  `SHA256SUMS.txt`. A `scverify/` directory was generated, but no SCVerify or Xcelium
  run was attempted.

Not done, and not attempted: DC synthesis of the RTL, and SCVerify. No Allo source
was patched.

## Files

`env.sh`, `b_csyn.py` (inputs); `run.tcl` (generated); `a1_sysgxx.log`,
`a1_g85.log`, `a2_eva.log` (verbatim, ANSI stripped); `b_csyn.log.gz` (full Catapult
transcript); `b_first50_warnings.txt`; `rtl.rpt`, `cycle.rpt`;
`reproduce_no_cosim.tail.txt`; `SHA256SUMS.txt` (the inputs at `c3de83f3`, the emitted
`kernel.cpp`, `run.tcl`, `rtl.v`, `concat_rtl.v` and the EVA generated-file diff).
Paths under the session scratchpad are written as `<scratch>`.
