# zhang-21 run of the SystemC harness — 2026-09-24

The first run of `tests/systemc/` past emission after it was reorganised, on the
host that has the licences. Input: `origin/main` at `07a6e7c`, checked out on local
disk (`/scratch`), with no Python and no `allo` import. Every input file's checksum
is in `INPUTS.sha256`.

| Tool | Version |
|---|---|
| Catapult | 2024.2/1130128 (Production Release) |
| Xcelium `xrun` | 24.03-s005 |
| SystemC (Catapult-bundled) | 2.3.3-Accellera, g++ from `$MGC_HOME/bin` |

## Verdicts

| Check | Result | Log |
|---|---|---|
| `synth_compute.tcl` on archived `stream_boundary.cpp` | analyze → extract clean, `concat_sim_rtl.v` written, 29 s | `catapult_synth_compute.log` |
| `synth_source.tcl`, same input | analyze → extract clean, 29 s | `catapult_synth_source.log` |
| `rtlsim/REPRO.sh`, `RUNNER=run_mulacc_xrun.sh` | matches `../../rtlsim/results.txt` on every case it runs: Stream and Channel PASS, Wire FAIL at all 18 pacings, LOCKSTEP PASS only at delay 3 and 4, all four breakage cases go red. 22 s | `rtlsim_REPRO_xrun.log` |
| `guard_experiment/` netlists, the four commands in `rtlsim/README.md` | Stream, Channel PASS and go red under `BREAK_DATA`; Wire FAIL on both `rtl_base` and `rtl_guard` | `rtlsim_guard_experiment_xrun.log` |
| EVA reference `examples/eva/generated/csim.sh` | compiles and runs, 13 s, and **every output is 0** | `eva_reference_csim.log` |

REPRO needs the original `pe_*` netlists, which the checkout does not include. They
came from `choonsik1/allo` at `0eff4888:agents/noc/rtl/<design>/rtl.v`, not
`choonsik1/SystemC-emitter` as `rtlsim/README.md` says (that repository is not
reachable). `pe_wire` has the same hash at `779e4350^`.

## What these runs do not show

- **`LOCKSTEP` on the `guard_experiment/` netlists does not elaborate.** `tb_mulacc.v:87`
  pokes `u_mul.mul_0_run_inst.v8_and_cse`, a net name that exists only in the original
  netlists. That is a mismatch between the testbench and the netlists, and REPRO never
  runs that combination. The `guard_experiment/` rows overwrote each other's work
  directory, so the `pe_wire` rows in that log come from `rtl_base`. The console run
  gave the same verdicts for `rtl_guard`.
- **The EVA reference csim cannot separate a fix from a regression.** The committed
  `kernel.cpp` is the `build_eva_systemc.py` emission at NSTEP=55, and `*.data` are
  gitignored, so it runs with no inputs. The EVA README also says the runtime-prime
  variant does not drain at 55. The meaningful A/B is `cosim_eva_systemc.py`
  (NSTEP=215, ramp golden). It must be emitted twice, with the old and the new
  emitter, and the whole `generated/` committed with its `input*.data` (`git add -f`)
  and `output*.data` for each run. The csim is then about 15 s per side here.
- **`cosim_tb_producer_consumer.v` and `cosim_tb_mem_port_reverse.v` were not run.**
  They need RTL from `df.build(..., mode="csyn")`, which requires an `allo` import, and
  no such RTL is committed.

---

*Addendum, same day, from the host without licences:* the three items above that
asked for work — the netlist provenance, the `LOCKSTEP` gap, and the NSTEP=215
EVA pair — are answered in `../../README.md` ("The emission finding, settled" and
"The EVA reference, and what the A/B pair had to be") and in
`../../eva_nstep215_ab_2026-09-24/`. The run record itself is left as it was
written.
