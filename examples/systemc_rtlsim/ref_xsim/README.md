# Reference xsim runs, for comparing against another simulator

Two runs of the **same `pe_wire` netlist**, kept so a run on a different
simulator (Xcelium, Questa) can be compared against a known-good and a
known-bad result rather than against a claim in a note. Trimmed to the logs;
`xsim.dir` (the compiled snapshot, ~350 KB per run) is dropped, and absolute
scratch paths are rewritten to `<RUNDIR>`.

Produced by `../run_mulacc.sh` under xsim v2023.2 on ace-01, 2026-09-18.

| | `fail_default/` | `pass_ACC_RST_DELAY3/` |
| --- | --- | --- |
| invocation | `pe_wire` | `pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3` |
| result | FAIL, 8/8 wrong | PASS |
| **cycles** | **12** | **22** |

Golden is `2 10 28 60 110 182 280 408`. The failing run gives
`0 0 0 2 2 10 10 28`.

## What to compare, in priority order

1. **The output pattern.** `0 0 0 2 2 10 10 28` is not noise -- it is the
   golden sequence shifted and repeated, which is what a consumer sampling an
   unhandshaked edge too early and too often produces. If another simulator
   gives this exact pattern, it is reproducing the same defect and not merely
   failing.
2. **The cycle count, which is the sharper tell.** The broken run finishes in
   **12 cycles and the correct one takes 22** -- *the failure is faster*. That
   is the free-running signature: nothing throttles `acc`, so it runs its whole
   loop and finishes early. A simulator that reproduces the wrong values but
   takes ~22 cycles is failing for some other reason and the diagnosis in
   `../README.md` does not transfer.
3. **The positive control.** `ACC_RST_DELAY=3` passing on identical RTL is what
   proves the wiring and the arithmetic are correct and only the lockstep is
   missing. If that does not pass on the other simulator, suspect the port of
   the harness before suspecting the design -- and check the fault injections,
   which must still go red.

Both runs reach `$finish` at `tb_mulacc.v` line 174 and capture 8/8 elements,
so neither is a hang or a truncated run; the failing one is a complete run with
wrong data.
