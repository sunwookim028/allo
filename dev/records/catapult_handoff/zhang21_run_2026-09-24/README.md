# zhang-21 run of the `pc_int32_systemc` handoff — 2026-09-24

Input: `../pc_int32_systemc/` at `main` `e42ba23`, copied unmodified to local `/scratch`.
Catapult Ultra Synthesis 2024.2/1130128, feature `CatapultUltra` checked out (LIC-14).
Input and output checksums are in `SHA256`.

## Verdict: synthesis is clean, but the pre-written criterion does not pass as worded

| # | Criterion | Result |
|---|---|---|
| 1 | `catapult.log` has `LIC-14`, exit 0 | **exit 0; LIC-14 is in stdout, not in `catapult.log`** — see below |
| 2 | no `^Error`, and no `SCHD-67`/`SCHD-30`/`HIER-47`/`ASSERT-1`/`CIN-` | **no error line of any kind.** `CIN-` does appear, on Info lines (CIN-1, CIN-52, …) and 17 CIN-124 warnings, so the literal regex matches. Read as "no *error* with these codes", it holds |
| 3 | `rtl.v` non-empty, contains `module top` | **pass** (30,200 bytes) |
| 4 | `cycle.rpt`: finite Latency and Throughput for `top` | **Throughput 1, Latency −1** — see below |

Numbers for `top` (`cycle.rpt`, `rtl.rpt`, 2.0 ns clock, nangate-45nm_beh):

| | |
|---|---|
| Throughput | 1 (every process) |
| Latency | −1 for `producer_0/run`, `consumer_0/run` and Design Total; 1 for the FIFO `Seq` |
| Reset length | 17 cycles (the 16-trip reset-action loop) |
| Total Area Score, post-assignment | **1555.2** (score units). 1689.0 post-scheduling, 1721.0 post-DP&FSM |
| Of which Reg / DataPath / FSM | 1207.1 (78 %) / 1512.1 / 43.1 |
| Critical path | 1.735 ns, slack +0.265 ns |
| Wall time | 33 s |

**Criterion 1 cannot hold with the `RUNME.md` command as written.** Catapult itself
writes a `catapult.log` in its working directory, which overwrites a `tee catapult.log`
of stdout. Catapult's own log does not include the LIC-13/LIC-14 lines; stdout does.
The run here captures stdout as `catapult_stdout.log` (LIC-14 present, exit 0). Either
tee to another name or check stdout.

**Criterion 4 as written fails.** Latency −1 is how Catapult reports a process whose
body is a non-terminating `while(1)` SC_THREAD. There is no first-output latency to
report, not an unbounded schedule. Throughput 1 is the finite figure. Whether −1
counts as a pass is for the criterion's author to decide. This record does not
reinterpret it.

**`RUNME.md` also sets `MGC_HOME=/opt/siemens/catapult/2024.2`.** The install root is
`/opt/siemens/catapult/2024.2/Mgc_home`, and the run used that path.

## csim

`./csim.sh` with Catapult's own g++ and libsystemc, then `diff output0.data
golden_output0.data`: **CSIM PASS** (`B = 1..8`).

## Negative control (directives stripped)

`-IO_MODE super` and `-SPECULATE true` removed: exit 0, no errors, the same cycle
table, area score 1555.2 and slack 0.265. `rtl.v` differs from the main run only in
the generated-at timestamp comment. **The two directives are inert on this design**,
so this control does not show that the criterion can go red. It was informative, as
`RUNME.md` anticipated.
