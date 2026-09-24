# zhang-21 run of `ppa_mac16` — 2026-09-24

The first end-to-end run of the repaired `mode="ppa"`. Input `../` at `main` `b1e9b81`,
`run.tcl` unmodified, run from a local `/scratch` copy. Catapult 2024.2/1130128 and
Xcelium 24.03-s005; licences `MGLS_LICENSE_FILE=1717@…` and `CDS_LIC_FILE=5280@…`.
The one deviation from `RUNME.md`: stdout went to local scratch rather than `$HOME`,
which is on slow NFS here, and that path was passed to `check_ppa.py` as its second
argument.

## Verdict: `check_ppa.py` PASS (exit 0)

`check_ppa.out` has the full output. Catapult exited 0 after 56 s. `use_ccs_block`
was **not** needed: SCVerify wrapped the `#pragma hls_design top` DUT directly.

| | |
|---|---|
| Power | **272.69 µW** = 252.19 dynamic + 20.50 static |
| Annotation | 100.00 % flop outputs, 100.00 % user nets |
| Testbench | `MAC16 TB errors=0`; `Simulation PASSED @ 18008500 ps` |
| `cycle.rpt` | latency 16, throughput 18 |
| `rtl.rpt` | total area 1180.529 score units |
| Instances | `mac16_core_inst` 272.09 µW; its FSM 4.07 µW |

It differs from the hand-written `mac.cpp` run (248.47 µW), as `RUNME.md` anticipates:
the stimulus shape is the same, but the port interface and netlist are not.

## The criterion can go red

`check_ppa.py` was run against three doctored copies of this run's outputs, without
re-running Catapult:

| Doctored input | Checker output |
|---|---|
| log says `MAC16 TB errors=3` | FAIL — `3. testbench self-check missing or non-zero` |
| `power.rpt` design-level Dynamic row zeroed | FAIL — `1. dynamic power is 0.0 uW` |
| `power.rpt` annotation row 0.00 % | FAIL — `2. only 0.00% of flop outputs annotated` |
