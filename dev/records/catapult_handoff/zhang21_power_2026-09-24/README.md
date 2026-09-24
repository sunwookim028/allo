# Catapult power on zhang-21 — 2026-09-24

Can the Catapult flow on this host produce power, not just area? **Yes, but not
through `mode="ppa"` as it stands.** It needs extra flow steps, a C++ testbench, and
Xcelium for the switching simulation.

## The measured number

A 16-tap int8 multiply-accumulate (`mac.cpp`), driven by 200 transactions from
`mac_tb.cpp` (TB errors = 0). nangate-45nm_beh at a 5 ns clock. Catapult 2024.2/1130128
with PowerPro, `report_pre_pwropt_Verilog`. Library `nangate45nm_nldm.lib`, operating
condition `typical` (process 1, 1.1 V, 25 °C).

| µW | Register | Combinational | Clock network | Total |
|---|---|---|---|---|
| Static | 4.42 | 14.27 | 0.10 | 18.80 |
| Dynamic | 61.78 | 147.96 | 19.93 | 229.67 |
| **Total** | 66.20 | 162.23 | 20.03 | **248.47** |

**Control:** the same design with every input 0 (`zero_activity/`) gives dynamic
83.80 µW. Combinational falls from 147.96 to 15.03 µW, and the clock network is
unchanged at 19.93 µW. The number follows the workload's activity.

## The three questions

1. **Is `power.rpt` produced, with real numbers?** Yes, once the flow runs
   `go switching` and `flow run /PowerAnalysis/report_pre_pwropt_Verilog` after
   `go extract`. The tcl that `mode="ppa"` emits stops at `go extract`, identical to
   `csyn`, so it never produces one. It also produces no `area.rpt`. Area is only in
   `rtl.rpt`.
2. **Licence.** PowerPro checks out
   `PProBase PProAnalysis PProCGopt PProWriteRTL PProPAWorker` from the same server
   (`1717@en-license-05`), all granted with nothing extra set. Nothing needs acquiring.
3. **Activity.** From simulation, not default toggle rates. `go switching` runs
   SCVerify: the C++ testbench drives the pre-power RTL in Xcelium, and the VCD is
   converted to SAIF. The SAIF annotated 100 % of primary inputs and 100 % of flop
   outputs. Internal combinational nets are propagated by PowerPro, not simulated.
   The number is **workload-specific: it is only as representative as the C++
   testbench.** The estimate is pre-layout RTL, uses a wireload model, and the clock
   tree is PowerPro's model. `LowPower/NO_VECTORS` exists for a vectorless estimate;
   it was not used.

## What it takes on this host

- **Simulator:** Catapult defaults to QuestaSim. `/opt/siemens/Questa/2024.2` is only
  Questa VIP and has no `vsim`, so that fails with "No rule to make target
  …/modelsim.ini". Use Xcelium:
  `/SCVerify/USE_NCSIM true`, `/NCSim/NC_ROOT /opt/cadence/XCELIUM2403` (the option is
  `NC_ROOT`; `Path` does not exist), plus `CDS_LIC_FILE`.
- **Activity format:** `/LowPower/SWITCHING_ACTIVITY_TYPE saif`. The default FSDB
  needs `NOVAS_INST_DIR`, which is unset.
- **Testbench:** a design with no C++ testbench cannot get activity. The SystemC
  handoff `pc_int32_systemc` and the archived `stream_boundary` have none that
  SCVerify can drive. On `stream_boundary`'s `compute_0` the switching step failed for
  that reason.
- **Parser:** `allo/backend/catapult.py` looks for `Total Power\s*:` and
  `area.rpt`'s `Total Area`. The real report is a table whose row is
  `Total … 248.47`, so `mode="ppa"` would print `N/A` for both even after a
  successful power run.
- The example Catapult ships for this,
  `shared/examples/docs/directives/pwr_switching_file/run.tcl`, references sources
  missing from this install, so it cannot run as shipped.

Run: `catapult -shell -f run.tcl` in a copy of this directory, with `MGC_HOME`,
`MGLS_LICENSE_FILE` and `CDS_LIC_FILE` exported. 81 s.
