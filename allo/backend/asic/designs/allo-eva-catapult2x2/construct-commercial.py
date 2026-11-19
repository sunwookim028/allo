"""Blocking EVA design through the SystemC/Catapult ASIC flow."""

import os
from pathlib import Path

from mflowgen.components import Graph, Node


SYSTEMC_OPTIONS = (
    "device=nangate-45nm_beh,"
    "generalize_pid_specializations=true,"
    "pid_generalization_policy=identity_only"
)


def _runtime_path(env_var, default, description, *, executable=False):
    """Resolve and validate a host dependency used while running the flow."""
    path = Path(os.environ.get(env_var, default)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} does not exist: {path} (override with {env_var})"
        )
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"{description} is not executable: {path}")
    return str(path)


def construct():
    """Build the complete two-stage graph for the local EVA source."""
    graph = Graph()
    this_dir = Path(__file__).resolve().parent
    asic_dir = this_dir.parent.parent
    nodes_dir = asic_dir / "nodes"

    allo_python = _runtime_path(
        "ALLO_PYTHON",
        Path.home() / ".conda/envs/allo-sunwoo/bin/python",
        "Allo Python interpreter",
        executable=True,
    )
    allo_setup_script = _runtime_path(
        "ALLO_SETUP_SCRIPT",
        "/work/shared/common/allo/setup-llvm-main.sh",
        "Allo LLVM setup script",
    )

    graph.sys_path.append(str(asic_dir / "adks"))
    graph.set_adk("freepdk-45nm")
    adk = graph.get_adk_node()

    flow_utilities = Node(str(nodes_dir / "asic-flow-utilities"))
    allo_build = Node(str(nodes_dir / "allo-asic-compilation"))
    testbench_generation = Node(
        str(nodes_dir / "allo-testbench-generation")
    )
    rtl_sim = Node(str(nodes_dir / "allo-rtl-sim"))
    ffgl_sim = Node(str(nodes_dir / "allo-ffgl-sim"))
    bagl_sim = Node(str(nodes_dir / "allo-bagl-sim"))
    power_analysis = Node(str(nodes_dir / "synopsys-pt-power"))
    macro_activity = Node(
        str(nodes_dir / "allo-macro-activity-extraction")
    )
    rtl_normalize = Node(str(nodes_dir / "sv2v-rtl-allo"))

    macro_plan = Node(str(nodes_dir / "allo-asic-macro-plan"))
    macro_synthesis = Node(
        str(nodes_dir / "commercial-batch-synthesis")
    )
    macro_pnr = Node(str(nodes_dir / "commercial-macro-pnr"))
    macro_physical_verify = Node(
        str(nodes_dir / "commercial-batch-physical-verify")
    )
    macro_signoff = Node(str(nodes_dir / "commercial-batch-signoff"))
    macro_publish = Node(str(nodes_dir / "commercial-macro-publish"))
    macro_power = Node(str(nodes_dir / "commercial-batch-macro-power"))

    assembly_plan = Node(str(nodes_dir / "allo-asic-assembly-plan"))
    rtl_assembly = Node(str(nodes_dir / "allo-asic-rtl-assembly"))
    full_chip_synthesis = Node(
        str(nodes_dir / "commercial-full-chip-synthesis")
    )
    physical_intent = Node(
        str(nodes_dir / "allo-asic-physical-intent")
    )
    full_chip_pnr = Node(str(nodes_dir / "commercial-full-chip-pnr"))
    full_chip_gdsmerge = Node(
        str(nodes_dir / "commercial-full-chip-gdsmerge")
    )
    full_chip_drc = Node(str(nodes_dir / "commercial-full-chip-drc"))
    full_chip_lvs = Node(str(nodes_dir / "commercial-full-chip-lvs"))
    flow_summary = Node(str(nodes_dir / "allo-asic-flow-summary"))
    flow_finalize = Node(str(nodes_dir / "asic-flow-finalize"))

    nodes = [
        flow_utilities,
        allo_build,
        testbench_generation,
        rtl_sim,
        ffgl_sim,
        bagl_sim,
        power_analysis,
        macro_activity,
        rtl_normalize,
        macro_plan,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        macro_publish,
        macro_power,
        assembly_plan,
        rtl_assembly,
        full_chip_synthesis,
        physical_intent,
        full_chip_pnr,
        full_chip_gdsmerge,
        full_chip_drc,
        full_chip_lvs,
        flow_summary,
        flow_finalize,
    ]
    for node in nodes:
        graph.add_node(node)

    utility_consumers = [
        allo_build,
        rtl_sim,
        ffgl_sim,
        bagl_sim,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        macro_power,
        full_chip_lvs,
    ]
    for node in utility_consumers:
        graph.connect_by_name(flow_utilities, node)

    # Compile once, then share the manifest, workload, and normalized RTL.
    graph.connect_by_name(allo_build, testbench_generation)
    graph.connect_by_name(allo_build, rtl_normalize)
    graph.connect_by_name(testbench_generation, rtl_sim)
    graph.connect_by_name(rtl_normalize, rtl_sim)

    # Stage 1: plan, harden, verify, characterize, and publish selected blocks.
    graph.connect_by_name(rtl_sim, macro_plan)
    for artifact in [
        "asic-manifest-final.json",
        "asic-manifest-final.tcl",
        "build-metadata.json",
    ]:
        graph.connect(rtl_normalize.o(artifact), macro_plan.i(artifact))
    graph.connect_by_name(macro_plan, macro_synthesis)
    graph.connect_by_name(macro_synthesis, macro_pnr)
    graph.connect_by_name(macro_pnr, macro_physical_verify)
    graph.connect_by_name(macro_physical_verify, macro_signoff)
    graph.connect_by_name(macro_signoff, macro_publish)
    graph.connect_by_name(rtl_normalize, macro_publish)
    for node in [
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
    ]:
        graph.connect_by_name(adk, node)

    # Stage 2: substitute the published views and implement the full chip.
    graph.connect_by_name(rtl_normalize, assembly_plan)
    graph.connect_by_name(macro_publish, assembly_plan)
    graph.connect_by_name(rtl_normalize, rtl_assembly)
    graph.connect_by_name(assembly_plan, rtl_assembly)
    graph.connect_by_name(macro_publish, rtl_assembly)
    graph.connect_by_name(rtl_assembly, full_chip_synthesis)
    graph.connect_by_name(macro_publish, full_chip_synthesis)
    graph.connect_by_name(adk, full_chip_synthesis)
    graph.connect_by_name(assembly_plan, physical_intent)
    graph.connect_by_name(rtl_assembly, physical_intent)
    graph.connect_by_name(macro_publish, physical_intent)
    graph.connect_by_name(full_chip_synthesis, physical_intent)
    graph.connect_by_name(rtl_assembly, full_chip_pnr)
    graph.connect_by_name(macro_publish, full_chip_pnr)
    graph.connect_by_name(full_chip_synthesis, full_chip_pnr)
    graph.connect_by_name(physical_intent, full_chip_pnr)
    graph.connect_by_name(adk, full_chip_pnr)

    # Full-chip physical verification.
    graph.connect_by_name(full_chip_pnr, full_chip_gdsmerge)
    graph.connect_by_name(macro_publish, full_chip_gdsmerge)
    graph.connect_by_name(adk, full_chip_gdsmerge)
    graph.connect_by_name(full_chip_gdsmerge, full_chip_drc)
    graph.connect_by_name(adk, full_chip_drc)
    graph.connect_by_name(full_chip_gdsmerge, full_chip_lvs)
    graph.connect_by_name(full_chip_pnr, full_chip_lvs)
    graph.connect_by_name(macro_publish, full_chip_lvs)
    graph.connect_by_name(adk, full_chip_lvs)

    # Workload-driven simulation and power analysis.
    graph.connect_by_name(testbench_generation, ffgl_sim)
    graph.connect_by_name(full_chip_synthesis, ffgl_sim)
    graph.connect_by_name(macro_publish, ffgl_sim)
    graph.connect_by_name(adk, ffgl_sim)

    graph.connect_by_name(testbench_generation, bagl_sim)
    graph.connect_by_name(rtl_assembly, bagl_sim)
    graph.connect_by_name(full_chip_pnr, bagl_sim)
    graph.connect_by_name(macro_publish, bagl_sim)
    graph.connect_by_name(adk, bagl_sim)

    graph.connect_by_name(full_chip_pnr, power_analysis)
    graph.connect_by_name(bagl_sim, power_analysis)
    graph.connect_by_name(macro_publish, power_analysis)
    graph.connect_by_name(adk, power_analysis)

    graph.connect_by_name(bagl_sim, macro_activity)
    graph.connect_by_name(assembly_plan, macro_activity)
    graph.connect_by_name(macro_publish, macro_activity)
    graph.connect_by_name(macro_activity, macro_power)
    graph.connect_by_name(macro_publish, macro_power)
    graph.connect(power_analysis.o("power.rpt"), macro_power.i("power.rpt"))
    graph.connect_by_name(adk, macro_power)

    # Collect every terminal result before final cleanup.
    graph.connect_by_name(macro_publish, flow_summary)
    for node in [
        allo_build,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        full_chip_synthesis,
        full_chip_pnr,
        full_chip_gdsmerge,
        full_chip_drc,
        full_chip_lvs,
    ]:
        graph.connect_by_name(node, flow_summary)
    graph.connect_by_name(physical_intent, flow_summary)
    graph.connect(
        ffgl_sim.o("simulation-report.json"),
        flow_summary.i("ffgl-simulation-report.json"),
    )
    graph.connect(
        bagl_sim.o("simulation-report.json"),
        flow_summary.i("bagl-simulation-report.json"),
    )
    graph.connect_by_name(macro_activity, flow_summary)
    graph.connect_by_name(macro_power, flow_summary)
    graph.connect_by_name(flow_summary, flow_finalize)

    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": str(
                this_dir / "eva_sb_syscredit_rtprime_skid_leanalu.py"
            ),
            "allo_entrypoint": "build",
            "backend": "systemc",
            "build_mode": "csyn",
            "backend_options": SYSTEMC_OPTIONS,
            "clock_period": 10.0,
            "macro_clock_period": 8.0,
            "python_bin": allo_python,
            "allo_setup_script": allo_setup_script,
            "top_module": "top",
            "design_name": "top",
            "report_design_name": "allo-eva-blocking_2x2_fp16_FIFO8_systemc",
            "adk": "freepdk-45nm",
            "adk_view": "view-standard",
            "min_macro_reuse": 2,
            "harden_repeated_hls_submodules": False,
            "bypass_macro_generation": False,
            "fold_fifos_into_macro": True,
            "enable_kernel_rotation": True,
            "interleave_macros": True,
            "enable_macro_channel_soft_blockages": True,
            "macro_channel_soft_blockage_fraction": 0.3,
            "peripheral_placement_sides": "all",
            "enable_gui": False,
            "nthreads": 4,
            "local_cpus": 4,
            "postroute_max_local_cpus": 4,
            "antenna_check_policy": "report",
            "drc_check_policy": "report",
            "macro_hold_target_slack": 0.050,
            "macro_setup_target_slack": 0.050,
            "hold_target_slack": 0.020,
            "hold_optimization_target_slack": 0.080,
            "stop_after_step": "none",
            "core_density_target": 0.70,
            "macro_core_density_target": 0.70,
            "use_separate_macro_density_target": False,
            "highest_macro_routing_layer": 5,
            "drc_nthreads": 4,
            "lvs_nthreads": 4,
            "flatten_effort": 1,
            "macro_separation_x": 10.0,
            "macro_separation_y": 10.0,
            "macro_halo": 0.5,
            "useful_skew": False,
            "ccopt_target_max_transition": 0.120,
            "clock_pin_depth_width_multiplier": 5.0,
            "macro_clock_root_buffer_cell": "CLKBUF_X1",
            "macro_clock_root_buffer_input_pin": "A",
            "macro_clock_root_buffer_output_pin": "Z",
            "kernel_separation_x": 10.0,
            "kernel_separation_y": 10.0,
            "macro_edge_keepout": 10.0,
            "sram_separation": 20.0,
            "google_sheets_enabled": False,
            "google_sheets_required": False,
            "allo_testbench_enabled": True,
            "allo_testbench_workload_factory": "testbench_workload",
            "allo_testbench_top_function": "top",
            "testbench_name": "allo_generated_testbench",
            "dut_name": "dut",
            "sdf_corner": "typ",
            "bagl_failure_policy": "report",
            "sdf_warning_policy": "report",
            "sdf_unmatched_timingcheck_policy": "report",
            "sdf_unmatched_iopath_policy": "report",
            "sdf_uphier_interconnect_policy": "report",
            "bagl_input_delay_ns": 0.025,
            "bagl_output_delay_ns": 0.025,
            "bagl_num_reset_cycles": 8,
            "waveform": True,
            "saif_instance": "allo_generated_testbench/dut",
            "analysis_mode": "averaged",
            "activity_source": "bagl_vcd",
            "lib_op_condition": "undefined",
            "macro_power_max_workers": 4,
            "cleanup_enabled": True,
        }
    )
    return graph


if __name__ == "__main__":
    construct()
