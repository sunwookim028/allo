# allo-asic

This directory contains the integrated hierarchical Allo flow, whose designs compile an Allo
  design, emit a backend-neutral ASIC manifest, harden reusable subblocks into macros, and
  assemble the full chip.

Shared implementation nodes live under `nodes/`; integrated examples live
under `designs/`. Generated mflowgen builds should be outside the importable
`allo/` package tree (for example in a sibling `build/` directory) so pytest and
helper scripts do not accidentally import a partially staged package. The Vitis
and SystemC/Catapult manifest contract is documented in
`docs/source/backends/asic_manifest.rst` and
`nodes/allo-asic-compilation/README.md`. The tutorial below gives a short introduction and walkthrough of the integrated Allo flow.

The maintained hierarchical nodes are direct children of `nodes/`. The older
flat RTL flow is retained only for legacy examples under
`nodes/legacy-flat-flow/`, and the ORFS nodes remain grouped under
`nodes/orfs/`. New Allo designs should use the hierarchical graph shown here,
not either legacy node set.

## Allo-ASIC Flow Background/Tutorial

Julian Bushlow, 10/2026

General info about the Allo-ASIC hierarchical flow. Assumes that reader knows parts of an ASIC flow and their role. A template for making new designs can be found in [`designs/template-design/`](designs/template-design/)

### Orchestration: mflowgen

File management can be a bottleneck of designer effort when making automatic ASIC flows, since the scripts need to know exactly where certain tools will dump their outputs so the next tool can use them. Manually specifying these paths can make sharing or modifying a flow tedious and brittle.

[mflowgen](https://mflowgen.readthedocs.io/en/latest/) is a modular flow organizer which neatly handles this file management. Each step can be packaged as a node in a flow graph, where the outputs of each step are assumed to be in the dedicated `outputs/` directory for each step's directory (and are also listed in the node's `configure.yml` file). If another node's `configure.yml` lists that output as one of its inputs, it will be copied via symlink into the node's `inputs/` directory and can be used from there.

Since ASIC flow steps can be accomplished using multiple procedures/tools and some steps are optional, a library of these nodes can be made and reused. This directory contains a hierarchical Allo <span>&rarr;</span> GDS flow which uses high-level design information from Allo designs to neatly partition and identify reuseable macros to speed up the design iteration process.

Note that a custom node can be defined just using a configure.yml file, so some designs build custom nodes in the design directory and call them inside their constructor files. This process requires user expertise though, so all needed nodes were made ahead of time for standardization.

### Making mflowgen Designs

Making a new mflowgen design involves setting up a constructor and an Allo
source file; the Allo-ASIC flow generates the testbenches automatically. The
structure below is for the maintained hierarchical commercial flow. The
archived flat and ORFS flows are not covered by this tutorial. Some sections
are required by mflowgen, while others are contracts used by the nodes in this
flow.

```
designs/MyDesign/
├── construct-commercial.py
├── allo-design.py
└── sram_manifest.yml       # only when SRAMs are used
```

#### SRAMs

An SRAM manifest must be used for SRAM generation using OpenRAM (note:  a different system will likely be used for the ASAP7 PDK, but a similar interface will be used). This manifest identifies each SRAM as used in the design with some basic information with other information being found in the [constructor file](#constructors). Note that this manifest is only required when using `sram_mode='generate'`

```
srams:
- name: sram_example
  word_size: 32
  num_words: 1024
	num_rw_ports: 2
	num_r_ports: 0
	num_w_ports: 0
	num_banks: 2
	words_per_row: 2
	write_size: 32
```

#### Allo Designs

The hierarchical flow requires designs to use the Allo dataflow programming constructs (see ['designs'] for examples).

The design file must also provide a build function with the interface below. The compilation node calls this function with the selected HLS backend and the configuration needed to emit the ASIC manifest. It is important to pass `target`, `mode`, and `configs` through without replacing them, they get replaced when compiling with the actual backend information specified in the [constructor file](#constructors)

```python
def build(project, target, mode, configs):
    module = df.build(
        top,
        target=target,
        mode=mode,
        project=Path(project).resolve(),
        wrap_io=False,
        configs=configs,
    )
    module()
```

Testbenches are generated automatically from a workload declared in the same design file. The design must provide a `testbench_workload()` function describing the top-level argument order, initial value of every argument, expected output arrays, comparison method, and timeout. Array shapes and NumPy dtypes must exactly match the corresponding Allo arguments; for example, an `Int(8)` array should use `np.int8`. The generated workload is reused for RTL, FFGL, and BAGL simulation as well as downstream power analysis.

```python
def testbench_workload():
    return {
        "top_function": "top",
        "call_signature": ["A", "B", "C"],
        "calls": [
            {
                "name": "example",
                "reset_before": True,
                "arguments": {
                    "A": input_a,
                    "B": input_b,
                    "C": initial_c,
                },
                "expected": {
                    "C": expected_c,
                },
                "comparison": {
                    "C": {"mode": "bit_exact"},
                },
            }
        ],
        "default_timeout_cycles": 10_000,
    }
```

Note that `reset_before` and `name` are both optional, and that `name` just controls which testbench name is printed in the simulation logs for readability.

#### Constructors

Constructor files are where most of the important parameters are set, including more physical information like clock period density targets, run settings like whether to generate SRAMs or how many cores to use, and logistical paths to any Allo/LLVM setup scripts and pre-generated SRAMs. The constructor file is also where nodes are actually connected into a graph to make a flow. For an example, look at [`designs/allo-example-systemc/construct-commercial.py`](designs/allo-example-systemc/construct-commercial.py).

The script takes the general form:
- define graph
- define list of parameters (to override defaults)
- instantiate nodes, connect to graph
- connect nodes together

The most commonly adjusted parameters and other paths to be set are grouped below. Most parameters already have defaults in their corresponding node's `configure.yml` file and do not have to be repeated in every constructor. Parameters are only included when a design needs to override those defaults.

```python
parameters = {
    #---------------------------------------------------------------------
    # Allo compilation
    #---------------------------------------------------------------------

    'construct_path': __file__,
    'allo_design_file': str(this_dir / 'allo_design.py'),
    'allo_entrypoint': 'build',
    'build_mode': 'csyn',

    'python_bin': allo_python,
    'allo_setup_script': allo_setup_script,

    #---------------------------------------------------------------------
    # HLS backend
    #---------------------------------------------------------------------

    'backend': 'vitis',
    'backend_options': 'device=u280',
```

`allo_design_file` identifies the Python file containing the Allo design, while `allo_entrypoint` selects the function called by the compilation node. `python_bin` and `allo_setup_script` select the Allo Python installation and matching LLVM environment setup script. The example constructors resolve these values from `ALLO_PYTHON` and `ALLO_SETUP_SCRIPT` and validate the paths before creating the graph.

The Vitis configuration shown above can be replaced with the following SystemC configuration:

```python
SYSTEMC_OPTIONS = (
    'device=nangate-45nm_beh,'
    'generalize_pid_specializations=true,'
    'pid_generalization_policy=identity_only'
)

# In the parameter dictionary:
'backend': 'systemc',
'backend_options': SYSTEMC_OPTIONS,
```

`backend='systemc'` asks Allo to emit SystemC using Connections channels and then uses Catapult as the synthesis engine. PID generalization allows processing elements with the same implementation but different coordinates to share one reusable implementation class.

Both backends produce the same manifest, workload, and RTL contracts consumed by the rest of the flow.

##### Design, ADK, and clock configuration

```python
    #---------------------------------------------------------------------
    # Design name and ADK
    #---------------------------------------------------------------------

    'design_name': 'top',
    'top_module': 'top',
    'report_design_name': '<descriptive design name>',

    'adk': 'freepdk-45nm',
    'adk_view': 'view-standard',

    #---------------------------------------------------------------------
    # Clock periods
    #---------------------------------------------------------------------

    'macro_clock_period': 8.0,
    'clock_period': 10.0,
```

`macro_clock_period` is the target period used during HLS and macro hardening. `clock_period` is the target period used for the assembled full chip. The full-chip period must be greater than or equal to the macro period so that each hardened macro is fast enough for the chip which instantiates it. It is recommended to have the clock period target be smaller for macros than the overall chip to leave slack for top-level wiring.

The backend-specific clock port is identified automatically from the generated RTL. Vitis designs use `ap_clk`, while SystemC designs use `clk`; the user does not need to specify this in the constructor.

##### RTL normalization and generated testbench

```python
    #---------------------------------------------------------------------
    # RTL normalization
    #---------------------------------------------------------------------

    'sv2v_bin': 'sv2v',
    'sv2v_defines': '',
    'sv2v_include_dirs': '',

    #---------------------------------------------------------------------
    # Generated testbench
    #---------------------------------------------------------------------

    'allo_testbench_enabled': True,
    'allo_testbench_workload_factory': 'testbench_workload',
    'allo_testbench_top_function': 'top',

    'testbench_name': 'allo_generated_testbench',
    'dut_name': 'dut',
```

The RTL normalization node ([see below](#RTL normalization (sv2v))) consumes RTL directly from the Allo compilation node and generates its own source manifest.

The testbench-generation node consumes the workload returned by the function `testbench_workload()`. Set parameters to match the testbench in your design.

##### Macro planning and physical intent

```python
    #---------------------------------------------------------------------
    # Macro planning
    #---------------------------------------------------------------------

    'min_macro_reuse': 2,
    'harden_repeated_hls_submodules': False,
    'bypass_macro_generation': False,
    'fold_fifos_into_macro': True,

    #---------------------------------------------------------------------
    # Hierarchical physical intent
    #---------------------------------------------------------------------

    'enable_kernel_rotation': True,
    'interleave_macros': False,

    'enable_macro_channel_soft_blockages': True,
    'macro_channel_soft_blockage_fraction': 0.3,
    'peripheral_placement_sides': 'all',

    'macro_separation_x': 5.0,
    'macro_separation_y': 5.0,
    'kernel_separation_x': 10.0,
    'kernel_separation_y': 10.0,
    'sram_separation': 10.0,
```

`min_macro_reuse` determines how many equivalent instances must exist before an implementation is hardened as a reusable macro. A value of 2 prevents kernels which only appear once from becoming macros while still hardening repeated processing elements.

`bypass_macro_generation` skips the macro-hardening stage entirely. `fold_fifos_into_macro` includes eligible stream FIFOs with the processing elements which own them (works since streams are point to point).

The physical-intent node uses the coordinates and channels recorded in the ASIC manifest to place related macros, determine legal rotations, and reserve routing space between communicating instances. Be sure to use suitable values for the PDK being used. Interleaving macros means that a more complicated placement algorithm is used than simply placing kernels next to each other based on connectivity information. Soft channel blockages between macros means that only physical-only standard cells and clock repair buffers can be placed in the channels between macros, with the fraction of that channel covered by the soft blockage tunable. Which side non-macro chip area is placed on is also tunable, with options including putting the area evenly on all sides, or only on some (done by passing those sides as a tuple to the parameter).

##### SRAM configuration

`sram_mode` selects how SRAM views are obtained:

```python
    #---------------------------------------------------------------------
    # SRAM source and generation configuration
    #---------------------------------------------------------------------

    # Valid modes: none, provided, bypass, generate
    'sram_mode': 'none',

    # Used by sram_mode='provided'
    'provided_sram_path': 'path/to/srams',

    # Used by sram_mode='generate' - OpenRAM configuration info
    'generate_method': 'openram',
    'sram_manifest': 'path/to/sram_manifest.yml',
    'openram_script': '',
    'tech_name': 'freepdk45',
    'process_corner': 'TT',
    'supply_voltage': 1.1,
    'temperature': 25,
    'check_lvsdrc': False,
    'route_supplies': True,
    'analytical_delay': True,
```

The supported modes are:

- `none`: publish an empty SRAM package
- `provided`: validate and package previously generated SRAM views
- `bypass`: repackage SRAM collateral supplied by an upstream node
- `generate`: generate SRAMs from `sram_manifest` using OpenRAM

If OpenRAM requires a different Python environment from Allo, its `python_bin` should be set directly on the SRAM node rather than in the shared parameter dictionary:

```python
sram_collateral.update_params(
    {
        'python_bin': openram_python,
    }
)
```

##### Synthesis configuration

```python
    #---------------------------------------------------------------------
    # Synthesis configuration
    #---------------------------------------------------------------------

    'flatten_effort': 1,
    'nthreads': 4,
    'high_effort_area_opt': False,
    'write_svsim_wrapper': False,
    'gate_clock': True,
    'uniquify_with_design_name': True,
    'suppress_msg': False,
```

The macro and full-chip synthesis nodes use different topographical defaults, so `topographical` mode should not normally be overridden globally. Macro synthesis uses topographical estimates, while full-chip synthesis consumes hardened macros as black boxes and leaves topographical mode disabled (both of these are set inside their respective node definitions).

##### ADK and technology files

```python
    #---------------------------------------------------------------------
    # ADK and technology files
    #---------------------------------------------------------------------

    'adk_tech_lef': 'inputs/adk/rtk-tech.lef',
    'adk_stdcell_lef': 'inputs/adk/stdcells.lef',
    'adk_gds_layer_map': 'inputs/adk/rtk-stream-out.map',
    'adk_qrc_lef_map': 'inputs/adk/pdk-qrc-lef.map',
    'adk_cap_table': 'inputs/adk/rtk-typical.captable',
    'adk_typical_lib': 'inputs/adk/stdcells.lib',
    'adk_bc_lib': 'inputs/adk/stdcells-bc.lib',
    'adk_wc_lib': 'inputs/adk/stdcells-wc.lib',
```

These paths are relative to the ADK package staged into each implementation node. They normally only need to be changed when adding or modifying an ADK.

##### General PNR configuration

```python
    #---------------------------------------------------------------------
    # General PNR operation
    #---------------------------------------------------------------------

    'enable_gui': False,
    'local_cpus': 4,
    'postroute_max_local_cpus': 4,

    # Valid values: none, floorplan, power, place, cts, route
    'stop_after_step': 'none',

    'process_node': 45,
    'gds_stream_out_units': 10000,

    'max_route_layer': 7,
    'highest_macro_routing_layer': 5,
    'base_layer_idx': 0,
    'pin_layer_offset': 3,
```

`stop_after_step` is mainly useful for debugging and partial physical-design runs. It should remain `none` for a complete flow.

The full-chip floorplan is generated by the physical-intent node, while individual macros use their own automatic or fixed floorplan settings. Max signal routing layers for both the top-level chip and inside macros can be tuned independently (macro limit must be less than or equal to top-level limit). Power stripes will be placed on the two metal layers above `max_route_layer`.

`enable_gui` determines whether Innovus is launched graphically or in headless mode. Headless mode is normally used for automated runs, while GUI mode is useful when debugging placement or routing. `local_cpus` sets the main Innovus thread limit, and `postroute_max_local_cpus` places a separate limit on the more memory-intensive post-route optimization steps. During macro PNR, these limits apply to each individual Innovus worker, so the total CPU use may be larger when multiple macros are implemented concurrently.

`process_node` tells Innovus the approximate process-node size used for its internal optimization settings. It normally comes from the ADK and should not be changed independently. `gds_stream_out_units` controls the coordinate precision used when writing GDS and should also remain consistent with the selected technology.

`base_layer_idx` records the technology's routing-layer index offset, while `pin_layer_offset` selects the ordinary signal-pin layer relative to that base. These layer-index parameters generally only need to change when porting the flow to another ADK.

##### Placement and timing optimization

```python
    #---------------------------------------------------------------------
    # Placement density
    #---------------------------------------------------------------------

    'core_density_target': 0.50,
    'macro_core_density_target': 0.70,
    'use_separate_macro_density_target': False,
    'cell_padding': 2,

    #---------------------------------------------------------------------
    # Timing optimization
    #---------------------------------------------------------------------

    'useful_skew': False,
    'ccopt_target_max_transition': 0.120,

    'macro_hold_target_slack': 0.050,
    'macro_setup_target_slack': 0.050,

    'hold_target_slack': 0.020,
    'hold_optimization_target_slack': 0.040,
    'setup_target_slack': 0.0,

    #---------------------------------------------------------------------
    # Macro clock boundary
    #---------------------------------------------------------------------

    'macro_clock_root_buffer_cell': 'CLKBUF_X1',
    'macro_clock_root_buffer_input_pin': 'A',
    'macro_clock_root_buffer_output_pin': 'Z',
    'clock_pin_depth_width_multiplier': 5.0,
```

The macro and full-chip stages have separate density and timing targets. `use_separate_macro_density_target` selects whether macro hardening uses `macro_core_density_target`; otherwise macros inherit `core_density_target`.

The dedicated macro clock-root buffer prevents the published macro clock pin from directly driving the aggregate internal clock load.

`cell_padding` adds placement-site spacing around the ADK cells selected for padding. This can relieve local routing congestion and leave room for optimization, at the cost of some additional area.

`useful_skew` allows clock-tree synthesis to intentionally vary clock arrival times to improve setup or hold timing. When it is disabled, CTS aims for a more conventionally balanced clock tree. `ccopt_target_max_transition` places an explicit transition-time target, in nanoseconds, on the clock tree. A value of zero leaves this choice to Innovus.

The setup and hold targets are all expressed in nanoseconds. `macro_setup_target_slack` and `macro_hold_target_slack` are used while hardening individual reusable macros. At the full-chip level, `setup_target_slack` controls setup optimization, while `hold_optimization_target_slack` is the margin Innovus attempts to reach during hold repair. `hold_target_slack` is the mandatory final threshold checked after signoff extraction. The optimization target can be larger than the mandatory threshold to provide margin for changes introduced during extraction and later optimization.

The macro clock-root buffer parameters must name a real standard cell and its input and output pins in the selected ADK. `clock_pin_depth_width_multiplier` controls how far the published clock pin extends into the macro as a multiple of the routing layer's minimum width. A deeper clock pin generally provides more room for full-chip clock routing and via access.

##### Power mesh and macro spacing

```python
    #---------------------------------------------------------------------
    # Power connectivity
    #---------------------------------------------------------------------

    'power_mesh_bot_layer': 8,
    'power_mesh_top_layer': 9,

    'primary_power_net': 'VDD',
    'primary_ground_net': 'VSS',

    'power_nets': 'VDD,VNW,VDDPST,POC,VDDCE,VDDPE',
    'ground_nets': 'VSS,VPW,VSSPST,VSSE',

    'power_pin_names': 'VDD',
    'ground_pin_names': 'VSS',

    #---------------------------------------------------------------------
    # Macro placement keepouts
    #---------------------------------------------------------------------

    'macro_halo': 0.5,
    'macro_pg_resource_util': 0.2,
    'macro_forbidden_space_to_macro': '20,20',
    'macro_min_space_to_core': '30,30',
    'macro_corner_keepout': '5,5',
    'macro_edge_keepout': 30.0,
    'macro_group_spacing': 20.0,
    'macro_group_max_depth': 2,

    #---------------------------------------------------------------------
    # Physical-only cells
    #---------------------------------------------------------------------

    'well_tap_cell': 'WELLTAP_X1',
    'well_tap_interval': 120,

    'lvs_exclude_cell_list': 'FILL*,WELLTAP*',
    'virtuoso_exclude_cell_list': 'FILL*,WELLTAP*',
```

`power_mesh_bot_layer` and `power_mesh_top_layer` select the lower and upper routing layers used by the full-chip power mesh. `primary_power_net` and `primary_ground_net` identify the main logical supply rails. The `power_nets` and `ground_nets` lists contain any additional technology supply names which should be connected to those rails, while `power_pin_names` and `ground_pin_names` identify the corresponding pins on standard cells and hard macros. These names are technology-specific and should normally come from the ADK.

`macro_halo` removes standard-cell placement rows immediately around each macro, leaving room for pin access and local routing. The physical-intent planner also uses it when deciding whether a short row fragment is large enough to keep. `macro_edge_keepout` reserves space between the core boundary and the macro region.

`macro_group_spacing` and `macro_group_max_depth` belong primarily to the older grouped-macro placement fallback. The normal Allo physical-intent path instead uses the macro, kernel, and SRAM separation settings described above. Similarly, `macro_pg_resource_util`, `macro_forbidden_space_to_macro`, `macro_min_space_to_core`, and `macro_corner_keepout` are compatibility settings retained from the shared Innovus template. They should normally remain at their defaults unless the generic placement behavior is being used.

`well_tap_cell` selects the technology well-tap cell, and `well_tap_interval` determines the maximum physical spacing between inserted taps. The flow verifies tap coverage after insertion and removes row fragments which cannot legally receive a tap. `lvs_exclude_cell_list` identifies physical-only cells which should be omitted from the LVS netlist, while `virtuoso_exclude_cell_list` controls the corresponding exclusions in the physical netlist exported for Virtuoso. Fill and well-tap cells are common exclusions because they are not part of the logical source hierarchy.

##### Simulation and SDF configuration

```python
    #---------------------------------------------------------------------
    # Common simulation configuration
    #---------------------------------------------------------------------

    'waveform': True,

    'ffgl_powerup_mode': 'random',
    'ffgl_powerup_seed': 2,

    'bagl_powerup_mode': 'random',
    'bagl_powerup_seed': 2,

    #---------------------------------------------------------------------
    # Back-annotated gate-level simulation
    #---------------------------------------------------------------------

    'sdf_corner': 'typ',

    'bagl_failure_policy': 'report',
    'sdf_warning_policy': 'report',
    'sdf_unmatched_timingcheck_policy': 'report',
    'sdf_unmatched_iopath_policy': 'report',
    'sdf_uphier_interconnect_policy': 'report',

    'bagl_input_delay_ns': 0.025,
    'bagl_output_delay_ns': 0.025,
    'bagl_num_reset_cycles': 8,
```

The supported FFGL and BAGL power-up modes are:

- `random`: initialize sequential state reproducibly using the corresponding seed
- `x`: leave sequential state unknown
- `zero`: initialize sequential state to zero
- `one`: initialize sequential state to one

Random seeds must be nonnegative integers, and VCS requires random seeds to be at least 2.

`sdf_corner` may be `typ`, `min`, or `max`.

The BAGL and SDF policies support:

- `error`: treat the corresponding failure or warning category as a flow failure
- `report`: preserve the result in the simulation report but allow later summary nodes to run

`bagl_failure_policy` controls whether a completed BAGL simulation which fails its workload or timing checks stops the flow. The four SDF policies control general SDF warnings, unmatched timing checks, unmatched IOPATH annotations, and ignored upward-hierarchy interconnect annotations separately.

`waveform` controls whether simulation retains its VCD output. The waveform is useful for debugging and is required when that simulation is selected as the source for downstream power analysis.

`bagl_input_delay_ns` adds a small delay before the testbench changes DUT inputs, while `bagl_output_delay_ns` adds margin before outputs are sampled. These are timing margins in nanoseconds rather than cycle counts. The BAGL node also extracts propagated-clock latency from the post-route constraints and accounts for it when driving and checking signals. `bagl_num_reset_cycles` determines how many clock cycles reset remains asserted before each transaction and may need to be increased for designs with a longer reset sequence.

##### Physical verification and signoff

```python
    #---------------------------------------------------------------------
    # Timing corners
    #---------------------------------------------------------------------

    'corner_setup': 'typical',
    'corner_hold': 'bc',

    #---------------------------------------------------------------------
    # DRC and antenna policies
    #---------------------------------------------------------------------

    'antenna_check_policy': 'report',
    'drc_check_policy': 'report',

    'drc_nthreads': 4,
    'drc_rule_deck': 'calibre-drc-block.rule',
    'drc_env_setup': 'undefined',

    #---------------------------------------------------------------------
    # LVS configuration
    #---------------------------------------------------------------------

    'lvs_nthreads': 4,
    'lvs_hcells_file': '',
    'lvs_connect_names': '',
    'lvs_verify_netlist': 1,
    'lvs_extra_spice_include': '',
    'lvs_power_name': 'VDD',
    'lvs_ground_name': 'VSS',
```

`antenna_check_policy` supports:

- `error`: require a clean antenna report
- `report`: record antenna violations without stopping the flow
- `off`: skip antenna checking

`drc_check_policy` supports:

- `error`: require clean non-antenna DRC and LVS results
- `report`: preserve incorrect DRC or LVS results in the flow summary without stopping the remaining reporting nodes

There is no separate `lvs_check_policy` in the integrated flow. Full-chip LVS currently uses the common physical-verification `drc_check_policy`.

`corner_setup` and `corner_hold` select the ADK timing-library corners used for setup and hold analysis during macro signoff. Their names must match corners published by the selected ADK.

`drc_nthreads` and `lvs_nthreads` independently control the processor counts used by Calibre DRC and LVS. `drc_rule_deck` names the rule deck staged by the ADK, while `drc_env_setup` optionally names an ADK-provided setup script which is sourced before Calibre is launched. Leaving it as `undefined` disables the extra setup step.

`lvs_power_name` and `lvs_ground_name` identify the source-netlist supply rails passed to V2LVS. They must be distinct and should agree with the primary power and ground nets used during PNR. `lvs_hcells_file` optionally provides a hierarchical-cell mapping file, and `lvs_connect_names` can specify differently named nets which Calibre should treat as connected. Empty values disable those additions.

`lvs_verify_netlist` controls Calibre's source-netlist verification setting, with 1 enabling the normal verification behavior. `lvs_extra_spice_include` accepts additional SPICE or CDL files for devices or hard macros whose source views are not already supplied by the ADK, SRAM package, or macro registry. The flow automatically adds the canonical LVS netlist published for each hardened macro, so this option is normally only needed for design- or technology-specific collateral.

##### Power analysis and reporting

```python
    #---------------------------------------------------------------------
    # Power analysis
    #---------------------------------------------------------------------

    'activity_source': 'bagl_vcd',
    'analysis_mode': 'averaged',
    'lib_op_condition': 'undefined',
    'saif_instance': 'allo_generated_testbench/dut',

    'macro_power_max_workers': 4,

    #---------------------------------------------------------------------
    # Reporting and cleanup
    #---------------------------------------------------------------------

    'google_sheets_enabled': False,
    'google_sheets_required': False,
    'cleanup_enabled': True,
}
```

The supported activity sources are `bagl_vcd`, `ffgl_vcd`, `rtl_vcd`, and `saif`. `analysis_mode` may be `averaged` or `time_based`.

`cleanup_enabled` removes large intermediate artifacts after the final reports have been collected. It should normally remain disabled while debugging and enabled for completed runs.

`saif_instance` identifies the testbench hierarchy which should be removed from the activity names before they are matched against the implemented design. Despite its historical name, this setting is used for both VCD and SAIF activity. It can be set to `auto` to derive the hierarchy from the generated testbench contract, or set explicitly when a nonstandard hierarchy is used.

`lib_op_condition` selects the Liberty operating condition used by PrimeTime. Leaving it as `undefined` allows PrimeTime to use the operating condition associated with the first target library. A named value should only be used when that condition exists in the selected ADK libraries.

`macro_power_max_workers` limits how many representative macro power jobs may run concurrently. The flow analyzes one representative of each macro class and multiplies its power by the number of reused instances before combining it with the full-chip shell power.

### Stepping Through the Flow

The integrated flow can be divided into five main stages. Most communication between these stages is handled through mflowgen inputs and outputs, so users ordinarily only need to define their Allo design, workload, and constructor parameters.

#### Allo Compilation and RTL Validation

The first stage runs the selected Allo compilation path. The Vitis path generates RTL directly, while the SystemC path generates Connections-based SystemC and uses Catapult for HLS. Both paths emit the same backend-neutral ASIC manifest describing the generated hierarchy, processes, ports, channels, and source files.

The compilation output is passed to the testbench-generation and RTL-normalization nodes. The testbench generator converts `testbench_workload()` into the testbench shared by the later simulation stages. The RTL-normalization node uses the source information emitted by compilation to determine the required source files and compilation order before invoking sv2v. Users do not need to maintain a separate sv2v source manifest for an integrated Allo design.

RTL simulation checks the normalized RTL against the generated workload before macro planning begins. This provides an early functional check before running synthesis or physical design.

#### Macro Planning and Hardening

The macro planner examines the ASIC manifest and groups instances with equivalent implementations. Implementations which meet `min_macro_reuse` are selected for hardening. One representative of each selected macro class is synthesized and placed and routed, rather than separately implementing every equivalent instance.

Each representative macro then passes through physical verification and timing signoff. Successfully implemented macros are published with the logical, physical, timing, and power views needed by the full-chip flow. The macro registry records which original RTL instances are represented by each published macro.

#### Full-Chip Assembly and Implementation

The assembly planner combines the normalized RTL with the macro registry. Instances selected for macro reuse are removed from the synthesizable RTL hierarchy and replaced with references to the corresponding hardened macros. Logic which was not selected for hardening remains as ordinary synthesizable RTL.

The remaining full-chip logic is synthesized into standard cells while the hardened macros are treated as black boxes. The physical-intent node uses hierarchy, process coordinates, channels, and macro geometry to generate the macro placement, orientation, floorplan, and routing constraints. Innovus then assembles and implements the complete design.

After routing, the flow merges the standard-cell, macro, and SRAM layouts into the final GDS. DRC and LVS operate on this assembled design.

#### Post-Implementation Validation

FFGL simulation runs after full-chip synthesis and checks the synthesized gate-level netlist without extracted delays. BAGL simulation runs after PNR and checks the routed netlist with SDF timing annotation. Both simulations use the workload generated from the original Allo design.

DRC checks the final layout for physical design-rule violations, while LVS compares the extracted layout connectivity against the logical implementation. The behavior of several checks is controlled by policy parameters. A policy set to `report` records a violation in the flow summary without necessarily stopping the complete flow. Consequently, a completed mflowgen run does not by itself mean that all simulations, timing checks, DRC checks, and LVS checks passed.

#### Power Analysis and Final Reporting

The flow uses activity from the selected RTL, FFGL, or BAGL simulation for power analysis. Full-chip shell power and representative macro power are analyzed separately. Power for each representative macro is multiplied by the number of equivalent instances before it is combined with the full-chip result.

The flow-summary node collects compilation, macro, simulation, timing, area, power, DRC, and LVS results. Its `outputs/flow-summary.txt` file is the easiest result to read manually, while `outputs/flow-summary.json` provides the same information in a machine-readable form. These reports, rather than the return status of `make` alone, should be used to decide whether a run was successful.

### Setting Up and Running an Example

The Vitis example in [`designs/allo-example-vitis/`](designs/allo-example-vitis/) and the SystemC/Catapult example in [`designs/allo-example-systemc/`](designs/allo-example-systemc/) are matching fixed 4-by-4 signed-8-bit designs intended as starting points for new work. Copy the example using the desired backend and modify its Allo design, workload, design names, clock targets, and physical-design parameters.

The constructors obtain the Allo Python executable and LLVM setup script from the environment:

```bash
export ALLO_PYTHON=/path/to/allo/environment/bin/python
export ALLO_SETUP_SCRIPT=/path/to/allo/llvm/setup.sh
```

The required commercial tools and licenses must also be available in the shell used to run mflowgen. Any machine-specific EDA setup should be performed separately from the Allo constructor.

Generated mflowgen builds should be placed outside the importable `allo/` package tree. From an empty build directory, generate a flow using:

```bash
mflowgen run --design /path/to/allo/allo/backend/asic/designs/<design-name>/construct-commercial.py
```

After the graph has been generated, `make list` shows the available steps and their numeric identifiers:

```bash
make list
```

`make status` reports which steps have completed, failed, or have not yet run:

```bash
make status
```

Running a numbered target executes that step and any unfinished dependencies:

```bash
make <step-number>
```

Running `make` without a target executes the complete flow:

```bash
make
```

When a step fails, first use `make status` to identify the failing node and then inspect that node’s log, reports, and `outputs/` directory.

## Python environments

The Allo-ASIC flow uses a small Conda environment to run mflowgen and its supporting utilities. This orchestration environment is separate from the environment containing Allo and its native MLIR extension.

A suitable orchestration environment can be created with:

```bash
conda create -n asicflow -c conda-forge python=3.12 pip numpy
conda activate asicflow
conda install ucb-bar::sv2v

python -m pip install --editable \
  "mflowgen @ git+https://github.com/mflowgen/mflowgen.git@aee0e5d640638bd38b44007d7258828eef66c641"
```

