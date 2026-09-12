# allo-asic

Repository for automatic ASIC flow scripts, including a simple flat flow and the in-progress AAAH (Allo-Aided ASIC Harness). A brief introduction to the flat flow as well as a tutorial is below. 

## Flat ASIC Flow Background/Tutorial

Julian Bushlow, 09/2026

File intended to give general info/a tutorial on this ASIC flow. Assumes that reader knows parts of an ASIC flow and their role. A template design can be found in [`designs/template-design/`](designs/template-design/)

### Orchestration: mflowgen

File management can be a bottleneck of designer effort when making automatic ASIC flows, since the scripts need to know exactly where certain tools will dump their outputs so the next tool can use them. Manually specifying these paths can make sharing or modifying a flow tedious and brittle. 

[mflowgen](https://mflowgen.readthedocs.io/en/latest/) is a modular flow organizer which neatly handles this file management. Each step can be packaged as a node in a flow graph, where the outputs of each step are assumed to be in the dedicated `outputs/` directory for each step's directory (and are also listed in the node's `configure.yml` file). If another node's `configure.yml` lists that output as one of its inputs, it will be copied via symlink into the node's `inputs/` directory and can be used from there. 

Since ASIC flow steps can be accomplished using multiple procedures/tools and some steps are optional, a library of these nodes can be made and reused. I put together a collection of nodes for a simple RTL <span>&rarr;</span> GDS flow.

Note that a custom node can be defined just using a configure.yml file, so some designs (like the GCD unit design I adapted from the mflowgen tutorials) build custom nodes in the design directory and call them inside their constructor files. This process requires user expertise though, so I tried to make all needed nodes so they can be simply instantiated.

### Making mflowgen Designs

Making a new mflowgen design involves setting up main constructor files and testbenches (the Allo-ASIC flow can generate the testbenches automatically). The structure below is for a design using commercial tools, open source tool builds use a slightly different structure. Some sections are required by mflowgen, others are contracts used by the nodes I set up for this flow. 

```
designs/MyDesign/
├── construct-commercial.py
├── sv2v_manifest.f
├── sram_manifest.yml       # only when SRAMs are used
└── testbench/
    ├── MyDesignTb.sv
    ├── testbench_manifest.f    # useful for multiple source files
    └── other used files
```

#### RTL/SRAMs

SV2V automatically parses RTL into a single readable file for the synthesis tools (resolving dependencies, converting SystemVerilog into Verilog, etc). This step requires knowing which files/directories are being used in the design, which directories to look for them in, and which order to compile them (some tools complain about modules/constructs being used before being defined). The sv2v manifest file contains the ordering information, with other information being found in the [constructor file](#constructors). An example SV2V manifest is below. Note that a package can explicitly be excluded from compilation by adding "!" in front of it (this is useful for when a logical memory implementation is needed but unwanted for ASIC synthesis).

```
# Packages
pkg/npu_config_pkg.sv
pkg/npu_isa_pkg.sv
pkg/npu_modes_pkg.sv
pkg/tile_agu_pkg.sv

# Shared utility cells
common/

# Core compute plane
core/dmu/
core/mxu/
core/sequencer/
core/spad/
core/top/
core/vpu/fpu/
core/vpu/

# Excludes
!core/mxu/systolic_tb_wrap.sv

```

A second manifest must also be used for SRAM generation using OpenRAM (note: we'll likely swap to a different system for ASAP7, but a similar interface will be used). This manifest identifies each SRAM as used in the design with some basic information with other information being found in the [constructor file](#constructors). Note that this manifest is only required when using `sram_mode='generate'`

```
srams:
- name: sram_bank_asic
    word_size: 32
    num_words: 1024
	num_rw_ports: 2
	num_r_ports: 0
	num_w_ports: 0
	num_banks: 1
	words_per_row: 2
	write_size: 32

- name: iram_asic
	word_size: 64
	num_words: 4096	
	num_rw_ports: 2	
	num_r_ports: 0	
	num_w_ports: 0	
	num_banks: 1	
	words_per_row: 4	
	write_size: 64
```

#### Testbenches

Unlike in the Allo-ASIC flow where testbench RTL files can be automatically generated, pure-RTL designs need a testbench to be provided by the user to allow for thorough simulation (the flow directly uses RTL, FFGL, and BAGL simulation for verification at each step) as well as downstream power simulation. To standardize, the following contract is used. 

The top-level structure of the testbench `<testbench_file>` must be

```
module <testbench_top>;

  <module name> <dut_instance> (
    // ...
  );

endmodule
```

Additionally, the testbench must also print `<pass_marker>` if all checks pass and `<failure_marker>` if any fail. The DUT should be placed at `<testbench_top.dut_instance>` for SDF annotation. The testbench collector packages the given testbench and provides information required by downstream nodes in JSON files.

#### Constructors

Constructor files are where most of the important parameters are tuned, including more physical information like clock period density targets, run settings like whether to generate SRAMs or how many cores to use, and logistical paths to RTL and pre-generated SRAMs. The constructor file is also where nodes are actually connected into a graph to make a flow. For an example, look at [`designs/tutorial-vvadd/construct-commercial.py`](designs/tutorial-vvadd/construct-commercial.py). 

The script takes the general form:
- define graph
- define list of parameters (to override defaults)
- instantiate nodes, connect to graph
- connect nodes together

A full list/explanation of all current parameters (with defaults when applicable) is below. Note that most of these do not have to be set at all and can be left unset in the constructor file.

```
parameters = {
    
    #---------------------------------------------------------------------
    # Design name/ADK
    #---------------------------------------------------------------------

    'construct_path': 'path/to/this/file', 
    'design_name': '<name of design>',
    'top_module': '<top module in design>',

    'adk': 'freepdk-45nm',                   # adk name defined in node
    'adk_view': 'view-tiny',

    #---------------------------------------------------------------------
    # RTL source paths and sv2v collection
    #---------------------------------------------------------------------

    'design_path': 'path/to/your/rtl',
    'manifest': 'path/to/sv2v-manifest',
    'sv2v_bin': 'sv2v',                      # change if using diff sv2v version
    'sv2v_defines': 'TARGET_ASIC=1',         # add compilation variables here
    'sv2v_include_dirs': '.',                # where other files in include  
                                             # statements can be found, separate 
                                             # with ":" character
    'normalize_rtl': False,                  # Whether or not to run sv2v
                                             # translation

    #---------------------------------------------------------------------
    # Testbench paths and configuration
    #---------------------------------------------------------------------

    'testbench_path': 'path/to/testbench/directory',
    'testbench_file': '<name of testbench file in directory>',
    'consume_upstream_testbench': False,     # if upstream node outputs testbench
    'testbench_manifest': '',                # if multiple files need compilation
    'testbench_top': 'tb',
    'dut_instance': 'dut',
    'testbench_include_dirs': '.',           # same format as SV2V
    'testbench_defines': '',
    'simulation_args_file': '',
    'pass_marker': 'TEST_PASS',
    'failure_marker': 'TEST_FAIL',
    'simulation_timeout_seconds': 3600,

    #---------------------------------------------------------------------
    # Common simulation configuration
    #---------------------------------------------------------------------

    'waveform': True,
    'xprop_enabled': True,
    'cleanup_enabled': True,

    #---------------------------------------------------------------------
    # SRAM source and generation configuration
    #---------------------------------------------------------------------

    # Valid modes: none, provided, bypass, generate
    'sram_mode': 'none',

    # Used by sram_mode='provided'
    'provided_sram_path': 'path/to/srams',

    # Used by sram_mode='generate'
    'generate_method': 'openram',
    'sram_manifest': 'path/to/sram/manifest',
    'python_bin': 'python',                     # use openram-supporting env bin
    'openram_script': '',
    'tech_name': 'freepdk45',
    'process_corner': 'TT',
    'supply_voltage': 1.1,
    'temperature': 25,
    'check_lvsdrc': False,
    'route_supplies': True,
    'analytical_delay': True,

    #---------------------------------------------------------------------
    # Clock and design constraints
    #---------------------------------------------------------------------

    'clock_period': 1.0,
    'clock_port': 'clk',
    'clock_name': 'ideal_clock',
    'constraints_file': '',
    'input_delay_fraction': 0.5,
    'output_delay_fraction': 0.0,
    'max_transition_fraction': 0.25,
    'max_fanout': 20,
    'clock_uncertainty': 0.0,

    #---------------------------------------------------------------------
    # Synthesis configuration
    #---------------------------------------------------------------------

    'flatten_effort': 3,
    'topographical': True,                  # must be False if using macros
    'nthreads': 16,
    'high_effort_area_opt': False,
    'write_svsim_wrapper': False,
    'gate_clock': True,
    'uniquify_with_design_name': False,
    'suppress_msg': False,

    'suppressed_msg': [
        'TFCHK-072',
        'TFCHK-014',
        'TFCHK-049',
        'TFCHK-050',
        'TFCHK-012',
        'TFCHK-073',
        'TFCHK-092',
        'PSYN-651',
        'PSYN-650',
    ],

    #---------------------------------------------------------------------
    # ADK and technology file paths used by PNR
    #---------------------------------------------------------------------

    'adk_tech_lef': 'inputs/adk/rtk-tech.lef',
    'adk_stdcell_lef': 'inputs/adk/stdcells.lef',
    'adk_gds_layer_map': 'inputs/adk/rtk-stream-out.map',
    'adk_qrc_lef_map': 'inputs/adk/pdk-qrc-lef.map',
    'adk_cap_table': 'inputs/adk/rtk-typical.captable',
    'adk_typical_lib': 'inputs/adk/stdcells.lib',
    'adk_bc_lib': 'inputs/adk/stdcells-bc.lib',
    'adk_wc_lib': 'inputs/adk/stdcells-wc.lib',

    #---------------------------------------------------------------------
    # General PNR operation
    #---------------------------------------------------------------------

    'enable_gui': True,
    'local_cpus': 16,
    'postroute_max_local_cpus': 8,

    # Valid values: none, floorplan, power, place, cts, route
    'stop_after_step': 'none',

    'process_node': 45,
    'gds_stream_out_units': 1000,
    'max_route_layer': 7,
    'base_layer_idx': 0,
    'pin_layer_offset': 3,

    #---------------------------------------------------------------------
    # Placement and optimization
    #---------------------------------------------------------------------

    'core_density_target': 0.7,
    'cell_padding': 2,

    'useful_skew': True,
    'useful_skew_ccopt_effort': 'standard',
    'ccopt_target_max_transition': 0.0,

    'signoff_engine': False,
    'hold_optimization_target_slack': 0.02,
    'setup_target_slack': 0.0,

    #---------------------------------------------------------------------
    # Floorplan configuration
    #---------------------------------------------------------------------

    # Valid values: auto, fixed
    'floorplan_mode': 'auto',

    # Used by floorplan_mode='auto'
    'floorplan_aspect_ratio': 1.0,

    # Required by floorplan_mode='fixed'
    'floorplan_width': '',
    'floorplan_height': '',

    #---------------------------------------------------------------------
    # Power mesh and physical power connectivity
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
    # Hard-macro placement
    #---------------------------------------------------------------------

    'macro_halo': 2.0,
    'macro_pg_resource_util': 0.2,
    'macro_forbidden_space_to_macro': '20,20',
    'macro_min_space_to_core': '30,30',
    'macro_corner_keepout': '5,5',
    'macro_edge_keepout': 30.0,
    'macro_group_spacing': 20.0,
    'macro_group_max_depth': 2,

    #---------------------------------------------------------------------
    # Well taps and physical-only cells
    #---------------------------------------------------------------------

    'well_tap_cell': 'WELLTAP_X1',
    'well_tap_interval': 120,

    'lvs_exclude_cell_list': 'FILL*,WELLTAP*',
    'virtuoso_exclude_cell_list': 'FILL*,WELLTAP*',

    #---------------------------------------------------------------------
    # Back-annotated gate-level simulation
    #---------------------------------------------------------------------

    'sdf_corner': 'typ',

    # Valid values generally include error and report
    'bagl_failure_policy': 'error',
    'sdf_warning_policy': 'report',
    'sdf_unmatched_timingcheck_policy': 'report',
    'sdf_unmatched_iopath_policy': 'report',
    'sdf_uphier_interconnect_policy': 'report',

    #---------------------------------------------------------------------
    # Timing signoff
    #---------------------------------------------------------------------

    'corner_setup': 'typical',
    'corner_hold': 'bc',
    'timing_check_policy': 'error',

    #---------------------------------------------------------------------
    # Innovus and Calibre DRC policies
    #---------------------------------------------------------------------

    'antenna_check_policy': 'report',
    'drc_check_policy': 'error',

    'drc_nthreads': 16,
    'drc_rule_deck': 'calibre-drc-block.rule',
    'drc_env_setup': 'undefined',

    #---------------------------------------------------------------------
    # LVS configuration
    #---------------------------------------------------------------------

    'lvs_nthreads': 16,
    'lvs_hcells_file': '',
    'lvs_connect_names': '',
    'lvs_verify_netlist': 1,
    'lvs_extra_spice_include': '',
    'lvs_power_name': 'VDD',
    'lvs_ground_name': 'VSS',
    'lvs_check_policy': 'error',

    #---------------------------------------------------------------------
    # Power analysis
    #---------------------------------------------------------------------

    # Valid activity sources include bagl_vcd and other sources supported by
    # the activity-preparation node.
    'activity_source': 'bagl_vcd',
    'analysis_mode': 'averaged',
    'lib_op_condition': 'undefined',
}
```

### mflowgen Usage & Examples

Using an mflowgen design is quite simple, after making sure all tools are available (use provided setup script if given one), make a build directory anywhere and then run

```
mflowgen run --design ~/allo-asic/designs/<design name>/construct-commercial.py
```

From there, the graph will be built. Run `make list` or `make status` to see which steps there are to be run. From there, running `make 2` will run all the steps required to build step 2, and so on. Running `make` will run all of them. The flow will fail if any step along the way fails its pre or postconditions

#### Vector Vector Add Example

To show an example of how a full design (with SRAMs and RTL outside of the repo) works, I used an agent to build a simple example with a vector vector adder which contains an SRAM. The actual design is a little contrived, but all that matters for this tutorial is that we have a functioning design complete with its own testbench. Zipped folders for both the RTL and the SRAMs are included at [this link](https://drive.google.com/drive/folders/1jt2y0ilfUMVtYLf1GeZU8Cxo9bmzvmZO?usp=drive_link) (SRAMs were generated using OpenRAM). Unzip and put these files anywhere in your file system.

Take a look at [`designs/tutorial-vvadd/construct-commercial.py`](designs/tutorial-vvadd/construct-commercial.py):

The file is what defines the flow graph and instantiates predefined nodes (found in [`nodes/`](nodes/)). The script takes the general form:
- define graph
- define list of parameters (to override defaults)
- define nodes, connect to graph
- connect nodes together

I set up the flat flow to have certain nodes and connections ahead of time, so the only thing to edit is the parameters. A full list of parameters can be found elsewhere in this doc, leaving them unset keeps them at their default values. Some parameters, including paths to RTL, need to be set for the design to work. I added annotations here for clarity.

##### Design & ADK

```
'construct_path': __file__,
'design_name': 'vvadd',
'top_module': 'vvadd',
'adk': 'freepdk-45nm',
'adk_view': 'view-standard',
```

- these parameters give the path to the constructor file, as well as design names and ADK names 

##### Physical Parameters

```
'clock_period': 10.0,
'hold_target_slack': 0.050,
'clock_port': 'clk',
'core_density_target': 0.50,
'floorplan_aspect_ratio': .67,
'topographical': False,
```

- Physical parameters like clock period, target hold time slack, name of the clock signal, target stdcell placement density, and floorplan aspect ratio
	- SRAM used in this design is a rectangular shape, so if the chip was set to a square aspect ratio it would protrude off the side. This is usually not a problem for larger designs, but a fixed floorplan can also be used.
- The flow uses Synopsys Design Compiler for synthesis. When `topographical` is set to True, the tool will estimate physical information and results. However, the estimates crash when black box macros (like SRAMs) are in use

##### RTL Path & SV2V

```
'design_path': 'path/to/unzipped/rtl/here',
'manifest': 'sv2v_manifest.f',
'sv2v_include_dirs': '.:include',
'normalize_rtl': True,
```

- `design_path` is where the path to the RTL is specified - one of the few parameters required to use the flow!
- The flow uses SV2V to convert large many-file designs into one neat file that can be parsed easily by Synopsys DC. It can also convert SystemVerilog to Verilog
	- 'normalize_rtl' determines whether SV2V is actually run - if set to False, translation is skipped
	- `manifest` gives the path to the sv2v manifest (see elsewhere in this guide for details) where the order in which the tool should parse and write files is given. 
		- Make sure that files are listed in the right order so that modules are not instantiated before being defined, synthesis tools may error
		- A manifest file should be written in the design directory, so this input should be able to stay the same
	- `sv2v_include_dirs` indicates where other files used in `include` statements can be found. In this example design, the only other file that needs to be included is where some design parameters are given
		- paths to files/directories that have to be included are separated by ':'

This example design has the following file tree which gets normalized into one file by SV2V:

```
vvadd-example-rtl/
├── compute
│   ├── adder16.v
│   └── vvadd_datapath.v
├── control
│   └── vvadd_controller.v
├── include
│   └── vvadd_defs.vh
├── memory
│   └── sram_1rw_adapter.v
└── top
    └── vvadd.v
```

##### SRAMs

```
'sram_mode': 'provided',
'provided_sram_path': '~/srams-vvadd-tutorial/',
# parameters for if generating SRAMs using OpenRAM
#'sram_mode': 'generate',
#'generate_method': 'openram',
#'sram_manifest': 'sram_manifest.yml',
#'python_bin': os.environ.get('OPENRAM_PYTHON', 'python'),
'power_pin_names': 'VDD,vdd',
'ground_pin_names': 'VSS,gnd',
```

- This design uses 1 SRAM. Supported modes for how the flow handles SRAMs includes "provided" where the user has to provide a path to the SRAMs (as in this case), "none" where the node becomes a passthrough (use when the design uses no SRAMs), "bypass" which simply repackages SRAMs emitted by an upstream node (unused in this flow),  and "generate" where the flow will use an SRAM compiler to generate the required SRAMs
	- Generating SRAMs takes a long time and can be tricky with using the compilers, so it's advised to generate SRAMs once and then reuse them later (using "provided" mode)
	- Setting up OpenRAM for this flow required much pain and requires a separate conda environment, so I included the required parameters (commented out) for using OpenRAM while using FreePDK45
	- Generated SRAMs will often have differently-named power and ground pins, so include those names along with the standard VDD and VSS in `power_pin_names` and `ground_pin_names`

##### DRC, LVS, & Simulation

```
'drc_check_policy': 'report',
'lvs_check_policy': 'error',

'testbench_path': 'testbench',
'testbench_file': 'VvaddTb.v',
'testbench_top': 'VvaddTb',
'pass_marker': 'PASS',
```

- When you don't fully care if your design has  DRC or LVS issues (may be preferred if PDK is flawed in some way/not set up well for tools to correctly find issues) and want to let the flow fully finish without stopping for errors, the DRC and LVS checking policies can be changed from error to report.
- Also include all parameters to describe the testbench being used for the design. Make sure to follow the testbench contract designed elsewhere in this file.


