#=========================================================================
# construct.py
#=========================================================================
# Commercial ASIC flow for the compute tile block
#
# Author : Julian Bushlow
# Date   : June 16, 2026
#

import os

from mflowgen.components import Graph, Node

def construct():

  g = Graph()

  #-----------------------------------------------------------------------
  # Parameters
  #-----------------------------------------------------------------------

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'
  openram_python = os.environ.get('OPENRAM_PYTHON', 'python')

  parameters = {
    'construct_path'      : __file__,
    'design_name'         : 'compute_tile',
    'clock_period'        : 10.0,
    'adk'                 : adk_name,
    'adk_view'            : adk_view,
    # Enable GUIs
    'enable_gui'          : True,
    # GLS Testbench
    'saif_instance'       : 'ComputeTileTb/compute_tile_inst',
    # Synthesis
    # Flatten effort 0 is strict hierarchy, 3 is full flattening
    'flatten_effort'      : 3,
    'topographical'       : False,
    # Postroute timing target slack
    'setup_target_slack'  : 0.000,
    'hold_target_slack'   : 0.050,
    # Utilization target
    'core_density_target' : 0.70,
    # SV2V params
    'top_module'        : 'compute_tile',
    'design_path'       : '../../../mininpu/npu/src',
    'sv2v_include_dirs' : '.:common:compute_tile:compute_tile/fpu',
    # OpenRAM params
    'sram_manifest'  : 'rtl/sram_manifest.yml',
    'python_bin'     : openram_python,
    'openram_script' : '',
    'tech_name'      : 'freepdk45',
    'process_corner' : 'TT',
    'supply_voltage' : 1.1,
    'temperature'    : 25,
    'check_lvsdrc'   : False,
    'route_supplies' : True,     ## shortcut to skip power routing
    'analytical_delay' : True,
    #'use_sram_cache' : True,
    #'sram_cache_path': 'srams',
    'consume_upstream_testbench' : True,
    'testbench_top'       : 'ComputeTileTb',
    'dut_instance'        : 'compute_tile_inst',
    'activity_source'     : 'bagl_vcd',
    'pass_marker'         : 'PASS',
  }

  #-----------------------------------------------------------------------
  # Create nodes
  #-----------------------------------------------------------------------

  this_dir  = os.path.dirname( os.path.abspath( __file__ ) )
  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  # ADK node

  g.sys_path.append(os.path.join(asic_dir, "adks"))
  g.set_adk( adk_name )
  adk = g.get_adk_node()

  testbench      = Node( this_dir + '/testbench'   )
  testbench_collect = Node( os.path.join(nodes_dir, 'testbench-collector')        )
  constraints    = Node( this_dir + '/constraints' )

  openram        = Node( os.path.join(nodes_dir, 'openram-sram-generation')       )
  sv2v           = Node( os.path.join(nodes_dir, 'sv2v-design-collector')         )
  synth          = Node( os.path.join(nodes_dir, 'synopsys-dc-synthesis')         )
  pnr            = Node( os.path.join(nodes_dir, 'cadence-innovus-pnr')           )
  pt_signoff     = Node( os.path.join(nodes_dir, 'synopsys-pt-timing-signoff')    )
  genlibdb       = Node( os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb')        )
  gdsmerge       = Node( os.path.join(nodes_dir, 'mentor-calibre-gdsmerge')       )
  drc            = Node( os.path.join(nodes_dir, 'mentor-calibre-drc')            )
  lvs            = Node( os.path.join(nodes_dir, 'mentor-calibre-lvs')            ) 
  utilities      = Node( os.path.join(nodes_dir, 'asic-flow-utilities')            )
  rtl_sim        = Node( os.path.join(nodes_dir, 'commercial-rtl-sim')             )
  ffgl_sim       = Node( os.path.join(nodes_dir, 'commercial-ffgl-sim')            )
  bagl_sim       = Node( os.path.join(nodes_dir, 'commercial-bagl-sim')            )
  power_est      = Node( os.path.join(nodes_dir, 'synopsys-pt-power')             )
  summary        = Node( os.path.join(nodes_dir, 'asic-flow-summary')              )
  finalize       = Node( os.path.join(nodes_dir, 'asic-flow-finalize')             )

  #-----------------------------------------------------------------------
  # Modify Nodes
  #-----------------------------------------------------------------------

  #-----------------------------------------------------------------------
  # Graph -- Add nodes
  #-----------------------------------------------------------------------

  g.add_node( sv2v              )
  g.add_node( testbench         )
  g.add_node( openram           )
  g.add_node( constraints       )
  g.add_node( synth             )
  g.add_node( pnr               )
  g.add_node( gdsmerge          )
  g.add_node( drc               )
  g.add_node( lvs               )
  g.add_node( testbench_collect )
  g.add_node( utilities         )
  g.add_node( rtl_sim           )
  g.add_node( ffgl_sim          )
  g.add_node( bagl_sim          )
  g.add_node( pt_signoff        )
  g.add_node( power_est         )
  g.add_node( genlibdb          )
  g.add_node( summary           )
  g.add_node( finalize          )

  #-----------------------------------------------------------------------
  # Graph -- Add edges
  #-----------------------------------------------------------------------

  # Connect by name

  for node in [synth, pnr, pt_signoff, genlibdb, gdsmerge, drc, lvs,
               ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name( adk, node )
  for node in [synth, pnr, pt_signoff, genlibdb, gdsmerge, lvs,
               rtl_sim, ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name( openram, node )
  g.connect_by_name( sv2v,           synth          )
  g.connect_by_name( sv2v,           rtl_sim        )
  g.connect_by_name( constraints,    synth          )
  g.connect_by_name( synth,          pnr            )
  g.connect_by_name( synth,          ffgl_sim       )
  for node in [pt_signoff, genlibdb, gdsmerge, drc, lvs, bagl_sim, power_est]:
    g.connect_by_name( pnr, node )
  g.connect_by_name( gdsmerge,       drc            )
  g.connect_by_name( gdsmerge,       lvs            )
  g.connect_by_name( testbench,      testbench_collect )
  for node in [rtl_sim, ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name( testbench_collect, node )
  for node in [rtl_sim, ffgl_sim, bagl_sim, lvs]:
    g.connect_by_name( utilities, node )
  g.connect_by_name( bagl_sim,       power_est      )
  for source, output, target in [
    (synth, 'synthesis-metrics.json', 'synthesis-metrics.json'),
    (pnr, 'pnr-metrics.json', 'pnr-metrics.json'),
    (pt_signoff, 'timing-metrics.json', 'timing-metrics.json'),
    (gdsmerge, 'gdsmerge-metrics.json', 'gdsmerge-metrics.json'),
    (drc, 'drc-metrics.json', 'drc-metrics.json'),
    (drc, 'drc-policy.json', 'drc-policy.json'),
    (lvs, 'lvs-metrics.json', 'lvs-metrics.json'),
    (power_est, 'power.rpt', 'power.rpt'),
    (power_est, 'activity-source.json', 'activity-source.json'),
    (rtl_sim, 'simulation-report.json', 'rtl-simulation-report.json'),
    (ffgl_sim, 'simulation-report.json', 'ffgl-simulation-report.json'),
    (bagl_sim, 'simulation-report.json', 'bagl-simulation-report.json'),
  ]:
    g.connect(source.o(output), summary.i(target))
  g.connect_by_name( summary,         finalize       )

  #-----------------------------------------------------------------------
  # Parameterize
  #-----------------------------------------------------------------------

  g.update_params( parameters )

  return g


if __name__ == '__main__':
  g = construct()
#  g.plot()
