#=========================================================================
# construct.py
#=========================================================================
# Demo with 16-bit GcdUnit
#
# Author : Christopher Torng
# Date   : June 2, 2019
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

  parameters = {
    'construct_path'      : __file__,
    'design_name'         : 'GcdUnit',
    'clock_period'        : 2.0,
    'adk'                 : adk_name,
    'adk_view'            : adk_view,
    # Enable GUIs
    'enable_gui'          : True,
    # GLS Testbench
    'saif_instance'       : 'GcdUnitTb/GcdUnit_inst',
    # Synthesis
    # Flatten effort 0 is strict hierarchy, 3 is full flattening
    'flatten_effort'      : 3,
    'topographical'       : True,
    # Postroute timing target slack
    'setup_target_slack'  : 0.000,
    'hold_target_slack'   : 0.050,
    # Utilization target
    'core_density_target' : 0.70,
    'consume_upstream_testbench' : True,
    'testbench_top'       : 'GcdUnitTb',
    'dut_instance'        : 'GcdUnit_inst',
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

  # Custom nodes

  rtl            = Node( this_dir + '/rtl'         )
  testbench      = Node( this_dir + '/testbench'   )
  testbench_collect = Node( os.path.join(nodes_dir, 'testbench-collector')        )
  constraints    = Node( this_dir + '/constraints' )

  # Default nodes

  synth          = Node( os.path.join(nodes_dir, 'synopsys-dc-synthesis')         )
  pnr            = Node( os.path.join(nodes_dir, 'cadence-innovus-pnr')           )
  pt_signoff     = Node( os.path.join(nodes_dir, 'synopsys-pt-timing-signoff'))
  genlibdb       = Node( os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb')    )
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

  g.add_node( rtl               )
  g.add_node( constraints       )
  g.add_node( synth             )
  g.add_node( pnr               )
  g.add_node( pt_signoff        )
  g.add_node( genlibdb          )
  g.add_node( gdsmerge          )
  g.add_node( drc               )
  g.add_node( lvs               )
  g.add_node( testbench         )
  g.add_node( testbench_collect )
  g.add_node( utilities         )
  g.add_node( rtl_sim           )
  g.add_node( ffgl_sim          )
  g.add_node( bagl_sim          )
  g.add_node( power_est         )
  g.add_node( summary           )
  g.add_node( finalize          )

  #-----------------------------------------------------------------------
  # Graph -- Add edges
  #-----------------------------------------------------------------------

  # Connect by name

  g.connect_by_name( adk,            synth          )
  g.connect_by_name( adk,            pnr            )
  g.connect_by_name( adk,            pt_signoff     )
  g.connect_by_name( adk,            gdsmerge       )
  g.connect_by_name( adk,            drc            )
  g.connect_by_name( adk,            lvs            )
  g.connect_by_name( adk,            genlibdb       )
  g.connect_by_name( adk,            ffgl_sim       )
  g.connect_by_name( adk,            bagl_sim       )

  g.connect_by_name( rtl,            synth          )
  g.connect_by_name( constraints,    synth          )

  g.connect_by_name( synth,          pnr            )
  g.connect_by_name( synth,          ffgl_sim       )
  for node in [pt_signoff, genlibdb, gdsmerge, drc, lvs, bagl_sim, power_est]:
    g.connect_by_name( pnr, node )

  g.connect_by_name( gdsmerge,       drc            )
  g.connect_by_name( gdsmerge,       lvs            )

  g.connect_by_name( rtl,            rtl_sim        )
  g.connect_by_name( testbench,      testbench_collect )
  for node in [rtl_sim, ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name( testbench_collect, node )
  for node in [rtl_sim, ffgl_sim, bagl_sim, lvs]:
    g.connect_by_name( utilities, node )

  g.connect_by_name( adk,            power_est      )
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
