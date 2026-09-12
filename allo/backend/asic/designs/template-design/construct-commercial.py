"""Template design to be filled in as needed"""

import os

from mflowgen.components import Graph, Node


def construct():
  """Build the commercial RTL-to-GDS graph."""
  graph = Graph()

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'
  
  parameters = {
    'construct_path': __file__,
    'design_name': '<design_name>',
    'top_module': '<design_name>',
    'adk': adk_name,
    'adk_view': adk_view,

    'clock_period': 10.0,
    'hold_target_slack': 0.050,
    'clock_port': 'clk',
    'core_density_target': 0.70,
    'topographical': True,
    
    'design_path': '<path/to/rtl>',
    'manifest': '<path/to/sv2v/manifest>',
    'sv2v_include_dirs': '.',
    'normalize_rtl': True,
    
    'sram_mode': 'provided',
    'provided_sram_path': '<path/to/SRAMs>',
    # parameters for if generating SRAMs using OpenRAM
    #'sram_mode': 'generate',
    #'generate_method': 'openram',
    #'sram_manifest': '<path/to/SRAM/manifest>',
    #'python_bin': os.environ.get('OPENRAM_PYTHON', 'python'),
    'power_pin_names': 'VDD',
    'ground_pin_names': 'VSS',

    'drc_check_policy': 'report',
    'lvs_check_policy': 'error',

    'testbench_path': '<path/to/testbench/directory>',
    'testbench_file': '<testbench file name>',
    'testbench_top': '<top level module in testbench file>',
    'pass_marker': 'PASS',
  }

  this_dir = os.path.dirname(os.path.abspath(__file__))
  asic_dir = os.path.dirname(os.path.dirname(this_dir))
  nodes_dir = os.path.join(asic_dir, 'nodes')

  graph.sys_path.append(os.path.join(asic_dir, 'adks'))
  graph.set_adk(adk_name)
  adk = graph.get_adk_node()

  node_names = [
    'sv2v-design-collector',
    'testbench-collector',
    'sram-collateral',
    'asic-flow-utilities',
    'commercial-rtl-sim',
    'synopsys-dc-synthesis',
    'commercial-ffgl-sim',
    'cadence-innovus-pnr',
    'commercial-bagl-sim',
    'synopsys-pt-timing-signoff',
    'synopsys-ptpx-genlibdb',
    'mentor-calibre-gdsmerge',
    'mentor-calibre-drc',
    'mentor-calibre-lvs',
    'synopsys-pt-power',
    'asic-flow-summary',
    'asic-flow-finalize',
  ]
  nodes = {name: Node(os.path.join(nodes_dir, name)) for name in node_names}
  for node in nodes.values():
    graph.add_node(node)

  sv2v      = nodes['sv2v-design-collector']
  testbench = nodes['testbench-collector']
  sram      = nodes['sram-collateral']
  utilities = nodes['asic-flow-utilities']
  rtl_sim   = nodes['commercial-rtl-sim']
  synth     = nodes['synopsys-dc-synthesis']
  ffgl_sim  = nodes['commercial-ffgl-sim']
  pnr       = nodes['cadence-innovus-pnr']
  bagl_sim  = nodes['commercial-bagl-sim']
  timing    = nodes['synopsys-pt-timing-signoff']
  genlibdb  = nodes['synopsys-ptpx-genlibdb']
  gdsmerge  = nodes['mentor-calibre-gdsmerge']
  drc       = nodes['mentor-calibre-drc']
  lvs       = nodes['mentor-calibre-lvs']
  power     = nodes['synopsys-pt-power']
  summary   = nodes['asic-flow-summary']
  finalize  = nodes['asic-flow-finalize']

  for node in [synth, pnr, timing, genlibdb, gdsmerge, drc, lvs,
               ffgl_sim, bagl_sim, power]:
    graph.connect_by_name(adk, node)

  for node in [synth, pnr, timing, genlibdb, gdsmerge, lvs,
               rtl_sim, ffgl_sim, bagl_sim, power, summary]:
    graph.connect_by_name(sram, node)

  graph.connect_by_name(sv2v, synth)
  graph.connect_by_name(sv2v, rtl_sim)
  for node in [rtl_sim, ffgl_sim, bagl_sim, power]:
    graph.connect_by_name(testbench, node)
  for node in [synth, pnr, timing, genlibdb, gdsmerge, lvs,
               rtl_sim, ffgl_sim, bagl_sim, power]:
    graph.connect_by_name(utilities, node)

  graph.connect_by_name(synth, pnr)
  graph.connect_by_name(synth, ffgl_sim)
  for node in [timing, genlibdb, gdsmerge, drc, lvs, bagl_sim, power]:
    graph.connect_by_name(pnr, node)
  graph.connect_by_name(gdsmerge, drc)
  graph.connect_by_name(gdsmerge, lvs)
  graph.connect_by_name(bagl_sim, power)

  metric_edges = [
    (synth, 'synthesis-metrics.json', 'synthesis-metrics.json'),
    (pnr, 'pnr-metrics.json', 'pnr-metrics.json'),
    (timing, 'timing-metrics.json', 'timing-metrics.json'),
    (gdsmerge, 'gdsmerge-metrics.json', 'gdsmerge-metrics.json'),
    (drc, 'drc-metrics.json', 'drc-metrics.json'),
    (drc, 'drc-policy.json', 'drc-policy.json'),
    (lvs, 'lvs-metrics.json', 'lvs-metrics.json'),
    (power, 'power.rpt', 'power.rpt'),
    (power, 'activity-source.json', 'activity-source.json'),
    (rtl_sim, 'simulation-report.json', 'rtl-simulation-report.json'),
    (ffgl_sim, 'simulation-report.json', 'ffgl-simulation-report.json'),
    (bagl_sim, 'simulation-report.json', 'bagl-simulation-report.json'),
  ]
  for source, source_name, target_name in metric_edges:
    graph.connect(source.o(source_name), summary.i(target_name))
  graph.connect_by_name(summary, finalize)

  graph.update_params(parameters)
  return graph


if __name__ == '__main__':
  construct()
