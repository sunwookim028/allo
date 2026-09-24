"""TinyTPU-isa, shipped T=4 configuration: DC synthesis only.

Relative cell area for comparing our own variants against each other. Memories
are flip-flops (sram_mode='none'), so this is NOT comparable with designs that
used provided SRAM macros (TestNPU-core, allo-mininpu-v2). No P&R, no routed
timing, no power, no DRC/LVS.
"""

import os

from mflowgen.components import Graph, Node

# Override with TINYTPU_RTL to synthesise another variant's rtl/ directory.
DEFAULT_RTL = ('/work/shared/users/phd/sk3463/projects/allo-main/'
               'examples/tinytpu/asic/shipped_t4')


def construct():
  graph = Graph()

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'
  design_path = os.environ.get('TINYTPU_RTL', DEFAULT_RTL)

  parameters = {
    'construct_path': __file__,
    'design_name': 'tinytpu_isa',
    'top_module': 'tinytpu_isa',
    'adk': adk_name,
    'adk_view': adk_view,

    # 3.33 ns is the Vitis target the RTL was emitted at.
    'clock_period': 3.33,
    'clock_port': 'ap_clk',
    'topographical': True,
    'flatten_effort': 3,

    'design_path': os.path.join(design_path, 'rtl'),
    'manifest': os.path.join(design_path, 'sv2v_manifest.f'),
    'sv2v_include_dirs': '.',
    # Vitis emits Verilog-2001; sv2v is unnecessary, and its output puts the
    # top module's (* CORE_GENERATION_INFO *) attribute on the module line,
    # which the collector's top-module check then misses.
    'normalize_rtl': False,

    'sram_mode': 'none',
  }

  # The vendored flow: allo/backend/asic/{nodes,adks} at the repository root,
  # three levels above examples/tinytpu/asic_synthesis/.
  this_dir = os.path.dirname(os.path.abspath(__file__))
  repo = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
  asic_dir = os.environ.get('ALLO_ASIC_FLOW',
                            os.path.join(repo, 'allo', 'backend', 'asic'))
  nodes_dir = os.path.join(asic_dir, 'nodes')
  if not os.path.isdir(nodes_dir):
    raise SystemExit(
      f'no node library at {nodes_dir}. Set ALLO_ASIC_FLOW to a checkout of it, '
      'or run tools/preflight.py to see what is missing.')

  graph.sys_path.append(os.path.join(asic_dir, 'adks'))
  graph.set_adk(adk_name)
  adk = graph.get_adk_node()

  node_names = [
    'sv2v-design-collector',
    'sram-collateral',
    'asic-flow-utilities',
    'synopsys-dc-synthesis',
  ]
  nodes = {name: Node(os.path.join(nodes_dir, name)) for name in node_names}
  for node in nodes.values():
    graph.add_node(node)

  sv2v = nodes['sv2v-design-collector']
  sram = nodes['sram-collateral']
  utilities = nodes['asic-flow-utilities']
  synth = nodes['synopsys-dc-synthesis']

  graph.connect_by_name(adk, synth)
  graph.connect_by_name(sram, synth)
  graph.connect_by_name(sv2v, synth)
  graph.connect_by_name(utilities, synth)

  graph.update_params(parameters)

  return graph


if __name__ == '__main__':
  g = construct()
