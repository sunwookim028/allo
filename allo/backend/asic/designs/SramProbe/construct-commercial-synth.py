#=========================================================================
# construct-commercial-synth.py
#=========================================================================
# Small synthesis/link probe for an OpenRAM-generated SRAM macro.

import os

from mflowgen.components import Graph, Node


def construct():

  g = Graph()

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'

  this_dir = os.path.dirname(os.path.abspath(__file__))

  parameters = {
    'construct_path'      : __file__,
    'design_name'         : 'sram_probe',
    'clock_period'        : 10.0,
    'adk'                 : adk_name,
    'adk_view'            : adk_view,
    'flatten_effort'      : 0,
    'topographical'       : True,
    'top_module'          : 'sram_probe',
    'design_path'         : os.path.join(this_dir, 'rtl'),
    'sv2v_include_dirs'   : '.',

    # OpenRAM params
    'sram_manifest'       : 'rtl/sram_manifest.yml',
    'python_bin'          : 'python',
    'openram_script'      : '',
    'tech_name'           : 'freepdk45',
    'process_corner'      : 'TT',
    'supply_voltage'      : 1.1,
    'temperature'         : 25,
    'check_lvsdrc'        : False,
    'route_supplies'      : True,
    'analytical_delay'    : True,
    'use_sram_cache'      : False,
    'sram_cache_path'     : '',
  }

  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  g.set_adk(adk_name)
  adk = g.get_adk_node()

  info        = Node('info', default=True)
  openram     = Node(os.path.join(nodes_dir, 'openram-sram-generation'))
  sv2v        = Node(os.path.join(nodes_dir, 'sv2v-design-collector'))
  constraints = Node(this_dir + '/constraints')
  synth       = Node(os.path.join(nodes_dir, 'synopsys-dc-synthesis'))

  g.add_node(info)
  g.add_node(openram)
  g.add_node(sv2v)
  g.add_node(constraints)
  g.add_node(synth)

  g.connect_by_name(adk, synth)
  g.connect_by_name(openram, synth)
  g.connect_by_name(sv2v, synth)
  g.connect_by_name(constraints, synth)

  g.update_params(parameters)

  return g


if __name__ == '__main__':
  g = construct()
