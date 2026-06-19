#=========================================================================
# construct-openram-probe.py
#=========================================================================
# Minimal flow for probing OpenRAM SRAM generation.

import os

from mflowgen.components import Graph, Node


def construct():

  g = Graph()

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'
  openram_python = os.environ.get('OPENRAM_PYTHON', 'python')

  parameters = {
    'construct_path'    : __file__,
    'design_name'       : 'sram_probe',
    'adk'               : adk_name,
    'adk_view'          : adk_view,

    # OpenRAM params
    'sram_manifest'    : 'rtl/sram_manifest_1rw.yml',
    'python_bin'       : openram_python,
    'openram_script'   : '',
    'tech_name'        : 'freepdk45',
    'process_corner'   : 'TT',
    'supply_voltage'   : 1.1,
    'temperature'      : 25,
    'check_lvsdrc'     : False,
    'route_supplies'   : True,
    'analytical_delay' : True,
    'use_sram_cache'   : False,
    'sram_cache_path'  : '',
  }

  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  info    = Node('info', default=True)
  openram = Node(os.path.join(nodes_dir, 'openram-sram-generation'))

  g.set_adk(adk_name)

  g.add_node(info)
  g.add_node(openram)

  g.update_params(parameters)

  return g


if __name__ == '__main__':
  g = construct()
