#=========================================================================
# construct-open.py
#=========================================================================
# NPU design from MiniNPU Cornell 
#
# Author : Julian Bushlow
# Date   : June 5, 2026
#

import os

from mflowgen.components import Graph, Node

def construct():

  g = Graph()

  #-----------------------------------------------------------------------
  # Parameters
  #-----------------------------------------------------------------------

  adk_name = 'freepdk-45nm'
  adk_view = 'view-tiny'

  parameters = {
    'construct_path' : __file__,
    'design_name'    : 'compute_tile',
    'orfs_platform'  : 'nangate45',
    'clock_period'   : 10.0,
    'adk'            : adk_name,
    'adk_view'       : adk_view,
    # Pick an image from Docker Hub "mflowgen/openroad-flow-scripts-base"
    # - https://hub.docker.com/repository/docker/mflowgen/openroad-flow-scripts-base/general
    #'orfs_image'     : 'mflowgen/openroad-flow-scripts-base:2024-0621-f0caba6',
    'orfs_image'     : 'openroad/orfs:26Q2-436-geb14d768b',

    'orfs_prune_checkpoints' : 1,
    'orfs_delete_flow_dir'   : 1,
  }

  #-----------------------------------------------------------------------
  # Create nodes
  #-----------------------------------------------------------------------

  this_dir  = os.path.dirname( os.path.abspath( __file__ ) )
  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  # ADK node

#  g.set_adk( adk_name )
#  adk = g.get_adk_node()

  # Custom nodes

  design = Node( this_dir + '/orfs-design' )

  # (modified) Default nodes

  common = Node( os.path.join(nodes_dir, 'orfs-common') )
  docker = Node( os.path.join(nodes_dir, 'orfs-docker-setup') )
  synth  = Node( os.path.join(nodes_dir, 'orfs-yosys-synthesis') )
  fplan  = Node( os.path.join(nodes_dir, 'orfs-openroad-floorplan') )
  place  = Node( os.path.join(nodes_dir, 'orfs-openroad-place') )
  cts    = Node( os.path.join(nodes_dir, 'orfs-openroad-cts') )
  route  = Node( os.path.join(nodes_dir, 'orfs-openroad-route') )
  finish = Node( os.path.join(nodes_dir, 'orfs-openroad-finish') )

  #-----------------------------------------------------------------------
  # Graph -- Add nodes
  #-----------------------------------------------------------------------

  g.add_node( common   )
  g.add_node( info     )
  g.add_node( design   )
  g.add_node( docker   )
  g.add_node( synth    )
  g.add_node( fplan    )
  g.add_node( place    )
  g.add_node( cts      )
  g.add_node( route    )
  g.add_node( finish   )

  #-----------------------------------------------------------------------
  # Graph -- Add edges
  #-----------------------------------------------------------------------

  g.connect_by_name( design,  synth  )

  g.connect_by_name( docker,  synth  )
  g.connect_by_name( docker,  fplan  )
  g.connect_by_name( docker,  place  )
  g.connect_by_name( docker,  cts    )
  g.connect_by_name( docker,  route  )
  g.connect_by_name( docker,  finish )

  g.connect_by_name( common, synth  )
  g.connect_by_name( common, fplan  )
  g.connect_by_name( common, place  )
  g.connect_by_name( common, cts    )
  g.connect_by_name( common, route  )
  g.connect_by_name( common, finish )

  g.connect_by_name( synth,   fplan  )
  g.connect_by_name( fplan,   place  )
  g.connect_by_name( place,   cts    )
  g.connect_by_name( cts,     route  )
  g.connect_by_name( route,   finish )

  #-----------------------------------------------------------------------
  # Parameterize
  #-----------------------------------------------------------------------

  g.update_params( parameters )

  return g


if __name__ == '__main__':
  g = construct()
