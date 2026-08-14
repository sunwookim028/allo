#=========================================================================
# pnr.tcl
#=========================================================================
# Single-node Cadence Innovus PNR flow.

if {![info exists env(design_name)] || $env(design_name) eq "" || $env(design_name) eq "undefined"} {
  if {[info exists env(top_module)] && $env(top_module) ne "" && $env(top_module) ne "undefined"} {
    set env(design_name) $env(top_module)
  }
}

if {![info exists env(design_name)] || $env(design_name) eq "" || $env(design_name) eq "undefined"} {
  error "Missing required parameter: design_name"
}

set design_name $env(design_name)

set results_dir     results
set reports_dir     reports
set checkpoints_dir checkpoints

file mkdir $results_dir
file mkdir $reports_dir
file mkdir $checkpoints_dir
file mkdir $checkpoints_dir/LEC

setDistributeHost -local

source inputs/adk/adk.tcl

proc env_list {name} {
  return [split [string map {, " "} $::env($name)]]
}

if {![info exists env(process_node)]} {
  if {[info exists ADK_PROCESS]} {
    set env(process_node) $ADK_PROCESS
  } else {
    set env(process_node) 45
  }
}
if {![info exists env(max_route_layer)]} {
  if {[info exists ADK_MAX_ROUTING_LAYER_INNOVUS]} {
    set env(max_route_layer) $ADK_MAX_ROUTING_LAYER_INNOVUS
  } else {
    set env(max_route_layer) 7
  }
}
if {![info exists env(base_layer_idx)]} {
  if {[info exists ADK_BASE_LAYER_IDX]} {
    set env(base_layer_idx) $ADK_BASE_LAYER_IDX
  } else {
    set env(base_layer_idx) 0
  }
}
if {![info exists env(gds_stream_out_units)]} {
  set env(gds_stream_out_units) 10000
}
if {![string is integer -strict $env(gds_stream_out_units)] || $env(gds_stream_out_units) <= 0} {
  error "gds_stream_out_units must be a positive integer"
}
if {![info exists env(pin_layer_offset)]} {
  set env(pin_layer_offset) 3
}
if {![info exists env(pin_vertical_layer_offset)]} {
  set env(pin_vertical_layer_offset) 4
}
if {![info exists env(pin_secondary_layer_offset)]} {
  set env(pin_secondary_layer_offset) 5
}
if {![info exists env(pin_min_pitch_multiplier)]} {
  set env(pin_min_pitch_multiplier) 2.0
}
if {![info exists env(pin_primary_depth_width_multiplier)]} {
  set env(pin_primary_depth_width_multiplier) 3.0
}
if {![info exists env(pin_other_depth_width_multiplier)]} {
  set env(pin_other_depth_width_multiplier) 2.0
}
if {![info exists env(pin_corner_keepout)]} {
  set env(pin_corner_keepout) 3.0
}
if {![info exists env(pin_primary_fraction)]} {
  set env(pin_primary_fraction) 0.75
}
if {![info exists env(pin_spread_multiple_stream_groups)]} {
  set env(pin_spread_multiple_stream_groups) true
}
if {![info exists env(pin_stream_group_min_width)]} {
  set env(pin_stream_group_min_width) 16
}
if {![info exists env(top_layer_obs_grid_tracks)]} {
  set env(top_layer_obs_grid_tracks) 4
}
if {![info exists env(top_layer_obs_pin_grid_tracks)]} {
  set env(top_layer_obs_pin_grid_tracks) 1
}
if {![info exists env(top_layer_obs_spacing_tracks)]} {
  set env(top_layer_obs_spacing_tracks) 1
}
if {![string is double -strict $env(pin_min_pitch_multiplier)] ||
    $env(pin_min_pitch_multiplier) < 1.0} {
  error "pin_min_pitch_multiplier must be a number greater than or equal to one"
}
foreach parameter {pin_primary_depth_width_multiplier pin_other_depth_width_multiplier} {
  if {![string is double -strict $env($parameter)] || $env($parameter) < 1.0} {
    error "$parameter must be a number greater than or equal to one"
  }
}
if {![string is integer -strict $env(pin_stream_group_min_width)] ||
    $env(pin_stream_group_min_width) < 1} {
  error "pin_stream_group_min_width must be a positive integer"
}
if {![string is integer -strict $env(top_layer_obs_grid_tracks)] ||
    $env(top_layer_obs_grid_tracks) < 1} {
  error "top_layer_obs_grid_tracks must be a positive integer"
}
if {![string is integer -strict $env(top_layer_obs_pin_grid_tracks)] ||
    $env(top_layer_obs_pin_grid_tracks) < 1 ||
    $env(top_layer_obs_grid_tracks) % $env(top_layer_obs_pin_grid_tracks) != 0} {
  error "top_layer_obs_pin_grid_tracks must be a positive divisor of top_layer_obs_grid_tracks"
}
if {![string is integer -strict $env(top_layer_obs_spacing_tracks)] ||
    $env(top_layer_obs_spacing_tracks) < 0} {
  error "top_layer_obs_spacing_tracks must be a non-negative integer"
}
if {![info exists env(power_mesh_bot_layer)]} {
  if {[info exists ADK_POWER_MESH_BOT_LAYER]} {
    set env(power_mesh_bot_layer) $ADK_POWER_MESH_BOT_LAYER
  } else {
    set env(power_mesh_bot_layer) 6
  }
}
if {![info exists env(power_mesh_top_layer)]} {
  if {[info exists ADK_POWER_MESH_TOP_LAYER]} {
    set env(power_mesh_top_layer) $ADK_POWER_MESH_TOP_LAYER
  } else {
    set env(power_mesh_top_layer) 7
  }
}
if {![info exists env(local_cpus)]} {
  set env(local_cpus) 16
}
if {![info exists env(postroute_max_local_cpus)]} {
  set env(postroute_max_local_cpus) 8
}
if {![info exists env(primary_power_net)]} {
  set env(primary_power_net) VDD
}
if {![info exists env(primary_ground_net)]} {
  set env(primary_ground_net) VSS
}
if {![info exists env(power_nets)]} {
  set env(power_nets) "VDD,VNW,VDDPST,POC,VDDCE,VDDPE"
}
if {![info exists env(ground_nets)]} {
  set env(ground_nets) "VSS,VPW,VSSPST,VSSE"
}
if {![info exists env(power_pin_names)]} {
  set env(power_pin_names) "VDD"
}
if {![info exists env(ground_pin_names)]} {
  set env(ground_pin_names) "VSS"
}
if {![info exists env(adk_tech_lef)]} {
  set env(adk_tech_lef) inputs/adk/rtk-tech.lef
}
if {![info exists env(adk_stdcell_lef)]} {
  set env(adk_stdcell_lef) inputs/adk/stdcells.lef
}
if {![info exists env(adk_gds_layer_map)]} {
  set env(adk_gds_layer_map) inputs/adk/rtk-stream-out.map
}
if {![info exists env(adk_qrc_lef_map)]} {
  set env(adk_qrc_lef_map) inputs/adk/pdk-qrc-lef.map
}
if {![info exists env(core_density_target)]} {
  set env(core_density_target) 0.7
}
if {![info exists env(useful_skew)]} {
  set env(useful_skew) true
}
if {![info exists env(useful_skew_ccopt_effort)]} {
  set env(useful_skew_ccopt_effort) standard
}
if {![info exists env(cell_padding)]} {
  set env(cell_padding) 0
}
if {![info exists env(macro_hold_target_slack)]} {
  set env(macro_hold_target_slack) 0.05
}
if {![info exists env(macro_setup_target_slack)]} {
  set env(macro_setup_target_slack) 0.05
}
if {![info exists env(postroute_optimization_passes)]} {
  set env(postroute_optimization_passes) 2
}
if {![string is integer -strict $env(postroute_optimization_passes)] || $env(postroute_optimization_passes) < 1} {
  error "postroute_optimization_passes must be a positive integer"
}
if {![info exists env(signoff_engine)]} {
  set env(signoff_engine) false
}
if {![info exists env(stop_after_step)]} {
  set env(stop_after_step) none
}
if {![info exists env(floorplan_mode)]} {
  set env(floorplan_mode) auto
}
if {![info exists env(floorplan_aspect_ratio)]} {
  set env(floorplan_aspect_ratio) 1.0
}
if {![info exists env(floorplan_width)]} {
  set env(floorplan_width) ""
}
if {![info exists env(floorplan_height)]} {
  set env(floorplan_height) ""
}
if {![info exists env(edge_mesh_margin)]} {
  set env(edge_mesh_margin) 2.0
}
if {![info exists env(edge_mesh_stripe_width_multiplier)]} {
  set env(edge_mesh_stripe_width_multiplier) 8
}
if {![info exists env(edge_mesh_stripe_spacing_multiplier)]} {
  set env(edge_mesh_stripe_spacing_multiplier) 8
}
if {![info exists env(macro_halo)]} {
  set env(macro_halo) 2.0
}
if {![info exists env(macro_pg_resource_util)]} {
  set env(macro_pg_resource_util) 0.2
}
if {![info exists env(macro_forbidden_space_to_macro)]} {
  set env(macro_forbidden_space_to_macro) "20,20"
}
if {![info exists env(macro_min_space_to_core)]} {
  set env(macro_min_space_to_core) "30,30"
}
if {![info exists env(macro_corner_keepout)]} {
  set env(macro_corner_keepout) "5,5"
}
if {![info exists env(macro_edge_keepout)]} {
  set env(macro_edge_keepout) 30.0
}
if {![info exists env(macro_group_spacing)]} {
  set env(macro_group_spacing) 10.0
}
if {![info exists env(macro_group_max_depth)]} {
  set env(macro_group_max_depth) 2
}
if {![info exists env(well_tap_cell)]} {
  if {[info exists ADK_WELL_TAP_CELL]} {
    set env(well_tap_cell) $ADK_WELL_TAP_CELL
  } else {
    set env(well_tap_cell) ""
  }
}
if {![info exists env(well_tap_interval)]} {
  if {[info exists ADK_WELL_TAP_INTERVAL]} {
    set env(well_tap_interval) $ADK_WELL_TAP_INTERVAL
  } else {
    set env(well_tap_interval) 120
  }
}
if {![info exists env(lvs_exclude_cell_list)]} {
  if {[info exists ADK_LVS_EXCLUDE_CELL_LIST]} {
    set env(lvs_exclude_cell_list) [string map {" " ","} $ADK_LVS_EXCLUDE_CELL_LIST]
  } else {
    set env(lvs_exclude_cell_list) ""
  }
}
if {![info exists env(virtuoso_exclude_cell_list)]} {
  if {[info exists ADK_VIRTUOSO_EXCLUDE_CELL_LIST]} {
    set env(virtuoso_exclude_cell_list) [string map {" " ","} $ADK_VIRTUOSO_EXCLUDE_CELL_LIST]
  } else {
    set env(virtuoso_exclude_cell_list) ""
  }
}

set vars(design)          $design_name
set vars(results_dir)     $results_dir
set vars(rpt_dir)         $reports_dir
set vars(max_route_layer) $env(max_route_layer)
set vars(local_cpus)      $env(local_cpus)

set primary_power_net  $env(primary_power_net)
set primary_ground_net $env(primary_ground_net)
set base_layer_idx     $env(base_layer_idx)
set pmesh_bot          $env(power_mesh_bot_layer)
set pmesh_top          $env(power_mesh_top_layer)
set pwr_net_list       [list $primary_power_net $primary_ground_net]
set pwr_net_list_rev   [list $primary_ground_net $primary_power_net]
set power_nets         [env_list power_nets]
set ground_nets        [env_list ground_nets]
set power_pin_names    [env_list power_pin_names]
set ground_pin_names   [env_list ground_pin_names]
set macro_forbidden_space_to_macro [env_list macro_forbidden_space_to_macro]
set macro_min_space_to_core        [env_list macro_min_space_to_core]
set macro_corner_keepout           [env_list macro_corner_keepout]
set lvs_exclude_cell_list          [env_list lvs_exclude_cell_list]
set virtuoso_exclude_cell_list     [env_list virtuoso_exclude_cell_list]

setMultiCpuUsage -localCpu $vars(local_cpus)

set valid_stop_steps {none floorplan power place cts route}
if {[lsearch -exact $valid_stop_steps $env(stop_after_step)] < 0} {
  error "Unsupported stop_after_step '$env(stop_after_step)'; expected one of: $valid_stop_steps"
}

proc save_design_checkpoint {} {
  global checkpoints_dir design_name

  file mkdir $checkpoints_dir
  file mkdir $checkpoints_dir/LEC
  file mkdir $checkpoints_dir/design.checkpoint

  saveDesign $checkpoints_dir/design.checkpoint/save.enc -compress
}

proc maybe_stop_after {phase} {
  if {$::env(stop_after_step) == $phase} {
    puts "Info: stop_after_step=$phase requested; saving checkpoint and exiting."
    save_design_checkpoint
    exit 0
  }
}

proc report_metrics {phase} {
  global reports_dir

  catch {create_snapshot -name $phase -categories design}
  catch {report_metric -file $reports_dir/${phase}.metrics.html -format vivid}
}

#-------------------------------------------------------------------------
# Read Design / MMMC
#-------------------------------------------------------------------------

set init_layout_view ""
set init_abstract_name ""
set sram_lef_files [lsort [glob -nocomplain inputs/srams/*/*.lef]]

set init_verilog "./inputs/design.v"
set init_mmmc_file "scripts/mmmc.tcl"
set init_lef_file [concat \
  [list $env(adk_tech_lef) $env(adk_stdcell_lef) $env(adk_tech_lef) $env(adk_stdcell_lef)] \
  $sram_lef_files \
]
set init_top_cell $design_name
set init_gnd_net $ground_nets
set init_pwr_net $power_nets

set_db init_no_new_assigns true

if {[file exists $env(adk_qrc_lef_map)]} {
  setExtractRCMode -lefTechFileMap $env(adk_qrc_lef_map)
}

init_design

set_power_analysis_mode -analysis_view analysis_default
setDesignMode -topRoutingLayer $vars(max_route_layer)
setDesignMode -process $env(process_node) -powerEffort high

#-------------------------------------------------------------------------
# Init / Floorplan / Macros
#-------------------------------------------------------------------------

setOptMode -timeDesignCompressReports false
if {[info exists ADK_HOLD_FIXING_CELL_LIST] && $ADK_HOLD_FIXING_CELL_LIST ne ""} {
  setOptMode -holdFixingCells $ADK_HOLD_FIXING_CELL_LIST
}

if {[info exists ADK_DONT_USE_CELL_LIST]} {
  foreach_in_collection cell [get_lib_cells $ADK_DONT_USE_CELL_LIST] {
    setDontUse [get_object_name $cell] true
  }
}

set core_aspect_ratio   $env(floorplan_aspect_ratio)
set core_density_target $env(core_density_target)

set M1_min_width   [dbGet [dbGetLayerByZ 1].minWidth]
set M1_min_spacing [dbGet [dbGetLayerByZ 1].minSpacing]

# A reusable PE exports sparse PG stripes at its boundary.  It deliberately
# does not reserve the large margins required by a private core ring.
set core_margin_t $env(edge_mesh_margin)
set core_margin_b $env(edge_mesh_margin)
set core_margin_r $env(edge_mesh_margin)
set core_margin_l $env(edge_mesh_margin)

if {$env(floorplan_mode) == "auto"} {
  floorPlan -r $core_aspect_ratio $core_density_target \
    $core_margin_l $core_margin_b $core_margin_r $core_margin_t
} elseif {$env(floorplan_mode) == "fixed"} {
  if {$env(floorplan_width) == "" || $env(floorplan_height) == ""} {
    error "floorplan_mode=fixed requires floorplan_width and floorplan_height"
  }

  floorPlan -d $env(floorplan_width) $env(floorplan_height) \
    $core_margin_l $core_margin_b $core_margin_r $core_margin_t
} else {
  error "Unsupported floorplan_mode '$env(floorplan_mode)'; expected 'auto' or 'fixed'"
}

setFlipping s

set macro_halo $env(macro_halo)
source scripts/place-grouped-macros.tcl
set has_hard_macros [place_grouped_macros]
set blocks [dbGet top.insts.cell.baseClass block -p2]

if {   [info exists ADK_END_CAP_CELL_LEFT]
    && [expr {$ADK_END_CAP_CELL_LEFT ne ""}]
    && [info exists ADK_END_CAP_CELL_RIGHT]
    && [expr {$ADK_END_CAP_CELL_RIGHT ne ""}] } {
  setEndCapMode -rightEdge $ADK_END_CAP_CELL_LEFT
  setEndCapMode -leftEdge  $ADK_END_CAP_CELL_RIGHT
  addEndCap -prefix ENDCAP
}

if {$has_hard_macros} {
  foreach inst $blocks {
    if {[dbGet $inst.isPhysOnly]} {
      continue
    }

    set box [dbGet $inst.box]
    if {[llength $box] == 1} {
      set box [lindex $box 0]
    }

    set llx [expr {[lindex $box 0] - $macro_halo}]
    set lly [expr {[lindex $box 1] - $macro_halo}]
    set urx [expr {[lindex $box 2] + $macro_halo}]
    set ury [expr {[lindex $box 3] + $macro_halo}]

    cutRow -area [list $llx $lly $urx $ury]
  }
}

if {$env(well_tap_cell) ne ""} {
  addWellTap -cell [list $env(well_tap_cell)] \
    -prefix WELLTAP \
    -cellInterval $env(well_tap_interval)

  verifyWellTap -cells [list $env(well_tap_cell)] \
    -report reports/welltap.rpt \
    -rule [expr {$env(well_tap_interval)/2}]
}

#-------------------------------------------------------------------------
# Pin Assignment / Path Groups / Init Reports
#-------------------------------------------------------------------------

set ports_layer [expr {$base_layer_idx + $env(pin_layer_offset)}]
set vertical_ports_layer [expr {$base_layer_idx + $env(pin_vertical_layer_offset)}]
set secondary_ports_layer [expr {$base_layer_idx + $env(pin_secondary_layer_offset)}]
if {$vertical_ports_layer > $vars(max_route_layer)} {
  error "Vertical signal-pin layer $vertical_ports_layer exceeds max_route_layer=$vars(max_route_layer)"
}
if {$secondary_ports_layer > $vars(max_route_layer)} {
  error "Secondary signal-pin layer $secondary_ports_layer exceeds max_route_layer=$vars(max_route_layer)"
}
set primary_pin_layer_obj [dbGetLayerByZ $ports_layer]
set vertical_pin_layer_obj [dbGetLayerByZ $vertical_ports_layer]
set secondary_pin_layer_obj [dbGetLayerByZ $secondary_ports_layer]
if {$primary_pin_layer_obj == 0 || $vertical_pin_layer_obj == 0 ||
    $secondary_pin_layer_obj == 0} {
  error "Signal-pin layer selection is not present in the loaded technology: horizontal=$ports_layer vertical=$vertical_ports_layer secondary_horizontal=$secondary_ports_layer"
}
set primary_pin_direction [dbGet $primary_pin_layer_obj.direction]
set vertical_pin_direction [dbGet $vertical_pin_layer_obj.direction]
set secondary_pin_direction [dbGet $secondary_pin_layer_obj.direction]
if {$primary_pin_direction ne $secondary_pin_direction} {
  error "East/west signal-pin layers must have the same preferred routing direction: layer $ports_layer is $primary_pin_direction but layer $secondary_ports_layer is $secondary_pin_direction"
}
if {[string equal -nocase $primary_pin_direction $vertical_pin_direction]} {
  error "North/south signal-pin layer $vertical_ports_layer must be orthogonal to east/west layer $ports_layer"
}
if {$env(pin_primary_fraction) <= 0.0 || $env(pin_primary_fraction) >= 1.0} {
  error "pin_primary_fraction must be strictly between zero and one"
}
if {![file exists inputs/pin-intent.tcl]} {
  error "Missing manifest-derived pin intent: inputs/pin-intent.tcl"
}
source inputs/pin-intent.tcl
set all_ports [lsort -dictionary [dbGet top.terms.name]]

# The pre-synthesis manifest uses one logical name for a vector port, while
# Innovus exposes a synthesized vector as individual terminals such as
# data[0], data[1], ... . Resolve an exact scalar first, then expand a logical
# bus name into all terminals with the corresponding bracketed prefix.
proc resolve_manifest_port {logical_port all_ports} {
  if {[lsearch -exact $all_ports $logical_port] >= 0} {
    return [list $logical_port]
  }
  set bus_prefix "${logical_port}\["
  set resolved {}
  foreach physical_port $all_ports {
    if {[string first $bus_prefix $physical_port] == 0} {
      lappend resolved $physical_port
    }
  }
  return [lsort -dictionary $resolved]
}

# Resolve a logical group as a unit. Keeping FIFO bundles, AXI channels and
# controls together makes the abstract interface regular. A group is moved to
# the secondary layer only when doing so best approaches the requested primary
# layer fraction; individual buses are expanded only after that decision.
proc resolve_manifest_group {logical_group all_ports} {
  set result {}
  foreach logical_port $logical_group {
    set resolved [resolve_manifest_port $logical_port $all_ports]
    if {[llength $resolved] == 0} {
      error "Pin intent references unknown top-level scalar or bus '$logical_port'"
    }
    set result [concat $result $resolved]
  }
  return $result
}

# Return the routing pitch declared by the loaded ADK for an Innovus layer.
proc pin_layer_pitch {layer_obj} {
  set candidates {}
  foreach attribute {pitchX pitchY} {
    set value [dbGet $layer_obj.$attribute]
    if {[llength $value] > 0} {
      set value [lindex $value 0]
      if {[string is double -strict $value] && $value > 0.0} {
        lappend candidates $value
      }
    }
  }
  if {[llength $candidates] == 0} {
    error "Loaded ADK does not provide a positive routing pitch for layer [dbGet $layer_obj.name]"
  }
  return [lindex [lsort -real $candidates] 0]
}

# Return the normal routing width. This remains the edge-parallel pin
# dimension; only depth into the macro is enlarged.
proc pin_layer_width {layer_obj} {
  foreach attribute {width minWidth} {
    set value [dbGet $layer_obj.$attribute]
    if {[llength $value] > 0} {
      set value [lindex $value 0]
      if {[string is double -strict $value] && $value > 0.0} {
        return $value
      }
    }
  }
  error "Loaded ADK does not provide a positive routing width for layer [dbGet $layer_obj.name]"
}

proc pin_edge_usable_length {side keepout} {
  set box [dbGet top.fPlan.box]
  if {[llength $box] == 1} {
    set box [lindex $box 0]
  }
  set width [expr {[lindex $box 2] - [lindex $box 0]}]
  set height [expr {[lindex $box 3] - [lindex $box 1]}]
  if {$side eq "TOP" || $side eq "BOTTOM"} {
    set length $width
  } else {
    set length $height
  }
  set usable [expr {$length - 2.0 * $keepout}]
  if {$usable < 0.0} {
    error "pin_corner_keepout=$keepout leaves no usable length on $side edge"
  }
  return $usable
}

# N pins require N-1 pitch intervals. Capacity is therefore floor(L/P)+1.
proc pin_layer_capacity {side layer_obj keepout pitch_multiplier} {
  set usable [pin_edge_usable_length $side $keepout]
  set adk_pitch [pin_layer_pitch $layer_obj]
  set required_pitch [expr {$adk_pitch * $pitch_multiplier}]
  set capacity [expr {int(floor($usable / $required_pitch)) + 1}]
  return [list $capacity $adk_pitch $required_pitch $usable]
}

# Pack complete semantic groups onto two layers. A small subset-sum search
# selects the primary-layer groups closest to the requested fraction while
# respecting both physical capacities. No bus or handshake group is divided.
proc pack_pin_groups_by_capacity {
  groups all_ports primary_capacity secondary_capacity fraction
  spread_multiple_streams stream_group_min_width
} {
  set resolved_groups {}
  set widths {}
  set total 0
  foreach group $groups {
    set resolved [resolve_manifest_group $group $all_ports]
    set width [llength $resolved]
    lappend resolved_groups $resolved
    lappend widths $width
    incr total $width
  }
  set large_group_count 0
  foreach width $widths {
    if {$width >= $stream_group_min_width} {
      incr large_group_count
    }
  }
  set require_secondary_large_group [expr {
    $spread_multiple_streams && $large_group_count >= 2
  }]
  if {$total <= $primary_capacity && !$require_secondary_large_group} {
    set primary {}
    foreach resolved $resolved_groups {
      set primary [concat $primary $resolved]
    }
    return [list $primary {} $widths]
  }
  if {$total > $primary_capacity + $secondary_capacity} {
    error "Pin groups require $total slots but the two layers provide only [expr {$primary_capacity + $secondary_capacity}]"
  }

  # Map reachable primary counts to the group indices assigned there.
  set states [dict create 0 {}]
  for {set index 0} {$index < [llength $widths]} {incr index} {
    set width [lindex $widths $index]
    set next $states
    dict for {count indices} $states {
      set candidate [expr {$count + $width}]
      if {$candidate <= $primary_capacity && ![dict exists $next $candidate]} {
        dict set next $candidate [concat $indices [list $index]]
      }
    }
    set states $next
  }

  set target [expr {$total * $fraction}]
  set best_count -1
  set best_score 1.0e30
  dict for {count indices} $states {
    set secondary_count [expr {$total - $count}]
    if {$secondary_count > $secondary_capacity} {
      continue
    }
    if {$require_secondary_large_group} {
      set secondary_has_large_group false
      for {set index 0} {$index < [llength $widths]} {incr index} {
        if {[lsearch -exact $indices $index] < 0 &&
            [lindex $widths $index] >= $stream_group_min_width} {
          set secondary_has_large_group true
          break
        }
      }
      if {!$secondary_has_large_group} {
        continue
      }
    }
    set score [expr {abs($count - $target)}]
    if {$score < $best_score || ($score == $best_score && $count > $best_count)} {
      set best_score $score
      set best_count $count
    }
  }
  if {$best_count < 0} {
    error "Whole pin groups with widths {$widths} cannot fit capacities primary=$primary_capacity secondary=$secondary_capacity"
  }

  set primary_indices [dict get $states $best_count]
  set primary {}
  set secondary {}
  for {set index 0} {$index < [llength $resolved_groups]} {incr index} {
    set resolved [lindex $resolved_groups $index]
    if {[lsearch -exact $primary_indices $index] >= 0} {
      set primary [concat $primary $resolved]
    } else {
      set secondary [concat $secondary $resolved]
    }
  }
  return [list $primary $secondary $widths]
}

# Place pins only on the middle portion of an edge.  SIDE spreading uses the
# complete block edge and can put pins directly in a corner, where a route on
# the neighboring edge (especially ap_clk) has very little legal access room.
proc edit_pins_with_corner_keepout {
  pins layer layer_obj side keepout depth_width_multiplier
} {
  if {[llength $pins] == 0} {
    return
  }
  if {$keepout < 0.0} {
    error "pin_corner_keepout must be nonnegative"
  }
  set box [dbGet top.fPlan.box]
  if {[llength $box] == 1} {
    set box [lindex $box 0]
  }
  set llx [lindex $box 0]
  set lly [lindex $box 1]
  set urx [lindex $box 2]
  set ury [lindex $box 3]
  switch -- $side {
    TOP {
      set start [list [expr {$llx + $keepout}] $ury]
      set end   [list [expr {$urx - $keepout}] $ury]
    }
    BOTTOM {
      # Innovus interprets RANGE endpoints as a directed walk around the block
      # boundary. Reverse the bottom edge so the range stays on that edge
      # instead of taking the long path through the other three sides.
      set start [list [expr {$urx - $keepout}] $lly]
      set end   [list [expr {$llx + $keepout}] $lly]
    }
    RIGHT {
      # Use the same perimeter direction as TOP and LEFT.
      set start [list $urx [expr {$ury - $keepout}]]
      set end   [list $urx [expr {$lly + $keepout}]]
    }
    LEFT {
      set start [list $llx [expr {$lly + $keepout}]]
      set end   [list $llx [expr {$ury - $keepout}]]
    }
    default {
      error "Unsupported pin side '$side'"
    }
  }
  set pin_width [pin_layer_width $layer_obj]
  set pin_depth [expr {$pin_width * $depth_width_multiplier}]
  editPin -layer $layer -pin $pins -spreadType RANGE -start $start -end $end \
    -pinWidth $pin_width -pinDepth $pin_depth -fixedPin true
}

set assigned_ports {}
set pin_assignment_report [open reports/pin-assignment.rpt w]

# Allocate splittable non-neighbor data buses only after floorplanning, when
# real edge capacities are known. Start with the two least-loaded edges, then
# use three or four only when the smaller subset cannot fit. Data slices remain
# contiguous; handshake pins remain one atomic group.
if {[info exists allo_asic_non_neighbor_split_bundles]} {
  array set split_capacity {}
  array set split_load {}
  foreach {compass innovus_side} {N TOP S BOTTOM E RIGHT W LEFT} {
    if {$compass eq "N" || $compass eq "S"} {
      set capacity_data [pin_layer_capacity $innovus_side \
        $vertical_pin_layer_obj $env(pin_corner_keepout) \
        $env(pin_min_pitch_multiplier)]
      set split_capacity($compass) [lindex $capacity_data 0]
    } else {
      set primary_data [pin_layer_capacity $innovus_side \
        $primary_pin_layer_obj $env(pin_corner_keepout) \
        $env(pin_min_pitch_multiplier)]
      set secondary_data [pin_layer_capacity $innovus_side \
        $secondary_pin_layer_obj $env(pin_corner_keepout) \
        $env(pin_min_pitch_multiplier)]
      set split_capacity($compass) [expr {
        [lindex $primary_data 0] + [lindex $secondary_data 0]
      }]
    }
    set logical_name allo_asic_signal_pins_${compass}
    set split_load($compass) 0
    foreach logical_port [set $logical_name] {
      incr split_load($compass) [llength [resolve_manifest_port $logical_port $all_ports]]
    }
  }

  foreach bundle $allo_asic_non_neighbor_split_bundles {
    lassign $bundle data_port data_width handshake_ports
    set ordered_sides {}
    set remaining_sides {N S E W}
    while {[llength $remaining_sides] > 0} {
      set best_side [lindex $remaining_sides 0]
      foreach side [lrange $remaining_sides 1 end] {
        set best_ratio [expr {double($split_load($best_side)) / $split_capacity($best_side)}]
        set side_ratio [expr {double($split_load($side)) / $split_capacity($side)}]
        if {$side_ratio < $best_ratio ||
            ($side_ratio == $best_ratio && [string compare $side $best_side] < 0)} {
          set best_side $side
        }
      }
      lappend ordered_sides $best_side
      set index [lsearch -exact $remaining_sides $best_side]
      set remaining_sides [lreplace $remaining_sides $index $index]
    }

    set chosen_count 0
    for {set side_count 2} {$side_count <= 4} {incr side_count} {
      set selected [lrange $ordered_sides 0 [expr {$side_count - 1}]]
      array unset trial_load
      array unset trial_groups
      foreach side {N S E W} {
        set trial_load($side) $split_load($side)
        set trial_groups($side) {}
      }
      set quotient [expr {$data_width / $side_count}]
      set remainder [expr {$data_width % $side_count}]
      set bit 0
      for {set slice 0} {$slice < $side_count} {incr slice} {
        set side [lindex $selected $slice]
        set slice_width [expr {$quotient + ($slice < $remainder ? 1 : 0)}]
        set pins {}
        for {set offset 0} {$offset < $slice_width} {incr offset} {
          if {$data_width == 1} {
            lappend pins $data_port
          } else {
            lappend pins "${data_port}\[$bit\]"
          }
          incr bit
        }
        if {[llength $pins] > 0} {
          lappend trial_groups($side) $pins
          incr trial_load($side) [llength $pins]
        }
      }
      if {[llength $handshake_ports] > 0} {
        set handshake_side [lindex $selected 0]
        foreach side [lrange $selected 1 end] {
          set handshake_ratio [expr {
            double($trial_load($handshake_side)) / $split_capacity($handshake_side)
          }]
          set side_ratio [expr {
            double($trial_load($side)) / $split_capacity($side)
          }]
          if {$side_ratio < $handshake_ratio} {
            set handshake_side $side
          }
        }
        lappend trial_groups($handshake_side) $handshake_ports
        incr trial_load($handshake_side) [llength $handshake_ports]
      }
      set fits true
      foreach side $selected {
        if {$trial_load($side) > $split_capacity($side)} {
          set fits false
        }
      }
      if {$fits} {
        set chosen_count $side_count
        foreach side {N S E W} {
          set split_load($side) $trial_load($side)
          set pin_variable allo_asic_signal_pins_${side}
          set group_variable allo_asic_signal_pin_groups_${side}
          foreach group $trial_groups($side) {
            set $pin_variable [concat [set $pin_variable] $group]
            lappend $group_variable $group
          }
        }
        puts $pin_assignment_report \
          "SPLIT $data_port width=$data_width sides=[join $selected ,] handshake={[join $handshake_ports { }]}"
        break
      }
    }
    if {$chosen_count == 0} {
      close $pin_assignment_report
      error "Non-neighbor stream $data_port cannot fit across all four macro edges"
    }
  }
}

setPinAssignMode -pinEditInBatch true
foreach {compass innovus_side} {N TOP S BOTTOM E RIGHT W LEFT} {
  set variable_name allo_asic_signal_pins_${compass}
  if {![info exists $variable_name]} {
    error "Pin intent does not define $variable_name"
  }
  set logical_side_ports [set $variable_name]
  set side_ports {}
  foreach logical_port $logical_side_ports {
    set resolved_ports [resolve_manifest_port $logical_port $all_ports]
    if {[llength $resolved_ports] == 0} {
      close $pin_assignment_report
      error "Pin intent references unknown top-level scalar or bus '$logical_port'"
    }
    puts $pin_assignment_report \
      "$compass $logical_port -> [join $resolved_ports { }]"
    foreach port $resolved_ports {
      if {[lsearch -exact $assigned_ports $port] >= 0} {
        close $pin_assignment_report
        error "Pin intent assigns synthesized terminal '$port' more than once"
      }
      lappend assigned_ports $port
      lappend side_ports $port
    }
  }
  set groups_variable_name allo_asic_signal_pin_groups_${compass}
  if {$compass eq "N" || $compass eq "S"} {
    # Pins leave a north/south boundary vertically, so keep every complete
    # group on the ADK's vertical-preferred signal layer (M4 in FreePDK45).
    set vertical_capacity_data [pin_layer_capacity $innovus_side \
      $vertical_pin_layer_obj $env(pin_corner_keepout) \
      $env(pin_min_pitch_multiplier)]
    if {[llength $side_ports] > [lindex $vertical_capacity_data 0]} {
      close $pin_assignment_report
      error "$compass pin groups require [llength $side_ports] slots but vertical layer $vertical_ports_layer provides only [lindex $vertical_capacity_data 0]"
    }
    set primary_side_ports $side_ports
    set secondary_side_ports {}
    if {[info exists $groups_variable_name]} {
      set group_widths {}
      foreach group [set $groups_variable_name] {
        lappend group_widths [llength [resolve_manifest_group $group $all_ports]]
      }
    } else {
      set group_widths [list [llength $side_ports]]
    }
    set side_primary_layer $vertical_ports_layer
    set side_primary_layer_obj $vertical_pin_layer_obj
    set primary_capacity_data $vertical_capacity_data
    set secondary_capacity_data [list 0 0.0 0.0 [lindex $vertical_capacity_data 3]]
  } elseif {[info exists $groups_variable_name]} {
    set primary_capacity_data [pin_layer_capacity $innovus_side \
      $primary_pin_layer_obj $env(pin_corner_keepout) \
      $env(pin_min_pitch_multiplier)]
    set secondary_capacity_data [pin_layer_capacity $innovus_side \
      $secondary_pin_layer_obj $env(pin_corner_keepout) \
      $env(pin_min_pitch_multiplier)]
    set resolved_layers [pack_pin_groups_by_capacity \
      [set $groups_variable_name] $all_ports \
      [lindex $primary_capacity_data 0] [lindex $secondary_capacity_data 0] \
      $env(pin_primary_fraction) \
      $env(pin_spread_multiple_stream_groups) \
      $env(pin_stream_group_min_width)]
    set primary_side_ports [lindex $resolved_layers 0]
    set secondary_side_ports [lindex $resolved_layers 1]
    set group_widths [lindex $resolved_layers 2]
    set side_primary_layer $ports_layer
    set side_primary_layer_obj $primary_pin_layer_obj
  } else {
    set primary_side_ports $side_ports
    set secondary_side_ports {}
    set group_widths [list [llength $side_ports]]
    set primary_capacity_data [pin_layer_capacity $innovus_side \
      $primary_pin_layer_obj $env(pin_corner_keepout) \
      $env(pin_min_pitch_multiplier)]
    set secondary_capacity_data [pin_layer_capacity $innovus_side \
      $secondary_pin_layer_obj $env(pin_corner_keepout) \
      $env(pin_min_pitch_multiplier)]
    set side_primary_layer $ports_layer
    set side_primary_layer_obj $primary_pin_layer_obj
  }
  puts $pin_assignment_report \
    "$compass SUMMARY total=[llength $side_ports] groups={$group_widths} primary_layer=$side_primary_layer primary=[llength $primary_side_ports] primary_capacity=[lindex $primary_capacity_data 0] primary_adk_pitch=[lindex $primary_capacity_data 1] primary_min_pitch=[lindex $primary_capacity_data 2] secondary_layer=$secondary_ports_layer secondary=[llength $secondary_side_ports] secondary_capacity=[lindex $secondary_capacity_data 0] secondary_adk_pitch=[lindex $secondary_capacity_data 1] secondary_min_pitch=[lindex $secondary_capacity_data 2] spread_multiple_streams=$env(pin_spread_multiple_stream_groups) stream_group_min_width=$env(pin_stream_group_min_width) usable_edge=[lindex $primary_capacity_data 3] corner_keepout=$env(pin_corner_keepout)"
  set primary_layer_name [dbGet $side_primary_layer_obj.name]
  set secondary_layer_name [dbGet $secondary_pin_layer_obj.name]
  foreach port $primary_side_ports {
    puts $pin_assignment_report "PLAN $port side=$compass layer=$primary_layer_name"
  }
  foreach port $secondary_side_ports {
    puts $pin_assignment_report "PLAN $port side=$compass layer=$secondary_layer_name"
  }
  if {[llength $primary_side_ports] > 0} {
    if {$side_primary_layer == $ports_layer} {
      set primary_depth_multiplier $env(pin_primary_depth_width_multiplier)
    } else {
      set primary_depth_multiplier $env(pin_other_depth_width_multiplier)
    }
    edit_pins_with_corner_keepout \
      $primary_side_ports $side_primary_layer $side_primary_layer_obj \
      $innovus_side $env(pin_corner_keepout) $primary_depth_multiplier
  }
  if {[llength $secondary_side_ports] > 0} {
    edit_pins_with_corner_keepout \
      $secondary_side_ports $secondary_ports_layer $secondary_pin_layer_obj \
      $innovus_side $env(pin_corner_keepout) \
      $env(pin_other_depth_width_multiplier)
  }
}
if {![info exists allo_asic_clock_pins] || ![info exists allo_asic_clock_side]} {
  close $pin_assignment_report
  error "Pin intent does not define dedicated clock-pin placement"
}
array set allo_innovus_side {N TOP S BOTTOM E RIGHT W LEFT}
set resolved_clock_ports [resolve_manifest_group $allo_asic_clock_pins $all_ports]
if {[llength $resolved_clock_ports] > 0} {
  editPin -layer $ports_layer -pin $resolved_clock_ports \
    -side $allo_innovus_side($allo_asic_clock_side) -spreadType CENTER \
    -fixedPin true
  foreach port $resolved_clock_ports {
    if {[lsearch -exact $assigned_ports $port] >= 0} {
      close $pin_assignment_report
      error "Clock terminal '$port' was also assigned as an ordinary signal pin"
    }
    lappend assigned_ports $port
  }
  puts $pin_assignment_report \
    "CLOCK side=$allo_asic_clock_side layer=$ports_layer ports=[join $resolved_clock_ports { }]"
}
setPinAssignMode -pinEditInBatch false
set unassigned_ports {}
foreach port $all_ports {
  if {[lsearch -exact $assigned_ports $port] < 0} {
    lappend unassigned_ports $port
  }
}
if {[llength $unassigned_ports] > 0} {
  close $pin_assignment_report
  error "Manifest pin intent leaves ports unassigned: $unassigned_ports"
}
close $pin_assignment_report

reset_path_group -all
resetPathGroupOptions

set inputs   [all_inputs -no_clocks]
set outputs  [all_outputs]
set icgs     [filter_collection [all_registers] "is_integrated_clock_gating_cell == true"]
set regs     [remove_from_collection [all_registers -edge_triggered] $icgs]
set allregs  [all_registers]

set blocks      [dbGet top.insts.cell.baseClass block -p2]
set macro_refs  [list]
set macros      [list]
set blocks_exist [expr [lindex $blocks 0] != 0]

if { $blocks_exist } {
  foreach b $blocks {
    set cell    [dbGet $b.cell]
    set isBlock [dbIsCellBlock $cell]
    set isPhys  [dbGet $b.isPhysOnly]
    if { [expr $isBlock && ! $isPhys] } {
      lappend macro_refs $b
      lappend macros [dbGet $b.name]
    }
  }
}

group_path -name In2Out -from $inputs -to $outputs

if { $allregs != "" } {
  group_path -name In2Reg  -from $inputs  -to $allregs
  group_path -name Reg2Out -from $allregs -to $outputs
}

if { $regs != "" } {
  group_path -name Reg2Reg -from $regs -to $regs
}

if { $allregs != "" && $icgs != "" } {
  group_path -name Reg2ClkGate -from $allregs -to $icgs
}

if { $macros != "" } {
  group_path -name All2Macro -to $macros
  group_path -name Macro2All -from $macros
  setPathGroupOptions All2Macro -effortLevel high
  setPathGroupOptions Macro2All -effortLevel high
}

if { $regs != "" } {
  setPathGroupOptions Reg2Reg -effortLevel high
}

timeDesign -preplace -prefix preplace -outDir reports -expandedViews
timeDesign -preplace -hold -expandedViews -prefix preplace -outDir reports
checkDesign -all
check_timing
report_ports > reports/init.ports.rpt

report_metrics init
maybe_stop_after floorplan

#-------------------------------------------------------------------------
# Power
#-------------------------------------------------------------------------

foreach pin $power_pin_names {
  globalNetConnect $primary_power_net -type pgpin -pin $pin -inst * -verbose
}

foreach pin $ground_pin_names {
  globalNetConnect $primary_ground_net -type pgpin -pin $pin -inst * -verbose
}

# Sparse edge mesh: one VDD/VSS stripe pair on each of two orthogonal upper
# layers.  The stripes span the core, meet at vias, and are exported as LEF PG
# pins.  There is intentionally no private ring around each reusable macro.
set pg_width [expr {$env(edge_mesh_stripe_width_multiplier) * $M1_min_width}]
set pg_spacing [expr {$env(edge_mesh_stripe_spacing_multiplier) * $M1_min_spacing}]

setViaGenMode -reset
setViaGenMode -viarule_preference default
setViaGenMode -ignore_DRC false
setAddStripeMode -reset
setAddStripeMode -stacked_via_bottom_layer [expr {$base_layer_idx + 1}] \
  -stacked_via_top_layer $pmesh_top

foreach layer [list $pmesh_bot $pmesh_top] {
  set layer_direction [dbGet [dbGetLayerByZ $layer].direction]
  if {[string equal -nocase $layer_direction "Vertical"]} {
    set stripe_direction vertical
    set start_from left
  } else {
    set stripe_direction horizontal
    set start_from bottom
  }
  addStripe -nets $pwr_net_list_rev -layer $layer \
    -direction $stripe_direction \
    -width $pg_width \
    -spacing $pg_spacing \
    -number_of_sets 1 \
    -start_from $start_from \
    -start_offset $pg_spacing
}

# Connect standard-cell rails and PG pins to the sparse upper-metal mesh.
sroute -nets $pwr_net_list

maybe_stop_after power

#-------------------------------------------------------------------------
# Placement
#-------------------------------------------------------------------------

if {$::env(useful_skew)} {
  setOptMode -usefulSkew       true
  setOptMode -usefulSkewPreCTS true
} else {
  setOptMode -usefulSkew false
}

if {[info exists ADK_CELLS_TO_BE_PADDED]} {
  specifyCellPad $ADK_CELLS_TO_BE_PADDED $env(cell_padding)
  reportCellPad -file reports/place.cellpad.rpt
}

if {[info exists ADK_MIN_INST_GAP] && $ADK_MIN_INST_GAP > 0} {
  setPlaceMode -place_detail_legalization_inst_gap $ADK_MIN_INST_GAP
  if {$ADK_MIN_INST_GAP > 1} {
    setPlaceMode -place_detail_use_no_diffusion_one_site_filler false
  }
}

setAnalysisMode -analysisType onChipVariation

setPlaceMode -place_global_cong_effort medium \
  -place_global_clock_gate_aware true \
  -place_global_place_io_pins false

setOptMode -fixFanoutLoad true
set_db opt_enable_podv2_clock_opt_flow true

place_opt_design -out_dir reports -prefix place

if {[info exists ADK_TIE_CELLS] && $ADK_TIE_CELLS ne ""} {
  set tie_cells $ADK_TIE_CELLS
} else {
  set tie_cells "LOGIC1_X1 LOGIC0_X1"
}

setTieHiLoMode -cell $tie_cells -maxDistance 20 -maxFanout 8
foreach cell $tie_cells {
  setDontUse $cell false
}
addTieHiLo
foreach cell $tie_cells {
  setDontUse $cell true
}

deleteCellPad *
refinePlace

checkPlace -verbose > reports/place.checkPlace.after_refine.rpt
checkPlace -macroBlockage -verbose > reports/place.checkPlace.macroBlockage.after_refine.rpt
reportDensityMap > reports/place.density.rpt

report_metrics place
maybe_stop_after place

#-------------------------------------------------------------------------
# CTS
#-------------------------------------------------------------------------

set_ccopt_property clone_clock_gates true
set_ccopt_property clone_clock_logic true
set_ccopt_property ccopt_merge_clock_gates true
set_ccopt_property ccopt_merge_clock_logic true
set_ccopt_property cts_merge_clock_gates true
set_ccopt_property cts_merge_clock_logic true

if {$::env(useful_skew)} {
  setOptMode -usefulSkew      true
  setOptMode -usefulSkewCCOpt $::env(useful_skew_ccopt_effort)
} else {
  setOptMode -usefulSkew false
}

set_db design_process_node $env(process_node)
set_db opt_enable_podv2_clock_opt_flow true
set_db route_design_detail_use_multi_cut_via_effort high
set_db route_design_with_litho_driven true
set_db timing_analysis_cppr both

if {[lsearch [all_constraint_modes] constraints_default] != -1} {
  set vars(constraints_default,post_cts_sdc_list) [concat ./inputs/design.sdc]
}

set restore [get_db timing_defer_mmmc_obj_updates]
set_db timing_defer_mmmc_obj_updates true
foreach mode [all_constraint_modes] {
  if {[info exists vars($mode,post_cts_sdc_list)]} {
    update_constraint_mode -name $mode \
      -sdc_files $vars($mode,post_cts_sdc_list)
  } else {
    foreach view [all_analysis_views] {
      set m [regsub _$view $mode ""]
      if {[info exists vars($m,post_cts_sdc_list)]} {
        update_constraint_mode -name $mode \
          -sdc_files $vars($m,post_cts_sdc_list)
      }
    }
  }
}
set_analysis_view -update_timing
set_db timing_defer_mmmc_obj_updates $restore

clock_opt_design -prefix cts -out_dir reports

report_ccopt_clock_trees -list_special_pins -filename reports/cts.clock_trees.rpt
report_ccopt_skew_groups -filename reports/cts.skew_groups.rpt
report_ccopt_clock_tree_structure -show_sinks -expand_generated_clock_trees independently -file reports/cts.structure.rpt

report_metrics cts
maybe_stop_after cts

#-------------------------------------------------------------------------
# Post-CTS Hold
#-------------------------------------------------------------------------

setOptMode -fixHoldAllowOverlap TRUE
setOptMode -fixHoldAllowSetupTnsDegrade false
setOptMode -holdTargetSlack  $::env(macro_hold_target_slack)
setOptMode -setupTargetSlack $::env(macro_setup_target_slack)

optDesign -postCTS -setup -outDir reports -prefix postcts_setup
optDesign -postCTS -hold -outDir reports -prefix postcts_hold

report_metrics postcts_hold

#-------------------------------------------------------------------------
# Route
#-------------------------------------------------------------------------

setAnalysisMode -cppr both
setDelayCalMode -siAware true -engine aae

if {[info exists ADK_ANTENNA_CELL] && $ADK_ANTENNA_CELL ne ""} {
  set antenna_cell $ADK_ANTENNA_CELL
} else {
  set antenna_cell ANTENNA_X1
}

setNanoRouteMode -drouteUseMultiCutViaEffort high \
  -routeWithLithoDriven true \
  -routeWithTimingDriven true \
  -routeAntennaCellName $antenna_cell \
  -routeInsertAntennaDiode true

routeDesign -placementCheck

setNanoRouteMode -droutePostRouteSpreadWire true -routeWithTimingDriven true
routeDesign -wireOpt
setNanoRouteMode -droutePostRouteSpreadWire false

setExtractRCMode -engine postRoute -effortLevel low

report_metrics route
maybe_stop_after route

#-------------------------------------------------------------------------
# Postroute Optimization
#-------------------------------------------------------------------------

setOptMode -verbose true
setOptMode -usefulSkewPostRoute true
setOptMode -fixHoldAllowSetupTnsDegrade false
setOptMode -holdTargetSlack  $::env(macro_hold_target_slack)
setOptMode -setupTargetSlack $::env(macro_setup_target_slack)

if { $::env(signoff_engine) } {
  setExtractRCMode -engine postRoute -effortLevel signoff
}

set need_restore_multi false
if {[getDistributeHost -mode] == "local"} {
  set ncpu [getMultiCpuUsage -localCpu]
  if {$ncpu > $env(postroute_max_local_cpus)} {
    set need_restore_multi true
    setMultiCpuUsage -localCpu $env(postroute_max_local_cpus)
  }
}

for {set pass 1} {$pass <= $env(postroute_optimization_passes)} {incr pass} {
  optDesign -postRoute -outDir reports -prefix postroute_setup_${pass} -setup
  if {$pass == 1} {
    optDesign -postRoute -outDir reports -prefix postroute_drv -drv
  }
  # Hold runs after setup in every pass so the final optimized database honors
  # the positive hold margin. Setup degradation is explicitly disallowed.
  optDesign -postRoute -outDir reports -prefix postroute_hold_${pass} -hold
}

# Fillers are deliberately inserted after timing optimization. Adding them
# before post-route optimization consumes the whitespace needed for delay cells
# and buffers, which prevented repair of small hold violations.
if {[info exists ADK_FILLER_CELLS] && $ADK_FILLER_CELLS ne ""} {
  set filler_cells $ADK_FILLER_CELLS
} else {
  set filler_cells "FILLCELL_X32 FILLCELL_X16 FILLCELL_X8 FILLCELL_X4 FILLCELL_X2 FILLCELL_X1"
}
setFillerMode -core $filler_cells -corePrefix FILL
addFiller

# Perform a narrow filler/local-route cleanup before extraction and the
# mandatory final DRC check; do not reopen global routing or optimization.
ecoRoute -target

if {$need_restore_multi == true} {
  setDistributeHost -local
  setMultiCpuUsage -localCpu $ncpu
}

#-------------------------------------------------------------------------
# Signoff
#-------------------------------------------------------------------------

update_names -nocase

setExtractRCMode -coupled true -effortLevel low
setAnalysisMode -analysisType onChipVariation -cppr both

set vars(active_rc_corners) [list]
foreach view [concat [all_setup_analysis_views] [all_hold_analysis_views]] {
  set corner [get_delay_corner [get_analysis_view $view -delay_corner] -rc_corner]
  if {[lsearch $vars(active_rc_corners) $corner] == -1 } {
    lappend vars(active_rc_corners) $corner
  }
}

set empty_corners [list]
foreach corner $vars(active_rc_corners) {
  if {![file exists [get_rc_corner $corner -qx_tech_file]]} {
    lappend empty_corners $corner
  }
}

setExtractRCMode -engine postRoute -effortLevel low -coupled true

extractRC
foreach corner $vars(active_rc_corners) {
  rcOut -rc_corner $corner -spef $corner.spef.gz
}

timeDesign -prefix signoff -signoff -reportOnly -outDir reports -expandedViews
timeDesign -prefix signoff \
  -signoff \
  -reportOnly \
  -hold \
  -outDir reports \
  -expandedViews

# Calibre's FreePDK45 LVS deck declares PRECISION 10000. Keep GDS stream
# precision explicit and independent of Innovus's internal ADK DBU precision;
# otherwise a legal 1000-DBU Innovus database is rejected before LVS begins.
set stream_out_units $env(gds_stream_out_units)

streamOut $results_dir/$design_name.gds.gz \
  -units ${stream_out_units} \
  -mapFile $env(adk_gds_layer_map)

set merge_files \
  [concat \
    [lsort [glob -nocomplain inputs/adk/*.gds*]] \
    [lsort [glob -nocomplain inputs/*.gds*]] \
    [lsort [glob -nocomplain inputs/srams/*/*.gds*]] \
  ]

streamOut $results_dir/$design_name-merged.gds \
  -units ${stream_out_units} \
  -mapFile $env(adk_gds_layer_map) \
  -uniquifyCellNames \
  -merge $merge_files

summaryReport -noHtml -outfile reports/signoff.summaryReport.rpt
verifyConnectivity -noAntenna
redirect -file reports/innovus-drc.rpt { verify_drc }
set drc_stream [open reports/innovus-drc.rpt r]
set drc_text [read $drc_stream]
close $drc_stream
if {![regexp -nocase {Verification Complete\s*:\s*0\s+Viols} $drc_text]} {
  error "Final Innovus DRC is not explicitly clean; see reports/innovus-drc.rpt"
}

set antenna_policy $::env(antenna_check_policy)
if {[lsearch -exact {error report off} $antenna_policy] < 0} {
  error "Unsupported antenna_check_policy '$antenna_policy'; expected error, report, or off"
}
if {$antenna_policy eq "off"} {
  set antenna_stream [open reports/innovus-antenna.rpt w]
  puts $antenna_stream "Antenna verification skipped by antenna_check_policy=off"
  close $antenna_stream
} else {
  set antenna_status [catch {
    redirect -file reports/innovus-antenna.rpt { verify_antena }
  } antenna_error]
  if {$antenna_status != 0} {
    set antenna_stream [open reports/innovus-antenna.rpt a]
    puts $antenna_stream "ERROR: verify_antena failed: $antenna_error"
    close $antenna_stream
    if {$antenna_policy eq "error"} {
      error "Innovus antenna verification failed: $antenna_error"
    }
  } elseif {$antenna_policy eq "error"} {
    set antenna_stream [open reports/innovus-antenna.rpt r]
    set antenna_text [read $antenna_stream]
    close $antenna_stream
    if {![regexp -nocase {Verification Complete\s*:\s*0\s+Viols} $antenna_text]} {
      error "Innovus antenna verification is not explicitly clean; see reports/innovus-antenna.rpt"
    }
  }
}

write_sdf $results_dir/$design_name.sdf
writeTimingCon $results_dir/$design_name.pt.sdc
sed -i "s/^current_design/\#current_design/" $results_dir/$design_name.pt.sdc
sed -i "s/get_design.*$/current_design\]/" $results_dir/$design_name.pt.sdc

saveNetlist $results_dir/$design_name.topdown.v

set lvs_exclude_list ""
foreach x $lvs_exclude_cell_list {
  append lvs_exclude_list [dbGet -u -e top.insts.cell.name $x] " "
}

saveNetlist -excludeLeafCell \
  -includePowerGround \
  -phys \
  -excludeCellInst $lvs_exclude_list \
  $results_dir/$design_name.lvs.v

set virtuoso_exclude_list ""
foreach x $virtuoso_exclude_cell_list {
  append virtuoso_exclude_list [dbGet -u -e top.physInsts.cell.name $x] " "
}

saveNetlist -excludeLeafCell \
  -phys \
  -excludeCellInst $virtuoso_exclude_list \
  $results_dir/$design_name.virtuoso.v

saveNetlist -excludeLeafCell $results_dir/$design_name.vcs.v
saveNetlist -includePowerGround -excludeLeafCell $results_dir/$design_name.vcs.pg.v

# Export resolved post-route rectangles from the configured top macro-routing
# layer. Unlike DEF path syntax, these database boxes already include wire
# width and generated-via enclosure geometry.
set top_obs_layer_obj [dbGetLayerByZ $vars(max_route_layer)]
if {$top_obs_layer_obj == 0} {
  error "Cannot generate top-layer occupancy OBS: layer $vars(max_route_layer) is absent"
}
set top_obs_layer_name [dbGet $top_obs_layer_obj.name]
set top_obs_pitch [pin_layer_pitch $top_obs_layer_obj]
set top_obs_geometry $reports_dir/top-layer-geometry.tsv
set top_obs_stream [open $top_obs_geometry w]
set top_obs_die_box [dbGet top.fPlan.box]
if {[llength $top_obs_die_box] == 1} {
  set top_obs_die_box [lindex $top_obs_die_box 0]
}
puts $top_obs_stream "DIE\t[join $top_obs_die_box \t]"
puts $top_obs_stream "LAYER\t$top_obs_layer_name"
puts $top_obs_stream "PITCH\t$top_obs_pitch"

proc emit_top_obs_rects {stream kind net_name rects} {
  set count 0
  foreach rect $rects {
    # Some Innovus database attributes return one rectangle as {{x y x y}},
    # while list(rect) attributes return {{...} {...}}. Normalize the former.
    if {[llength $rect] == 1 && [llength [lindex $rect 0]] == 4} {
      set rect [lindex $rect 0]
    }
    if {[llength $rect] == 4} {
      puts $stream "RECT\t$kind\t$net_name\t[join $rect \t]"
      incr count
    }
  }
  return $count
}

proc top_obs_object_net_name {object} {
  set net_name [dbGet -e $object.net.name]
  if {[llength $net_name] != 1 || $net_name eq ""} {
    error "Top-layer geometry object $object has no unique owning net"
  }
  # dbGet may return a one-element Tcl list rendered with braces for names that
  # contain bus brackets. lindex returns the canonical element without list
  # serialization, so the TSV agrees with LEF PIN names.
  return [lindex $net_name 0]
}

set top_obs_rect_count 0
set top_obs_objects [dbQuery \
  -areas [list $top_obs_die_box] \
  -layers [list $top_obs_layer_name] \
  -objType {wire sWire viaInst sViaInst}]
foreach object $top_obs_objects {
  set object_type [dbGet $object.objType]
  if {$object_type eq "wire"} {
    set object_net [top_obs_object_net_name $object]
    incr top_obs_rect_count [emit_top_obs_rects \
      $top_obs_stream wire $object_net [dbGet $object.box]]
  } elseif {$object_type eq "sWire"} {
    set object_net [top_obs_object_net_name $object]
    incr top_obs_rect_count [emit_top_obs_rects \
      $top_obs_stream special_wire $object_net [dbGet $object.box]]
  } elseif {$object_type eq "viaInst" || $object_type eq "sViaInst"} {
    set object_net [top_obs_object_net_name $object]
    set prefix [expr {$object_type eq "sViaInst" ? "special_via" : "via"}]
    if {[dbGet $object.via.botLayer.name] eq $top_obs_layer_name} {
      incr top_obs_rect_count [emit_top_obs_rects \
        $top_obs_stream ${prefix}_bottom $object_net [dbGet $object.botRects]]
    }
    if {[dbGet $object.via.topLayer.name] eq $top_obs_layer_name} {
      incr top_obs_rect_count [emit_top_obs_rects \
        $top_obs_stream ${prefix}_top $object_net [dbGet $object.topRects]]
    }
  } else {
    close $top_obs_stream
    error "Unexpected dbQuery object type '$object_type' in top-layer occupancy extraction"
  }
}
close $top_obs_stream
if {$top_obs_rect_count == 0} {
  error "Top-layer occupancy query found no geometry on $top_obs_layer_name; refusing to emit an unsafe unobstructed abstract"
}

# Keep conservative full-layer OBS below the top layer, but omit the single
# full rectangle on the top layer. Embedded block OBS on otherwise uncovered
# layers is retained by -extractBlockObs.
write_lef_abstract \
  -stripePin \
  -specifyTopLayer $vars(max_route_layer) \
  -excludeObsLayers [list $top_obs_layer_name] \
  -extractBlockObs \
  -PGPinLayers [list $pmesh_bot $pmesh_top] \
  -noCutObs \
  $reports_dir/$design_name.base.lef

exec python3 scripts/build_coarse_top_layer_obs.py \
  --input-lef $reports_dir/$design_name.base.lef \
  --geometry $top_obs_geometry \
  --output-lef $results_dir/$design_name.lef \
  --report $reports_dir/top-layer-obs.rpt \
  --grid-tracks $env(top_layer_obs_grid_tracks) \
  --pin-grid-tracks $env(top_layer_obs_pin_grid_tracks) \
  --spacing-tracks $env(top_layer_obs_spacing_tracks)

# Diagnostic only: compare the requested scalar side/layer contract with the
# final abstract. Keep macro PNR usable while the abstract-export behavior is
# being characterized; mismatches are recorded rather than made fatal.
if {[catch {
  exec python3 scripts/check_final_lef_pins.py \
    reports/pin-assignment.rpt $results_dir/$design_name.lef \
    reports/final-lef-pin-check.rpt
} pin_check_error]} {
  set pin_check_stream [open reports/final-lef-pin-check.rpt w]
  puts $pin_check_stream "CHECK_ERROR $pin_check_error"
  close $pin_check_stream
}

defOut -routing -allLayers $results_dir/$design_name.def.gz

report_area -verbose > reports/signoff.area.rpt

report_metrics signoff
save_design_checkpoint
