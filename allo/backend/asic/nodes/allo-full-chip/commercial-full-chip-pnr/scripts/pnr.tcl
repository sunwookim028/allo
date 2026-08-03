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
if {![info exists env(gds_stream_out_units)]} {
  set env(gds_stream_out_units) 10000
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
if {![info exists env(pin_layer_offset)]} {
  set env(pin_layer_offset) 3
}
if {![info exists env(power_mesh_bot_layer)]} {
  if {[info exists ADK_POWER_MESH_BOT_LAYER]} {
    set env(power_mesh_bot_layer) $ADK_POWER_MESH_BOT_LAYER
  } else {
    set env(power_mesh_bot_layer) 8
  }
}
if {![info exists env(power_mesh_top_layer)]} {
  if {[info exists ADK_POWER_MESH_TOP_LAYER]} {
    set env(power_mesh_top_layer) $ADK_POWER_MESH_TOP_LAYER
  } else {
    set env(power_mesh_top_layer) 9
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
if {![info exists env(hold_target_slack)]} {
  set env(hold_target_slack) 0.05
}
if {![info exists env(setup_target_slack)]} {
  set env(setup_target_slack) 0.0
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
set allo_macro_lef_files [lsort [glob -nocomplain inputs/macro-registry/*/*.lef]]
source inputs/physical-intent.tcl

set init_verilog "./inputs/design.v"
set init_mmmc_file "scripts/mmmc.tcl"
set init_lef_file [concat \
  [list $env(adk_tech_lef) $env(adk_stdcell_lef) $env(adk_tech_lef) $env(adk_stdcell_lef)] \
  $sram_lef_files \
  $allo_macro_lef_files \
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

if {[info exists ADK_DONT_USE_CELL_LIST]} {
  foreach_in_collection cell [get_lib_cells $ADK_DONT_USE_CELL_LIST] {
    setDontUse [get_object_name $cell] true
  }
}

set core_aspect_ratio   $env(floorplan_aspect_ratio)
set core_density_target $env(core_density_target)

set M1_min_width   [dbGet [dbGetLayerByZ 1].minWidth]
set M1_min_spacing [dbGet [dbGetLayerByZ 1].minSpacing]

set savedvars(p_ring_width)   [expr 48 * $M1_min_width]
set savedvars(p_ring_spacing) [expr 24 * $M1_min_spacing]

set core_margin_t [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_b [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_r [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_l [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]

if {$env(floorplan_mode) == "physical_intent"} {
  # -s specifies the planned core dimensions; -d would interpret them as die
  # dimensions and shrink the usable core by the four power-grid margins.
  floorPlan -s $allo_core_width $allo_core_height \
    $core_margin_l $core_margin_b $core_margin_r $core_margin_t
} elseif {$env(floorplan_mode) == "auto"} {
  floorPlan -r $core_aspect_ratio $core_density_target \
    $core_margin_l $core_margin_b $core_margin_r $core_margin_t
} elseif {$env(floorplan_mode) == "fixed"} {
  if {$env(floorplan_width) == "" || $env(floorplan_height) == ""} {
    error "floorplan_mode=fixed requires floorplan_width and floorplan_height"
  }

  floorPlan -d $env(floorplan_width) $env(floorplan_height) \
    $core_margin_l $core_margin_b $core_margin_r $core_margin_t
} else {
  error "Unsupported floorplan_mode '$env(floorplan_mode)'; expected 'physical_intent', 'auto', or 'fixed'"
}

setFlipping s

set macro_halo $env(macro_halo)
source scripts/place-grouped-macros.tcl
# The original grouped SRAM placer is retained above for structural continuity
# and fallback research. The generated physical intent now coordinates optional
# SRAM perimeter packing and Allo PE/kernel tiling in one collision-free plan.
set placed_hard_macros [place_allo_physical_intent]
set has_hard_macros [expr {$placed_hard_macros > 0}]
set intent_rpt [open reports/physical-intent-placement.rpt w]
puts $intent_rpt "placed_hard_macros $placed_hard_macros"
puts $intent_rpt "core_width $allo_core_width"
puts $intent_rpt "core_height $allo_core_height"
close $intent_rpt
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
  # Preserve all useful inter-macro placement area. The generated intent cuts
  # only geometrically short row fragments that would otherwise fail welltap
  # insertion; ordinary per-block cuts below enforce each macro's true halo.
  set cut_short_fragments [cut_allo_short_row_fragments]
  set fragment_cut_rpt [open reports/selective-row-fragment-cuts.rpt w]
  puts $fragment_cut_rpt "cut_short_fragments $cut_short_fragments"
  close $fragment_cut_rpt
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
set all_ports [dbGet top.terms.name]
set num_ports [llength $all_ports]
set split_idx [expr ($num_ports + 1) / 2]
set pins_left_half  [lrange $all_ports 0 [expr $split_idx - 1]]
set pins_right_half [lrange $all_ports $split_idx end]

editPin -layer $ports_layer -pin $pins_left_half  -side LEFT  -spreadType SIDE
editPin -layer $ports_layer -pin $pins_right_half -side RIGHT -spreadType SIDE

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

sroute -nets $pwr_net_list

set M2_direction [dbGet [dbGet head.layers.name [expr $base_layer_idx + 2] -p].direction]

if { $M2_direction == "Vertical" } {
  addRing -nets $pwr_net_list -type core_rings -follow core \
    -layer [list top $pmesh_top bottom $pmesh_top left $pmesh_bot right $pmesh_bot] \
    -width $savedvars(p_ring_width) \
    -spacing $savedvars(p_ring_spacing) \
    -offset $savedvars(p_ring_spacing) \
    -extend_corner {tl tr bl br lt lb rt rb}

  set M1_min_width    [dbGet [dbGetLayerByZ [expr $base_layer_idx + 1]].minWidth]
  set M1_route_pitchX [dbGet [dbGetLayerByZ [expr $base_layer_idx + 1]].pitchX]

  set pmesh_bot_str_width [expr  8 *  3 * $M1_min_width]
  set pmesh_bot_str_pitch [expr 4 * 10 * $M1_route_pitchX]

  set pmesh_bot_str_intraset_spacing [expr $pmesh_bot_str_pitch - $pmesh_bot_str_width]
  set pmesh_bot_str_interset_pitch   [expr 2*$pmesh_bot_str_pitch]

  setViaGenMode -reset
  setViaGenMode -viarule_preference default
  setViaGenMode -ignore_DRC false
  setAddStripeMode -reset
  setAddStripeMode -stacked_via_bottom_layer [expr $base_layer_idx + 1] \
    -stacked_via_top_layer $pmesh_top

  addStripe -nets $pwr_net_list_rev -layer $pmesh_bot -direction vertical \
    -width $pmesh_bot_str_width \
    -spacing $pmesh_bot_str_intraset_spacing \
    -set_to_set_distance $pmesh_bot_str_interset_pitch \
    -max_same_layer_jog_length $pmesh_bot_str_pitch \
    -padcore_ring_bottom_layer_limit $pmesh_bot \
    -padcore_ring_top_layer_limit $pmesh_top \
    -start [expr $pmesh_bot_str_pitch]

  set pmesh_top_str_width [expr  8 *  3 * $M1_min_width]
  set pmesh_top_str_pitch [expr 4 * 10 * $M1_route_pitchX]
  set pmesh_top_str_intraset_spacing [expr $pmesh_top_str_pitch - $pmesh_top_str_width]
  set pmesh_top_str_interset_pitch   [expr 2*$pmesh_top_str_pitch]

  setViaGenMode -reset
  setViaGenMode -viarule_preference default
  setViaGenMode -ignore_DRC false
  setAddStripeMode -reset
  setAddStripeMode -stacked_via_bottom_layer $pmesh_bot \
    -stacked_via_top_layer $pmesh_top

  addStripe -nets $pwr_net_list_rev -layer $pmesh_top -direction horizontal \
    -width $pmesh_top_str_width \
    -spacing $pmesh_top_str_intraset_spacing \
    -set_to_set_distance $pmesh_top_str_interset_pitch \
    -max_same_layer_jog_length $pmesh_top_str_pitch \
    -padcore_ring_bottom_layer_limit $pmesh_bot \
    -padcore_ring_top_layer_limit $pmesh_top \
    -start [expr $pmesh_top_str_pitch]
} else {
  set M3_min_width    [dbGet [dbGetLayerByZ 3].minWidth]
  set M3_route_pitchX [dbGet [dbGetLayerByZ 3].pitchX]
  set M3_str_width            [expr  3 * $M3_min_width]
  set M3_str_pitch            [expr 10 * $M3_route_pitchX]
  set M3_str_intraset_spacing [expr $M3_str_pitch - $M3_str_width]
  set M3_str_interset_pitch   [expr 2*$M3_str_pitch]
  set M3_str_offset           [expr $M3_str_pitch + $M3_route_pitchX/2 - $M3_str_width/2]

  setViaGenMode -reset
  setViaGenMode -viarule_preference default
  setViaGenMode -ignore_DRC false
  setAddStripeMode -reset
  setAddStripeMode -stacked_via_bottom_layer 1 \
    -stacked_via_top_layer 3

  addStripe -nets $pwr_net_list_rev -layer 3 -direction vertical \
    -width $M3_str_width \
    -spacing $M3_str_intraset_spacing \
    -set_to_set_distance $M3_str_interset_pitch \
    -start_offset $M3_str_offset

  addRing -nets $pwr_net_list -type core_rings -follow core \
    -layer [list top $pmesh_bot bottom $pmesh_bot left $pmesh_top right $pmesh_top] \
    -width $savedvars(p_ring_width) \
    -spacing $savedvars(p_ring_spacing) \
    -offset $savedvars(p_ring_spacing) \
    -extend_corner {tl tr bl br lt lb rt rb}

  set pmesh_bot_str_width [expr  8 * $M3_str_width]
  set pmesh_bot_str_pitch [expr 10 * $M3_str_pitch]
  set pmesh_bot_str_intraset_spacing [expr $pmesh_bot_str_pitch - $pmesh_bot_str_width]
  set pmesh_bot_str_interset_pitch   [expr 2*$pmesh_bot_str_pitch]

  setViaGenMode -reset
  setViaGenMode -viarule_preference default
  setViaGenMode -ignore_DRC false
  setAddStripeMode -reset
  setAddStripeMode -stacked_via_bottom_layer 3 \
    -stacked_via_top_layer $pmesh_top

  addStripe -nets $pwr_net_list_rev -layer $pmesh_bot -direction horizontal \
    -width $pmesh_bot_str_width \
    -spacing $pmesh_bot_str_intraset_spacing \
    -set_to_set_distance $pmesh_bot_str_interset_pitch \
    -max_same_layer_jog_length $pmesh_bot_str_pitch \
    -padcore_ring_bottom_layer_limit $pmesh_bot \
    -padcore_ring_top_layer_limit $pmesh_top \
    -start [expr $pmesh_bot_str_pitch]

  set pmesh_top_str_width [expr 16 * $M3_str_width]
  set pmesh_top_str_pitch [expr 20 * $M3_str_pitch]
  set pmesh_top_str_intraset_spacing [expr $pmesh_top_str_pitch - $pmesh_top_str_width]
  set pmesh_top_str_interset_pitch   [expr 2*$pmesh_top_str_pitch]

  setViaGenMode -reset
  setViaGenMode -viarule_preference default
  setViaGenMode -ignore_DRC false
  setAddStripeMode -reset
  setAddStripeMode -stacked_via_bottom_layer $pmesh_bot \
    -stacked_via_top_layer $pmesh_top

  addStripe -nets $pwr_net_list_rev -layer $pmesh_top -direction vertical \
    -width $pmesh_top_str_width \
    -spacing $pmesh_top_str_intraset_spacing \
    -set_to_set_distance $pmesh_top_str_interset_pitch \
    -max_same_layer_jog_length $pmesh_top_str_pitch \
    -padcore_ring_bottom_layer_limit $pmesh_bot \
    -padcore_ring_top_layer_limit $pmesh_top \
    -start [expr $pmesh_top_str_pitch/2]
}

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
setOptMode -fixHoldAllowSetupTnsDegrade true

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
  -routeAntennaCellName $antenna_cell \
  -routeInsertAntennaDiode true

if {[info exists ADK_FILLER_CELLS] && $ADK_FILLER_CELLS ne ""} {
  set filler_cells $ADK_FILLER_CELLS
} else {
  set filler_cells "FILLCELL_X32 FILLCELL_X16 FILLCELL_X8 FILLCELL_X4 FILLCELL_X2 FILLCELL_X1"
}

routeDesign -placementCheck

setNanoRouteMode -droutePostRouteSpreadWire true -routeWithTimingDriven false
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
setOptMode -holdTargetSlack  $::env(hold_target_slack)
setOptMode -setupTargetSlack $::env(setup_target_slack)

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

optDesign -postRoute -outDir reports -prefix postroute_setup -setup
optDesign -postRoute -outDir reports -prefix postroute_drv -drv
optDesign -postRoute -outDir reports -prefix postroute_hold -hold

# Preserve whitespace for routing, timing repair, and antenna-diode insertion.
# Physical-only filler cells are safe to add after those operations complete.
setFillerMode -core $filler_cells -corePrefix FILL
addFiller

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

set stream_out_units $env(gds_stream_out_units)

streamOut $results_dir/$design_name.gds.gz \
  -units ${stream_out_units} \
  -mapFile $env(adk_gds_layer_map)

set merge_files \
  [concat \
    [lsort [glob -nocomplain inputs/adk/*.gds*]] \
    [lsort [glob -nocomplain inputs/*.gds*]] \
    [lsort [glob -nocomplain inputs/srams/*/*.gds*]] \
    [lsort [glob -nocomplain inputs/macro-registry/*/*.gds*]] \
  ]

streamOut $results_dir/$design_name-merged.gds \
  -units ${stream_out_units} \
  -mapFile $env(adk_gds_layer_map) \
  -uniquifyCellNames \
  -merge $merge_files

summaryReport -noHtml -outfile reports/signoff.summaryReport.rpt
verifyConnectivity -noAntenna
verify_drc
verifyProcessAntenna

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

write_lef_abstract \
  -stripePin \
  -specifyTopLayer $vars(max_route_layer) \
  -PGPinLayers [list $pmesh_bot $pmesh_top] \
  -noCutObs \
  $results_dir/$design_name.lef

defOut -routing -allLayers $results_dir/$design_name.def.gz

report_area -verbose > reports/signoff.area.rpt

report_metrics signoff
save_design_checkpoint
