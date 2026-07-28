#=========================================================================
# innovus_single_node_clean_reference.tcl
#=========================================================================
# Clean reference version of the current Mini-NPU Innovus flow.
# This is meant to be readable starting material for a hand-owned single
# PNR node. It preserves the important command sequence and flags from the
# current mflowgen/IFF run, but drops wrapper plumbing and generated comments.

set design_name compute_tile

set results_dir     results
set reports_dir     reports
set checkpoints_dir checkpoints

set vars(design)        $design_name
set vars(results_dir)   $results_dir
set vars(rpt_dir)       $reports_dir
set vars(max_route_layer) 7
set vars(local_cpus)    16

file mkdir $results_dir
file mkdir $reports_dir
file mkdir $checkpoints_dir
file mkdir $checkpoints_dir/LEC

setDistributeHost -local
setMultiCpuUsage -localCpu 16

source inputs/adk/adk.tcl

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

proc save_phase {phase} {
  global checkpoints_dir design_name

  file mkdir $checkpoints_dir
  file mkdir $checkpoints_dir/LEC
  file mkdir $checkpoints_dir/${phase}.checkpoint

  saveDesign $checkpoints_dir/${phase}.checkpoint/save.enc -compress
  saveNetlist $checkpoints_dir/LEC/${phase}.v.gz

  if {$phase == "signoff"} {
    file mkdir $checkpoints_dir/design.checkpoint
    saveDesign $checkpoints_dir/design.checkpoint/save.enc -compress
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
  [list inputs/adk/rtk-tech.lef inputs/adk/stdcells.lef inputs/adk/rtk-tech.lef inputs/adk/stdcells.lef] \
  $sram_lef_files \
]
set init_top_cell $design_name
set init_gnd_net "VSS VPW VSSPST VSSE"
set init_pwr_net "VDD VNW VDDPST POC VDDCE VDDPE"

set_db init_no_new_assigns true

if {[file exists inputs/adk/pdk-qrc-lef.map]} {
  setExtractRCMode -lefTechFileMap inputs/adk/pdk-qrc-lef.map
}

init_design

set_power_analysis_mode -analysis_view analysis_default
setMaxRouteLayer 7
setDesignMode -process 45 -powerEffort high

#-------------------------------------------------------------------------
# Init / Floorplan / Macros
#-------------------------------------------------------------------------

setOptMode -timeDesignCompressReports false

if {[info exists ADK_DONT_USE_CELL_LIST]} {
  foreach_in_collection cell [get_lib_cells $ADK_DONT_USE_CELL_LIST] {
    setDontUse [get_object_name $cell] true
  }
}

set core_aspect_ratio   1.00
set core_density_target $env(core_density_target)
set pwr_net_list {VDD VSS}

set M1_min_width   [dbGet [dbGetLayerByZ 1].minWidth]
set M1_min_spacing [dbGet [dbGetLayerByZ 1].minSpacing]

set savedvars(p_ring_width)   [expr 48 * $M1_min_width]
set savedvars(p_ring_spacing) [expr 24 * $M1_min_spacing]

set core_margin_t [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_b [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_r [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]
set core_margin_l [expr ([llength $pwr_net_list] * ($savedvars(p_ring_width) + $savedvars(p_ring_spacing))) + $savedvars(p_ring_spacing)]

floorPlan -r $core_aspect_ratio $core_density_target \
  $core_margin_l $core_margin_b $core_margin_r $core_margin_t

setFlipping s
planDesign

# Optional place to replace planDesign with explicit SRAM placement/keepouts.
# If kept, fix SRAM macro placement here before row cutting, taps, and power.

if {   [info exists ADK_END_CAP_CELL_LEFT]
    && [expr {$ADK_END_CAP_CELL_LEFT ne ""}]
    && [info exists ADK_END_CAP_CELL_RIGHT]
    && [expr {$ADK_END_CAP_CELL_RIGHT ne ""}] } {
  setEndCapMode -rightEdge $ADK_END_CAP_CELL_LEFT
  setEndCapMode -leftEdge  $ADK_END_CAP_CELL_RIGHT
  addEndCap -prefix ENDCAP
}

set macro_halo 2.0
set blocks [dbGet top.insts.cell.baseClass block -p2]

if {![expr {[llength $blocks] == 0 || [lindex $blocks 0] == 0 || [lindex $blocks 0] == "0x0"}]} {
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

if {[info exists ADK_WELL_TAP_CELL] && [expr {$ADK_WELL_TAP_CELL ne ""}]} {
  addWellTap -cell [list $ADK_WELL_TAP_CELL] \
    -prefix WELLTAP \
    -cellInterval $ADK_WELL_TAP_INTERVAL

  verifyWellTap -cells [list $ADK_WELL_TAP_CELL] \
    -report reports/welltap.rpt \
    -rule [expr $ADK_WELL_TAP_INTERVAL/2]
}

#-------------------------------------------------------------------------
# Pin Assignment / Path Groups / Init Reports
#-------------------------------------------------------------------------

set ports_layer [expr $ADK_BASE_LAYER_IDX + 3]
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
save_phase init

#-------------------------------------------------------------------------
# Power
#-------------------------------------------------------------------------

foreach pin {VDD VPWR VNW VPB vcc vdd} {
  globalNetConnect VDD -type pgpin -pin $pin -inst * -verbose
}

foreach pin {VSS VGND VPW VNB vssx gnd} {
  globalNetConnect VSS -type pgpin -pin $pin -inst * -verbose
}

sroute -nets {VDD VSS}

if {[info exists ADK_BASE_LAYER_IDX]} {
  set base_layer_idx $ADK_BASE_LAYER_IDX
} else {
  set base_layer_idx 0
}

set M2_direction [dbGet [dbGet head.layers.name [expr $base_layer_idx + 2] -p].direction]
set pmesh_bot $ADK_POWER_MESH_BOT_LAYER
set pmesh_top $ADK_POWER_MESH_TOP_LAYER

if { $M2_direction == "Vertical" } {
  addRing -nets {VDD VSS} -type core_rings -follow core \
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

  addStripe -nets {VSS VDD} -layer $pmesh_bot -direction vertical \
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

  addStripe -nets {VSS VDD} -layer $pmesh_top -direction horizontal \
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

  addStripe -nets {VSS VDD} -layer 3 -direction vertical \
    -width $M3_str_width \
    -spacing $M3_str_intraset_spacing \
    -set_to_set_distance $M3_str_interset_pitch \
    -start_offset $M3_str_offset

  addRing -nets {VDD VSS} -type core_rings -follow core \
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

  addStripe -nets {VSS VDD} -layer $pmesh_bot -direction horizontal \
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

  addStripe -nets {VSS VDD} -layer $pmesh_top -direction vertical \
    -width $pmesh_top_str_width \
    -spacing $pmesh_top_str_intraset_spacing \
    -set_to_set_distance $pmesh_top_str_interset_pitch \
    -max_same_layer_jog_length $pmesh_top_str_pitch \
    -padcore_ring_bottom_layer_limit $pmesh_bot \
    -padcore_ring_top_layer_limit $pmesh_top \
    -start [expr $pmesh_top_str_pitch/2]
}

save_phase power

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

setDesignMode -process 45 -powerEffort high
setAnalysisMode -analysisType onChipVariation

setPlaceMode -place_global_cong_effort medium \
  -place_global_clock_gate_aware true \
  -place_global_place_io_pins false

setOptMode -fixFanoutLoad true
set_db opt_enable_podv2_clock_opt_flow true

place_opt_design -out_dir reports -prefix place

setTieHiLoMode -cell "LOGIC1_X1  LOGIC0_X1" -maxDistance 20 -maxFanout 8
foreach cell {LOGIC1_X1 LOGIC0_X1} {
  setDontUse $cell false
}
addTieHiLo
foreach cell {LOGIC1_X1 LOGIC0_X1} {
  setDontUse $cell true
}

deleteCellPad *
refinePlace

checkPlace -verbose > reports/place.checkPlace.after_refine.rpt
checkPlace -macroBlockage -verbose > reports/place.checkPlace.macroBlockage.after_refine.rpt
reportDensityMap > reports/place.density.rpt

report_metrics place
save_phase place

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

set_db design_process_node 45
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
save_phase cts

#-------------------------------------------------------------------------
# Post-CTS Hold
#-------------------------------------------------------------------------

setOptMode -fixHoldAllowOverlap TRUE
setDesignMode -process 45 -powerEffort high
setOptMode -fixHoldAllowSetupTnsDegrade true

optDesign -postCTS -hold -outDir reports -prefix postcts_hold

report_metrics postcts_hold
save_phase postcts_hold

#-------------------------------------------------------------------------
# Route
#-------------------------------------------------------------------------

setAnalysisMode -cppr both
setDelayCalMode -siAware true -engine aae

setNanoRouteMode -drouteUseMultiCutViaEffort high \
  -routeWithLithoDriven true \
  -routeAntennaCellName ANTENNA_X1 \
  -routeInsertAntennaDiode true

setFillerMode -core "FILLCELL_X32  FILLCELL_X16  FILLCELL_X8  FILLCELL_X4  FILLCELL_X2  FILLCELL_X1" \
  -corePrefix FILL
addFiller

routeDesign -placementCheck

setNanoRouteMode -droutePostRouteSpreadWire true -routeWithTimingDriven false
routeDesign -wireOpt
setNanoRouteMode -droutePostRouteSpreadWire false

setExtractRCMode -engine postRoute -effortLevel low

report_metrics route
save_phase route

#-------------------------------------------------------------------------
# Postroute Hold
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
  if {$ncpu > 8} {
    set need_restore_multi true
    setMultiCpuUsage -localCpu 8
  }
}

optDesign -postRoute -outDir reports -prefix postroute_hold -hold

if {$need_restore_multi == true} {
  setDistributeHost -local
  setMultiCpuUsage -localCpu $ncpu
}

save_phase postroute_hold

#-------------------------------------------------------------------------
# Signoff
#-------------------------------------------------------------------------

update_names -nocase

setDesignMode -process 45
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

if { [info exists ADK_DBU_PRECISION] } {
  set stream_out_units $ADK_DBU_PRECISION
} else {
  set stream_out_units 1000
}

streamOut $results_dir/$design_name.gds.gz \
  -units ${stream_out_units} \
  -mapFile inputs/adk/rtk-stream-out.map

set merge_files \
  [concat \
    [lsort [glob -nocomplain inputs/adk/*.gds*]] \
    [lsort [glob -nocomplain inputs/*.gds*]] \
    [lsort [glob -nocomplain inputs/srams/*/*.gds*]] \
  ]

streamOut $results_dir/$design_name-merged.gds \
  -units ${stream_out_units} \
  -mapFile inputs/adk/rtk-stream-out.map \
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
foreach x $ADK_LVS_EXCLUDE_CELL_LIST {
  append lvs_exclude_list [dbGet -u -e top.insts.cell.name $x] " "
}

saveNetlist -excludeLeafCell \
  -phys \
  -excludeCellInst $lvs_exclude_list \
  $results_dir/$design_name.lvs.v

set virtuoso_exclude_list ""
foreach x $ADK_VIRTUOSO_EXCLUDE_CELL_LIST {
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
  -PGPinLayers [list $ADK_POWER_MESH_BOT_LAYER $ADK_POWER_MESH_TOP_LAYER] \
  -noCutObs \
  $results_dir/$design_name.lef

defOut -routing -allLayers $results_dir/$design_name.def.gz

report_area -verbose > reports/signoff.area.rpt

report_metrics signoff
save_phase signoff
