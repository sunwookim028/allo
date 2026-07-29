#=========================================================================
# add-endcaps-welltaps.tcl
#=========================================================================
# Author : Christopher Torng
# Date   : March 4, 2020

# Add end caps if ADK contains end caps

if {   [info exists ADK_END_CAP_CELL_LEFT]
    && [expr {$ADK_END_CAP_CELL_LEFT ne ""}]
    && [info exists ADK_END_CAP_CELL_RIGHT]
    && [expr {$ADK_END_CAP_CELL_RIGHT ne ""}] } {
  # The -rightEdge option is for precap (left edge of core rows)
  # The -leftEdge option is for postcap (right edge of core rows)
  setEndCapMode -rightEdge $ADK_END_CAP_CELL_LEFT
  setEndCapMode -leftEdge  $ADK_END_CAP_CELL_RIGHT
  addEndCap     -prefix    ENDCAP
} else {
  echo "Warning: mflowgen skipping end cap insertion because none found in ADK"
}

# block off some rows to prevent overlap with macros

# Cut standard-cell rows under hard macros before welltap insertion.
# Otherwise addWellTap may place taps through SRAM macro areas, and
# verifyWellTap may later report coverage gaps around deleted taps.

set macro_halo 3.0
set blocks [dbGet top.insts.cell.baseClass block -p2]

if {[llength $blocks] == 0 || [lindex $blocks 0] == 0 || [lindex $blocks 0] == "0x0"} {
  puts "Info: No macro/block instances found for row cutting"
} else {
  foreach inst $blocks {
    if {[dbGet $inst.isPhysOnly]} {
      continue
    }

    set macro_name [dbGet $inst.name]
    set box [dbGet $inst.box]
    if {[llength $box] == 1} {
      set box [lindex $box 0]
    }

    set llx [expr {[lindex $box 0] - $macro_halo}]
    set lly [expr {[lindex $box 1] - $macro_halo}]
    set urx [expr {[lindex $box 2] + $macro_halo}]
    set ury [expr {[lindex $box 3] + $macro_halo}]

    puts "Info: Cutting rows around macro $macro_name: $llx $lly $urx $ury"
    cutRow -area [list $llx $lly $urx $ury]
  }
}

# Add well taps if ADK contains well taps

if {[info exists ADK_WELL_TAP_CELL] && [expr {$ADK_WELL_TAP_CELL ne ""}]} {
  addWellTap -cell [list $ADK_WELL_TAP_CELL] \
             -prefix       WELLTAP \
             -cellInterval $ADK_WELL_TAP_INTERVAL 


  verifyWellTap -cells [list $ADK_WELL_TAP_CELL] \
                -report reports/welltap.rpt \
                -rule   [ expr $ADK_WELL_TAP_INTERVAL/2 ]
} else {
  echo "Warning: mflowgen skipping well tap insertion because no well taps found in ADK"
}


