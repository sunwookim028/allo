#=========================================================================
# Project FreePDK45 ADK policy overlay
#=========================================================================

# Preserve all variables supplied by the selected upstream ADK view.
set adk_overlay_dir [file dirname [file normalize [info script]]]
source [file join $adk_overlay_dir adk-base.tcl]

# Scan flip-flops can be used by synthesis as muxed functional registers.
# Their gate models are X-pessimistic when the scan enable depends on an
# uninitialized next-state cone, which can prevent reset from initializing a
# Vitis one-hot FSM during FFGL simulation. Keep this PDK-specific naming
# knowledge in the ADK so generic synthesis and P&R nodes remain portable.
set ADK_SCAN_CELL_LIST [list */SDFF*]

if {[info exists ADK_DONT_USE_CELL_LIST]} {
  set ADK_DONT_USE_CELL_LIST [concat \
    $ADK_DONT_USE_CELL_LIST \
    $ADK_SCAN_CELL_LIST]
} else {
  set ADK_DONT_USE_CELL_LIST $ADK_SCAN_CELL_LIST
}

unset adk_overlay_dir
