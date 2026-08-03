#=========================================================================
# START.tcl
#=========================================================================
# Single-node Innovus PNR flow.

if {[catch {source -verbose scripts/pnr.tcl} error_message error_options]} {
  puts stderr "ERROR: macro PNR Tcl failed: $error_message"
  if {[dict exists $error_options -errorinfo]} {
    puts stderr [dict get $error_options -errorinfo]
  }
  exit 1
}

exit 0
