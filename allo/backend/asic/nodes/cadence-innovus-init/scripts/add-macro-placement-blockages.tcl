#=========================================================================
# add-macro-placement-blockages.tcl
#=========================================================================
# Author : Julian Bushlow
# Date   : July 25, 2026

set macro_halo 2.0

set blocks [dbGet top.insts.cell.baseClass block -p2]

if {[llength $blocks] == 0 || [lindex $blocks 0] == 0 || [lindex $blocks 0] == "0x0"} {
  puts "Info: No macro/block instances found for halos"
  return
}

foreach inst $blocks {
  if {[dbGet $inst.isPhysOnly]} {
    continue
  }

  set name [dbGet $inst.name]
  puts "Info: Adding halo around macro $name"
  addHaloToBlock $macro_halo $macro_halo $macro_halo $macro_halo $name
}
