#=========================================================================
# add-macro-placement-blockages.tcl
#=========================================================================
# Author : Julian Bushlow
# Date   : July 25, 2026

set macro_halo 2.0

foreach inst [dbGet top.insts.cell.baseClass block -p2] {
  set name [dbGet $inst.name]
  set box  [dbGet $inst.box]

  set llx [expr [lindex $box 0] - $macro_halo]
  set lly [expr [lindex $box 1] - $macro_halo]
  set urx [expr [lindex $box 2] + $macro_halo]
  set ury [expr [lindex $box 3] + $macro_halo]

  createPlaceBlockage \
    -name "PBLOCK_MACRO_${name}" \
    -type hard \
    -box [list $llx $lly $urx $ury]
}
