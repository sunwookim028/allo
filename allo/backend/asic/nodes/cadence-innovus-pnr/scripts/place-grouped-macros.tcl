#=========================================================================
# place-grouped-macros.tcl
#=========================================================================
# Deterministic first-pass hard-macro placement.

proc mg_clean_box {box} {
  if {[llength $box] == 1} {
    return [lindex $box 0]
  }
  return $box
}

proc mg_inst_name {inst} {
  return [dbGet $inst.name]
}

proc mg_cell_name {inst} {
  return [dbGet $inst.cell.name]
}

proc mg_parent_prefix {inst_name} {
  set parts [split $inst_name "/"]
  set n [llength $parts]

  if {$n <= 2} {
    return ""
  }

  # Drop the instance name and one likely generate/leaf wrapper level. This
  # keeps common SRAM-bank wrappers together without requiring user overrides.
  return [join [lrange $parts 0 [expr {$n - 3}]] "/"]
}

proc mg_macro_size {inst} {
  set cell [dbGet $inst.cell]
  set width ""
  set height ""

  catch {set width  [dbGet $cell.size_x]}
  catch {set height [dbGet $cell.size_y]}

  if {$width eq "" || $height eq "" || $width == 0 || $height == 0} {
    set box [mg_clean_box [dbGet $inst.box]]
    set width  [expr {[lindex $box 2] - [lindex $box 0]}]
    set height [expr {[lindex $box 3] - [lindex $box 1]}]
  }

  return [list $width $height]
}

proc mg_get_mfg_grid {} {
  set grid 0.005
  catch {
    set candidate [dbGet head.mfgGrid]
    if {$candidate ne "" && $candidate > 0} {
      set grid $candidate
    }
  }
  return $grid
}

proc mg_snap_down {value grid} {
  return [expr {floor($value / $grid) * $grid}]
}

proc mg_snap_up {value grid} {
  return [expr {ceil($value / $grid) * $grid}]
}

proc mg_has_r90_symmetry {inst} {
  # The legal-orientation attribute is version/library dependent in Innovus.
  # Keep the first deterministic macro placer conservative until pin-side and
  # symmetry probing are verified on real macro LEFs.
  return 0
}

proc mg_orient_for_edge {inst edge width height} {
  set can_r90 [mg_has_r90_symmetry $inst]

  if {($edge eq "bottom" || $edge eq "top") && $height > $width && $can_r90} {
    return R90
  }

  if {($edge eq "left" || $edge eq "right") && $width > $height && $can_r90} {
    return R90
  }

  return R0
}

proc mg_oriented_size {width height orient} {
  if {$orient eq "R90" || $orient eq "R270" || $orient eq "MYR90" || $orient eq "MXR90"} {
    return [list $height $width]
  }
  return [list $width $height]
}

proc mg_group_key {inst} {
  set name [mg_inst_name $inst]
  set cell [mg_cell_name $inst]
  set prefix [mg_parent_prefix $name]
  return "${cell}@@${prefix}"
}

proc mg_compare_group_area {a b} {
  set area_a [dict get $a area]
  set area_b [dict get $b area]

  if {$area_a < $area_b} {
    return 1
  } elseif {$area_a > $area_b} {
    return -1
  }
  return [string compare [dict get $a key] [dict get $b key]]
}

proc mg_choose_pack {count edge macro_w macro_h spacing max_depth} {
  if {$count <= 1} {
    return [list 1 1 $macro_w $macro_h]
  }

  set depth [expr {int(floor(sqrt(double($count))))}]
  if {$depth < 1} {
    set depth 1
  }
  if {$depth > $max_depth} {
    set depth $max_depth
  }

  if {$edge eq "left" || $edge eq "right"} {
    set cols $depth
    set rows [expr {int(ceil(double($count) / double($cols)))}]
  } else {
    set rows $depth
    set cols [expr {int(ceil(double($count) / double($rows)))}]
  }

  set group_w [expr {$cols * $macro_w + ($cols - 1) * $spacing}]
  set group_h [expr {$rows * $macro_h + ($rows - 1) * $spacing}]

  return [list $cols $rows $group_w $group_h]
}

proc mg_edge_score {edge cursor group_w group_h core_w core_h keepout spacing} {
  if {$edge eq "bottom" || $edge eq "top"} {
    set span $core_w
    set used [expr {$cursor + $group_w}]
    set intrusion $group_h
  } else {
    set span $core_h
    set used [expr {$cursor + $group_h}]
    set intrusion $group_w
  }

  if {$used > ($span - 2 * $keepout)} {
    return ""
  }

  return [expr {$intrusion * 1000000.0 + $used + $spacing}]
}

proc place_grouped_macros {} {
  global reports_dir macro_halo

  if {![info exists ::env(macro_edge_keepout)]} {
    set ::env(macro_edge_keepout) 30.0
  }
  if {![info exists ::env(macro_group_spacing)]} {
    set ::env(macro_group_spacing) 10.0
  }
  if {![info exists ::env(macro_group_max_depth)]} {
    set ::env(macro_group_max_depth) 2
  }

  set keepout $::env(macro_edge_keepout)
  set spacing $::env(macro_group_spacing)
  set max_depth $::env(macro_group_max_depth)
  set grid [mg_get_mfg_grid]

  set blocks [dbGet top.insts.cell.baseClass block -p2]
  set first_block [lindex $blocks 0]
  set has_blocks [expr {[llength $blocks] > 0 && $first_block ne "0" && $first_block ne "0x0"}]

  if {!$has_blocks} {
    puts "Info: No hard macros found for grouped macro placement"
    return 0
  }

  array unset group_members
  array unset group_area

  foreach inst $blocks {
    if {[dbGet $inst.isPhysOnly]} {
      continue
    }

    set key [mg_group_key $inst]
    set size [mg_macro_size $inst]
    set area [expr {[lindex $size 0] * [lindex $size 1]}]

    lappend group_members($key) $inst
    if {![info exists group_area($key)]} {
      set group_area($key) 0.0
    }
    set group_area($key) [expr {$group_area($key) + $area}]
  }

  set group_list [list]
  foreach key [array names group_members] {
    lappend group_list [dict create key $key area $group_area($key) members $group_members($key)]
  }

  if {[llength $group_list] == 0} {
    puts "Info: No placeable hard macros found for grouped macro placement"
    return 0
  }

  set group_list [lsort -command mg_compare_group_area $group_list]

  set core_box [mg_clean_box [dbGet top.fPlan.coreBox]]
  set core_llx [lindex $core_box 0]
  set core_lly [lindex $core_box 1]
  set core_urx [lindex $core_box 2]
  set core_ury [lindex $core_box 3]
  set core_w [expr {$core_urx - $core_llx}]
  set core_h [expr {$core_ury - $core_lly}]

  array set cursor {
    bottom 0.0
    top    0.0
    right  0.0
    left   0.0
  }

  set rpt [open "$reports_dir/macro.grouped_placement.rpt" w]
  puts $rpt "# grouped macro placement"
  puts $rpt "# core_box: $core_box"
  puts $rpt "# keepout: $keepout"
  puts $rpt "# spacing: $spacing"
  puts $rpt "# max_depth: $max_depth"

  foreach group $group_list {
    set key [dict get $group key]
    set members [dict get $group members]
    set count [llength $members]
    set first_inst [lindex $members 0]
    set size [mg_macro_size $first_inst]
    set raw_w [lindex $size 0]
    set raw_h [lindex $size 1]

    set best_edge ""
    set best_score ""
    set best_orient R0
    set best_pack ""
    set best_macro_size ""

    foreach edge {bottom top right left} {
      set orient [mg_orient_for_edge $first_inst $edge $raw_w $raw_h]
      set osize [mg_oriented_size $raw_w $raw_h $orient]
      set macro_w [lindex $osize 0]
      set macro_h [lindex $osize 1]
      set pack [mg_choose_pack $count $edge $macro_w $macro_h $spacing $max_depth]
      set group_w [lindex $pack 2]
      set group_h [lindex $pack 3]
      set score [mg_edge_score $edge $cursor($edge) $group_w $group_h $core_w $core_h $keepout $spacing]

      if {$score ne "" && ($best_score eq "" || $score < $best_score)} {
        set best_edge $edge
        set best_score $score
        set best_orient $orient
        set best_pack $pack
        set best_macro_size $osize
      }
    }

    if {$best_edge eq ""} {
      set best_edge bottom
      set best_orient [mg_orient_for_edge $first_inst bottom $raw_w $raw_h]
      set best_macro_size [mg_oriented_size $raw_w $raw_h $best_orient]
      set best_pack [mg_choose_pack $count bottom [lindex $best_macro_size 0] [lindex $best_macro_size 1] $spacing $max_depth]
      puts "Warning: Macro group $key does not fit available edge cursors; placing on bottom with possible overflow"
    }

    set cols [lindex $best_pack 0]
    set rows [lindex $best_pack 1]
    set group_w [lindex $best_pack 2]
    set group_h [lindex $best_pack 3]
    set macro_w [lindex $best_macro_size 0]
    set macro_h [lindex $best_macro_size 1]

    if {$best_edge eq "bottom"} {
      set origin_x [expr {$core_llx + $keepout + $cursor(bottom)}]
      set origin_y [expr {$core_lly + $keepout}]
      set cursor(bottom) [expr {$cursor(bottom) + $group_w + $spacing}]
    } elseif {$best_edge eq "top"} {
      set origin_x [expr {$core_llx + $keepout + $cursor(top)}]
      set origin_y [expr {$core_ury - $keepout - $group_h}]
      set cursor(top) [expr {$cursor(top) + $group_w + $spacing}]
    } elseif {$best_edge eq "right"} {
      set origin_x [expr {$core_urx - $keepout - $group_w}]
      set origin_y [expr {$core_lly + $keepout + $cursor(right)}]
      set cursor(right) [expr {$cursor(right) + $group_h + $spacing}]
    } else {
      set origin_x [expr {$core_llx + $keepout}]
      set origin_y [expr {$core_lly + $keepout + $cursor(left)}]
      set cursor(left) [expr {$cursor(left) + $group_h + $spacing}]
    }

    set origin_x [mg_snap_up $origin_x $grid]
    set origin_y [mg_snap_up $origin_y $grid]

    puts $rpt "group $key edge=$best_edge orient=$best_orient count=$count cols=$cols rows=$rows origin=($origin_x,$origin_y) size=($group_w,$group_h)"

    set i 0
    foreach inst $members {
      set col [expr {$i % $cols}]
      set row [expr {int($i / $cols)}]
      set x [mg_snap_up [expr {$origin_x + $col * ($macro_w + $spacing)}] $grid]
      set y [mg_snap_up [expr {$origin_y + $row * ($macro_h + $spacing)}] $grid]
      set name [mg_inst_name $inst]

      puts $rpt "  place $name $x $y $best_orient"
      placeInstance $name $x $y $best_orient
      incr i
    }
  }

  close $rpt

  addHaloToBlock $macro_halo $macro_halo $macro_halo $macro_halo -allMacro
  setInstancePlacementStatus -allHardMacros -status fixed
  checkPlace -macroBlockage -verbose > reports/init.checkPlace.macroBlockage.after_grouped_macro_place.rpt

  return 1
}
