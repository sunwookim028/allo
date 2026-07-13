#=========================================================================
# create-sram-mwlib.tcl
#=========================================================================
# Import one SRAM LEF into a Synopsys Milkyway library.
#
# Expected variables:
#   sram_lef
#   sram_mwlib
#   sram_tech_file
#

if {![info exists sram_lef]} {
  error "Missing required variable: sram_lef"
}

if {![info exists sram_mwlib]} {
  error "Missing required variable: sram_mwlib"
}

if {![info exists sram_tech_file]} {
  error "Missing required variable: sram_tech_file"
}

if {![file exists $sram_lef]} {
  error "SRAM LEF does not exist: $sram_lef"
}

if {![file exists $sram_tech_file]} {
  error "Milkyway technology file does not exist: $sram_tech_file"
}

if {[file isdirectory $sram_mwlib]} {
  file delete -force $sram_mwlib
}

create_mw_lib -technology $sram_tech_file $sram_mwlib
open_mw_lib $sram_mwlib

read_lef $sram_lef

close_mw_lib
exit

