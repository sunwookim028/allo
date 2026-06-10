#=========================================================================
# lib2db.tcl
#=========================================================================
# Use Synopsys Library Compiler to convert lib to db
#
# Author : Julian Bushlow
# Date   : June 10, 2026
#

#-------------------------------------------------------------------------
# Extract db model
#-------------------------------------------------------------------------

enable_write_lib_mode

read_lib ${::env(design_name)}.lib
write_lib -format db $::env(design_name) -output ${::env(design_name)}.db

exit
