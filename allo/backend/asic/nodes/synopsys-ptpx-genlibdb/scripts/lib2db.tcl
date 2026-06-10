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

set design_name $::env(design_name)

read_lib ${design_name}.lib
write_lib -format db $design_name -output ${design_name}.db

exit
