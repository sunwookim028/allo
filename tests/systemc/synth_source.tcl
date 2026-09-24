# Catapult synth of the stream-interface DUT emitted by stream_boundary.py.
#   run:  catapult -shell -f synth_source.tcl   (from anywhere)
# The generated SystemC it reads is an archived emitter output, kept under
# dev/records/systemc/generated/; regenerate it with
#   python tests/systemc/stream_boundary.py
options set Input/CppStandard c++11
solution new -state initial
# The repository root, found by searching UPWARD for a marker -- never by
# counting levels (dev/roadmap.md).
set root [file normalize [file dir [info script]]]
while {![file exists [file join $root pyproject.toml]]} {
  set up [file dirname $root]
  if {$up eq $root} { error "no repository root above [info script]" }
  set root $up
}
solution file add $root/dev/records/systemc/generated/stream_boundary.cpp -type C++
directive set -DESIGN_HIERARCHY {source_0}
go analyze
go compile
solution library add nangate-45nm_beh
directive set -CLOCKS {clk {-CLOCK_PERIOD 5.0}}
go assembly
go extract
project save
exit
