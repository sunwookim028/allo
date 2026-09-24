# Commercial RTL simulation

Runs the normalized `design.v` with the packaged user testbench. The testbench
contract supplies the simulation top, DUT instance, pass/failure markers, and
timeout. VCS consumes SystemVerilog directly through the packaged file list.
