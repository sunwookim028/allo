README
==========================================================================
Author : Christopher Torng
Date   : June 7, 2019

This ASIC design kit uses FreePDK45 and the NanGate Open Cell Library.

More information is available in the README of the standard base kit
located here:

- https://github.com/mflowgen/freepdk-45nm

This repository only contains the tiny base kit necessary to push a
design through synthesis and place and route but not further (e.g.,
DRC, LVS). The standard base kit contains the technology and library
files to run through synthesis, place and route, and signoff steps.

`vcs-compile.args` is published with either view and selects the Nangate
library's common gate-level behavior with `TETRAMAX`. BAGL additionally reads
`vcs-bagl.args`; its `NTC` and `RECREM` defines make the model's combined
negative timing checks match the `SETUPHOLD` and `RECREM` records emitted in
Innovus SDF. Keeping timing-only defines in the BAGL file preserves the prior
zero-delay FFGL behavior.
