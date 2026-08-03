# Commercial full-chip PNR

This node begins as a direct copy of `cadence-innovus-pnr`; its script structure
and comments are retained. Stage-2 changes add hardened macro Liberty/LEF/GDS
views, consume the generated physical-intent dimensions and placement Tcl, and
publish runtime and placement reports. The original grouped macro script remains
in place, while the generated plan coordinates optional SRAM perimeter packing
and Allo kernel/PE tiling before standard-cell placement.

The optional `srams` input retains the original directory interface. When it is
absent or empty, physical-intent generation and PNR operate as a PE-only flow.
