# Allo ASIC physical intent

This dependency-free planner reads actual LEF dimensions, separation parameters,
the whole-region stream graph, the synthesized netlist, and optional SRAM views.
It packs SRAM instances sequentially around the perimeter to preserve a large
rectangular interior, tiles PE macros inside kernel clusters, and performs a
bounded deterministic kernel-slot optimization weighted by cross-kernel streams.
Innovus remains responsible for final manufacturing-grid and routing legality.
