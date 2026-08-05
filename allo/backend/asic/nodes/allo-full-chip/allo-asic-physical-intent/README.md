# Allo ASIC physical intent

Within each systolic kernel, PID columns increase east and PID rows increase
south. Innovus coordinates increase north, so the planner emits larger PID rows
at smaller Y coordinates. This makes the west/east and north/south stream pin
bundles generated during macro planning face their connected neighbors.

This dependency-free planner reads actual LEF dimensions, separation parameters,
the whole-region stream graph, the synthesized netlist, and optional SRAM views.
It packs SRAM instances sequentially around the perimeter to preserve a large
rectangular interior, tiles PE macros inside kernel clusters, and performs a
bounded deterministic kernel-slot optimization weighted by cross-kernel streams.
Innovus remains responsible for final manufacturing-grid and routing legality.

When `enable_kernel_rotation` is true, the planner may rotate a complete
semantic kernel cluster, including hardened HLS-only child modules attributed
to that kernel. Legal choices come from the published LEF `SYMMETRY`. The
deterministic objective combines packed region area with weighted distance
between kernels connected in the pre-HLS whole-region stream graph.
