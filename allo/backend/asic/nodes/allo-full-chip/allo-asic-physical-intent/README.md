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

`interleave_macros` defaults to `False`, preserving the original rigid
one-cluster-per-semantic-kernel placement. When explicitly enabled, the planner
examines concrete cross-kernel endpoint PIDs and accepts interleaving only for
a complete, exact, repeated D4-plus-offset stencil. At least two anchor PIDs
must repeat the same stencil; all hardened PIDs in both kernels must be covered;
each target PID must belong to exactly one composite group; and neither kernel
may have multiple hardened macros at one PID. Accepted disjoint kernel pairs
are tiled as repeated composite groups using normal `macro_separation_x/y`.
Irregular, partial, ambiguous, overlapping, or many-kernel relationships retain
the existing rigid-kernel fallback and `kernel_separation_x/y` behavior.

The JSON output records accepted/rejected inference, its D4 transform, offset
stencil, coverage, and fallback policy. Semantic kernel identity remains on
each placement; `spatial_cluster` separately identifies a composite cluster.
