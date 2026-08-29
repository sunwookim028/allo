# Allo ASIC physical intent

Within each systolic kernel, PID columns increase east and PID rows increase
south. Innovus coordinates increase north, so the planner emits larger PID rows
at smaller Y coordinates. This makes the west/east and north/south stream pin
bundles generated during macro planning face their connected neighbors.

This dependency-free planner reads actual LEF dimensions, separation parameters,
the whole-region stream graph, the synthesized netlist, and optional SRAM views.
It packs SRAM instances sequentially around the perimeter and uses the
`stream_grid` PE-macro placement algorithm by default. Innovus remains
responsible for final manufacturing-grid and routing legality.

## Stream-grid macro placement

The stream-grid placer treats every hardened macro instance as a graph node and
every pre-HLS stream as a weighted edge. Stream bit width is the edge weight.
Concrete PID deltas provide north/south/east/west ordering within mapped kernels;
directional kernel suffixes are used only for boundary kernels because the
current whole-region manifest does not yet carry a cardinal-side field.

Placement uses two coarse physical grids. A macro occupies the number of cells
required by its exact LEF dimensions plus `macro_separation_x/y`. The deliberately
small normalized objective is:

```
stream distance + bounding-box area + 2 * overlap + direction violations
```

Only strictly improving moves are accepted. An active queue reconsiders only a
moved macro, its graph neighbors, and nearby overlapping macros. Each level ends
when the queue empties or its bounded evaluation budget is reached. Residual
overlaps are legalized deterministically, largest rectangles first. Final
horizontal and vertical compaction moves macros by the whole removable gap and
keeps only cost-improving moves; it does not perform repeated one-site scoots.

The JSON `kernel_packing_policy.optimization` record includes both grid pitches,
evaluations, accepted moves, convergence status, normalized cost components,
remaining coarse-grid overlap, legalization/compaction moves, final region size,
and macro coverage. Coarse-grid overlap is expected and does not imply an illegal
final placement.

Set `macro_placement_algorithm=legacy_cluster_grid` to restore the previous rigid
semantic-kernel clustering and slot-swap implementation. The legacy D4
interleaver remains available on that path. The following optional knobs bound
the new optimizer without changing its cost function:

- `stream_grid_max_passes` (default 16): maximum active evaluations per macro
  at each resolution.
- `stream_grid_minimum_improvement` (default `1e-5`): minimum normalized gain
  required to accept a move.

When `enable_kernel_rotation` is true, the planner may rotate a complete
semantic kernel cluster, including hardened HLS-only child modules attributed
to that kernel. Legal choices come from the published LEF `SYMMETRY`. The
deterministic objective combines packed region area with weighted distance
between kernels connected in the pre-HLS whole-region stream graph.

On the `legacy_cluster_grid` path, `interleave_macros` defaults to `False`,
preserving the original rigid one-cluster-per-semantic-kernel placement. When
explicitly enabled, the planner
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
