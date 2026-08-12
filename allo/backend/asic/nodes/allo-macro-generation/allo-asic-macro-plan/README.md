# Allo ASIC macro planning

This node validates the enriched Allo ASIC manifest against normalized Verilog,
selects proven RTL-equivalence classes whose `member_count` meets
`min_macro_reuse`, and packages one canonical representative per selected class.

Equivalent Vitis modules often use different port names. The node validates
direction and width by ordinal port position, records the explicit mapping, and
replaces noncanonical definitions in `residual-design.v` with zero-logic alias
wrappers. The canonical definition is omitted so that Stage 2 must link the
hardened macro view.

`harden_repeated_hls_submodules` controls identical modules that Vitis creates
multiple times inside one semantic Allo PE. It defaults to `False`. When it is
enabled, the plan labels these as `repeated_hls_submodule` and retains their
owning Allo kernel, rather than treating anonymous HLS pipeline modules as new
semantic PEs.

For every canonical representative, the node also maps semantic stream ordinals
onto Vitis FIFO/valid RTL bundles and emits `pin-intent.json` plus a dependency-
free `pin-intent.tcl`. Explicit compiler compass directions take precedence;
otherwise same-kernel PID displacement, stream-axis hints, and a recorded
dataflow fallback are used in that order. Clock/control pins use the south side.
Non-stream interfaces do not affect equivalence or D4 selection; each AXI
channel and logical vector remains intact while those groups are load-balanced
across sides not occupied by semantic stream traffic. The PNR worker validates
that every RTL port is assigned exactly once.
It also emits an Innovus D4 orientation for every equivalent member by proving
that the member's desired stream sides are a rotation/reflection of the
canonical pin pattern.

`fold_fifos_into_macro` defaults to `False`. When enabled for a hierarchical
run, semantic PE candidates are promoted from their inner Vitis pipeline to the
complete outer kernel module. The planner locates every FIFO driven by an Allo
`put` bundle, generates a producer-owned kernel-plus-FIFO wrapper, and hardens
that wrapper as the canonical macro. Its external stream pins are the folded
FIFO's dequeue-side interface. Allo's point-to-point stream rule is treated as
a compiler contract; the planner still requires an unambiguous realized RTL
binding, recognized FIFO ports, and matching kernel/FIFO clock and reset nets.

Wrappers with different external shapes or FIFO module sets are split into
separate derived equivalence classes before applying `min_macro_reuse`. Module
names remain canonical-class based. Concrete instances are named
`<kernel>_<pid...>` during Stage-2 assembly—for example `compute_3_5`—without
sacrificing macro reuse. Bypass/flat mode always disables folding.
