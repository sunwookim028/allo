# Allo ASIC macro planning

This node validates the enriched Allo ASIC manifest against normalized Verilog,
selects proven RTL-equivalence classes whose `member_count` meets
`min_macro_reuse`, and packages one canonical representative per selected class.

Equivalent Vitis modules often use different port names. The node validates
direction and width by ordinal port position, records the explicit mapping, and
replaces noncanonical definitions in `residual-design.v` with zero-logic alias
wrappers. The canonical definition is omitted so that Stage 2 must link the
hardened macro view.

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
