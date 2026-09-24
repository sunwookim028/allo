# Commercial full-chip PNR

This node begins as a direct copy of `cadence-innovus-pnr`; its script structure
and comments are retained. Stage-2 changes add hardened macro Liberty/LEF/GDS
views, consume the generated physical-intent dimensions and placement Tcl, and
publish runtime and placement reports. The original grouped macro script remains
in place, while the generated plan coordinates optional SRAM perimeter packing
and Allo kernel/PE tiling before standard-cell placement.

The optional `srams` input retains the original directory interface. When it is
absent or empty, physical-intent generation and PNR operate as a PE-only flow.

Hold repair uses `hold_optimization_target_slack` during both post-CTS and
post-route optimization; its default is 0.20 ns. The separate mandatory
`hold_target_slack` defaults to 0.15 ns, providing convergence and extraction
headroom instead of treating barely nonnegative slack as sufficient for SDF
simulation. A full PNR run fails its postconditions if final signoff hold WNS
misses the mandatory target. The metrics JSON records post-CTS, post-route,
and signoff setup/hold summaries. Detailed path reports produced by the setup
and hold `timeDesign` calls are retained under `reports/`.
