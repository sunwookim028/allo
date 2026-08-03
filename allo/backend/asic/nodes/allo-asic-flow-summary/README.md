# Allo ASIC flow summary

This dependency-free reporting node reads the published macro registry, preserved
tool reports, and explicit node timing metrics. It emits JSON for automation, Tcl
for commercial-flow scripts, and a short human-readable text report. It does not
scrape `mflowgen-run.log`; unavailable measurements remain explicitly unavailable.
It is connected after full-chip GDS merge, DRC, and LVS and records their node
runtimes plus the explicit full-chip DRC result count and LVS status.
