# Commercial back-annotated gate-level simulation

Runs the routed netlist with top-level SDF, ADK models, optional SRAM models,
four-state checking, negative timing checks, categorized SDF warnings, timing
violation detection, and the common user testbench contract. Generic phase-1
BAGL annotates only the top-level SDF; per-hard-macro SDF annotation belongs to
the phase-2 macro registry.

Technology-specific standard-cell model defines are read from optional common
`vcs-compile.args` and BAGL-only `vcs-bagl.args` files in the selected ADK. The
generic node owns strict SDF annotation and runtime timing-violation policy, but
does not hard-code library macros such as FreePDK45/Nangate's `NTC`, `RECREM`,
or `TETRAMAX`.
