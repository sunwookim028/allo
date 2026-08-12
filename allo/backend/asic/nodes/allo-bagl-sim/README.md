# Allo back-annotated gate-level simulation

This node runs the generated self-checking Allo testbench against the
post-route VCS netlist with required SDF annotation, `xprop=tmerge`, negative
timing-check support, and a VCD suitable for PrimeTime PX. It annotates both the
recomputed top-level SDF and every hardened-macro SDF using exact instance names
from `macro-collateral.json`; `sdf-annotation-manifest.json` records that mapping.

Clock source latency is extracted from `design.pt.sdc`. The testbench contract
supplies the common flow clock period, and BAGL validates it against the node
parameter. AXI memory-BFM outputs are driven after one half period plus the
propagated-clock compensation and input margin, keeping them stable before the
next active edge. Reset, input, and output margins remain configurable. The JSON report separates top
and macro annotation coverage, timing violations, unknown DUT behavior, and SDF
warning categories. Category policies can promote unmatched timing checks,
unmatched IOPATHs, or ignored up-hierarchy interconnect warnings independently.
Optional SRAM and macro functional models are loaded recursively; LVS and
power-ground macro netlists remain excluded to avoid duplicate definitions.
