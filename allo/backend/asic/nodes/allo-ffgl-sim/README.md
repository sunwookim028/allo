# Allo functional gate-level simulation

This node runs the generated self-checking Allo testbench against the mapped
full-chip synthesis netlist with VCS zero-delay semantics and `xprop=tmerge`.
It recursively loads Verilog/SystemVerilog models from optional `srams/` and
`macro-registry/` directory inputs; a missing or empty optional directory is
silently skipped. LVS and power-ground netlist variants are excluded so they do
not duplicate the functional macro definitions. Success requires the
testbench's `ALLO_TEST_PASS` marker and a nonempty VCD.
