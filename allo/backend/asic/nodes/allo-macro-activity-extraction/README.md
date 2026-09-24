# Allo macro activity extraction

This node selects the first assembly-plan member as the representative for each
canonical macro class and uses Synopsys `vcd2saif` to extract that instance's
BAGL activity. The output manifest records the exact VCD scope, SAIF filename,
canonical module, and reuse count used downstream.

The BAGL simulation must preserve hardened-macro hierarchy. A missing scope is
treated as an extraction failure instead of allowing PrimeTime to fall back to
unannotated/default activity.
