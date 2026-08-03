# Full-chip RTL and collateral assembly

This node rewrites each selected concrete PE instance to the canonical hardened
macro module and renames its named ports with the proven registry map. It removes
all replaced PE definitions from the normalized RTL and emits registry-relative
DB, Liberty, LEF, GDS, and functional-Verilog lists. The generated interface
stubs are for inspection/lint only; synthesis should resolve unresolved canonical
instances through the macro DB views.
