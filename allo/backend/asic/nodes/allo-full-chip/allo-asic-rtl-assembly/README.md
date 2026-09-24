# Full-chip RTL and collateral assembly

This node rewrites each selected concrete PE instance to the canonical hardened
macro module and renames its named ports with the proven registry map. It removes
all replaced PE definitions from the normalized RTL and emits registry-relative
DB, Liberty, LEF, GDS, and functional-Verilog lists. The generated interface
stubs are for inspection/lint only; synthesis should resolve unresolved canonical
instances through the macro DB views.

When FIFO folding is enabled, rewriting also removes each planned top-level
FIFO instance and replaces the complete producer kernel with its canonical
kernel-plus-FIFO macro. The replacement connection list preserves the FIFO's
former consumer-side nets. Assembly checks that the number of deleted FIFOs
and rewritten kernels exactly matches the plan; canonical interface stubs are
read from the published macro functional model because wrapper definitions do
not exist in the original normalized RTL.
