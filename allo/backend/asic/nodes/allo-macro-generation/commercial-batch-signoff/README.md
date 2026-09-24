# Commercial batch signoff

This node runs PrimeTime timing signoff followed by Liberty extraction and
Library Compiler DB generation for every hardened macro class. It publishes
logical, timing, parasitic, LEF, and GDS views together. Per-macro power and VCS
simulation are intentionally not part of this Stage 1 implementation.

After all workers stop, every timing, model-extraction, and Library Compiler
`*.log` is replayed to stdout in stable class/path order, with begin/end
delimiters and from a `finally` path. Separate per-worker logs are retained.
