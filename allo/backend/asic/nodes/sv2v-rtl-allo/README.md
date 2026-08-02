# Allo RTL normalization node

This node consumes the backend synthesis RTL directory published by
`allo-asic-compilation`, runs sv2v over the complete module set, and emits one
normalized `design.v`. It forwards both ASIC manifests and build metadata and
also preserves a copy of the unnormalized source RTL for debugging.
