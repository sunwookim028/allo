# Allo RTL normalization node

This node consumes the backend synthesis RTL directory published by
`allo-asic-compilation`, runs sv2v over the selected module set, and emits one
normalized `design.v`. Vitis retains its complete generated RTL set. Catapult
normalizes only the self-contained `concat_rtl.v`, avoiding duplicate module
definitions from `rtl.v` or simulation wrappers. It forwards both ASIC
manifests and build metadata and preserves the source RTL for debugging.
