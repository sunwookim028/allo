# Generic RTL collection and normalization

This node preserves the existing `design_path`, `manifest`,
`sv2v_include_dirs`, and `top_module` interface. Manifest entries are resolved
relative to `design_path`; files retain manifest order, directory entries expand
one level in sorted order, and `!entry` excludes files.

Relative `design_path` and `manifest` parameters are resolved from the design
constructor. The node packages selected sources and include headers under
`source-rtl`, emits one canonical `design.v`, and records resolved paths,
defines, hashes, modules, and the sv2v version in `rtl-collection.json`.

With `normalize_rtl=True`, sv2v emits the canonical single-file `design.v`.
With `normalize_rtl=False`, the node does not invoke sv2v: it publishes the
ordered Verilog/SystemVerilog closure, packaged headers, defines, and include
directories through `rtl-source-package` and the downstream-ready
`rtl-sources.f`/`rtl-sources.tcl` manifests. The compatibility `design.v`
remains available, but synthesis and RTL simulation consume the source package
so SystemVerilog include and compile-order semantics are preserved.

Packaged copies receive one narrow Design Compiler compatibility rewrite:
`parameter string` and `localparam string` become untyped parameters because
the DC Presto synthesis frontend rejects the typed-string qualifier. The
user-owned source files are not modified, and other SystemVerilog is preserved.
