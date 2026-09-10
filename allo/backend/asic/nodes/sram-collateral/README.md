# Standard SRAM collateral node

This node gives every downstream ASIC stage one SRAM artifact contract while
keeping SRAM acquisition policy at the graph boundary. It always publishes an
`srams` directory, `sram-contract.json`, `sram-metadata.json`, and a concise
`sram-info.txt` summary.

`sram_mode` selects one of four paths:

- `none` publishes an empty package with `num_srams: 0`.
- `provided` validates and copies `provided_sram_path`, resolved relative to
  the design constructor.
- `bypass` validates and republishes an upstream `inputs/srams` package.
- `generate` reads `sram_manifest` and invokes `generate_method`. Only
  `openram` is supported initially.

Each nonempty SRAM is a directory named for the macro. It must contain a
Verilog simulation model, Liberty and compiled DB timing views, LEF, GDS, and
SPICE or CDL. The JSON contract records every view using paths relative to the
published `srams` directory.

The OpenRAM backend preserves the generation parameters and script from
`openram-sram-generation`. The obsolete cache shortcut has been removed from
both nodes: previously generated collateral belongs in `provided` mode rather
than masquerading as generation. An OpenRAM run checks for the selected Python,
PyYAML, OpenRAM/compiler script, Bash, and `lc_shell` before starting.
