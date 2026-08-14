# Commercial macro PNR

This Stage 1 node launches one isolated Innovus process for every synthesized
canonical macro class. The existing single-node Innovus implementation is
reused inside each worker, but all work directories and reports remain separate.
The complete selected batch is launched concurrently.

Signal pins are placed from the planner's whole-region Allo channel graph rather
than split arbitrarily between two edges. Each semantic stream's complete Vitis
handshake bundle stays together on its requested N/S/E/W side; controls and
auxiliary interfaces follow explicit planner policies. Unknown, duplicate, or
unassigned RTL ports are fatal. Logical vector names from the pre-synthesis
manifest are expanded into naturally ordered synthesized Innovus terminals
(`name[0]`, `name[1]`, ...) before placement, and the exact resolution is saved
in `reports/pin-assignment.rpt`. Layer allocation is capacity-driven: the
usable edge length and routing pitch are read from the loaded ADK, the required
center pitch is scaled by `pin_min_pitch_multiplier`, and whole semantic groups
are packed across the primary and secondary layers. A stream group is never
split merely to meet a target layer fraction. The resulting pins are explicitly
fixed during batched `editPin` placement so later placement optimization cannot
legally redistribute them. The report records group widths, ADK pitch, required
pitch, and capacity for both layers on every edge.

After abstract export, `check_final_lef_pins.py` compares those planned scalar
assignments with the actual LEF rectangles. Its
`reports/final-lef-pin-check.rpt` output is diagnostic rather than a strict
postcondition. North/south edge pins use the configured vertical-preferred
layer (`pin_vertical_layer_offset`, M4 in FreePDK45); east/west pins use the
horizontal-preferred primary/secondary layers (M3/M5 in FreePDK45). When an
east/west edge contains at least two groups of
`pin_stream_group_min_width` terminals, the default
`pin_spread_multiple_stream_groups` policy requires at least one complete
large group on M5 even if all pins geometrically fit on M3. Small control and
auxiliary groups remain on M3 when possible.
FreePDK45 standard cells use uppercase `VDD` and `VSS`; only those PG-pin names
are requested. Lowercase compatibility aliases are not probed because Innovus
reports an unmatched `globalNetConnect` pattern as an error even after the real
uppercase connection succeeds.

The macro power plan is a sparse edge mesh, not a core ring. One VDD/VSS stripe
pair spans each of two orthogonal upper-metal layers and is exported as LEF PG
geometry. `sroute` connects standard-cell rails to this mesh. This retains the
internal distribution required to power placed cells while avoiding a private,
wide ring and its four large margins around every tiled PE.

After all workers stop, every worker `*.log` is replayed to stdout in stable
class/path order, with begin/end delimiters and from a `finally` path. Thus the
node's `mflowgen-run.log` is complete on success and failure without removing
the individual logs.

Before launching Innovus, each worker now verifies that its synthesized netlist,
SDC, and JSON/Tcl pin intent are real readable files rather than missing or
dangling paths. `START.tcl` catches PNR Tcl errors and explicitly exits nonzero,
stdin is closed to prevent headless Innovus from waiting at an interactive
prompt, and the complete worker process group has a configurable six-hour
`worker_timeout_seconds` watchdog. Start/finish messages remain visible in the
main terminal while detailed logs stay isolated until deterministic replay.

Timing closure uses positive 50 ps setup and hold targets by default. Setup and
hold are repaired after CTS, routing remains timing-driven, and configurable
post-route optimization passes end with hold repair while prohibiting setup-TNS
degradation. Fill cells are inserted only after those passes so optimization
has whitespace available for delay buffers and resizing.

The configured highest macro-routing layer uses a hybrid occupancy abstract.
Four-track tiles keep the macro interior compact, while tiles around exported
pins and their access paths are refined to `top_layer_obs_pin_grid_tracks`
(one track by default). Post-route geometry retains its owning-net name. A pin
opening is rejected if it overlaps another net's clearance-expanded metal, and
each pin must have a continuous clear fine-grid path to a macro boundary. This
replaces the unsafe earlier behavior that removed an entire coarse OBS tile
whenever any pin touched it.

LEF `USE POWER` and `USE GROUND` pins are handled as distributed power-grid
shapes rather than point-to-point signal terminals. Their real top-layer pin
cells are published after verifying matching routed geometry on the owning
net, without fabricating a separate top-layer boundary corridor for every PG
stripe. Signal and clock pins retain the strict unrelated-net clearance and
continuous boundary-path checks.

Signal-pin rectangles retain the technology routing width parallel to the
macro edge and extend farther inward. The primary horizontal layer defaults to
three routing widths of depth; vertical and secondary horizontal layers use two
widths. Pin center pitch and edge capacity therefore remain unchanged while
the detailed router receives a longer landing area for pin escape and vias.
After timing repair and filler insertion, macro PNR runs a targeted
`ecoRoute -target` cleanup before extraction and the mandatory final DRC check.
