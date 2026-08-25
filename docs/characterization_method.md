<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# How the device numbers are measured

Every latency, delay and area in `allo/backend/rtl/devices/` comes from one of
two measurement campaigns run against the vendor tools. This document says what
each campaign measures, how to rebuild it, and which of its choices are free and
which are load bearing. It exists so that a number in a fabric table can be
traced back to a path in a real netlist, and so that someone standing up the
harness on a new part does not have to rediscover the traps.

Two campaigns, because the compiler asks two different questions. Native
combinational operators (an adder, a comparator, a barrel shifter) are logic the
emitter builds itself, so we characterize the logic. Operator cores (a
floating-point adder, an integer divider) are vendor IP the emitter instantiates
behind an extern module, so we characterize the delivered core. The two share a
resource vocabulary and almost nothing else.

## 1. Combinational logic

### 1.1 What is being measured

The compiler asks two questions of every native operator kind, and both are
functions of the operand width, so each kind is swept over a set of widths and
what lands in the device table is a curve rather than a number.

The timing question is how much of a clock period one instance eats when it sits
between two registers. The path we want is the whole register to register cone:
the launching flip-flop's clock to out, the routing into the structure, the
structure's own logic, the routing out, and the capturing flip-flop's setup.
That is exactly what a scheduler is deciding when it asks whether two operations
can chain inside one cycle, so measuring anything narrower would understate it.

The register floor, meaning the same path with no structure in it at all, is
measured separately as its own wire DUT. The model charges the floor once per
cycle rather than once per operator, so a chain of n operators costs the floor
plus n marginal steps, not n full paths. Keeping the floor as a separate
measurement is what makes that decomposition possible.

The area question is how many cells of each kind one instance occupies, and here
we want the structure alone with no surrounding registers counted. That is why
the two questions get two different DUT shapes rather than one shared shape.

### 1.2 Verilog templates

The area DUT is bare combinational logic with no clock and no registers:

```verilog
module a_add_w128(
  input [127:0] i0,
  input [127:0] i1,
  output [127:0] y
);
  assign y = i0 + i1;
endmodule
```

The delay DUT is exactly register, structure, register:

```verilog
module d_add_w32(
  input clk,
  input [31:0] i0_i,
  input [31:0] i1_i,
  output [31:0] y
);
  (* dont_touch = "yes" *) reg [31:0] i0;
  (* dont_touch = "yes" *) reg [31:0] i1;
  (* dont_touch = "yes" *) reg [31:0] o;
  always @(posedge clk) begin
    i0 <= i0_i;
    i1 <= i1_i;
    o <= i0 + i1;
  end
  assign y = o;
endmodule
```

The `dont_touch` on all three registers is load bearing rather than defensive.
Without it the operand registers are absorbed into the logic or retimed across
it, and the path that gets timed is no longer the path that was asked about. The
floor DUT is the same shape with the operation replaced by a plain assignment,
so that what remains is clock to out plus routing plus setup.

Generating these from a script rather than writing them by hand matters for one
reason beyond convenience: the width sweep is a single list, and every place
that list is duplicated is a place where a later widening silently covers only
half the sweep.

### 1.3 TCL and Vivado commands

Area mode synthesizes and stops:

```tcl
create_project -in_memory -part $part
read_verilog -quiet -sv duts/$dut.v
synth_design -top $dut -part $part -mode out_of_context \
             -flatten_hierarchy none -no_lc
```

Both synthesis flags are there to keep the cell count attributable. `-no_lc`
stops two unrelated logic functions from being packed into one LUT, which would
make an eight bit operator look cheaper next to a sixty four bit one for reasons
that have nothing to do with either. `-flatten_hierarchy none` keeps the
structure's own boundary so the count is the structure's and not the wrapper's.

Delay mode runs the full implementation, because a synthesis estimate does not
include routing and routing is a large fraction of the path at these widths:

```tcl
synth_design -top $dut -part $part -mode out_of_context
create_clock -period $period -name clk [get_ports clk]
opt_design -quiet
place_design -quiet
route_design -quiet
```

The path query is generic on purpose, so that it survives every renaming and
every DUT shape in the sweep:

```tcl
set regs [get_cells -hier -quiet -filter {IS_SEQUENTIAL}]
set p [lindex [get_timing_paths -quiet -delay_type max -max_paths 1 \
                   -from $regs -to $regs] 0]
get_property DATAPATH_DELAY $p
get_property LOGIC_LEVELS $p
```

Two details in that query are worth keeping. The startpoint set is every
sequential cell rather than `all_registers`, because a block RAM is a startpoint
the flip-flop list does not hold, and the read path out of one is exactly what a
storage row's read delay is. And `DATAPATH_DELAY` is the number we want rather
than the slack, since the slack depends on the clock period we happened to pick
and the datapath delay does not.

The cell histogram is a plain walk:

```tcl
foreach c [get_cells -hier -quiet] {
  dict incr h [get_property REF_NAME $c]
}
```

For storage realizations one number is not enough, because a RAM's read path
starts at the RAM and its write path ends there, and which of the two is worse
is not the question either row is asking. Those DUTs are timed per (startpoint
reference, endpoint reference) pair, keeping the worst path of each kind, with
`-max_paths` deep enough that a slower path kind cannot crowd out a faster one
(a block RAM's read endpoints will otherwise fill the whole list and hide the
write path entirely).

### 1.4 Cautions

`read_verilog` accumulates into the session's source set, and a file that fails
to parse stays in it. Every later `synth_design` in the same session then dies
with a module not found error, so one malformed DUT costs every job queued
behind it. Recreating the in-memory project per job is what drops the bad file,
and it is cheap.

Changing the part inside a live session is not supported by every Vivado
release and has been seen to crash outright, so the sweep runs one session per
part and shards within a part.

Pick the clock period loosely. The number being recorded is the datapath delay,
not the slack, so a period that fails timing wastes the run without improving
the measurement, while a period that is comfortably met still routes normally.

Any cap in the harness on which jobs run has to be derived from the sweep
definition rather than written as a literal. A hardcoded width limit on the
implementation jobs will silently skip every wider delay point and record only
the area, and a sweep that cannot widen reports success either way.

Check that the structure survives synthesis. An operator whose operands the
tool can constant fold, or whose output is unused, measures as nothing at all,
and a zero in an area column reads exactly like a legitimately free operator.

## 2. Catalog IP

### 2.1 What is being measured

An operator core arrives as a synthesized checkpoint, so unlike the fabric
sweep there is nothing to write. The compiler needs three things from it.

The latency in cycles is an input rather than an output: the core is requested
at a specific depth and the run confirms it built at that depth. This matters
because the compiler treats a declared latency as a correctness contract, and a
core that quietly built deeper than asked would be sampled early by every
consumer the scheduler placed.

The area is the core's own cells, excluding anything the measurement harness
puts around it, counted from the same routed netlist the arcs are timed on. Two
things follow from taking it there rather than from a run of its own. It is the
core as a design gets it, with the operation channel tied to the constant the
emitter ties it to, so the modes a row never uses are optimized away in the
measurement exactly as they are in a build: a double compare counts 23 LUTs, not
the 117 the general core holds. And the count is in the same resource vocabulary
as every fabric number, so a slice mux is `muxf` rather than folded into `lut`.

The timing is three arcs, not one number. The path is again register to
register through the core, for the same reason as the fabric sweep: that is how
the emitter instantiates it, between real registers, and the routing on both
sides is part of what the schedule has to fit. But the compiler charges the
three segments of that path in three different places, so they are timed apart:

| arc | from | to | declared as |
| --- | --- | --- | --- |
| in | the wrapper's register | the core's first internal register | `in_delay_ns`, less the register floor |
| int | the core's internal registers | each other | `min_period_ns` |
| out | the core's last internal register | the wrapper's register | `out_delay_ns` |

The in arc is what a chain in front of the core has to share a cycle with, the
out arc is what a chain behind it starts from, and the int arc is a path no
schedule can split, so it is a floor on the clock rather than a term in a
chain. Their max is the period the row needs for a cycle of its own, which is
the whole register-to-register path -- so measuring the three separately loses
nothing and tells the scheduler where the time goes.

One number for all three is the failure this replaced. A single worst-path
query answers only "does the row hold this clock", and a row that has to answer
it at some clock ends up warranted at whatever clock the campaign happened to
run at. That number then reads as a physical limit later. It is not one: it is
a record of the campaign, and a core measured at 300 MHz whose internal arc is
0.9 ns will refuse a 500 MHz design for no reason anyone can see. Time the arcs,
declare the arcs, and let the model take the max.

Since an IP cannot be generated between registers, the shape has to be built
around it.

### 2.2 TCL and Vivado commands

Create and configure the core, then synthesize it out of context:

```tcl
create_ip -name $core -vendor xilinx.com -library ip -version * \
          -module_name $name -dir $ipdir
set_property -dict $cfg [get_ips $name]
synth_ip [get_ips $name]
```

`create_ip -dir` fails unless the directory already exists, which is worth
handling once rather than per job.

Out-of-context synthesis writes a stub next to the checkpoint, and the stub is
the port list. Reading it is more robust than deriving port names from the core
version, because the names change between configurations:

```
  input aclk /* synthesis syn_isclock = 1 */;
  input aclken;
  input s_axis_a_tvalid;
  input [31:0]s_axis_a_tdata;
  output m_axis_result_tvalid;
  output [31:0]m_axis_result_tdata;
```

A line matcher over `(input|output)`, an optional range, and a name recovers
that. From it, generate a wrapper in which every data port passes through a
register and the clock and enable pass straight through:

```verilog
module wrap_fadd_l7(
  input wclk,
  input aclken,
  input [31:0] s_axis_a_tdata_i,
  ...
  output [31:0] m_axis_result_tdata
);
  (* dont_touch = "yes" *) reg [31:0] q_s_axis_a_tdata;
  (* dont_touch = "yes" *) reg [31:0] q_m_axis_result_tdata;
  always @(posedge wclk) begin
    q_s_axis_a_tdata <= s_axis_a_tdata_i;
    q_m_axis_result_tdata <= w_m_axis_result_tdata;
  end
  fadd_l7 u_core (.aclk(wclk), .aclken(aclken),
                  .s_axis_a_tdata(q_s_axis_a_tdata), ...,
                  .m_axis_result_tdata(w_m_axis_result_tdata));
  assign m_axis_result_tdata = q_m_axis_result_tdata;
endmodule
```

Then synthesize the wrapper in the same in-memory project. Vivado's non-project
flow links the already synthesized IP checkpoint automatically, so the core is
brought in as a netlist rather than rebuilt from RTL:

```tcl
read_verilog wrap.v
synth_design -top wrap_$name -part $part -mode out_of_context
if {[get_property IS_BLACKBOX [get_cells u_core]]} { error "core not linked" }
create_clock -period $period -name wclk [get_ports wclk]
opt_design -quiet
place_design -quiet
route_design -quiet
```

The blackbox assertion is the guard that makes the whole method trustworthy. If
the core were rebuilt from RTL instead of linked, the wrapper's registers could
be absorbed into a DSP block and the path timed would no longer be the core's.
Failing loudly there is much better than a plausible number.

The timing query is the same generic worst sequential to sequential path as the
fabric sweep, since the wrapper has now given the design that shape.

Area has to exclude the wrapper, and an IP checkpoint keeps the core's internal
hierarchy, so an unfiltered walk would count module instances alongside the
primitives inside them:

```tcl
get_cells -hier -quiet -filter {IS_PRIMITIVE && NAME =~ u_core/*}
```

One validation is worth running whenever the method changes: re-measure a set of
cores that already have numbers and compare the primitive histograms. If the
histograms are identical, the core was linked and not rebuilt, and any change in
frequency is a change in what path was visible rather than a change in what was
built.

### 2.3 Options that have to agree with the scheduler and the emitter

The core is configured, and some of those configuration choices are not free.
They describe a contract that the scheduler assumes and the emitter emits, so
changing one side without the other produces a design that simulates and then
fails in hardware, or an area number for a core nobody instantiates.

Flow control. Today the only supported contract is a clock enable, so the cores
are built with `Flow_Control=NonBlocking`, `Has_ACLKEN=true` and
`Has_RESULT_TREADY=false`. That is a fixed latency pipeline with a global stall
input and no back pressure, which is precisely what the emitter's `ce` style
emits and what the scheduler assumes when it pins every consumer to a fixed
sampling cycle. There is also a `free` style, meaning the same pipeline with no
enable at all, for cores that sit outside any stalling region.

An `elastic` style, meaning a real valid and ready handshake with variable
latency, is declared in the vocabulary and rejected everywhere. It is worth
understanding why before anyone adds cores for it. The scheduler currently pins
each consumer to the core's declared latency, and the emitter emits the free
running port shape, so declaring a core elastic today would produce a design
that samples the result at a static cycle regardless of when it actually
arrives. Supporting it means a node level elasticity contract in the scheduler
first, a handshake port shape in the emitter, and only then a core built with
back pressure enabled. All three have to land together, and the characterization
changes with them, because a core with `Has_RESULT_TREADY=true` has different
area and a different critical path.

Latency pinning. The core must be requested at an explicit depth and confirmed
at it, because the compiler's declared latency is a correctness contract rather
than an estimate. On the Xilinx floating point core this means clearing the
maximum latency flag before setting the depth, since the depth parameter is
disabled until then. Left alone, every core silently builds at its own maximum
and the whole table describes cores nobody asked for.

Pipelining. The device model declares whether one instance accepts a new input
every cycle, and the scheduler uses that both to bound a cyclic region's
initiation interval and to decide whether an operator may be shared. The
non-blocking configuration above is fully pipelined, so the declaration and the
core agree. A core built in a mode that accepts an input only every k cycles has
to be declared non-pipelined or the emitter will feed it faster than it can
accept.

Two mismatches worth knowing about, both currently unresolved. First, the core
carries a valid pipeline (`s_axis_*_tvalid` in, `m_axis_result_tvalid` out, one
bit deep by the core's latency) that our extern module has no port for, and the
area we charge includes it. On a latency one compare, where the shipped area is
two flip-flops, the valid register is one of the two. Second, on this core
family a floating point add and subtract are the same physical core selected by
a runtime operation port, which is why the two rows carry identical area; we
declare them as two operators, so a design using both is charged for two cores
where one plus a selector would do.

More generally, our extern module's port shape is data in, clock, enable, data
out, while the core presents AXI stream names and the extra ports above.
Instantiating a real core behind our extern therefore needs a thin adapter that
renames, ties the valid inputs high and drives the operation port to a constant.
That is a known gap in the external IP story rather than a defect in the
measurement.

### 2.4 Vendor tool behaviour to work around

Property order matters twice over. `set_property -dict` applies its list in
order, and on the floating point core changing the operation type resets the
latency to the new type's default. So the shape of the core has to be settled
first and the depth set last. This is easy to miss because nothing reports it;
the run succeeds and produces a core at the wrong depth.

Parameters can be disabled by other parameters. The latency field is inert until
the maximum latency flag is cleared, as described above.

The available precision list is version dependent. A core version that offers
half, single, double and custom does not offer bfloat16, which then has to be
spelled as the custom format it is, with eight exponent bits and eight fraction
bits counting the hidden one.

Clock port names vary with the interface. An AXI stream core calls it `aclk`, a
plain one `clk` or `CLK`, so the harness should look for any of them rather than
assume.

`get_timing_paths` ranks by SLACK, and every arc here wants the longest DELAY.
The two orders agree only when every endpoint carries the same clock skew and
setup requirement, which across a core's internal registers they do not, so
`-max_paths 1` can return a path shorter than the worst one. The check that
catches it is arithmetic: the three arcs of one core must have the same maximum
as the whole register-to-register path, and taking each over a deep path list
is what makes them.

Finally, the failure that motivated the wrapper in the first place. If the
design is constrained with a clock and nothing else, `get_timing_paths` returns
register to register paths only, because port to register and register to port
paths are unconstrained without input and output delays. A core whose registers
all sit on one side of its arithmetic, or which is absorbed entirely into a DSP
block and uses that block's internal pipeline, then has no such path and the
query returns nothing. Writing that as a sentinel is where the real damage
starts, for two reasons. The sentinel gets read as a fact later, and it was read
two incompatible ways in the same table: as combinational with no maximum for
the compares, whose rows shipped annotated that way, and as unknown for the
multipliers, whose shorter candidates were declined. And where a path does exist
but is not the one you meant, no sentinel appears at all: the compares reported
the delay between two valid pipeline flip-flops as if it were the core's
frequency. That is an artifact rather than an optimistic measurement, and no
inspection of the numbers separates the two. Only re-measuring does.

## 3. Resource vocabularies across architectures

The device model treats a resource as a name with a capacity and nothing else,
so the vocabulary is the fabric's own and the compiler only adds and multiplies
what it is given. That makes adding a new architecture mostly a matter of
naming, but three things do have to be got right.

### 3.1 What each family declares

Primary resources are what the part quotes. Derived resources are computed from
a primary one because the part does not quote them separately, and the divisor
is a property of the architecture rather than of the die.

| model resource | kind | 7 series | UltraScale Plus | Versal |
| --- | --- | --- | --- | --- |
| `lut` | primary | quoted | quoted | quoted |
| `ff` | primary | quoted | quoted | quoted |
| `dsp` | primary | quoted | quoted | quoted |
| `bram36` | primary | quoted | quoted | quoted |
| `uram288` | primary | absent | quoted | quoted |
| `carry4` | derived | `lut` / 4 | not declared | not declared |
| `carry8` | derived | not declared | `lut` / 8 | `lut` / 8 |
| `slicem_lut` | derived | `lut` / 2 | `lut` / 2 | `lut` / 2 |

A part that lacks a resource declares it absent rather than zero, and every
storage realization that requires it is then left undeclared. So a 7 series part
simply has no ultra RAM storage row rather than one that can never be chosen.
The carry rows differ in name and not only in divisor, which keeps a cost
written against one architecture from silently evaluating on another.

### 3.2 Counting a netlist into those resources

The cell histogram from a placed design maps onto the model like this. Counting
is where the families actually differ, so the last column is the one to read
when standing up a new architecture.

| model resource | cells counted | per family |
| --- | --- | --- |
| `lut` | LUT1 through LUT6, LUT6_2, LUT6CY, MUXF7, MUXF8, SRL16E, SRLC32E | LUT6CY is Versal only |
| `ff` | FDRE, FDSE | same everywhere |
| `dsp` | the top level slice cell only | DSP48E1, DSP48E2, DSP58 |
| `carry4` | CARRY4 | 7 series |
| `carry8` | CARRY8, or LOOKAHEAD8 | LOOKAHEAD8 is Versal |
| nothing | GND, VCC, LUTCY1, LUTCY2, any `DSP_*` leaf, DSP58C | tie-offs and sub-cells |

Four of those rows are traps rather than conveniences. The shift register
primitives count as lookup tables because that is the site they occupy, not
because they are logic. The `DSP_*` leaves such as `DSP_MULTIPLIER` or
`DSP_OUTPUT` are components inside one slice, so counting them multiplies the
real usage severalfold; DSP58C is likewise a distinct cell and not the plain
Versal slice. LUTCY1 and LUTCY2 are the two halves of a LUT6CY and counting
all three double counts. And Versal's carry appears as LOOKAHEAD8 alongside
LUT6CY rather than as a CARRY cell at all, which is why it maps onto `carry8`
by name: the LOOKAHEAD8 count of a given core matches the CARRY8 count of the
same core on UltraScale row for row.

### 3.3 Storage realizations

Storage is tiled rather than counted, since one tile serves any shape of array
that fits inside it.

| realization | resource it spends | bits per tile | available on |
| --- | --- | --- | --- |
| register file | `lut` and `ff` | not tiled, linear in depth and width | all |
| distributed RAM | `slicem_lut` | 64 | all |
| shift register | `slicem_lut` | 32 | all |
| block RAM | `bram36` | 36864 | all |
| ultra RAM | `uram288` | 294912 | UltraScale Plus, Versal |

Shift register extraction has a depth threshold below which the chain stays in
flip-flops. Above it, the site count per bit is the depth divided by 32, rounded
up. That is a formula rather than a table, and writing it as a table means it
stops being true past the last sampled depth.

### 3.4 Differences that reach past the vocabulary

Two architectural facts are not expressible by renaming a resource.

Versal's DSP58 has native floating point primitives, so a multiply accumulate
binds to a single fused core rather than to a multiplier and an adder, and a
plain floating point add runs at a depth the other families cannot reach. Any
model that assumes an add and a multiply are separately bound will misprice that
part. The usual latency against area trade also does not appear there at all,
because the primitive is simultaneously the fastest and the cheapest option.

In the other direction, on a DSP poor part the DSP implementation of an operator
can be the shorter one while the fabric implementation is the cheaper one, which
is the reverse of the UltraScale case. Neither implementation is intrinsically
the fast one or the small one. Which is which is a property of the part and the
target period, and that is the reason the library carries candidate rows rather
than a single row per operator.
