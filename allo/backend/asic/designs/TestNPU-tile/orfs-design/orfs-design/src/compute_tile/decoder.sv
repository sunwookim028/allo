`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: decoder
// Description: Combinational ISA decoder. Slices a 64-bit instruction word into
//              opcode / flags / op0 / op1 / op2 fields per the layout in
//              `mininpu/isa.py` (see also `npu_isa_pkg.sv`):
//
//                 63      56 55      48 47      32 31      16 15       0
//                 ┌────────┬──────────┬──────────┬──────────┬──────────┐
//                 │opcode:8│ flags:8  │ op0:16   │ op1:16   │ op2:16   │
//                 └────────┴──────────┴──────────┴──────────┴──────────┘
//
//              Per-instruction interpretation (matmul row indices, flag bits,
//              etc.) is derived in the sequencer; this module is layout-only.
//////////////////////////////////////////////////////////////////////////////////

module decoder (
    input  logic [63:0] instr_decode,
    output logic [ 7:0] op_decode,
    output logic [ 7:0] flags_decode,
    output logic [15:0] op0_decode,
    output logic [15:0] op1_decode,
    output logic [15:0] op2_decode
);

  import npu_isa_pkg::*;

  assign op_decode    = instr_decode[FIELDS_INSTR_OP_LSB+:FIELDS_INSTR_OP_WIDTH];
  assign flags_decode = instr_decode[FIELDS_INSTR_FLAGS_LSB+:FIELDS_INSTR_FLAGS_WIDTH];
  assign op0_decode   = instr_decode[FIELDS_INSTR_OP0_LSB+:FIELDS_INSTR_OP0_WIDTH];
  assign op1_decode   = instr_decode[FIELDS_INSTR_OP1_LSB+:FIELDS_INSTR_OP1_WIDTH];
  assign op2_decode   = instr_decode[FIELDS_INSTR_OP2_LSB+:FIELDS_INSTR_OP2_WIDTH];

endmodule
