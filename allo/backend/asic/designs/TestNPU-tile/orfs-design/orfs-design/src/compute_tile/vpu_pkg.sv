// Private constants for the VPU subsystem (vpu.sv + vpu_op.sv).
//
// As of Stage 0 of the NPU-ISA roadmap the legacy VPU/VLOAD/VSTORE/VCOMPUTE
// dispatch is no longer driven by the ISA decoder; the modules remain as RTL
// stubs reusable by Stage 2 once `vec.*` opcodes are wired in. These
// constants used to live in `npu_isa_pkg.sv` (FIELDS_VPU_*, VPU_TYPE_*,
// VPU_OP_*); they were moved here when the public ISA was redesigned so the
// VPU subsystem stays self-contained until the Stage 2 rewrite.

package vpu_pkg;

  // VPU sub-type values driven on the `vpu_type` input port.
  localparam logic [2:0] VPU_TYPE_SCALAR = 3'd0;
  localparam logic [2:0] VPU_TYPE_VLOAD = 3'd1;
  localparam logic [2:0] VPU_TYPE_VSTORE = 3'd2;
  localparam logic [2:0] VPU_TYPE_VCOMPUTE = 3'd3;

  // VPU ALU opcode values driven on the `vpu_opcode` / `opcode` ports.
  // Bit width matches the historical 3-bit VPU OPCODE field.
  localparam int VPU_OPCODE_WIDTH = 3;

  localparam logic [VPU_OPCODE_WIDTH-1:0] VPU_OP_ADD = 3'd0;
  localparam logic [VPU_OPCODE_WIDTH-1:0] VPU_OP_SUB = 3'd1;
  localparam logic [VPU_OPCODE_WIDTH-1:0] VPU_OP_RELU = 3'd2;
  localparam logic [VPU_OPCODE_WIDTH-1:0] VPU_OP_MUL = 3'd3;
  localparam logic [VPU_OPCODE_WIDTH-1:0] VPU_OP_D_RELU = 3'd4;

endpackage
