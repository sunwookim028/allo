// ALU-style VPU operations
module vpu_op #(
    parameter int DATA_W = 32,
    parameter int OP_W   = 3
) (
    input  logic [DATA_W-1:0] operand0,
    input  logic [DATA_W-1:0] operand1,
    input  logic [  OP_W-1:0] opcode,
    output logic [DATA_W-1:0] result_out
);

  import vpu_pkg::*;

  // internal signalas for computation result storing
  logic [DATA_W-1:0] result;
  logic [DATA_W-1:0] adder_a, adder_b;
  logic [DATA_W-1:0] adder_result;
  logic [DATA_W-1:0] relu_result;
  logic [DATA_W-1:0] d_relu_result;
  logic [DATA_W-1:0] mul_a, mul_b;
  logic [DATA_W-1:0] mul_result;

  // fp32 adder instance ; this can be adjusted for fxp
  logic [DATA_W-1:0] operand1_neg;
  assign operand1_neg = {~operand1[DATA_W-1], operand1[DATA_W-2:0]};

  assign adder_a = operand0;
  assign adder_b = (opcode == VPU_OP_SUB) ? operand1_neg : operand1;
  /* verilator lint_off PINMISSING */
  fp32_add #(
      .FORMAT("FP32")
  ) fp32_adder (
      .a(adder_a),
      .b(adder_b),
      .result(adder_result)
  );
  /* verilator lint_on PINMISSING */

  assign mul_a = operand0;
  assign mul_b = operand1;

  /* verilator lint_off PINMISSING */
  fp32_mul #(
      .FORMAT("FP32")
  ) fp32_multiplier (
      .a(mul_a),
      .b(mul_b),
      .result(mul_result)
  );
  /* verilator lint_on PINMISSING */

  // ReLU operation
  assign relu_result   = (!operand0[DATA_W-1]) ? operand0 : {DATA_W{1'b0}};

  // ReLU deriv
  assign d_relu_result = (operand0[DATA_W-1] || operand0 == 32'h00000000) ? '0 : 32'h3f800000;

  // opcode decoding + proper operation
  always @(*) begin
    case (opcode)
      VPU_OP_ADD: begin
        result = adder_result;
      end
      VPU_OP_SUB: begin
        result = adder_result;
      end
      VPU_OP_RELU: begin
        result = relu_result;
      end
      VPU_OP_MUL: begin
        result = mul_result;
      end
      VPU_OP_D_RELU: begin
        result = d_relu_result;
      end
      default: begin
        result = {DATA_W{1'b0}};
      end
    endcase
  end

  assign result_out = result;

endmodule

