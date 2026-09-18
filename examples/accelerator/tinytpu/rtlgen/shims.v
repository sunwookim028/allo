module mul_f32_f32_f32_l2(
  input [31:0] a,
  input [31:0] b,
  input clk,
  input ce,
  output [31:0] y
);
  mul_f32_f32_f32_l2_core u (
    .aclk(clk),
    .aclken(ce),
    .s_axis_a_tvalid(1'b1),
    .s_axis_a_tdata(a),
    .s_axis_b_tvalid(1'b1),
    .s_axis_b_tdata(b),
    .m_axis_result_tvalid(),
    .m_axis_result_tdata(y)
  );
endmodule

module add_f32_f32_f32_l2(
  input [31:0] a,
  input [31:0] b,
  input clk,
  input ce,
  output [31:0] y
);
  add_f32_f32_f32_l2_core u (
    .aclk(clk),
    .aclken(ce),
    .s_axis_a_tvalid(1'b1),
    .s_axis_a_tdata(a),
    .s_axis_b_tvalid(1'b1),
    .s_axis_b_tdata(b),
    .s_axis_operation_tvalid(1'b1),
    .s_axis_operation_tdata(8'b00000000),
    .m_axis_result_tvalid(),
    .m_axis_result_tdata(y)
  );
endmodule

module sub_f32_f32_f32_l2(
  input [31:0] a,
  input [31:0] b,
  input clk,
  input ce,
  output [31:0] y
);
  sub_f32_f32_f32_l2_core u (
    .aclk(clk),
    .aclken(ce),
    .s_axis_a_tvalid(1'b1),
    .s_axis_a_tdata(a),
    .s_axis_b_tvalid(1'b1),
    .s_axis_b_tdata(b),
    .s_axis_operation_tvalid(1'b1),
    .s_axis_operation_tdata(8'b00000001),
    .m_axis_result_tvalid(),
    .m_axis_result_tdata(y)
  );
endmodule

module cmp_f32_f32_u1_l1_ugt(
  input [31:0] a,
  input [31:0] b,
  input clk,
  input ce,
  output y
);
  wire [7:0] result;
  cmp_f32_f32_u1_l1_core u (
    .aclk(clk),
    .aclken(ce),
    .s_axis_a_tvalid(1'b1),
    .s_axis_a_tdata(a),
    .s_axis_b_tvalid(1'b1),
    .s_axis_b_tdata(b),
    .s_axis_operation_tvalid(1'b1),
    .s_axis_operation_tdata(8'b00100100),
    .m_axis_result_tvalid(),
    .m_axis_result_tdata(result)
  );
  assign y = result[0:0];
endmodule

module cmp_f32_f32_u1_l1_uno(
  input [31:0] a,
  input [31:0] b,
  input clk,
  input ce,
  output y
);
  wire [7:0] result;
  cmp_f32_f32_u1_l1_core u (
    .aclk(clk),
    .aclken(ce),
    .s_axis_a_tvalid(1'b1),
    .s_axis_a_tdata(a),
    .s_axis_b_tvalid(1'b1),
    .s_axis_b_tdata(b),
    .s_axis_operation_tvalid(1'b1),
    .s_axis_operation_tdata(8'b00000100),
    .m_axis_result_tvalid(),
    .m_axis_result_tdata(result)
  );
  assign y = result[0:0];
endmodule
