module rf_0_run (
  clk, rst, done, v12, v13, v14, v15, v17, v18
);
  input clk;
  input rst;
  output done;
  reg done;
  input [4:0] v12;
  input [4:0] v13;
  input [15:0] v14;
  input v15;
  output [15:0] v17;
  reg [15:0] v17;
  output [15:0] v18;
  reg [15:0] v18;


  // Interconnect Declarations
  wire [1:0] fsm_output;
  wire or_dcpl_6;
  wire or_dcpl_9;
  reg [1:0] while_v23_1_0_sva;
  reg v15_svs;
  reg [15:0] mgc_stateful_rf_0_tag_2_1_sva;
  reg [15:0] mgc_stateful_rf_0_tag_2_2_sva;
  reg [15:0] mgc_stateful_rf_0_tag_2_0_sva;
  reg [15:0] mgc_stateful_rf_0_tag_2_3_sva;
  reg [15:0] while_v25_sva;
  wire [15:0] mgc_stateful_rf_0_tag_2_3_sva_mx1;
  wire [15:0] mgc_stateful_rf_0_tag_2_2_sva_mx1;
  wire [15:0] mgc_stateful_rf_0_tag_2_1_sva_mx1;
  wire [15:0] mgc_stateful_rf_0_tag_2_0_sva_mx1;

  wire[15:0] while_v30_mux_nl;
  wire[15:0] operator_33_true_acc_nl;
  wire[16:0] nl_operator_33_true_acc_nl;
  wire or_7_nl;
  wire or_8_nl;
  wire or_10_nl;
  wire or_11_nl;

  // Interconnect Declarations for Component Instantiations 
  rf_0_run_run_fsm rf_0_run_run_fsm_inst (
      .clk(clk),
      .rst(rst),
      .fsm_output(fsm_output)
    );
  assign or_7_nl = or_dcpl_6 | (~ (while_v23_1_0_sva[0]));
  assign mgc_stateful_rf_0_tag_2_3_sva_mx1 = MUX_v_16_2_2(while_v25_sva, mgc_stateful_rf_0_tag_2_3_sva,
      or_7_nl);
  assign or_8_nl = or_dcpl_6 | (while_v23_1_0_sva[0]);
  assign mgc_stateful_rf_0_tag_2_2_sva_mx1 = MUX_v_16_2_2(while_v25_sva, mgc_stateful_rf_0_tag_2_2_sva,
      or_8_nl);
  assign or_10_nl = or_dcpl_9 | (~ (while_v23_1_0_sva[0]));
  assign mgc_stateful_rf_0_tag_2_1_sva_mx1 = MUX_v_16_2_2(while_v25_sva, mgc_stateful_rf_0_tag_2_1_sva,
      or_10_nl);
  assign or_11_nl = or_dcpl_9 | (while_v23_1_0_sva[0]);
  assign mgc_stateful_rf_0_tag_2_0_sva_mx1 = MUX_v_16_2_2(while_v25_sva, mgc_stateful_rf_0_tag_2_0_sva,
      or_11_nl);
  assign or_dcpl_6 = ~((while_v23_1_0_sva[1]) & v15_svs);
  assign or_dcpl_9 = (while_v23_1_0_sva[1]) | (~ v15_svs);
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      v17 <= 16'b0000000000000000;
      v18 <= 16'b0000000000000000;
      while_v23_1_0_sva <= 2'b00;
      v15_svs <= 1'b0;
    end
    else begin
      v17 <= MUX_v_16_2_2(16'b0000000000000000, while_v30_mux_nl, (fsm_output[1]));
      v18 <= MUX_v_16_2_2(16'b0000000000000000, operator_33_true_acc_nl, (fsm_output[1]));
      while_v23_1_0_sva <= v13[1:0];
      v15_svs <= v15;
    end
  end
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      done <= 1'b0;
    end
    else if ( ~ (fsm_output[1]) ) begin
      done <= 1'b1;
    end
  end
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      mgc_stateful_rf_0_tag_2_3_sva <= 16'b0000000000000000;
    end
    else if ( v15_svs & (while_v23_1_0_sva==2'b11) & (fsm_output[1]) & (~((v13[1:0]==2'b11)
        & v15)) ) begin
      mgc_stateful_rf_0_tag_2_3_sva <= mgc_stateful_rf_0_tag_2_3_sva_mx1;
    end
  end
  always @(posedge clk) begin
    if ( v15 ) begin
      while_v25_sva <= v14;
    end
  end
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      mgc_stateful_rf_0_tag_2_2_sva <= 16'b0000000000000000;
    end
    else if ( v15_svs & (while_v23_1_0_sva==2'b10) & (fsm_output[1]) & ((v13[1:0]!=2'b10)
        | (~ v15)) ) begin
      mgc_stateful_rf_0_tag_2_2_sva <= mgc_stateful_rf_0_tag_2_2_sva_mx1;
    end
  end
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      mgc_stateful_rf_0_tag_2_1_sva <= 16'b0000000000000000;
    end
    else if ( v15_svs & (while_v23_1_0_sva==2'b01) & (fsm_output[1]) & ((v13[1:0]!=2'b01)
        | (~ v15)) ) begin
      mgc_stateful_rf_0_tag_2_1_sva <= mgc_stateful_rf_0_tag_2_1_sva_mx1;
    end
  end
  always @(posedge clk or negedge rst) begin
    if ( ~ rst ) begin
      mgc_stateful_rf_0_tag_2_0_sva <= 16'b0000000000000000;
    end
    else if ( v15_svs & (while_v23_1_0_sva==2'b00) & (fsm_output[1]) & ((v13[1:0]!=2'b00)
        | (~ v15)) ) begin
      mgc_stateful_rf_0_tag_2_0_sva <= mgc_stateful_rf_0_tag_2_0_sva_mx1;
    end
  end
  assign while_v30_mux_nl = MUX_v_16_4_2(mgc_stateful_rf_0_tag_2_0_sva_mx1, mgc_stateful_rf_0_tag_2_1_sva_mx1,
      mgc_stateful_rf_0_tag_2_2_sva_mx1, mgc_stateful_rf_0_tag_2_3_sva_mx1, v12[1:0]);
  assign nl_operator_33_true_acc_nl = v18 + 16'b0000000000000001;
  assign operator_33_true_acc_nl = nl_operator_33_true_acc_nl[15:0];

  function automatic [15:0] MUX_v_16_2_2;
    input [15:0] input_0;
    input [15:0] input_1;
    input  sel;
    reg [15:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_16_2_2 = result;
  end
  endfunction


  function automatic [15:0] MUX_v_16_4_2;
    input [15:0] input_0;
    input [15:0] input_1;
    input [15:0] input_2;
    input [15:0] input_3;
    input [1:0] sel;
    reg [15:0] result;
  begin
    case (sel)
      2'b00 : begin
        result = input_0;
      end
      2'b01 : begin
        result = input_1;
      end
      2'b10 : begin
        result = input_2;
      end
      default : begin
        result = input_3;
      end
    endcase
    MUX_v_16_4_2 = result;
  end
  endfunction

endmodule
