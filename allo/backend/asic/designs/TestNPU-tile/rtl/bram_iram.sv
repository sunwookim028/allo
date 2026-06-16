`timescale 1ns / 1ps

// Behavioral model for blk_mem_gen_1 (Instruction RAM)
// 4096×64-bit dual-port SRAM, 12-bit address, 1-cycle registered output latency.
// Matches Xilinx Block RAM IP: 1-cycle registered output latency.
module blk_mem_gen_1 (
    input clka,
    input ena,
    input [0:0] wea,
    input [11:0] addra,
    input [63:0] dina,
    output reg [63:0] douta,
    input clkb,
    input enb,
    input [0:0] web,
    input [11:0] addrb,
    input [63:0] dinb,
    output reg [63:0] doutb
);
    // Dual-port BRAM: two separate clock domains write to mem[].
    // MULTIDRIVEN is expected for a true dual-port RAM; suppress here.
    /* verilator lint_off MULTIDRIVEN */
    reg [63:0] mem [0:4095];
    /* verilator lint_on MULTIDRIVEN */
    integer i;
    initial begin
        for (i = 0; i < 4096; i = i + 1) mem[i] = 0;
    end

    // 1-cycle read latency (matches Xilinx BRAM IP)
    always @(posedge clka) begin
        if (ena) begin
            if (wea) mem[addra] <= dina;
            douta <= mem[addra];
        end
    end

    always @(posedge clkb) begin
        if (enb) begin
            if (web) mem[addrb] <= dinb;
            doutb <= mem[addrb];
        end
    end
endmodule
