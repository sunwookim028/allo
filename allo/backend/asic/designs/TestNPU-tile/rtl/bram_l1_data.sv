`timescale 1ns / 1ps

// Behavioral model for blk_mem_gen_0 (L1 Data BRAM)
// 8192×32-bit dual-port SRAM, 13-bit address, 1-cycle registered output latency.
// Matches Xilinx Block RAM IP: 1-cycle registered output latency.
module blk_mem_gen_0 (
    input clka,
    input ena,
    input [0:0] wea,
    input [12:0] addra,
    input [31:0] dina,
    output reg [31:0] douta,
    input clkb,
    input enb,
    input [0:0] web,
    input [12:0] addrb,
    input [31:0] dinb,
    output reg [31:0] doutb
);
    /* verilator lint_off MULTIDRIVEN */
    reg [31:0] mem [0:8191];
    /* verilator lint_on MULTIDRIVEN */
    integer i;
    initial begin
        for (i = 0; i < 8192; i = i + 1) mem[i] = 0;
    end

    // 1-cycle read latency (matches Xilinx BRAM IP)
    /* verilator lint_off MULTIDRIVEN */
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
    /* verilator lint_on MULTIDRIVEN */
endmodule

// VCD waveform dump — enabled by compiling with -DVCD_DUMP
// Usage: make test_data_integrity_rtl VCD=1
`ifdef VCD_DUMP
module vcd_dump;
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tpu);
    end
endmodule
`endif
