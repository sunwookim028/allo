// GENERATED — do not edit directly.
// Regenerate with: make -C npu generate-sv-pkgs
// Source: npu/params.py

/* verilator lint_off UNUSEDPARAM */
package npu_config_pkg;

  // Systolic array dimension (N×N PEs)
  localparam N = 8;

  // FP32 word width used throughout the data path
  localparam DATA_WIDTH = 32;

  // L1 BRAM: 2^10 = 1024 words (4 KB @ 32 b/word)
  localparam L1_ADDR_WIDTH = 10;

  // L1 wide bus: 8 × DATA_WIDTH, used for DMA and compute-side 256-bit access
  localparam L1_DATA_WIDTH = 256;

  // L2 SRAM: 2^15 = 32768 words (128 KB @ 32 b/word)
  localparam L2_ADDR_WIDTH = 15;

  // Device memory: 2^16 = 65536 words (256 KB @ 32 b/word)
  localparam DM_ADDR_WIDTH = 16;

  // Device memory wide bus: 8 × DATA_WIDTH, matching L1_DATA_WIDTH
  localparam DM_DATA_WIDTH = 256;

  // BRAM registered-output latency (cycles from address-valid to data-valid)
  localparam MEM_LATENCY = 2;

  // FPU pipeline stages per arithmetic unit (fp32_mul + fp32_add each). Total PE MAC latency = 2 × PE_LATENCY
  localparam PE_LATENCY = 0;

  // Instruction RAM address bits
  localparam IRAM_ADDR_WIDTH = 12;

  // Instruction RAM depth = 2^IRAM_ADDR_WIDTH = 4096
  localparam IRAM_DEPTH = 4096;

  // Instruction word width in bits
  localparam INSTR_WIDTH = 64;

endpackage
/* verilator lint_on UNUSEDPARAM */
