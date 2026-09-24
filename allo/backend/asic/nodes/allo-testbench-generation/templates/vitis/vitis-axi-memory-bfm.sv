`timescale 1ns/1ps

module vitis_axi_memory_bfm #(
  parameter integer ADDR_WIDTH = 64,
  parameter integer DATA_WIDTH = 32,
  parameter integer ELEMENT_WIDTH = DATA_WIDTH,
  parameter integer ID_WIDTH = 1,
  parameter logic [ADDR_WIDTH-1:0] BASE_ADDR = 'h0,
  parameter integer MEMORY_WORDS = 1048576
) (
  input  logic clk,
  input  logic reset_n,
  input  logic awvalid,
  output logic awready,
  input  logic [ADDR_WIDTH-1:0] awaddr,
  input  logic [ID_WIDTH-1:0] awid,
  input  logic [7:0] awlen,
  input  logic [2:0] awsize,
  input  logic [1:0] awburst,
  input  logic [1:0] awlock,
  input  logic [3:0] awcache,
  input  logic [2:0] awprot,
  input  logic [3:0] awqos,
  input  logic [3:0] awregion,
  input  logic awuser,
  input  logic wvalid,
  output logic wready,
  input  logic [DATA_WIDTH-1:0] wdata,
  input  logic [DATA_WIDTH/8-1:0] wstrb,
  input  logic wlast,
  input  logic [ID_WIDTH-1:0] wid,
  input  logic wuser,
  input  logic arvalid,
  output logic arready,
  input  logic [ADDR_WIDTH-1:0] araddr,
  input  logic [ID_WIDTH-1:0] arid,
  input  logic [7:0] arlen,
  input  logic [2:0] arsize,
  input  logic [1:0] arburst,
  input  logic [1:0] arlock,
  input  logic [3:0] arcache,
  input  logic [2:0] arprot,
  input  logic [3:0] arqos,
  input  logic [3:0] arregion,
  input  logic aruser,
  output logic rvalid,
  input  logic rready,
  output logic [DATA_WIDTH-1:0] rdata,
  output logic rlast,
  output logic [ID_WIDTH-1:0] rid,
  output logic ruser,
  output logic [1:0] rresp,
  output logic bvalid,
  input  logic bready,
  output logic [1:0] bresp,
  output logic [ID_WIDTH-1:0] bid,
  output logic buser
);
  localparam integer BYTES_PER_WORD = DATA_WIDTH / 8;
  localparam integer ELEMENTS_PER_WORD = DATA_WIDTH / ELEMENT_WIDTH;
  logic [DATA_WIDTH-1:0] memory [0:MEMORY_WORDS-1];

  logic write_active;
  logic [ADDR_WIDTH-1:0] write_address;
  logic [8:0] write_beats_left;
  logic [ID_WIDTH-1:0] write_id;
  logic read_active;
  logic [ADDR_WIDTH-1:0] read_address;
  logic [8:0] read_beats_left;
  logic [ID_WIDTH-1:0] read_id;
  integer byte_index;
  integer word_index;
  real drive_delay_ns;

  task automatic load_hex(input string file_name, input integer element_count);
    integer file_handle;
    integer scan_status;
    integer element_index;
    integer packed_word_index;
    integer packed_lane_index;
    logic [ELEMENT_WIDTH-1:0] element_value;
    begin
      file_handle = $fopen(file_name, "r");
      if (file_handle == 0)
        $fatal(1, "cannot open memory initialization file %s", file_name);
      for (element_index = 0; element_index < element_count;
           element_index = element_index + 1) begin
        scan_status = $fscanf(file_handle, "%h", element_value);
        if (scan_status != 1)
          $fatal(1, "cannot read element %0d from %s", element_index, file_name);
        packed_word_index = element_index / ELEMENTS_PER_WORD;
        packed_lane_index = element_index % ELEMENTS_PER_WORD;
        if (packed_word_index >= MEMORY_WORDS)
          $fatal(1, "initialization file exceeds BFM memory");
        if (packed_lane_index == 0)
          memory[packed_word_index] = '0;
        memory[packed_word_index][packed_lane_index*ELEMENT_WIDTH +: ELEMENT_WIDTH]
          = element_value;
      end
      $fclose(file_handle);
    end
  endtask

  task automatic check_hex(input string file_name, input integer element_count);
    integer file_handle;
    integer scan_status;
    integer element_index;
    integer packed_word_index;
    integer packed_lane_index;
    logic [ELEMENT_WIDTH-1:0] expected_element;
    begin
      file_handle = $fopen(file_name, "r");
      if (file_handle == 0)
        $fatal(1, "cannot open expected memory file %s", file_name);
      for (element_index = 0; element_index < element_count;
           element_index = element_index + 1) begin
        scan_status = $fscanf(file_handle, "%h", expected_element);
        if (scan_status != 1)
          $fatal(1, "cannot read expected element %0d from %s",
                 element_index, file_name);
        packed_word_index = element_index / ELEMENTS_PER_WORD;
        packed_lane_index = element_index % ELEMENTS_PER_WORD;
        if (memory[packed_word_index][packed_lane_index*ELEMENT_WIDTH +: ELEMENT_WIDTH]
            !== expected_element)
          $fatal(1, "memory mismatch at index %0d: expected 0x%0h, got 0x%0h",
                 element_index, expected_element,
                 memory[packed_word_index]
                       [packed_lane_index*ELEMENT_WIDTH +: ELEMENT_WIDTH]);
      end
      $fclose(file_handle);
    end
  endtask

  initial begin
    if (ELEMENT_WIDTH <= 0 || ELEMENT_WIDTH > DATA_WIDTH ||
        DATA_WIDTH % ELEMENT_WIDTH != 0)
      $fatal(1, "ELEMENT_WIDTH must divide DATA_WIDTH");
    if (!$value$plusargs("ALLO_BAGL_BFM_DRIVE_DELAY_NS=%f", drive_delay_ns))
      drive_delay_ns = 0.0;
    awready = 1'b0;
    wready = 1'b0;
    arready = 1'b0;
    rvalid = 1'b0;
    rdata = '0;
    rlast = 1'b0;
    rid = '0;
    ruser = 1'b0;
    rresp = 2'b00;
    bvalid = 1'b0;
    bresp = 2'b00;
    bid = '0;
    buser = 1'b0;
    write_active = 1'b0;
    read_active = 1'b0;
  end

  always @(posedge clk) begin
    if (!reset_n) begin
      awready <= #(drive_delay_ns) 1'b0;
      wready <= #(drive_delay_ns) 1'b0;
      arready <= #(drive_delay_ns) 1'b0;
      rvalid <= #(drive_delay_ns) 1'b0;
      bvalid <= #(drive_delay_ns) 1'b0;
      write_active <= 1'b0;
      read_active <= 1'b0;
    end else begin
      awready <= #(drive_delay_ns) !write_active && !bvalid;
      wready <= #(drive_delay_ns) write_active;
      arready <= #(drive_delay_ns) !read_active && !rvalid;
      if (awvalid && awready) begin
        if (awburst != 2'b01)
          $fatal(1, "only AXI INCR write bursts are supported");
        if ((1 << awsize) != BYTES_PER_WORD)
          $fatal(1, "AXI write beat size does not match DATA_WIDTH");
        write_active <= 1'b1;
        write_address <= awaddr;
        write_beats_left <= {1'b0, awlen} + 1'b1;
        write_id <= awid;
        awready <= #(drive_delay_ns) 1'b0;
      end

      if (wvalid && wready) begin
        if (write_address < BASE_ADDR)
          $fatal(1, "AXI write address 0x%0h is below configured base 0x%0h",
                 write_address, BASE_ADDR);
        word_index = (write_address - BASE_ADDR) / BYTES_PER_WORD;
        if (word_index >= MEMORY_WORDS)
          $fatal(1, "AXI write exceeds BFM memory");
        for (byte_index = 0; byte_index < BYTES_PER_WORD; byte_index = byte_index + 1)
          if (wstrb[byte_index])
            memory[word_index][byte_index*8 +: 8] <= wdata[byte_index*8 +: 8];
        write_address <= write_address + BYTES_PER_WORD;
        write_beats_left <= write_beats_left - 1'b1;
        if (write_beats_left == 1) begin
          if (!wlast)
            $fatal(1, "missing WLAST on final AXI write beat");
          write_active <= 1'b0;
          wready <= #(drive_delay_ns) 1'b0;
          bvalid <= #(drive_delay_ns) 1'b1;
          bid <= #(drive_delay_ns) write_id;
        end
      end
      if (bvalid && bready)
        bvalid <= #(drive_delay_ns) 1'b0;

      if (arvalid && arready) begin
        if (arburst != 2'b01)
          $fatal(1, "only AXI INCR read bursts are supported");
        if ((1 << arsize) != BYTES_PER_WORD)
          $fatal(1, "AXI read beat size does not match DATA_WIDTH");
        read_active <= 1'b1;
        read_address <= araddr;
        read_beats_left <= {1'b0, arlen} + 1'b1;
        read_id <= arid;
        arready <= #(drive_delay_ns) 1'b0;
      end

      if (read_active && !rvalid) begin
        if (read_address < BASE_ADDR)
          $fatal(1, "AXI read address 0x%0h is below configured base 0x%0h",
                 read_address, BASE_ADDR);
        word_index = (read_address - BASE_ADDR) / BYTES_PER_WORD;
        if (word_index >= MEMORY_WORDS)
          $fatal(1, "AXI read exceeds BFM memory");
        rdata <= #(drive_delay_ns) memory[word_index];
        rid <= #(drive_delay_ns) read_id;
        rlast <= #(drive_delay_ns) (read_beats_left == 1);
        rvalid <= #(drive_delay_ns) 1'b1;
      end
      if (rvalid && rready) begin
        rvalid <= #(drive_delay_ns) 1'b0;
        read_address <= read_address + BYTES_PER_WORD;
        read_beats_left <= read_beats_left - 1'b1;
        if (read_beats_left == 1) begin
          read_active <= 1'b0;
          rlast <= #(drive_delay_ns) 1'b0;
        end
      end
    end
  end
endmodule
