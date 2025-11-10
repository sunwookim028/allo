//=============================================================================
// Example 6: RAM Module Implementation
//=============================================================================
// Demonstrates: Static arrays as RAM with read/write interface
// Hardware Impact: Static array becomes BRAM with address decoding
//
// Interface: ram(write_addr, read_addr, in, write_en, out)
//   - write_addr: address to write to
//   - read_addr: address to read from
//   - in: input data to write
//   - write_en: write enable signal
//   - out: output data (always reads from read_addr)
//
// Note: The 'mem' argument was removed as it was unused. The static array
// 'memory' is internal to each kernel and not shared. For shared memory
// across kernels, see ex08_shared_memory.

// Version A: Basic RAM with separate read/write addresses
void ram_a(int write_addr, int read_addr, int in, int write_en, int* out) {
    // Static array: becomes BRAM in hardware
    // Size: 1024 words (adjustable)
    static int memory[1024];
    
    // INTERFACE pragma: Specifies how function arguments connect to hardware
    // - m_axi: Memory-mapped AXI4 interface (for external memory access)
    // - s_axilite: AXI4-Lite interface (for control/status registers)
    // - ap_none: No interface (direct connection)
    // Here: out uses ap_none (direct output), others use ap_none (scalar inputs)
    #pragma HLS INTERFACE ap_none port=out
    #pragma HLS INTERFACE ap_none port=write_addr
    #pragma HLS INTERFACE ap_none port=read_addr
    #pragma HLS INTERFACE ap_none port=in
    #pragma HLS INTERFACE ap_none port=write_en
    
    // bind_storage pragma: Specifies physical storage resource for variables
    // - ram_1p (impl=bram): Single-port Block RAM (one read OR write per cycle)
    // - ram_2p (impl=bram): Dual-port Block RAM (simultaneous read AND write)
    // - RAM_1P_URAM: Single-port Ultra RAM (for very large arrays)
    // - RAM_2P_URAM: Dual-port Ultra RAM
    // This maps the static array to a specific BRAM resource
    #pragma HLS bind_storage variable=memory type=ram_1p impl=bram
    
    // Write operation: only if write_en is true
    if (write_en && write_addr >= 0 && write_addr < 1024) {
        memory[write_addr] = in;
    }
    
    // Read operation: always outputs value at read_addr
    // If never written, outputs undefined value (initialized to 0)
    if (read_addr >= 0 && read_addr < 1024) {
        *out = memory[read_addr];
    } else {
        *out = 0;  // Out of bounds
    }
}

// Version B: Dual-port RAM (read and write can happen simultaneously)
void ram_b(int write_addr, int read_addr, int in, int write_en, int* out) {
    static int memory[1024];
    
    #pragma HLS INTERFACE ap_none port=out
    #pragma HLS INTERFACE ap_none port=write_addr
    #pragma HLS INTERFACE ap_none port=read_addr
    #pragma HLS INTERFACE ap_none port=in
    #pragma HLS INTERFACE ap_none port=write_en
    
    // bind_storage: Use dual-port BRAM for simultaneous read/write
    // ram_2p (impl=bram) allows independent read and write ports
    #pragma HLS bind_storage variable=memory type=ram_2p impl=bram
    
    // Write port
    if (write_en && write_addr >= 0 && write_addr < 1024) {
        memory[write_addr] = in;
    }
    
    // Read port (independent, can happen simultaneously)
    if (read_addr >= 0 && read_addr < 1024) {
        *out = memory[read_addr];
    } else {
        *out = 0;
    }
}

// Version C: RAM with initialization (undefined values = 0)
void ram_c(int write_addr, int read_addr, int in, int write_en, int* out) {
    // Initialize to 0 (undefined values are 0)
    static int memory[1024] = {0};
    
    #pragma HLS INTERFACE ap_none port=out
    #pragma HLS INTERFACE ap_none port=write_addr
    #pragma HLS INTERFACE ap_none port=read_addr
    #pragma HLS INTERFACE ap_none port=in
    #pragma HLS INTERFACE ap_none port=write_en
    
    // bind_storage: Single-port BRAM with initialization
    #pragma HLS bind_storage variable=memory type=ram_1p impl=bram
    
    // Write operation
    if (write_en && write_addr >= 0 && write_addr < 1024) {
        memory[write_addr] = in;
    }
    
    // Read operation
    if (read_addr >= 0 && read_addr < 1024) {
        *out = memory[read_addr];
    } else {
        *out = 0;
    }
}

// Version D: RAM with explicit undefined value handling
void ram_d(int write_addr, int read_addr, int in, int write_en, int* out) {
    // Track which addresses have been written
    static int memory[1024];
    static bool written[1024] = {false};  // Track initialization
    
    #pragma HLS INTERFACE ap_none port=out
    #pragma HLS INTERFACE ap_none port=write_addr
    #pragma HLS INTERFACE ap_none port=read_addr
    #pragma HLS INTERFACE ap_none port=in
    #pragma HLS INTERFACE ap_none port=write_en
    
    // bind_storage: Single-port BRAM for data
    #pragma HLS bind_storage variable=memory type=ram_1p impl=bram
    // Partition written array to improve access (small array, uses registers)
    #pragma HLS ARRAY_PARTITION variable=written cyclic factor=32
    
    // Write operation
    if (write_en && write_addr >= 0 && write_addr < 1024) {
        memory[write_addr] = in;
        written[write_addr] = true;
    }
    
    // Read operation
    if (read_addr >= 0 && read_addr < 1024) {
        if (written[read_addr]) {
            *out = memory[read_addr];
        } else {
            // Undefined value: return 0 or could use a sentinel value
            *out = 0;  // Or could return a special "undefined" marker
        }
    } else {
        *out = 0;
    }
}

// Version E: Simple single-port RAM (most common pattern)
void ram_e(int addr, int in, int write_en, int* out) {
    // Single-port: read and write share same address
    static int memory[1024] = {0};
    
    #pragma HLS INTERFACE ap_none port=out
    #pragma HLS INTERFACE ap_none port=addr
    #pragma HLS INTERFACE ap_none port=in
    #pragma HLS INTERFACE ap_none port=write_en
    
    // bind_storage: Single-port BRAM
    #pragma HLS bind_storage variable=memory type=ram_1p impl=bram
    
    if (write_en && addr >= 0 && addr < 1024) {
        // Write mode
        memory[addr] = in;
        *out = in;  // Write-through (optional)
    } else if (addr >= 0 && addr < 1024) {
        // Read mode
        *out = memory[addr];
    } else {
        *out = 0;
    }
}
