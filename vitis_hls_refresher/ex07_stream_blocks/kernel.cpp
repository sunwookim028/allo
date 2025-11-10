//=============================================================================
// Example 7: HLS Stream of Blocks Library
//=============================================================================
// Demonstrates: Extending static notion to arbitrary objects/structures
// Hardware Impact: Static objects become registers/BRAM based on size
//
// This example shows how static works with:
//   - Structures/classes
//   - Arrays of structures
//   - Complex data types
//   - Stream interfaces

#include "hls_stream.h"
#include <ap_int.h>

// Define a custom structure (block)
struct DataBlock {
    int value;
    int timestamp;
    bool valid;
    
    // Default constructor (important for static initialization)
    DataBlock() : value(0), timestamp(0), valid(false) {}
    
    // Constructor with parameters
    DataBlock(int v, int t, bool vld) : value(v), timestamp(t), valid(vld) {}
};

// Version A: Static structure (single object)
void stream_blocks_a(DataBlock* in, DataBlock* out) {
    // Static structure: becomes registers
    static DataBlock buffer;
    
    // Store input
    buffer = *in;
    
    // Output with delay (simple buffering)
    *out = buffer;
}

// Version B: Static array of structures (history buffer)
void stream_blocks_b(DataBlock* in, DataBlock* out, int delay) {
    // Static array of structures: becomes BRAM or registers
    static DataBlock history[16];
    
    #pragma HLS ARRAY_PARTITION variable=history cyclic factor=4
    
    // Shift register pattern
    for (int i = 15; i > 0; i--) {
        history[i] = history[i-1];
    }
    history[0] = *in;
    
    // Output delayed version
    if (delay >= 0 && delay < 16) {
        *out = history[delay];
    } else {
        *out = history[0];
    }
}

// Version C: Static structure with initialization
void stream_blocks_c(DataBlock* in, DataBlock* out) {
    // Static structure with explicit initialization
    static DataBlock accumulator = DataBlock(0, 0, false);
    
    // Accumulate values
    if (in->valid) {
        accumulator.value += in->value;
        accumulator.timestamp = in->timestamp;
        accumulator.valid = true;
    }
    
    *out = accumulator;
}

// Version D: Using HLS Stream interface (recommended for streaming)
void stream_blocks_d(hls::stream<DataBlock>& in_stream, 
                     hls::stream<DataBlock>& out_stream) {
    // Static buffer for stream processing
    static DataBlock buffer[4];
    
    #pragma HLS ARRAY_PARTITION variable=buffer complete
    #pragma HLS PIPELINE II=1
    
    // Read from stream
    DataBlock in_data = in_stream.read();
    
    // Shift buffer
    for (int i = 3; i > 0; i--) {
        buffer[i] = buffer[i-1];
    }
    buffer[0] = in_data;
    
    // Write delayed output to stream
    out_stream.write(buffer[3]);  // 3-cycle delay
}

// Version E: Complex structure with nested arrays
struct ComplexBlock {
    int header[4];
    int data[8];
    int checksum;
    
    ComplexBlock() {
        for (int i = 0; i < 4; i++) header[i] = 0;
        for (int i = 0; i < 8; i++) data[i] = 0;
        checksum = 0;
    }
};

void stream_blocks_e(ComplexBlock* in, ComplexBlock* out) {
    // Static complex structure: large, likely maps to BRAM
    static ComplexBlock buffer[4];
    
    #pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=2
    
    // Shift operation
    for (int i = 3; i > 0; i--) {
        buffer[i] = buffer[i-1];
    }
    buffer[0] = *in;
    
    *out = buffer[2];  // 2-cycle delay
}

// Version F: Static structure with reset capability
void stream_blocks_f(DataBlock* in, DataBlock* out, bool reset) {
    static DataBlock state = DataBlock(0, 0, false);
    
    if (reset) {
        state = DataBlock(0, 0, false);
        *out = state;
        return;
    }
    
    // State machine logic
    if (in->valid) {
        state.value = in->value;
        state.timestamp = in->timestamp;
        state.valid = true;
    }
    
    *out = state;
}

