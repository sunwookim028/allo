//=============================================================================
// Testbench for Example 7: HLS Stream of Blocks
//=============================================================================

#include <iostream>
#include "hls_stream.h"

// Forward declarations
struct DataBlock {
    int value;
    int timestamp;
    bool valid;
    
    DataBlock() : value(0), timestamp(0), valid(false) {}
    DataBlock(int v, int t, bool vld) : value(v), timestamp(t), valid(vld) {}
};

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

extern void stream_blocks_a(DataBlock* in, DataBlock* out);
extern void stream_blocks_b(DataBlock* in, DataBlock* out, int delay);
extern void stream_blocks_c(DataBlock* in, DataBlock* out);
extern void stream_blocks_d(hls::stream<DataBlock>& in_stream, 
                     hls::stream<DataBlock>& out_stream);
extern void stream_blocks_e(ComplexBlock* in, ComplexBlock* out);
extern void stream_blocks_f(DataBlock* in, DataBlock* out, bool reset);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 7: HLS Stream of Blocks\n";
    std::cout << "========================================\n\n";
    
    // Test Version A: Static structure
    std::cout << "--- Version A: Static Structure ---\n";
    DataBlock in_a(10, 100, true);
    DataBlock out_a;
    
    stream_blocks_a(&in_a, &out_a);
    std::cout << "Input: value=" << in_a.value << ", timestamp=" << in_a.timestamp << "\n";
    std::cout << "Output: value=" << out_a.value << ", timestamp=" << out_a.timestamp << "\n";
    std::cout << "  Static structure stores the block\n\n";
    
    // Test Version B: Static array of structures
    std::cout << "--- Version B: Static Array of Structures ---\n";
    std::cout << "Testing delay=3 (3-cycle delay)\n\n";
    
    for (int i = 1; i <= 5; i++) {
        DataBlock in_b(i * 10, i * 100, true);
        DataBlock out_b;
        
        stream_blocks_b(&in_b, &out_b, 3);
        std::cout << "Call " << i << ": in=" << in_b.value;
        if (i <= 3) {
            std::cout << ", out=" << out_b.value << " (buffer filling)\n";
        } else {
            std::cout << ", out=" << out_b.value << " (expected: " << ((i-3)*10) << ")\n";
        }
    }
    std::cout << "\n";
    
    // Test Version C: Static accumulator
    std::cout << "--- Version C: Static Accumulator ---\n";
    DataBlock in_c(5, 1, true);
    DataBlock out_c;
    
    std::cout << "Call 1: value=5\n";
    stream_blocks_c(&in_c, &out_c);
    std::cout << "  Accumulator: value=" << out_c.value << " (expected: 5)\n";
    
    in_c.value = 10;
    in_c.timestamp = 2;
    std::cout << "Call 2: value=10\n";
    stream_blocks_c(&in_c, &out_c);
    std::cout << "  Accumulator: value=" << out_c.value << " (expected: 15)\n";
    
    in_c.value = 20;
    in_c.timestamp = 3;
    std::cout << "Call 3: value=20\n";
    stream_blocks_c(&in_c, &out_c);
    std::cout << "  Accumulator: value=" << out_c.value << " (expected: 35)\n\n";
    // Test Version D: HLS Stream interface (demonstrates efficiency)
    std::cout << "--- Version D: HLS Stream Interface ---\n";
    std::cout << "Demonstrates efficiency of streaming vs pointer-based\n\n";
    
    // Create streams
    hls::stream<DataBlock> in_stream;
    hls::stream<DataBlock> out_stream;
    
    // Baseline: Pointer-based version B (similar functionality)
    std::cout << "Baseline (Version B - pointer-based):\n";
    DataBlock in_baseline, out_baseline;
    for (int i = 1; i <= 6; i++) {
        in_baseline = DataBlock(i * 10, i * 100, true);
        stream_blocks_b(&in_baseline, &out_baseline, 3);
        if (i > 3) {
            std::cout << "  Call " << i << ": in=" << in_baseline.value 
                      << ", out=" << out_baseline.value 
                      << " (expected: " << ((i-3)*10) << ")\n";
        }
    }
    
    std::cout << "\nStreaming (Version D - hls::stream):\n";
    // Fill stream with data
    for (int i = 1; i <= 6; i++) {
        DataBlock in_data(i * 10, i * 100, true);
        in_stream.write(in_data);
    }
    
    // Process stream (simulating pipeline)
    std::cout << "  Processing stream with II=1 pipeline...\n";
    for (int i = 1; i <= 6; i++) {
        stream_blocks_d(in_stream, out_stream);
        DataBlock result = out_stream.read();
        if (i > 3) {
            std::cout << "  Cycle " << i << ": in=" << (i * 10)
                      << ", out=" << result.value 
                      << " (expected: " << ((i-3)*10) << ")\n";
        }
    }
    
    std::cout << "\nKey Advantages of Stream Interface:\n";
    std::cout << "  - FIFO semantics: Natural for streaming data\n";
    std::cout << "  - Pipeline-friendly: Achieves II=1 easily\n";
    std::cout << "  - Hardware-efficient: Maps to FIFOs in hardware\n";
    std::cout << "  - Backpressure handling: Automatic flow control\n";
    std::cout << "  - No manual pointer management: Safer and cleaner\n\n";
    
    // Test Version E: Complex structure
    std::cout << "--- Version E: Complex Structure ---\n";
    ComplexBlock in_e;
    ComplexBlock out_e;
    
    // Initialize input
    for (int i = 0; i < 4; i++) {
        in_e.header[i] = i + 1;
    }
    for (int i = 0; i < 8; i++) {
        in_e.data[i] = i * 10;
    }
    in_e.checksum = 100;
    
    std::cout << "Input: header[0]=" << in_e.header[0] << ", data[0]=" << in_e.data[0] << "\n";
    
    // Feed multiple times to fill buffer
    for (int i = 0; i < 3; i++) {
        stream_blocks_e(&in_e, &out_e);
        std::cout << "Call " << (i+1) << ": output header[0]=" << out_e.header[0] << "\n";
    }
    std::cout << "\n";
    
    // Test Version F: Static with reset
    std::cout << "--- Version F: Static Structure with Reset ---\n";
    DataBlock in_f(50, 500, true);
    DataBlock out_f;
    
    std::cout << "Call 1: value=50\n";
    stream_blocks_f(&in_f, &out_f, false);
    std::cout << "  State: value=" << out_f.value << " (expected: 50)\n";
    
    in_f.value = 75;
    std::cout << "Call 2: value=75\n";
    stream_blocks_f(&in_f, &out_f, false);
    std::cout << "  State: value=" << out_f.value << " (expected: 75)\n";
    
    std::cout << "Call 3: reset=true\n";
    stream_blocks_f(&in_f, &out_f, true);
    std::cout << "  State: value=" << out_f.value << " (expected: 0, reset)\n";
    
    std::cout << "Call 4: value=100\n";
    in_f.value = 100;
    stream_blocks_f(&in_f, &out_f, false);
    std::cout << "  State: value=" << out_f.value << " (expected: 100)\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "  - Static works with any C++ type (structs, classes)\n";
    std::cout << "  - Small structures → registers\n";
    std::cout << "  - Large structures → BRAM\n";
    std::cout << "  - Arrays of structures → BRAM\n";
    std::cout << "  - HLS Stream interface for efficient streaming\n";
    std::cout << "  - Initialization important for static objects\n";
    std::cout << "========================================\n";
    
    return 0;
}

