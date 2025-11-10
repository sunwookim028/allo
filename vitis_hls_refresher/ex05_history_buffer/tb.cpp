//=============================================================================
// Testbench for Example 5: History Buffer
//=============================================================================

#include <iostream>
#include <cassert>

extern void history_buffer_a(int* buf, int reset, int in, int* out, int time);
extern void history_buffer_b(int* buf, int reset, int in, int* out, int time);
extern void history_buffer_c(int* buf, int reset, int in, int* out);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 5: History Buffer with Shift Registers\n";
    std::cout << "========================================\n\n";
    
    int buf[16] = {0};
    int result;
    
    // Test Version A: Configurable delay
    std::cout << "--- Version A: Configurable Delay (time=3) ---\n";
    std::cout << "Testing delay of 3 cycles...\n\n";
    
    // Reset first
    history_buffer_a(buf, 1, 0, &result, 3);
    std::cout << "Reset: out = " << result << " (expected: 0)\n";
    
    // Feed sequence: 10, 20, 30, 40, 50
    std::cout << "\nInput sequence: 10, 20, 30, 40, 50\n";
    std::cout << "Expected output (time=3): 0, 0, 0, 10, 20\n\n";
    
    std::cout << "Call 1: in=10\n";
    history_buffer_a(buf, 0, 10, &result, 3);
    std::cout << "  out = " << result << " (expected: 0, buffer not filled yet)\n";
    
    std::cout << "Call 2: in=20\n";
    history_buffer_a(buf, 0, 20, &result, 3);
    std::cout << "  out = " << result << " (expected: 0)\n";
    
    std::cout << "Call 3: in=30\n";
    history_buffer_a(buf, 0, 30, &result, 3);
    std::cout << "  out = " << result << " (expected: 0)\n";
    
    std::cout << "Call 4: in=40\n";
    history_buffer_a(buf, 0, 40, &result, 3);
    std::cout << "  out = " << result << " (expected: 10, from 3 calls ago)\n";
    
    std::cout << "Call 5: in=50\n";
    history_buffer_a(buf, 0, 50, &result, 3);
    std::cout << "  out = " << result << " (expected: 20, from 3 calls ago)\n\n";
    
    // Test different delay values
    std::cout << "--- Testing Different Delay Values ---\n";
    history_buffer_a(buf, 1, 0, &result, 1);  // Reset
    
    std::cout << "Feeding: 100, 200, 300\n";
    history_buffer_a(buf, 0, 100, &result, 1);
    std::cout << "  time=1: out = " << result << " (expected: 0)\n";
    
    history_buffer_a(buf, 0, 200, &result, 1);
    std::cout << "  time=1: out = " << result << " (expected: 100)\n";
    
    history_buffer_a(buf, 0, 300, &result, 2);
    std::cout << "  time=2: out = " << result << " (expected: 100, from 2 calls ago)\n";
    
    history_buffer_a(buf, 0, 400, &result, 3);
    std::cout << "  time=3: out = " << result << " (expected: 100, from 3 calls ago)\n\n";
    
    // Test Version C: Fixed delay (time=3)
    std::cout << "--- Version C: Fixed Delay (time=3) ---\n";
    std::cout << "Most efficient implementation for fixed delay\n\n";
    
    history_buffer_c(buf, 1, 0, &result);  // Reset
    
    std::cout << "Input sequence: 1, 2, 3, 4, 5, 6\n";
    std::cout << "Expected output: 0, 0, 0, 1, 2, 3\n\n";
    
    for (int i = 1; i <= 6; i++) {
        history_buffer_c(buf, 0, i, &result);
        std::cout << "Call " << i << ": in=" << i << ", out=" << result;
        if (i <= 3) {
            std::cout << " (expected: 0, buffer filling)";
        } else {
            std::cout << " (expected: " << (i-3) << ", from 3 calls ago)";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "  - Static array creates shift register\n";
    std::cout << "  - Each element stores value from N cycles ago\n";
    std::cout << "  - Hardware: Register chain (tapped delay line)\n";
    std::cout << "  - Fixed delay is more efficient than configurable\n";
    std::cout << "========================================\n";
    
    return 0;
}

