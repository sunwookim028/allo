//=============================================================================
// Testbench for Example 2: Static Variables in Loops
//=============================================================================

#include <iostream>
#include <cassert>

extern void top_a(int data[10], int* result);
extern void top_b(int data[10], int* result);
extern void top_c(int data[10], int reset, int* result);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 2: Static Variables in Loops\n";
    std::cout << "========================================\n\n";
    
    int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int result;
    
    // Test Version A: Non-static accumulator
    std::cout << "--- Version A: Non-static accumulator ---\n";
    std::cout << "Call 1:\n";
    top_a(data, &result);
    std::cout << "  Result: " << result << " (expected: 55)\n";
    
    std::cout << "Call 2:\n";
    top_a(data, &result);
    std::cout << "  Result: " << result << " (expected: 55, same as Call 1)\n";
    std::cout << "  Note: acc resets to 0 each call\n\n";
    
    // Test Version B: Static accumulator
    std::cout << "--- Version B: Static accumulator ---\n";
    std::cout << "Call 1:\n";
    top_b(data, &result);
    std::cout << "  Result: " << result << " (expected: 55)\n";
    
    std::cout << "Call 2:\n";
    top_b(data, &result);
    std::cout << "  Result: " << result << " (expected: 110, accumulates!)\n";
    std::cout << "  Note: acc retains value between calls\n\n";
    
    // Test Version C: Static with reset
    std::cout << "--- Version C: Static with reset ---\n";
    std::cout << "Call 1 (reset=1):\n";
    top_c(data, 1, &result);
    std::cout << "  Result: " << result << " (expected: 55)\n";
    
    std::cout << "Call 2 (reset=0):\n";
    top_c(data, 0, &result);
    std::cout << "  Result: " << result << " (expected: 110)\n";
    
    std::cout << "Call 3 (reset=1):\n";
    top_c(data, 1, &result);
    std::cout << "  Result: " << result << " (expected: 55, reset works)\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Hardware Implications:\n";
    std::cout << "  Version A: acc is wire/logic (reset each call)\n";
    std::cout << "  Version B: acc is register (persists, may need reset logic)\n";
    std::cout << "  Version C: acc is register with reset control\n";
    std::cout << "========================================\n";
    
    return 0;
}

