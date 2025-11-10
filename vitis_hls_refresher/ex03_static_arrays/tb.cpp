//=============================================================================
// Testbench for Example 3: Static Arrays
//=============================================================================

#include <iostream>

extern void top_a(int data[10], int* result);
extern void top_b(int data[100], int* result);
extern void top_c(int data[10], int* result);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 3: Static Arrays\n";
    std::cout << "========================================\n\n";
    
    // Test Version A: Non-static array
    std::cout << "--- Version A: Non-static array ---\n";
    int data_a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int result_a;
    top_a(data_a, &result_a);
    std::cout << "  Result: " << result_a << " (expected: 110)\n";
    std::cout << "  Hardware: Array mapped to registers/wires\n\n";
    
    // Test Version B: Static array (large)
    std::cout << "--- Version B: Static array (large) ---\n";
    int data_b[100];
    for (int i = 0; i < 100; i++) {
        data_b[i] = i + 1;
    }
    int result_b;
    top_b(data_b, &result_b);
    std::cout << "  Result: " << result_b << " (expected: 10100)\n";
    std::cout << "  Hardware: Array mapped to BRAM\n\n";
    
    // Test Version C: Static array with initialization
    std::cout << "--- Version C: Static array with init ---\n";
    int data_c[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int result_c;
    
    std::cout << "Call 1:\n";
    top_c(data_c, &result_c);
    std::cout << "  Result: " << result_c << " (expected: 550)\n";
    
    std::cout << "Call 2 (static array retains values):\n";
    top_c(data_c, &result_c);
    std::cout << "  Result: " << result_c << " (expected: 1100, accumulates)\n";
    std::cout << "  Hardware: Array persists across calls\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Memory Resource Mapping:\n";
    std::cout << "  Non-static small arrays: Registers/Flip-flops\n";
    std::cout << "  Static large arrays: BRAM/URAM\n";
    std::cout << "  Static arrays persist across function calls\n";
    std::cout << "========================================\n";
    
    return 0;
}

