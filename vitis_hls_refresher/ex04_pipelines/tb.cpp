//=============================================================================
// Testbench for Example 4: Static Variables in Pipelines
//=============================================================================

#include <iostream>

extern void top_a(int data[10], int* result);
extern void top_b(int data[10], int* result);
extern void top_c(int data[10], int* result);
extern void top_d(int data[10], int* result);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 4: Static Variables in Pipelines\n";
    std::cout << "========================================\n\n";
    
    int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int result;
    
    // Test Version A: Non-static in pipeline
    std::cout << "--- Version A: Non-static accumulator ---\n";
    top_a(data, &result);
    std::cout << "  Result: " << result << " (expected: 55)\n";
    std::cout << "  II=1 achievable: Yes (dependency within loop only)\n\n";
    
    // Test Version B: Static accumulator (problematic)
    std::cout << "--- Version B: Static accumulator ---\n";
    std::cout << "Call 1:\n";
    top_b(data, &result);
    std::cout << "  Result: " << result << " (expected: 55)\n";
    
    std::cout << "Call 2:\n";
    top_b(data, &result);
    std::cout << "  Result: " << result << " (expected: 110, accumulates!)\n";
    std::cout << "  II=1 achievable: Problematic (static persists across calls)\n\n";
    
    // Test Version C: Static shift register
    std::cout << "--- Version C: Static shift register ---\n";
    top_c(data, &result);
    std::cout << "  Result: " << result << "\n";
    std::cout << "  II=1 achievable: Yes (shift register pattern)\n\n";
    
    // Test Version D: Static state machine
    std::cout << "--- Version D: Static state machine ---\n";
    std::cout << "Call 1:\n";
    top_d(data, &result);
    std::cout << "  Result: " << result << "\n";
    
    std::cout << "Call 2:\n";
    top_d(data, &result);
    std::cout << "  Result: " << result << " (state persists)\n";
    std::cout << "  II=1 achievable: Yes (but state transitions may cause stalls)\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Pipeline Insights:\n";
    std::cout << "  - Static variables can break II=1 pipelines\n";
    std::cout << "  - Use static for shift registers and state machines\n";
    std::cout << "  - Avoid static accumulators in pipelined loops\n";
    std::cout << "========================================\n";
    
    return 0;
}

