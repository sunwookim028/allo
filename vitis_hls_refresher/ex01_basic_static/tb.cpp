//=============================================================================
// Testbench for Example 1: Basic Static Variable
//=============================================================================

#include <iostream>
#include <cassert>

extern void top(int in, int* out);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 1: Basic Static Variable\n";
    std::cout << "========================================\n\n";
    
    int result;
    
    // Call 1: Both variables start fresh (non_static) or retain state (static)
    std::cout << "Call 1: in = 5\n";
    top(5, &result);
    std::cout << "  Result: " << result << "\n";
    std::cout << "  Expected: (5+1) + (0+5) = 11\n\n";
    
    // Call 2: static_var retains its value from Call 1
    std::cout << "Call 2: in = 10\n";
    top(10, &result);
    std::cout << "  Result: " << result << "\n";
    std::cout << "  Expected: (10+1) + (5+10) = 26\n";
    std::cout << "  Note: static_var persists, non_static resets\n\n";
    
    // Call 3: Demonstrate accumulation
    std::cout << "Call 3: in = 3\n";
    top(3, &result);
    std::cout << "  Result: " << result << "\n";
    std::cout << "  Expected: (3+1) + (15+3) = 22\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Key Insight:\n";
    std::cout << "  - non_static: Reset every call (wire/logic)\n";
    std::cout << "  - static_var: Retains value (register)\n";
    std::cout << "========================================\n";
    
    return 0;
}

