//=============================================================================
// Testbench for Example 6: RAM Module
//=============================================================================

#include <iostream>
#include <cassert>

extern void ram_a(int write_addr, int read_addr, int in, int write_en, int* out);
extern void ram_b(int write_addr, int read_addr, int in, int write_en, int* out);
extern void ram_c(int write_addr, int read_addr, int in, int write_en, int* out);
extern void ram_d(int write_addr, int read_addr, int in, int write_en, int* out);
extern void ram_e(int addr, int in, int write_en, int* out);

int main() {
    std::cout << "========================================\n";
    std::cout << "Example 6: RAM Module Implementation\n";
    std::cout << "========================================\n\n";
    
    int result;
    
    // Test Version A: Basic RAM
    std::cout << "--- Version A: Basic RAM (separate read/write) ---\n";
    
    std::cout << "\n1. Write value 100 to address 5\n";
    ram_a(5, 0, 100, 1, &result);
    std::cout << "   Write: addr=5, data=100, write_en=1\n";
    
    std::cout << "\n2. Read from address 5\n";
    ram_a(0, 5, 0, 0, &result);
    std::cout << "   Read: addr=5, out=" << result << " (expected: 100)\n";
    
    std::cout << "\n3. Read from unwritten address 10\n";
    ram_a(0, 10, 0, 0, &result);
    std::cout << "   Read: addr=10, out=" << result << " (undefined, likely 0)\n";
    
    std::cout << "\n4. Write value 200 to address 10\n";
    ram_a(10, 0, 200, 1, &result);
    std::cout << "   Write: addr=10, data=200\n";
    
    std::cout << "\n5. Read from address 10\n";
    ram_a(0, 10, 0, 0, &result);
    std::cout << "   Read: addr=10, out=" << result << " (expected: 200)\n";
    
    std::cout << "\n6. Try to write with write_en=0 (should not write)\n";
    ram_a(5, 0, 999, 0, &result);
    std::cout << "   Write attempt: addr=5, data=999, write_en=0\n";
    
    std::cout << "\n7. Read from address 5 (should still be 100)\n";
    ram_a(0, 5, 0, 0, &result);
    std::cout << "   Read: addr=5, out=" << result << " (expected: 100, unchanged)\n\n";
    
    // Test Version B: Dual-port RAM
    std::cout << "--- Version B: Dual-port RAM ---\n";
    std::cout << "Can read and write simultaneously\n\n";
    
    std::cout << "Simultaneous operation:\n";
    std::cout << "  Writing 300 to address 20\n";
    std::cout << "  Reading from address 5\n";
    ram_b(20, 5, 300, 1, &result);
    std::cout << "  Read result: " << result << " (expected: 100, from previous test)\n";
    
    ram_b(0, 20, 0, 0, &result);
    std::cout << "  Verify write: addr=20, out=" << result << " (expected: 300)\n\n";
    
    // Test Version C: RAM with initialization
    std::cout << "--- Version C: RAM with Initialization ---\n";
    std::cout << "All locations initialized to 0\n\n";
    
    std::cout << "Read from unwritten address 50:\n";
    ram_c(0, 50, 0, 0, &result);
    std::cout << "  out=" << result << " (expected: 0, initialized)\n\n";
    
    // Test Version D: RAM with undefined tracking
    std::cout << "--- Version D: RAM with Undefined Tracking ---\n";
    std::cout << "Tracks which addresses have been written\n\n";
    
    std::cout << "Read from unwritten address 100:\n";
    ram_d(0, 100, 0, 0, &result);
    std::cout << "  out=" << result << " (undefined, returns 0)\n";
    
    std::cout << "\nWrite 500 to address 100:\n";
    ram_d(100, 0, 500, 1, &result);
    
    std::cout << "\nRead from address 100:\n";
    ram_d(0, 100, 0, 0, &result);
    std::cout << "  out=" << result << " (expected: 500)\n\n";
    
    // Test Version E: Single-port RAM
    std::cout << "--- Version E: Single-port RAM ---\n";
    std::cout << "Read and write share same address\n\n";
    
    std::cout << "Write 777 to address 7:\n";
    ram_e(7, 777, 1, &result);
    std::cout << "  Write: addr=7, data=777, out=" << result << "\n";
    
    std::cout << "\nRead from address 7:\n";
    ram_e(7, 0, 0, &result);
    std::cout << "  Read: addr=7, out=" << result << " (expected: 777)\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "  - Static array becomes BRAM in hardware\n";
    std::cout << "  - Write only occurs when write_en=1\n";
    std::cout << "  - Read always outputs value (undefined if never written)\n";
    std::cout << "  - Dual-port allows simultaneous read/write\n";
    std::cout << "  - Single-port shares address for read/write\n";
    std::cout << "  - Note: Each kernel has its own static memory (not shared)\n";
    std::cout << "  - For shared memory across kernels, see ex08_shared_memory\n";
    std::cout << "========================================\n";
    
    return 0;
}
