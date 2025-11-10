//=============================================================================
// Example 3: Static Arrays and Memory Resource Mapping
//=============================================================================
// Demonstrates: How static arrays map to FPGA memory resources (BRAM/URAM/Registers)
// Hardware Impact: static arrays become BRAM/URAM, non-static arrays become wires/registers

// Version A: Non-static array (local, small)
void top_a(int data[10], int* result) {
    int local_array[10];  // Non-static: synthesized as registers/wires
    #pragma HLS ARRAY_PARTITION variable=local_array complete
    
    for (int i = 0; i < 10; i++) {
        local_array[i] = data[i] * 2;
    }
    
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += local_array[i];
    }
    *result = sum;
}

// Version B: Static array (large, mapped to BRAM)
void top_b(int data[100], int* result) {
    static int static_array[100];  // Static: synthesized as BRAM
    #pragma HLS ARRAY_PARTITION variable=static_array block factor=4
    
    for (int i = 0; i < 100; i++) {
        static_array[i] = data[i] * 2;
    }
    
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += static_array[i];
    }
    *result = sum;
}

// Version C: Static array with initialization
void top_c(int data[10], int* result) {
    static int static_init[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};  // Initialized once
    
    for (int i = 0; i < 10; i++) {
        static_init[i] = static_init[i] + data[i];
    }
    
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += static_init[i];
    }
    *result = sum;
}

