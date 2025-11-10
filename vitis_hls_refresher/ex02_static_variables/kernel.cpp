//=============================================================================
// Example 2: Static Variables in Loops
//=============================================================================
// Demonstrates: How static variables affect loop pipelining and resource usage
// Hardware Impact: static variables create registers that persist across iterations

// Version A: Non-static accumulator
void top_a(int data[10], int* result) {
    int acc = 0;  // Non-static: reset every iteration
    #pragma HLS PIPELINE
    for (int i = 0; i < 10; i++) {
        acc = acc + data[i];
    }
    *result = acc;
}

// Version B: Static accumulator
void top_b(int data[10], int* result) {
    static int acc = 0;  // Static: retains value across iterations
    #pragma HLS PIPELINE
    for (int i = 0; i < 10; i++) {
        acc = acc + data[i];
    }
    *result = acc;
}

// Version C: Static with reset capability
void top_c(int data[10], int reset, int* result) {
    static int acc = 0;
    if (reset) {
        acc = 0;  // Manual reset
    }
    #pragma HLS PIPELINE
    for (int i = 0; i < 10; i++) {
        acc = acc + data[i];
    }
    *result = acc;
}

