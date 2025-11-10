//=============================================================================
// Example 4: Static Variables in Pipelined Loops
//=============================================================================
// Demonstrates: How static variables affect pipeline initiation interval (II)
// Hardware Impact: static variables can create dependencies that prevent II=1

// Version A: Non-static accumulator in pipeline
void top_a(int data[10], int* result) {
    int acc = 0;  // Non-static: new register per iteration
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < 10; i++) {
        acc = acc + data[i];  // Read-after-write dependency
    }
    *result = acc;
}

// Version B: Static accumulator in pipeline (problematic!)
void top_b(int data[10], int* result) {
    static int acc = 0;  // Static: shared register across calls AND iterations
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < 10; i++) {
        acc = acc + data[i];  // Cannot pipeline well - static persists!
    }
    *result = acc;
}

// Version C: Static array as shift register
void top_c(int data[10], int* result) {
    static int shift_reg[5] = {0};  // Static: shift register pattern
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    
    for (int i = 0; i < 10; i++) {
        // Shift register operation
        for (int j = 4; j > 0; j--) {
            shift_reg[j] = shift_reg[j-1];
        }
        shift_reg[0] = data[i];
    }
    
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += shift_reg[i];
    }
    *result = sum;
}

// Version D: Proper use of static for state machine
void top_d(int data[10], int* result) {
    static int state = 0;  // Static: state variable
    static int counter = 0;
    
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < 10; i++) {
        if (state == 0) {
            counter += data[i];
            if (counter > 50) {
                state = 1;
            }
        } else {
            counter -= data[i];
        }
    }
    *result = counter;
}

