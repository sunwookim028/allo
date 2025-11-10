//=============================================================================
// Example 5: History Buffer with Shift Registers
//=============================================================================
// Demonstrates: Static arrays as shift registers for delayed output
// Hardware Impact: Static array becomes register chain (tapped delay line)
//
// Interface: history_buffer(buf, reset, in, out, time=1)
//   - buf: internal static buffer (shift register)
//   - reset: reset signal (clears buffer)
//   - in: input value
//   - out: output value (delayed by 'time' cycles)
//   - time: delay amount (default 1, meaning output from 1 invocation ago)

// Version A: Basic shift register with configurable delay
void history_buffer_a(int* buf, int reset, int in, int* out, int time) {
    // Static shift register: persists across calls
    // Size determined by maximum delay needed
    static int shift_reg[16] = {0};  // Support up to 16 cycles delay
    
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    
    // Reset logic: clear entire buffer
    if (reset) {
        for (int i = 0; i < 16; i++) {
            shift_reg[i] = 0;
        }
        *out = 0;
        return;
    }
    
    // Shift operation: move all values one position
    // This creates a tapped delay line
    for (int i = 15; i > 0; i--) {
        shift_reg[i] = shift_reg[i-1];
    }
    shift_reg[0] = in;  // Insert new value at front
    
    // Output: tap at position 'time'
    // time=1 means output from 1 cycle ago (shift_reg[1])
    // time=3 means output from 3 cycles ago (shift_reg[3])
    if (time >= 1 && time < 16) {
        *out = shift_reg[time];
    } else {
        *out = shift_reg[1];  // Default to 1 cycle delay
    }
}

// Version B: Optimized with pipeline
void history_buffer_b(int* buf, int reset, int in, int* out, int time) {
    static int shift_reg[16] = {0};
    
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    #pragma HLS PIPELINE II=1
    
    if (reset) {
        for (int i = 0; i < 16; i++) {
            #pragma HLS UNROLL
            shift_reg[i] = 0;
        }
        *out = 0;
        return;
    }
    
    // Pipelined shift operation
    for (int i = 15; i > 0; i--) {
        #pragma HLS UNROLL
        shift_reg[i] = shift_reg[i-1];
    }
    shift_reg[0] = in;
    
    // Output selection
    if (time >= 1 && time < 16) {
        *out = shift_reg[time];
    } else {
        *out = shift_reg[1];
    }
}

// Version C: Fixed delay (time=3) - most efficient
void history_buffer_c(int* buf, int reset, int in, int* out) {
    // Fixed 3-cycle delay: optimized shift register
    static int shift_reg[4] = {0};  // Need 4 elements for 3-cycle delay
    
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    #pragma HLS PIPELINE II=1
    
    if (reset) {
        shift_reg[0] = 0;
        shift_reg[1] = 0;
        shift_reg[2] = 0;
        shift_reg[3] = 0;
        *out = 0;
        return;
    }
    
    // Simple 3-stage shift
    shift_reg[3] = shift_reg[2];
    shift_reg[2] = shift_reg[1];
    shift_reg[1] = shift_reg[0];
    shift_reg[0] = in;
    
    // Output from 3 cycles ago
    *out = shift_reg[3];
}

