//=============================================================================
// Example 1: Basic Static Variable Behavior
//=============================================================================
// Demonstrates: static vs non-static local variables in HLS
// Hardware Impact: static variables become registers/BRAM, non-static become wires

    static int static_var = 0;
void top(int in, int* out) {
    // Non-static variable: synthesized as wire/logic
    // Reset on every function call
    int non_static = 0;
    
    // Static variable: synthesized as register (persists across calls)
    // Initialized once, retains value between function calls
    
    // Simple computation
    non_static = in + 1;
    static_var = static_var + in;
    
    *out = non_static + static_var;
}

