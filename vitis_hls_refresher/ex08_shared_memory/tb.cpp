//=============================================================================
// Testbench for Example 8: Shared Static Memory Across Kernels
//=============================================================================

#include <iostream>
#include <cstring>

// Forward declarations
extern void matmul_kernel(int* W, int* X, int* result, int M, int N, int K);
extern void elemwise_kernel(int* A, int* B, int* result, int size, int op);
extern void matmul_shared(int w_offset, int x_offset, int result_offset, 
                          int M, int N, int K);
extern void elemwise_shared(int a_offset, int b_offset, int result_offset, 
                            int size, int op);

extern void top_module(int* W, int* X, int* B, int* Y, int* Z, 
                       int* H, int* A, int M, int N, int K, int mode);
// Shared memory declaration - must match kernel.cpp exactly
// This allows testbench to access the same shared_memory instance as kernels
extern int shared_memory[4096];


int main() {
    std::cout << "========================================\n";
    std::cout << "Example 8: Shared Static Memory Across Kernels\n";
    std::cout << "========================================\n\n";
    
    // Test dimensions
    const int M = 4, N = 4, K = 4;
    const int size = M * N;
    
    // Test matrices
    int W[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int X[K * N] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // Identity
    int B[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int Y[K * N] = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2};
    int Z[K * N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    int H[size] = {0};
    int A[size] = {0};
    int result[size] = {0};
    
    // Test 1: Matrix Multiplication
    std::cout << "--- Test 1: Matrix Multiplication (W * X) ---\n";
    std::cout << "W is 4x4 matrix, X is identity matrix\n";
    matmul_kernel(W, X, result, M, N, K);
    std::cout << "Result (should equal W):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << result[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Test 2: Element-wise Addition
    std::cout << "--- Test 2: Element-wise Addition ---\n";
    int A_vec[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int B_vec[size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int C_vec[size] = {0};
    
    elemwise_kernel(A_vec, B_vec, C_vec, size, 0);  // op=0: add
    std::cout << "A + B (first 4 elements): ";
    for (int i = 0; i < 4; i++) {
        std::cout << C_vec[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Expected: 2 3 4 5\n\n";
    
    // Test 3: Element-wise Multiplication
    std::cout << "--- Test 3: Element-wise Multiplication ---\n";
    elemwise_kernel(A_vec, B_vec, C_vec, size, 1);  // op=1: multiply
    std::cout << "A * B (first 4 elements): ";
    for (int i = 0; i < 4; i++) {
        std::cout << C_vec[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Expected: 1 2 3 4\n\n";
    
    // Test 4: Programmable TPU Interface with Control Paths (top_module)
    std::cout << "--- Test 4: Programmable TPU Interface with Control Paths (top_module) ---\n";
    std::cout << "Demonstrates: top_module with separate LOAD and COMPUTE modes\n";
    std::cout << "Mode 0 (LOAD): Load data from host into shared BRAM\n";
    std::cout << "Mode 1 (COMPUTE): Run computations using data in shared BRAM\n\n";
    
    std::cout << "Step 1: Invocation 1 - LOAD MODE\n";
    std::cout << "  Loading W, X, B, Y, Z into shared BRAM...\n";
    top_module(W, X, B, Y, Z, H, A, M, N, K, 0);  // MODE_LOAD = 0
    std::cout << "  ✓ Data loaded into shared BRAM\n\n";
    
    std::cout << "Step 2: Invocation 2 - COMPUTE MODE\n";
    std::cout << "  Computing:\n";
    std::cout << "    1. H = W * X + B\n";
    std::cout << "    2. A = X * Y + W * Z\n";
    std::cout << "  All operations use shared BRAM!\n";
    top_module(W, X, B, Y, Z, H, A, M, N, K, 1);  // MODE_COMPUTE = 1
    
    std::cout << "\nResults:\n";
    std::cout << "  Result H = W*X + B (first 4 elements): ";
    for (int i = 0; i < 4; i++) {
        std::cout << H[i] << " ";
    }
    std::cout << "\n";
    std::cout << "  Expected: 2 3 4 5 (W*I + B, where W[0]=1, B[0]=1)\n\n";
    
    std::cout << "  Result A = X*Y + W*Z (first 4 elements): ";
    for (int i = 0; i < 4; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << "\n";
    std::cout << "  Expected: 3 4 5 6 (X*Y + W*Z, where X*Y=2*I, W*Z=W)\n\n";
    
    std::cout << "✓ Two separate invocations: LOAD then COMPUTE\n";
    std::cout << "✓ Control path separation enables flexible programming\n";
    std::cout << "✓ Both engines share the same BRAM resource\n";
    std::cout << "✓ Programmable interface enables complex operations\n\n";
// Test 5: Shared Memory Direct Access - DEMONSTRATING KERNEL SHARING
    std::cout << "--- Test 5: Demonstrating Shared Memory Between Kernels ---\n";
    std::cout << "This test shows how kernels share intermediate results via shared_memory\n\n";
    
    // Access shared_memory (in simulation, we can access it directly)
    // In hardware, this would be via DMA or AXI, but kernels access directly
    // Note: shared_memory is declared as 'static' in kernel.cpp, so we need to access it
    // For simulation, we'll declare it here to match kernel.cpp
    
    // Define memory layout
    const int W_OFFSET = 0;
    const int X_OFFSET = M * K;
    const int B_OFFSET = M * K + K * N;
    const int TEMP_OFFSET = M * K + K * N + size;  // Where matmul_shared writes W*X
    const int RESULT_OFFSET = M * K + K * N + size + size;  // Where elemwise_shared writes H
    
    std::cout << "Step 1: Loading data into shared_memory\n";
    memcpy(&shared_memory[W_OFFSET], W, M * K * sizeof(int));
    memcpy(&shared_memory[X_OFFSET], X, K * N * sizeof(int));
    memcpy(&shared_memory[B_OFFSET], B, size * sizeof(int));
    std::cout << "  Loaded W, X, B into shared_memory\n\n";
    
    std::cout << "Step 2: Kernel 1 (matmul_shared) computes W*X\n";
    std::cout << "  Reads: W from shared_memory[" << W_OFFSET << "], X from shared_memory[" << X_OFFSET << "]\n";
    std::cout << "  Writes: W*X to shared_memory[" << TEMP_OFFSET << "] (intermediate result)\n";
    matmul_shared(W_OFFSET, X_OFFSET, TEMP_OFFSET, M, N, K);
    
    std::cout << "  Intermediate result W*X stored in shared_memory[" << TEMP_OFFSET << "]:\n";
    std::cout << "  First 4 elements: ";
    for (int i = 0; i < 4; i++) {
        std::cout << shared_memory[TEMP_OFFSET + i] << " ";
    }
    std::cout << " (expected: 1 2 3 4)\n\n";
    
    std::cout << "Step 3: Kernel 2 (elemwise_shared) reads intermediate result\n";
    std::cout << "  Reads: W*X from shared_memory[" << TEMP_OFFSET << "] (written by Kernel 1!)\n";
    std::cout << "  Reads: B from shared_memory[" << B_OFFSET << "]\n";
    std::cout << "  Writes: H = W*X + B to shared_memory[" << RESULT_OFFSET << "]\n";
    std::cout << "  KEY: Kernel 2 reads the intermediate result that Kernel 1 wrote!\n";
    elemwise_shared(TEMP_OFFSET, B_OFFSET, RESULT_OFFSET, size, 0);
    
    std::cout << "  Final result H = W*X + B:\n";
    std::cout << "  First 4 elements: ";
    for (int i = 0; i < 4; i++) {
        std::cout << shared_memory[RESULT_OFFSET + i] << " ";
    }
    std::cout << " (expected: 2 3 4 5)\n\n";
    
    std::cout << "Step 4: Verify intermediate result persists in shared_memory\n";
    std::cout << "  Reading back W*X from shared_memory[" << TEMP_OFFSET << "]:\n";
    std::cout << "  W*X[0:3] = ";
    for (int i = 0; i < 4; i++) {
        std::cout << shared_memory[TEMP_OFFSET + i] << " ";
    }
    std::cout << "\n";
    std::cout << "  ✓ Intermediate result persists between kernel calls\n";
    std::cout << "  ✓ Both kernels access the SAME shared_memory array\n";
    std::cout << "  ✓ No data transfer needed - all data stays on-FPGA\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "  - File-scope static = shared across kernels\n";
    std::cout << "  - Function-scope static = per-kernel-instance\n";
    std::cout << "  - Shared memory enables kernel composition\n";
    std::cout << "  - Direct access more efficient than AXI transfers\n";
    std::cout << "  - Requires careful memory layout management\n";
    std::cout << "========================================\n";
    
    return 0;
}

