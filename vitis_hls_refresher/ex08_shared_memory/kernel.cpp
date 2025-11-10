//=============================================================================
// Example 8: Shared Static Memory Across Kernels
//=============================================================================
// Demonstrates: Properly shared static variables across multiple kernels
// Hardware Impact: Shared BRAM accessible by multiple kernels
//
// This example shows:
//   - MatMul kernel: Computes matrix multiplication W*X
//   - Elemwise kernel: Computes element-wise operations (add, multiply)
//   - Shared memory: Both kernels access the same static array
//   - Host programming: Multiple tensor operations (H=W*X+B, A=X*Y+W*Z)

// Shared memory: Accessible by all kernels
// This must be declared at file scope (not inside a function) to be shared
// Forward declaration - actual definition in kernel.cpp
int shared_memory[4096];  // Shared across all kernels - DEFINITION

// Interface pragmas for shared memory access
// When shared memory is accessed via pointers, use m_axi for external access
// or ap_memory for internal shared access

//=============================================================================
// Kernel 1: Matrix Multiplication Engine
//=============================================================================
// Computes: C = A * B
// Uses shared_memory for input/output tensors
void matmul_kernel(int* W, int* X, int* result, int M, int N, int K) {
    // W: MxK matrix (stored in shared_memory starting at offset 0)
    // X: KxN matrix (stored in shared_memory starting at offset M*K)
    // result: MxN matrix (stored in shared_memory starting at offset M*K + K*N)
    
    #pragma HLS INTERFACE m_axi port=W depth=4096
    #pragma HLS INTERFACE m_axi port=X depth=4096
    #pragma HLS INTERFACE m_axi port=result depth=4096
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return
    
    // Copy inputs to shared memory
    int w_offset = 0;
    int x_offset = M * K;
    int result_offset = M * K + K * N;
    
    // Load W matrix into shared memory
    for (int i = 0; i < M * K; i++) {
        #pragma HLS PIPELINE II=1
        shared_memory[w_offset + i] = W[i];
    }
    
    // Load X matrix into shared memory
    for (int i = 0; i < K * N; i++) {
        #pragma HLS PIPELINE II=1
        shared_memory[x_offset + i] = X[i];
    }
    
    // Compute matrix multiplication: result = W * X
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += shared_memory[w_offset + i * K + k] * 
                       shared_memory[x_offset + k * N + j];
            }
            shared_memory[result_offset + i * N + j] = sum;
        }
    }
    
    // Copy result back
    for (int i = 0; i < M * N; i++) {
        #pragma HLS PIPELINE II=1
        result[i] = shared_memory[result_offset + i];
    }
}

//=============================================================================
// Kernel 2: Element-wise Operations Engine
//=============================================================================
// Computes: C = A op B (element-wise add or multiply)
// Uses shared_memory for input/output tensors
void elemwise_kernel(int* A, int* B, int* result, int size, int op) {
    // A: input vector (stored in shared_memory starting at offset 0)
    // B: input vector (stored in shared_memory starting at offset size)
    // result: output vector (stored in shared_memory starting at offset 2*size)
    // op: 0 = add, 1 = multiply
    
    #pragma HLS INTERFACE m_axi port=A depth=4096
    #pragma HLS INTERFACE m_axi port=B depth=4096
    #pragma HLS INTERFACE m_axi port=result depth=4096
    #pragma HLS INTERFACE s_axilite port=size
    #pragma HLS INTERFACE s_axilite port=op
    #pragma HLS INTERFACE s_axilite port=return
    
    int a_offset = 0;
    int b_offset = size;
    int result_offset = 2 * size;
    
    // Load inputs to shared memory
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        shared_memory[a_offset + i] = A[i];
        shared_memory[b_offset + i] = B[i];
    }
    
    // Perform element-wise operation
    if (op == 0) {
        // Add: C = A + B
        for (int i = 0; i < size; i++) {
            #pragma HLS PIPELINE II=1
            shared_memory[result_offset + i] = 
                shared_memory[a_offset + i] + shared_memory[b_offset + i];
        }
    } else {
        // Multiply: C = A * B
        for (int i = 0; i < size; i++) {
            #pragma HLS PIPELINE II=1
            shared_memory[result_offset + i] = 
                shared_memory[a_offset + i] * shared_memory[b_offset + i];
        }
    }
    
    // Copy result back
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        result[i] = shared_memory[result_offset + i];
    }
}

//=============================================================================
// Top Module: Coordinates Multiple Operations
//=============================================================================
// Demonstrates: Host programming multiple tensor operations
// Operations:
//   1. H = W * X + B  (matrix multiply + bias)
//   2. A = X * Y + W * Z  (two matrix multiplies + add)
void top_module(int* W, int* X, int* B, int* Y, int* Z, 
                int* H, int* A, 
                int M, int N, int K) {
    // H = W * X + B
    // Step 1: Compute W * X using matmul kernel
    int temp_H[256];  // Temporary storage for W*X
    matmul_kernel(W, X, temp_H, M, N, K);
    
    // Step 2: Add bias B (element-wise)
    elemwise_kernel(temp_H, B, H, M * N, 0);  // op=0 means add
    
    // A = X * Y + W * Z
    // Step 1: Compute X * Y
    int temp_XY[256];
    matmul_kernel(X, Y, temp_XY, K, N, K);
    
    // Step 2: Compute W * Z
    int temp_WZ[256];
    matmul_kernel(W, Z, temp_WZ, M, N, K);
    
    // Step 3: Add results (element-wise)
    elemwise_kernel(temp_XY, temp_WZ, A, M * N, 0);  // op=0 means add
}

//=============================================================================
// Alternative: Direct Shared Memory Access (More Efficient)
//=============================================================================
// This version directly uses shared_memory without copying through AXI
// More efficient but requires careful memory management

void matmul_shared(int w_offset, int x_offset, int result_offset, 
                   int M, int N, int K) {
    // Direct access to shared_memory using offsets
    // No AXI transfers needed - data already in shared memory
    
    #pragma HLS INTERFACE s_axilite port=w_offset
    #pragma HLS INTERFACE s_axilite port=x_offset
    #pragma HLS INTERFACE s_axilite port=result_offset
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return
    
    // Compute matrix multiplication directly in shared memory
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += shared_memory[w_offset + i * K + k] * 
                       shared_memory[x_offset + k * N + j];
            }
            shared_memory[result_offset + i * N + j] = sum;
        }
    }
}

void elemwise_shared(int a_offset, int b_offset, int result_offset, 
                     int size, int op) {
    // Direct access to shared_memory using offsets
    
    #pragma HLS INTERFACE s_axilite port=a_offset
    #pragma HLS INTERFACE s_axilite port=b_offset
    #pragma HLS INTERFACE s_axilite port=result_offset
    #pragma HLS INTERFACE s_axilite port=size
    #pragma HLS INTERFACE s_axilite port=op
    #pragma HLS INTERFACE s_axilite port=return
    
    // Perform element-wise operation directly in shared memory
    if (op == 0) {
        // Add
        for (int i = 0; i < size; i++) {
            #pragma HLS PIPELINE II=1
            shared_memory[result_offset + i] = 
                shared_memory[a_offset + i] + shared_memory[b_offset + i];
        }
    } else {
        // Multiply
        for (int i = 0; i < size; i++) {
            #pragma HLS PIPELINE II=1
            shared_memory[result_offset + i] = 
                shared_memory[a_offset + i] * shared_memory[b_offset + i];
        }
    }
}

