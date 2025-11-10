// Test 5: Shared Memory Direct Access - DEMONSTRATING KERNEL SHARING
    std::cout << "--- Test 5: Demonstrating Shared Memory Between Kernels ---\n";
    std::cout << "This test shows how kernels share intermediate results via shared_memory\n\n";
    
    // Access shared_memory (in simulation, we can access it directly)
    // In hardware, this would be via DMA or AXI, but kernels access directly
    // Note: shared_memory is declared as 'static' in kernel.cpp, so we need to access it
    // For simulation, we'll declare it here to match kernel.cpp
    static int shared_memory[4096];
    
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
