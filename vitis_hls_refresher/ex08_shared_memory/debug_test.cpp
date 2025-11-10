#include <iostream>
#include <cstring>

extern int shared_memory[4096];
extern void matmul_shared(int w_offset, int x_offset, int result_offset, int M, int N, int K);

int main() {
    const int M = 4, N = 4, K = 4;
    int W[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int X[K * N] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    
    const int W_OFFSET = 0;
    const int X_OFFSET = M * K;
    const int TEMP_OFFSET = M * K + K * N + M * N;
    
    std::cout << "Before loading:\n";
    std::cout << "  shared_memory[0:3] = ";
    for (int i = 0; i < 4; i++) std::cout << shared_memory[i] << " ";
    std::cout << "\n";
    
    memcpy(&shared_memory[W_OFFSET], W, M * K * sizeof(int));
    memcpy(&shared_memory[X_OFFSET], X, K * N * sizeof(int));
    
    std::cout << "After loading:\n";
    std::cout << "  shared_memory[0:3] = ";
    for (int i = 0; i < 4; i++) std::cout << shared_memory[i] << " ";
    std::cout << "\n";
    std::cout << "  shared_memory[16:19] = ";
    for (int i = 0; i < 4; i++) std::cout << shared_memory[X_OFFSET + i] << " ";
    std::cout << "\n";
    
    std::cout << "Calling matmul_shared...\n";
    matmul_shared(W_OFFSET, X_OFFSET, TEMP_OFFSET, M, N, K);
    
    std::cout << "After matmul:\n";
    std::cout << "  shared_memory[" << TEMP_OFFSET << ":" << TEMP_OFFSET + 3 << "] = ";
    for (int i = 0; i < 4; i++) std::cout << shared_memory[TEMP_OFFSET + i] << " ";
    std::cout << "\n";
    
    return 0;
}
