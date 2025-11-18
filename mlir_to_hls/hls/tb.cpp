#include "kernel.h"

#include <array>
#include <iostream>

int main() {
  constexpr std::array<std::int32_t, 6> inputs = {1, 2, 3, -4, 10, -7};
  constexpr std::array<std::int32_t, inputs.size()> expected = {1, 3, 6, 2, 12, 5};

  bool pass = true;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::int32_t out_value = 0;
    accumulate_kernel(inputs[i], &out_value);
    if (out_value != expected[i]) {
      std::cerr << "Mismatch at iteration " << i << ": got " << out_value
                << ", expected " << expected[i] << '\n';
      pass = false;
    }
  }

  if (!pass) {
    std::cerr << "Test FAILED\n";
    return 1;
  }

  std::cout << "Test PASSED\n";
  return 0;
}




