#pragma once

#include <cstdint>
#include "bram_defs.h"

namespace mini_tpu {

inline Command MakeLoad(std::uint32_t host_addr, std::uint32_t bram_addr,
                        std::uint32_t length_words) {
  return Command{OP_LOAD, host_addr, bram_addr, 0u, length_words, 0};
}

inline Command MakeStore(std::uint32_t bram_addr, std::uint32_t host_addr,
                         std::uint32_t length_words) {
  return Command{OP_STORE, bram_addr, host_addr, 0u, length_words, 0};
}

inline Command MakeMatMul(std::uint32_t a_addr, std::uint32_t b_addr,
                          std::uint32_t out_addr) {
  return Command{OP_MATMUL, a_addr, b_addr, out_addr, MATRIX_WORDS, 0};
}

inline Command MakeVecAdd(std::uint32_t a_addr, std::uint32_t b_addr,
                          std::uint32_t out_addr, std::uint32_t length_words) {
  return Command{OP_VECADD, a_addr, b_addr, out_addr, length_words, 0};
}

inline Command MakeScale(std::uint32_t in_addr, std::uint32_t out_addr,
                         std::uint32_t length_words, std::int32_t scalar) {
  return Command{OP_SCALE, in_addr, out_addr, 0u, length_words, scalar};
}

}  // namespace mini_tpu


