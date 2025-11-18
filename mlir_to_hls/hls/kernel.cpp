#include "kernel.h"

extern "C" void accumulate_kernel(std::int32_t in_value, std::int32_t *out_total) {
#pragma HLS INLINE off
  static std::int32_t accumulator = 0;
  accumulator += in_value;
  *out_total = accumulator;
}




