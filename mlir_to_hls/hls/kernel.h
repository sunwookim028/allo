#pragma once

#include <cstdint>

// Simple scalar accumulator kernel that keeps state in a static variable.
// Every invocation adds the incoming value to the persistent accumulator and
// writes the new total into `*out_total`.
extern "C" void accumulate_kernel(std::int32_t in_value, std::int32_t *out_total);




