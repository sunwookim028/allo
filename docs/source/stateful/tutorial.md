Vivado/Vitis HLS Backend

In this tutorial, we will demonstrate how to leverage the Allo DSL to generate stateful Vivado/Vitis HLS C++ code for FPGA applications.

Import Allo

First, we import the necessary packages.

import allo
from allo.ir.types import int32, stateful

Algorithm Definition

We will define a simple stateful scalar accumulator in this tutorial. A stateful function maintains internal state across invocations. This allows us to model algorithms such as running sums, counters, moving averages, and finite-state machines with optimized hardware structures. 

Allo provides the special type annotation, stateful(), for declaring persistent variables whose values are updated each time the function is executed. These variables behave like static variables in C or registers in hardware designs, preserving values across calls. 

Having imported the relevant types from Allo, we define the computation of the accumulator. It takes a single integer input x, adds it to a persistent state variable acc, and returns the updated state. Since Allo enforces strict typing, the state must be explicitly declared as stateful(int32), and its initial value must  match the annotated type. 

def accumulator(x: int32) -> int32:
	state: stateful(int32) = 0
	state = state + x
	return state

We can then inspect the lowered MLIR translation. 

s = allo.customize(accumulator)
print(s.module)

module {
  memref.global "private" @__stateful_accumulator_state_1 : memref<i32> = dense<0>
  func.func @accumulator(%arg0: i32) -> i32 attributes {itypes = "s", otypes = "s"} {
    %0 = memref.get_global @__stateful_accumulator_state_1 : memref<i32>
    %1 = affine.load %0[] {from = "state"} : memref<i32>
    %2 = arith.extsi %1 : i32 to i33
    %3 = arith.extsi %arg0 : i32 to i33
    %4 = arith.addi %2, %3 : i33
    %5 = arith.trunci %4 : i33 to i32
    affine.store %5, %0[] {to = "state"} : memref<i32>
    %6 = affine.load %0[] {from = "state"} : memref<i32>
    return %6 : i32
  }
}

Codegen for Vivado/Vitis HLS

To generate Vivado/Vitis HLS code, we can change the target of the .build() function in order to target different backends. Here, we use vhls as the target to generate Vivado/Vitis HLS code, which will directly return the generated code as a string.

code = s.build(target=”vhls”)
print(code)

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
void accumulator(
  int32_t v0,
  int32_t *v1
) {     // L3
  static int32_t __stateful_accumulator_state_1 = {0};  // L2
  // placeholder for const int32_t __stateful_accumulator_state_1     // L4
  int32_t v3 = __stateful_accumulator_state_1;  // L5
  ap_int<33> v4 = v3;   // L6
  ap_int<33> v5 = v0;   // L7
  ap_int<33> v6 = v4 + v5;      // L8
  int32_t v7 = v6;      // L9
  __stateful_accumulator_state_1 = v7;  // L10
  *v1 = __stateful_accumulator_state_1; // L11
}

Allo also provides an easy way to invoke Vitis HLS. Users can provide the desired synthesis mode supported by Vitis HLS (e.g., sw_emu, hw_emu, and hw) and the target project folder name. Allo will automatically generate the HLS project and invoke the compiler to generate the RTL design. 

mod = s.build(target=”vitis_hls”, mode=”hw_emu”, project=”accum.prj”)

After running the above instruction, we can see an accum.prj folder is generated in the current directory:

host.cpp : The host (CPU) OpenCL code that invokes the generated accumulator
kernel.cpp :  The generated accumulator code
Makefile : Defined shorthands for compiling the project. 

You have to modify the host.cpp manually (for now). Later, we may improve this.

Allo also provides an easy way to invoke the (legacy) Vivado HLS. Users can specify the desired synthesis mode supported by Vivado HLS (e.g., csim, csyn, cosim, impl) and the target project folder name. Allo will automatically generate the HLS project and use run.tcl for the project script. 
