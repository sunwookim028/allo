module {
  func.func @k_masks(%arg0: memref<8xi32>, %arg1: memref<8xi32>) attributes {itypes = "uu", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a", unsigned} : memref<8xi32>
      %alloc = memref.alloc() {name = "x", unsigned} : memref<i32>
      affine.store %0, %alloc[] {to = "x"} : memref<i32>
      %1 = affine.load %alloc[] {from = "x", unsigned} : memref<i32>
      %c-1_i32 = arith.constant -1 : i32
      %c-1_i32_0 = arith.constant -1 : i32
      %2 = arith.andi %1, %c-1_i32_0 : i32
      %3 = affine.load %alloc[] {from = "x", unsigned} : memref<i32>
      %c-2147483648_i32 = arith.constant -2147483648 : i32
      %c-2147483648_i32_1 = arith.constant -2147483648 : i32
      %4 = arith.andi %3, %c-2147483648_i32_1 : i32
      %5 = arith.ori %2, %4 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_2 = arith.constant 1 : i32
      %c31_i32 = arith.constant 31 : i32
      %c31_i32_3 = arith.constant 31 : i32
      %6 = arith.shli %c1_i32_2, %c31_i32_3 : i32
      %7 = arith.ori %5, %6 : i32
      affine.store %7, %arg1[%arg2] {to = "o"} : memref<8xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
