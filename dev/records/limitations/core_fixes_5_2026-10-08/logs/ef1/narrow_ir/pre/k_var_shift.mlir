module {
  func.func @k_var_shift(%arg0: memref<8xi32>, %arg1: memref<8xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a"} : memref<8xi32>
      %c7_i32 = arith.constant 7 : i32
      %c7_i32_0 = arith.constant 7 : i32
      %1 = arith.andi %0, %c7_i32_0 : i32
      %alloc = memref.alloc() {name = "k"} : memref<i32>
      affine.store %1, %alloc[] {to = "k"} : memref<i32>
      %2 = affine.load %alloc[] {from = "k"} : memref<i32>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %3 = arith.shli %c1_i32_1, %2 : i32
      %4 = affine.load %alloc[] {from = "k"} : memref<i32>
      %5 = arith.extsi %4 : i32 to i33
      %c1_i32_2 = arith.constant 1 : i32
      %c1_i32_3 = arith.constant 1 : i32
      %6 = arith.extsi %c1_i32_3 : i32 to i33
      %7 = arith.addi %5, %6 : i33
      %8 = arith.trunci %7 : i33 to i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_4 = arith.constant 3 : i32
      %9 = arith.shli %c3_i32_4, %8 : i32
      %10 = arith.ori %3, %9 : i32
      affine.store %10, %arg1[%arg2] {to = "o"} : memref<8xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
