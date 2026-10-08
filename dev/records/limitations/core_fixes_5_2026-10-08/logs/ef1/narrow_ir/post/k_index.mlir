module {
  func.func @k_index(%arg0: memref<8xi32>, %arg1: memref<8xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 1 to 7 {
      %0 = affine.load %arg0[%arg2 + 1] {from = "a"} : memref<8xi32>
      %1 = affine.load %arg0[%arg2 - 1] {from = "a"} : memref<8xi32>
      %2 = arith.extsi %0 : i32 to i33
      %3 = arith.extsi %1 : i32 to i33
      %4 = arith.addi %2, %3 : i33
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_0 = arith.constant 1 : i32
      %5 = arith.index_cast %c1_i32_0 : i32 to index
      %6 = arith.shli %arg2, %5 : index
      %7 = arith.extsi %4 : i33 to i34
      %8 = arith.index_cast %6 : index to i34
      %9 = arith.addi %7, %8 : i34
      %10 = arith.trunci %9 : i34 to i32
      affine.store %10, %arg1[%arg2] {to = "o"} : memref<8xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
