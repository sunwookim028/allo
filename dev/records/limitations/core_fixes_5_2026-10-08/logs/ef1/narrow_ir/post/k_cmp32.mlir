module {
  func.func @k_cmp32(%arg0: memref<8xi32>, %arg1: memref<8xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a"} : memref<8xi32>
      %alloc = memref.alloc() {name = "x"} : memref<i32>
      affine.store %0, %alloc[] {to = "x"} : memref<i32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %alloc_1 = memref.alloc() {name = "c"} : memref<i32>
      affine.store %c0_i32_0, %alloc_1[] {to = "c"} : memref<i32>
      %1 = affine.load %alloc[] {from = "x"} : memref<i32>
      %c-1_i32 = arith.constant -1 : i32
      %c-1_i32_2 = arith.constant -1 : i32
      %2 = arith.cmpi eq, %1, %c-1_i32_2 : i32
      scf.if %2 {
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_5 = arith.constant 1 : i32
        affine.store %c1_i32_5, %alloc_1[] {to = "c"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "x"} : memref<i32>
      %c5_i32 = arith.constant 5 : i32
      %c5_i32_3 = arith.constant 5 : i32
      %c0_i32_4 = arith.constant 0 : i32
      %4 = arith.subi %c0_i32_4, %c5_i32_3 : i32
      %5 = arith.cmpi slt, %3, %4 : i32
      scf.if %5 {
        %7 = affine.load %alloc_1[] {from = "c"} : memref<i32>
        %8 = arith.extsi %7 : i32 to i33
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_5 = arith.constant 2 : i32
        %9 = arith.extsi %c2_i32_5 : i32 to i33
        %10 = arith.addi %8, %9 : i33
        %11 = arith.trunci %10 : i33 to i32
        affine.store %11, %alloc_1[] {to = "c"} : memref<i32>
      }
      %6 = affine.load %alloc_1[] {from = "c"} : memref<i32>
      affine.store %6, %arg1[%arg2] {to = "o"} : memref<8xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
