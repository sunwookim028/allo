module {
  func.func @k_i8_neg(%arg0: memref<8xi8>, %arg1: memref<8xi8>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a"} : memref<8xi8>
      %alloc = memref.alloc() {name = "x"} : memref<i8>
      affine.store %0, %alloc[] {to = "x"} : memref<i8>
      %1 = affine.load %alloc[] {from = "x"} : memref<i8>
      %c2_i32 = arith.constant 2 : i32
      %c2_i32_0 = arith.constant 2 : i32
      %c0_i32 = arith.constant 0 : i32
      %2 = arith.subi %c0_i32, %c2_i32_0 : i32
      %3 = arith.extsi %1 : i8 to i32
      %4 = arith.andi %3, %2 : i32
      %5 = arith.extsi %4 : i32 to i33
      %c128_i32 = arith.constant 128 : i32
      %c128_i32_1 = arith.constant 128 : i32
      %6 = arith.extsi %c128_i32_1 : i32 to i33
      %7 = arith.subi %5, %6 : i33
      %8 = arith.trunci %7 : i33 to i8
      affine.store %8, %arg1[%arg2] {to = "o"} : memref<8xi8>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
