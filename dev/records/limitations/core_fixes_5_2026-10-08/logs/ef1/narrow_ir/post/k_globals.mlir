module {
  func.func @k_globals(%arg0: memref<8xi16>, %arg1: memref<8xi16>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a"} : memref<8xi16>
      %alloc = memref.alloc() {name = "x"} : memref<i16>
      affine.store %0, %alloc[] {to = "x"} : memref<i16>
      %1 = affine.load %alloc[] {from = "x"} : memref<i16>
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %2 = arith.trunci %c3_i32_0 : i32 to i16
      %3 = arith.shli %1, %2 : i16
      %c8_i32 = arith.constant 8 : i32
      %c8_i32_1 = arith.constant 8 : i32
      %4 = arith.extsi %c8_i32_1 : i32 to i64
      %c4_i32 = arith.constant 4 : i32
      %c4_i32_2 = arith.constant 4 : i32
      %5 = arith.extsi %c4_i32_2 : i32 to i64
      %6 = arith.muli %4, %5 : i64
      %7 = arith.extui %3 : i16 to i65
      %8 = arith.extsi %6 : i64 to i65
      %9 = arith.addi %7, %8 : i65
      %10 = arith.extsi %9 : i65 to i66
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_3 = arith.constant 1 : i32
      %11 = arith.extsi %c1_i32_3 : i32 to i66
      %12 = arith.subi %10, %11 : i66
      %13 = arith.trunci %12 : i66 to i16
      affine.store %13, %arg1[%arg2] {to = "o"} : memref<8xi16>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
