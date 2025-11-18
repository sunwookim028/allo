module {
  memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<0>
  func.func @test(%arg0: i32) -> i32 attributes {itypes = "s", otypes = "s"} {
    %0 = memref.get_global @acc_stateful_5700927378943735518 : memref<i32>
    %1 = affine.load %0[] {from = "acc"} : memref<i32>
    %2 = arith.addi %1, %arg0 : i32
    affine.store %2, %0[] {to = "acc"} : memref<i32>
    %3 = affine.load %0[] {from = "acc"} : memref<i32>
    return %3 : i32
  }
}



