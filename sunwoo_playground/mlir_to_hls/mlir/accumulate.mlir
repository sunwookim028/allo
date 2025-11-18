module {
  memref.global "private" @acc_state : memref<i32> = dense<0> : memref<i32> { hls.static }

  func.func @accumulate_kernel(%in: i32, %out: memref<i32>) attributes {top} {
    %state = memref.get_global @acc_state : memref<i32>
    %current = memref.load %state[] : memref<i32>
    %updated = arith.addi %current, %in : i32
    memref.store %updated, %state[] : memref<i32>
    memref.store %updated, %out[] : memref<i32>
    func.return
  }
}


