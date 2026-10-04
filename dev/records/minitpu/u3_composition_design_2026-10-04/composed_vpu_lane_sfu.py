@df.region()
def vpu_lane_sfu(
    OPS: UInt(32)[N_OPS],
    A: UInt(32)[N_OPS],
    B: UInt(32)[N_OPS],
    OUT: UInt(32)[N_OPS],
):
    to_alu: Stream[UInt(64), QD]
    alu_out: Stream[UInt(64), QD]
    sfu_out: Stream[UInt(64), QD]

    @df.kernel(mapping=[1], args=[OPS, A, B])
    def issue(ops: UInt(32)[N_OPS], a_mem: UInt(32)[N_OPS], b_mem: UInt(32)[N_OPS]):
        for i in range(N_OPS):
            word: UInt(64) = 0
            word[0:16] = a_mem[i]
            word[16:32] = b_mem[i]
            word[32:36] = ops[i]
            to_alu.put(word)

    @df.kernel(mapping=[1])
    def alu(): # a stub: add is integer add, every other op passes `a` through
        for i in range(N_OPS):
            word: UInt(64) = to_alu.get()
            a: UInt(16) = word[0:16]
            b: UInt(16) = word[16:32]
            op: UInt(4) = word[32:36]
            r: UInt(16) = a
            if op == OP_ADD:
                r = a + b
            out: UInt(64) = word
            out[0:16] = r
            alu_out.put(out)

    @df.kernel(mapping=[1], args=[OUT])
    def writeback(out_mem: UInt(32)[N_OPS]):
        for i in range(N_OPS):
            word: UInt(64) = sfu_out.get()
            out_mem[i] = word[0:16]

    @df.kernel(mapping=[1])
    def sfu(): # a stub: GELU is `x ^ 0x5555` here; the real SFU is track A's S1
        for i in range(N_OPS):
            word: UInt(64) = alu_out.get()
            op: UInt(4) = word[32:36]
            r: UInt(16) = word[0:16]
            if op == OP_GELU:
                r = r ^ 0x5555
            out: UInt(64) = word
            out[0:16] = r
            sfu_out.put(out)
