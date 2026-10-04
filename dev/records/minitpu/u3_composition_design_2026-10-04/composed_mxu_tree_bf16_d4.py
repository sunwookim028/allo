@df.region()
def mxu_tree_bf16_acc24_d4(
    A: UInt(32)[N_ROWS * DIM],
    W: UInt(32)[DIM * DIM],
    OUT: UInt(32)[N_ROWS * DIM],
):
    me_lhs: Stream[UInt(DIM * MAC_IN_BITS), QD]    # one row of DIM operands
    me_out: Stream[UInt(DIM * MAC_OUT_BITS), QD]    # one row of DIM results

    @df.kernel(mapping=[1], args=[A])
    def me_feed(a_mem: UInt(32)[N_ROWS * DIM]):
        for work in range(N_ROWS):
            packed: UInt(DIM * MAC_IN_BITS) = 0
            with allo.meta_for(DIM) as r:
                a_word: UInt(32) = a_mem[work * DIM + r]
                a: MAC_IN = a_word
                packed[MAC_IN_BITS * r : MAC_IN_BITS * (r + 1)] = a
            me_lhs.put(packed)

    @df.kernel(mapping=[1], args=[W])
    def matrix_engine(w_tile: UInt(32)[DIM * DIM]):
        for work in range(N_ROWS):
            packed: UInt(DIM * MAC_IN_BITS) = me_lhs.get()
            out_word: UInt(DIM * MAC_OUT_BITS) = 0
            node: MAC_ACC[2 * DIM - 1]
            with allo.meta_for(DIM) as c:
                with allo.meta_for(DIM) as r:
                    a: MAC_IN = packed[MAC_IN_BITS * r:MAC_IN_BITS * (r + 1)]
                    w_word: UInt(32) = w_tile[r * DIM + c]
                    w: MAC_IN = w_word
                    node[r] = MAC_MUL(a, w)
                with allo.meta_for(DIM - 1) as n:
                    node[DIM + n] = MAC_ADD(node[2 * n], node[2 * n + 1])
                out_word[MAC_OUT_BITS * c:MAC_OUT_BITS * (c + 1)] = MAC_PACK(node[2 * DIM - 2])
            me_out.put(out_word)

    @df.kernel(mapping=[1], args=[OUT])
    def me_sink(out_mem: UInt(32)[N_ROWS * DIM]):
        for work in range(N_ROWS):
            out_word: UInt(DIM * MAC_OUT_BITS) = me_out.get()
            with allo.meta_for(DIM) as c:
                lane: MAC_OUT = out_word[MAC_OUT_BITS * c : MAC_OUT_BITS * (c + 1)]
                word: UInt(32) = lane
                out_mem[work * DIM + c] = word
