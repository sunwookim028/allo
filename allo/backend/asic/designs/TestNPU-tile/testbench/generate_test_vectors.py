"""Generate the self-checking vectors for the legacy compute_tile snapshot.

Keep this script dependency-free: the mflowgen testbench step runs with the
system Python on ASIC hosts, where NumPy is not guaranteed to be installed.
"""

import struct

N = 8
OP_MXU_LOAD_W = 0x00
OP_MATMUL_TILE = 0x01
OP_VREG_LOAD = 0x32
OP_HALT = 0x5F
ACC_DST = 2 * N
SPM_W = 0
SPM_X = N
VREG_X = 0


def fp32_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def encode_instr(op, flags=0, op0=0, op1=0, op2=0):
    return (
        ((op & 0xFF) << 56)
        | ((flags & 0xFF) << 48)
        | ((op0 & 0xFFFF) << 32)
        | ((op1 & 0xFFFF) << 16)
        | (op2 & 0xFFFF)
    )


def pack_program_word(word):
    return f"{word & ((1 << 64) - 1):064x}"


def pack_spad_row_bf16(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= ((fp32_bits(value) >> 16) & 0xFFFF) << (i * 16)
    return f"{result:064x}"


def pack_spad_row_fp32(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= (fp32_bits(value) & 0xFFFFFFFF) << (i * 32)
    return f"{result:064x}"


def main():
    # Fixed, small integers are exactly representable in bf16 and keep the
    # gate-level result deterministic while exercising positive, negative, and
    # zero values.  The legacy MXU implements X @ W^T.
    w_mat = [[float(((3 * row + 2 * col + 1) % 5) - 2) for col in range(N)]
             for row in range(N)]
    x_mat = [[float(((2 * row + 3 * col + 2) % 5) - 2) for col in range(N)]
             for row in range(N)]
    expected = [
        [sum(x_mat[row][k] * w_mat[col][k] for k in range(N))
         for col in range(N)]
        for row in range(N)
    ]

    program = [
        # The integrated tile sources MXU activations from eight consecutive
        # VREGs, so widen each BF16 SPAD row into VREG before starting the MXU.
        *(encode_instr(OP_VREG_LOAD, op0=VREG_X + row, op1=SPM_X + row)
          for row in range(N)),
        encode_instr(OP_MXU_LOAD_W, op0=SPM_W),
        encode_instr(OP_MATMUL_TILE, op0=ACC_DST, op1=VREG_X),
        encode_instr(OP_HALT),
    ]

    rows = []
    rows.extend(pack_program_word(word) for word in program)
    rows.extend(pack_spad_row_bf16(w_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_bf16(x_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_fp32(expected[i]) for i in range(N))

    with open("test_vectors.txt", "w", encoding="ascii") as f:
        for row in rows:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
