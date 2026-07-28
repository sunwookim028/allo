"""Generate the self-checking vectors for the legacy compute_tile snapshot.

Keep this script dependency-free: the mflowgen testbench step runs with the
system Python on ASIC hosts, where NumPy is not guaranteed to be installed.
"""

import struct

N = 8
OP_MATMUL_TILE = 0x01
FLUSH_INSTR = 0x5200000000000000
ACC_DST = 2 * N
SPM_A = N
SPM_B = 0


def fp16_bits(value):
    """Return the IEEE-754 binary16 encoding of *value*."""
    return int.from_bytes(struct.pack(">e", float(value)), byteorder="big")


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


def pack_spad_row_fp16(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= (fp16_bits(value) & 0xFFFF) << (i * 16)
    return f"{result:064x}"


def pack_spad_row_fp32(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= (fp32_bits(value) & 0xFFFFFFFF) << (i * 32)
    return f"{result:064x}"


def main():
    # Fixed, small integers are exactly representable in fp16 and keep the
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
        encode_instr(OP_MATMUL_TILE, op0=ACC_DST, op1=SPM_A, op2=SPM_B),
        FLUSH_INSTR,
    ]

    rows = []
    rows.extend(pack_program_word(word) for word in program)
    rows.extend(pack_spad_row_fp16(w_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_fp16(x_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_fp32(expected[i]) for i in range(N))

    with open("test_vectors.txt", "w", encoding="ascii") as f:
        for row in rows:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
