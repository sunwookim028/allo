// M-R1 probe: one RTL top with a MemPort (16 x 32-bit RAM) AND two ready/valid stream ports.
// Per command word c: read M[c[3:0]], write M[c[3:0]] = M[c[3:0]] + c[31:16], emit the old value on st.
// c[15] marks the last command; done rises after its status word is taken.
module mixed_probe (
  input  logic        clk, rst_n,
  input  logic [31:0] cmd_data, input logic cmd_valid, output logic cmd_ready,
  output logic [31:0] st_data, output logic st_valid, input logic st_ready,
  output logic [3:0]  m_addr, output logic m_ce, output logic m_we, output logic [31:0] m_d,
  input  logic [31:0] m_q,
  output logic        done
);
  localparam logic [2:0] IDLE = 0, RD = 1, WR = 2, OUT = 3, FIN = 4;
  logic [2:0] s;
  logic [31:0] c, q;
  assign cmd_ready = (s == IDLE);
  assign m_ce = (s == RD) || (s == WR);   // RD: read request; WR: m_q holds the read, write it back
  assign m_we = (s == WR);
  assign m_addr = c[3:0];
  assign m_d = m_q + {16'b0, c[31:16]};
  assign st_valid = (s == OUT);
  assign st_data = q;
  assign done = (s == FIN);
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin s <= IDLE; c <= '0; q <= '0; end
    else case (s)
      IDLE: if (cmd_valid) begin c <= cmd_data; s <= RD; end
      RD:   s <= WR;
      WR:   begin q <= m_q; s <= OUT; end
      OUT:  if (st_ready) s <= c[15] ? FIN : IDLE;
      default: s <= FIN;
    endcase
endmodule
