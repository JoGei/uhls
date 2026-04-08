module GEN_FF_MEM #(
  parameter W = 32,
  parameter DEPTH = 256
) (
  input clk,
  input [31:0] addr,
  input signed [W-1:0] wdata,
  input we,
  output signed [W-1:0] rdata
);
  localparam integer IDX_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

  reg signed [W-1:0] mem_q [0:DEPTH-1];
  wire [IDX_W-1:0] addr_idx;

  assign addr_idx = addr[IDX_W-1:0];
  assign rdata = (addr < DEPTH) ? mem_q[addr_idx] : {W{1'b0}};

  always @(posedge clk) begin
    if (we && addr < DEPTH) begin
      mem_q[addr_idx] <= wdata;
    end
  end
endmodule
