module GEN_MUL_PSEUDOPIPE_II1D3 (
  input clk,
  input rst,
  input signed [31:0] a,
  input signed [31:0] b,
  output signed [31:0] y_lo,
  output signed [31:0] y_hi
);
  reg signed [31:0] stage1_a_q;
  reg signed [31:0] stage1_b_q;
  wire signed [63:0] y_n;

  assign y_n = stage1_a_q * stage1_b_q;
  assign y_lo = y_n[31:0];
  assign y_hi = y_n[63:32];

  always @(posedge clk) begin
    if (rst) begin
      stage1_a_q <= 32'sd0;
      stage1_b_q <= 32'sd0;
    end else begin
      stage1_a_q <= a;
      stage1_b_q <= b;
    end
  end
endmodule
