module GEN_DIV_PSEUDOPIPE_II1D3 #(
  parameter W = 32
) (
  input clk,
  input rst,
  input signed [W-1:0] a,
  input signed [W-1:0] b,
  input [0:0] op,
  output signed [W-1:0] div,
  output signed [W-1:0] mod
);
  reg signed [W-1:0] stage1_a_q;
  reg signed [W-1:0] stage1_b_q;
  reg [0:0] stage1_op_q;

  reg signed [W-1:0] stage2_a_q;
  reg signed [W-1:0] stage2_b_q;
  reg [0:0] stage2_op_q;

  assign div = (stage2_b_q == 'sd0) ? 'sd0 : (stage2_a_q / stage2_b_q);
  assign mod = (stage2_b_q == 'sd0) ? 'sd0 : (stage2_a_q % stage2_b_q);

  always @(posedge clk) begin
    if (rst) begin
      stage1_a_q <= 'sd0;
      stage1_b_q <= 'sd0;
      stage1_op_q <= 'b0;
      stage2_a_q <= 'sd0;
      stage2_b_q <= 'sd0;
      stage2_op_q <= 'b0;
    end else begin
      stage1_a_q <= a;
      stage1_b_q <= b;
      stage1_op_q <= op;
      stage2_a_q <= stage1_a_q;
      stage2_b_q <= stage1_b_q;
      stage2_op_q <= stage1_op_q;
    end
  end
endmodule
