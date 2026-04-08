module GEN_ALU (
  input signed [31:0] a,
  input signed [31:0] b,
  input [4:0] op,
  output reg signed [31:0] y
);
  always @(*) begin
    case (op)
      5'd0:  y = a + b;
      5'd1:  y = a & b;
      5'd2:  y = (a == b) ? 32'sd1 : 32'sd0;
      5'd3:  y = (a >= b) ? 32'sd1 : 32'sd0;
      5'd4:  y = (a > b)  ? 32'sd1 : 32'sd0;
      5'd5:  y = (a <= b) ? 32'sd1 : 32'sd0;
      5'd6:  y = (a < b)  ? 32'sd1 : 32'sd0;
      5'd7:  y = a;
      5'd8:  y = (a != b) ? 32'sd1 : 32'sd0;
      5'd9:  y = -a;
      5'd10: y = ~a;
      5'd11: y = a | b;
      5'd12: y = a <<< b[4:0];
      5'd13: y = a >>> b[4:0];
      5'd14: y = a - b;
      5'd15: y = a ^ b;
      default: y = 32'sd0;
    endcase
  end
endmodule
