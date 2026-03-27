// Single-port synchronous RAM — infers to M9K block RAM on Cyclone IV
// Read-during-write returns old data.

module sp_ram #(
    parameter ADDR_W = 10,
    parameter DATA_W = 8
)(
    input  wire                clk,
    input  wire                we,
    input  wire [ADDR_W-1:0]   addr,
    input  wire [DATA_W-1:0]   din,
    output reg  [DATA_W-1:0]   dout
);

    (* ramstyle = "M9K" *)
    reg [DATA_W-1:0] mem [0:(1<<ADDR_W)-1];

    always @(posedge clk) begin
        if (we)
            mem[addr] <= din;
        dout <= mem[addr];
    end

endmodule
