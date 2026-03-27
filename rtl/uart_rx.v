/*
 * Copyright (c) 2026
 *
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 */

// UART Receiver - 115200 baud, 8N1, 50MHz clock
// Oversamples at 16x baud rate, samples at midpoint of each bit.

module uart_rx (
    input  wire       clk,       // 50 MHz
    input  wire       rst_n,
    input  wire       rx_pin,    // serial input
    output reg  [7:0] rx_data,   // received byte
    output reg        rx_valid   // pulse high for 1 cycle when rx_data is valid
);

    // 50_000_000 / 115_200 = 434.03 clocks per bit
    localparam CLKS_PER_BIT = 434;
    localparam HALF_BIT     = 217;  // sample at midpoint

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [8:0]  clk_cnt;    // max 434
    reg [2:0]  bit_idx;    // 0..7
    reg [7:0]  shift_reg;
    reg [1:0]  rx_sync;    // 2-stage synchronizer

    wire rx_s = rx_sync[1];

    // Synchronize rx_pin to clk domain
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rx_sync <= 2'b11;
        else
            rx_sync <= {rx_sync[0], rx_pin};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            clk_cnt   <= 0;
            bit_idx   <= 0;
            shift_reg <= 0;
            rx_data   <= 0;
            rx_valid  <= 0;
        end else begin
            rx_valid <= 1'b0;  // default: no valid pulse

            case (state)
                S_IDLE: begin
                    if (rx_s == 1'b0) begin  // start bit detected
                        state   <= S_START;
                        clk_cnt <= 0;
                    end
                end

                S_START: begin
                    if (clk_cnt == HALF_BIT - 1) begin
                        if (rx_s == 1'b0) begin  // still low at midpoint → valid start
                            state   <= S_DATA;
                            clk_cnt <= 0;
                            bit_idx <= 0;
                        end else begin
                            state <= S_IDLE;  // false start
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        shift_reg <= {rx_s, shift_reg[7:1]};  // LSB first
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        rx_data  <= shift_reg;
                        rx_valid <= 1'b1;
                        state    <= S_IDLE;
                        clk_cnt  <= 0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
            endcase
        end
    end

endmodule
