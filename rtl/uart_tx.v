/*
 * Copyright (c) 2026
 *
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 */

// UART Transmitter - 115200 baud, 8N1, 50MHz clock

module uart_tx (
    input  wire       clk,       // 50 MHz
    input  wire       rst_n,
    input  wire [7:0] tx_data,   // byte to send
    input  wire       tx_start,  // pulse to begin transmission
    output reg        tx_pin,    // serial output
    output reg        tx_busy    // high while transmitting
);

    localparam CLKS_PER_BIT = 434;  // 50M / 115200

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0] state;
    reg [8:0] clk_cnt;
    reg [2:0] bit_idx;
    reg [7:0] shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            clk_cnt   <= 0;
            bit_idx   <= 0;
            shift_reg <= 0;
            tx_pin    <= 1'b1;  // idle high
            tx_busy   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx_pin <= 1'b1;
                    if (tx_start) begin
                        state     <= S_START;
                        shift_reg <= tx_data;
                        clk_cnt   <= 0;
                        tx_busy   <= 1'b1;
                    end
                end

                S_START: begin
                    tx_pin <= 1'b0;  // start bit
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        state   <= S_DATA;
                        clk_cnt <= 0;
                        bit_idx <= 0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                S_DATA: begin
                    tx_pin <= shift_reg[0];  // LSB first
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt   <= 0;
                        shift_reg <= {1'b0, shift_reg[7:1]};
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
                    tx_pin <= 1'b1;  // stop bit
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        state   <= S_IDLE;
                        clk_cnt <= 0;
                        tx_busy <= 1'b0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
            endcase
        end
    end

endmodule
