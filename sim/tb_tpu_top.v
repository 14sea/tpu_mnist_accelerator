// Full system testbench for TPU demo
// Simulates UART communication: loads a tiny input, runs a 4→4 layer, reads result.

`timescale 1ns / 1ps

module tb_tpu_top;

    reg  clk, rst_n;
    reg  rxd;       // UART RX (to FPGA)
    wire txd;       // UART TX (from FPGA)
    wire [3:0] led;

    tpu_top uut (
        .CLOCK(clk), .RESET(rst_n),
        .RXD(rxd), .TXD(txd), .LED(led)
    );

    // 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    // UART timing: 115200 baud → ~8681 ns per bit
    localparam BIT_TIME = 8681;

    // ── UART TX helper (PC → FPGA) ──
    task uart_send_byte;
        input [7:0] data;
        integer b;
        begin
            rxd = 0;  // start bit
            #(BIT_TIME);
            for (b = 0; b < 8; b = b + 1) begin
                rxd = data[b];  // LSB first
                #(BIT_TIME);
            end
            rxd = 1;  // stop bit
            #(BIT_TIME);
        end
    endtask

    task uart_send_bytes;
        input [0:255] data;  // up to 32 bytes packed
        input integer len;
        integer idx;
        begin
            for (idx = 0; idx < len; idx = idx + 1)
                uart_send_byte(data[idx*8 +: 8]);
        end
    endtask

    // ── UART RX helper (FPGA → PC) ──
    reg [7:0] rx_byte;
    reg       rx_done;

    task uart_recv_byte;
        output [7:0] data;
        integer b;
        begin
            // Wait for start bit (falling edge)
            @(negedge txd);
            #(BIT_TIME / 2);  // center of start bit
            if (txd !== 0) begin
                $display("ERROR: invalid start bit");
                data = 8'hFF;
            end else begin
                #(BIT_TIME);  // skip to first data bit
                for (b = 0; b < 8; b = b + 1) begin
                    data[b] = txd;
                    #(BIT_TIME);
                end
                // stop bit
                if (txd !== 1)
                    $display("WARNING: stop bit not high");
            end
        end
    endtask

    // ── Test scenario ──
    // Small test: 4→4 layer with known weights
    // W = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity)
    // b = [100, 200, 300, 400]
    // x = [10, 20, 30, 40]
    // has_relu = 1, requant_mult = 65535 (≈1.0, so output ≈ acc)
    // acc = W@x + b = [110, 220, 330, 440]
    // requant: (acc * 65535) >> 16 ≈ acc (for small values)
    // ReLU + clamp [0,127]: [110, 127, 127, 127]
    //   (220 > 127, clamped)

    reg [7:0] resp;
    integer idx;

    initial begin
        $dumpfile("tb_tpu_top.vcd");
        $dumpvars(0, tb_tpu_top);

        rxd   = 1;  // UART idle
        rst_n = 0;
        #200;
        rst_n = 1;
        #200;

        $display("\n=== TPU Top Testbench ===\n");

        // ── 1. LOAD_INPUT: 4 bytes ──
        $display("[%0t] Sending LOAD_INPUT (4 bytes)...", $time);
        uart_send_byte(8'hAA);   // sync
        uart_send_byte(8'h55);
        uart_send_byte(8'h01);   // cmd = LOAD_INPUT
        uart_send_byte(8'h00);   // len_h = 0
        uart_send_byte(8'h04);   // len_l = 4
        uart_send_byte(8'd10);   // x[0] = 10
        uart_send_byte(8'd20);   // x[1] = 20
        uart_send_byte(8'd30);   // x[2] = 30
        uart_send_byte(8'd40);   // x[3] = 40

        // Wait for ACK
        uart_recv_byte(resp);
        $display("[%0t] LOAD_INPUT response: 0x%02x %s",
                 $time, resp, (resp == 8'h06) ? "(ACK)" : "(ERROR)");

        // ── 2. RUN_LAYER: 4→4, relu, identity weights ──
        $display("\n[%0t] Sending RUN_LAYER (4→4, relu)...", $time);
        uart_send_byte(8'hAA);   // sync
        uart_send_byte(8'h55);
        uart_send_byte(8'h02);   // cmd = RUN_LAYER
        uart_send_byte(8'h00);   // in_dim_h
        uart_send_byte(8'h04);   // in_dim_l = 4
        uart_send_byte(8'h00);   // out_dim_h
        uart_send_byte(8'h04);   // out_dim_l = 4
        uart_send_byte(8'h01);   // has_relu = 1
        uart_send_byte(8'hFF);   // requant_mult_h = 0xFF
        uart_send_byte(8'hFF);   // requant_mult_l = 0xFF → 65535

        // Bias for output group 0: [100, 200, 300, 400] as int32 LE
        // bias[0] = 100 = 0x00000064
        uart_send_byte(8'h64); uart_send_byte(8'h00);
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        // bias[1] = 200 = 0x000000C8
        uart_send_byte(8'hC8); uart_send_byte(8'h00);
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        // bias[2] = 300 = 0x0000012C
        uart_send_byte(8'h2C); uart_send_byte(8'h01);
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        // bias[3] = 400 = 0x00000190
        uart_send_byte(8'h90); uart_send_byte(8'h01);
        uart_send_byte(8'h00); uart_send_byte(8'h00);

        // Weight tile (1 tile since in_dim=4):
        // W[0][0..3] = [1, 0, 0, 0] (identity row 0)
        uart_send_byte(8'h01); uart_send_byte(8'h00);
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        // W[1][0..3] = [0, 1, 0, 0]
        uart_send_byte(8'h00); uart_send_byte(8'h01);
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        // W[2][0..3] = [0, 0, 1, 0]
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        uart_send_byte(8'h01); uart_send_byte(8'h00);
        // W[3][0..3] = [0, 0, 0, 1]
        uart_send_byte(8'h00); uart_send_byte(8'h00);
        uart_send_byte(8'h00); uart_send_byte(8'h01);

        // Wait for ACK
        uart_recv_byte(resp);
        $display("[%0t] RUN_LAYER response: 0x%02x %s",
                 $time, resp, (resp == 8'h06) ? "(ACK)" : "(ERROR)");

        // ── 3. GET_RESULT ──
        $display("\n[%0t] Sending GET_RESULT...", $time);
        uart_send_byte(8'hAA);
        uart_send_byte(8'h55);
        uart_send_byte(8'h03);

        // Read response: len_h, len_l, has_relu, data[0..len-1]
        uart_recv_byte(resp);
        $display("  len_h = %0d", resp);
        uart_recv_byte(resp);
        $display("  len_l = %0d", resp);
        uart_recv_byte(resp);
        $display("  has_relu = %0d", resp);

        // Read 4 output bytes (int8 since has_relu=1)
        for (idx = 0; idx < 4; idx = idx + 1) begin
            uart_recv_byte(resp);
            $display("  output[%0d] = %0d", idx, resp);
        end

        $display("\nExpected: output = [110, 127, 127, 127]");
        $display("  (acc=[110,220,330,440], requant≈1.0, clamp to [0,127])");

        $display("\n=== Testbench Done ===\n");
        #(BIT_TIME * 5);
        $finish;
    end

    // Timeout
    initial begin
        #(BIT_TIME * 500);
        $display("ERROR: Simulation timeout");
        $finish;
    end

endmodule
