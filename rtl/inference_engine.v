/*
 * Copyright (c) 2026
 *
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 */

// Inference Engine — orchestrates layer-by-layer MLP computation
//
// Uses M9K block RAMs for input/output buffers (not registers) to fit in EP4CE6.
// All RAM accesses are single-byte, sequential, with 1-cycle read latency.
//
// UART Protocol: see tpu_top.v or host_infer.py for details.

module inference_engine (
    input  wire        clk,
    input  wire        rst_n,
    // UART RX
    input  wire [7:0]  rx_data,
    input  wire        rx_valid,
    // UART TX
    output reg  [7:0]  tx_data,
    output reg         tx_start,
    input  wire        tx_busy,
    // Systolic array
    output reg         sa_en,
    output reg         sa_load_weight,
    output reg  [1:0]  sa_w_row_sel,
    output reg  [1:0]  sa_w_col_sel,
    output reg  signed [7:0]  sa_w_data,
    output reg  [31:0] sa_x_in,        // packed {x3, x2, x1, x0}
    input  wire [127:0] sa_result,      // packed {r3, r2, r1, r0}
    // Status LEDs
    output reg  [3:0]  status_led
);

    // ─── Unpack sa_result ───
    wire signed [31:0] sa_res0 = sa_result[ 31:  0];
    wire signed [31:0] sa_res1 = sa_result[ 63: 32];
    wire signed [31:0] sa_res2 = sa_result[ 95: 64];
    wire signed [31:0] sa_res3 = sa_result[127: 96];

    // ─── Block RAM instances ───
    // input_ram: 1024 × 8-bit (stores input activations, also receives next-layer output)
    reg         in_ram_we;
    reg  [9:0]  in_ram_addr;
    reg  [7:0]  in_ram_din;
    wire [7:0]  in_ram_dout;

    sp_ram #(.ADDR_W(10), .DATA_W(8)) input_ram (
        .clk(clk), .we(in_ram_we), .addr(in_ram_addr),
        .din(in_ram_din), .dout(in_ram_dout)
    );

    // output_ram: 512 × 8-bit (stores layer output for GET_RESULT)
    reg         out_ram_we;
    reg  [8:0]  out_ram_addr;
    reg  [7:0]  out_ram_din;
    wire [7:0]  out_ram_dout;

    sp_ram #(.ADDR_W(9), .DATA_W(8)) output_ram (
        .clk(clk), .we(out_ram_we), .addr(out_ram_addr),
        .din(out_ram_din), .dout(out_ram_dout)
    );

    // ─── State encoding ───
    localparam [4:0]
        S_IDLE          = 5'd0,
        S_SYNC1         = 5'd1,
        S_CMD           = 5'd2,
        S_LI_LEN_H     = 5'd3,
        S_LI_LEN_L     = 5'd4,
        S_LI_DATA      = 5'd5,
        S_RL_IN_DIM_H  = 5'd6,
        S_RL_IN_DIM_L  = 5'd7,
        S_RL_OUT_DIM_H = 5'd8,
        S_RL_OUT_DIM_L = 5'd9,
        S_RL_RELU      = 5'd10,
        S_RL_MULT_H    = 5'd11,
        S_RL_MULT_L    = 5'd12,
        S_RL_BIAS      = 5'd13,
        S_RL_BIAS_INIT = 5'd14,
        S_RL_WEIGHT    = 5'd15,
        S_RL_LOAD_W    = 5'd16,
        S_RL_READ_X    = 5'd17,  // pre-read input values from BRAM
        S_RL_COMPUTE   = 5'd18,
        S_RL_STORE     = 5'd19,  // write requant results to RAMs
        S_RL_COPY_BACK = 5'd20, // copy output_ram → input_ram for next layer
        S_RL_SEND_ACK  = 5'd21,
        S_GR_LEN_H     = 5'd22,
        S_GR_LEN_L     = 5'd23,
        S_GR_RELU      = 5'd24,
        S_GR_DATA      = 5'd25,
        S_GR_WAIT      = 5'd26;  // wait for RAM read latency

    reg [4:0] state;

    // ─── Layer parameters ───
    reg [9:0]  in_dim;
    reg [9:0]  out_dim;
    reg        has_relu_reg;
    reg [15:0] requant_mult;

    // ─── Counters ───
    reg [9:0]  byte_cnt;
    reg [9:0]  input_len;
    reg [9:0]  out_group;       // current output group base (0, 4, 8, ...)
    reg [9:0]  in_tile;         // current input tile (0, 1, 2, ...)
    reg [3:0]  bias_bcnt;       // 0..15
    reg [3:0]  w_load_cnt;      // 0..15
    reg [3:0]  compute_cnt;     // 0..9
    reg [2:0]  read_x_cnt;      // 0..4 for pre-reading inputs
    reg [3:0]  store_cnt;       // 0..7 for storing results (4 int8 or 16 int32)
    reg [9:0]  output_buf_len;
    reg [9:0]  copy_cnt;          // for S_RL_COPY_BACK

    // ─── Small register buffers ───
    reg signed [7:0]  weight_tile [0:15];   // 16 bytes (registers OK)
    reg signed [31:0] bias_buf [0:3];       // 4 × int32 (registers OK)
    reg signed [31:0] acc [0:3];            // 4 × int32 accumulators
    reg signed [7:0]  x_reg [0:3];          // 4 pre-read input values

    // ─── Requantization (combinational) ───
    wire signed [47:0] rq_prod_0 = acc[0] * $signed({1'b0, requant_mult});
    wire signed [47:0] rq_prod_1 = acc[1] * $signed({1'b0, requant_mult});
    wire signed [47:0] rq_prod_2 = acc[2] * $signed({1'b0, requant_mult});
    wire signed [47:0] rq_prod_3 = acc[3] * $signed({1'b0, requant_mult});
    wire signed [31:0] rq_val_0 = rq_prod_0[47:16];
    wire signed [31:0] rq_val_1 = rq_prod_1[47:16];
    wire signed [31:0] rq_val_2 = rq_prod_2[47:16];
    wire signed [31:0] rq_val_3 = rq_prod_3[47:16];
    wire [7:0] rq_clamp_0 = (rq_val_0 < 0) ? 8'd0 : (rq_val_0 > 127) ? 8'd127 : rq_val_0[7:0];
    wire [7:0] rq_clamp_1 = (rq_val_1 < 0) ? 8'd0 : (rq_val_1 > 127) ? 8'd127 : rq_val_1[7:0];
    wire [7:0] rq_clamp_2 = (rq_val_2 < 0) ? 8'd0 : (rq_val_2 > 127) ? 8'd127 : rq_val_2[7:0];
    wire [7:0] rq_clamp_3 = (rq_val_3 < 0) ? 8'd0 : (rq_val_3 > 127) ? 8'd127 : rq_val_3[7:0];

    // Store data register (int8 or int32 bytes, prepared for sequential write)
    reg [7:0] store_data [0:15]; // up to 16 bytes per output group (4 int32)
    reg [3:0] store_len;          // number of bytes to write
    reg       store_to_input;     // also write to input_ram?

    // TX helpers
    reg [9:0] tx_idx;

    wire [9:0] in_base = {in_tile, 2'b00};  // in_tile * 4

    // ─── Main FSM ───
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            sa_en           <= 1'b0;
            sa_load_weight  <= 1'b0;
            sa_w_row_sel    <= 2'd0;
            sa_w_col_sel    <= 2'd0;
            sa_w_data       <= 8'sd0;
            sa_x_in         <= 32'd0;
            tx_data         <= 8'd0;
            tx_start        <= 1'b0;
            status_led      <= 4'd0;
            in_ram_we       <= 1'b0;
            in_ram_addr     <= 10'd0;
            in_ram_din      <= 8'd0;
            out_ram_we      <= 1'b0;
            out_ram_addr    <= 9'd0;
            out_ram_din     <= 8'd0;
            in_dim          <= 10'd0;
            out_dim         <= 10'd0;
            has_relu_reg    <= 1'b0;
            requant_mult    <= 16'd0;
            byte_cnt        <= 10'd0;
            input_len       <= 10'd0;
            out_group       <= 10'd0;
            in_tile         <= 10'd0;
            bias_bcnt       <= 4'd0;
            w_load_cnt      <= 4'd0;
            compute_cnt     <= 4'd0;
            read_x_cnt      <= 3'd0;
            store_cnt       <= 4'd0;
            store_len       <= 4'd0;
            store_to_input  <= 1'b0;
            output_buf_len  <= 10'd0;
            copy_cnt        <= 10'd0;
            tx_idx          <= 10'd0;
            for (i = 0; i < 4; i = i + 1) begin
                acc[i]      <= 32'sd0;
                bias_buf[i] <= 32'sd0;
                x_reg[i]    <= 8'sd0;
            end
        end else begin
            // Defaults
            tx_start       <= 1'b0;
            sa_load_weight <= 1'b0;
            sa_en          <= 1'b0;
            in_ram_we      <= 1'b0;
            out_ram_we     <= 1'b0;

            case (state)

            // ═══════════ IDLE / SYNC / CMD ═══════════
            S_IDLE: begin
                status_led <= 4'b0001;
                if (rx_valid && rx_data == 8'hAA)
                    state <= S_SYNC1;
            end

            S_SYNC1: begin
                if (rx_valid)
                    state <= (rx_data == 8'h55) ? S_CMD : S_IDLE;
            end

            S_CMD: begin
                if (rx_valid) begin
                    case (rx_data)
                        8'h01:   state <= S_LI_LEN_H;
                        8'h02:   state <= S_RL_IN_DIM_H;
                        8'h03:   state <= S_GR_LEN_H;
                        default: state <= S_IDLE;
                    endcase
                end
            end

            // ═══════════ LOAD_INPUT ═══════════
            S_LI_LEN_H: begin
                if (rx_valid) begin
                    input_len[9:8] <= rx_data[1:0];
                    state <= S_LI_LEN_L;
                end
            end

            S_LI_LEN_L: begin
                if (rx_valid) begin
                    input_len[7:0] <= rx_data;
                    byte_cnt <= 10'd0;
                    state    <= S_LI_DATA;
                end
            end

            S_LI_DATA: begin
                status_led <= 4'b0010;
                if (rx_valid) begin
                    in_ram_we   <= 1'b1;
                    in_ram_addr <= byte_cnt;
                    in_ram_din  <= rx_data;
                    if (byte_cnt == input_len - 10'd1) begin
                        tx_data  <= 8'h06;
                        tx_start <= 1'b1;
                        state    <= S_IDLE;
                    end else begin
                        byte_cnt <= byte_cnt + 10'd1;
                    end
                end
            end

            // ═══════════ RUN_LAYER — header ═══════════
            S_RL_IN_DIM_H: begin
                if (rx_valid) begin in_dim[9:8] <= rx_data[1:0]; state <= S_RL_IN_DIM_L; end
            end
            S_RL_IN_DIM_L: begin
                if (rx_valid) begin in_dim[7:0] <= rx_data; state <= S_RL_OUT_DIM_H; end
            end
            S_RL_OUT_DIM_H: begin
                if (rx_valid) begin out_dim[9:8] <= rx_data[1:0]; state <= S_RL_OUT_DIM_L; end
            end
            S_RL_OUT_DIM_L: begin
                if (rx_valid) begin out_dim[7:0] <= rx_data; state <= S_RL_RELU; end
            end
            S_RL_RELU: begin
                if (rx_valid) begin has_relu_reg <= rx_data[0]; state <= S_RL_MULT_H; end
            end
            S_RL_MULT_H: begin
                if (rx_valid) begin requant_mult[15:8] <= rx_data; state <= S_RL_MULT_L; end
            end
            S_RL_MULT_L: begin
                if (rx_valid) begin
                    requant_mult[7:0] <= rx_data;
                    out_group  <= 10'd0;
                    bias_bcnt  <= 4'd0;
                    state      <= S_RL_BIAS;
                    status_led <= 4'b0100;
                end
            end

            // ═══════════ RUN_LAYER — per output group ═══════════

            // ── Receive 16 bias bytes (4 × int32 LE) ──
            S_RL_BIAS: begin
                if (rx_valid) begin
                    case (bias_bcnt[1:0])
                        2'd0: bias_buf[bias_bcnt[3:2]][ 7: 0] <= rx_data;
                        2'd1: bias_buf[bias_bcnt[3:2]][15: 8] <= rx_data;
                        2'd2: bias_buf[bias_bcnt[3:2]][23:16] <= rx_data;
                        2'd3: bias_buf[bias_bcnt[3:2]][31:24] <= rx_data;
                    endcase
                    if (bias_bcnt == 4'd15)
                        state <= S_RL_BIAS_INIT;
                    else
                        bias_bcnt <= bias_bcnt + 4'd1;
                end
            end

            // ── Initialize accumulators with bias ──
            S_RL_BIAS_INIT: begin
                for (i = 0; i < 4; i = i + 1)
                    acc[i] <= bias_buf[i];
                in_tile  <= 10'd0;
                byte_cnt <= 10'd0;
                state    <= S_RL_WEIGHT;
            end

            // ── Receive weight tile (16 bytes) ──
            S_RL_WEIGHT: begin
                if (rx_valid) begin
                    weight_tile[byte_cnt[3:0]] <= rx_data;
                    if (byte_cnt[3:0] == 4'd15) begin
                        w_load_cnt <= 4'd0;
                        state      <= S_RL_LOAD_W;
                    end else begin
                        byte_cnt <= byte_cnt + 10'd1;
                    end
                end
            end

            // ── Load 16 weights into systolic array (16 cycles) ──
            S_RL_LOAD_W: begin
                sa_load_weight <= 1'b1;
                sa_w_row_sel   <= w_load_cnt[3:2];
                sa_w_col_sel   <= w_load_cnt[1:0];
                sa_w_data      <= weight_tile[w_load_cnt];
                if (w_load_cnt == 4'd15) begin
                    read_x_cnt <= 3'd0;
                    // Issue first read address
                    in_ram_addr <= in_base;
                    state       <= S_RL_READ_X;
                end else begin
                    w_load_cnt <= w_load_cnt + 4'd1;
                end
            end

            // ── Pre-read 4 input values from BRAM (5 cycles) ──
            // Cycle 0: addr=base+0 issued in LOAD_W, data arrives now (wait 1 more)
            // Cycle 1: capture data for x[0], issue addr=base+1
            // Cycle 2: capture data for x[1], issue addr=base+2
            // Cycle 3: capture data for x[2], issue addr=base+3
            // Cycle 4: capture data for x[3], done
            // Pre-read 4 input values from BRAM.
            // sp_ram has 1-cycle read latency: addr set at posedge N → dout valid at posedge N+1.
            // LOAD_W last cycle set addr = in_base, so dout = mem[in_base] at cnt=0.
            // We must advance addr each cycle so dout stays pipelined.
            S_RL_READ_X: begin
                case (read_x_cnt)
                    3'd0: begin
                        // dout = mem[in_base] (addr set in LOAD_W). Don't re-issue — advance.
                        in_ram_addr <= in_base + 10'd1;
                        read_x_cnt  <= 3'd1;
                    end
                    3'd1: begin
                        x_reg[0]    <= in_ram_dout;  // mem[in_base]
                        in_ram_addr <= in_base + 10'd2;
                        read_x_cnt  <= 3'd2;
                    end
                    3'd2: begin
                        x_reg[1]    <= in_ram_dout;  // mem[in_base+1]
                        in_ram_addr <= in_base + 10'd3;
                        read_x_cnt  <= 3'd3;
                    end
                    3'd3: begin
                        x_reg[2]    <= in_ram_dout;  // mem[in_base+2]
                        read_x_cnt  <= 3'd4;
                    end
                    3'd4: begin
                        x_reg[3]    <= in_ram_dout;  // mem[in_base+3]
                        compute_cnt <= 4'd0;
                        state       <= S_RL_COMPUTE;
                    end
                    default: state <= S_IDLE;
                endcase
            end

            // ── Systolic computation (10 cycles) ──
            S_RL_COMPUTE: begin
                sa_en <= (compute_cnt <= 4'd7);

                sa_x_in[ 7: 0] <= (compute_cnt == 4'd0) ? x_reg[0] : 8'sd0;
                sa_x_in[15: 8] <= (compute_cnt == 4'd1) ? x_reg[1] : 8'sd0;
                sa_x_in[23:16] <= (compute_cnt == 4'd2) ? x_reg[2] : 8'sd0;
                sa_x_in[31:24] <= (compute_cnt == 4'd3) ? x_reg[3] : 8'sd0;

                // Continuous accumulation
                acc[0] <= acc[0] + sa_res0;
                acc[1] <= acc[1] + sa_res1;
                acc[2] <= acc[2] + sa_res2;
                acc[3] <= acc[3] + sa_res3;

                if (compute_cnt == 4'd9) begin
                    in_tile <= in_tile + 10'd1;
                    if ({in_tile + 10'd1, 2'b00} >= in_dim) begin
                        // All input tiles done → prepare store
                        store_cnt <= 4'd0;
                        state     <= S_RL_STORE;
                        // Prepare store data (combinational from rq_ wires next cycle)
                    end else begin
                        byte_cnt <= 10'd0;
                        state    <= S_RL_WEIGHT;
                    end
                end else begin
                    compute_cnt <= compute_cnt + 4'd1;
                end
            end

            // ── Store results to output_ram only (no input_ram writes during layer) ──
            // For has_relu: write 4 int8 bytes to output_ram
            // For !has_relu: write 16 int32 LE bytes to output_ram
            // Input_ram is updated in S_RL_COPY_BACK after all groups are done.
            S_RL_STORE: begin
                if (has_relu_reg) begin
                    // int8 mode: 4 writes to output_ram
                    out_ram_we   <= 1'b1;
                    out_ram_addr <= out_group[8:0] + store_cnt[1:0];
                    case (store_cnt[1:0])
                        2'd0: out_ram_din <= rq_clamp_0;
                        2'd1: out_ram_din <= rq_clamp_1;
                        2'd2: out_ram_din <= rq_clamp_2;
                        2'd3: out_ram_din <= rq_clamp_3;
                    endcase

                    if (store_cnt == 4'd3) begin
                        output_buf_len <= out_dim;
                        out_group <= out_group + 10'd4;
                        if (out_group + 10'd4 >= out_dim) begin
                            status_led <= 4'b1000;
                            copy_cnt   <= 10'd0;
                            state      <= S_RL_COPY_BACK;
                            // Don't override out_ram_addr here — let the
                            // last store write complete at the correct address.
                            // S_RL_COPY_BACK will issue addr 0 in its first cycle.
                        end else begin
                            bias_bcnt <= 4'd0;
                            state     <= S_RL_BIAS;
                        end
                    end else begin
                        store_cnt <= store_cnt + 4'd1;
                    end
                end else begin
                    // int32 mode: 16 bytes to output_ram only
                    out_ram_we   <= 1'b1;
                    out_ram_addr <= {out_group[6:0], 2'b00} + store_cnt[3:0];
                    case (store_cnt)
                        4'd0:  out_ram_din <= acc[0][ 7: 0];
                        4'd1:  out_ram_din <= acc[0][15: 8];
                        4'd2:  out_ram_din <= acc[0][23:16];
                        4'd3:  out_ram_din <= acc[0][31:24];
                        4'd4:  out_ram_din <= acc[1][ 7: 0];
                        4'd5:  out_ram_din <= acc[1][15: 8];
                        4'd6:  out_ram_din <= acc[1][23:16];
                        4'd7:  out_ram_din <= acc[1][31:24];
                        4'd8:  out_ram_din <= acc[2][ 7: 0];
                        4'd9:  out_ram_din <= acc[2][15: 8];
                        4'd10: out_ram_din <= acc[2][23:16];
                        4'd11: out_ram_din <= acc[2][31:24];
                        4'd12: out_ram_din <= acc[3][ 7: 0];
                        4'd13: out_ram_din <= acc[3][15: 8];
                        4'd14: out_ram_din <= acc[3][23:16];
                        4'd15: out_ram_din <= acc[3][31:24];
                    endcase

                    if (store_cnt == 4'd15) begin
                        output_buf_len <= {out_dim, 2'b00};  // out_dim * 4
                        out_group <= out_group + 10'd4;
                        if (out_group + 10'd4 >= out_dim) begin
                            status_led <= 4'b1000;
                            state      <= S_RL_SEND_ACK;
                        end else begin
                            bias_bcnt <= 4'd0;
                            state     <= S_RL_BIAS;
                        end
                    end else begin
                        store_cnt <= store_cnt + 4'd1;
                    end
                end
            end

            // ── Copy output_ram → input_ram (for relu layers, prepares next layer) ──
            // Pipeline: addr issued 1 cycle before capture.
            // copy_cnt=0: out_ram_addr was set in S_RL_STORE transition; wait for dout.
            // copy_cnt=1..N: capture dout, write to in_ram, advance addr.
            // copy_cnt=N+1: capture last byte, write, done.
            // Copy output_ram → input_ram for relu layers (prepares next layer input).
            // sp_ram read latency: addr at posedge N → dout valid at posedge N+1.
            // copy_cnt=0: issue addr=0
            // copy_cnt=1: dout has stale data, issue addr=1 (pipeline fill)
            // copy_cnt=2: dout=mem[0], write in_ram[0], issue addr=2
            // copy_cnt=K: dout=mem[K-2], write in_ram[K-2], issue addr=K
            // copy_cnt=len+1: dout=mem[len-1], write in_ram[len-1], done
            S_RL_COPY_BACK: begin
                if (copy_cnt == 10'd0) begin
                    out_ram_addr <= 9'd0;
                    copy_cnt     <= 10'd1;
                end else if (copy_cnt == 10'd1) begin
                    // Pipeline fill — dout not yet valid for addr 0
                    out_ram_addr <= 9'd1;
                    copy_cnt     <= 10'd2;
                end else if (copy_cnt <= output_buf_len + 10'd1) begin
                    // dout = mem[copy_cnt-2]. Write to input_ram.
                    in_ram_we   <= 1'b1;
                    in_ram_addr <= copy_cnt - 10'd2;
                    in_ram_din  <= out_ram_dout;
                    if (copy_cnt <= output_buf_len)
                        out_ram_addr <= copy_cnt[8:0];
                    copy_cnt <= copy_cnt + 10'd1;
                end else begin
                    state <= S_RL_SEND_ACK;
                end
            end

            S_RL_SEND_ACK: begin
                if (!tx_busy) begin
                    tx_data  <= 8'h06;
                    tx_start <= 1'b1;
                    state    <= S_IDLE;
                end
            end

            // ═══════════ GET_RESULT ═══════════
            S_GR_LEN_H: begin
                if (!tx_busy) begin
                    tx_data  <= {6'b0, output_buf_len[9:8]};
                    tx_start <= 1'b1;
                    state    <= S_GR_LEN_L;
                end
            end

            S_GR_LEN_L: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= output_buf_len[7:0];
                    tx_start <= 1'b1;
                    state    <= S_GR_RELU;
                end
            end

            S_GR_RELU: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= {7'b0, has_relu_reg};
                    tx_start <= 1'b1;
                    tx_idx   <= 10'd0;
                    // Issue first read
                    out_ram_addr <= 9'd0;
                    state <= (output_buf_len == 0) ? S_IDLE : S_GR_WAIT;
                end
            end

            // Wait 1 cycle for RAM read latency
            S_GR_WAIT: begin
                state <= S_GR_DATA;
            end

            S_GR_DATA: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= out_ram_dout;
                    tx_start <= 1'b1;
                    if (tx_idx == output_buf_len - 10'd1) begin
                        state <= S_IDLE;
                    end else begin
                        tx_idx       <= tx_idx + 10'd1;
                        out_ram_addr <= tx_idx[8:0] + 9'd1;
                        // Data will be ready 1 cycle later, but tx_busy
                        // takes many cycles (UART bit time), so data is ready
                        // well before the next non-busy cycle.
                    end
                end
            end

            endcase
        end
    end

endmodule
