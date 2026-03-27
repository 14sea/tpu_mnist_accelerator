// TPU Demo Top Level — AX301 (EP4CE6F17C8)
// 50MHz clock, UART 115200 baud, 4×4 systolic array

module tpu_top (
    input  wire       CLOCK,    // 50 MHz (PIN_E1)
    input  wire       RESET,    // active-low push button (PIN_E15)
    input  wire       RXD,      // UART receive (PIN_M2, from PL2303)
    output wire       TXD,      // UART transmit (PIN_G1, to PL2303)
    output wire [3:0] LED       // status LEDs (active-low on AX301)
);

    wire clk   = CLOCK;
    wire rst_n = RESET;

    // ── UART RX ──
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx u_uart_rx (
        .clk      (clk),
        .rst_n    (rst_n),
        .rx_pin   (RXD),
        .rx_data  (rx_data),
        .rx_valid (rx_valid)
    );

    // ── UART TX ──
    wire [7:0] tx_data;
    wire       tx_start;
    wire       tx_busy;

    uart_tx u_uart_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .tx_data  (tx_data),
        .tx_start (tx_start),
        .tx_pin   (TXD),
        .tx_busy  (tx_busy)
    );

    // ── Systolic Array ──
    wire        sa_en;
    wire        sa_load_weight;
    wire [1:0]  sa_w_row_sel;
    wire [1:0]  sa_w_col_sel;
    wire signed [7:0]  sa_w_data;
    wire [31:0]  sa_x_in;     // packed {x3, x2, x1, x0}
    wire [127:0] sa_result;   // packed {r3, r2, r1, r0}

    systolic_array_4x4 u_sa (
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (sa_en),
        .load_weight  (sa_load_weight),
        .w_row_sel    (sa_w_row_sel),
        .w_col_sel    (sa_w_col_sel),
        .w_data       (sa_w_data),
        .x_in         (sa_x_in),
        .result       (sa_result)
    );

    // ── Inference Engine ──
    wire [3:0] status_led;

    inference_engine u_engine (
        .clk            (clk),
        .rst_n          (rst_n),
        .rx_data        (rx_data),
        .rx_valid       (rx_valid),
        .tx_data        (tx_data),
        .tx_start       (tx_start),
        .tx_busy        (tx_busy),
        .sa_en          (sa_en),
        .sa_load_weight (sa_load_weight),
        .sa_w_row_sel   (sa_w_row_sel),
        .sa_w_col_sel   (sa_w_col_sel),
        .sa_w_data      (sa_w_data),
        .sa_x_in        (sa_x_in),
        .sa_result      (sa_result),
        .status_led     (status_led)
    );

    // AX301 LEDs are active-low
    assign LED = ~status_led;

endmodule
