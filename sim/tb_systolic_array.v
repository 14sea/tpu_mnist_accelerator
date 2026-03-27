// Testbench for 4×4 systolic array (packed bus interface)
// Verifies tile MAC computation against known values.

`timescale 1ns / 1ps

module tb_systolic_array;

    reg clk, rst_n;
    reg en, load_weight;
    reg [1:0] w_row_sel, w_col_sel;
    reg signed [7:0] w_data;
    reg [31:0] x_in;        // packed {x3, x2, x1, x0}
    wire [127:0] result;    // packed {r3, r2, r1, r0}

    // Unpack results for display
    wire signed [31:0] res0 = result[ 31:  0];
    wire signed [31:0] res1 = result[ 63: 32];
    wire signed [31:0] res2 = result[ 95: 64];
    wire signed [31:0] res3 = result[127: 96];

    systolic_array_4x4 uut (
        .clk(clk), .rst_n(rst_n), .en(en),
        .load_weight(load_weight),
        .w_row_sel(w_row_sel), .w_col_sel(w_col_sel),
        .w_data(w_data),
        .x_in(x_in), .result(result)
    );

    initial clk = 0;
    always #10 clk = ~clk;

    integer i, j;
    integer errors;

    reg signed [7:0] W [0:3][0:3];
    reg signed [7:0] X [0:3];
    reg signed [31:0] expected [0:3];

    task load_all_weights;
        integer r, c;
        begin
            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 4; c = c + 1) begin
                    @(posedge clk);
                    load_weight <= 1;
                    w_row_sel   <= r[1:0];
                    w_col_sel   <= c[1:0];
                    w_data      <= W[r][c];
                end
            end
            @(posedge clk);
            load_weight <= 0;
        end
    endtask

    task run_compute;
        integer c;
        begin
            for (c = 0; c < 10; c = c + 1) begin
                @(posedge clk);
                en <= 1;
                x_in <= {
                    (c == 3) ? X[3] : 8'sd0,
                    (c == 2) ? X[2] : 8'sd0,
                    (c == 1) ? X[1] : 8'sd0,
                    (c == 0) ? X[0] : 8'sd0
                };
            end
            @(posedge clk);
            en <= 0;
            x_in <= 32'd0;
        end
    endtask

    task compute_expected;
        integer r, c;
        begin
            for (r = 0; r < 4; r = r + 1) begin
                expected[r] = 0;
                for (c = 0; c < 4; c = c + 1)
                    expected[r] = expected[r] + W[r][c] * X[c];
            end
        end
    endtask

    task check_results;
        input [8*20-1:0] test_name;
        integer r;
        reg signed [31:0] res [0:3];
        begin
            res[0] = res0; res[1] = res1; res[2] = res2; res[3] = res3;
            for (r = 0; r < 4; r = r + 1) begin
                if (res[r] !== expected[r]) begin
                    $display("FAIL %0s: result[%0d] = %0d, expected %0d",
                             test_name, r, res[r], expected[r]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        $dumpfile("tb_systolic_array.vcd");
        $dumpvars(0, tb_systolic_array);

        rst_n = 0; en = 0; load_weight = 0;
        w_row_sel = 0; w_col_sel = 0; w_data = 0; x_in = 0;
        errors = 0;

        repeat(3) @(posedge clk); rst_n = 1; repeat(2) @(posedge clk);

        // Test 1: Identity
        $display("\n=== Test 1: Identity weights ===");
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1)
            W[i][j] = (i == j) ? 8'sd1 : 8'sd0;
        X[0] = 10; X[1] = 20; X[2] = 30; X[3] = 40;
        compute_expected;
        load_all_weights; run_compute; repeat(3) @(posedge clk);
        check_results("Test1");
        $display("  result = [%0d, %0d, %0d, %0d]", res0, res1, res2, res3);

        // Test 2: All ones
        $display("\n=== Test 2: All-ones ===");
        rst_n = 0; repeat(2) @(posedge clk); rst_n = 1; repeat(2) @(posedge clk);
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) W[i][j] = 1;
        X[0] = 1; X[1] = 2; X[2] = 3; X[3] = 4;
        compute_expected;
        load_all_weights; run_compute; repeat(3) @(posedge clk);
        check_results("Test2");
        $display("  result = [%0d, %0d, %0d, %0d]", res0, res1, res2, res3);

        // Test 3: Signed
        $display("\n=== Test 3: Signed values ===");
        rst_n = 0; repeat(2) @(posedge clk); rst_n = 1; repeat(2) @(posedge clk);
        W[0][0]= 2; W[0][1]=-1; W[0][2]= 3; W[0][3]= 0;
        W[1][0]=-1; W[1][1]= 2; W[1][2]= 0; W[1][3]= 3;
        W[2][0]= 1; W[2][1]= 1; W[2][2]=-1; W[2][3]=-1;
        W[3][0]= 0; W[3][1]= 0; W[3][2]= 1; W[3][3]= 2;
        X[0] = 10; X[1] = -5; X[2] = 3; X[3] = 7;
        compute_expected;
        load_all_weights; run_compute; repeat(3) @(posedge clk);
        check_results("Test3");
        $display("  result = [%0d, %0d, %0d, %0d]", res0, res1, res2, res3);

        // Summary
        $display("\n========================================");
        if (errors == 0) $display("ALL TESTS PASSED");
        else $display("FAILED: %0d errors", errors);
        $display("========================================\n");
        #100; $finish;
    end

endmodule
