# Timing constraints for TPU Demo

# 50 MHz system clock
create_clock -name clk_50m -period 20.000 [get_ports {CLOCK}]

# UART is slow — no special constraints needed
set_false_path -from [get_ports {RXD}]
set_false_path -to   [get_ports {TXD}]
set_false_path -to   [get_ports {LED[*]}]
set_false_path -from [get_ports {RESET}]
