#!/usr/bin/env python3
"""Diagnostic: check what the FPGA is sending on UART."""
import serial, time, sys

port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
ser = serial.Serial(port, 115200, timeout=0.5)
ser.reset_input_buffer()

# 1) Listen for unsolicited data for 3 seconds
print("=== Phase 1: Listening for unsolicited data (3s) ===")
deadline = time.time() + 3.0
while time.time() < deadline:
    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        print(f"  Received {len(data)} bytes: {data.hex(' ')} | {data!r}")
    time.sleep(0.1)

# 2) Send sync only (AA 55) followed by invalid command (FF)
print("\n=== Phase 2: Send AA 55 FF (invalid cmd) ===")
ser.reset_input_buffer()
ser.write(bytes([0xAA, 0x55, 0xFF]))
ser.flush()
time.sleep(0.5)
if ser.in_waiting:
    data = ser.read(ser.in_waiting)
    print(f"  Response: {data.hex(' ')} | {data!r}")
else:
    print("  No response (good — FSM returned to IDLE)")

# 3) Send a tiny LOAD_INPUT: AA 55 01 00 04 + 4 bytes
print("\n=== Phase 3: Send LOAD_INPUT (4 bytes) ===")
ser.reset_input_buffer()
cmd = bytes([0xAA, 0x55, 0x01, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04])
ser.write(cmd)
ser.flush()
time.sleep(1.0)
if ser.in_waiting:
    data = ser.read(ser.in_waiting)
    print(f"  Response: {data.hex(' ')} | {data!r}")
else:
    print("  No response (TIMEOUT — FPGA did not ACK)")

# 4) Read anything else
print("\n=== Phase 4: Drain remaining (2s) ===")
deadline = time.time() + 2.0
while time.time() < deadline:
    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        print(f"  Late data: {data.hex(' ')} | {data!r}")
    time.sleep(0.1)

ser.close()
print("\nDone.")
