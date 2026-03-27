#!/usr/bin/env python3
#
# Copyright (c) 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
#

"""Debug: compare FPGA vs Python layer-by-layer for a single sample."""
import json, os, struct, sys, time
import numpy as np
import serial

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
SYNC = bytes([0xAA, 0x55])

def pad4(n):
    return ((n + 3) // 4) * 4

def compute_requant_mult(meta):
    return min(int(round(meta['requant_scale'] * (1 << 16))), 65535)

class DebugTPU:
    def __init__(self, port='/dev/ttyUSB0'):
        self.ser = serial.Serial(port, 115200, timeout=10)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(1.0)
        self.ser.reset_input_buffer()
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
            time.sleep(0.01)
        self.ser.reset_input_buffer()

        with open(os.path.join(WEIGHTS_DIR, 'model_meta.json')) as f:
            self.meta = json.load(f)
        self.layers = []
        for lm in self.meta['layers']:
            W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                            dtype=np.int8).reshape(lm['out_features'], lm['in_features'])
            b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']),
                            dtype=np.int32)
            self.layers.append((lm, W, b))

    def _send(self, data):
        self.ser.write(data)
        self.ser.flush()

    def _recv(self, n):
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _wait_ack(self):
        for attempt in range(16):
            resp = self._recv(1)
            if resp[0] == 0x06:
                return
            print(f"    [warn] expected ACK, got 0x{resp[0]:02x}")
        raise RuntimeError("ACK not received")

    def load_input(self, x_int8):
        data = x_int8.tobytes()
        length = len(data)
        cmd = SYNC + bytes([0x01, (length >> 8) & 0xFF, length & 0xFF])
        self._send(cmd + data)
        self._wait_ack()

    def run_layer(self, layer_idx):
        lm, W_raw, b_raw = self.layers[layer_idx]
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        has_relu = 1 if lm['activation'] == 'relu' else 0
        requant_mult = compute_requant_mult(lm)
        in_dim_pad = pad4(in_dim)
        out_dim_pad = pad4(out_dim)

        W = np.zeros((out_dim_pad, in_dim_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_raw
        b = np.zeros(out_dim_pad, dtype=np.int32)
        b[:out_dim] = b_raw

        header = SYNC + bytes([
            0x02,
            (in_dim_pad >> 8) & 0xFF, in_dim_pad & 0xFF,
            (out_dim_pad >> 8) & 0xFF, out_dim_pad & 0xFF,
            has_relu,
            (requant_mult >> 8) & 0xFF, requant_mult & 0xFF,
        ])
        self._send(header)

        n_out_groups = out_dim_pad // 4
        n_in_tiles = in_dim_pad // 4
        for g in range(n_out_groups):
            bias_bytes = b""
            for i in range(4):
                bias_bytes += struct.pack('<i', int(b[g * 4 + i]))
            self._send(bias_bytes)
            for t in range(n_in_tiles):
                tile = W[g*4:g*4+4, t*4:t*4+4]
                self._send(tile.tobytes())
        self._wait_ack()

    def get_result(self):
        self._send(SYNC + bytes([0x03]))
        resp = self._recv(3)
        length = (resp[0] << 8) | resp[1]
        has_relu = resp[2]
        data = self._recv(length)
        return has_relu, data

    def python_layer(self, x_int8, layer_idx):
        """Bit-accurate Python reference computation."""
        lm, W_raw, b_raw = self.layers[layer_idx]
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        has_relu = lm['activation'] == 'relu'
        requant_mult = compute_requant_mult(lm)
        in_dim_pad = pad4(in_dim)
        out_dim_pad = pad4(out_dim)

        W = np.zeros((out_dim_pad, in_dim_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_raw
        b = np.zeros(out_dim_pad, dtype=np.int32)
        b[:out_dim] = b_raw

        x = np.zeros(in_dim_pad, dtype=np.int8)
        x[:len(x_int8)] = x_int8

        # Tile-based MAC (matching FPGA tile iteration)
        acc = b.astype(np.int64).copy()  # start with bias
        n_out_groups = out_dim_pad // 4
        n_in_tiles = in_dim_pad // 4
        for g in range(n_out_groups):
            for t in range(n_in_tiles):
                tile_w = W[g*4:g*4+4, t*4:t*4+4].astype(np.int64)
                tile_x = x[t*4:t*4+4].astype(np.int64)
                for r in range(4):
                    for c in range(4):
                        acc[g*4 + r] += int(tile_w[r, c]) * int(tile_x[c])

        if has_relu:
            # Requantize
            out = np.zeros(out_dim_pad, dtype=np.int8)
            for i in range(out_dim_pad):
                val = int(acc[i]) * requant_mult
                val = val >> 16
                val = max(0, min(127, val))
                out[i] = np.int8(val)
            return out
        else:
            return acc[:out_dim_pad].astype(np.int32)


def main():
    samples = np.fromfile(os.path.join(WEIGHTS_DIR, 'test_samples.bin'),
                          dtype=np.int8).reshape(-1, 784)
    labels = np.fromfile(os.path.join(WEIGHTS_DIR, 'test_labels.bin'),
                         dtype=np.uint8)

    tpu = DebugTPU()
    sample_idx = 0
    label = labels[sample_idx]
    image = samples[sample_idx]
    x_float = np.clip(image.astype(np.float32) / 127.0, 0, 1)
    x_int8 = np.clip(np.round(x_float * 127), 0, 127).astype(np.int8)

    print(f"Sample {sample_idx}, label={label}")
    print(f"Input (first 20): {x_int8[:20]}")

    # Load input to FPGA
    in_pad = pad4(len(x_int8))
    x_pad = np.zeros(in_pad, dtype=np.int8)
    x_pad[:len(x_int8)] = x_int8
    print(f"\nLoading input ({len(x_pad)} bytes)...")
    tpu.load_input(x_pad)
    print("  ACK received")

    # Run each layer and compare
    py_x = x_pad.copy()
    for li in range(len(tpu.layers)):
        lm = tpu.layers[li][0]
        print(f"\n--- Layer {li}: {lm['in_features']}→{lm['out_features']} (relu={lm['activation']}) ---")

        # Python reference
        py_out = tpu.python_layer(py_x, li)
        print(f"  Python output (first 10): {py_out[:10]}")

        # FPGA
        print(f"  Running on FPGA...")
        tpu.run_layer(li)
        print(f"  ACK received")

        has_relu, raw = tpu.get_result()
        print(f"  GET_RESULT: {len(raw)} bytes, has_relu={has_relu}")

        if has_relu:
            fpga_out = np.frombuffer(raw, dtype=np.int8)
        else:
            fpga_out = np.frombuffer(raw, dtype=np.int32)

        fpga_out = fpga_out[:len(py_out)]
        print(f"  FPGA output (first 10):   {fpga_out[:10]}")

        # Compare
        match = np.array_equal(py_out[:lm['out_features']], fpga_out[:lm['out_features']])
        if match:
            print(f"  ✓ MATCH")
        else:
            diff = py_out[:lm['out_features']].astype(np.int64) - fpga_out[:lm['out_features']].astype(np.int64)
            print(f"  ✗ MISMATCH! Max diff: {np.max(np.abs(diff))}")
            # Show first few mismatches
            mismatches = np.where(diff != 0)[0]
            for idx in mismatches[:8]:
                print(f"    [{idx}] python={py_out[idx]}, fpga={fpga_out[idx]}")

        # Use Python output as next layer input (to isolate errors per layer)
        py_x = py_out

    # Final prediction
    if has_relu:
        scores = np.frombuffer(raw, dtype=np.int8)
    else:
        scores = np.frombuffer(raw, dtype=np.int32)
    out_dim = tpu.layers[-1][0]['out_features']
    scores = scores[:out_dim]
    pred = int(np.argmax(scores))
    print(f"\nFPGA prediction: {pred}, label: {label}")
    print(f"Python prediction: {int(np.argmax(py_out[:out_dim]))}")

    tpu.ser.close()

if __name__ == '__main__':
    main()
