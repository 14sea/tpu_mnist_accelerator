#!/usr/bin/env python3
#
# Copyright (c) 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
#

"""
Host-side UART inference script for TPU Demo.
Sends MNIST images to FPGA, receives classification results.

Usage:
    python host_infer.py [--port /dev/ttyUSB0] [--baud 115200] [--samples 10]
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import serial

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
SYNC = bytes([0xAA, 0x55])


def pad4(n):
    return ((n + 3) // 4) * 4


def compute_requant_mult(meta):
    requant_scale = meta['requant_scale']
    return min(int(round(requant_scale * (1 << 16))), 65535)


class TPUHost:
    def __init__(self, port, baud=115200, timeout=10):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        # Wait for FPGA to settle, then drain any stale bytes accumulated
        # during JTAG bitstream programming (TXD pin glitches → PL2303 buffer)
        time.sleep(1.0)
        self.ser.reset_input_buffer()
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
            time.sleep(0.01)
        self.ser.reset_input_buffer()

        # Load model metadata and weights
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
        """Send raw bytes."""
        self.ser.write(data)
        self.ser.flush()

    def _recv(self, n):
        """Receive exactly n bytes."""
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _wait_ack(self, retries=8):
        """Wait for ACK byte (0x06), discarding any stale bytes first."""
        for attempt in range(retries):
            resp = self._recv(1)
            if resp[0] == 0x06:
                return
            # Stale byte — log and keep draining
            print(f"  [warn] expected ACK, got 0x{resp[0]:02x} (attempt {attempt+1}/{retries})")
        raise RuntimeError(f"ACK not received after {retries} attempts")

    def load_input(self, x_int8):
        """
        CMD 0x01: Send input vector to FPGA.
        x_int8: numpy int8 array (raw pixel values, 0-127)
        """
        data = x_int8.tobytes()
        length = len(data)
        cmd = SYNC + bytes([0x01, (length >> 8) & 0xFF, length & 0xFF])
        self._send(cmd + data)
        self._wait_ack()

    def run_layer(self, layer_idx):
        """
        CMD 0x02: Run one layer's computation on FPGA.
        Streams bias + weight tiles in the order the FPGA expects.
        """
        lm, W_raw, b_raw = self.layers[layer_idx]
        in_dim = lm['in_features']
        out_dim = lm['out_features']
        has_relu = 1 if lm['activation'] == 'relu' else 0
        requant_mult = compute_requant_mult(lm)

        in_dim_pad = pad4(in_dim)
        out_dim_pad = pad4(out_dim)

        # Pad weights and bias
        W = np.zeros((out_dim_pad, in_dim_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_raw
        b = np.zeros(out_dim_pad, dtype=np.int32)
        b[:out_dim] = b_raw

        # Send header
        header = SYNC + bytes([
            0x02,
            (in_dim_pad >> 8) & 0xFF, in_dim_pad & 0xFF,
            (out_dim_pad >> 8) & 0xFF, out_dim_pad & 0xFF,
            has_relu,
            (requant_mult >> 8) & 0xFF, requant_mult & 0xFF,
        ])
        self._send(header)

        # Send bias + weight data for each output group
        n_out_groups = out_dim_pad // 4
        n_in_tiles = in_dim_pad // 4

        for g in range(n_out_groups):
            # Bias: 4 × int32 LE = 16 bytes
            bias_bytes = b""
            for i in range(4):
                bias_bytes += struct.pack('<i', int(b[g * 4 + i]))
            self._send(bias_bytes)

            # Weight tiles: n_in_tiles × 16 bytes each
            for t in range(n_in_tiles):
                # 4 rows × 4 cols of int8
                tile = W[g*4:g*4+4, t*4:t*4+4]  # (4, 4) int8
                self._send(tile.tobytes())

        self._wait_ack()

    def get_result(self):
        """
        CMD 0x03: Read output buffer from FPGA.
        Returns: (has_relu, raw_bytes)
        """
        self._send(SYNC + bytes([0x03]))
        resp = self._recv(3)  # len_h, len_l, has_relu
        length = (resp[0] << 8) | resp[1]
        has_relu = resp[2]
        data = self._recv(length)
        return has_relu, data

    def infer(self, image_float, verbose=False):
        """
        Run full inference on one image.
        image_float: (784,) float32 in [0, 1]
        Returns: (predicted_digit, scores)
        """
        # Quantize input
        x_int8 = np.clip(np.round(image_float * 127), 0, 127).astype(np.int8)

        # Pad input to multiple of 4 (784 is already multiple of 4)
        in_dim_pad = pad4(len(x_int8))
        x_pad = np.zeros(in_dim_pad, dtype=np.int8)
        x_pad[:len(x_int8)] = x_int8

        if verbose:
            print(f"  Loading input ({len(x_pad)} bytes)...")
        self.load_input(x_pad)

        # Run each layer
        for i in range(len(self.layers)):
            lm = self.layers[i][0]
            if verbose:
                print(f"  Running layer {i}: {lm['in_features']}→{lm['out_features']}...")
            t0 = time.time()
            self.run_layer(i)
            dt = time.time() - t0
            if verbose:
                print(f"    Done in {dt:.2f}s")

        # Get result
        has_relu, raw = self.get_result()

        if has_relu:
            # int8 output
            scores = np.frombuffer(raw, dtype=np.int8)
        else:
            # int32 output
            scores = np.frombuffer(raw, dtype=np.int32)

        out_dim = self.layers[-1][0]['out_features']
        scores = scores[:out_dim]
        predicted = int(np.argmax(scores))

        return predicted, scores

    def close(self):
        self.ser.close()


def main():
    parser = argparse.ArgumentParser(description='TPU Demo Host Inference')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples')
    args = parser.parse_args()

    # Load test data
    samples = np.fromfile(os.path.join(WEIGHTS_DIR, 'test_samples.bin'),
                          dtype=np.int8).reshape(-1, 784)
    labels = np.fromfile(os.path.join(WEIGHTS_DIR, 'test_labels.bin'),
                         dtype=np.uint8)

    n = min(args.samples, len(labels))

    print(f"Connecting to FPGA on {args.port} @ {args.baud} baud...")
    tpu = TPUHost(args.port, args.baud)

    correct = 0
    t_total = time.time()

    for i in range(n):
        image_float = np.clip(samples[i].astype(np.float32) / 127.0, 0, 1)
        print(f"\nSample {i} (label={labels[i]}):")
        t0 = time.time()
        pred, scores = tpu.infer(image_float, verbose=True)
        dt = time.time() - t0
        match = "✓" if pred == labels[i] else "✗"
        print(f"  {match} Predicted: {pred} (label: {labels[i]}) [{dt:.2f}s total]")
        if pred == labels[i]:
            correct += 1

    dt_total = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{n} = {correct/n*100:.1f}%")
    print(f"Total time: {dt_total:.1f}s ({dt_total/n:.1f}s per sample)")
    tpu.close()


if __name__ == '__main__':
    main()
