#!/usr/bin/env python3
"""
Host-side CNN inference: im2col conv on host (INT8 numpy) + FC layers on FPGA (UART).

The im2col technique converts each conv layer into a matrix multiply:
  - For each output spatial position, gather its receptive-field patch as a row vector
  - Stack rows → patch matrix  (H_out×W_out, C×kH×kW)
  - Multiply by weight matrix  (N_filters, C×kH×kW)^T
  → equivalent to the convolution, computed with the existing 4×4 systolic array

Conv layers run entirely on the host (INT8 numpy, fast).
FC layers are sent to the FPGA via the same UART protocol as the MLP demo.

Usage:
  ../../.env/bin/python host_cnn_infer.py [--port /dev/ttyUSB0] [--samples 10]
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


# ─── Host-side INT8 conv via im2col ──────────────────────────────────────────

def im2col_single(x, kH, kW, stride=1, pad=0):
    """
    x: (C, H, W)  int8 or int32
    Returns: (H_out*W_out, C*kH*kW)  int32

    This is the KEY operation: each row of the result is one flattened receptive-field
    patch.  Multiplying this matrix by the weight matrix (F, C*kH*kW).T gives the same
    result as a conv2d — the entire convolution becomes a single matrix multiply.
    """
    C, H, W = x.shape
    H_out = (H - kH + 2 * pad) // stride + 1
    W_out = (W - kW + 2 * pad) // stride + 1

    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

    k       = np.arange(C * kH * kW)
    c_idx   = k // (kH * kW)
    kh_idx  = (k % (kH * kW)) // kW
    kw_idx  = k % kW

    h_s = np.arange(H_out) * stride
    w_s = np.arange(W_out) * stride

    cols = x[c_idx[:, None, None],
             h_s[None, :, None] + kh_idx[:, None, None],
             w_s[None, None, :] + kw_idx[:, None, None]]

    return cols.reshape(C * kH * kW, H_out * W_out).T.astype(np.int32)


def conv_host_int8(x_int8, lm, verbose=False):
    """
    INT8 convolution on host via im2col.
    x_int8: (C, H, W)
    Returns: (F, H_out, W_out) int8  post-ReLU + requantized
    """
    kH   = lm['kernel_h']
    kW   = lm['kernel_w']
    F    = lm['out_channels']
    K    = lm['patch_size']
    mult = lm['requant_mult']

    W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                    dtype=np.int8).reshape(F, K)      # (F, K)
    b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']),
                    dtype=np.int32)                   # (F,)

    # ── im2col: each row = one receptive-field patch ──────────────────────
    patches = im2col_single(x_int8.astype(np.int32), kH, kW)  # (P, K)
    P       = patches.shape[0]

    if verbose:
        print(f'    im2col: input {x_int8.shape} → patch matrix ({P}, {K})')
        print(f'    matmul: ({P},{K}) × ({K},{F}) + bias → ({P},{F}) accumulators')

    # ── matmul = the "convolution" ─────────────────────────────────────────
    acc = patches @ W.T.astype(np.int32) + b          # (P, F)  int32 accumulators

    # ── requantize → int8 ─────────────────────────────────────────────────
    out = np.clip((acc.astype(np.int64) * mult) >> 16, 0, 127).astype(np.int8)

    H_in, W_in = x_int8.shape[1], x_int8.shape[2]
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    return out.reshape(H_out, W_out, F).transpose(2, 0, 1)   # (F, H_out, W_out)


def maxpool_int8(x, size=2):
    C, H, W = x.shape
    H_out, W_out = H // size, W // size
    xr = x[:, :H_out * size, :W_out * size]
    return xr.reshape(C, H_out, size, W_out, size).max(axis=(2, 4)).astype(np.int8)


# ─── FPGA UART client (FC layers) ────────────────────────────────────────────

class FPGAClient:
    """
    Handles UART communication with the FPGA inference engine.
    Reuses the same CMD 0x01 / 0x02 / 0x03 protocol as the MLP demo.
    """

    def __init__(self, port, baud=115200, timeout=10):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Drain any stale bytes from JTAG programming (PL2303 buffer glitch)
        time.sleep(1.0)
        self.ser.reset_input_buffer()
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
            time.sleep(0.01)
        self.ser.reset_input_buffer()

    def _send(self, data):
        self.ser.write(data)
        self.ser.flush()

    def _recv(self, n):
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f'Expected {n} bytes, got {len(data)}')
        return data

    def _wait_ack(self, retries=8):
        for attempt in range(retries):
            resp = self._recv(1)
            if resp[0] == 0x06:
                return
            print(f'  [warn] expected ACK, got 0x{resp[0]:02x} (attempt {attempt+1}/{retries})')
        raise RuntimeError(f'ACK not received after {retries} attempts')

    def load_input(self, x_int8):
        """CMD 0x01 — load input vector into FPGA input_ram."""
        data   = x_int8.tobytes()
        length = len(data)
        self._send(SYNC + bytes([0x01, (length >> 8) & 0xFF, length & 0xFF]) + data)
        self._wait_ack()

    def run_fc_layer(self, lm, W_int8, b_int32):
        """
        CMD 0x02 — run one FC layer on the FPGA systolic array.
        lm:     layer metadata dict
        W_int8: (out_dim, in_dim) int8
        b_int32:(out_dim,) int32
        """
        in_dim   = lm['in_features']
        out_dim  = lm['out_features']
        has_relu = 1 if lm['activation'] == 'relu' else 0
        mult     = lm['requant_mult']

        in_pad  = pad4(in_dim)
        out_pad = pad4(out_dim)

        W = np.zeros((out_pad, in_pad), dtype=np.int8)
        W[:out_dim, :in_dim] = W_int8
        b = np.zeros(out_pad, dtype=np.int32)
        b[:out_dim] = b_int32

        header = SYNC + bytes([
            0x02,
            (in_pad  >> 8) & 0xFF, in_pad  & 0xFF,
            (out_pad >> 8) & 0xFF, out_pad & 0xFF,
            has_relu,
            (mult >> 8) & 0xFF, mult & 0xFF,
        ])
        self._send(header)

        n_out_groups = out_pad // 4
        n_in_tiles   = in_pad  // 4

        for g in range(n_out_groups):
            # Bias: 4 × int32 LE
            bias_bytes = b''
            for i in range(4):
                bias_bytes += struct.pack('<i', int(b[g * 4 + i]))
            self._send(bias_bytes)

            # Weight tiles: n_in_tiles × 16 bytes
            for t in range(n_in_tiles):
                tile = W[g*4:g*4+4, t*4:t*4+4]   # (4, 4) int8
                self._send(tile.tobytes())

        self._wait_ack()

    def get_result(self):
        """CMD 0x03 — read output buffer."""
        self._send(SYNC + bytes([0x03]))
        resp   = self._recv(3)      # len_h, len_l, has_relu
        length = (resp[0] << 8) | resp[1]
        has_relu = resp[2]
        data   = self._recv(length)
        return has_relu, data

    def close(self):
        self.ser.close()


# ─── Full CNN inference host ──────────────────────────────────────────────────

class CNNHost:
    def __init__(self, port, baud=115200):
        meta_path = os.path.join(WEIGHTS_DIR, 'cnn_meta.json')
        if not os.path.exists(meta_path):
            print(f'ERROR: {meta_path} not found. Run model/train_cnn.py first.')
            sys.exit(1)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.conv_layers = [l for l in self.meta['layers'] if l['type'] == 'conv']
        self.fc_layers   = [l for l in self.meta['layers'] if l['type'] == 'fc']
        self.conv_info   = self.meta['conv_info']

        # Pre-load FC weights
        self.fc_weights = []
        for lm in self.fc_layers:
            W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                            dtype=np.int8).reshape(lm['out_features'], lm['in_features'])
            b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']), dtype=np.int32)
            self.fc_weights.append((W, b))

        print(f'Connecting to FPGA on {port} @ {baud} baud...')
        self.fpga = FPGAClient(port, baud)
        print('  Connected (UART buffer drained).')

    def infer(self, image_int8_flat, verbose=False):
        """
        Full CNN inference for one image.
        image_int8_flat: (784,) int8 pixels [0..127]
        Returns: (predicted_digit, scores_int32)
        """
        x = image_int8_flat.reshape(1, 28, 28)    # (C=1, H=28, W=28)

        # ── Stage 1: Conv layers on HOST (INT8 im2col) ────────────────────
        for idx, lm in enumerate(self.conv_layers):
            ci = self.conv_info[idx]
            if verbose:
                print(f'  [HOST] Conv{idx}: im2col({ci["in_channels"]}ch '
                      f'{ci["in_h"]}×{ci["in_w"]}'
                      f', kernel {ci["kH"]}×{ci["kW"]}) → '
                      f'{ci["out_channels"]}×{ci["out_h"]}×{ci["out_w"]}')
            t0 = time.time()
            x = conv_host_int8(x, lm, verbose=verbose)    # (F, H_out, W_out)
            x = maxpool_int8(x, size=ci['pool'])
            if verbose:
                print(f'    → {x.shape}  ({time.time()-t0:.3f}s)')

        # ── Flatten to 1-D vector (input for first FC) ────────────────────
        x_flat = x.ravel().astype(np.int8)   # (200,) int8
        in_pad = pad4(len(x_flat))
        x_pad  = np.zeros(in_pad, dtype=np.int8)
        x_pad[:len(x_flat)] = x_flat

        if verbose:
            print(f'  Flatten → {len(x_flat)} values (padded to {in_pad})')

        # ── Stage 2: FC layers on FPGA (UART) ─────────────────────────────
        self.fpga.load_input(x_pad)

        for i, (lm, (W, b)) in enumerate(zip(self.fc_layers, self.fc_weights)):
            if verbose:
                print(f'  [FPGA] FC{i}: {lm["in_features"]}→{lm["out_features"]}'
                      f'  relu={lm["activation"]=="relu"}')
            t0 = time.time()
            self.fpga.run_fc_layer(lm, W, b)
            if verbose:
                print(f'    Done ({time.time()-t0:.2f}s)')

        # ── Read result ────────────────────────────────────────────────────
        has_relu, raw = self.fpga.get_result()
        if has_relu:
            scores = np.frombuffer(raw, dtype=np.int8)
        else:
            scores = np.frombuffer(raw, dtype=np.int32)

        n_out  = self.fc_layers[-1]['out_features']
        scores = scores[:n_out].astype(np.int32)
        return int(np.argmax(scores)), scores

    def close(self):
        self.fpga.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CNN Demo Host Inference')
    parser.add_argument('--port',    default='/dev/ttyUSB0')
    parser.add_argument('--baud',    type=int, default=115200)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    samples_path = os.path.join(WEIGHTS_DIR, 'test_samples.bin')
    labels_path  = os.path.join(WEIGHTS_DIR, 'test_labels.bin')
    if not os.path.exists(samples_path):
        print('ERROR: test_samples.bin not found. Run model/train_cnn.py first.')
        sys.exit(1)

    samples = np.fromfile(samples_path, dtype=np.int8).reshape(-1, 784)
    labels  = np.fromfile(labels_path,  dtype=np.uint8)
    n       = min(args.samples, len(labels))

    host = CNNHost(args.port, args.baud)

    print(f'\nInference pipeline:')
    print(f'  Conv(1→4,5×5)+Pool → Conv(4→8,3×3)+Pool   [host numpy INT8 im2col]')
    print(f'  FC(200→64)+ReLU → FC(64→10)                [FPGA 4×4 systolic array]')
    print(f'\nRunning {n} samples...\n')

    correct = 0
    t_start = time.time()

    for i in range(n):
        label = labels[i]
        print(f'Sample {i} (label={label}):')
        t0 = time.time()
        pred, scores = host.infer(samples[i], verbose=args.verbose)
        dt = time.time() - t0
        mark = '✓' if pred == label else '✗'
        print(f'  {mark} predicted={pred}  [{dt:.2f}s]  scores={scores}')
        if pred == label:
            correct += 1

    total = time.time() - t_start
    print(f'\n{"="*50}')
    print(f'Accuracy: {correct}/{n} = {correct/n*100:.1f}%')
    print(f'Total: {total:.1f}s  ({total/n:.2f}s/sample)')

    host.close()


if __name__ == '__main__':
    main()
