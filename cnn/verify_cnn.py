#!/usr/bin/env python3
"""
Bit-accurate Python simulation of the full CNN inference pipeline.

Conv layers:  INT8 im2col + matmul (host side, matches what host_cnn_infer.py does)
FC layers:    INT8 tile-based MAC   (matches FPGA inference_engine.v exactly)

Usage:
  ../../.env/bin/python verify_cnn.py
"""

import json
import os
import sys
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


# ─── Utilities ────────────────────────────────────────────────────────────────

def pad4(n):
    return ((n + 3) // 4) * 4


def load_meta():
    with open(os.path.join(WEIGHTS_DIR, 'cnn_meta.json')) as f:
        return json.load(f)


# ─── Host-side INT8 conv via im2col ──────────────────────────────────────────

def im2col_single(x, kH, kW, stride=1, pad=0):
    """
    x: (C, H, W)  — single image, already int8/int32
    Returns: (H_out*W_out, C*kH*kW)
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

    # (C*kH*kW, H_out, W_out)
    cols = x[c_idx[:, None, None],
             h_s[None, :, None] + kh_idx[:, None, None],
             w_s[None, None, :] + kw_idx[:, None, None]]

    return cols.reshape(C * kH * kW, H_out * W_out).T   # (H_out*W_out, C*kH*kW)


def conv_int8(x_int8, lm):
    """
    x_int8: (C, H, W)  int8 input (single image)
    Returns: (F, H_out, W_out)  int8 — post-ReLU + requantized
    """
    kH, kW      = lm['kernel_h'], lm['kernel_w']
    F           = lm['out_channels']
    patch_size  = lm['patch_size']

    W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                    dtype=np.int8).reshape(F, patch_size)        # (F, K)
    b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']),
                    dtype=np.int32)                               # (F,)

    cols = im2col_single(x_int8.astype(np.int32), kH, kW)        # (P, K)
    acc  = cols @ W.T.astype(np.int32) + b                       # (P, F)

    # Requantize: (acc * mult) >> 16, clamp [0, 127]
    mult = lm['requant_mult']
    out  = np.clip((acc.astype(np.int64) * mult) >> 16, 0, 127).astype(np.int8)

    C_in = x_int8.shape[0]
    H_in = x_int8.shape[1]
    W_in = x_int8.shape[2]
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    return out.reshape(H_out, W_out, F).transpose(2, 0, 1)       # (F, H_out, W_out)


def maxpool_int8(x, size=2):
    """x: (C, H, W) int8 → (C, H//size, W//size) int8"""
    C, H, W = x.shape
    H_out = H // size
    W_out = W // size
    xr = x[:, :H_out * size, :W_out * size]
    return xr.reshape(C, H_out, size, W_out, size).max(axis=(2, 4)).astype(np.int8)


# ─── FPGA-style INT8 FC (tile-based, matches inference_engine.v) ──────────────

def fc_int8_tile(x_int8, lm):
    """
    Tile-based FC matching FPGA 4×4 systolic array behaviour.
    x_int8: (in_dim,)  int8
    Returns: (out_dim,) int8 (relu) or int32 (no relu)
    """
    in_dim   = lm['in_features']
    out_dim  = lm['out_features']
    has_relu = lm['activation'] == 'relu'
    mult     = lm['requant_mult']

    in_pad  = pad4(in_dim)
    out_pad = pad4(out_dim)

    W_raw = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                        dtype=np.int8).reshape(out_dim, in_dim)
    b_raw = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']), dtype=np.int32)

    # Zero-pad
    x_pad = np.zeros(in_pad,  dtype=np.int8)
    x_pad[:in_dim] = x_int8[:in_dim]

    W_pad = np.zeros((out_pad, in_pad), dtype=np.int8)
    W_pad[:out_dim, :in_dim] = W_raw
    b_pad = np.zeros(out_pad, dtype=np.int32)
    b_pad[:out_dim] = b_raw

    n_out_groups = out_pad // 4
    n_in_tiles   = in_pad  // 4

    if has_relu:
        output = np.zeros(out_pad, dtype=np.int8)
    else:
        output = np.zeros(out_pad, dtype=np.int32)

    for g in range(n_out_groups):
        acc = b_pad[g * 4:(g + 1) * 4].astype(np.int32).copy()

        for t in range(n_in_tiles):
            W_tile = W_pad[g*4:g*4+4, t*4:t*4+4].astype(np.int32)
            x_tile = x_pad[t*4:t*4+4].astype(np.int32)
            acc += W_tile @ x_tile

        if has_relu:
            for i in range(4):
                prod    = np.int64(acc[i]) * np.int64(mult)
                scaled  = int(prod >> 16)
                clamped = max(0, min(127, scaled))
                output[g * 4 + i] = np.int8(clamped)
        else:
            output[g * 4:g * 4 + 4] = acc

    return output[:out_dim]


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_cnn(image_int8_flat, meta, verbose=False):
    """
    image_int8_flat: (784,) int8 — pixels in [0, 127]
    Returns: (predicted_digit, scores)
    """
    layers      = meta['layers']
    conv_layers = [l for l in layers if l['type'] == 'conv']
    fc_layers   = [l for l in layers if l['type'] == 'fc']
    conv_info   = meta['conv_info']

    # Reshape to (1, 28, 28)
    x = image_int8_flat.reshape(1, 28, 28)

    # ── Conv + Pool (host INT8) ────────────────────────────────────────────
    for idx, lm in enumerate(conv_layers):
        ci = conv_info[idx]
        if verbose:
            print(f'  Conv{idx}: {ci["in_channels"]}×{ci["in_h"]}×{ci["in_w"]}'
                  f' → {ci["out_channels"]}×{ci["out_h"]}×{ci["out_w"]}'
                  f'  [im2col, host INT8, requant_mult={lm["requant_mult"]}]')
        x = conv_int8(x, lm)                       # (F, H_out, W_out) int8
        x = maxpool_int8(x, size=ci['pool'])        # (F, H//pool, W//pool) int8
        if verbose:
            print(f'    → after pool: {x.shape}  min={x.min()} max={x.max()}')

    # ── Flatten ───────────────────────────────────────────────────────────
    x_flat = x.ravel()    # (200,) int8
    if verbose:
        print(f'  Flatten → {len(x_flat)} values')

    # ── FC layers (FPGA tile-based MAC) ────────────────────────────────────
    x = x_flat
    for lm in fc_layers:
        if verbose:
            print(f'  FC: {lm["in_features"]}→{lm["out_features"]}'
                  f'  relu={lm["activation"]=="relu"}'
                  f'  requant_mult={lm["requant_mult"]}')
        x = fc_int8_tile(x, lm)
        if verbose:
            print(f'    → min={x.min()} max={x.max()}')

    scores    = x.astype(np.int32) if x.dtype != np.int32 else x
    predicted = int(np.argmax(scores))
    return predicted, scores


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    meta_path = os.path.join(WEIGHTS_DIR, 'cnn_meta.json')
    if not os.path.exists(meta_path):
        print(f'ERROR: {meta_path} not found.')
        print('Run model/train_cnn.py first to generate weights.')
        sys.exit(1)

    meta = load_meta()
    print('CNN architecture:')
    print('  Conv(1→4, 5×5) + ReLU + MaxPool(2×2) → 12×12×4 = 576  [host INT8 im2col]')
    print('  Conv(4→8, 3×3) + ReLU + MaxPool(2×2) →   5×5×8 = 200  [host INT8 im2col]')
    print('  FC(200→64) + ReLU                                       [FPGA tile MAC]')
    print('  FC(64→10)                                               [FPGA tile MAC]')
    print(f'\n  Float accuracy (train.py): {meta.get("float_accuracy", "n/a"):.4f}')
    print(f'  Quantized acc (500 samp):  {meta.get("quantized_accuracy_500", "n/a"):.4f}')

    samples_path = os.path.join(WEIGHTS_DIR, 'test_samples.bin')
    labels_path  = os.path.join(WEIGHTS_DIR, 'test_labels.bin')
    if not os.path.exists(samples_path):
        print('\nERROR: test_samples.bin not found — run train_cnn.py first.')
        sys.exit(1)

    samples = np.fromfile(samples_path, dtype=np.int8).reshape(-1, 784)
    labels  = np.fromfile(labels_path,  dtype=np.uint8)
    n       = len(labels)

    print(f'\nRunning bit-accurate simulation on {n} test samples...\n')
    correct = 0

    for i in range(n):
        label = labels[i]
        print(f'Sample {i} (label={label}):')
        pred, scores = run_cnn(samples[i], meta, verbose=True)
        mark = '✓' if pred == label else '✗'
        print(f'  {mark}  predicted={pred}  scores={scores}\n')
        if pred == label:
            correct += 1

    print(f'Accuracy: {correct}/{n} = {correct/n*100:.1f}%')


if __name__ == '__main__':
    main()
