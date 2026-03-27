#!/usr/bin/env python3
"""
Bit-accurate Python simulation of the FPGA inference engine.
Matches the hardware's integer arithmetic exactly:
  - int8 weights/activations, int32 accumulators
  - 4×4 tile-based MAC with bias initialization
  - Requantization: (acc * requant_mult) >> 16, clamp [0, 127]
  - Padding to multiples of 4
"""

import numpy as np
import json
import os
import struct

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


def pad4(n):
    """Round up to next multiple of 4."""
    return ((n + 3) // 4) * 4


def load_layer(meta):
    """Load quantized weights and bias for one layer."""
    w_path = os.path.join(WEIGHTS_DIR, meta['w_file'])
    b_path = os.path.join(WEIGHTS_DIR, meta['b_file'])

    # Weights: stored as (out_features, in_features) row-major int8
    W = np.fromfile(w_path, dtype=np.int8).reshape(meta['out_features'], meta['in_features'])
    # Bias: stored as int32
    b = np.fromfile(b_path, dtype=np.int32)

    return W, b


def compute_requant_mult(meta):
    """Compute 16-bit fixed-point requantization multiplier."""
    requant_scale = meta['requant_scale']
    mult = int(round(requant_scale * (1 << 16)))
    return min(mult, 65535)  # clamp to 16-bit unsigned


def pad_weights(W, b, in_dim_pad, out_dim_pad):
    """Zero-pad weight matrix and bias to multiples of 4."""
    out_f, in_f = W.shape
    W_pad = np.zeros((out_dim_pad, in_dim_pad), dtype=np.int8)
    W_pad[:out_f, :in_f] = W
    b_pad = np.zeros(out_dim_pad, dtype=np.int32)
    b_pad[:out_f] = b
    return W_pad, b_pad


def systolic_tile_mac(W_tile, x_tile):
    """
    Simulate 4×4 systolic array tile computation.
    W_tile: (4, 4) int8
    x_tile: (4,) int8
    Returns: (4,) int32 partial sums
    """
    # Each row i: sum_j W_tile[i][j] * x_tile[j]
    result = np.zeros(4, dtype=np.int32)
    for i in range(4):
        for j in range(4):
            result[i] += np.int32(W_tile[i, j]) * np.int32(x_tile[j])
    return result


def inference_layer(x_int8, W, b, has_relu, requant_mult):
    """
    Simulate one layer's computation, matching FPGA tile-by-tile processing.
    x_int8: (in_dim,) int8 input
    W: (out_dim, in_dim) int8 weights (padded)
    b: (out_dim,) int32 bias (padded)
    Returns: (out_dim,) int8 (if relu) or (out_dim,) int32 (if no relu)
    """
    out_dim, in_dim = W.shape
    n_out_groups = out_dim // 4
    n_in_tiles = in_dim // 4

    if has_relu:
        output = np.zeros(out_dim, dtype=np.int8)
    else:
        output = np.zeros(out_dim, dtype=np.int32)

    for g in range(n_out_groups):
        # Initialize accumulators with bias
        acc = np.array([b[g*4 + i] for i in range(4)], dtype=np.int32)

        # Process input tiles
        for t in range(n_in_tiles):
            W_tile = W[g*4:g*4+4, t*4:t*4+4]  # (4, 4)
            x_tile = x_int8[t*4:t*4+4]          # (4,)
            acc += systolic_tile_mac(W_tile, x_tile)

        # Requantize
        if has_relu:
            for i in range(4):
                # Match hardware: (acc * mult) >> 16, arithmetic right shift
                prod = np.int64(acc[i]) * np.int64(requant_mult)
                scaled = int(prod >> 16)  # arithmetic right shift
                clamped = max(0, min(127, scaled))
                output[g*4 + i] = np.int8(clamped)
        else:
            for i in range(4):
                output[g*4 + i] = acc[i]

    return output


def run_inference(image_float, verbose=True):
    """
    Run full 3-layer MLP inference matching FPGA behavior.
    image_float: (784,) float32 in [0, 1]
    Returns: predicted digit (0-9), raw output scores
    """
    with open(os.path.join(WEIGHTS_DIR, 'model_meta.json')) as f:
        meta = json.load(f)

    # Quantize input: [0, 1] → int8 [0, 127]
    x = np.clip(np.round(image_float * 127), 0, 127).astype(np.int8)

    for layer_meta in meta['layers']:
        W_raw, b_raw = load_layer(layer_meta)
        in_dim = layer_meta['in_features']
        out_dim = layer_meta['out_features']
        has_relu = layer_meta['activation'] == 'relu'
        requant_mult = compute_requant_mult(layer_meta)

        in_dim_pad = pad4(in_dim)
        out_dim_pad = pad4(out_dim)

        # Pad input
        x_pad = np.zeros(in_dim_pad, dtype=np.int8)
        x_pad[:len(x)] = x[:in_dim]

        # Pad weights/bias
        W_pad, b_pad = pad_weights(W_raw, b_raw, in_dim_pad, out_dim_pad)

        if verbose:
            print(f"  Layer {layer_meta['index']}: {in_dim}→{out_dim} "
                  f"(padded {in_dim_pad}→{out_dim_pad}), "
                  f"relu={has_relu}, requant_mult={requant_mult}")

        x = inference_layer(x_pad, W_pad, b_pad, has_relu, requant_mult)

    # For output layer (no relu): raw int32 accumulators
    scores = x[:meta['layers'][-1]['out_features']]
    predicted = int(np.argmax(scores))

    if verbose:
        print(f"  Scores: {scores}")
        print(f"  Predicted: {predicted}")

    return predicted, scores


def main():
    # Load test samples
    samples_path = os.path.join(WEIGHTS_DIR, 'test_samples.bin')
    labels_path = os.path.join(WEIGHTS_DIR, 'test_labels.bin')

    samples = np.fromfile(samples_path, dtype=np.int8).reshape(-1, 784)
    labels = np.fromfile(labels_path, dtype=np.uint8)

    n_samples = len(labels)
    correct = 0

    print(f"Running bit-accurate inference on {n_samples} test samples...\n")

    for i in range(n_samples):
        # Convert int8 [0, 127] back to float [0, 1] for the inference function
        image_float = np.clip(samples[i].astype(np.float32) / 127.0, 0, 1)
        print(f"Sample {i} (label={labels[i]}):")
        pred, scores = run_inference(image_float, verbose=True)
        match = "✓" if pred == labels[i] else "✗"
        print(f"  {match} predicted={pred}, label={labels[i]}\n")
        if pred == labels[i]:
            correct += 1

    print(f"Accuracy: {correct}/{n_samples} = {correct/n_samples*100:.1f}%")


if __name__ == '__main__':
    main()
