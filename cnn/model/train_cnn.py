#!/usr/bin/env python3
"""
Train a small CNN on MNIST, quantize INT8 (PTQ), export weights.

Architecture:
  Input: 28×28×1
  Conv(1→4, 5×5, valid) + ReLU + MaxPool(2×2) → 12×12×4 = 576   [host INT8 im2col]
  Conv(4→8, 3×3, valid) + ReLU + MaxPool(2×2) → 5×5×8  = 200   [host INT8 im2col]
  FC(200→64) + ReLU                                               [FPGA]
  FC(64→10)                                                       [FPGA, argmax]

Usage:
  ../../.env/bin/python model/train_cnn.py [--epochs 20] [--lr 0.01] [--batch 64]
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')

# ─── Architecture constants ───────────────────────────────────────────────────
CONV_INFO = [
    {'kH': 5, 'kW': 5, 'stride': 1, 'pad': 0,
     'in_channels': 1, 'out_channels': 4,
     'in_h': 28, 'in_w': 28, 'out_h': 24, 'out_w': 24, 'pool': 2,
     'out_h_pool': 12, 'out_w_pool': 12},
    {'kH': 3, 'kW': 3, 'stride': 1, 'pad': 0,
     'in_channels': 4, 'out_channels': 8,
     'in_h': 12, 'in_w': 12, 'out_h': 10, 'out_w': 10, 'pool': 2,
     'out_h_pool': 5, 'out_w_pool': 5},
]
FC_INFO = [
    {'in': 200, 'out': 64, 'relu': True},
    {'in': 64,  'out': 10, 'relu': False},
]

# ─── im2col / col2im ──────────────────────────────────────────────────────────

def im2col(x, kH, kW, stride=1, pad=0):
    """
    x: (N, C, H, W)
    Returns: (N*H_out*W_out, C*kH*kW)  — each row is one receptive-field patch
    """
    N, C, H, W = x.shape
    H_out = (H - kH + 2 * pad) // stride + 1
    W_out = (W - kW + 2 * pad) // stride + 1

    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   mode='constant')

    k = np.arange(C * kH * kW)
    c_idx  = k // (kH * kW)
    kh_idx = (k % (kH * kW)) // kW
    kw_idx = k % kW

    h_starts = np.arange(H_out) * stride   # (H_out,)
    w_starts = np.arange(W_out) * stride   # (W_out,)

    # Advanced indexing → (N, C*kH*kW, H_out, W_out)
    h_idx = h_starts[None, :, None] + kh_idx[:, None, None]   # (K, H_out, 1)
    w_idx = w_starts[None, None, :] + kw_idx[:, None, None]   # (K, 1, W_out)
    cols = x[:, c_idx[:, None, None], h_idx, w_idx]           # (N,K,H_out,W_out)

    # Reshape to (N*H_out*W_out, C*kH*kW)
    cols = cols.reshape(N, C * kH * kW, H_out * W_out)        # (N,K,P)
    cols = cols.transpose(0, 2, 1).reshape(-1, C * kH * kW)   # (N*P, K)
    return cols


def col2im(dcols, x_shape, kH, kW, stride=1, pad=0):
    """
    dcols:   (N*H_out*W_out, C*kH*kW)
    x_shape: (N, C, H, W)  — original shape before padding
    Returns: dx (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_out = (H - kH + 2 * pad) // stride + 1
    W_out = (W - kW + 2 * pad) // stride + 1
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad

    # dcols: (N*H_out*W_out, C*kH*kW) → (N, C*kH*kW, H_out, W_out)
    dcols_4d = dcols.reshape(N, H_out, W_out, C * kH * kW)
    dcols_4d = dcols_4d.transpose(0, 3, 1, 2)       # (N, K, H_out, W_out)

    k = np.arange(C * kH * kW)
    c_idx  = k // (kH * kW)
    kh_idx = (k % (kH * kW)) // kW
    kw_idx = k % kW

    h_starts = np.arange(H_out) * stride
    w_starts = np.arange(W_out) * stride

    dx_pad = np.zeros((N, C, H_pad, W_pad), dtype=dcols.dtype)

    # For each kernel position: no repeated (h,w) indices when stride≥1, so += is safe
    for ki in range(C * kH * kW):
        c  = c_idx[ki]
        kh = kh_idx[ki]
        kw = kw_idx[ki]
        dx_pad[:, c,
               h_starts[:, None] + kh,
               w_starts[None, :] + kw] += dcols_4d[:, ki, :, :]

    if pad > 0:
        return dx_pad[:, :, pad:-pad, pad:-pad]
    return dx_pad


# ─── Layer primitives ─────────────────────────────────────────────────────────

def conv_forward(x, W, b, stride=1, pad=0):
    """x:(N,C,H,W)  W:(F,C,kH,kW)  b:(F,)  →  out:(N,F,H_out,W_out)"""
    N, C, H, Ww = x.shape
    F, _, kH, kW = W.shape
    H_out = (H  - kH + 2 * pad) // stride + 1
    W_out = (Ww - kW + 2 * pad) // stride + 1

    cols  = im2col(x, kH, kW, stride, pad)    # (N*P, C*kH*kW)
    W_row = W.reshape(F, -1)                   # (F, C*kH*kW)
    out   = cols @ W_row.T + b                 # (N*P, F)
    out   = out.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)

    cache = (x, W, b, stride, pad, cols)
    return out, cache


def conv_backward(dout, cache):
    x, W, b, stride, pad, cols = cache
    N, C, H, Ww = x.shape
    F, _, kH, kW = W.shape
    H_out, W_out = dout.shape[2], dout.shape[3]

    dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, F)   # (N*P, F)
    W_row  = W.reshape(F, -1)

    dW     = (dout_r.T @ cols).reshape(W.shape)
    db     = dout_r.sum(axis=0)
    dcols  = dout_r @ W_row                               # (N*P, C*kH*kW)
    dx     = col2im(dcols, x.shape, kH, kW, stride, pad)
    return dx, dW, db


def maxpool_forward(x, size=2):
    N, C, H, W = x.shape
    H_out, W_out = H // size, W // size
    xr   = x.reshape(N, C, H_out, size, W_out, size)
    out  = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None]).astype(np.float64)
    # Distribute gradient evenly when there are ties
    mask /= mask.sum(axis=(3, 5), keepdims=True)
    cache = (x.shape, mask, size)
    return out, cache


def maxpool_backward(dout, cache):
    x_shape, mask, size = cache
    N, C, H, W = x_shape
    H_out, W_out = H // size, W // size
    dx = dout[:, :, :, None, :, None] * mask
    return dx.reshape(N, C, H, W)


def relu_forward(x):
    return np.maximum(0.0, x), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def fc_forward(x, W, b):
    out   = x @ W.T + b
    cache = (x, W)
    return out, cache


def fc_backward(dout, cache):
    x, W = cache
    dx   = dout @ W
    dW   = dout.T @ x
    db   = dout.sum(axis=0)
    return dx, dW, db


def softmax_cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_l  = np.exp(logits)
    probs  = exp_l / exp_l.sum(axis=1, keepdims=True)
    N      = logits.shape[0]
    loss   = -np.log(probs[np.arange(N), labels] + 1e-15).mean()
    dloss  = probs.copy()
    dloss[np.arange(N), labels] -= 1.0
    dloss  /= N
    return loss, dloss


# ─── CNN model ────────────────────────────────────────────────────────────────

class CNN:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        self.params = {
            'W1': rng.randn(4, 1, 5, 5).astype(np.float64) * np.sqrt(2.0 / (1 * 5 * 5)),
            'b1': np.zeros(4, dtype=np.float64),
            'W2': rng.randn(8, 4, 3, 3).astype(np.float64) * np.sqrt(2.0 / (4 * 3 * 3)),
            'b2': np.zeros(8, dtype=np.float64),
            'W3': rng.randn(64, 200).astype(np.float64) * np.sqrt(2.0 / 200),
            'b3': np.zeros(64, dtype=np.float64),
            'W4': rng.randn(10, 64).astype(np.float64) * np.sqrt(2.0 / 64),
            'b4': np.zeros(10, dtype=np.float64),
        }
        self.vel = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        """x: (N,1,28,28)"""
        p = self.params
        c1, cc1 = conv_forward(x,   p['W1'], p['b1'])   # (N,4,24,24)
        r1, cr1 = relu_forward(c1)
        p1, cp1 = maxpool_forward(r1, 2)                 # (N,4,12,12)

        c2, cc2 = conv_forward(p1,  p['W2'], p['b2'])   # (N,8,10,10)
        r2, cr2 = relu_forward(c2)
        p2, cp2 = maxpool_forward(r2, 2)                 # (N,8,5,5)

        fl  = p2.reshape(p2.shape[0], -1)                # (N,200)

        f3, cf3 = fc_forward(fl,  p['W3'], p['b3'])     # (N,64)
        r3, cr3 = relu_forward(f3)

        f4, cf4 = fc_forward(r3,  p['W4'], p['b4'])     # (N,10)

        caches = (cc1, cr1, cp1, cc2, cr2, cp2,
                  fl.shape, cf3, cr3, cf4)
        return f4, caches

    def backward(self, dlogits, caches):
        (cc1, cr1, cp1, cc2, cr2, cp2, fl_shape, cf3, cr3, cf4) = caches
        g = {}

        dr3, g['W4'], g['b4'] = fc_backward(dlogits, cf4)
        df3 = relu_backward(dr3, cr3)
        dfl, g['W3'], g['b3'] = fc_backward(df3, cf3)

        dp2 = dfl.reshape(fl_shape[0], 8, 5, 5)
        dr2 = maxpool_backward(dp2, cp2)
        dc2 = relu_backward(dr2, cr2)
        dp1, g['W2'], g['b2'] = conv_backward(dc2, cc2)

        dr1 = maxpool_backward(dp1, cp1)
        dc1 = relu_backward(dr1, cr1)
        _,   g['W1'], g['b1'] = conv_backward(dc1, cc1)
        return g

    def predict(self, x):
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)

    def step(self, grads, lr=0.01, momentum=0.9):
        for k in self.params:
            self.vel[k] = momentum * self.vel[k] - lr * grads[k]
            self.params[k] += self.vel[k]


# ─── Post-training quantization ───────────────────────────────────────────────

def calibrate_activations(model, X_calib):
    """
    Run float forward on calibration data, record per-layer activation maxima.
    Returns dict of layer_name → max float activation value.
    """
    act_max = {'relu1': 0.0, 'relu2': 0.0, 'relu3': 0.0}
    bs = 64
    N  = len(X_calib)
    p  = model.params

    for i in range(0, N, bs):
        xb = X_calib[i:i + bs]
        c1, _ = conv_forward(xb, p['W1'], p['b1'])
        r1     = np.maximum(0.0, c1)
        p1, _  = maxpool_forward(r1, 2)
        act_max['relu1'] = max(act_max['relu1'], float(r1.max()))

        c2, _ = conv_forward(p1, p['W2'], p['b2'])
        r2     = np.maximum(0.0, c2)
        p2, _  = maxpool_forward(r2, 2)
        act_max['relu2'] = max(act_max['relu2'], float(r2.max()))

        fl    = p2.reshape(p2.shape[0], -1)
        f3, _ = fc_forward(fl, p['W3'], p['b3'])
        r3    = np.maximum(0.0, f3)
        act_max['relu3'] = max(act_max['relu3'], float(r3.max()))

    return act_max


def quantize_and_export(model, act_max):
    """
    Quantize all layers to INT8 (PTQ) and write binary weight/bias files.
    Returns list of layer metadata dicts.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    layers   = []
    x_scale  = 1.0 / 127.0          # input: float [0,1] → int8 [0,127]

    layer_specs = [
        ('conv0', 'W1', 'b1', 4,  True,  'relu1', 'conv', (4, 1, 5, 5)),
        ('conv1', 'W2', 'b2', 8,  True,  'relu2', 'conv', (8, 4, 3, 3)),
        ('fc0',   'W3', 'b3', 64, True,  'relu3', 'fc',   None),
        ('fc1',   'W4', 'b4', 10, False, None,    'fc',   None),
    ]

    for name, wk, bk, out_ch, has_relu, act_key, ltype, conv_shape in layer_specs:
        W_f = model.params[wk]
        b_f = model.params[bk]

        w_scale   = float(np.abs(W_f).max()) / 127.0
        out_scale = (act_max[act_key] / 127.0) if has_relu else 1.0

        W_int8 = np.clip(np.round(W_f / w_scale), -127, 127).astype(np.int8)

        acc_scale = w_scale * x_scale          # accumulator scale
        b_int32   = np.round(b_f / acc_scale).astype(np.int32)

        requant_scale = (acc_scale / out_scale) if has_relu else 0.0
        requant_mult  = (min(int(round(requant_scale * (1 << 16))), 65535)
                         if has_relu else 0)

        lm = {
            'name':         name,
            'type':         ltype,
            'activation':   'relu' if has_relu else 'none',
            'w_scale':      w_scale,
            'x_scale':      x_scale,
            'out_scale':    out_scale,
            'requant_scale': requant_scale,
            'requant_mult': requant_mult,
            'w_file':       f'{name}_weights.bin',
            'b_file':       f'{name}_bias.bin',
        }

        if ltype == 'conv':
            F, C, kH, kW = conv_shape
            patch_size = C * kH * kW
            lm.update({
                'out_channels': F,
                'in_channels':  C,
                'kernel_h':     kH,
                'kernel_w':     kW,
                'patch_size':   patch_size,
            })
            W_save = W_int8.reshape(F, patch_size)    # (F, patch_size)
        else:
            out_dim, in_dim = W_f.shape
            lm.update({
                'in_features':  in_dim,
                'out_features': out_dim,
            })
            W_save = W_int8

        W_save.tofile(os.path.join(WEIGHTS_DIR, lm['w_file']))
        b_int32.tofile(os.path.join(WEIGHTS_DIR, lm['b_file']))

        layers.append(lm)
        x_scale = out_scale    # chain scales

    return layers


# ─── Quantized accuracy check (pure Python, bit-accurate) ────────────────────

def _quant_conv(x_int8, lm):
    """INT8 conv via im2col, returns int8 result post-ReLU."""
    W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                    dtype=np.int8).reshape(lm['out_channels'], lm['patch_size'])
    b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']), dtype=np.int32)
    kH, kW = lm['kernel_h'], lm['kernel_w']

    cols   = im2col(x_int8.astype(np.int32), kH, kW)   # (N*P, K)
    acc    = cols @ W.T.astype(np.int32) + b            # (N*P, F)

    N    = x_int8.shape[0]
    H_in = x_int8.shape[2]
    W_in = x_int8.shape[3]
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    F     = lm['out_channels']

    scaled = (acc.astype(np.int64) * lm['requant_mult']) >> 16
    out    = np.clip(scaled, 0, 127).astype(np.int8)
    return out.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)


def _quant_fc(x_int8, lm):
    """INT8 FC, padded to multiples of 4, returns int8 (relu) or int32."""
    in_dim  = lm['in_features']
    out_dim = lm['out_features']
    in_pad  = ((in_dim  + 3) // 4) * 4
    out_pad = ((out_dim + 3) // 4) * 4
    has_relu = lm['activation'] == 'relu'

    W = np.fromfile(os.path.join(WEIGHTS_DIR, lm['w_file']),
                    dtype=np.int8).reshape(out_dim, in_dim)
    b = np.fromfile(os.path.join(WEIGHTS_DIR, lm['b_file']), dtype=np.int32)

    N = x_int8.shape[0]
    x_pad = np.zeros((N, in_pad),  dtype=np.int32)
    x_pad[:, :in_dim] = x_int8.astype(np.int32)[:, :in_dim]

    W_pad = np.zeros((out_pad, in_pad), dtype=np.int32)
    W_pad[:out_dim, :in_dim] = W.astype(np.int32)
    b_pad = np.zeros(out_pad, dtype=np.int32)
    b_pad[:out_dim] = b

    acc = x_pad @ W_pad.T + b_pad     # (N, out_pad)

    if has_relu:
        scaled = (acc.astype(np.int64) * lm['requant_mult']) >> 16
        out    = np.clip(scaled, 0, 127).astype(np.int8)
    else:
        out = acc.astype(np.int32)

    return out[:, :out_dim]


def eval_quantized(layers, X_test, y_test, n=500):
    """Check INT8 accuracy on first n test samples."""
    conv_layers = [l for l in layers if l['type'] == 'conv']
    fc_layers   = [l for l in layers if l['type'] == 'fc']
    n = min(n, len(X_test))
    correct = 0

    for i in range(n):
        xi = X_test[i:i + 1]    # (1, 1, 28, 28)
        x  = np.clip(np.round(xi * 127), 0, 127).astype(np.int8)

        for lm in conv_layers:
            x = _quant_conv(x, lm)
            # MaxPool 2×2
            ps = CONV_INFO[conv_layers.index(lm)]['pool']
            H  = x.shape[2] // ps * ps
            W  = x.shape[3] // ps * ps
            x  = x[:, :, :H, :W].reshape(1, x.shape[1], H // ps, ps, W // ps, ps)
            x  = x.max(axis=(3, 5)).astype(np.int8)

        x = x.reshape(1, -1)    # flatten → (1, 200)

        for lm in fc_layers:
            x = _quant_fc(x, lm)

        pred = int(np.argmax(x[0]))
        if pred == y_test[i]:
            correct += 1

    return correct / n


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_mnist():
    print("Loading MNIST from sklearn (may download on first run)...")
    mnist   = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X       = mnist.data.astype(np.float64) / 255.0
    y       = mnist.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y)

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test  = X_test.reshape(-1, 1, 28, 28)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ─── Training loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs (default 20, ~15-30 min on CPU)')
    parser.add_argument('--lr',     type=float, default=0.01)
    parser.add_argument('--batch',  type=int, default=64)
    args = parser.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_mnist()

    np.random.seed(42)
    model = CNN(seed=42)

    n_train     = len(X_train)
    best_acc    = 0.0
    best_params = None

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, batch={args.batch}")
    print("Conv layers run in numpy → training can take several minutes per epoch.\n")

    for epoch in range(args.epochs):
        idx   = np.random.permutation(n_train)
        total = 0.0
        nb    = 0
        lr    = args.lr if epoch < 10 else args.lr * 0.1

        for i in range(0, n_train, args.batch):
            xb = X_train[idx[i:i + args.batch]]
            yb = y_train[idx[i:i + args.batch]]

            logits, caches = model.forward(xb)
            loss, dlogits  = softmax_cross_entropy(logits, yb)
            grads          = model.backward(dlogits, caches)
            model.step(grads, lr=lr)

            total += loss
            nb    += 1

            if nb % 50 == 0:
                sys.stdout.write(f'\r  [{epoch+1}/{args.epochs}] batch {nb}: loss={total/nb:.4f}')
                sys.stdout.flush()

        # Quick validation on 1000 test samples
        preds = []
        for i in range(0, 1000, 64):
            preds.extend(model.predict(X_test[i:i + 64]))
        acc = (np.array(preds[:1000]) == y_test[:1000]).mean()

        if acc > best_acc:
            best_acc    = acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        print(f'\r  Epoch {epoch+1:2d}/{args.epochs}: loss={total/nb:.4f}'
              f'  val_acc={acc:.4f}  best={best_acc:.4f}')

    model.params = best_params
    print(f'\nBest validation accuracy: {best_acc:.4f}')

    # Full test accuracy
    preds = []
    for i in range(0, len(X_test), 64):
        preds.extend(model.predict(X_test[i:i + 64]))
    float_acc = (np.array(preds[:len(X_test)]) == y_test).mean()
    print(f'Float accuracy (full test set): {float_acc:.4f}')

    # PTQ
    print('\nCalibrating activations on 2000 training samples...')
    act_max = calibrate_activations(model, X_train[:2000])
    for k, v in act_max.items():
        print(f'  {k}: max={v:.4f}  →  scale={v/127:.6f}')

    layers = quantize_and_export(model, act_max)

    print('\nChecking quantized accuracy on 500 test samples...')
    q_acc = eval_quantized(layers, X_test, y_test, n=500)
    print(f'  Quantized accuracy: {q_acc:.4f}')

    # Save 10 test samples (flat int8, same format as MLP test_samples.bin)
    imgs = np.clip(np.round(X_test[:10, 0].reshape(10, 784) * 127), 0, 127).astype(np.int8)
    lbls = y_test[:10].astype(np.uint8)
    imgs.tofile(os.path.join(WEIGHTS_DIR, 'test_samples.bin'))
    lbls.tofile(os.path.join(WEIGHTS_DIR, 'test_labels.bin'))
    print(f'Saved 10 test samples, labels: {list(lbls)}')

    # Save metadata
    meta = {
        'description': 'INT8 CNN for MNIST. Conv layers run on host via im2col; '
                        'FC layers run on FPGA via existing UART protocol.',
        'float_accuracy':          float_acc,
        'quantized_accuracy_500':  q_acc,
        'conv_info':               CONV_INFO,
        'layers':                  layers,
    }
    meta_path = os.path.join(WEIGHTS_DIR, 'cnn_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nAll weights written to {WEIGHTS_DIR}/')
    files = [f for f in os.listdir(WEIGHTS_DIR) if not f.startswith('.')]
    for fn in sorted(files):
        size = os.path.getsize(os.path.join(WEIGHTS_DIR, fn))
        print(f'  {fn:30s}  {size:7d} bytes')


if __name__ == '__main__':
    main()
