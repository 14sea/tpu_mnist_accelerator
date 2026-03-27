#!/usr/bin/env python3
"""
MNIST MLP 訓練 + 8-bit 對稱量化 + 權重導出
網絡結構: 784 -> 128 -> 64 -> 10
量化方案: 每層獨立 per-tensor 對稱量化 (int8)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
import json
import os

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')

# ─────────────────────────────────────────
# 1. 數據準備
# ─────────────────────────────────────────
def load_mnist():
    print("載入 MNIST 數據集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0   # 歸一化到 [0, 1]
    y = mnist.target.astype(np.int32)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 2. 訓練
# ─────────────────────────────────────────
def train(X_train, y_train):
    print("\n訓練 MLP (784->128->64->10)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=30,
        batch_size=256,
        learning_rate_init=0.001,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train, y_train)
    return clf


# ─────────────────────────────────────────
# 3. 浮點推理精度
# ─────────────────────────────────────────
def evaluate_float(clf, X_test, y_test):
    acc = clf.score(X_test, y_test)
    print(f"\n浮點模型精度: {acc*100:.2f}%")
    return acc


# ─────────────────────────────────────────
# 4. 8-bit 對稱量化
# ─────────────────────────────────────────
def calc_scale(arr, bits=8):
    """計算對稱量化 scale = max(|x|) / (2^(bits-1) - 1)"""
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return np.float32(1.0)
    return np.float32(max_val / (2 ** (bits - 1) - 1))


def quantize_model(clf, X_calibrate):
    """
    正確的量化方案：
    - 權重: int8, scale = max(|W|) / 127
    - 偏置: int32, scale = w_scale * x_scale  ← 關鍵：與累加器同scale
    - 激活: int8, scale 由校準集決定
    FPGA 累加器: acc_int32 = W_int8 @ x_int8 + b_int32
    重量化: x_next = clip(round(acc * (w_scale * x_scale / out_scale)), -128, 127)
    """
    print("\n進行 8-bit 對稱量化（正確偏置scale方案）...")

    layers = []
    activation = X_calibrate.copy()

    # 輸入 scale：像素 [0,1] → int8 [0,127]
    x_scale = np.float32(1.0 / 127.0)

    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        # 權重量化
        w_scale = calc_scale(W)
        W_q = np.clip(np.round(W / w_scale), -128, 127).astype(np.int8)

        # 偏置量化: 使用累加器 scale = w_scale * x_scale，存為 int32
        acc_scale = w_scale * x_scale
        b_q = np.clip(np.round(b / acc_scale), -(2**31), 2**31 - 1).astype(np.int32)

        # 前向計算校準集激活，確定輸出 scale
        z = activation @ W + b
        if i < len(clf.coefs_) - 1:
            activation = np.maximum(0, z)
        else:
            activation = z

        out_scale = calc_scale(activation) if i < len(clf.coefs_) - 1 else np.float32(1.0)

        layers.append({
            'W_q':       W_q,           # int8 (in, out)
            'b_q':       b_q,           # int32 (out,)
            'w_scale':   w_scale,       # float32
            'x_scale':   x_scale,       # float32（本層輸入 scale）
            'out_scale': out_scale,     # float32（本層輸出 scale → 下一層 x_scale）
            'in_features':  W.shape[0],
            'out_features': W.shape[1],
        })

        print(f"  Layer {i}: {W.shape[0]:>4}→{W.shape[1]:<4}"
              f"  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}"
              f"  out_scale={out_scale:.6f}")

        x_scale = out_scale   # 下一層輸入 scale

    return layers


# ─────────────────────────────────────────
# 5. 量化推理（模擬 FPGA 行為）
# ─────────────────────────────────────────
def quantized_inference(layers, x_float):
    """
    模擬 FPGA int8 脈動陣列推理
    FPGA 計算流程：
      acc_int32 = W_int8 @ x_int8 + b_int32
      x_next_int8 = clip(round(acc * requant_scale), 0, 127)  # ReLU 後非負
    """
    # 輸入量化 [0,1] → int8 [0,127]
    x = np.clip(np.round(x_float * 127), 0, 127).astype(np.int8)

    for i, layer in enumerate(layers):
        W_q  = layer['W_q'].astype(np.int32)    # (in, out)
        b_q  = layer['b_q'].astype(np.int32)    # (out,)

        # int32 累加（模擬脈動陣列 MAC）
        acc = x.astype(np.int32) @ W_q + b_q   # (out,)

        # 重量化因子: acc * w_scale * x_scale = 浮點值
        # 再除以 out_scale 得到下一層 int8
        requant = (layer['w_scale'] * layer['x_scale']) / layer['out_scale']

        if i < len(layers) - 1:
            # ReLU + 重量化到 [0, 127]
            out = np.maximum(0, np.round(acc.astype(np.float32) * requant))
            x = np.clip(out, 0, 127).astype(np.int8)
        else:
            # 輸出層：直接用累加器做 argmax（不需要重量化）
            return int(np.argmax(acc.astype(np.float32) * requant))

    return -1


def evaluate_quantized(layers, X_test, y_test, n_samples=2000):
    print(f"\n量化模型精度評估（前 {n_samples} 個樣本）...")
    correct = 0
    for i in range(n_samples):
        pred = quantized_inference(layers, X_test[i])
        if pred == y_test[i]:
            correct += 1
    acc = correct / n_samples
    print(f"量化模型精度: {acc*100:.2f}%")
    return acc


# ─────────────────────────────────────────
# 6. 導出權重（供 FPGA 使用）
# ─────────────────────────────────────────
def export_weights(layers):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    meta = {'layers': []}

    for i, layer in enumerate(layers):
        # 權重矩陣: shape (in, out) → 存為 (out, in) 方便 FPGA 按行讀取
        W = layer['W_q'].T.astype(np.int8)    # (out, in)  int8
        b = layer['b_q'].astype(np.int32)     # (out,)     int32（累加器同 scale）

        w_path = os.path.join(WEIGHTS_DIR, f'layer{i}_weights.bin')
        b_path = os.path.join(WEIGHTS_DIR, f'layer{i}_bias.bin')

        W.tofile(w_path)
        b.tofile(b_path)

        # requant_scale 供 FPGA 重量化使用（可轉為定點移位）
        requant = float(layer['w_scale'] * layer['x_scale'] / layer['out_scale'])

        meta['layers'].append({
            'index':        i,
            'in_features':  int(layer['in_features']),
            'out_features': int(layer['out_features']),
            'w_scale':      float(layer['w_scale']),
            'x_scale':      float(layer['x_scale']),
            'out_scale':    float(layer['out_scale']),
            'requant_scale': requant,
            'w_file':       f'layer{i}_weights.bin',  # int8, (out, in)
            'b_file':       f'layer{i}_bias.bin',     # int32, (out,)
            'w_bytes':      int(W.nbytes),
            'b_bytes':      int(b.nbytes),
            'activation':   'relu' if i < len(layers) - 1 else 'none',
        })

        print(f"  Layer {i}: W={W.shape} ({W.nbytes} bytes) → {w_path}")
        print(f"          b={b.shape} ({b.nbytes} bytes) → {b_path}")

    meta_path = os.path.join(WEIGHTS_DIR, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n模型元數據 → {meta_path}")

    # 同時導出 10 個測試樣本供 FPGA 驗證
    return meta


def export_test_samples(X_test, y_test, n=10):
    samples_path = os.path.join(WEIGHTS_DIR, 'test_samples.bin')
    labels_path  = os.path.join(WEIGHTS_DIR, 'test_labels.bin')

    samples = np.clip(np.round(X_test[:n] * 127), 0, 127).astype(np.int8)
    labels  = y_test[:n].astype(np.uint8)

    samples.tofile(samples_path)
    labels.tofile(labels_path)
    print(f"測試樣本（{n} 個）→ {samples_path}")
    print(f"測試標籤           → {labels_path}")


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_mnist()

    clf = train(X_train, y_train)
    evaluate_float(clf, X_test, y_test)

    # 用 1000 個校準樣本計算激活 scale
    layers = quantize_model(clf, X_train[:1000])
    evaluate_quantized(layers, X_test, y_test)

    print("\n導出量化權重...")
    export_weights(layers)
    export_test_samples(X_test, y_test)

    print("\n完成。weights/ 目錄內容:")
    for f in sorted(os.listdir(WEIGHTS_DIR)):
        path = os.path.join(WEIGHTS_DIR, f)
        print(f"  {f:40s} {os.path.getsize(path):>8} bytes")
