---
name: ai-for-science-tf-to-pytorch
description: TensorFlow 或 Keras 模型改写到 PyTorch 的通用 Skill，适用于在华为 Ascend NPU 或其他依赖 PyTorch 生态的平台上完成层级映射、权重转换、逐层数值验证和端到端精度对比，尤其适合 ProteinBERT、DeepFRI 这类科学模型的跨框架迁移。
keywords:
  - ai-for-science
  - tensorflow
  - keras
  - pytorch
  - conversion
  - weight-mapping
---

# TensorFlow/Keras to PyTorch Conversion

A systematic approach to converting TF/Keras models to PyTorch with
numerical precision verification. Derived from real production experience
converting ProteinBERT (6-layer attention model, 16M params, 145 weight arrays)
and DeepFRI (GCN + CNN protein function prediction with CuDNNLSTM language model).

## When to Use

- Target platform has no TF support (e.g. Ascend NPU with torch_npu)
- Need PyTorch ecosystem features (torchscript, ONNX, distributed)
- Migrating research code from TF to PyTorch for maintenance
- Converting pretrained TF weights to PyTorch format

## Conversion Workflow

```
1. Analyze TF model    → Map every layer, activation, parameter shape
2. Rewrite in PyTorch  → Create equivalent nn.Module hierarchy
3. Convert weights     → Map TF arrays to PyTorch state_dict
4. Verify layer-by-layer → Compare intermediate outputs at every layer
5. Fix discrepancies   → Address framework-specific differences
6. End-to-end validate → Compare final outputs, run downstream tasks
```

## Step 1: Analyze the TF Model

Before writing any PyTorch code, fully understand the TF model:

```python
# Print all layers and shapes
for layer in model.layers:
    print(f"{layer.name:40s} type={type(layer).__name__:25s} output_shape={layer.output_shape}")

# Print all weight shapes and names
for i, var in enumerate(model.variables):
    print(f"[{i:3d}] {var.name:50s} shape={var.shape}")
```

Key things to document:
- Layer names and their order (may differ from code order)
- Weight shapes and dtypes
- Activation functions used (and whether they're fused into layers)
- Custom layers (especially attention mechanisms)
- Default hyperparameters (epsilon, momentum, etc.)

## Step 2: Layer-by-Layer Conversion Rules

### Dense / Linear

| Property | TF/Keras | PyTorch |
|----------|----------|--------|
| Class | `keras.layers.Dense` | `nn.Linear` |
| Kernel shape | `(in_features, out_features)` | `(out_features, in_features)` |
| Activation | Fused: `Dense(n, activation='gelu')` | Separate: `nn.Linear(n)` + `F.gelu()` |
| Weight conversion | `kernel` | `.weight = kernel.T` ← **must transpose** |
| Bias conversion | `bias` | `.bias = bias` (direct copy) |

### Embedding

| Property | TF/Keras | PyTorch |
|----------|----------|--------|
| Class | `keras.layers.Embedding` | `nn.Embedding` |
| Weight shape | `(vocab_size, embed_dim)` | `(vocab_size, embed_dim)` |
| Weight conversion | `embeddings` | `.weight = embeddings` (direct copy, **no transpose**) |

### Conv1D

| Property | TF/Keras | PyTorch |
|----------|----------|--------|
| Class | `keras.layers.Conv1D` | `nn.Conv1d` |
| Input format | `(batch, length, channels)` | `(batch, channels, length)` |
| Kernel shape | `(kernel_size, in_ch, out_ch)` | `(out_ch, in_ch, kernel_size)` |
| Weight conversion | `kernel` | `.weight = np.transpose(kernel, (2,1,0))` |
| Padding "same" | Built-in, may use asymmetric padding | `padding='same'` (PyTorch 1.9+) |

**Important**: When using Conv1d in PyTorch, transpose the input before and after:
```python
# TF: hidden_seq shape (batch, length, channels) - Conv1D operates directly
# PyTorch:
seq_t = hidden_seq.transpose(1, 2)        # (batch, channels, length)
out = F.gelu(self.conv(seq_t)).transpose(1, 2)  # back to (batch, length, channels)
```

### LayerNormalization / LayerNorm

| Property | TF/Keras | PyTorch |
|----------|----------|--------|
| Class | `keras.layers.LayerNormalization` | `nn.LayerNorm` |
| **Default epsilon** | **1e-3 (0.001)** | **1e-5 (0.00001)** |
| Weight names | `gamma`, `beta` | `weight`, `bias` |
| Weight conversion | Direct copy | Direct copy |

> **⚠️ CRITICAL**: Always set `nn.LayerNorm(dim, eps=1e-3)` to match TF default.
> This 100x epsilon difference propagates through deep networks and causes
> massive numerical divergence. This is the #1 most common silent bug.

### BatchNormalization / BatchNorm

| Property | TF/Keras | PyTorch |
|----------|----------|--------|
| Class | `keras.layers.BatchNormalization` | `nn.BatchNorm1d` / `nn.BatchNorm2d` |
| **Default epsilon** | **1e-3 (0.001)** | **1e-5 (0.00001)** |
| Default momentum | 0.99 (as `momentum`) | 0.1 (as `1 - momentum`) |
| Weight names | `gamma`, `beta`, `moving_mean`, `moving_variance` | `weight`, `bias`, `running_mean`, `running_var` |
| Weight conversion | Direct copy | Direct copy |

> **⚠️ CRITICAL**: Same 100x epsilon trap as LayerNorm. Always set
> `nn.BatchNorm1d(dim, eps=1e-3)`. When BN `moving_variance` contains
> near-zero values (common after training), the wrong epsilon causes
> division-by-near-zero → output explodes → softmax saturates to 0/1
> → model outputs all zeros. This was the #1 bug in DeepFRI CNN conversion.

### CuDNNLSTM / LSTM

TF `CuDNNLSTM` is a GPU-optimized LSTM with different weight layout than
standard `keras.layers.LSTM`. It's commonly found in older TF models.

| Property | TF CuDNNLSTM | TF Keras LSTM | PyTorch `nn.LSTM` |
|----------|-------------|---------------|-------------------|
| Kernel shape | `(input, 4*H)` | `(input, 4*H)` | `weight_ih: (4*H, input)` |
| Recurrent kernel | `(H, 4*H)` | `(H, 4*H)` | `weight_hh: (4*H, H)` |
| **Bias shape** | **`(8*H,)`** ← two groups | `(4*H,)` single | `bias_ih: (4*H,)` + `bias_hh: (4*H,)` |
| Recurrent activation | sigmoid (fixed) | configurable | sigmoid (default) |

**CuDNNLSTM weight conversion:**
```python
# Kernel and recurrent kernel: just transpose
pt_weight_ih = tf_kernel.T           # (input, 4*H) -> (4*H, input)
pt_weight_hh = tf_recurrent_kernel.T # (H, 4*H) -> (4*H, H)

# Bias: CuDNNLSTM stores TWO bias groups concatenated
# bias[:4*H] = input-hidden bias,  bias[4*H:] = hidden-hidden bias
pt_bias_ih = tf_bias[:4 * hidden_dim]
pt_bias_hh = tf_bias[4 * hidden_dim:]
```

**Standard Keras LSTM weight conversion:**
```python
pt_weight_ih = tf_kernel.T
pt_weight_hh = tf_recurrent_kernel.T
pt_bias_ih = tf_bias                       # single group (4*H,)
pt_bias_hh = np.zeros(4 * hidden_dim)     # PyTorch needs both; set hh to zero
```

> **⚠️ WARNING**: CuDNNLSTM models are **GPU-only** weights. When converted to
> PyTorch `nn.LSTM`, the output is mathematically equivalent per-step, but
> accumulated floating-point differences over long sequences can be amplified
> by downstream layers (e.g. graph convolutions). If the original model has
> both GPU (CuDNNLSTM) and CPU (standard LSTM) versions, prefer the CPU version
> for non-CUDA deployments (e.g. Ascend NPU).

### GELU Activation

| Property | TF/Keras (>=2.4) | PyTorch |
|----------|-----------------|--------|
| Default | Exact: `x * Φ(x)` | Exact: `x * Φ(x)` |
| Approximate | N/A | `F.gelu(x, approximate="tanh")` |
| Match? | ✅ Both use exact by default | Use `F.gelu()` without approximate |

## Step 3: K.dot and High-Dimensional Tensor Operations

This is the **most dangerous** part of conversion. TF's `K.dot` has different
semantics than any single PyTorch operation.

### K.dot Behavior

`K.dot(A, B)` contracts **A's last dimension** with **B's second-to-last dimension**
(standard matrix multiplication semantics, generalized to higher dimensions).

```python
# Example: K.dot(S, W) where S:(batch, length, d_in), W:(n_heads, d_in, d_out)
# Contracts S[..., d_in] with W[..., d_in, :] -> result: (batch, length, n_heads, d_out)
```

### ⚠️ Common einsum Mistakes

```python
# W shape: (n_heads, d_in, d_out)  -  d_in is the SECOND dimension (index 1)

# ❌ WRONG - contracts with W's LAST dimension
VS = torch.einsum('bls,hvs->bhlv', S, W)  # s matches W's dim 2 (last)

# ✅ CORRECT - contracts with W's SECOND dimension
VS = torch.einsum('bls,hsv->bhlv', S, W)  # s matches W's dim 1 (second-to-last)
```

### Verification Pattern

Always verify K.dot equivalence with random data:
```python
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import torch

S = np.random.randn(2, 5, 128).astype(np.float32)
W = np.random.randn(4, 128, 64).astype(np.float32)

tf_result = K.dot(tf.constant(S), tf.constant(W)).numpy()   # (2, 5, 4, 64)
pt_result = torch.einsum('bls,hsv->blhv', torch.tensor(S), torch.tensor(W)).numpy()

np.testing.assert_allclose(tf_result, pt_result, atol=1e-6)  # must pass
```

### K.permute_dimensions Mapping

After K.dot, TF typically reorders dimensions with `K.permute_dimensions`.
Map the full chain:

```python
# TF: result = K.permute_dimensions(K.dot(S, W), (0, 2, 1, 3))
#     K.dot:    (batch, length, heads, dim)  
#     permute:  (batch, heads, length, dim)
#
# PyTorch equivalent (single einsum):
result = torch.einsum('bls,hsv->bhlv', S, W)  # directly (batch, heads, length, dim)
```

## Step 4: Weight Mapping

### Determine Weight Order

TF pickle/checkpoint weight order may NOT match code definition order.
Always inspect the actual order first:

```python
import pickle
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
for i, w in enumerate(data['weights']):
    print(f"[{i:3d}] shape={str(w.shape):25s}")
```

Then map each weight to the corresponding PyTorch parameter by shape matching.

### Weight Conversion Checklist

For each weight array:
- [ ] Dense kernel: **transpose** `.T`
- [ ] Dense bias: direct copy
- [ ] Conv1D kernel: **transpose** `np.transpose(w, (2,1,0))`
- [ ] Conv1D bias: direct copy
- [ ] LayerNorm gamma/beta: direct copy, **set eps=1e-3**
- [ ] BatchNorm gamma/beta/moving_mean/moving_var: direct copy, **set eps=1e-3**
- [ ] Embedding: direct copy (no transpose)
- [ ] CuDNNLSTM kernel/recurrent_kernel: **transpose** `.T`
- [ ] CuDNNLSTM bias `(8*H,)`: **split** into `bias_ih[:4H]` + `bias_hh[4H:]`
- [ ] Standard LSTM kernel/recurrent_kernel: **transpose** `.T`
- [ ] Standard LSTM bias `(4*H,)`: copy to `bias_ih`, set `bias_hh` to zeros
- [ ] Custom attention weights (Wq/Wk/Wv): usually direct copy, but verify shape

## Step 5: Layer-by-Layer Verification

This is the most important step. Never skip it.

### Create Debug Scripts

Create paired debug scripts that print identical statistics at every layer:

```python
# Both GPU (TF) and NPU/CPU (PyTorch) scripts should print:
for layer_name, output in layer_outputs:
    print(f"{layer_name:45s} shape={str(output.shape):25s} "
          f"mean={output.mean():.6f} std={output.std():.6f} "
          f"min={output.min():.6f} max={output.max():.6f}")
```

### Identify Divergence Point

Compare outputs layer by layer. The FIRST layer where mean/std diverge
significantly (>1e-3) reveals the bug location.

Typical pattern:
```
embedding:              GPU mean=0.000829  NPU mean=0.000829  ✅ match
dense-global-input:     GPU mean=-0.005728 NPU mean=-0.005728 ✅ match
global-merge1-norm-b1:  GPU mean=-0.031465 NPU mean=0.017317  ❌ DIVERGED
```
→ Bug is between `dense-global-input` and `global-merge1-norm-block1`.

### Common Divergence Sources (ranked by frequency)

1. **Normalization epsilon** (LayerNorm/BatchNorm 1e-3 vs 1e-5) - causes output explosion or gradual drift
2. **einsum dimension mismatch** (K.dot semantics) - causes total output corruption
3. **CuDNNLSTM bias splitting** (8*H bias not split into ih+hh) - causes wrong LSTM output
4. **Conv1D weight transpose wrong** - causes garbage output from convolutions
5. **Dense weight not transposed** - causes completely wrong linear projections
6. **Activation variant mismatch** (approximate vs exact GELU) - minor drift

## Step 6: Fine-tuning Verification

When converting fine-tuning code, also verify:

### Hidden Layer Concatenation

If TF code uses `get_model_with_hidden_layers_as_outputs` or similar to
concatenate hidden layers for the fine-tuning head:

1. Check exactly WHICH layers TF collects (filter by name + type)
2. Some layer names in filters may not exist in the model (dead filters)
3. Match the concatenated dimension exactly

### Optimizer Differences

| Property | TF Adam | PyTorch Adam |
|----------|---------|-------------|
| Default epsilon | 1e-7 | 1e-8 |
| Weight decay | Separate `AdamW` | `weight_decay` param |
| LR schedule | Callbacks | `lr_scheduler` |

These cause minor training dynamics differences but should not affect
final model quality significantly.

## Quick Reference: Conversion Cheat Sheet

```python
# Dense
pt_weight = tf_kernel.T
pt_bias = tf_bias

# Conv1D  
pt_weight = np.transpose(tf_kernel, (2, 1, 0))
pt_bias = tf_bias

# LayerNorm
nn.LayerNorm(dim, eps=1e-3)  # MUST set eps to match TF
pt_weight = tf_gamma
pt_bias = tf_beta

# BatchNorm
nn.BatchNorm1d(dim, eps=1e-3)  # MUST set eps to match TF
pt_weight = tf_gamma
pt_bias = tf_beta
pt_running_mean = tf_moving_mean
pt_running_var = tf_moving_variance

# Embedding
pt_weight = tf_embeddings  # no transpose

# CuDNNLSTM (bias shape 8*H)
pt_weight_ih = tf_kernel.T
pt_weight_hh = tf_recurrent_kernel.T
pt_bias_ih = tf_bias[:4*H]
pt_bias_hh = tf_bias[4*H:]

# Standard Keras LSTM (bias shape 4*H)
pt_weight_ih = tf_kernel.T
pt_weight_hh = tf_recurrent_kernel.T
pt_bias_ih = tf_bias
pt_bias_hh = np.zeros(4*H)

# K.dot(S[b,l,s], W[h,s,v]) -> einsum('bls,hsv->blhv', S, W)
# K.dot contracts last dim of A with second-to-last dim of B
```

For detailed weight mapping examples and real conversion scripts,
read `references/weight_conversion_example.md`.

## 配套脚本

- 数值对比辅助脚本：`python scripts/compare_arrays.py reference.npz candidate.npz --atol 1e-5 --rtol 1e-4`

## 参考资料

- 跨框架数值验证清单：[`references/numerical-validation.md`](references/numerical-validation.md)
