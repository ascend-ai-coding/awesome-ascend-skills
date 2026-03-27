# ProteinBERT TF→PyTorch 迁移详细参考

## 权重转换映射表

### TF Pickle 权重顺序（共 145 个数组）

```
[0]   global_input_dense/kernel  (8943, 512)  → .weight = kernel.T
[1]   global_input_dense/bias    (512,)       → .bias = bias
[2]   embedding-seq-input        (26, 128)    → .weight = embeddings（不转置）
[3-25]   block1（23 个数组）
[26-48]  block2
[49-71]  block3
[72-94]  block4
[95-117] block5
[118-140] block6
[141] output-seq/kernel           (128, 26)   → .weight = kernel.T
[142] output-seq/bias             (26,)       → .bias = bias
[143] output-annotations/kernel   (512, 8943) → .weight = kernel.T
[144] output-annotations/bias     (8943,)     → .bias = bias
```

### 每个 Block 内部权重顺序（23 个数组）

```
global_to_seq_dense/kernel + bias    → .weight(.T) + .bias
narrow_conv/kernel + bias            → .weight(transpose 2,1,0) + .bias
wide_conv/kernel + bias              → .weight(transpose 2,1,0) + .bias
seq_merge1_norm/gamma + beta         → seq_norm1.weight + .bias
seq_dense/kernel + bias              → .weight(.T) + .bias
seq_merge2_norm/gamma + beta         → seq_norm2.weight + .bias
global_dense1/kernel + bias          → .weight(.T) + .bias
global_attention/Wq, Wk, Wv          → 直接复制（3 个数组）
global_merge1_norm/gamma + beta      → global_norm1.weight + .bias
global_dense2/kernel + bias          → .weight(.T) + .bias
global_merge2_norm/gamma + beta      → global_norm2.weight + .bias
```

## GlobalAttention 实现对照

### TF 原始实现
```python
# K.dot 收缩 A 的最后一维和 B 的倒数第二维
VS = K.permute_dimensions(keras.activations.gelu(K.dot(S, self.Wv)), (0, 2, 1, 3))
QX = K.tanh(K.dot(X, self.Wq))
KS = K.permute_dimensions(K.tanh(K.dot(S, self.Wk)), (0, 2, 3, 1))
Z = K.softmax(K.batch_dot(QX_batched, KS_batched) / sqrt_d_key)
Y = K.batch_dot(Z, VS_batched)
```

### PyTorch 等价实现
```python
# s 在 W 的第二个维度（index 1），对应 K.dot 的收缩语义
VS = F.gelu(torch.einsum('bls,hsv->bhlv', seq_repr, self.Wv))  # (b,h,l,v)
QX = torch.tanh(torch.einsum('bg,hgk->bhk', global_repr, self.Wq))  # (b,h,k)
KS = torch.tanh(torch.einsum('bls,hsk->bhkl', seq_repr, self.Wk))  # (b,h,k,l)
attn = F.softmax(torch.bmm(QX.unsqueeze(1), KS).squeeze(1) / sqrt_d_key, dim=-1)
Y = torch.bmm(attn.unsqueeze(1), VS).squeeze(1)
```

### 为什么 einsum('bls,hvs->...') 是错的

`K.dot(S, W)` 其中 S:(b,l,s)、W:(h,s,v)：
- 收缩 S 的 dim 2 (s) 和 W 的 dim 1 (s) → 标准矩阵乘法
- 结果: (b, l, h, v)

`einsum('bls,hvs->bhlv')` 收缩 s 与 W 的 dim 2（最后一维）：
- 等价于 `S @ W.transpose(-1,-2)`，计算完全不同
- 从第 1 个 block 开始就会产生巨大的数值偏差

随机数据验证：
```python
import numpy as np, tensorflow as tf, torch
import tensorflow.keras.backend as K

S = np.random.randn(2, 5, 128).astype(np.float32)
W = np.random.randn(4, 128, 64).astype(np.float32)

tf_result = np.transpose(K.dot(tf.constant(S), tf.constant(W)).numpy(), (0,2,1,3))
pt_correct = torch.einsum('bls,hsv->bhlv', torch.tensor(S), torch.tensor(W)).numpy()
pt_wrong = torch.einsum('bls,hvs->bhlv', torch.tensor(S), torch.tensor(W)).numpy()

print(f'正确写法 vs TF: max_diff = {np.abs(tf_result - pt_correct).max():.2e}')  # 0.00e+00
print(f'错误写法 vs TF: max_diff = {np.abs(tf_result - pt_wrong).max():.2e}')    # ~60
```

## 隐藏层拼接（微调 head 输入）

### TF get_model_with_hidden_layers_as_outputs 过滤逻辑

```python
# seq 方向: 按名字或 LayerNorm 类型过滤
seq_layers = [l for l in model.layers if
    3D形状 and (名字 in ['input-seq-encoding','dense-seq-input','output-seq']
                or isinstance(LayerNormalization))]
# “input-seq-encoding” 在模型中不存在（实际名: embedding-seq-input）
# “dense-seq-input” 也不存在
# 结果: 仅 12个LayerNorm + 1个output-seq = 12×128 + 26 = 1562 维

# global 方向:
global_layers = [l for l in model.layers if
    2D形状 and (名字 in ['input_annotations','dense-global-input','output-annotations']
                or isinstance(LayerNormalization))]
# “input_annotations”（下划线）与实际名“input-annotations”（连字符）不匹配
# “dense-global-input” 匹配 ✅
# 结果: 1×512 + 12×512 + 1×8943 = 15599 维
```

### PyTorch 等价实现
```python
# 不收集 embedding（匹配 TF 行为）
global_outputs = [h_g]  # 仅 dense-global-input
for block in blocks:
    # ... seq 路径 ...
    h_g = block.global_norm1(...)  ; global_outputs.append(h_g)
    h_g = block.global_norm2(...)  ; global_outputs.append(h_g)
out_ann = sigmoid(output_dense(h_g)); global_outputs.append(out_ann)
concat_global = torch.cat(global_outputs, dim=-1)  # (batch, 15599)
```

## 调试方法论

### 逐层对比流程

1. 创建 GPU/NPU 配对脚本，在每层打印相同格式的 mean/std/min/max
2. 用相同的 3 条测试序列、seq_len=512 运行
3. 逐行对比，第一个 mean 偏差 > 1e-3 的层即为 bug 所在

### 实际发现并修复的 Bug（按发现顺序）

| # | Bug | 影响 | 定位方式 |
|---|-----|------|----------|
| 1 | local 维度 1690 ≠ 1562 | Embedding 多拼了 128 维 | Shape 对比 |
| 2 | LayerNorm eps 1e-5 ≠ 1e-3 | 数值逐层漂移 | TF 默认值检查 |
| 3 | einsum 'bls,hvs' ≠ 'bls,hsv' | 从 block1 开始完全错误 | 逐层 diff 定位 |
| 4 | einsum 输出 'bhlk' ≠ 'bhkl' | KS 维度顺序错误 | 同上 |
| 5 | FTModel head 输入 512 ≠ 15599 | 训练提前 Early Stopping | 训练日志对比 |

## 环境注意事项

- **pyyaml**: torch_npu 运行时依赖，未列入官方 requirements
- **TF + torch_npu 冲突**: 同环境安装会导致 `aclnnEmbedding` 算子失败，必须用独立 conda 环境
- **推荐环境**: Python 3.11 + PyTorch 2.5.1 + torch_npu 2.5.1 + CANN 8.3.RC1
