# npu_format_cast API 详解

本文档提供 `torch_npu.npu_format_cast` 系列 API 的完整参考，用于 IDE 代码提示与补全。

## API 签名与参数

### torch_npu.npu_format_cast

```python
torch_npu.npu_format_cast(tensor, acl_format, customize_dtype=None)
```

- `tensor`：NPU 上的 `torch.Tensor`（需先 `.npu()`）
- `acl_format`：目标存储格式，可为 **`int`** 或 **`torch_npu.Format`** 枚举成员
- `customize_dtype`：可选，用于 ONNX 等场景的自定义 dtype
- 返回：新张量（不修改原张量）

### torch_npu.npu_format_cast_

```python
torch_npu.npu_format_cast_(tensor, acl_format)
```

- 同上，但为 **in-place** 版本，直接修改 `tensor` 的格式

### torch_npu.get_npu_format

```python
torch_npu.get_npu_format(tensor)
```

- 返回张量当前 NPU 存储格式（`torch_npu.Format` 或整型）

## Format 枚举（torch_npu.Format）

### 常用枚举值

| 枚举名 | 值 | 常见用途 |
|--------|----|----------|
| `Format.NCHW` | 0 | 默认 4D 卷积布局 |
| `Format.NHWC` | 1 | 通道在后的 4D 布局 |
| `Format.ND` | 2 | 通用 ND 布局 |
| `Format.NC1HWC0` | 3 | Conv/BatchNorm 等算子常用 |
| `Format.FRACTAL_Z` | 4 | 3D 卷积等 |
| `Format.FRACTAL_NZ` | 29 | 线性/矩阵乘、Attention 权重等 |
| `Format.NDC1HWC0` | 32 | 5D |
| `Format.FRACTAL_Z_3D` | 33 | 3D 卷积 |
| `Format.UNDEFINED` | -1 | 未定义 |

### 扩展枚举值

| 枚举名 | 值 | 用途 |
|--------|----|----|
| `NC1HWC0_C04` | 12 | 特殊 4D 布局 |
| `HWCN` | 16 | 通道在后的布局 |
| `NDHWC` | 27 | 5D 通道在后 |
| `NCDHW` | 30 | 5D 通道在前 |
| `NC` | 35 | 2D |
| `NCL` | 47 | 序列布局 |
| `FRACTAL_NZ_C0_*` | 50-54 | 分块矩阵 |

## 代码提示与补全规则

### 1. 补全第二参数

当用户输入 `torch_npu.npu_format_cast(x, ` 时，提示 `acl_format` 可选为 `int` 或 `torch_npu.Format.xxx`，并列出常用枚举（如 `Format.NCHW`、`Format.NHWC`、`Format.FRACTAL_NZ`、`Format.NC1HWC0`）。

### 2. 补全 Format 枚举

当用户输入 `torch_npu.Format.` 时，提示上述枚举成员列表。

### 3. 配对使用

若代码中已有 `get_npu_format(t)`，在需要转成相同格式时，可提示：
```python
torch_npu.npu_format_cast(other, torch_npu.get_npu_format(t))
```

### 4. 常见场景示例

**线性层权重量子化/迁移到 NPU：**
```python
torch_npu.npu_format_cast(weight.npu(), 29)  # FRACTAL_NZ
```

**与参数格式一致的梯度：**
```python
torch_npu.npu_format_cast(p.grad, torch_npu.get_npu_format(p))
```

**模块迁移时 BN/Conv 的 NC1HWC0：**
```python
torch_npu.npu_format_cast(tensor, 3)  # 或 Format.NC1HWC0
```

## 相关文档

- [API 索引总览](./api_index.md)
- [Flash Attention 指南](./flash_attention_guide.md)
- [量化算子指南](./quantization_guide.md)
