# 数据类型支持矩阵与约束

## 概述

Triton 支持丰富的数据类型，包括整数类型（int1/8/16/32/64, uint8/16/32/64）、浮点类型（float16/32/64, bfloat16）以及多种 FP8 格式（fp8e4nv, fp8e4b15, fp8e4b8, fp8e5, fp8e5b16）。在昇腾 NPU 上，由于硬件架构差异，部分数据类型的支持状态与 GPU 不同：uint8/16/32/64 和 fp64 在多数操作中不支持，FP8 格式的支持程度因操作类型而异。

理解 NPU 上的数据类型支持矩阵、对齐要求和隐式类型提升规则，是编写正确且高性能 Triton 算子的基础。选择合适的数据类型不仅影响计算精度，还直接影响内存对齐、片上存储利用率和计算单元的执行效率。

**关键词**：数据类型、dtype、FP8、bfloat16、float16、类型提升、integer_promote、computation_type、对齐、Cube 计算

## 关键概念

| 概念 | 说明 | NPU 特殊约束 |
|------|------|-------------|
| 数据类型 (dtype) | Triton 中张量的元素类型 | 部分 GPU 类型在 NPU 不支持 |
| 隐式类型提升 | 二元运算中自动提升操作数类型 | 遵循 C 语言标准转换规则 |
| 对齐要求 | 数据访问的内存对齐约束 | VV 算子 32B，CV 算子 512B |
| FP8 格式 | 8-bit 浮点数的多种编码格式 | NPU 支持多种 FP8，但 Cube 计算中受限 |
| BF16 vs FP16 | 两种 16-bit 浮点格式 | BF16 精度低但范围大，FP16 精度高但范围小 |

## 详细内容

### 1. Triton 支持的所有数据类型

#### 1.1 数据类型完整列表

源码定义（[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L288-L293)）：

```python
class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']
```

#### 1.2 数据类型详细规格

| 类型 | Triton 名称 | 位数 | 尾数位 | 指数偏移 | 类别 |
|------|------------|------|--------|----------|------|
| bool | `tl.int1` | 1 | - | - | 无符号整数 |
| int8 | `tl.int8` | 8 | - | - | 有符号整数 |
| int16 | `tl.int16` | 16 | - | - | 有符号整数 |
| int32 | `tl.int32` | 32 | - | - | 有符号整数 |
| int64 | `tl.int64` | 64 | - | - | 有符号整数 |
| uint8 | `tl.uint8` | 8 | - | - | 无符号整数 |
| uint16 | `tl.uint16` | 16 | - | - | 无符号整数 |
| uint32 | `tl.uint32` | 32 | - | - | 无符号整数 |
| uint64 | `tl.uint64` | 64 | - | - | 无符号整数 |
| fp8e4nv | `tl.float8e4nv` | 8 | 3 (E4M3) | 7 | 浮点 |
| fp8e4b15 | `tl.float8e4b15` | 8 | 3 (E4M3) | 15 | 浮点 |
| fp8e4b8 | `tl.float8e4b8` | 8 | 3 (E4M3) | 8 | 浮点 |
| fp8e5 | `tl.float8e5` | 8 | 2 (E5M2) | 15 | 浮点 |
| fp8e5b16 | `tl.float8e5b16` | 8 | 2 (E5M2) | 16 | 浮点 |
| float16 | `tl.float16` | 16 | 10 | 15 | 浮点 |
| bfloat16 | `tl.bfloat16` | 16 | 7 | 127 | 浮点 |
| float32 | `tl.float32` | 32 | 23 | 127 | 浮点 |
| float64 | `tl.float64` | 64 | 52 | 1023 | 浮点 |

#### 1.3 FP8 格式详解

| FP8 格式 | 编码 | 尾数位 | 指数位 | 指数偏移 | 典型用途 |
|----------|------|--------|--------|----------|----------|
| fp8e4nv | E4M3FN | 3 | 4 | 7 | NVIDIA 标准 FP8，前向传播 |
| fp8e4b15 | E4M3 | 3 | 4 | 15 | 华为昇腾格式 |
| fp8e4b8 | E4M3 | 3 | 4 | 8 | 中间偏移格式 |
| fp8e5 | E5M2 | 2 | 5 | 15 | 反向传播梯度 |
| fp8e5b16 | E5M2 | 2 | 5 | 16 | 扩展范围格式 |

Ascend 后端支持的 FP8 格式（[compiler.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L835)）：

```python
supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")
deprecated_fp8_dtypes: Tuple[str] = ()
```

### 2. NPU 上各数据类型的支持状态

#### 2.1 综合支持矩阵

| 数据类型 | tl.load/tl.store | tl.dot | 数学运算 | 类型转换 | NPU 支持状态 |
|----------|-----------------|--------|----------|----------|-------------|
| int1 (bool) | 支持 | 支持(dot输入) | 部分支持 | 支持 | 部分支持 |
| int8 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| int16 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| int32 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| int64 | 支持 | 不支持 | 部分支持 | 支持 | 部分支持 |
| uint8 | 不支持 | 不支持 | 部分支持 | 支持 | 不支持 |
| uint16 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| uint32 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| uint64 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| float16 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| bfloat16 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| float32 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| float64 | 不支持 | 不支持 | 部分支持 | 不支持 | 不支持 |
| fp8e4nv | 部分支持 | 不支持（A2/A3）/ dot_scaled（910_95） | 不支持 | 支持 | 部分支持（910_95 支持转换和 dot_scaled） |
| fp8e4b15 | 部分支持 | 不支持（A2/A3）/ dot_scaled（910_95） | 不支持 | 支持 | 部分支持（910_95 支持转换和 dot_scaled） |
| fp8e5 | 部分支持 | 不支持（A2/A3）/ dot_scaled（910_95） | 不支持 | 支持 | 部分支持（910_95 支持转换和 dot_scaled） |

#### 2.2 tl.dot 支持的数据类型

| 输入类型 | GPU | NPU (Ascend) | 说明 |
|----------|-----|-------------|------|
| int8 | 支持 | 支持 | 矩阵乘法输入 |
| int16 | 支持 | 支持 | 矩阵乘法输入 |
| int32 | 支持 | 支持 | 矩阵乘法输入 |
| float16 | 支持 | 支持 | 矩阵乘法输入 |
| bfloat16 | 支持 | 支持 | 矩阵乘法输入 |
| float32 | 支持 | 支持 | 矩阵乘法输入 |
| bool | 支持 | 支持 | Triton 内部转为 int8 |
| uint8/16/32/64 | 支持 | 不支持 | 硬件限制 |
| float64 | 支持 | 不支持 | 硬件限制 |

**tl.dot 特殊限制**：
- acc 不支持 FP16，硬件默认使用 FP32 累加
- `max_num_imprecise_acc` 暂不支持
- `out_dtype` 对比 GPU 缺乏 int8 和 FP16 的类型支持
- NPU 默认 `input_precision` 为 `"ieee"`，可选 `"hf32"`

#### 2.3 tl.load/tl.store 支持的数据类型

| 数据类型 | GPU | NPU (Ascend) |
|----------|-----|-------------|
| int8 | 支持 | 支持 |
| int16 | 支持 | 支持 |
| int32 | 支持 | 支持 |
| int64 | 支持 | 支持 |
| uint8 | 支持 | 不支持 |
| uint16 | 支持 | 不支持 |
| uint32 | 支持 | 不支持 |
| uint64 | 支持 | 不支持 |
| float16 | 支持 | 支持 |
| float32 | 支持 | 支持 |
| float64 | 支持 | 不支持 |
| bfloat16 | 支持 | 支持 |
| bool | 支持 | 支持 |

#### 2.4 类型转换 (cast) 支持矩阵

| 数据类型 | GPU | NPU (Ascend) |
|----------|-----|-------------|
| int8 | 支持 | 支持 |
| int16 | 支持 | 支持 |
| int32 | 支持 | 支持 |
| int64 | 支持 | 支持 |
| uint8 | 支持 | 支持 |
| uint16 | 支持 | 不支持 |
| uint32 | 支持 | 不支持 |
| uint64 | 支持 | 不支持 |
| float16 | 支持 | 支持 |
| float32 | 支持 | 支持 |
| bfloat16 | 支持 | 支持 |
| float8e4 | 支持 | 部分支持（910_95 支持） |
| float8e5 | 支持 | 部分支持（910_95 支持） |
| float64 | 支持 | 不支持 |
| bool | 支持 | 支持 |

### 3. 数据类型对齐要求

#### 3.1 内存对齐约束

| 场景 | 对齐要求 | 说明 |
|------|----------|------|
| VV 类算子（纯 Vector） | 尾轴 32B 对齐 | UB 访问要求 |
| CV 类算子（Cube+Vector） | 尾轴 512B 对齐 | Cube 计算要求 |
| UB 存储 | 32B 对齐 | 硬件要求 |
| L1 存储 | 32B 对齐 | 硬件要求 |
| L0A/L0B/L0C | 512B 对齐 | 硬件要求 |

#### 3.2 对齐计算示例

```python
# float16 (2 bytes) 的对齐计算
# VV 算子: 尾轴元素数 >= 32 / 2 = 16
# CV 算子: 尾轴元素数 >= 512 / 2 = 256

# float32 (4 bytes) 的对齐计算
# VV 算子: 尾轴元素数 >= 32 / 4 = 8
# CV 算子: 尾轴元素数 >= 512 / 4 = 128

# bfloat16 (2 bytes) 的对齐计算
# VV 算子: 尾轴元素数 >= 32 / 2 = 16
# CV 算子: 尾轴元素数 >= 512 / 2 = 256
```

#### 3.3 尾轴不对齐的性能影响

当尾轴大小不满足对齐要求时，硬件会自动补齐，导致：
- 额外的内存占用
- 无效数据的搬运和计算
- 性能明显恶化

```python
# 不推荐: shape (2048, 3), bfloat16
# 尾轴 3 * 2 = 6 bytes, 不满足 32B 对齐
# 硬件会补齐到 16 * 2 = 32B, 浪费 13 个元素的空间

# 推荐: 使用借轴转置技巧
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3))
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
```

### 4. 隐式类型提升规则

#### 4.1 integer_promote 规则

源码定义（[semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L46-L58)）：

```python
def integer_promote_impl(a_ty, b_ty):
    a_rank = a_ty.int_bitwidth
    b_rank = b_ty.int_bitwidth
    a_sn = a_ty.int_signedness
    b_sn = b_ty.int_signedness

    if a_sn == b_sn:
        return a_ty if a_rank > b_rank else b_ty
    elif a_sn == SIGNEDNESS.UNSIGNED:
        return a_ty if a_rank >= b_rank else b_ty
    elif b_sn == SIGNEDNESS.UNSIGNED:
        return b_ty if b_rank >= a_rank else b_ty
```

**规则总结**：

| 操作数 A | 操作数 B | 提升结果 | 说明 |
|----------|----------|----------|------|
| int8 | int16 | int16 | 同符号，取宽类型 |
| int32 | int8 | int32 | 同符号，取宽类型 |
| int8 | uint8 | int16 | 有符号 vs 无符号，取更宽类型 |
| int16 | uint8 | int16 | 有符号宽度 >= 无符号宽度，取有符号类型 |
| int8 | uint16 | uint16 | 无符号宽度 > 有符号宽度，取无符号类型 |

#### 4.2 computation_type 规则

源码定义（[semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L61-L100)）：

```python
def computation_type_impl(a_ty, a_is_scalar, b_ty, b_is_scalar, div_or_mod):
    # 0) 标量不参与提升（如果类型级别更低）
    if a_is_scalar != b_is_scalar:
        scalar_ty, tensor_ty = (a_ty, b_ty) if a_is_scalar else (b_ty, a_ty)
        if scalar_ty.kind().value <= tensor_ty.kind().value:
            if div_or_mod and (tensor_ty in (float16, bfloat16)):
                return float32
            return tensor_ty

    # 1) 有 fp64 则提升为 fp64
    if a_ty.is_fp64() or b_ty.is_fp64():
        return float64

    # 2) 有 fp32 则提升为 fp32
    if a_ty.is_fp32() or b_ty.is_fp32():
        return float32

    # 3) 有 fp16 则提升为 fp16（除法/取模提升为 fp32）
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return float32
        return float16

    # 4) bf16 仅在两边都是 bf16 时保持，否则提升为 fp32
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return bfloat16
        return float32

    # 5) 不同的 fp8 类型提升为 fp16
    if a_ty.is_fp8() and b_ty.is_fp8():
        return a_ty if a_ty == b_ty else float16

    # 6) 整数类型使用 integer_promote
    return integer_promote_impl(a_ty, b_ty)
```

**规则总结**：

| 操作数 A | 操作数 B | 运算 | 提升结果 | 说明 |
|----------|----------|------|----------|------|
| float16 | float16 | +,-,* | float16 | 同类型保持 |
| float16 | float16 | /,% | float32 | 除法/取模提升精度 |
| bfloat16 | bfloat16 | +,-,* | bfloat16 | 同类型保持 |
| bfloat16 | float16 | +,-,* | float32 | 混合提升为 fp32 |
| bfloat16 | bfloat16 | /,% | float32 | 除法/取模提升精度 |
| float32 | float16 | 任意 | float32 | 有 fp32 则提升 |
| fp8e4nv | fp8e5 | 任意 | float16 | 不同 fp8 提升为 fp16 |
| int8 | int32 | 任意 | int32 | 整数类型提升 |

### 5. FP8 在 Cube 计算中的使用

#### 5.1 FP8 支持现状

在 Ascend NPU 上，FP8 数据类型的支持有以下特点：

- **类型转换**：FP8 与 fp16/bf16/fp32 之间的类型转换支持
- **tl.dot 输入**：当前 Ascend A2/A3 系列**不支持** FP8 作为 `tl.dot` 的输入（硬件限制）
- **dot_scaled**：A2/A3 系列 `tl.dot_scaled` 不支持 FP4 和 FP8 格式（硬件限制）；910_95 系列支持 FP8 的 dot_scaled

```python
# FP8 类型转换示例
@triton.jit
def fp8_cast_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 从 GM 加载 fp8 数据并转换为 fp16
    x_fp8 = tl.load(x_ptr + offsets, mask=mask)
    x_fp16 = x_fp8.to(tl.float16)
    result = x_fp16 * 2.0
    tl.store(out_ptr + offsets, result, mask=mask)
```

#### 5.2 FP8 在 910_95 系列上的支持

910_95 系列对 FP8 的支持有所增强：

```python
# 910_95 系列上的 FP8 转换
from triton.backends.ascend.language.cann.extension.vec_ops import ascend_cast_impl

# FP8 <-> 浮点类型转换通过 create_fp_to_fp 实现
# fp8e4b15 使用 convert_custom_types 路径
```

### 6. bfloat16 与 float16 的性能差异

#### 6.1 格式对比

| 特性 | float16 (fp16) | bfloat16 (bf16) |
|------|----------------|-----------------|
| 总位数 | 16 | 16 |
| 尾数位 | 10 | 7 |
| 指数位 | 5 | 8 |
| 指数偏移 | 15 | 127 |
| 表示范围 | ~5.96e-8 ~ 65504 | ~1.18e-38 ~ 3.39e+38 |
| 精度 | 较高（约 3.3 位十进制） | 较低（约 2.1 位十进制） |
| 与 fp32 转换 | 需要调整指数和尾数 | 仅截断/扩展尾数 |

#### 6.2 选择建议

| 场景 | 推荐类型 | 原因 |
|------|----------|------|
| 矩阵乘法输入 | bfloat16 | 范围与 fp32 一致，不易溢出 |
| 需要高精度的累加 | float32 | 累加器默认使用 fp32 |
| 推理场景 | float16 / bfloat16 | 根据模型精度要求选择 |
| 训练前向传播 | bfloat16 | 范围大，梯度稳定 |
| 训练反向传播 | float32 或 bfloat16 | 梯度计算需要足够精度 |
| 与 fp32 混合运算 | bfloat16 | bf16 与 fp32 转换开销更低 |

#### 6.3 性能对比

```python
# bfloat16 与 float16 在 NPU 上的性能差异
# 1. tl.dot: 两者性能相当，都是 Cube 单元原生支持
# 2. 向量运算: 两者性能相当，都是 Vector 单元原生支持
# 3. 类型转换: bfloat16 <-> float32 转换更高效（仅截断/扩展尾数）
# 4. 精度: float16 精度更高，bfloat16 范围更大

# 注意: bfloat16 与 float16 混合运算会提升为 float32
@triton.jit
def mixed_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)  # bfloat16
    b = tl.load(b_ptr + offsets, mask=mask)  # float16
    # a + b 会自动提升为 float32
    result = a + b  # float32
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)
```

### 7. 代码示例：数据类型选择最佳实践

#### 7.1 推荐的数据类型选择策略

```python
import torch
import torch_npu
import triton
import triton.language as tl

# 策略一: 矩阵乘法使用 bfloat16 输入 + float32 累加
@triton.jit
def matmul_bf16_kernel(A, B, C, M, N, K,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器使用 float32 保证精度
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # 输入使用 bfloat16，范围大不易溢出
        a = tl.load(A + offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(B + (offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        # tl.dot 自动将 bf16 输入累加到 fp32
        acc += tl.dot(a, b)

    # 输出转回 bfloat16
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

#### 7.2 避免不支持的类型

```python
# 错误: 使用 uint8 类型
# x = tl.load(x_ptr + offsets, mask=mask)  # x_ptr 指向 uint8 数据 -> 不支持!

# 正确: 使用 int8 代替 uint8
x_int8 = tl.load(x_ptr_int8 + offsets, mask=mask)

# 如果确实需要处理 uint8 数据，先在 host 端转换
# x_npu = x_original.to(torch.int8).npu()
```

#### 7.3 FP8 类型转换最佳实践

```python
@triton.jit
def fp8_inference_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 加载 FP8 数据
    x_fp8 = tl.load(x_ptr + offsets, mask=mask)

    # 转换为 float16 进行计算（FP8 不能直接参与大多数运算）
    x_fp16 = x_fp8.to(tl.float16)

    # 在 float16 精度下计算
    result = x_fp16 * 2.0 + 1.0

    # 转换回 FP8 存储
    tl.store(out_ptr + offsets, result.to(x_fp8.dtype), mask=mask)
```

#### 7.4 处理 bool 类型的特殊规则

```python
@triton.jit
def bool_handling_kernel(mask_ptr, data_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # bool (int1) 在 Triton 内部会转为 int8 进行运算
    cond = tl.load(mask_ptr + offsets, mask=mask).to(tl.int1)
    data = tl.load(data_ptr + offsets, mask=mask)

    # 使用 tl.where 进行条件选择
    result = tl.where(cond, data, tl.zeros_like(data))
    tl.store(out_ptr + offsets, result, mask=mask)
```

#### 7.5 int64 在 NPU 上的使用

```python
from triton.backends.ascend.language.cann.extension.core import int64 as al_int64

@triton.jit
def large_offset_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 大偏移量需要 int64，使用 al.int64() 包装
    large_offset = al_int64(pid) * al_int64(BLOCK_SIZE)
    full_offsets = large_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = full_offsets < N

    data = tl.load(data_ptr + full_offsets, mask=mask)
    tl.store(data_ptr + full_offsets, data * 2, mask=mask)
```

## NPU 适配要点

1. **避免使用不支持的类型**：uint8/16/32/64 和 fp64 在多数操作中不支持，使用 int8/16/32 替代 uint 类型
2. **矩阵乘法推荐 bfloat16**：bf16 范围与 fp32 一致，不易溢出，且与 fp32 转换开销低
3. **累加器使用 float32**：`tl.dot` 的累加器默认使用 fp32，不要使用 fp16 作为 acc
4. **FP8 仅用于存储**：当前 FP8 主要用于数据存储和传输，计算前需转换为 fp16/bf16/fp32
5. **注意对齐要求**：VV 算子 32B 对齐，CV 算子 512B 对齐，尾轴不满足时自动补齐影响性能
6. **bool 类型转为 int8**：Triton 内部将 bool 转为 int8 运算，int8 会占用更大片上空间，注意 UB 溢出
7. **所有 tensor 总和不超过 UB 限制**：A2/A3 系列开启 double buffer 时不超过 96KB，910_95 系列开启 double buffer 时不超过 128KB

## 常见问题

**Q1: 为什么 uint8 在 NPU 上不支持？**

A: 昇腾 NPU 的硬件指令集不原生支持 uint8 类型的内存加载和计算。如果需要处理 uint8 数据，建议在 host 端先转换为 int8 或 int16，再传入 Triton kernel。

**Q2: bfloat16 和 float16 应该选哪个？**

A: 推荐优先使用 bfloat16，原因：1) bf16 的数值范围与 fp32 一致，不易溢出；2) bf16 与 fp32 的转换仅需截断/扩展尾数，开销更低；3) 在矩阵乘法场景中两者性能相当。如果对精度要求极高（如某些科学计算），则使用 float16 或 float32。

**Q3: FP8 能否直接用于 tl.dot？**

A: 当前 Ascend A2/A3 系列不支持 FP8 作为 `tl.dot` 的输入。FP8 数据需要先转换为 fp16/bf16/fp32 后再进行矩阵乘法。910_95 系列支持 FP8 类型转换和 `tl.dot_scaled` 的 FP8 输入。

**Q4: 为什么 bool 类型会导致 UB 溢出？**

A: Triton 内部将 bool (int1) 转为 int8 进行运算，int8 占用 1 字节而非 1 比特。如果大量使用 bool 类型的 tensor，实际内存占用会比预期大很多。建议控制 bool tensor 的数量和大小。

**Q5: 混合 bfloat16 和 float16 运算会怎样？**

A: 根据 `computation_type` 规则，bfloat16 和 float16 混合运算会自动提升为 float32。这意味着额外的类型转换开销和更高的内存占用。建议在同一 kernel 中统一使用一种 16-bit 浮点类型。

**Q6: 如何处理大偏移量的指针运算？**

A: 当数据量超过 2^31（int32 范围）时，需要使用 int64 进行偏移量计算。使用 `al.int64()` 包装 Python 整数，或在 kernel 内部使用 `.to(tl.int64)` 转换。参见上文"int64 在 NPU 上的使用"示例。

## 相关文档

- [SPMD 模型在 NPU 上的映射](./01-spmd-on-npu.md)
- [Grid/Program ID 与 AI Core 对应关系](./02-grid-and-program-id.md)
- [NPU 内存层次](./03-memory-model.md)
- [硬件架构概览](../../docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md)

## 源文件参考

- [core.py - dtype 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L288-L683) - 数据类型定义和标量类型
- [semantic.py - integer_promote](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L46-L58) - 整数类型提升规则
- [semantic.py - computation_type](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L61-L100) - 计算类型推导规则
- [compiler.py - AscendOptions](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L835) - Ascend 编译选项（FP8 支持）
- [core.py (extension)](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) - NPU 扩展 API
- [dot.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Linear_Algebra_Ops/dot.md) - tl.dot 支持规格
- [tl.load.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md) - tl.load 支持规格
- [cast.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Creation_Ops/cast.md) - 类型转换支持规格
