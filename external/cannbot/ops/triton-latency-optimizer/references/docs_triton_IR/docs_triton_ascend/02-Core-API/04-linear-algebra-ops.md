# 线性代数操作 API - tl.dot / tl.dot_scaled 及 Cube 管线

## 概述

本文档详细描述 Triton-Ascend 中的线性代数操作 API，包括 `tl.dot` 和 `tl.dot_scaled`。矩阵乘法是深度学习中最核心的计算操作，在 Ascend NPU 上由 Cube 计算单元（矩阵计算单元）执行，对应 GPU 上的 Tensor Core。理解 Cube 管线的工作方式、数据布局要求（ND2NZ 转换）、分块策略以及精度控制，对于在 NPU 上编写高性能矩阵乘法 kernel 至关重要。

关键词：tl.dot, tl.dot_scaled, 矩阵乘法, Cube, Tensor Core, ND2NZ, 分块, allow_tf32, input_precision, hf32, 精度, 数据布局

---

## API 参考

### tl.dot

计算两个块的矩阵乘积。

```python
triton.language.dot(
    input,
    other,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=float32,
    _builder=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | 2D 或 3D tensor | 第一个输入矩阵。支持 int8, fp16, bf16, fp32 类型 |
| `other` | 2D 或 3D tensor | 第二个输入矩阵。必须与 `input` 相同 dtype |
| `acc` | 2D 或 3D tensor，可选 | 累加器张量。若不为 None，结果加到该张量上 |
| `input_precision` | `str`，可选 | 精度模式。NPU 可选值：`"ieee"`（默认）、`"hf32"`。GPU 可选值还包括 `"tf32"`、`"tf32x3"` |
| `allow_tf32` | `bool`，可选 | **已弃用**。若为 True，在 NPU 上等价于 `input_precision="hf32"` |
| `max_num_imprecise_acc` | `int`，可选 | 低精度累加次数限制。**NPU 暂不支持**，传入会被忽略 |
| `out_dtype` | `tl.dtype`，默认 `float32` | 输出数据类型。NPU 支持 float32 和 int32 |

**返回值**：矩阵乘法结果张量

**约束**：
- `input` 和 `other` 必须同为 2D 或同为 3D
- 3D 情况下第一维为 batch 维度
- `input` 的最后一维必须等于 `other` 的倒数第二维（K 维度）
- 两个输入的 dtype 必须相同
- 输入矩阵的维度必须满足最小尺寸要求（由 `min_dot_size` 决定）
- `out_dtype=bfloat16` 不支持，需使用 float32/float16 后手动 cast

**NPU 精度模式映射**：

| GPU 精度模式 | NPU 精度模式 | 说明 |
|-------------|-------------|------|
| `"tf32"` | `"hf32"` | NPU 使用 hf32 替代 tf32 |
| `"tf32x3"` | 不支持 | NPU 不支持 tf32x3 |
| `"ieee"` | `"ieee"` | IEEE float32 精度 |
| `allow_tf32=True` | `"hf32"` | 自动映射为 hf32 |

**数据类型支持（Ascend）**：

| 输入类型 | int8 | fp16 | bf16 | fp32 | int1 |
|---------|------|------|------|------|------|
| 支持 | √ | √ | √ | √ | √ |

| 输出类型 | float32 | int32 | float16 | bfloat16 |
|---------|---------|-------|---------|----------|
| 支持 | √ | √（int8 输入时） | √ | ×（需手动 cast） |

### tl.dot_scaled

计算微缩放格式（microscaling format）的矩阵乘法。

```python
triton.language.dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    out_dtype=float32,
    lhs_k_pack=True,
    rhs_k_pack=True,
    _builder=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `lhs` | 2D tensor | 左矩阵，f8/f6/f4 格式打包在 int32 中 |
| `lhs_scale` | tensor | 左矩阵缩放因子，ue8m0 float8 类型（表示为 int8 张量） |
| `lhs_format` | `str` | 左矩阵格式：`"e4m3"` / `"e5m2"` / `"e2m3"` / `"e3m2"` / `"e2m1"` / `"bf16"` / `"fp16"` |
| `rhs` | 2D tensor | 右矩阵 |
| `rhs_scale` | tensor，可选 | 右矩阵缩放因子 |
| `rhs_format` | `str` | 右矩阵格式 |
| `acc` | 2D tensor，可选 | 累加器 |
| `out_dtype` | `tl.dtype`，默认 `float32` | 仅支持 float32 |
| `lhs_k_pack` | `bool`，默认 `True` | 左矩阵是否沿 K 维打包 |
| `rhs_k_pack` | `bool`，默认 `True` | 右矩阵是否沿 K 维打包 |

**约束**：
- `out_dtype` 必须为 float32
- 910B 上 lhs/rhs 仅支持 bf16/fp16；910-95 上额外支持 uint8/e5m2/e4m3
- lhs 和 rhs 必须相同 dtype
- lhs_format 和 rhs_format 必须在允许的格式列表中

---

## 代码示例

### 基础用法：矩阵乘法 kernel

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### 进阶用法：int8 量化矩阵乘法与 dot_scaled

```python
@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                     mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        offs_k = offs_k + BLOCK_K

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

---

## NPU 适配要点

### 1. Cube 计算单元的映射

Ascend NPU 的 Cube 单元是专用的矩阵计算引擎，类似于 GPU 的 Tensor Core。`tl.dot` 操作在 NPU 上映射为 Cube 矩阵乘指令：

- **GPU Tensor Core**：支持 fp16/bf16/int8 的矩阵乘，累加精度为 fp32/int32
- **NPU Cube**：同样支持 fp16/bf16/int8 的矩阵乘，累加精度为 fp32/int32

Cube 单元的计算吞吐量远高于 Vector 单元，是矩阵乘法性能的关键。

### 2. 矩阵乘法的分块策略

NPU 上矩阵乘法的分块策略需要考虑以下因素：

- **UB 容量限制**：分块大小受 UB 容量约束，过大的 BLOCK_M/BLOCK_N/BLOCK_K 可能导致 UB 溢出
- **Cube 利用率**：分块大小应匹配 Cube 单元的计算粒度，以充分利用硬件
- **数据复用**：K 维度分块（BLOCK_K）允许在 UB 中复用 A/B 矩阵的数据块

典型的分块配置：
- BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32（fp16）
- BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64（fp32，需要更多 UB）

### 3. 数据布局要求（ND2NZ 转换）

NPU Cube 单元要求数据以特定的 ZZ（Nz）格式排列，而 Triton kernel 中的数据通常是 ND（行优先）格式。编译器会自动插入 **ND2NZ**（ND 到 Nz）的数据格式转换指令：

- **输入转换**：A/B 矩阵从 ND 格式转换为 Nz 格式，供 Cube 单元使用
- **输出转换**：Cube 输出从 Nz 格式转换回 ND 格式，供后续 Vector 操作使用

这些转换会消耗额外的 UB 空间和计算周期，是 NPU 矩阵乘法的额外开销来源。优化建议：
- 尽量增大分块大小，摊薄格式转换的开销
- 减少 K 维度的分块次数（增大 BLOCK_K），减少重复转换

### 4. input_precision 在 NPU 上的行为

| 精度模式 | GPU 行为 | NPU 行为 |
|---------|---------|---------|
| `"ieee"` | 使用 IEEE float32 精度 | 使用 IEEE float32 精度 |
| `"tf32"` | 使用 TF32 精度（19-bit） | **自动映射为 `"hf32"`** |
| `"hf32"` | 不适用 | NPU 特有的半精度 float32 格式 |
| `"tf32x3"` | 使用三倍 TF32 精度 | **不支持** |

当 `input_precision="hf32"` 且输入不是 fp32 时，精度设置会被忽略，回退到默认的 `"ieee"` 模式。

### 5. max_num_imprecise_acc 不支持

NPU 上 `max_num_imprecise_acc` 参数暂不支持，传入的值会被忽略并设为 0。这意味着所有累加操作都使用完整精度。

### 6. out_dtype 限制

- `out_dtype=bfloat16` 不支持，需要使用 float32 或 float16 后手动 cast：`result.to(tl.bfloat16)`
- int8 输入时，`out_dtype` 默认为 int32
- fp16/bf16 输入时，`out_dtype` 默认为 float32

---

## 常见问题

**Q1: 为什么 tf32 在 NPU 上映射为 hf32？**

A: NPU 的 Cube 单元不支持 TF32 格式，但支持类似的半精度 float32 格式（hf32）。hf32 是 NPU 特有的精度模式，精度介于 fp16 和 fp32 之间，与 TF32 的设计目标类似。

**Q2: 如何选择 BLOCK_M/BLOCK_N/BLOCK_K 的大小？**

A: 需要权衡 UB 容量和 Cube 利用率。较大的分块可以提高 Cube 利用率，但可能超出 UB 容量。建议从 BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 开始，根据实际 UB 使用情况调整。

**Q3: ND2NZ 转换的开销有多大？**

A: 转换开销取决于数据量和 Cube 计算量之比。对于大型矩阵乘法（M, N, K 较大），转换开销占比很小；对于小型矩阵乘法，转换开销可能成为瓶颈。

**Q4: dot_scaled 在哪些 NPU 上可用？**

A: `dot_scaled` 在 910-95 上支持 bf16/fp16/uint8/e5m2/e4m3 格式；在 910B 上仅支持 bf16/fp16 格式。

**Q5: 为什么 acc 不能使用 fp16？**

A: NPU Cube 单元的累加器精度固定为 fp32（浮点）或 int32（整数），不支持 fp16 累加。这是硬件设计决定，为了保证计算精度。

---

## 相关文档

- [01-memory-ops.md](./01-memory-ops.md) - 内存操作 API
- [02-math-ops.md](./02-math-ops.md) - 数学运算 API
- [03-reduction-ops.md](./03-reduction-ops.md) - 归约操作

## 源码参考

- [core.py - dot 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1518-L1558)
- [core.py - dot_scaled 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1561-L1581)
- [semantic.py - dot 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1559-L1626)
- [semantic.py - dot_scaled 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1659-L1730)
