# 归约操作模式（Reduction Pattern）

## 概述

归约操作是并行计算中的基本模式，将一组值压缩为更少数量的值（如求和、最大值、最小值）。在 Triton 中，归约操作广泛应用于 Softmax、LayerNorm、Flash Attention 等 kernel。NPU 上的归约需要特别注意 UB 空间限制和多核同步机制。

| 关键概念 | 说明 |
|---------|------|
| 行归约 | 对每行的元素执行归约，结果为每行一个标量 |
| 列归约 | 对每列的元素执行归约，结果为每列一个标量 |
| 全局归约 | 对整个 tensor 执行归约，结果为单个标量 |
| 多轴归约 | 沿多个轴同时归约 |
| `tl.sum / tl.max / tl.min` | Triton 内置归约操作 |
| `tl.argmax / tl.argmin` | 带索引的归约操作 |
| `tl.reduce` | 通用归约操作，自定义归约函数 |
| 归约+后处理融合 | 将归约结果直接用于后续计算，避免额外内存访问 |

## 行归约模式

行归约是最常见的归约模式，对每行独立执行归约操作：

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(input_ptr, output_ptr,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(row, axis=0)
    tl.store(output_ptr + row_idx, row_sum)


def row_sum(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    row_sum_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### 行最大值

```python
@triton.jit
def row_max_kernel(input_ptr, output_ptr,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row = tl.load(row_start + cols, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=0)
    tl.store(output_ptr + row_idx, row_max)
```

### 大 N 列的行归约

当 `n_cols` 超过单个 BLOCK_SIZE 时，需要分块累加：

```python
@triton.jit
def row_sum_large_kernel(input_ptr, output_ptr,
                         n_rows, n_cols,
                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        partial_sum += row
    result = tl.sum(partial_sum, axis=0)
    tl.store(output_ptr + row_idx, result)
```

## 列归约模式

列归约对每列独立执行归约，需要多个 program 协作完成同一列的归约：

```python
@triton.jit
def col_sum_kernel(input_ptr, output_ptr,
                   n_rows, n_cols,
                   BLOCK_M: tl.constexpr):
    col_idx = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    mask = rows < n_rows
    col_data = tl.load(input_ptr + rows[:, None] * n_cols + col_idx, mask=mask[:, None], other=0.0)
    col_sum = tl.sum(col_data, axis=0)
    tl.store(output_ptr + col_idx, col_sum)
```

### 使用原子操作实现列归约

当行数很大时，可以使用原子操作将部分和累加到输出：

```python
@triton.jit
def col_sum_atomic_kernel(input_ptr, output_ptr,
                          n_rows, n_cols,
                          BLOCK_M: tl.constexpr):
    row_start = tl.program_id(0) * BLOCK_M
    cols = tl.arange(0, n_cols)
    rows = row_start + tl.arange(0, BLOCK_M)
    mask = rows[:, None] < n_rows
    data = tl.load(input_ptr + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0)
    partial_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr + cols, partial_sum, mask=cols < n_cols)
```

> **NPU 注意**：`atomic_add` 不支持多核 add+保存中间结果。如果需要保存中间结果，应改用普通 add + 自旋锁方式。

## 全局归约模式

全局归约将整个 tensor 归约为单个值：

```python
@triton.jit
def global_sum_kernel(input_ptr, output_ptr,
                      n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr, block_sum)
```

### 两阶段全局归约

对于大 tensor，使用两阶段归约更高效：

```python
@triton.jit
def global_sum_stage1_kernel(input_ptr, partial_sums,
                             n_elements, num_blocks,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(data, axis=0)
    tl.store(partial_sums + pid, block_sum)


@triton.jit
def global_sum_stage2_kernel(partial_sums, output_ptr,
                             num_blocks,
                             BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    data = tl.load(partial_sums + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(data, axis=0)
    tl.store(output_ptr, total)
```

## 多轴归约

对多个轴同时归约，需要逐步处理每个轴：

```python
@triton.jit
def multi_axis_sum_kernel(input_ptr, output_ptr,
                          M, N, K,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    m_idx = pid
    row_start = input_ptr + m_idx * N * K
    total = tl.zeros([1], dtype=tl.float32)
    for n_off in range(0, N * K, BLOCK_SIZE):
        offsets = n_off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * K
        data = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(data, axis=0)
    tl.store(output_ptr + m_idx, total)
```

## 归约+后处理融合

将归约结果直接用于后续计算，避免额外的内存读写：

### Softmax 中的融合归约

```python
@triton.jit
def fused_softmax_kernel(output_ptr, input_ptr,
                         n_rows, n_cols,
                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row = tl.load(row_start + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    output_start = output_ptr + row_idx * n_cols
    tl.store(output_start + cols, result, mask=mask)
```

### LayerNorm 中的融合归约

```python
@triton.jit
def fused_layernorm_kernel(output_ptr, input_ptr, weight_ptr, bias_ptr,
                           n_rows, n_cols, eps,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)
    y = (x - mean) * rstd * w + b

    output_start = output_ptr + row_idx * n_cols
    tl.store(output_start + cols, y, mask=mask)
```

## NPU 归约操作支持矩阵

| 归约操作 | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|---------|------|-------|-------|-------|------|------|------|------|
| `tl.sum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.max` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.min` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.argmax` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| `tl.argmin` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| `tl.reduce` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.xor_sum` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* |

> ✓* 表示 Triton 内部将 bool 转为 int8 进行运算

## 常见问题（Q&A）

**Q: 归约结果精度不够？**

A: 在 kernel 内部使用 fp32 精度进行归约计算，最终写回时再转换为目标类型。`tl.sum` 对 fp16 输入可能产生精度损失。

**Q: 列归约性能差？**

A: 列归约的内存访问模式不连续，导致带宽利用率低。考虑转置输入后使用行归约，或使用 atomic_add 累加部分和。

**Q: atomic_add 在 NPU 上有什么限制？**

A: NPU 不支持 `atomic_add` 实现多核 add+保存中间结果。`atomic_or/atomic_xor/atomic_and/atomic_xchg/atomic_cas` 暂不支持在 loop 中使用。`sem` 只支持默认值 `"acq_rel"`，`scope` 只支持默认值 `"gpu"`。

**Q: 如何实现自定义归约操作？**

A: 使用 `tl.reduce`：

```python
@triton.jit
def my_reduce(a, b):
    return tl.maximum(a, b)

result = tl.reduce(data, axis=0, combine_fn=my_reduce)
```

**Q: 大 tensor 全局归约如何避免 atomic_add 精度问题？**

A: 使用两阶段归约：第一阶段每个 program 计算部分和并写入独立位置，第二阶段汇总部分和。避免使用 atomic_add。

## 相关文档

- [02-fused-softmax.md](./02-fused-softmax.md) - 融合 Softmax 模式
- [04-layer-norm.md](./04-layer-norm.md) - LayerNorm 模式
- [05-flash-attention.md](./05-flash-attention.md) - Flash Attention 模式
- [08-api-support-matrix.md](../09-Reference/01-api-support-matrix.md) - API 支持矩阵
