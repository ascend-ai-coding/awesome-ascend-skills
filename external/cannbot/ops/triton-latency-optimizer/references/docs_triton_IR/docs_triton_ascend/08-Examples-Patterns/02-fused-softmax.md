# 融合 Softmax 模式（Fused Softmax Pattern）

## 概述

融合 Softmax 是 Triton 中展示 kernel fusion 优势的经典案例。原生 PyTorch 实现 Softmax 需要多次读写 Global Memory，而融合 kernel 将所有计算放在片上完成，大幅减少内存访问次数。在 NPU 上，Softmax 是典型的 Vector 计算密集 + 归约操作组合。

| 关键概念 | 说明 |
|---------|------|
| Kernel Fusion | 将多个算子融合为单个 kernel，减少 Global Memory 读写 |
| 数值稳定性 | 减去最大值防止 exp 溢出 |
| 行归约 | 对每行独立执行 max/sum 归约 |
| `tl.max / tl.sum` | Triton 归约操作 |
| `tl.range` | 带软件流水线（num_stages）的循环迭代器 |
| `triton.next_power_of_2` | 向上取 2 的幂，满足 Triton block 大小要求 |
| `other=-float('inf')` | load 的默认填充值，对 softmax 计算无影响 |

## 完整代码示例

```python
import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_programs = 32
    y = torch.empty_like(x)
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)
    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

## 数值稳定性技巧（减最大值）

Softmax 的数学定义为：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

直接计算 `exp(x_i)` 在 `x_i` 较大时会导致数值溢出。利用 Softmax 的平移不变性：

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

减去最大值后，所有 `exp` 的参数都 <= 0，避免了溢出。

```python
row_minus_max = row - tl.max(row, axis=0)
numerator = tl.exp(row_minus_max)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
```

### 关键步骤分解

| 步骤 | 操作 | 输入 | 输出 | 内存访问 |
|-----|------|------|------|---------|
| 1 | 加载行数据 | Global Memory | SRAM | 1次读 |
| 2 | 计算行最大值 | SRAM | 标量 | 0 |
| 3 | 减最大值 | SRAM | SRAM | 0 |
| 4 | 计算 exp | SRAM | SRAM | 0 |
| 5 | 求和 | SRAM | 标量 | 0 |
| 6 | 除法 | SRAM | SRAM | 0 |
| 7 | 写回结果 | SRAM | Global Memory | 1次写 |

融合 kernel 仅需 1 次读 + 1 次写，而原生实现需要 5 次读 + 3 次写。

## 归约轴处理

### 行归约模式

Softmax 对每一行独立归约，`axis=0` 表示对列维度归约：

```python
row_max = tl.max(row, axis=0)
row_sum = tl.sum(numerator, axis=0)
```

- `row` 的 shape 为 `[BLOCK_SIZE]`
- `tl.max(row, axis=0)` 返回标量，即该行的最大值
- `tl.sum(numerator, axis=0)` 返回标量，即该行 exp 值的总和

### 多行并行

每个 program 处理多行，通过 `tl.range` 循环实现：

```python
row_start = tl.program_id(0)
row_step = tl.num_programs(0)
for row_idx in tl.range(row_start, n_rows, row_step):
    ...
```

- `row_start`：当前 program 起始行
- `row_step`：步长等于 program 总数，实现行级交错分配
- 这种分配方式有助于负载均衡

### 2 的幂填充

Triton 要求 block 大小为 2 的幂：

```python
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

当 `n_cols` 不是 2 的幂时，`BLOCK_SIZE > n_cols`，需要用 `mask` 保护：

```python
mask = col_offsets < n_cols
row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
```

`other=-float('inf')` 确保 padding 位置的 exp 值为 0，不影响求和结果。

## NPU 上的性能优化

### 1. Program 数量调优

NPU 的 Vector 核心数量与 GPU SM 数量不同，需要根据实际硬件调整：

```python
num_programs = 32
num_programs = min(num_programs, n_rows)
```

可以通过 `get_npu_properties()` 获取实际核心数：

```python
from triton.runtime import driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

num_cores = get_npu_properties()["num_aicore"]
```

### 2. BLOCK_SIZE 选择

NPU 亲和 512B 对齐。对于 fp32 数据，推荐 BLOCK_SIZE 为 128 的倍数：

| n_cols 范围 | 推荐 BLOCK_SIZE |
|------------|----------------|
| <= 128 | 128 |
| <= 256 | 256 |
| <= 512 | 512 |
| <= 1024 | 1024 |

### 3. UB 空间管理

每行数据加载到 UB 后，需要额外空间存储中间结果。对于 fp32 数据：

```
UB 占用 ≈ BLOCK_SIZE * 4B * 3（row, numerator, output）
```

确保 `3 * BLOCK_SIZE * 4 <= 96KB`（A2/A3）或 `3 * BLOCK_SIZE * 4 <= 128KB`（910_95），即 `BLOCK_SIZE <= 8192`（A2/A3）或 `BLOCK_SIZE <= 10922`（910_95）。

### 4. Autotune 配置

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(...):
    ...
```

## 常见问题（Q&A）

**Q: Softmax 结果精度不够，与 PyTorch 差异较大？**

A: 检查是否正确减去了最大值。NPU 上 `tl.exp` 是近似计算，对于 fp16 输入可能有精度差异。建议在 kernel 内部使用 fp32 精度计算，最终写回时再转换。

**Q: n_cols 很大时（如 8192+）性能下降？**

A: 大 n_cols 会导致单行数据超出 UB 容量。需要将行分块处理，每块加载一部分列，增量更新 max 和 sum。

**Q: causal softmax 如何实现？**

A: 在 load 时使用 causal mask，将未来位置的值设为 `-inf`：

```python
causal_mask = col_offsets <= row_idx
row = tl.load(input_ptrs, mask=mask & causal_mask, other=-float('inf'))
```

**Q: NPU 上 softmax 比 GPU 慢？**

A: Softmax 是 memory-bound 操作，性能取决于内存带宽。确保 BLOCK_SIZE 选择合理，并使用 autotune 找到最优配置。

## 相关文档

- [01-vector-add.md](./01-vector-add.md) - 向量加法模式
- [05-flash-attention.md](./05-flash-attention.md) - Flash Attention 模式（含在线 Softmax）
- [06-reduction-pattern.md](./06-reduction-pattern.md) - 归约操作模式
- 源码参考：[02-fused-softmax.py (upstream)](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials/02-fused-softmax.py)
- 源码参考：[02_fused_softmax_example.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/examples/02_fused_softmax_example.md)
