# 扫描与排序操作 API

## 概述

本文档详细描述 Triton-Ascend 中的扫描与排序操作 API，包括 `associative_scan`、`cumsum`、`cumprod`、`histogram` 和 `sort`。扫描操作（也称为前缀和/前缀扫描）沿指定轴对张量元素进行累积计算，排序操作对张量元素进行排序。在 Ascend NPU 上，这些操作由 Vector 计算单元执行，其实现方式和性能特征与 GPU 有所不同。排序操作基于 bitonic merge sort 算法，要求张量大小为 2 的幂。

关键词：扫描, 排序, associative_scan, cumsum, cumprod, histogram, sort, bitonic, 前缀和, NPU, Vector

---

## API 参考

### tl.associative_scan

沿指定轴对张量执行关联扫描（associative scan / prefix scan）。

```python
triton.language.associative_scan(
    input,
    axis,
    combine_fn,
    reverse=False,
    _builder=None,
    _generator=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` 或 `tuple(tensor)` | 输入张量，或张量元组（多输出扫描） |
| `axis` | `int` | 扫描轴 |
| `combine_fn` | `Callable` | 组合函数，必须标记 `@triton.jit`。接受两个标量参数，返回组合结果 |
| `reverse` | `bool`，默认 `False` | 是否沿轴反向扫描 |

**返回值**：扫描结果张量（或张量元组）

**约束**：
- `combine_fn` 必须满足**结合律**（associative）
- 所有输入张量必须形状相同
- `axis` 必须在有效范围内

**语义**：对于 `combine_fn = f`，`associative_scan(x, axis, f)` 产生：
- `result[0] = x[0]`
- `result[1] = f(x[0], x[1])`
- `result[2] = f(f(x[0], x[1]), x[2])`
- ...

### tl.cumsum

沿指定轴计算累积和（前缀和）。

```python
triton.language.cumsum(input, axis=0, reverse=False) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int`，默认 `0` | 扫描轴 |
| `reverse` | `bool`，默认 `False` | 是否反向计算 |

**实现**：`associative_scan(input, axis, lambda a, b: a + b, reverse)`

**NPU 注意**：bfloat16 输入会自动提升为 float32 执行。

### tl.cumprod

沿指定轴计算累积乘积。

```python
triton.language.cumprod(input, axis=0, reverse=False) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int`，默认 `0` | 扫描轴 |
| `reverse` | `bool`，默认 `False` | 是否反向计算 |

**实现**：`associative_scan(input, axis, lambda a, b: a * b, reverse)`

**NPU 注意**：bfloat16 输入会自动提升为 float32 执行。

### tl.histogram

基于输入张量计算直方图。

```python
triton.language.histogram(input, num_bins, _builder=None, _generator=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量，**必须为 1D 且为整数类型** |
| `num_bins` | `int` | 直方图桶的数量 |

**返回值**：形状为 `(num_bins,)` 的 int32 张量

**约束**：
- 输入必须是 1D 张量
- 输入必须是整数类型
- 桶宽度为 1，起始位置为 0（即第 i 个桶统计值为 i 的元素数量）

### tl.sort

沿指定维度对张量排序。

```python
triton.language.sort(x, dim=None, descending=False) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 输入张量 |
| `dim` | `int`，可选 | 排序维度。若为 None，沿最后一维排序。**当前仅支持最后一维** |
| `descending` | `bool`，默认 `False` | 是否降序排序 |

**约束**：
- 仅支持沿最后一维排序
- 排序维度的大小必须是 2 的幂（基于 bitonic merge sort 实现）

---

## 代码示例

### 基础用法：cumsum / cumprod

```python
import triton
import triton.language as tl

@triton.jit
def cumsum_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    result = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def cumprod_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)

    result = tl.cumprod(x, axis=0)
    tl.store(out_ptr + offsets, result, mask=mask)
```

### 进阶用法：自定义 associative_scan 与 sort

```python
@triton.jit
def scan_combine(a, b):
    return a + b

@triton.jit
def custom_scan_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    result = tl.associative_scan(x, axis=0, combine_fn=scan_combine)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def sort_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    sorted_x = tl.sort(x, dim=0)
    tl.store(out_ptr + offsets, sorted_x, mask=mask)
```

---

## NPU 适配要点

### 1. 扫描操作在 NPU 上的实现

NPU 上的 `associative_scan` 由 Vector 计算单元执行。与 GPU 的实现类似，采用并行扫描算法（Blelloch scan 或 Hillis-Steele scan），时间复杂度为 O(log n)。

关键实现细节：
- 扫描操作在 IR 层面通过 `create_scan` 创建
- `combine_fn` 被编译为 scan region 内的内联函数
- 扫描的中间结果存储在 UB 中

**性能注意**：扫描操作需要 O(log n) 步，每步都需要读写 UB，因此比简单的逐元素操作慢得多。对于大张量，扫描可能成为性能瓶颈。

### 2. 排序算法的 NPU 适配

Triton 的 `sort` 操作基于 **bitonic merge sort**（双调归并排序）算法实现：

1. **Bitonic merge sort**：一种适合并行硬件的排序算法
2. **时间复杂度**：O(n log^2 n)
3. **空间复杂度**：O(1) 额外空间（原地排序）

算法步骤：
1. 将输入 reshape 为超立方体形状 `[2, 2, ..., 2]`
2. 迭代执行 bitonic merge 步骤
3. 每一步包含多轮 compare-and-swap 操作
4. 最终 reshape 回原始形状

**NPU 限制**：
- 排序维度的大小必须是 2 的幂
- 仅支持沿最后一维排序
- 排序操作涉及大量 reshape 和比较操作，性能可能不如专用排序硬件

### 3. histogram 的约束

`histogram` 操作在 NPU 上有以下约束：
- 输入必须是 1D 整数张量
- 桶从 0 开始，宽度为 1
- 输出为 int32 类型

对于浮点输入或非均匀桶的需求，需要手动实现（使用 `atomic_add` 或 `associative_scan`）。

### 4. reverse 参数

`cumsum` 和 `cumprod` 支持 `reverse` 参数：
- `reverse=False`（默认）：从左到右扫描
- `reverse=True`：从右到左扫描

反向扫描在 NPU 上的实现与正向扫描对称，性能特征相同。

### 5. bfloat16 精度提升

与归约操作类似，`cumsum` 和 `cumprod` 中的 bfloat16 输入会自动提升为 float32 执行（通过 `_promote_bfloat16_to_float32`），因为 NPU 不支持 bf16 的部分算术操作。

---

## 常见问题

**Q1: associative_scan 的 combine_fn 有什么要求？**

A: `combine_fn` 必须标记 `@triton.jit`，接受两个标量参数，返回组合结果。最重要的是必须满足**结合律**：`f(f(a, b), c) = f(a, f(b, c))`。不满足结合律的函数会导致错误结果。

**Q2: 为什么 sort 只支持 2 的幂大小？**

A: Triton 的 sort 基于 bitonic merge sort 实现，该算法要求输入大小为 2 的幂。如果需要排序非 2 幂大小的数据，可以填充到最近的 2 的幂，排序后截取有效部分。

**Q3: cumsum 和 reduce + store 有什么区别？**

A: `cumsum` 保留所有中间结果（前缀和），输出形状与输入相同。`reduce(sum)` 只返回最终总和，输出是标量或缩减维度的张量。

**Q4: 如何实现反向 cumsum（后缀和）？**

A: 使用 `cumsum(x, axis=0, reverse=True)`。

**Q5: histogram 能处理浮点数据吗？**

A: 不能。`histogram` 仅支持 1D 整数输入。对于浮点数据，需要先离散化为整数（如 `bin_index = (x * scale).to(tl.int32)`），再使用 histogram 或 atomic_add 统计。

---

## 相关文档

- [03-reduction-ops.md](./03-reduction-ops.md) - 归约操作
- [05-atomic-ops.md](./05-atomic-ops.md) - 原子操作
- [06-shape-ops.md](./06-shape-ops.md) - 张量形状操作

## 源码参考

- [core.py - associative_scan 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2150-L2200)
- [core.py - histogram 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2190-L2200)
- [standard.py - cumsum/cumprod 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L305-L331)
- [standard.py - sort 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L430-L451)
- [semantic.py - associative_scan 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1792-L1809)
- [semantic.py - histogram 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1840-L1843)
