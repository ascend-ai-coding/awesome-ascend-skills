# care_padding 优化

## 概述

`care_padding` 是 Triton-Ascend 在 `tl.load` 操作中引入的 NPU 专属优化参数。当使用 mask 加载数据时，未被 mask 覆盖的 padding 区域需要填充默认值。`care_padding` 控制是否对 padding 区域进行显式填充，直接影响 MTE2 搬运指令与 Vector 计算指令的并行度，是访存密集型算子性能优化的关键手段。

## 关键概念

| 概念 | 说明 | 默认值 |
|------|------|--------|
| care_padding | 控制 tl.load 是否关心 padding 区域的数据正确性 | True |
| padding 区域 | mask=False 对应的内存位置，即未被有效数据覆盖的区域 | - |
| MTE2 | 数据从 GM 搬入 UB 的搬运流水线 | - |
| Vector 初始化 | 使用 Vector Core 对内存区域进行初始化填充 | care_padding=True 时触发 |
| 存算并行 | MTE2 搬运与 Vector 计算重叠执行 | care_padding=False 时更容易实现 |

## care_padding 参数的含义

`care_padding` 是 `tl.load` 的可选参数，定义如下：

```python
tl.load(
    pointer,
    mask=None,
    other=None,
    care_padding=True,  # NPU 专属参数
    ...
)
```

源码参考：[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1621-L1638)

### 参数行为

| care_padding | other 参数 | padding 区域行为 | 对性能的影响 |
|:---:|:---:|------|------|
| True | None | padding 区域填充 0（与 GPU 行为一致） | MTE2 与 Vector 存在依赖，降低并行度 |
| True | 指定值 | padding 区域填充指定值 | MTE2 与 Vector 存在依赖，降低并行度 |
| False | 任意 | padding 区域为随机值（未定义） | MTE2 与 Vector 无依赖，提升并行度 |
| - | 非 None | care_padding 不生效，other 值覆盖 | 取决于 other 的使用方式 |

> 注意：当 `other` 参数不为 None 时，`care_padding` 不生效，因为 `other` 已经显式指定了填充值。

## care_padding=True（默认）：确保 padding 区域数据正确

### 行为详解

当 `care_padding=True` 时，NPU 需要确保 padding 区域的数据与 GPU 行为一致（默认填充 0 或 other 指定值）。实现方式是：

1. **Vector 初始化**：先用 Vector Core 对 data 指向的全部内存空间设置为指定值
2. **MTE2 搬运**：再使用 MTE2 指令搬运数据到 data 指向的部分内存空间

这导致 MTE2 和 Vector 产生依赖——MTE2 必须等待 Vector 初始化完成后才能开始搬运，无法高效并行。

```
care_padding=True 时的执行时序：
Vector: |==初始化全0==|                    |==计算==|
MTE2:                  |==搬运有效数据==|
                                ↑ 必须等待初始化完成
```

### 适用场景

- padding 区域的数据会被后续计算使用
- 算子正确性依赖于 padding 区域的值
- 不确定 padding 区域是否影响结果时（安全选择）

## care_padding=False：跳过 padding 处理，提升性能

### 行为详解

当 `care_padding=False` 时，NPU 跳过 Vector 初始化步骤，直接使用 MTE2 搬运有效数据。padding 区域的值是未定义的（随机值），但 MTE2 搬运不再依赖 Vector 初始化，可以实现存算并行。

```
care_padding=False 时的执行时序：
MTE2:   |==搬运1==|==搬运2==|==搬运3==|
Vector:           |==计算1==|==计算2==|==计算3==|
                  ↑ 无需等待，直接并行
```

### 性能提升原理

| 阶段 | care_padding=True | care_padding=False |
|------|-------------------|---------------------|
| 步骤1 | Vector 初始化 padding 区域 | 无 |
| 步骤2 | MTE2 搬运有效数据 | MTE2 搬运有效数据 |
| 步骤3 | Vector 计算 | Vector 计算 |
| 并行性 | MTE2 等待 Vector 初始化 | MTE2 与 Vector 计算并行 |
| 总耗时 | 初始化 + 搬运 + 计算 | 搬运与计算重叠 |

## 何时可以安全地使用 care_padding=False

### 安全条件

使用 `care_padding=False` 的前提是 **padding 区域的数据不影响最终计算结果**。以下场景可以安全使用：

### 场景1：padding 区域的数据不会被使用

```python
# 典型场景：element-wise 操作，mask 外的值不会被 store
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    # padding 区域的值在 x+y 后仍会被 mask 过滤掉
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
    output = x + y
    # store 时使用相同的 mask，padding 区域不会被写出
    tl.store(out_ptr + offsets, output, mask=mask)
```

### 场景2：padding 区域参与计算但会被后续 mask 过滤

```python
# 典型场景：reduction 操作，padding 区域的值不影响结果
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    # 如果 padding 区域为 0，不影响 sum 结果
    # 但如果 padding 区域为随机值，sum 结果会错误！
    # 此时不能使用 care_padding=False，除非 other=0
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, result)
```

> **重要**：对于 reduction 操作（sum、max 等），padding 区域的值会影响结果。此时不能使用 `care_padding=False`，除非能确保 padding 值为 0（sum）或 -inf（max）等不影响结果的值。

### 场景3：padding 区域的值被 where/select 覆盖

```python
# 典型场景：条件选择操作
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # inp 的 padding 区域会被 fill_mask 过滤
    input_vals = tl.load(inp + offsets, mask=mask, care_padding=False)
    fill_mask_vals = tl.load(expand_mask + offsets, mask=mask).to(tl.int1)
    # where 会选择 fill_mask_vals 为 True 的位置填充 value
    # padding 区域的 input_vals 不影响结果
    result = tl.where(fill_mask_vals, value, input_vals)
    tl.store(out + offsets, result, mask=mask)
```

### 不安全场景

| 场景 | 原因 | 正确做法 |
|------|------|----------|
| Reduction (sum) | padding 区域的随机值会被累加 | 使用 `other=0.0` 而非 `care_padding=False` |
| Reduction (max) | padding 区域的随机值可能成为最大值 | 使用 `other=-float('inf')` |
| 中间结果依赖 padding | 后续计算使用 padding 区域的值 | 保持 `care_padding=True` |
| Store 无 mask | padding 区域的值会被写出 | 确保 store 有正确的 mask |

## 代码示例

### 示例1：基本优化

```python
@triton.jit
def npu_vector_add_kernel(
    input,
    output,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    N: tl.constexpr = BLOCK_SIZE
    idx = tl.arange(0, N)
    mask = idx < M
    # 优化前：care_padding=True（默认），MTE2 等待 Vector 初始化
    # data = tl.load(input + idx, mask=mask)
    # 优化后：care_padding=False，MTE2 与 Vector 并行
    data = tl.load(input + idx, mask=mask, care_padding=False)
```

### 示例2：多输入算子优化

```python
@triton.jit
def fused_bias_gelu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # x 和 bias 的 padding 区域都不会被使用
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    bias = tl.load(bias_ptr + offsets, mask=mask, care_padding=False)

    # GELU 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_bias = x + bias
    result = x_bias * 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (x_bias + 0.044715 * x_bias * x_bias * x_bias)))

    tl.store(out_ptr + offsets, result, mask=mask)
```

### 示例3：不能使用 care_padding=False 的场景

```python
@triton.jit
def sum_kernel_unsafe(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # 错误！padding 区域的随机值会被 sum 累加
    # x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)

    # 正确做法：使用 other=0.0 确保 padding 区域为 0
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    result = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, result)
```

### 示例4：结合 for 循环 Tiling 使用

```python
@triton.jit
def process_kernel(
    input_ptr, output_ptr, n,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n

        # for 循环 + care_padding=False = 最佳存算并行
        data = tl.load(input_ptr + offsets, mask=mask, care_padding=False)
        result = data * 2.0 + 1.0
        tl.store(output_ptr + offsets, result, mask=mask)
```

## NPU 适配要点

1. **默认行为与 GPU 一致**：`care_padding=True` 时，NPU 的行为与 GPU 完全一致，padding 区域填充 0
2. **care_padding=False 是 NPU 专属优化**：GPU 上不存在此参数，迁移时需注意
3. **优先在访存密集型算子中使用**：计算密集型算子的瓶颈不在搬运，care_padding 优化效果有限
4. **结合 for 循环使用效果更佳**：for 循环 Tiling + care_padding=False 能最大化存算并行
5. **安全性第一**：不确定时保持 `care_padding=True`，确认安全后再改为 False

## 常见问题 (Q&A)

**Q1: care_padding=False 会导致结果错误吗？**

A: 如果 padding 区域的数据不影响最终计算结果，则不会。关键判断标准是：mask=False 位置的数据是否会被后续计算使用或被 store 写出。如果 store 时使用了相同的 mask 过滤，且中间计算不依赖 padding 值做 reduction，则是安全的。

**Q2: care_padding 和 other 参数有什么关系？**

A: 当 `other` 不为 None 时，`care_padding` 不生效，因为 `other` 已经显式指定了填充值。只有当 `other=None` 时，`care_padding` 才有意义：`care_padding=True` 填充 0，`care_padding=False` 不填充。

**Q3: 如何验证 care_padding=False 是否安全？**

A: 对比优化前后的计算结果，如果完全一致则安全。也可以分析算子逻辑：检查 padding 区域的值是否被 store 写出，是否参与 reduction 等全局计算。

**Q4: care_padding=False 能提升多少性能？**

A: 取决于算子特征。对于访存密集型算子，提升可达 10%-30%，因为消除了 Vector 初始化的同步开销。对于计算密集型算子，提升较小。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [06-data-movement-optimization.md](./06-data-movement-optimization.md) - 数据搬运优化
- [07-profiling-guide.md](./07-profiling-guide.md) - 性能分析与瓶颈定位

### 源码参考

- [core.py - tl.load care_padding 参数](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1621-L1638)
- [semantic.py - load 函数](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1198-L1213)
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md) - 指令并行优化
