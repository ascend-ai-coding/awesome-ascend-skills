# 混合策略自动选择

## 概述

同一算子在不同 shape 或数据类型下可能需要不同的优化策略。通过在 Host 侧根据运行时特征自动选择最优策略，可以确保算子在各种场景下都达到最佳性能。

## 触发条件

**当代码中存在以下特征时，应考虑混合策略自动选择：**

1. **Shape 相关的条件分支**：代码中已存在根据 shape 选择不同 kernel 的逻辑
2. **数据类型相关的条件分支**：不同数据类型需要不同的计算策略
3. **性能瓶颈差异**：不同 shape 下的主要性能瓶颈不同（如 small grid vs large grid）

## 优化方法

### 典型策略组合

#### 策略 A：Small batch / Small groups → 并行规约

```python
# 适合小 batch 场景：使用 atomic_add 并行规约
@triton.jit
def reduce_small_batch(
    input_ptr, output_ptr,
    batch_size, group_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * group_size
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # 使用 atomic_add 并行写入结果
    tl.atomic_add(output_ptr + (offsets // group_size), data, mask=mask)
```

#### 策略 B：Large batch / Large groups → 原始规约

```python
# 适合大 batch 场景：避免 atomic 开销，使用串行规约
@triton.jit
def reduce_large_batch(
    input_ptr, output_ptr,
    batch_size, group_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 每个 program 处理一个完整的 group
    group_idx = pid
    sum_val = 0.0
    
    for i in range(0, group_size, BLOCK_SIZE):
        offsets = group_idx * group_size + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (group_idx + 1) * group_size
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(data)
    
    tl.store(output_ptr + group_idx, sum_val)
```

#### 策略 C：FP32 → 禁用改变求和顺序的优化

```python
# FP32 场景：保持精确求和顺序，避免精度损失
@triton.jit
def sum_fp32_exact(
    input_ptr, output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 严格按顺序累加，不启用并行优化
    ...
```

#### 策略 D：FP16/BF16 → 可启用并行优化

```python
# FP16/BF16 场景：可启用并行优化，利用低精度的高吞吐
@triton.jit
def sum_fp16_parallel(
    input_ptr, output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 启用并行规约，利用 fp16 的高带宽
    ...
```

### Host 侧动态选择

```python
class Model(nn.Module):
    def forward(self, x, dim=-1):
        batch_size = x.shape[0]
        group_size = x.shape[dim]
        dtype = x.dtype
        
        # 策略选择逻辑
        if batch_size * group_size < 4096:
            # 小数据量：并行规约
            grid = (num_cores,)
            kernel = reduce_small_batch
        else:
            # 大数据量：串行规约，避免 atomic 开销
            grid = (batch_size,)
            kernel = reduce_large_batch
        
        # 数据类型策略
        if dtype == torch.float32:
            # FP32：禁用改变顺序的优化
            pass
        elif dtype in (torch.float16, torch.bfloat16):
            # FP16/BF16：可启用并行优化
            pass
        
        kernel[grid](x, output, ...)
        return output
```

## 关键点

1. **策略选择依据**：
   - **Shape 阈值**：根据 batch_size、group_size、total_elements 等选择
   - **数据类型**：fp32 需要精度保护，fp16/bf16 可启用更多优化
   - **硬件特征**：根据 NPU 核数、带宽等选择

2. **策略切换开销**：
   - Host 侧条件判断开销极小（微秒级）
   - 确保策略切换的收益远大于开销

3. **策略一致性**：
   - 不同策略的计算结果必须一致（精度允许范围内）
   - 边界处理逻辑必须等价

## 性能收益

- **Small batch 场景**：并行规约避免启动过多核，性能提升 1.5x~3.0x
- **Large batch 场景**：串行规约避免 atomic 竞争，性能提升 1.2x~2.0x
- **综合收益**：在 batch 分布广泛的场景中，几何平均加速比提升 10%~30%

## 常见错误

### 错误 1：策略选择条件与实际情况不符

```python
# ❌ 错误：条件判断错误，导致小 batch 走了大 batch 路径
if batch_size > 1:  # 几乎所有情况都走大 batch 路径
    kernel = reduce_large_batch

# ✅ 正确：基于 workload 特征选择
if batch_size * group_size < 4096:
    kernel = reduce_small_batch
else:
    kernel = reduce_large_batch
```

### 错误 2：策略间结果不一致

```python
# ❌ 错误：两条路径的边界处理不一致，导致结果差异
# 路径 A：mask=offsets < total
# 路径 B：mask=offsets < (group_idx + 1) * group_size

# ✅ 正确：确保边界处理逻辑等价
```

### 错误 3：过多策略导致维护困难

```python
# ❌ 错误：4 条以上路径，维护成本高，收益递减
if condition_a:
    kernel_a
elif condition_b:
    kernel_b
elif condition_c:
    kernel_c
elif condition_d:
    kernel_d

# ✅ 正确：2-3 条路径覆盖主要场景
if small_workload:
    kernel_small
else:
    kernel_large
```

## 适用算子

| 算子类型 | 适用性 | 典型策略组合 |
|----------|--------|-------------|
| Reduction | 高 | small→atomic, large→serial |
| Normalization | 高 | small→parallel stats, large→two-pass |
| Element-wise | 中 | fp32→exact, fp16→fast |
| Matmul | 低 | 策略复杂，一般由 autotune 覆盖 |

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| 混合策略 | Host 侧动态选择 | 不同场景下均达到最优 |
| 自动选择 | 基于 shape/dtype 条件 | 消除单一策略的短板 |

**核心原则：**
1. 策略数量：2-3 条路径即可覆盖大部分场景
2. 选择依据：基于 workload 特征（shape、dtype、硬件）
3. 结果一致性：不同策略的计算结果必须等价

---

## 来自 SKILL.md 的原始描述（优化点 14：混合策略自动选择）

**适用条件**：同一算子在不同 shape 或数据类型下需要不同优化策略

**典型代码特征**：
```python
# 问题：单一策略无法覆盖所有 shape
if some_condition:
    # 策略 A: 适合小 shape
    kernel_a[grid](...)
else:
    # 策略 B: 适合大 shape
    kernel_b[grid](...)
```

**判断逻辑**：
- 检查是否存在 shape 相关的条件分支选择不同 kernel
- 检查是否存在数据类型相关的条件分支选择不同策略
- 检查不同策略是否针对不同的性能瓶颈（如 small grid vs large grid）
- 若存在 → 涉及

**参考策略**：
- small batch / small groups → 并行规约（atomic_add）
- large batch / large groups → 原始规约（避免 atomic 开销）
- fp32 → 禁用改变求和顺序的优化
- fp16/bf16 → 可启用并行优化

**命中条件**：代码中存在 shape 或数据类型相关的条件分支选择不同 kernel 或策略

**参考文档**：`references/mixed_strategy.md`

---
