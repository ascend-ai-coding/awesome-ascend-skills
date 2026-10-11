# Reduce 算子优化

> 适用于需要聚合多个值的归约操作

## 适用算子

**基础归约**: sum, mean, max, min, prod
**归一化**: softmax, logsoftmax, layernorm, batchnorm
**统计**: variance, std

## 通用归约策略

### 1. 块内归约 + 原子操作

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作写回全局内存
    tl.atomic_add(output_ptr, block_sum)
```

### 2. 减少规约精度损失

**关键**: 如果需要在 FP16 或 BF16 的数据上执行计算性规约（除了max, min的规约计算），应在规约计算前将其强制转换为 FP32，以避免低精度累加带来的数值误差。

```python
# 错误：直接用 fp16/bf16 累加，精度损失大
data = tl.load(input_ptr + offsets, mask=mask, other=0.0)  # data 为 fp16/bf16
block_sum = tl.sum(data, axis=0)  # 低精度累加
carry = carry + block_sum  # 低精度累加

# 正确：在执行累加计算前转为 fp32，在 fp32 上完成规约
data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
data = data.to(tl.float32)        # 强制提升为 fp32
block_sum = tl.sum(data, axis=0)  # 高精度累加
carry = carry + block_sum  # 高精度累加

# 如果输出要求 fp16/bf16，在最终 store 前转回
tl.store(output_ptr, block_sum.to(input_ptr.dtype.element_ty))
```

**原则**：
- 在执行规约操作前 `.to(tl.float32)`
- 如果涉及多次规约，累积多次规约结果的累加器对象精度应为`tl.float32`
- 涉及计算的规约操作（除了max, min的规约操作）均在 FP32 上执行
- 在最后 `tl.store` 前按需转回原始数据类型

### 3. 数值稳定性处理

**关键**: 对于涉及 exp 的操作（softmax、logsoftmax），必须减去最大值防止溢出。

```python
# 错误：错误：直接 exp 可能溢出
scores = tl.math.exp2(x)

# 正确：正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.math.exp2(x - max_val)
```

## 4. 小输出表规约（Histogram / Bincount / 小输出 Scatter）

### 适用特征

- 每个输入元素通过标量映射落入一个小输出表（通常 `bins <= 256`）。
- 存在跨线程/跨核写冲突，直接使用全局 `atomic_add` 会严重竞争。
- 输出表足够小，可完整放进一个 program 的 UB / 寄存器空间。

### 默认架构：dual-kernel local + reduce

```text
Kernel 1 (local): 每个 program/core 负责一段输入，维护一个局部小表
                  输出 partial[num_cores, bins]
Kernel 2 (reduce): 每个 bin 一个 program，汇总所有 core 的 partial
                  输出 hist[bins]
```

### 实现模式 A：Match-Matrix（推荐 bins <= 128）

骨架：

```text
local_hist = zeros([bins])

for each block in core's input partition:
    x_tile = load(block)
    bin_idx = compute_bin_index(x_tile)        # fp32 计算后 clamp 到 [0, bins-1]
    valid  = compute_valid_mask(x_tile)        # 边界 + NaN 检查

    # 构造 [BLOCK_SIZE, bins] 布尔匹配矩阵
    matches = (bin_idx[:, broadcast_to_bins] == bin_range[broadcast_to_block]) & valid[:, broadcast_to_bins]
    counts  = sum(matches, axis=BLOCK_SIZE_DIM)

    local_hist += counts

store(local_hist -> partial[core_idx, :])
```

- 优点：完全向量化，无动态索引。
- 缺点：`BLOCK_SIZE * bins` 矩阵面积随 bins 增大而增大，容易 UB 溢出。

### 实现模式 B：Conditional Increment

骨架：

```text
local_hist = zeros([bins])

for each block in core's input partition:
    x_tile = load(block)
    bin_idx = compute_bin_index(x_tile)        # fp32 计算后 clamp
    valid  = compute_valid_mask(x_tile)

    for each element j in block:
        if valid[j]:
            local_hist[bin_idx[j]] += 1

store(local_hist -> partial[core_idx, :])
```

- 优点：无额外矩阵面积。
- 缺点：依赖编译器把动态索引优化成 scalar/vector 混合路径；部分 NPU 编译器会标量降级。

### 选择建议

| bins 范围 | 推荐模式 | 原因 |
|---|---|---|
| <= 64 | 优先 match-matrix，可尝试 BLOCK_SIZE=64/128 | 矩阵面积小，向量化收益大 |
| 65~128 | match-matrix 或 conditional increment | 按 UB 余量和编译器行为选择 |
| > 128 | 优先 conditional increment，match-matrix 易 UB 溢出 | 矩阵面积过大 |

### Reduce Kernel 骨架

```text
bin_idx = program_id(BIN_AXIS)
vals    = load(partial[valid_cores, bin_idx])
total   = sum(vals)
store(total -> hist[bin_idx])
```

### Host 侧调度骨架

```text
x_flat = flatten(x)
N      = numel(x_flat)
num_cores = dynamic_core_count(N, BLOCK_SIZE, PHYSICAL_CORES)
partial   = allocate([num_cores, bins])

launch local_kernel[grid=num_cores](x_flat, partial, ...)
launch reduce_kernel[grid=bins](partial, hist, ...)
```

### 关键约束

1. **首版禁止全局原子累加**：小输出表规约必须先走 dual-kernel local + reduce。
2. **`MAX_BINS` 必须作为编译期常量参数传入**，值取运行时 `bins`。
3. **bin 索引计算在 fp32 上完成**，最后 clamp 到 `[0, bins-1]`。
4. **局部表必须零初始化**。
5. **输入必须展平**。
6. **动态核数**：小 shape 1 核，大 shape 扩展到物理核数。
