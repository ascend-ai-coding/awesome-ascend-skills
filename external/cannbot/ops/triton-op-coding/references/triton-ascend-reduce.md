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
