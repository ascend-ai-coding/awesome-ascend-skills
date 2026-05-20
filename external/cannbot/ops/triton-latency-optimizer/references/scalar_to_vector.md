# Scalar 转 Vector 优化模式

## 概述

在昇腾 NPU Triton kernel 开发中，scalar 操作会占用 Scalar 计算单元，无法充分利用 NPU 的 Vector 计算单元（支持 256 字节 SIMD 并行）。通过将 scalar 操作转换为 vector 操作，可以显著提升算子性能，尤其在数据并行计算场景中效果更为明显。

## 优化场景

当在代码中遇到以下模式时，可应用此优化：

1. **标量广播操作**：使用 Python 标量或标量指针与向量数据进行计算
2. **标量规约操作**：使用标量变量进行累加、累乘等规约计算
3. **标量控制流**：使用 if-else 分支处理向量数据
4. **标量索引计算**：使用标量循环生成内存访问索引
5. **标量数学函数**：使用 Python math 模块进行逐元素数学计算
6. **整数比较**：避免int类型的比较
7. **整数除法和取余**：避免int类型的除法和取余

## 优化方法

### 1. 标量广播 → 向量广播

**原始代码（scalar 计算）**

```python
scalar_val = 0.5  # Python 标量
x = tl.load(x_ptr + offsets, mask=mask)
result = x * scalar_val  # scalar 广播，无法启用 vector 加速
```

**优化后代码（vector 计算）**

```python
scalar_val = 0.5
x = tl.load(x_ptr + offsets, mask=mask)
scalar_vec = tl.full([BLOCK_SIZE], scalar_val, dtype=x.dtype)  # 转为 vector
result = x * scalar_vec  # 启用 vector 广播计算
```

### 2. 标量规约 → 向量分块规约

**原始代码（scalar 计算）**

```python
sum_val = 0.0  # 标量累加器
for n in range(N):
    val = tl.load(x_ptr + row_offset + n)
    sum_val += val  # 标量加法，循环依赖
result = sum_val
```

**优化后代码（vector 计算）**

```python
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # vector 累加器
for n_start in range(0, N, BLOCK_SIZE):
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    n_mask = n_offsets < N
    x_vec = tl.load(x_ptr + row_offset + n_offsets, mask=n_mask)
    acc += x_vec  # vector 加法，无循环依赖
result = tl.sum(acc, axis=0)  # vector 规约
```

### 2.1 2D 向量化规约

**原始代码（每个 program 只处理 1 个 expert，1D 标量比较）**

```python
@triton.jit
def count_naive(topk_ids_ptr, expert_num_tokens_ptr, num_experts, topk_numel,
                expert_map, HAS_EXPERT_MAP: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    curr_expert = tl.program_id(0)  # 标量索引
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)  # 1D 累加器
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        expert_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, ...)
        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)  # 标量比较
        acc = acc + has_curr_expert
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))
```

**优化后代码（每个 program 同时处理 EXPERT_BLOCK 个 expert，2D 向量化）**

```python
@triton.jit
def count_2d(topk_ids_ptr, expert_num_tokens_ptr, num_experts: tl.constexpr,
             topk_numel: tl.constexpr, expert_map, HAS_EXPERT_MAP: tl.constexpr,
             BLOCK_SIZE: tl.constexpr, EXPERT_BLOCK: tl.constexpr):
    # 向量索引：同时表示 EXPERT_BLOCK 个 expert
    curr_expert = tl.program_id(0) * EXPERT_BLOCK + tl.arange(0, EXPERT_BLOCK)
    offsets = tl.arange(0, BLOCK_SIZE)
    # 2D 累加器：(EXPERT_BLOCK, BLOCK_SIZE)
    acc = tl.zeros((EXPERT_BLOCK, BLOCK_SIZE), dtype=tl.float32)
    cntx = (topk_numel - 1) // BLOCK_SIZE + 1
    for x in range(cntx):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, mask=mask, other=-1)
        # 广播：(EXPERT_BLOCK, 1) == (1, BLOCK_SIZE) -> (EXPERT_BLOCK, BLOCK_SIZE)
        has_curr_expert = (expert_ids[None, :] == curr_expert[:, None]).to(tl.float32)
        has_curr_expert = tl.where(mask, has_curr_expert, 0.0)
        acc = acc + has_curr_expert
    # 沿 BLOCK_SIZE 轴规约，每行一个结果
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc, axis=1))
```

**关键变换**：
- `curr_expert` 从标量 → 向量（`tl.arange(0, EXPERT_BLOCK)`）
- `acc` 从 `(BLOCK_SIZE,)` 1D → `(EXPERT_BLOCK, BLOCK_SIZE)` 2D
- 比较操作通过广播实现 2D 向量化：`expert_ids[None, :] == curr_expert[:, None]`
- Grid = `(num_experts // EXPERT_BLOCK,)`，更接近物理核数


### 3. 标量控制流 → 向量掩码

**场景一：两个分支的计算逻辑的定义域一样，均不会出现nan或inf等无效输出**
**原始代码（scalar 控制流）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
if x > 0:  # 标量条件，导致 SIMD 分支分化
    result = tl.exp(x)
else:
    result = tl.cos(x)
```

**优化后代码（vector 掩码）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
cond_mask = x > 0  # vector 比较，返回布尔向量
exp_result = tl.exp(x)
cos_result = tl.cos(x)
result = cond_mask * exp_result + ~cond_mask * cos_result  # vector 加法，无分支
```

**场景二：两个分支的计算逻辑定义域不一样，至少有一个会出现nan或inf等无效输出**
**原始代码（scalar 控制流）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
if x > 0:  # 标量条件，导致 SIMD 分支分化
    result = tl.exp(x)
else:
    result = tl.log(x)
```

**优化后代码（vector 掩码）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
cond_mask = x > 0  # vector 比较，返回布尔向量
exp_result = tl.exp(x)
log_result = tl.log(x)
result = tl.where(cond_mask, exp_result, log_result)  # vector 选择，无分支
```

### 4. 标量索引 → 向量索引

**原始代码（scalar 索引）**


```python
for i in range(BLOCK_SIZE):
    idx = tl.load(indices_ptr + start + i)  # 标量加载
    val = tl.load(input_ptr + idx)  # 标量地址计算
    tl.store(output_ptr + start + i, val)
```

**优化后代码（vector 索引）**

```python
# 方案一：先将indices和input都搬到ub，再使用tl.gather收集并输出，适用于input较小的场景，加速比较大
indices_offsets = indices_start + tl.arange(0, BLOCK_SIZE)
indices_mask = indices_offsets < indices_len
indices = tl.load(indices_ptr + indices_offsets, mask=indices_mask)
input_mask = indices_offsets < input_len
input_data = tl.load(input_ptr + indices_offsets, mask=input_mask)
gathered_data = tl.gather(input_data, indices, 0)
tl.store(output_ptr + indices_offsets, gathered_data, mask=indices_mask)

# 方案二：将indices搬到ub，再使用al.gather_out_to_ub扩展接口收集GM上的input数据，再进行输出，对UB空间占用较小，但是当前这个接口无法编译通过
# import triton.language.extra.cann.extension as al
# pid = tl.program_id(0)
# indices_start = pid * BLOCK_SIZE
# indices_offsets = indices_start + tl.arange(0, BLOCK_SIZE)
# indices_mask = indices_offsets < indices_len
# indices = tl.load(indices_ptr + indices_offsets, mask=indices_mask)
# gathered_data = al.gather_out_to_ub(
#     src=input_ptr,
#     index=indices,
#     index_boundary=input_len,
#     dim=0,
#     src_stride=(1, ),
#     end_offset=(indices_len, ),
#     start_offset=(0, )
# )
# tl.store(output_ptr + indices_offsets, gathered_data, mask=indices_mask)

```

### 5. 标量数学函数 → 向量数学函数

**原始代码（scalar 数学）**

```python
import math  # 错误：使用 Python math 模块

x = tl.load(x_ptr + offsets, mask=mask)
result = math.exp(x)  # 这将导致标量循环
```

**优化后代码（vector 数学）**
```python
x = tl.load(x_ptr + offsets, mask=mask)
result = tl.exp(x)  # vector 指数函数
# 其他 vector 函数：tl.log, tl.sqrt, tl.sin, tl.pow 等
```

### 6. int类型比较 → 向量比较

**优化前（整数比较退化为标量）**

```python
is_invalid_tok = tok < 0          # i64/i32类型
valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)   # 标量比较
```

**优化后（转换为浮点数启用向量）**

```python
is_invalid_tok = tok.to(tl.float32) < 0               # 转换float32类型
valid_block = (block_id.to(tl.float32) < max_num_blocks_per_req) & (block_id.to(tl.float32) >= 0)   # vector比较
```

### 7. int类型除法/取余 → 向量除法/取余

**优化前（整数除法/取余退化为标量）**

```python
c = a // b          # i32标量除法
d = a % b           # i32标量取余
```

**优化后（转换为浮点数启用向量）**

```python
c = a.to(tl.float32) // b.to(tl.float32)            # 转换float32类型
d = a - (a // b) * b  # 公式转换
```

## 关键点

1. **类型一致性**：确保 vector 操作中的数据类型一致，避免隐式类型转换回退到 scalar
2. **广播语义**：使用 `tl.full` 或 `tl.broadcast_to` 显式创建 vector 常量，避免 Python 标量自动广播
3. **规约融合**：使用 `tl.sum`, `tl.max`, `tl.min` 等内置 vector 规约函数，而非手动循环累加
4. **分支消除**：使用 `mask` 和加法结合 替代 if-else，避免标量控制流导致的 SIMD divergence
5. **内存对齐**：确保 BLOCK_SIZE 满足 128 字节对齐（如 FP16 类型下 BLOCK_SIZE 为 64 的倍数），以启用 vector 加载/存储
6. **索引比较类型转换**：对于 tl.where 中的整数索引比较，先 .to(tl.float32) 再比较，以启用向量比较指令。
7. **标量比较类型转换**：对于 `int32` 和 `int64` 整数标量比较，先 .to(tl.float32) 再比较，以启用向量比较指令。
8. **标量除法类型转换**：对于 `int32` 和 `int64` 中的整数除法操作，先 .to(tl.float32) 再计算，以启用向量计算指令。
9. **标量取余类型转换**：对于 `int32` 和 `int64` 中的整数取余操作， 使用`a - (a // b) * b`的形式计算，以启用向量计算指令。

## 性能收益

将 scalar 操作转换为 vector 操作，可在昇腾 NPU 上获得显著性能提升：

- **标量广播优化**：10-20% 加速，通过消除标量指令开销
- **标量规约优化**：5-128 倍加速（取决于 vector 并行度，FP16 理论加速比 128 倍）
- **控制流优化**：消除 SIMD 分支分化，提升 Vector 利用率至 90%+
- **整体 kernel 优化**：在 LayerNorm、Softmax 等带宽受限算子中，端到端性能提升 2-3 倍

**实测数据参考**：在 LayerNorm 算子中，将 mean/variance 计算从标量累加改为 vector 分块规约，UB 利用率从 35% 提升至 78%，kernel 执行时间减少 62%。
