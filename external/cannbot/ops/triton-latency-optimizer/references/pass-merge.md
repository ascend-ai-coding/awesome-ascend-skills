# 减少遍历优化

## 概述

减少数据遍历次数是 Triton 性能优化的核心手段。有两种主要方法：

| 方法 | 适用场景 | 核心思路 |
|------|---------|---------|
| **Pass 合并** | 多个统计量计算 | 同时计算多个统计量，减少遍历次数 |
| **循环消除** | 数据量小 (N ≤ BLOCK_SIZE) | 一次加载所有数据，消除循环 |

---

## 方法一：Pass 合并

### 问题描述

**问题：** 多次遍历数据，每次独立计算统计量，导致重复内存访问。

```python
# 问题代码：3-pass BatchNorm
# Pass 1: 计算 mean
for ...:
    data = tl.load(...)
    mean += tl.sum(data)

# Pass 2: 计算 variance（再次遍历！）
for ...:
    data = tl.load(...)  # 重复加载
    var += tl.sum((data - mean) ** 2)

# Pass 3: 归一化（第三次遍历！）
for ...:
    data = tl.load(...)  # 第三次加载
    tl.store(...)
```

### 优化原理

利用数学公式，在单次遍历中同时计算多个统计量：

```
mean = sum(x) / count
var = sum(x²) / count - mean²

证明：
var = E[(x - mean)²]
    = E[x²] - 2·mean·E[x] + mean²
    = E[x²] - mean²
    = sum(x²)/count - mean²
```

### 优化代码

```python
# Pass 1: 同时计算 sum 和 sum_sq
sum_val = 0.0
sum_sq = 0.0

for ...:
    data = tl.load(...)
    sum_val += tl.sum(data)
    sum_sq += tl.sum(data * data)  # 同时累加

mean = sum_val / count
var = sum_sq / count - mean * mean

# Pass 2: 归一化
for ...:
    ...
```

### 案例：BatchNorm2d

```python
@triton.jit
def batchnorm_2pass(
    input_ptr, output_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, stride_n, stride_c, stride_h, stride_w,
    eps: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr,
):
    c = tl.program_id(0)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    count = N * H * W

    # Pass 1: 同时计算 sum 和 sum_sq
    sum_val = 0.0
    sum_sq = 0.0

    for n in range(N):
        for h in range(H):
            for w_start in range(0, W, BLOCK_SIZE_HW):
                data = tl.load(...)
                sum_val += tl.sum(data)
                sum_sq += tl.sum(data * data)

    mean = sum_val / count
    var = sum_sq / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize
    for n in range(N):
        for h in range(H):
            for w_start in range(0, W, BLOCK_SIZE_HW):
                data = tl.load(...)
                output = (data - mean) * inv_std * gamma + beta
                tl.store(...)
```

**性能：** 3-pass → 2-pass，延迟从 73.67ms → ~52ms，**加速 1.42x**

---

## 方法二：循环消除

### 问题描述

**问题：** Triton 对 Python for 循环优化有限，大量循环是性能杀手。

```python
# 问题代码：循环多次加载
for col_offset in range(0, n_cols, BLOCK_SIZE):
    vals = tl.load(...)  # 循环内加载
    max_val = update(max_val, vals)

for col_offset in range(0, n_cols, BLOCK_SIZE):
    vals = tl.load(...)  # 再次加载
    sum_exp = update(sum_exp, vals)

for col_offset in range(0, n_cols, BLOCK_SIZE):
    vals = tl.load(...)  # 第三次加载
    tl.store(...)
```

### 为什么 Triton 循环是性能杀手

| 循环次数 | 延迟 | 原因 |
|----------|------|------|
| 1 | ~1 ms | 无循环开销 |
| 10 | ~10 ms | 线性增长 |
| 100 | ~100 ms | 线性增长 |

**原因分析**:
1. **循环展开有限**: 编译器不会激进展开所有循环
2. **无法向量化**: 循环体被视为串行操作
3. **每次迭代独立编译**: 增加编译开销
4. **无法流水线**: 循环边界动态检查

### 优化原理

当 `N <= BLOCK_SIZE` 时，可以一次加载所有数据，消除循环：

```python
# 优化：一次加载，多次使用
BLOCK_SIZE = triton.next_power_of_2(N)  # 确保 N <= BLOCK_SIZE
vals = tl.load(...)  # 一次加载

# 直接在加载的数据上操作
max_val = tl.max(vals, axis=1, keep_dims=True)
sum_val = tl.sum(vals, axis=1, keep_dims=True)
output = compute(vals, max_val, sum_val)
tl.store(...)
```

### 案例：Log Softmax

#### 原始实现

```python
@triton.jit
def log_softmax_original(...):
    row_idx = tl.program_id(0)

    # Phase 1: 循环计算 max
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)
        max_val = tl.max(tl.maximum(vals, max_val))

    # Phase 2: 循环计算 sum
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)  # 第二次加载
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals)

    # Phase 3: 循环存储
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)  # 第三次加载
        output = vals - max_val - tl.log(sum_exp)
        tl.store(...)
```

**问题分析：**
- 3 次循环，3 次数据加载
- 每次 load 需要单独的内存访问
- Grid = (M,)，kernel launch 开销大

#### 优化实现

```python
@triton.jit
def log_softmax_optimized(
    input_ptr, output_ptr,
    stride_in, stride_out,
    n_rows, n_cols,
    BLOCK: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < n_rows
    col_offs = tl.arange(0, BLOCK)
    col_mask = col_offs < n_cols
    mask_2d = row_mask[:, None] & col_mask[None, :]

    # 一次加载所有数据
    x = tl.load(
        input_ptr + row_offs[:, None] * stride_in + col_offs[None, :],
        mask=mask_2d, other=-float('inf')
    )

    # 在同一数据上完成所有计算
    row_max = tl.max(x, axis=1, keep_dims=True)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    row_sum = tl.sum(tl.where(mask_2d, exp_x, 0.0), axis=1, keep_dims=True)
    output = x_shifted - tl.log(row_sum)

    tl.store(output_ptr + row_offs[:, None] * stride_out + col_offs[None, :], output, mask=mask_2d)
```

**性能：** 延迟从 82.32μs → 7.97μs，**加速 10.3x**

### 性能收益

| 场景 | 原始 | 优化后 | 收益 |
|-----|------|--------|------|
| Softmax (3 phase) | 3 次加载 | 1 次加载 | **3x** |
| LayerNorm (2 phase) | 2 次加载 | 1 次加载 | **2x** |
| Log Softmax | 82.32 μs | 7.97 μs | **10.3x** |

---

## 两种方法对比

| 对比项 | Pass 合并 | 循环消除 |
|-------|---------|---------|
| **适用条件** | 任意数据量 | N ≤ BLOCK_SIZE |
| **优化对象** | 多个统计量计算 | 循环结构 |
| **核心操作** | 同时计算 sum+sum_sq 等 | 一次加载多次使用 |
| **收益来源** | 减少遍历次数 | 减少加载次数 + 减少 kernel launch |
| **可组合性** | 可与维度合并组合 | 可与多行并行组合 |

### 适用条件

**Pass 合并：**
- ✅ 多个统计量可同时计算（mean+var, sum+sum_sq）
- ❌ 统计量之间有依赖关系（Softmax 的 sum 依赖 max）

**循环消除：**
- ✅ N ≤ BLOCK_SIZE（可一次性加载）
- ❌ N > BLOCK_SIZE（需保留循环累积）

---

## 大数据量处理

当数据量超过 BLOCK_SIZE 时，需要保留循环但优化为累积式：

```python
# 累积式循环模板
@triton.jit
def kernel_large_n(
    input_ptr, output_ptr,
    stride_in, stride_out,
    n_rows, n_cols,
    BLOCK: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < n_rows

    # 初始化累加器
    max_acc = -float('inf')  # shape: [ROWS_PER_BLOCK, 1]
    sum_acc = 0.0

    # 循环处理列
    for col_start in range(0, n_cols, BLOCK):
        col_offs = col_start + tl.arange(0, BLOCK)
        col_mask = col_offs < n_cols
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # 加载当前块
        x = tl.load(
            input_ptr + row_offs[:, None] * stride_in + col_offs[None, :],
            mask=mask_2d, other=-float('inf')
        )

        # 累积统计量
        block_max = tl.max(x, axis=1, keep_dims=True)
        # ... 使用 Welford 算法或其他累积方法

    # 最终处理...
```

**关键点：** 循环内不要重复加载相同数据，而是在循环内累积结果。

---

## 其他常见应用

### LayerNorm：2-pass → 1-pass

```python
# 原始：2-pass
for i in range(N):
    mean += x[i]
mean /= N

for i in range(N):
    var += (x[i] - mean) ** 2
var /= N

# 优化：1-pass
for i in range(N):
    sum_val += x[i]
    sum_sq += x[i] ** 2
mean = sum_val / N
var = sum_sq / N - mean ** 2
```

### Softmax：3-pass → 2-pass

```python
# 原始：3-pass (max + sum + normalize)
for i in range(N):
    max_val = max(max_val, x[i])

for i in range(N):
    sum_exp += exp(x[i] - max_val)

for i in range(N):
    output[i] = exp(x[i] - max_val) / sum_exp

# 注意：无法进一步合并，因为 sum 依赖 max
```

---

## 常见错误

### 错误 1：忘记调整 BLOCK_SIZE

```python
# ❌ 错误：BLOCK_SIZE 不够大
BLOCK_SIZE = 256  # 如果 N = 512，会丢失数据
vals = tl.load(...)

# ✅ 正确：确保 N <= BLOCK_SIZE
BLOCK_SIZE = triton.next_power_of_2(N)
```

### 错误 2：忘记更新除数

```python
# ❌ 错误：有 mask 时仍用固定 count
sum_val += tl.sum(data)
mean = sum_val / (N * H * W)  # 实际元素数可能更少

# ✅ 正确：跟踪实际元素数
valid_count = tl.sum(mask)
mean = sum_val / valid_count
```

### 错误 3：循环内重复加载

```python
# ❌ 错误：循环内多次加载同一数据
for col_start in range(0, n_cols, BLOCK):
    x = tl.load(...)
    max_val = update(max_val, x)

for col_start in range(0, n_cols, BLOCK):
    x = tl.load(...)  # 重复加载！
    sum_val = update(sum_val, x)

# ✅ 正确：循环内累积，减少加载
for col_start in range(0, n_cols, BLOCK):
    x = tl.load(...)
    max_val = update(max_val, x)
    sum_val = update(sum_val, f(x))  # 同时处理
```

---

## 总结

| 方法 | 适用场景 | 核心操作 | 收益 |
|------|---------|---------|------|
| Pass 合并 | 多统计量计算 | 同时计算 sum+sum_sq | 减少遍历次数 |
| 循环消除 | N ≤ BLOCK_SIZE | 一次加载多次使用 | 减少加载 + 减少 launch |

**选择依据：**
- 数据量小：优先循环消除
- 数据量大 + 多统计量：优先 Pass 合并
- 两者可组合使用

**核心原则：**
- 减少数据加载次数
- 寻找可同时计算的统计量
- 循环内累积而非重复加载
