# 减少遍历优化

## 概述

减少数据遍历次数是 Triton 性能优化的核心手段。有两种主要方法：

**优化优先级：循环消除 > Pass 合并**

| 优先级 | 方法 | 适用场景 | 核心思路 | 收益 |
|------|------|---------|---------|------|
|**P0(优先)**| **循环消除** | 数据量小，UB不溢出 | 自适应计算`BLOCK_SIZE`, 且满足`BLOCK_SIZE ≥ N`，单次加载消除所有循环 | **最高**：同时消除循环 + 合并 Pass |
|**P1(兜底)**| **Pass 合并** | 数据量大，UB受限 | 同时计算多个统计量 | 减少遍历次数 |

**决策流程：**
1. 先检查当前 `BLOCK_SIZE` 是否为固定值（如 `BLOCK_SIZE: tl.constexpr = 1024` 或调用侧固定传入）
2. 自适应计算 `BLOCK_SIZE = triton.next_power_of_2(N)`
   - **即使当前固定 BLOCK_SIZE 已 ≥ N，也必须执行此调整**：固定 BLOCK_SIZE 在 N 较小时会产生大量无效 mask 计算，浪费 Vector 周期
3. 然后检查 UB 约束: BLOCK_SIZE × dtype_size × (input + output + 中间变量峰值) ≤ 192KB
4. 若通过 → 应用循环消除（P0）
5. 若不通过 → 应用 Pass 合并（P1）

#### 循环消除的自适应 BLOCK_SIZE 计算
  
```python
# 根据实际维度 N 自适应计算最小覆盖 BLOCK_SIZE
BLOCK_SIZE = triton.next_power_of_2(N)  # 或 1 << (N - 1).bit_length()

# UB 安全检查（以 float32 为例）
max_elements = 192 * 1024 // dtype_size // (input + output + 中间变量峰值)
if BLOCK_SIZE > max_elements:
    # UB 溢出，回退到 Pass 合并
    use_pass_merge = True
```

---

## 方法一：循环消除(优先尝试)

### 问题描述

**问题：** Triton 对 Python for 循环优化有限，大量循环是性能杀手。当数据量 `N` 可通过自适应调整 `BLOCK_SIZE` 一次性覆盖时，必须同时将 `BLOCK_SIZE` 从固定值改为调用侧自适应计算。

**⚠️ 特别注意：即使当前固定的 `BLOCK_SIZE` 已经 ≥ `N`（循环本来就只迭代一次），仍然必须改为自适应计算。** 原因是：
- 若 `BLOCK_SIZE=1024` 而 `N=64` 时，`tl.arange(0, 1024)` 会产生 960 个无效 mask 位置
- 这些无效位置仍会被 Vector 单元处理，造成大量空转周期
- 同时占用更多 UB 内存，降低单个 AICore 的并行处理能力
- 自适应计算 `BLOCK_SIZE = triton.next_power_of_2(N) = 64` 可完全消除无效计算

```python
# 问题代码：循环多次加载，且 BLOCK_SIZE 固定
BLOCK_SIZE = 1024
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
| 512 | ~700 ms | **20x 慢于无循环版本** |

**原因分析**:
1. **循环展开有限**: 编译器不会激进展开所有循环
2. **无法向量化**: 循环体被视为串行操作
3. **每次迭代独立编译**: 增加编译开销
4. **无法流水线**: 循环边界动态检查

### 优化原理

- 当分块大小 `BLOCK_SIZE >= N` 时，则range(0, N, BLOCK_SIZE)仅迭代一次，可消除循环；
- 然后必须将 `BLOCK_SIZE` 从固定值改为调用侧自适应计算 `BLOCK_SIZE = triton.next_power_of_2(N)`，要不然存在无效计算；

**原始代码（多次for循环遍历）**

```python
@triton.jit
def example_kernel(..., N, BLOCK_SIZE: tl.constexpr, ...):
    # First pass
    for n_start in range(0, N, BLOCK_SIZE):
        n_offset = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offset < N
        x = tl.load(row_start + n_offset, mask=mask, other=0.0)
        tmp1 += tl.sigmoid(x, ...)
        ...
    tmp1_sum = tl.sum(tmp1, ...)

    # Second pass
    for n_start in range(0, N, BLOCK_SIZE):
        n_offset = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offset < N
        x = tl.load(row_start + n_offset, mask=mask, other=0.0)
        tmp2 = tl.exp(x - tmp1_sum, ...)
        ...

    # Third pass
    for n_start in range(0, N, BLOCK_SIZE):
        n_offset = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offset < N
        x = tl.load(row_start + n_offset, mask=mask, other=0.0)
        ...

# Python 调用侧：BLOCK_SIZE为固定值
example_kernel[grid](..., N, BLOCK_SIZE=256, ...)
```

**优化后代码（BLOCK_SIZE 自适应，消除 for 循环）**

`BLOCK_SIZE` 保持 `tl.constexpr` 声明（`tl.arange` 需要编译时已知大小），但在 **Python 调用侧**根据 `N` 自适应计算后传入：

```python
@triton.jit
def example_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr, 
):
    n_offset = tl.arange(0, BLOCK_SIZE)
    mask = n_offset < N
    x = tl.load(input_ptr + n_offset, mask=mask, other=0.0)

    tmp1 = tl.sigmoid(x)
    tmp1_sum = tl.sum(tmp1)
    tmp2 = tl.exp(x - tmp1_sum)
    tl.store(output_ptr + n_offset, tmp2, mask=mask)

# Python 调用侧：根据实际 N 自适应计算 BLOCK_SIZE
BLOCK_SIZE = triton.next_power_of_2(N)  # 确保 BLOCK_SIZE >= N
kernel[grid](input_ptr, output_ptr, N, BLOCK_SIZE=BLOCK_SIZE)
```

**关键要点**：
- Kernel 内部声明 `BLOCK_SIZE: tl.constexpr`，但不给固定默认值
- Python 侧使用 `triton.next_power_of_2(N)` 计算最小覆盖大小
- 检查 UB 约束后再调用 kernel，避免 UB 溢出

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

# Python 调用侧：根据实际 n_cols 自适应选择 BLOCK
BLOCK = triton.next_power_of_2(n_cols)
kernel[grid](input_ptr, output_ptr, stride_in, stride_out, n_rows, n_cols, BLOCK=BLOCK, ROWS_PER_BLOCK=ROWS_PER_BLOCK)
```

**性能：** 延迟从 82.32μs → 7.97μs，**加速 10.3x**

### 性能收益

| 场景 | 原始 | 优化后 | 收益 |
|-----|------|--------|------|
| Softmax (3 phase) | 3 次加载 | 1 次加载 | **3x** |
| LayerNorm (2 phase) | 2 次加载 | 1 次加载 | **2x** |
| Log Softmax | 82.32 μs | 7.97 μs | **10.3x** |

---

## 方法二：Pass 合并

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

利用数学公式（如online softmax），在单次遍历中同时计算多个统计量：

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

# 优化：2-pass
max_val = float('-inf')
sum_exp = 0.0
for i in range(N):
    old_max = max_val
    max_val = max(max_val, x[i])
    sum_exp *= exp(old_max - max_val)
    sum_exp += exp(x[i] - max_val)

for i in range(N):
    output[i] = exp(x[i] - max_val) / sum_exp
```

---

## 常见错误

### 错误 1：循环消除时未同步调整 BLOCK_SIZE（最常见错误）

**⚠️ 循环消除必须与自适应 BLOCK_SIZE 一起完成，禁止单独执行其中之一。**

```python
# ❌ 错误 1a：只把 for 循环去掉，但 BLOCK_SIZE 仍然是固定值 256
# 后果：当 N=64 时，tl.arange(0, 256) 产生大量无效掩码计算，未真正利用循环消除优势
@triton.jit
def kernel(..., N, BLOCK_SIZE: tl.constexpr = 256):
    # 原本有 for n_start in range(0, N, BLOCK_SIZE):  被错误地直接删掉
    n_offset = tl.arange(0, BLOCK_SIZE)
    mask = n_offset < N
    x = tl.load(input_ptr + n_offset, mask=mask, other=0.0)
    ...

kernel[grid](..., BLOCK_SIZE=256)  # 调用侧仍是固定值

# ❌ 错误 1b：kernel 内给 BLOCK_SIZE 固定默认值，调用侧未根据 N 自适应传入
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr = 256):  # 固定值
    for n in range(0, N, BLOCK_SIZE):  # 当 N 变化时无法消除循环
        ...

# ❌ 错误 1c：BLOCK_SIZE 作为普通参数（非 tl.constexpr），tl.arange 编译失败
@triton.jit
def kernel(..., BLOCK_SIZE):  # 缺少 tl.constexpr
    offs = tl.arange(0, BLOCK_SIZE)  # 编译错误：需要编译时常量

# ❌ 错误 1d：调用侧忘记根据 N 计算 BLOCK_SIZE
kernel[grid](..., BLOCK_SIZE=256)  # 始终固定 256

# ✅ 正确：kernel 声明为 tl.constexpr 但不设默认值，调用侧自适应传入，且循环与 BLOCK_SIZE 调整同步完成
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    ...

BLOCK_SIZE = triton.next_power_of_2(N)
kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)
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

---

## 来自 SKILL.md 的原始描述（优化点 7：Pass 消除合并优化）

**适用条件**：代码中存在多次遍历相同数据计算不同统计量

**典型代码特征**：
```python
# 特征 1：多个独立循环遍历相同数据
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

# 特征 2：kernel调用侧未根据实际 N 自适应计算 BLOCK_SIZE，而是传入固定值（如BLOCK_SIZE=1024）
@triton.jit
def kernel(..., N, BLOCK_SIZE: tl.constexpr):
    for n_start in range(0, N, BLOCK_SIZE):  # 当 BLOCK_SIZE >= N 时可消除循环
        ...

kernel(..., N, BLOCK_SIZE=1024)
```

**判断逻辑**：
- 检查是否可以通过自适应计算 `BLOCK_SIZE` 消除循环：
  - 如果 `BLOCK_SIZE` 当前是固定的 `tl.constexpr` 或者调用侧传入了固定值，而实际数据维度 `N` 是变量
  - **无论当前 BLOCK_SIZE 是否已 >= N**：
    - 若当前 BLOCK_SIZE < N：令 `BLOCK_SIZE = triton.next_power_of_2(N)` 可使得 `range(0, N, BLOCK_SIZE)` 从多次迭代变为仅迭代一次
    - **若当前 BLOCK_SIZE 已 >= N：循环虽然只迭代一次，但固定 BLOCK_SIZE 在 N 较小时会产生大量无效 mask 计算（`tl.arange(0, 1024)` 仅前 64 个有效），浪费 Vector 单元周期，且可能占用过多 UB 影响并行度。必须将 BLOCK_SIZE 改为自适应计算。**
  - 若满足 UB 约束（`BLOCK_SIZE * dtype_size * (input + output + 中间变量峰值) <= 192KB`）
  - → 涉及，**必须同时执行：(a) 消除循环；(b) 将 `BLOCK_SIZE` 从固定值改为 Python 调用侧自适应计算后传入。二者缺一不可，禁止只做循环消除而保留固定 BLOCK_SIZE。**
- 检查是否存在多个独立的循环遍历相同数据
  - 检查是否可以同时计算多个统计量（如 sum + sum_sq 可同时计算 mean + var）
  - 如果存在多次遍历且可合并 → 涉及
- 如果只有单次遍历，或统计量之间有依赖无法合并 → 不涉及，跳过

**命中条件**：代码中存在多次遍历相同数据，可通过自适应计算 BLOCK_SIZE 实现循环消除；或者可以对多次遍历进行合并计算

**参考文档**：`references/pass-merge.md`

---
