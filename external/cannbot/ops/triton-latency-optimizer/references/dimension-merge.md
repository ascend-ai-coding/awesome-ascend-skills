# 维度合并优化

## 问题描述

**问题：** 多层嵌套循环，每层循环都有开销，重复计算索引。

```python
# 问题代码：3层循环
for n in range(N):           # 64 次
    for h in range(H):       # 512 次
        for w_start in range(0, W, BLOCK_SIZE):  # 1 次
            # 每次外层循环都要重新计算
            base_offset = n * stride_n + c * stride_c + h * stride_h
            data = tl.load(input_ptr + base_offset + ...)
```

**开销分析：**
- 循环嵌套层数：3 层
- 外层循环总次数：N × H = 64 × 512 = 32768 次
- base_offset 计算：32768 次
- 每次内层循环只有 1 次迭代（W=512, BLOCK_SIZE=512）

## 优化方案

**原理：** 合并连续维度，减少循环层数，提高内存访问连续性。

### 维度合并

```python
# 原始：分块 W 维度
for n in range(N):
    for h in range(H):
        for w_start in range(0, W, BLOCK_SIZE):
            ...

# 优化：合并 H 和 W 为 HW 维度
HW = H * W
for n in range(N):
    for hw_start in range(0, HW, BLOCK_SIZE):
        ...
```

### 索引转换

对于 NCHW contiguous 布局，合并后可简化为连续访问：

```python
# 原始：通过 h, w 分别计算索引
offsets = n * stride_n + c * stride_c + h * stride_h + w * stride_w

# 优化：连续线性访问
HW = H * W
channel_base = c * stride_c  # c * H * W
batch_offset = n * stride_n + channel_base
offsets = batch_offset + hw_offs  # 连续访问
```

## 案例：BatchNorm2d

### 原始实现

```python
@triton.jit
def batchnorm_original(
    input_ptr, output_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, stride_n, stride_c, stride_h, stride_w,
    eps: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr,
):
    c = tl.program_id(0)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    count = N * H * W
    mean = 0.0

    # 3层循环
    for n in range(N):
        for h in range(H):
            base_offset = n * stride_n + c * stride_c + h * stride_h
            for w_start in range(0, W, BLOCK_SIZE_HW):
                w_offs = w_start + tl.arange(0, BLOCK_SIZE_HW)
                mask = w_offs < W
                data = tl.load(input_ptr + base_offset + w_offs * stride_w, mask=mask, other=0.0)
                mean += tl.sum(data)

    mean = mean / count
    # ... 后续 pass 省略
```

### 维度合并后

```python
@triton.jit
def batchnorm_dim_merge(
    input_ptr, output_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, stride_n, stride_c, stride_h, stride_w,
    eps: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr,
):
    c = tl.program_id(0)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    # 维度合并
    HW = H * W
    channel_base = c * stride_c

    mean = 0.0

    # 2层循环
    for n in range(N):
        batch_offset = n * stride_n + channel_base
        for hw_start in range(0, HW, BLOCK_SIZE_HW):
            hw_offs = hw_start + tl.arange(0, BLOCK_SIZE_HW)
            mask = hw_offs < HW

            # 连续内存访问
            offsets = batch_offset + hw_offs
            data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            mean += tl.sum(data)

    count = (N * HW).to(tl.float32)
    mean = mean / count
    # ... 后续 pass 省略
```

### 性能对比

| 指标 | 原始 N→H→W | 优化 N→(HW) | 收益 |
|------|-----------|-------------|------|
| 循环层数 | 3 层 | 2 层 | 减少 1 层 |
| 外层循环次数 | N × H = 32768 | N = 64 | **减少 512 倍** |
| base_offset 计算 | 32768 次 | 64 次 | **减少 512 倍** |
| 内层循环迭代 | W/BS = 1 | HW/BS = 512 | 增加（但总循环次数减少） |
| 内存访问 | 间隔访问 | 连续访问 | 更好的局部性 |

**实测性能：**
- 原始 N→H→W：~52 ms（已做 pass 合并）
- 优化 N→(HW)：~20 ms
- **加速比：2.6x**

## 优势分析

### 1. 减少循环开销

```python
# 原始：N × H = 32768 次外层循环
for n in range(N):
    for h in range(H):
        # 循环开销：判断、跳转、计数

# 优化：N = 64 次外层循环
for n in range(N):
    # 循环开销减少 512 倍
```

### 2. 减少重复计算

```python
# 原始：每次 H 循环都要计算
for n in range(N):
    for h in range(H):
        base_offset = n * stride_n + c * stride_c + h * stride_h  # 32768 次

# 优化：只计算 N 次
for n in range(N):
    batch_offset = n * stride_n + channel_base  # 64 次
```

### 3. 提高内存连续性

```python
# 原始：间隔访问
# [h=0, w=0..W] [h=1, w=0..W] [h=2, w=0..W] ...
# 每个 H 之间可能有其他 channel 的数据

# 优化：连续访问
# [h=0..H, w=0..W] 一个大连续块
# 更好的缓存利用和内存带宽
```

### 4. 减少 mask 开销

```python
# 原始：每个 W 分块都要做 mask
for w_start in range(0, W, BLOCK_SIZE):
    mask = w_offs < W  # 每次都要计算

# 优化：只有边界需要 mask
for hw_start in range(0, HW, BLOCK_SIZE):
    mask = hw_offs < HW  # 大部分是全 True，只有最后一块需要
```

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | 维度间内存连续（NCHW 的 H×W） |
| ✅ 适用 | 维度间无依赖关系 |
| ⚠️ 注意 | 合并后索引计算是否正确 |
| ❌ 不适用 | 维度间有复杂依赖 |

## 常见错误

### 错误 1：忘记转换索引

```python
# ❌ 错误：HW 合并后仍用 w 索引
HW = H * W
for hw_start in range(0, HW, BLOCK_SIZE):
    hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
    offsets = ... + hw_offs * stride_w  # 错误！

# ✅ 正确：连续访问
offsets = batch_offset + hw_offs  # 直接加偏移
```

### 错误 2：非连续布局直接合并

```python
# ❌ 错误：非连续布局不能直接用线性索引
# 如果 tensor 被转置过，H×W 不是连续的

# ✅ 正确：先 ensure contiguous
if not x.is_contiguous():
    x = x.contiguous()
```

### 错误 3：合并有依赖的维度

```python
# ❌ 错误：某些算子 H 和 W 有依赖
# 如 convolution 中 kernel 需要分别访问 H 和 W

# ✅ 正确：只合并独立的维度
```

## 其他案例

### Softmax

```python
# 原始：按行处理
for row in range(M):
    for col_start in range(0, N, BLOCK_SIZE):
        ...

# 优化：不需要维度合并，本身就是 2D 问题
# 但可以多行并行处理
```

### Reduce

```python
# 原始：按维度 reduce
for i in range(N):
    for j in range(M):
        result[i] += data[i, j]

# 优化：合并内层维度
for i in range(N):
    for jm_start in range(0, M, BLOCK_SIZE):
        jm_offs = jm_start + tl.arange(0, BLOCK_SIZE)
        data = tl.load(input_ptr + i * M + jm_offs, ...)
        result[i] += tl.sum(data)
```

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| 维度合并 | 合并连续维度 | 减少循环层数 + 减少重复计算 + 连续访问 |

**核心：**
- 合并连续、独立的维度
- 利用 contiguous 内存布局
- 减少循环开销和重复计算
