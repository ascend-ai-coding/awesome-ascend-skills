# 分核优化

## 核心原则

**NPU 设备有多个 AI Core（通常 40 或 48 个），选择合适的核数是性能优化的关键。**

| 问题 | 影响 |
|------|------|
| Grid 远大于物理核数 | Kernel launch 开销大，调度开销大 |
| Grid 远小于物理核数 | 硬件利用率低，算力浪费 |
| Grid 大于实际需要（数据量小） | 部分核空转，产生不必要的调度开销 |
| Grid ≈ min(实际需要, 物理核数) | **最优** |

**关键洞察：** Grid 应该匹配"实际需要"和"物理核数"中的较小值。当数据量很小时，不需要启动全部物理核。

## 一、Grid 大小优化

### 问题：发射核数不合理

```python
# 问题1：Grid = (128,) 核数过多，远超 48 核
grid = (batch_size,)  # 如果 batch_size=128，远超 48 核
kernel[grid](...)

# 问题2：Grid = (2,) 核数过少
grid = (batch_size // 64,)  # 如果 batch_size=128，只有 2 核
kernel[grid](...)

# 问题3：固定启动 48 核，但数据量只需要 1 个 block
grid = (num_cores,)  # n_elements=100, BLOCK_SIZE=2048, 实际只需 1 个 block，却启动 48 个核
kernel[grid](...)
```

### 优化：动态匹配实际需要

**推荐方案：根据实际 block 数量动态计算 grid**

```python
# NPU 通常有 40 或 48 个物理核
num_cores = 48

# 计算总共需要多少个 block
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

# Grid 取实际需要 block 数和物理核数中的较小值
# 避免小 shape 下启动过多核产生空转调度开销
grid_size = min(num_blocks, num_cores)
grid = (grid_size,)
```

**对比：固定核数 vs 动态核数**

| 场景 | n_elements | BLOCK_SIZE | num_blocks | 固定 grid=(num_cores,) | 动态 grid=min(num_blocks, num_cores) |
|------|-----------|------------|-----------|----------------|------------------------------|
| 极小 shape | 100 | 2048 | 1 | 47 核空转 | 1 核执行 |
| 小 shape | 4096 | 2048 | 2 | 46 核空转 | 2 核执行 |
| 中等 shape | 65536 | 2048 | 32 | 32 核工作，16 核空转 | 32 核执行 |
| 大 shape | 1048576 | 2048 | 512 | 48 核工作（最优） | 48 核执行（最优） |

### 案例：Softmax 算子

**原始实现：Grid 过大**

```python
M = 112  # 行数
N = 256  # 列数

# Grid = (M,) = 112 个核，远超 48 核
grid = (M,)

@triton.jit
def softmax_kernel_naive(...):
    row_idx = tl.program_id(0)
    # 每个 program 只处理 1 行
    row_data = tl.load(ptr + row_idx * stride + col_offs, mask=mask)
    # ... 计算 softmax
```

**优化后：合理核数**

```python
M = 112
N = 256
ROWS_PER_BLOCK = 4  # 每个 program 处理 4 行

# Grid = (M / ROWS_PER_BLOCK,) = 28 个核，接近最优
grid = (triton.cdiv(M, ROWS_PER_BLOCK),)

@triton.jit
def softmax_kernel_optimized(...):
    pid = tl.program_id(0)
    # 每个 program 处理 ROWS_PER_BLOCK 行
    row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < M

    # 2D 加载：[ROWS_PER_BLOCK, N]
    row_data = tl.load(
        ptr + row_offs[:, None] * stride + col_offs[None, :],
        mask=row_mask[:, None] & col_mask[None, :]
    )
    # ... 计算 softmax（向量化处理多行）
```

**收益对比：**

| 指标 | 原始 | 优化后 |
|------|------|--------|
| Grid 大小 | 112 | 28 |
| 每个 program 处理 | 1 行 = 256 元素 | 4 行 = 1024 元素 |
| 核数利用率 | 过饱和 | **接近最优** |

## 二、UB 大小约束

**NPU UB (Unified Buffer) 通常为 192KB，tile 大小必须满足：**

```
tile_size × dtype_size × buffers ≤ 192KB
```

### 计算示例

```python
# float32 (4 bytes)
# 单 buffer 最大元素数
max_elements = 192 * 1024 / 4 = 49152

# 考虑多 buffer (如 load + store)
# 单 buffer 最大元素数
max_elements_per_buffer = 192 * 1024 / 4 / 2 = 24576

# 常见的 BLOCK_SIZE 选择
BLOCK_SIZE = 8192   # 32KB，安全
BLOCK_SIZE = 16384  # 64KB，安全
BLOCK_SIZE = 32768  # 128KB，接近上限
```

### Tiling 大小选择

| BLOCK_SIZE | 内存占用 (float32) | 安全性 |
|-----------|------------------|--------|
| 4096 | 16 KB | 非常安全 |
| 8192 | 32 KB | 安全 |
| 16384 | 64 KB | 安全 |
| 32768 | 128 KB | 接近上限 |
| 49152 | 192 KB | 可能溢出 |

**原则：在不溢出的前提下，尽量使用大的 BLOCK_SIZE**

## 三、多行并行优化 (1D → 2D Tiling)

### 问题描述

**问题：** 每个 program 只处理 1 行数据，导致 kernel launch 开销大，向量化效率低。

```python
# 问题代码：每个 program 处理 1 行
row_idx = tl.program_id(0)
x = tl.load(ptr + row_idx * stride + cols, mask=mask)
# ... 处理 1 行数据
```

### 优化方案

**方案：** 每个 program 处理 `ROWS_PER_BLOCK` 行，使用 2D 向量化加载。

```python
# 优化代码：每个 program 处理多行
pid = tl.program_id(0)
row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)  # 多行索引
row_mask = row_offs < n_rows

# 2D 加载：[ROWS_PER_BLOCK, BLOCK_SIZE]
x = tl.load(
    ptr + row_offs[:, None] * stride + col_offs[None, :],
    mask=row_mask[:, None] & col_mask[None, :],
)
```

### 性能收益

| 指标 | 原始 | 优化后 |
|-----|------|--------|
| Grid 大小 | `(M,)` | `(M / ROWS_PER_BLOCK,)` |
| Kernel launch 开销 | 高 | 降低 ROWS_PER_BLOCK 倍 |
| 向量化效率 | 1D | 2D 向量化 |
| 典型加速 | 1x | **10-30x** |

### ROWS_PER_BLOCK 选择

| N (列数) | 推荐 ROWS_PER_BLOCK | 原因 |
|---------|-------------------|------|
| < 256 | 16-32 | 列向量化已足够 |
| 256-1024 | 8-16 | 平衡寄存器压力 |
| > 1024 | 4-8 | 避免寄存器溢出 |

### 变换规则

| 单行版本 | 多行版本 |
|---------|---------|
| `row_id = tl.program_id(0)` | `row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)` |
| `grid = (M,)` | `grid = (triton.cdiv(M, ROWS_PER_BLOCK),)` |
| 1D 索引 `ptr + row_id * stride` | 2D 索引 `ptr + row_offs[:, None] * stride` |

### 常见错误：用外层 for 循环串行处理多行

**问题代码**：试图让每个 program 处理多行数据，但错误地使用了外层 for 循环串行处理。

```python
# 错误：每个 program 通过外层循环串行处理多个 expert
# 导致每个 program 重复遍历输入数据多次
pid = tl.program_id(0)
experts_per_core = tl.cdiv(num_experts, NUM_CORES)
start_expert = pid * experts_per_core

for _ in range(experts_per_core):
    curr_expert = ...
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        expert_ids = tl.load(topk_ids_ptrs, ...)
        # ... 重复遍历 topk_numel
    curr_expert += 1
```

**性能劣化原因**：
- 每个 program 将 `topk_ids` 重复加载 `experts_per_core` 次
- 虽然 grid 变小了，但每个 program 的访存量线性增加
- 计算密度没有提升，访存带宽却成倍增加

**正确做法**：使用 2D 向量化 + 广播，让每个 program 在一次遍历中并行处理多行数据。

```python
# 正确：每个 program 同时处理 EXPERT_BLOCK 个 expert
pid = tl.program_id(0)
curr_expert = pid * EXPERT_BLOCK + tl.arange(0, EXPERT_BLOCK)  # 向量索引

acc = tl.zeros((EXPERT_BLOCK, BLOCK_SIZE), dtype=tl.float32)
for x in range(cntx):
    mask = offsets < (topk_numel - x * BLOCK_SIZE)
    expert_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, mask=mask, other=-1)
    # 广播：(EXPERT_BLOCK, 1) == (1, BLOCK_SIZE) -> (EXPERT_BLOCK, BLOCK_SIZE)
    has_curr_expert = (expert_ids[None, :] == curr_expert[:, None]).to(tl.float32)
    acc = acc + has_curr_expert

# 每行一个结果
result = tl.sum(acc, axis=1)
tl.store(expert_num_tokens_ptr + curr_expert, result)
```

**关键差异**：
- 错误方式：外层 for 循环串行处理 `experts_per_core` 个 expert，每个 expert 独立遍历输入
- 正确方式：通过广播将 `EXPERT_BLOCK` 个 expert 的比较同时向量化，**只遍历一次**输入数据
- Grid = `(num_experts // EXPERT_BLOCK,)`，每个 program 的计算密度提升 `EXPERT_BLOCK` 倍

### 模板代码

#### 模式 1: Row-wise Reduction (softmax, row-max, row-sum)

```python
@triton.jit
def row_reduce_kernel(
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

    # 2D 加载
    x = tl.load(
        input_ptr + row_offs[:, None] * stride_in + col_offs[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0
    )

    # 沿列方向 reduce (每行一个值)
    row_result = tl.max(x, axis=1, keep_dims=True)  # 或 tl.sum

    # 存储结果
    tl.store(output_ptr + row_offs[:, None], row_result, mask=row_mask[:, None])
```

#### 模式 2: Element-wise Operation (activation, copy)

```python
@triton.jit
def elementwise_kernel(
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

    # 2D 加载
    x = tl.load(
        input_ptr + row_offs[:, None] * stride_in + col_offs[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
    )

    # 逐元素操作 (自动向量化)
    y = tl.exp(x)  # 或其他 elementwise 操作

    # 2D 存储
    tl.store(
        output_ptr + row_offs[:, None] * stride_out + col_offs[None, :],
        y,
        mask=row_mask[:, None] & col_mask[None, :],
    )
```

#### 模式 3: Reduction + Element-wise (softmax, layer-norm)

```python
@triton.jit
def softmax_kernel(
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

    # 加载数据
    x = tl.load(
        input_ptr + row_offs[:, None] * stride_in + col_offs[None, :],
        mask=mask_2d, other=-float('inf')
    )

    # Phase 1: 计算 max
    row_max = tl.max(x, axis=1, keep_dims=True)

    # Phase 2: 计算 sum(exp(x - max))
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    row_sum = tl.sum(tl.where(mask_2d, exp_x, 0.0), axis=1, keep_dims=True)

    # Phase 3: 计算 output
    output = x_shifted - tl.log(row_sum)

    # 存储
    tl.store(
        output_ptr + row_offs[:, None] * stride_out + col_offs[None, :],
        output, mask=mask_2d
    )
```

## 四、分核策略选择

### 策略对比

| 策略 | 适用场景 | Grid 大小 |
|------|---------|---------|
| **一维分核** | 单维度处理（如逐行） | min(N / BLOCK, num_cores) |
| **多行并行** | 行级 reduce（如 softmax） | min(M / ROWS_PER_BLOCK, num_cores) |

### 选择依据

1. **计算数据量与核数匹配**
   - 总数据量 / 每个 program 处理量 ≈ 物理核数
   - 但 grid 不超过实际需要的 block 数

2. **避免过度细粒度**
   - 每个 program 处理足够数据（至少几千元素）

3. **考虑内存访问模式**
   - 连续访问优于随机访问
   - 2D 加载优于多次 1D 加载

## 五、NPU vs GPU 分核语义差异

### 关键差异

| 特性 | GPU | NPU |
|------|-----|-----|
| Grid 含义 | 逻辑并行实例数 | 直接映射到物理核 |
| 超额订阅 | SM 会自动调度 | AI Core 按顺序执行 |
| 最优 Grid | 可远大于 SM 数 | min(实际需要, AI Core 数) |
| 多核利用率 | SM 动态调度 | 静态绑定 |

**GPU 行为：** Grid 可以远大于 SM 数，GPU 会自动调度，多余的 block 在 SM 上排队等待。

**NPU 行为：** Grid 直接映射到物理核，超额部分串行执行，导致调度开销累积。因此 Grid 应取 `min(实际需要的 block 数, 物理核数)`。

### 案例对比

```python
# 场景：处理 128 个 batch

# GPU 风格（不适用于 NPU）
grid = (128,)  # GPU: 128 block 在 SM 上动态调度，效率高
              # NPU: 128 个核串行执行，调度开销大

# NPU 优化风格（大数据量）
grid = (48,)   # NPU: 48 核并行，每个处理 128/48 ≈ 3 个 batch

# NPU 优化风格（小数据量，只需 2 个 block）
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)  # = 2
grid_size = min(num_blocks, 48)  # = 2
grid = (grid_size,)  # 只启动 2 个核，避免 46 个核空转
```

## 六、Tiling 策略选择

### 案例：GELU 算子

**Easy Kernel（简单但效率低）：**

```python
@triton.jit
def gelu_kernel_easy(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    y = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(y_ptr + offs, y, mask=mask)

# Grid = (n_elements // BLOCK_SIZE,)
# 如果 n_elements = 128 * 1024 * 1024, BLOCK_SIZE = 1024
# Grid = 131072，远超 48 核
```

**Better Kernel（优化的 Tiling 策略）：**

```python
@triton.jit
def gelu_kernel_better(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = tl.minimum(start + BLOCK_SIZE, n_elements)

    # 循环处理这个 block 负责的所有数据
    for i in range(start, end, 256):  # 内部 tiling
        offs = i + tl.arange(0, 256)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask)
        y = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))
        tl.store(y_ptr + offs, y, mask=mask)

# 动态 grid：根据实际需要 block 数和物理核数取较小值
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
grid_size = min(num_blocks, 48)
grid = (grid_size,)
```

**收益对比：**

| 指标 | Easy Kernel | Better Kernel |
|------|-------------|---------------|
| Grid 大小 | 131072 | min(num_blocks, 48) |
| 核数利用率 | 过饱和，调度开销大 | 按需分配，避免空转 |
| 调度开销 | 131072 次 kernel launch | 动态控制 |

### Tiling 策略选择原则

```python
# 策略 1：外部 tiling（适合小数据量）
# Grid = (n_elements // BLOCK_SIZE,)
# 每个 program 处理 BLOCK_SIZE 数据
# 适用：n_elements < 48 * BLOCK_SIZE

# 策略 2：内部 tiling（适合大数据量）
# Grid = min(triton.cdiv(n_elements, BLOCK_SIZE), num_cores)
# 每个 program 内部循环处理
# 适用：所有场景（推荐）
```

## 七、编译优化选项

### 常用编译选项

| 选项 | 作用 | 适用场景 |
|------|------|---------|
| `multibuffer=True` | 启用多缓冲，隐藏内存延迟 | Vector 算子，内存密集型 |
| `multibuffer=False` | 禁用多缓冲 | 计算密集型，寄存器压力大 |
| `unit_flag=True` | 生成独立的计算单元 | 简单算子，无复杂控制流 |
| `unit_flag=False` | 不生成独立单元 | 复杂算子，有分支 |

### 使用方式

```python
@triton.jit
def kernel(...):
    ...

# 在调用时指定
kernel[grid](
    ...,
    multibuffer=True,   # 编译选项
    unit_flag=True,
)
```

### 选择建议

| 算子类型 | multibuffer | unit_flag | 原因 |
|---------|-------------|-----------|------|
| Element-wise (add, mul, gelu) | True | True | 简单、内存密集 |
| Reduction (sum, mean) | True | False | 有归约操作 |
| 复杂控制流 | False | False | 寄存器压力大 |

## 常见错误

### 错误 1：Grid 远超物理核数

```python
# 错误：Grid = (1024,) 远超 48 核
grid = (batch_size * height * width // BLOCK,)

# 正确：Grid = min(实际需要, 48)
num_blocks = triton.cdiv(total_work, BLOCK_SIZE)
grid = (min(num_blocks, num_cores),)
```

### 错误 2：固定启动全部核数（小 shape 场景）

```python
# 错误：无论数据量大小，固定启动 48 核
# n_elements = 100 时，num_blocks = 1，却启动 48 个核，47 个核空转
grid = (num_cores,)

# 正确：根据实际 block 数量动态决定
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
grid_size = min(num_blocks, num_cores)
grid = (grid_size,)
```

### 错误 3：Tile 过小

```python
# 错误：BLOCK_SIZE = 64，UB 利用率低
BLOCK_SIZE = 64  # 只用 256 bytes

# 正确：BLOCK_SIZE = 8192，充分利用 UB
BLOCK_SIZE = 8192  # 用 32KB
```

### 错误 4：忽略 UB 上限

```python
# 错误：BLOCK_SIZE 过大导致 UB 溢出
BLOCK_SIZE = 65536  # 256KB > 192KB UB

# 正确：确保不溢出
BLOCK_SIZE = 32768  # 128KB < 192KB UB
```

### 错误 5：GPU 风格分核

```python
# 错误：GPU 风格，Grid 远大于核数
grid = (n_elements // 1024,)  # 可能是 131072

# 正确：NPU 风格，Grid 按需匹配
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
grid = (min(num_blocks, num_cores),)
```

## 总结

| 优化点 | 原则 | 方法 |
|-------|------|------|
| Grid 大小 | min(实际需要, 物理核数) | `grid_size = min(triton.cdiv(n, BLOCK), num_cores)` |
| Tile 大小 | 接近但不超过 UB | BLOCK_SIZE ≤ 49152 (float32) |
| 多行并行 | 减少 Grid 大小 | 2D 向量化加载 |
| Tiling 策略 | 大数据用内部 tiling | 动态 grid，内部循环 |
| 编译选项 | 内存密集用 multibuffer | multibuffer=True |
