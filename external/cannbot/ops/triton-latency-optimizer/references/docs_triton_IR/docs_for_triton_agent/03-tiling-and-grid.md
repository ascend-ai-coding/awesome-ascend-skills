# Tiling 与 Grid 分核策略

## 触发条件

当 Agent 需要确定 Tiling 大小或 Grid 分核数时，参考本文档：

- 编写新 Triton kernel，需要设定 `BLOCK_SIZE` / `BLOCK_M` / `BLOCK_N` 等 Tiling 参数
- 现有 kernel 使用固定 Tiling，需改为 Autotune 动态选择
- Grid 分核数超过物理核数，需要收缩优化
- 每个程序只处理一行数据，SIMD 利用率低，需改为多行处理
- 遇到 UB overflow 编译错误，需重新计算 Tiling

---

## 核心知识

### 1. UB 容量计算

#### 各型号 UB 容量

| 芯片型号 | UB 硬件大小 | 编译器可用大小 | 开启 multiBuffer 后可用 | 源码值 (bits) |
|----------|------------|--------------|----------------------|--------------|
| Ascend910B (A2) | 192 KB | 192 KB | 96 KB | 1572864 |
| **Ascend910_95** | **256 KB** | **248 KB** | **124 KB** | **2031616** |
| Ascend950 | 256 KB | 248 KB | 124 KB | 2031616 |

> **910_95 关键**：UB 硬件为 256KB，但编译器预留 8KB，实际可用 **248KB**（= 256KB - 8KB）。源码中 `UbSize=2031616 bits`。

源码参考：[utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py#L41-L50)

#### UB 可用空间公式

```
UB可用 = UB总容量 - 预留空间
910B:  192 KB = 196608 bytes
910_95: 248 KB = 253952 bytes

开启 multiBuffer 后:
UB可用 = UB可用 / 2
910B:  96 KB = 98304 bytes
910_95: 124 KB = 126976 bytes
```

#### UB 空间预算公式

```
所需UB空间 = (输入张量数 + 输出张量数 + 中间张量数) x BLOCK_SIZE x dtype_bytes x multiBuffer副本数

约束: 所需UB空间 <= UB可用空间
```

#### 32B 对齐处理

Vector 算子场景下，Tensor 尾轴大小须被 32Bytes 整除：

```python
def align_to_32(size_bytes):
    return ((size_bytes + 31) // 32) * 32

# 各数据类型的最小对齐元素数
# FP16 (2B): 尾轴元素数须为 16 的倍数
# BF16 (2B): 尾轴元素数须为 16 的倍数
# FP32 (4B): 尾轴元素数须为 8 的倍数
# INT8 (1B):  尾轴元素数须为 32 的倍数
# INT32 (4B): 尾轴元素数须为 8 的倍数
```

单值缓冲区（如归约结果 mean/var/max/sum）逻辑上只需 4B（FP32），但硬件要求 32B 对齐，实际分配 32B。

CV 融合算子（含 `tl.dot`）要求 512B 对齐：
- FP16: 尾轴元素数须为 256 的倍数
- FP32: 尾轴元素数须为 128 的倍数

---

### 2. 核间切分策略

核间切分决定多个 AI Core 如何并行处理数据。核心原则：**独立性、负载均衡、数据局部性**。

#### 获取物理核数

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
aicore_num = properties["num_aicore"]          # Cube 核数
vectorcore_num = properties["num_vectorcore"]   # Vector 核数

# 910_95: vectorcore_num = aicore_num * 2
# 例: Ascend910_9581 -> aicore_num=32, vectorcore_num=64
```

#### 算子类型与核数选择

| 算子类型 | 判断依据 | 使用核数 |
|---------|---------|---------|
| 纯 Vector 算子 | kernel 中无 `tl.dot` / `tl.matmul` | `num_vectorcore` |
| CV 融合算子 | kernel 中含 `tl.dot` / `tl.matmul` | `num_aicore` |

#### 切分模式

**模式 1：按 Batch 维度切分**

适用场景：输入 `[B, D]`，算子在 D 维度上独立计算（如 LayerNorm、RMSNorm、Softmax）

```python
# 每个 Core 处理的 batch 数
batch_per_core = ceil(B / num_cores)

# 当前 Core 的 batch 范围
core_id = tl.program_id(0)
batch_start = core_id * batch_per_core
batch_end = min((core_id + 1) * batch_per_core, B)
```

**模式 2：按行切分（矩阵运算）**

适用场景：矩阵乘法 `C[M,N] = A[M,K] x B[K,N]`，按 M 维度切分

```python
rows_per_core = ceil(M / num_cores)
row_start = core_id * rows_per_core
row_end = min((core_id + 1) * rows_per_core, M)
```

**模式 3：按特征维度切分**

适用场景：输入 `[B, D]`，需要在 D 维度上并行，B 维度有依赖

```python
features_per_core = ceil(D / num_cores)
feature_start = core_id * features_per_core
feature_end = min((core_id + 1) * features_per_core, D)
```

#### 负载均衡处理

当数据量不能被核数整除时：

```python
base_batch = B // num_cores
remainder = B % num_cores

if core_id < remainder:
    batch_start = core_id * (base_batch + 1)
    batch_end = batch_start + base_batch + 1
else:
    batch_start = remainder * (base_batch + 1) + (core_id - remainder) * base_batch
    batch_end = batch_start + base_batch
```

---

### 3. 核内切分策略

核内切分决定单个 Core 如何利用 UB 空间分块处理数据。

#### 一级分块（BLOCK_SIZE）

决定每个核处理的总数据量，影响 Grid 分核数：

```python
NUM_BLOCKS = ceil(N / BLOCK_SIZE)
coreDim = NUM_BLOCKS

# 约束:
# 1. coreDim <= 65535 (UINT16_MAX)
# 2. 推荐对齐到物理核数
# 3. BLOCK_SIZE 应为 2 的幂次
```

#### 二级分块（BLOCK_SIZE_SUB）

当一级分块仍超出 UB 容量时，在 for 循环内进一步切分：

```python
num_sub_blocks = ceil(BLOCK_SIZE / BLOCK_SIZE_SUB)

# 约束:
# 1. 单次迭代 UB 使用量 <= UB可用 / multiBuffer副本数
# 2. BLOCK_SIZE_SUB 应满足 32B 对齐
# 3. BLOCK_SIZE_SUB 应为 2 的幂次
```

#### 存算并行

910_95 平台默认 `multiBuffer=False`（910B 默认为 True），需显式设置 `multibuffer=True` 才能支持数据搬运（MTE2）与计算（Vector/Cube）重叠执行。前提条件：

1. **for 循环 Tiling**：算子内必须有多次迭代
2. **无数据依赖**：当前迭代不依赖下一迭代搬入的数据
3. **UB 空间充足**：multiBuffer 需额外 UB 空间（可用空间减半）

---

### 4. Grid 收缩到物理核数

当 Grid 分核数超过物理核数时，核启动开销急剧增加。优化思路：**将 Grid 收缩到物理核数，让每个核内部循环处理多个任务**。

#### 判断条件

| 分核类型 | 条件判断 |
|---------|---------|
| 一维分核 | `grid > num_cores` |
| 二维分核 | `grid1 x grid2 > num_cores` |
| 三维分核 | `grid1 x grid2 x grid3 > num_cores` |

#### 一维分核场景

优化前：

```python
@triton.jit
def kernel(X, Y, BLOCK_SIZE: tl.constexpr, ...):
    row_id = tl.program_id(0)
    off = row_id * BLOCK_SIZE
    x = tl.load(X + off)
    # ... calc ...

grid = triton.cdiv(X.shape[0], BLOCK_SIZE)
kernel[(grid,)](X, Y, ...)
```

优化后：

```python
@triton.jit
def kernel(X, Y, BLOCK_SIZE: tl.constexpr, NUM_TASKS: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    for row_id in range(pid, NUM_TASKS, NUM_CORES):
        off = row_id * BLOCK_SIZE
        x = tl.load(X + off)
        # ... calc ...
        tl.store(Y + off, result)

def launch_kernel(x, y, block_size, num_cores):
    num_tasks = triton.cdiv(x.shape[0], block_size)
    grid = (num_cores,)
    kernel[grid](x, y, block_size, num_tasks, num_cores)
```

#### 二维分核场景

优化前：

```python
@triton.jit
def kernel(X, Y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ...):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    # ... calc ...

grid1 = triton.cdiv(M, BLOCK_M)
grid2 = triton.cdiv(N, BLOCK_N)
kernel[(grid1, grid2)](X, Y, ...)
```

优化后：

```python
@triton.jit
def kernel(X, Y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           NUM_TASKS: tl.constexpr, NUM_COL_TASKS: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    for task_id in range(pid, NUM_TASKS, NUM_CORES):
        row_id = task_id // NUM_COL_TASKS
        col_id = task_id % NUM_COL_TASKS
        # ... calc ...

def launch_kernel(x, y, block_m, block_n, num_cores):
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    num_tasks = grid_m * grid_n
    num_col_tasks = grid_n
    grid = (num_cores,)
    kernel[grid](x, y, block_m, block_n, num_tasks, num_col_tasks, num_cores)
```

#### 三维分核场景

优化前：

```python
@triton.jit
def kernel(X, Y, ...):
    batch_id = tl.program_id(0)
    row_id = tl.program_id(1)
    col_id = tl.program_id(2)
    # ... calc ...

grid0 = triton.cdiv(BATCH, BLOCK_B)
grid1 = triton.cdiv(M, BLOCK_M)
grid2 = triton.cdiv(N, BLOCK_N)
kernel[(grid0, grid1, grid2)](X, Y, ...)
```

优化后：

```python
@triton.jit
def kernel(X, Y, BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           NUM_TASKS: tl.constexpr, NUM_ROW_TASKS: tl.constexpr,
           NUM_COL_TASKS: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    for task_id in range(pid, NUM_TASKS, NUM_CORES):
        batch_id = task_id // (NUM_ROW_TASKS * NUM_COL_TASKS)
        rem = task_id % (NUM_ROW_TASKS * NUM_COL_TASKS)
        row_id = rem // NUM_COL_TASKS
        col_id = rem % NUM_COL_TASKS
        # ... calc ...

def launch_kernel(x, y, block_b, block_m, block_n, num_cores):
    grid_b = triton.cdiv(BATCH, block_b)
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    num_tasks = grid_b * grid_m * grid_n
    num_row_tasks = grid_m * grid_n
    num_col_tasks = grid_n
    grid = (num_cores,)
    kernel[grid](x, y, block_b, block_m, block_n,
                 num_tasks, num_row_tasks, num_col_tasks, num_cores)
```

#### Grid 收缩注意事项

- `NUM_TASKS` 和 `NUM_CORES` 必须传递为 `tl.constexpr`，以启用编译器优化
- 不适用于任务间有依赖、必须按 program_id 顺序执行的场景
- 不适用于使用了 `tl.barrier()` 或 `tl.NamedBarrier` 进行同步的分核策略
- 环境变量 `TRITON_ALL_BLOCKS_PARALLEL=1` 可启用编译器自动 Grid 收缩（仅当逻辑核间可并行时）

---

### 5. 单行到多行任务优化

当每个 program 只处理一行数据时，单次 load/store 数据量过小，无法充分利用 NPU Vector 单元的 SIMD 能力。引入 `BLOCK_ROWS` 参数，让每个 program 同时处理多行。

#### 优化前（每 program 1 行）

```python
@triton.jit
def kernel(X_ptr, Y_ptr, X_stride, Y_stride, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    X_offs = row_id * X_stride + tl.arange(0, BLOCK_SIZE)
    X_mask = tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE
    X = tl.load(X_ptr + X_offs, mask=X_mask)
    # ... calc ...
    Y_offs = row_id * Y_stride + tl.arange(0, BLOCK_SIZE)
    tl.store(Y_ptr + Y_offs, X, mask=X_mask)

def kernel_launch(X):
    M, N = X.view(-1, X.shape[-1]).shape
    Y = torch.empty_like(X)
    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)
    kernel[grid](X, Y, X.stride(0), Y.stride(0), BLOCK_SIZE)
```

#### 优化后（每 program BLOCK_ROWS 行 + Autotune）

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": BM})
        for BM in [1, 2, 4, 8, 16, 32]
    ],
    key=["X_stride", "total_rows"]
)
@triton.jit
def kernel(X_ptr, Y_ptr, X_stride, Y_stride, total_rows,
           BLOCK_ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offs < total_rows
    col_offs = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offs < X_stride
    X = tl.load(X_ptr + (row_offs[:, None] * X_stride + col_offs[None, :]),
                mask=row_mask[:, None] & col_mask[None, :])
    # ... calc ...
    tl.store(Y_ptr + (row_offs[:, None] * Y_stride + col_offs[None, :]),
             result, mask=row_mask[:, None] & col_mask[None, :])

def kernel_launch(X):
    M, N = X.view(-1, X.shape[-1]).shape
    Y = torch.empty_like(X)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROWS"]),)
    BLOCK_SIZE = triton.next_power_of_2(N)
    kernel[grid](X, Y, X.stride(0), Y.stride(0), total_rows=M, BLOCK_SIZE=BLOCK_SIZE)
```

#### 关键点

- **2D mask 构造**：`row_mask[:, None] & col_mask[None, :]`
- **offset 广播**：`row_offs[:, None] * stride + col_offs[None, :]`
- **BLOCK_ROWS 候选值**：`[1, 2, 4, 8, 16, 32]`，2 的幂次
- **key 参数**：使用 `stride` 和 `total_rows` 作为 key
- **grid 计算**：`grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROWS"]),)`
- 与 Autotune Tiling 优化互补：BLOCK_ROWS 调整行方向，BLOCK_SIZE 调整列方向

---

### 6. Autotune 配置

#### NPU 上的 Autotune 规则

**关键：NPU 上 Autotune 的 Config 只含 Tiling 参数，不含 `num_stages` 和 `num_warps`。**

- `num_warps`：NPU 无 Warp 概念，此参数无效
- `num_stages`：NPU 无软件流水线概念，此参数无效

#### 单参数 Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM})
        for BM in [128, 256, 512, 1024, 2048]
    ],
    key=["n_numel"]
)
@triton.jit
def kernel(..., n_numel, BLOCK_SIZE: tl.constexpr):
    ...
```

#### 多参数 Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M_SIZE": BM, "BLOCK_N_SIZE": BN})
        for BM in [128, 256, 512, 1024, 2048]
        for BN in [128, 256, 512, 1024, 2048]
    ],
    key=["M", "N"]
)
@triton.jit
def kernel(..., M, N, BLOCK_M_SIZE: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    ...
```

#### Autotune 配置要点

| 要点 | 说明 |
|------|------|
| 装饰器位置 | 必须放在 `@triton.jit` 上方 |
| key 参数 | 当 key 中的值变化时重新选择最优配置，常用 `n_numel`、`M`、`N`、`seq_len` |
| 候选值范围 | 2 的幂次，从小到大覆盖 |
| 小值选择 | 覆盖小 shape 场景，把物理核数用满 |
| 大值选择 | 覆盖大 shape 场景，把 UB 空间用满 |
| 边界处理 | 如果最优配置总在候选列表边界，需扩展范围 |
| n_numel 确定 | 通常是 kernel 处理的主要维度大小，如 total_elements、total_tokens |
| 验证 | 设置 `TRITON_PRINT_AUTOTUNING=1` 打印最优参数信息 |

---

## 910_95 特别注意

### UB 容量差异

| 项目 | 910B | 910_95 |
|------|------|--------|
| UB 硬件大小 | 192 KB | 256 KB |
| **编译器可用** | **192 KB** | **248 KB（预留 8KB）** |
| multiBuffer 后可用 | 96 KB | 124 KB |
| 源码值 (bits) | 1572864 | 2031616 |

> **注意**：源码 [utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py#L48-L49) 中 910_95 的 `ub_size_in_kbytes = 256`，这是硬件规格值。编译器实际可用为 248KB（预留 8KB 给编译器内部使用，源码 `UbSize = 2031616; // bits = 248KB, reserve 8KB for compiler`）。

### 核数差异

910_95 系列 Vector Core 数 = AI Core 数 x 2：

| 型号 | AI Core | Vector Core |
|------|---------|-------------|
| Ascend910_9581 | 32 | 64 |
| Ascend910_9599 | 36 | 72 |

### Tiling 参数上限调整

910_95 的 UB 比 910B 大约 29%（248KB vs 192KB），Tiling 候选值上限可相应提高：

```python
# 910B 的最大 BLOCK_SIZE 估算
# FP16, 3个张量(input+output+intermediate), multiBuffer=2
# max_BLOCK = 192*1024 / (3 * 2 * 2) = 16384

# 910_95 的最大 BLOCK_SIZE 估算
# max_BLOCK = 248*1024 / (3 * 2 * 2) = 21233 -> 实际取 16384 或 20480 (2的幂次近似)
```

---

## UB 空间预算计算示例

### RMSNorm（FP16 输入，FP32 计算）

输入形状 `[B, D]`，每个 batch 独立计算。

| 缓冲区 | 数据类型 | 大小 (D=4096) |
|--------|---------|--------------|
| 输入 x | FP16 | D x 2 = 8192 B |
| 升精度 x_fp32 | FP32 | D x 4 = 16384 B |
| 平方 x_sq | FP32 | D x 4 = 16384 B |
| 均值 mean | FP32 | 32 B（对齐） |
| Gamma | FP32 | D x 4 = 16384 B |
| RMS 值 | FP32 | 32 B（对齐） |
| 输出 y | FP16 | D x 2 = 8192 B |
| **合计** | | **65632 B ≈ 64 KB** |

```
910B:  192KB / 64KB ≈ 3 个 batch/iteration
910_95: 248KB / 64KB ≈ 3 个 batch/iteration（D=4096 时 UB 较紧张）

若开启 multiBuffer:
910B:  96KB / 64KB ≈ 1 个 batch/iteration
910_95: 124KB / 64KB ≈ 1 个 batch/iteration
```

### LayerNorm（FP16 输入，FP32 计算）

| 缓冲区 | 数据类型 | 大小 (D=4096) |
|--------|---------|--------------|
| 输入 x | FP16 | D x 2 = 8192 B |
| 升精度 x_fp32 | FP32 | D x 4 = 16384 B |
| 均值 mean | FP32 | 32 B |
| 方差 var | FP32 | 32 B |
| Gamma | FP32 | D x 4 = 16384 B |
| Beta | FP32 | D x 4 = 16384 B |
| 输出 y | FP16 | D x 2 = 8192 B |
| **合计** | | **65600 B ≈ 64 KB** |

### Softmax（FP16 输入，FP32 计算）

| 缓冲区 | 数据类型 | 大小 (D=4096) |
|--------|---------|--------------|
| 输入 x | FP16 | D x 2 = 8192 B |
| 升精度 x_fp32 | FP32 | D x 4 = 16384 B |
| Max 值 | FP32 | 32 B |
| Sum 值 | FP32 | 32 B |
| Exp 缓冲区 | FP32 | D x 4 = 16384 B |
| 输出 y | FP16 | D x 2 = 8192 B |
| **合计** | | **49376 B ≈ 48 KB** |

```
910B:  192KB / 48KB ≈ 4 个 batch/iteration
910_95: 248KB / 48KB ≈ 5 个 batch/iteration

若开启 multiBuffer:
910B:  96KB / 48KB ≈ 2 个 batch/iteration
910_95: 124KB / 48KB ≈ 2 个 batch/iteration
```

### MatMul（FP16）

`C[M,N] = A[M,K] x B[K,N]`，核内按 K 维度分块累加：

| 缓冲区 | 大小 |
|--------|------|
| A 块 | BLOCK_M x BLOCK_K x 2 B |
| B 块 | BLOCK_K x BLOCK_N x 2 B |
| C 累加 | BLOCK_M x BLOCK_N x 4 B (FP32) |
| **合计** | 2*BK*(BM+BN) + 4*BM*BN |

```
示例: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, FP16
合计 = 2*64*(128+128) + 4*128*128 = 32768 + 65536 = 98304 B ≈ 96 KB

910B:  192KB / 96KB ≈ 2x（刚好可行，但开启 multiBuffer 后不足）
910_95: 248KB / 96KB ≈ 2x（开启 multiBuffer 后 124KB > 96KB，可行）
```

---

## 典型 Tiling 候选值范围

### 1D 场景（单维度 BLOCK_SIZE）

```python
# Vector 算子（无 tl.dot）
configs = [
    triton.Config({"BLOCK_SIZE": BS})
    for BS in [128, 256, 512, 1024, 2048, 4096]
]

# 小 shape 场景（D < 4096）
configs = [
    triton.Config({"BLOCK_SIZE": BS})
    for BS in [64, 128, 256, 512, 1024]
]
```

### 2D 场景（MatMul BLOCK_M x BLOCK_N）

```python
# 矩阵乘法
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK})
    for BM in [64, 128, 256]
    for BN in [64, 128, 256]
    for BK in [32, 64]
]
```

### 行方向（BLOCK_ROWS）

```python
# 单行 -> 多行优化
configs = [
    triton.Config({"BLOCK_ROWS": BR})
    for BR in [1, 2, 4, 8, 16, 32]
]
```

### 候选值选择原则

| 原则 | 说明 |
|------|------|
| 2 的幂次 | 候选值必须是 2 的幂次，便于硬件对齐 |
| 从小到大覆盖 | 小值匹配物理核数，大值匹配 UB 容量 |
| UB 约束上限 | 最大候选值对应的 UB 使用量不超过可用空间 |
| 对齐约束 | FP16 场景 BLOCK_SIZE 须为 16 的倍数（32B/2B=16） |
| coreDim 约束 | `ceil(N / BLOCK_SIZE) <= 65535` |
| 边界扩展 | 如果 autotune 总选到边界值，需扩展候选范围 |

---

## 代码模式汇总

### 模式 1：固定 Tiling -> Autotune

```python
# 优化前
@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2, mask=mask)

# 优化后
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": BS}) for BS in [128, 256, 512, 1024, 2048]],
    key=["n"]
)
@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2, mask=mask)
```

### 模式 2：Grid 收缩 + 核内循环

```python
# 优化前: grid 可能远超物理核数
grid = (triton.cdiv(N, BLOCK_SIZE),)
kernel[grid](...)

# 优化后: grid 固定为物理核数，核内循环
@triton.jit
def kernel(..., NUM_TASKS: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    for task_id in range(pid, NUM_TASKS, NUM_CORES):
        # 处理 task_id 对应的数据块
        ...

num_tasks = triton.cdiv(N, BLOCK_SIZE)
grid = (num_cores,)
kernel[grid](..., num_tasks, num_cores)
```

### 模式 3：单行 -> 多行 + Autotune

```python
# 优化前: 每 program 1 行
row_id = tl.program_id(0)
X = tl.load(X_ptr + row_id * stride + tl.arange(0, BLOCK_SIZE))

# 优化后: 每 program BLOCK_ROWS 行
@triton.autotune(
    configs=[triton.Config({"BLOCK_ROWS": BR}) for BR in [1, 2, 4, 8, 16]],
    key=["total_rows"]
)
@triton.jit
def kernel(..., total_rows, BLOCK_ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offs < total_rows
    col_offs = tl.arange(0, BLOCK_SIZE)
    X = tl.load(X_ptr + row_offs[:, None] * stride + col_offs[None, :],
                mask=row_mask[:, None] & (col_offs[None, :] < BLOCK_SIZE))
```

### 模式 4：二级分块避免 UB 溢出

```python
# 当 BLOCK_SIZE 对应的 UB 使用量超限时，引入 BLOCK_SIZE_SUB
@triton.jit
def kernel(inp, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_idx in range(num_sub_blocks):
        offsets = base_offset + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        data = tl.load(inp + offsets, mask=mask)
        result = compute(data)
        tl.store(out + offsets, result, mask=mask)
```

---

## Tiling 策略设计检查清单

### 核间切分

- [ ] 切分维度选择合理（考虑数据独立性）
- [ ] 负载均衡（每个 Core 处理的数据量相近）
- [ ] 无跨核通信（每个 Core 独立完成任务）
- [ ] 边界处理正确（最后一个 Core 的数据范围）
- [ ] Grid 分核数不超过物理核数（或已做 Grid 收缩）

### 核内切分

- [ ] 所有缓冲区都已列出
- [ ] 缓冲区总大小 < UB 可用空间（910_95: 248KB, 910B: 192KB）
- [ ] 单值缓冲区分配 32B 空间
- [ ] 精度转换策略明确（是否需要升/降精度）
- [ ] 开启 multiBuffer 后 UB 仍够用（可用空间减半）
- [ ] for 循环存在以支持存算并行

### 对齐检查

- [ ] UB 缓冲区地址 32 字节对齐
- [ ] 单值缓冲区分配 32B 空间
- [ ] BLOCK_SIZE 满足数据类型对齐（FP16: 16 的倍数，FP32: 8 的倍数）
- [ ] CV 融合算子满足 512B 对齐

### Autotune 检查

- [ ] Config 中不含 `num_stages` 和 `num_warps`
- [ ] 候选值为 2 的幂次
- [ ] key 参数选择代表数据规模的变量
- [ ] 装饰器位于 `@triton.jit` 上方

---

## 常见错误与解决方案

| 错误 | 症状 | 原因 | 解决方案 |
|------|------|------|---------|
| UB 溢出 | `ub overflow, requires xxx bits while yyy bits available` | 缓冲区总大小超 UB 容量 | 减小 BLOCK_SIZE 或引入 BLOCK_SIZE_SUB 二级分块 |
| 对齐错误 | 硬件错误或性能下降 | 缓冲区地址未对齐 / 单值缓冲区分配不足 | 使用 `align_to_32()` 计算地址，单值缓冲区统一分配 32B |
| 负载不均衡 | 部分 Core 提前完成 | 数据量不能被核数整除 | 使用动态分配策略 |
| Grid 过大 | 性能下降，核启动开销高 | Grid 远超物理核数 | Grid 收缩到物理核数 + 核内循环 |
| SIMD 利用率低 | 单行处理效率低 | 每 program 只处理 1 行 | 引入 BLOCK_ROWS 多行处理 |
| Autotune 无效 | 不同规模性能差异大 | Config 含 num_stages/num_warps | 移除 GPU 专属参数，只保留 Tiling 参数 |
| coreDim 超限 | `coreDim can't be greater than UINT16_MAX` | `ceil(N/BLOCK_SIZE) > 65535` | 增大 BLOCK_SIZE 或使用 Grid 收缩模式 |

---

## 相关文档链接

- [00-hardware-quick-ref.md](../docs_for_triton_agent/00-hardware-quick-ref.md) -- 910_95 硬件速查手册（UB 容量、核数、对齐要求）
- [01-migration-overview.md](../docs_for_triton_agent/01-migration-overview.md) -- GPU 到 NPU 迁移概览（num_warps/num_stages 移除说明）
- [02-tiling-strategy.md](../docs_triton_ascend/05-Performance-Optimization/02-tiling-strategy.md) -- 分块策略详解（存算并行、multiBuffer、对齐要求）
- [utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py) -- UB 容量和硬件参数源码
- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) -- Triton 算子开发指南
