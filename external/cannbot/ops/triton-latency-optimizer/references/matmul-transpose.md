# Ascend Matmul Transpose 优化

## 概述

本文档覆盖 Ascend NPU 上 Triton 矩阵乘法转置类算子（MatmulBothTrans / MatmulTransA / MatmulTransB / BMM / Linear）的系统性性能优化方法。

转置 matmul 的核心优化哲学是 **保证 `tl.dot` 输入 tile 的内存连续性 + 按 workload 显式双路径分发**。转置输入不能直接对离散跨步 tile 调用 `tl.dot`；生成时禁止混用其他类别的经验（如 reduce 类的 split-K 或 elementwise 的广播技巧）。

## 适用算子

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| MatmulBothTrans | `matmul-both-trans` | A[K,M]、B[N,K]，C = A.T @ B.T | 双路径：kernel 内连续加载后 `tl.trans` + Host 侧 transpose 兜底 |
| MatmulTransA | `matmul-transa` | A[M,N]、B[M,K]，C = A.T @ B | 双路径：A 侧 `[BLOCK_K,BLOCK_M]` 加载后转置 + Host 侧 transpose 兜底 |
| MatmulTransB | `matmul-transb` | A[M,K]、B[N,K]，C = A @ B.T | 双路径：B 侧 `[BLOCK_N,BLOCK_K]` 加载后转置 + Host 侧 transpose 兜底 |
| Matmul / BMM / Linear | `matmul` | 标准 [M,K] @ [K,N] | 标准 tiling，无需转置，直接连续加载 |

## 核心优化点 Checklist（首次生成必查）

### M1 禁止将离散转置 tile 直接送入 `tl.dot`

- **必须**保证 `tl.dot` 的两个输入在加载后是连续的行主序 tile。
- **禁止**对转置输入按非连续维切片后直接进入 `tl.dot`（如把 `[K,M]` 视为 `[M,K].T` 却按 `[M,K]` 行主序切片，得到跨步访存）。
- **Why:** Cube 对离散跨步 tile 利用率极低，典型加速比跌至 0.05x~0.2x。

### M2 Grid 大小必须按实际 cube core 数裁剪

- **必须**通过 `torch_npu.npu.npu_config.get_device_limit(0).get('cube_core_num', 20)` 获取实际 cube core 数。
- **禁止**硬编码核数或超过物理核数 launch。
- **推荐模式**：`grid = (min(num_cores, NUM_BLOCKS),)`，每个 program 循环处理多个输出 tile。
- **Why:** grid 超过物理核数会引入额外调度开销，小 shape 尤其敏感；cube_core_num 在 ascend910b1 上为 20。

### M3 必须按 workload 显式双路径分发

- **必须**在 Host 侧根据 `M * N * K` 阈值选择 kernel 内转置路径或 Host 侧 transpose+contiguous 路径。
- **禁止**用单一 kernel 覆盖所有 shape，或无条件走 Host 侧 transpose。
- **Why:** 小/中 shape 的 Host 侧 `transpose+contiguous` 固定拷贝开销会主导总延迟；极大 shape 下拷贝成本可被计算收益摊平。
- **默认阈值**：`M * N * K >= 10**12` 才启用 Host 侧 transpose 路径，让评测范围内绝大多数 shape 走 kernel 内转置。

### M4 累加器强制使用 `tl.float32`

- **必须**使用 `tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` 作为累加器。
- **禁止**用 fp16/bf16 累加器直接调用 `tl.dot`。
- **Why:** fp16/bf16 累加会引入显著精度误差，verify 比对失败；最终 store 前再 cast 回原 dtype。

### M5 Autotune key 必须使用 `['M', 'N', 'K']`

- **必须**以 `'M'`, `'N'`, `'K'` 作为 `@triton.autotune` 的 key。
- **禁止**用无关维度作为 key，或在小 path 和大 path 混用同一组 config。
- **Why:** 不同 (M,N,K) 组合的最优 tile 差异大，正确 key 才能让编译器为不同 shape 选择合适 config。
- **典型配置**：
  - kernel 内转置路径：`{16,16,16}` / `{32,32,32}` / `{64,64,64}` / `{64,64,32}` / `{128,64,64}` / `{64,128,64}` / `{128,128,64/128}`。
  - Host 侧 transpose 后的标准路径：`{128,128,128}` / `{256,64,64}` / `{64,256,64}`。

## 优化方法

### 通用 Host 侧分派决策树

**优化前（错误：全路径 Host 侧 transpose，小 shape 拷贝开销主导）**

```python
A_t = A.transpose(0, 1).contiguous()
B_t = B.transpose(0, 1).contiguous()
launch_standard_row_major_matmul_kernel(A_t, B_t, C)
```

**优化后（正确：按 workload 阈值双路径分发）**

```python
C = torch.empty((M, N), device=A.device, dtype=A.dtype)

if M * N * K >= 10 ** 12:
    # 极大 shape：拷贝成本可被计算收益摊平
    A_t = A.transpose(0, 1).contiguous()
    B_t = B.transpose(0, 1).contiguous()
    launch_standard_row_major_matmul_kernel(A_t, B_t, C)
else:
    # 小/中/大（评测范围内）shape：避免 Host 侧拷贝开销
    launch_matmul_transpose_kernel(A, B, C)
```

### MatmulBothTrans（C = A.T @ B.T）

**算子类别**: `matmul-both-trans`
**典型特征**: A 为 `[K, M]`，B 为 `[N, K]`，计算 `C = A.T @ B.T`，输出 C 为 `[M, N]`
**性能基准**: 50/50 pass，几何平均加速比 **1.1844x** vs torch

**错误做法（离散跨步切片）**

```python
# A [K, M] 被当作 [M, K].T，却按 [M, K] 行主序切片 -> 跨步访存
a_m = off_m + tl.arange(0, BLOCK_M)
a_k = k + tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + a_m[:, None] * K + a_k[None, :]  # stride=K，不连续
a = tl.load(a_ptrs, mask=..., other=0.0)
```

**正确做法（连续加载后转置）**

```python
@triton.jit
def matmul_both_trans_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    grid_size = tl.num_programs(0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, grid_size):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx - block_m * NUM_BLOCKS_N
        off_m = block_m * BLOCK_M
        off_n = block_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            # A [K, M] row-major: load [BLOCK_K, BLOCK_M] contiguous then trans
            a_dm = off_m + tl.arange(0, BLOCK_M)
            a_dk = k + tl.arange(0, BLOCK_K)
            a_tile_ptrs = a_ptr + a_dk[:, None] * M + a_dm[None, :]
            a_tile_mask = (a_dk[:, None] < K) & (a_dm[None, :] < M)
            a_tile = tl.load(a_tile_ptrs, mask=a_tile_mask, other=0.0)
            a = tl.trans(a_tile)  # [BLOCK_M, BLOCK_K]

            # B [N, K] row-major: load [BLOCK_N, BLOCK_K] contiguous then trans
            b_dn = off_n + tl.arange(0, BLOCK_N)
            b_dk = k + tl.arange(0, BLOCK_K)
            b_tile_ptrs = b_ptr + b_dn[:, None] * K + b_dk[None, :]
            b_tile_mask = (b_dn[:, None] < N) & (b_dk[None, :] < K)
            b_tile = tl.load(b_tile_ptrs, mask=b_tile_mask, other=0.0)
            b = tl.trans(b_tile)  # [BLOCK_K, BLOCK_N]

            accumulator = tl.dot(a, b, acc=accumulator)

        c_dm = off_m + tl.arange(0, BLOCK_M)
        c_dn = off_n + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + c_dm[:, None] * N + c_dn[None, :]
        c_mask = (c_dm[:, None] < M) & (c_dn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
```

### MatmulTransA（C = A.T @ B）

**算子类别**: `matmul-transa`
**典型特征**: A 为 `[M, N]`，B 为 `[M, K]`，计算 `C = A.T @ B`，输出 C 为 `[N, K]`
**性能基准**: 50/50 pass，几何平均加速比 **0.8395x** vs torch

**正确做法（A 侧转置，B 侧直接加载）**

```python
@triton.jit
def matmul_transa_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    grid_size = tl.num_programs(0)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_M)
    NUM_BLOCKS_K = tl.cdiv(K, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_N * NUM_BLOCKS_K

    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, grid_size):
        block_n = block_idx // NUM_BLOCKS_K
        block_k = block_idx - block_n * NUM_BLOCKS_K
        off_n = block_n * BLOCK_M
        off_k = block_k * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for m in range(0, M, BLOCK_K):
            # A [M, N]: load [BLOCK_K, BLOCK_M] contiguous then trans
            a_dm = m + tl.arange(0, BLOCK_K)
            a_dn = off_n + tl.arange(0, BLOCK_M)
            a_tile_ptrs = a_ptr + a_dm[:, None] * N + a_dn[None, :]
            a_tile_mask = (a_dm[:, None] < M) & (a_dn[None, :] < N)
            a_tile = tl.load(a_tile_ptrs, mask=a_tile_mask, other=0.0)
            a = tl.trans(a_tile)  # [BLOCK_M, BLOCK_K]

            # B [M, K]: load [BLOCK_K, BLOCK_N] directly
            b_dm = m + tl.arange(0, BLOCK_K)
            b_dk = off_k + tl.arange(0, BLOCK_N)
            b_ptrs = b_ptr + b_dm[:, None] * K + b_dk[None, :]
            b_mask = (b_dm[:, None] < M) & (b_dk[None, :] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            accumulator = tl.dot(a, b, acc=accumulator)

        c_dn = off_n + tl.arange(0, BLOCK_M)
        c_dk = off_k + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + c_dn[:, None] * K + c_dk[None, :]
        c_mask = (c_dn[:, None] < N) & (c_dk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
```

### MatmulTransB（C = A @ B.T）

**算子类别**: `matmul-transb`
**典型特征**: A 为 `[M, K]`，B 为 `[N, K]`，计算 `C = A @ B.T`，输出 C 为 `[M, N]`
**性能基准**: 50/50 pass，几何平均加速比 **0.8052x** vs torch

**正确做法（B 侧转置，A 侧直接加载）**

```python
@triton.jit
def matmul_transb_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        off_m = block_m * BLOCK_M
        off_n = block_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            # A [M, K]: load [BLOCK_M, BLOCK_K] directly
            a_m = off_m + tl.arange(0, BLOCK_M)
            a_k = k + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + a_m[:, None] * K + a_k[None, :]
            a_mask = (a_m[:, None] < M) & (a_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # B [N, K]: load [BLOCK_N, BLOCK_K] contiguous then trans
            b_n = off_n + tl.arange(0, BLOCK_N)
            b_k = k + tl.arange(0, BLOCK_K)
            b_ptrs = b_ptr + b_n[:, None] * K + b_k[None, :]
            b_mask = (b_n[:, None] < N) & (b_k[None, :] < K)
            b_tile_nk = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b = tl.trans(b_tile_nk)  # [BLOCK_K, BLOCK_N]

            accumulator = tl.dot(a, b, acc=accumulator)

        c_m = off_m + tl.arange(0, BLOCK_M)
        c_n = off_n + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + c_m[:, None] * N + c_n[None, :]
        c_mask = (c_m[:, None] < M) & (c_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
```

## 性能基准

### MatmulBothTrans

| Shape 类型 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 小 shape（如 [4,8] / [6,4]） | 3 | 1.17x ~ 1.50x | kernel 内转置路径调度开销低 |
| 中等 shape（如 [64,128] / [96,64]） | 6 | 0.82x ~ 2.00x | fp16/bf16 通常优于 fp32 |
| 大 shape（如 [512,1024] / [768,512]） | 3 | 0.48x ~ 0.83x | 大 K 维循环次数多，为主要短板 |
| 全量 | 50 | **1.1844x** | 50/50 精度通过 |

### MatmulTransA

| Shape 类型 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 小 shape（如 [1,1] / [2,2]） | 12 | 1.10x ~ 1.77x | 小矩阵 launch 开销尚可接受 |
| 中等 shape（如 [64,128] / [64,256]） | 12 | 0.16x ~ 1.62x | fp16/bf16 部分 shape 出现明显短板 |
| 大 shape（如 [128,512] / [128,256]） | 6 | 0.29x ~ 1.52x | K 维循环长，调度敏感 |
| 全量 | 50 | **0.8395x** | 50/50 精度通过，达 0.8x 目标 |

### MatmulTransB

| Shape 类型 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 小 shape（如 [1,1] / [7,13]） | 12 | 0.88x ~ 1.22x | 小矩阵部分 shape 已接近或略低于 torch |
| 中等 shape（如 [64,128] / [128,256]） | 12 | 0.28x ~ 1.15x | fp16 表现优于 fp32/bf16 |
| 大 shape（如 [512,256] / [1024,256]） | 6 | 0.41x ~ 1.07x | 宽 K 维导致循环次数多 |
| 全量 | 50 | **0.8052x** | 50/50 精度通过，达 0.8x 目标 |

## 常见陷阱

### MatmulBothTrans 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 两侧都按 `[M,K]` / `[K,N]` 语义离散切片 | 跨步访存，Cube 利用率极低 | A `[K,M]` 按 `[BLOCK_K,BLOCK_M]` 加载后 trans；B `[N,K]` 按 `[BLOCK_N,BLOCK_K]` 加载后 trans |
| 全路径 Host 侧 transpose+contiguous | 小 shape 拷贝开销主导 | 保守阈值 `M*N*K >= 10**12` |
| grid 超过 cube core 数 | 调度开销增大 | `grid = (min(num_cores, NUM_BLOCKS),)` |
| fp16/bf16 累加器 | 精度误差导致 verify 失败 | `tl.float32` 累加器 |

### MatmulTransA / MatmulTransB 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 转置侧与非转置侧加载形状混淆 | A `[M,N]` 需要 `[BLOCK_K,BLOCK_M]`+trans；B `[M,K]` 直接 `[BLOCK_K,BLOCK_N]` | 明确语义后选择加载 tile 形状 |
| autotune key 错误 | 用无关维度导致 config 错配 | key 使用 `['M','N','K']` |
| 大 path 与小 path 共用同一组 autotune config | 大 path 需要更大 tile，小 path 需要更小 tile | 分两套 config：小 path 16/32/64/128，大 path 128/256 |
| Host 侧 transpose 阈值过低 | 评测 shape 落入慢路径 | 默认 `10**12`，以目标 shape 分布上的实测几何平均为唯一标准 |

### 通用跨算子陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 使用 `vector_core_num=40` 而非 `cube_core_num=20` | 调度 overhead 增加 | 读取 `cube_core_num` 并裁剪 grid |
| `accumulator += tl.dot(a, b)` 隐式行为 | 可能与显式 `acc=` 有差异 | 统一使用 `tl.dot(a, b, acc=accumulator)` |
| 忽略 mask 导致越界读写 | 非整除 shape 时越界 | 所有 `tl.load` / `tl.store` 加 mask |
| BLOCK 过大超出 UB | fp32 累加下大 tile 溢出 UB | 结合 autotune 和硬件容量选择，避免单 tile 过大 |
