# 矩阵乘法模式（Matrix Multiplication Pattern）

## 概述

矩阵乘法是深度学习中最核心的计算操作，也是 Triton 优化难度最高的 kernel 之一。NPU 上的矩阵乘法需要充分利用 Cube 计算单元（矩阵乘加速器），通过分块策略、存算并行、对角线分核等优化手段达到高性能。

| 关键概念 | 说明 |
|---------|------|
| 分块策略 | BLOCK_M/BLOCK_N/BLOCK_K 三维分块，适配片上存储 |
| `tl.dot` | 矩阵乘法核心操作，映射到 Cube 计算单元 |
| 累加器 | 使用 fp32 精度累加，避免精度损失 |
| `al.compile_hint` | 编译提示，指导 NPU 编译器优化 |
| 对角线分核 | NPU 特有优化，减少 Bank 冲突和提升 L2Cache 命中率 |
| `tl.arange` + 指针算术 | 构建多维指针块 |
| Autotune | 自动搜索最优分块配置 |

## GPU 矩阵乘法 Kernel (不适合NPU)

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c
```

## 分块策略（BLOCK_M/BLOCK_N/BLOCK_K）

矩阵乘法的核心思想是将大矩阵分解为小块，在片上存储中完成计算：

```
C[M, N] = sum_k A[M, K] * B[K, N]

分块后：
C[m_block, n_block] = sum_k_block A[m_block, k_block] * B[k_block, n_block]
```

### 分块大小选择

| 参数 | 含义 | 影响 |
|------|------|------|
| BLOCK_SIZE_M | M 轴分块大小 | 影响 A 矩阵片上占用 |
| BLOCK_SIZE_N | N 轴分块大小 | 影响 B 矩阵片上占用和 L2 命中率 |
| BLOCK_SIZE_K | K 轴分块大小 | 影响循环次数和 Cube 利用率 |

### NPU 推荐分块配置

NPU 芯片亲和 512B 对齐场景，以下分块通用性能较好：

```python
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 256
```

### UB 空间估算

```
UB 占用 ≈ BLOCK_M * BLOCK_K * sizeof(A) + BLOCK_K * BLOCK_N * sizeof(B) + BLOCK_M * BLOCK_N * sizeof(acc)
```

对于 bf16 输入 + fp32 累加：

```
= 128 * 256 * 2 + 256 * 256 * 2 + 128 * 256 * 4
= 65536 + 131072 + 131072
= 327680 bytes ≈ 320KB
```

需要开启 double buffer（96KB * 2 = 192KB，A2/A3）或调整分块大小以适配 UB 限制。910_95 系列 UB 为 256KB，空间更充裕。

## Cube 计算单元利用

### tl.dot 映射

`tl.dot(a, b, acc)` 在 NPU 上映射到 Cube 矩阵乘指令：

```python
accumulator = tl.dot(a, b, accumulator)
```

- `a`：shape `[BLOCK_M, BLOCK_K]`，左矩阵
- `b`：shape `[BLOCK_K, BLOCK_N]`，右矩阵
- `accumulator`：shape `[BLOCK_M, BLOCK_N]`，累加器
- 累加器使用 fp32 精度，NPU 硬件默认 fp32 累加

### compile_hint: dot_pad_only_k

NPU 上 `tl.dot` 要求输入矩阵在 M 和 N 维度对齐。`dot_pad_only_k` 提示编译器仅在 K 维度进行 padding：

```python
a = tl.load(a_ptrs, mask=..., other=0.0)
al.compile_hint(a, "dot_pad_only_k")
b = tl.load(b_ptrs, mask=..., other=0.0)
al.compile_hint(b, "dot_pad_only_k")
accumulator = tl.dot(a, b, accumulator)
```

此提示告诉编译器 M 和 N 维度已经是合法的，只需在 K 维度补零，减少不必要的 padding 开销。

## 优化版矩阵乘法（NPU 特有）

### 对角线分核策略

传统水平分核方式在大 shape 下存在两个问题：
1. 同一时间大量核心访问同一块左矩阵内存，产生 Bank 冲突
2. 右矩阵较大时超出 L2Cache 容量，导致 Cache Miss

对角线分核按 N*N 方块沿对角线方向分配任务，优化 Bank 冲突和 L2Cache 命中率：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel(
        mat_a, mat_b, mat_c,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD

            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                al.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                al.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
    else:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                al.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                al.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
```

### MultiBuffer 优化

`al.multibuffer` 可以为 tensor 创建多个缓冲区副本，实现存算并行：

```python
import triton.language.extra.cann.extension as al

mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
al.multibuffer(mat_a_block, 2)
```

### Host 端调用

```python
from triton.runtime import driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

def triton_matmul(mat_a, mat_b):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    num_cores = get_npu_properties()["num_aicore"]
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    return mat_c
```

## NPU 特有优化总结

| 优化技术 | 说明 | 适用场景 |
|---------|------|---------|
| 对角线分核 | 减少 Bank 冲突，提升 L2Cache 命中率 | M 和 N 方向均超过阈值块数 |
| `compile_hint("dot_pad_only_k")` | 仅在 K 维度 padding | 所有 matmul kernel |
| `al.multibuffer` | 多缓冲区实现存算并行 | Cube+Vector 混合 kernel |
| 512B 对齐分块 | BLOCK_M=128, BLOCK_N=256, BLOCK_K=256 | 通用场景 |
| bf16 输出 | 使用 bf16 替代 fp16 | NPU 上 bf16 性能更优 |
| `num_cores` Grid | 使用实际 AICore 数量作为 grid 大小 | NPU 多核调度 |

## 常见问题（Q&A）

**Q: matmul 结果精度与 PyTorch 不一致？**

A: NPU 上 `tl.dot` 使用 fp32 累加器。bf16/fp16 输入的 matmul 精度差异在 1e-2 范围内属正常。使用 `torch.testing.assert_close` 时设置合理的 `atol` 和 `rtol`。

**Q: 编译报 UB overflow 错误？**

A: 减小 BLOCK_M/BLOCK_N/BLOCK_K，确保所有片上 tensor 总和不超过 UB 限制：A2/A3 系列为 96KB（开启 double buffer）或 192KB（关闭 double buffer），910_95 系列为 128KB（开启 double buffer）或 256KB（关闭 double buffer）。

**Q: 对角线分核比水平分核慢？**

A: 对角线分核在 M 和 N 方向均超过阈值（如 8 块）时才有优势。小矩阵使用传统顺序分核即可。

**Q: 如何实现融合激活函数的 matmul？**

A: 在 K 循环结束后、写回前，对累加器应用激活函数：

```python
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
```

## 相关文档

- [04-layer-norm.md](./04-layer-norm.md) - LayerNorm 模式
- [05-flash-attention.md](./05-flash-attention.md) - Flash Attention 模式
- [07-custom-op-example.md](./07-custom-op-example.md) - 自定义算子示例
- 源码参考：[03-matrix-multiplication.py (upstream)](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials/03-matrix-multiplication.py)
- 源码参考：[13-matrix-multiplication-optimized.py (Ascend)](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/tutorials/13-matrix-multiplication-optimized.py)
