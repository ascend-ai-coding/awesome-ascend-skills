# Flash Attention 模式（Flash Attention Pattern）

## 概述

Flash Attention 是近年来最重要的注意力机制优化算法，通过分块计算和在线 Softmax 更新，将注意力计算的内存复杂度从 O(N^2) 降低到 O(N)。Triton 实现的 Flash Attention 充分利用了 `tl.dot` 和 `tl.make_block_ptr` 等特性，在 NPU 上可以通过 CV 融合、存算并行等优化进一步提升性能。

| 关键概念 | 说明 |
|---------|------|
| 分块计算 | 将 Q/K/V 分块加载到片上，避免 O(N^2) 内存 |
| 在线 Softmax | 增量更新 max 和 sum，避免二次遍历 |
| `tl.make_block_ptr` | 创建块指针，简化多维指针管理 |
| `tl.advance` | 推进块指针偏移 |
| `tl.dot` | 矩阵乘法，映射到 Cube 计算单元 |
| STAGE | 控制因果注意力（causal）的分块范围 |
| `qk_scale * 1.44269504` | 将 scale 转换为 log2 域，配合 `exp2` 使用 |

## Flash Attention Kernel 结构

### 前向内循环

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i
```

### 前向主 Kernel

```python
@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q = tl.load(Q_block_ptr)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        V.dtype.element_ty == tl.float8e5)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        V.dtype.element_ty == tl.float8e5)

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

## 分块计算策略

### 2D Grid 映射

Flash Attention 使用 2D grid：

```
grid = (N_CTX // BLOCK_M, Z * H, 1)
```

- `program_id(0)`：Q 的行块索引（start_m）
- `program_id(1)`：batch * head 的组合索引（off_hz）

### STAGE 机制

| STAGE 值 | 含义 | 内循环范围 |
|----------|------|-----------|
| 1 | 非因果注意力 | lo=0, hi=N_CTX |
| 2 | 因果注意力的对角块 | lo=start_m*BLOCK_M, hi=(start_m+1)*BLOCK_M |
| 3 | 因果注意力（先 off-band 再 on-band） | 两次内循环 |

### 分块大小选择

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| BLOCK_M | 64, 128 | Q 行块大小 |
| BLOCK_N | 32, 64 | K/V 列块大小 |
| HEAD_DIM | 64, 128 | 注意力头维度 |

## Softmax 在线更新

Flash Attention 的核心创新是 Softmax 的在线更新算法，避免将完整的 N*N 注意力矩阵存储到 Global Memory。

### 算法原理

维护两个运行状态：
- `m_i`：当前最大值（每行一个）
- `l_i`：当前归一化因子（每行一个）

每处理一个新的 K/V 块：

```python
qk = tl.dot(q, k)
m_ij = tl.maximum(m_i, tl.max(qk, 1))
alpha = tl.math.exp2(m_i - m_ij)
l_i = l_i * alpha + l_ij
acc = acc * alpha[:, None]
acc = tl.dot(p, v, acc)
m_i = m_ij
```

### 最终归一化

```python
m_i += tl.math.log2(l_i)
acc = acc / l_i[:, None]
```

### log2 域优化

使用 `exp2` / `log2` 替代 `exp` / `log`，在某些硬件上更高效：

```python
qk_scale *= 1.44269504  # 1/log(2)
p = tl.math.exp2(qk)
alpha = tl.math.exp2(m_i - m_ij)
```

## NPU 上的优化

### 1. CV 融合（Cube + Vector）

Flash Attention 的 QK^T 计算使用 Cube（`tl.dot`），Softmax 和最终归一化使用 Vector。NPU 上可以通过 `al.sync_block_all` / `al.sync_block_set` / `al.sync_block_wait` 实现 Cube 和 Vector 的同步：

```python
import triton.language.extra.cann.extension as al

al.sync_block_set(sender="cube", event_id=0)
al.sync_block_wait(sender="cube", event_id=0)
```

### 2. 存算并行

使用 `al.multibuffer` 创建多缓冲区，实现数据搬运和计算的流水线并行：

```python
al.multibuffer(q_block, 2)
```

### 3. make_block_ptr 注意事项

NPU 上 `tl.make_block_ptr` 的 `order` 参数需要正确设置：
- 行优先：`order=(1, 0)`（最后一维连续）
- 列优先：`order=(0, 1)`（第一维连续）

NPU 只允许通过调整 `order` 参数表达转置语义，不能通过调整 `stride` 参数的顺序实现转置。

### 4. fp8_v 支持

NPU 当前不支持 fp8 数据类型（`tl.float8e5`），使用时需确保 V 为 fp16 或 bf16。

### 5. Autotune 配置

```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [1]
    for w in [4, 8]
]
```

NPU 上建议 `num_stages=1`，因为 NPU 的软件流水线策略与 GPU 不同。

## 常见问题（Q&A）

**Q: Flash Attention 结果与 PyTorch 参考实现差异较大？**

A: 检查 `sm_scale` 是否正确设置。在线 Softmax 使用 log2 域计算，确保 `qk_scale *= 1.44269504` 的转换正确。

**Q: causal=True 时结果不对？**

A: 检查 STAGE 参数设置。causal=True 时 STAGE=3，内循环会先处理 off-band 块（STAGE=1），再处理 on-band 块（STAGE=2）。

**Q: make_block_ptr 编译报错？**

A: NPU 上 `make_block_ptr` 与复杂循环和分支语句搭配使用可能出现编译问题。可以改用手动指针算术。

**Q: NPU 上 Flash Attention 性能不佳？**

A: 确保使用了 `compile_hint("dot_pad_only_k")`、`multibuffer` 和 CV 同步等 NPU 特有优化。调整 BLOCK_M/BLOCK_N 以适配 UB 空间。

## 相关文档

- [02-fused-softmax.md](./02-fused-softmax.md) - 融合 Softmax 模式
- [03-matmul.md](./03-matmul.md) - 矩阵乘法模式
- [06-reduction-pattern.md](./06-reduction-pattern.md) - 归约操作模式
- 源码参考：[06-fused-attention.py (upstream)](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials/06-fused-attention.py)
