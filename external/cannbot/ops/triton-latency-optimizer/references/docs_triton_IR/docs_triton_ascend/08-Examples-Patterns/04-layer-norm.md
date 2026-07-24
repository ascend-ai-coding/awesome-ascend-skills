# LayerNorm 模式（Layer Normalization Pattern）

## 概述

LayerNorm 是 Transformer 模型中的关键组件，需要对每个特征维度计算均值和方差，然后进行归一化和线性变换。Triton 的融合 LayerNorm kernel 将前向计算中的多次内存访问合并为单次，大幅提升性能。反向传播中使用并行归约策略高效计算梯度。

| 关键概念 | 说明 |
|---------|------|
| 均值计算 | 对每行求和后除以 N |
| 方差计算 | `E[(x - mean)^2]`，使用 `x^2` 的均值减去均值的平方 |
| `rstd` | `1 / sqrt(var + eps)`，预计算倒数避免重复除法 |
| 融合前向 | 将 mean/var/normalize/linear 合并为单次 kernel |
| 并行归约 | 反向传播中分组累加 dw/db，减少原子操作竞争 |
| `tl.atomic_cas` | 自旋锁，保护共享缓冲区的并发写入 |

## 完整 Kernel 代码

### 前向 Kernel

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

### 反向 Kernel

```python
@triton.jit
def _layer_norm_bwd_dx_fused(
    DX, DY, DW, DB, X, W, Mean, Rstd, Lock,
    stride, N,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    tl.store(DX + cols, dx, mask=mask)

    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
```

## 归约操作模式

### 均值计算

LayerNorm 的均值计算采用分块累加模式：

```python
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _mean += a
mean = tl.sum(_mean, axis=0) / N
```

当 `N > BLOCK_SIZE` 时，需要多次加载并累加。`tl.sum` 对块内元素求和得到标量。

### 方差计算技巧

方差使用 `Var(x) = E[x^2] - E[x]^2` 的等价形式：

```python
_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    x = tl.where(cols < N, x - mean, 0.)
    _var += x * x
var = tl.sum(_var, axis=0) / N
```

这里直接计算 `(x - mean)^2` 的均值，而非 `E[x^2] - E[x]^2`，因为后者在数值上可能产生负方差。

### rstd 预计算

```python
rstd = 1 / tl.sqrt(var + eps)
```

预计算 `1/sqrt(var+eps)` 而非每次除以 `sqrt(var+eps)`，避免重复计算。

## 前向调用封装

```python
class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layer_norm_fwd_fused[(M, )](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        _layer_norm_bwd_dx_fused[(M, )](
            dx, dy, _dw, _db, x, w, m, v, locks,
            x_arg.stride(0), N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps)

        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply
```

## NPU 上的优化

### 1. BLOCK_SIZE 选择

NPU 上 UB 空间有限，BLOCK_SIZE 需要适配：

```python
MAX_FUSED_SIZE = 65536 // x.element_size()
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
```

### 2. num_warps 配置

NPU 的 Vector 核心调度与 GPU warp 概念不同，但 `num_warps` 参数仍影响调度：

```python
num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
```

### 3. 反向传播优化

NPU 上 `atomic_add` 不支持多核 add+保存中间结果，需要修改为普通 add 保存。LayerNorm 反向使用自旋锁（`atomic_cas`）保护共享缓冲区，这在 NPU 上是可行的。

### 4. GROUP_SIZE_M 调优

| N 范围 | GROUP_SIZE_M | 说明 |
|--------|-------------|------|
| <= 1024 | 256 | 更多并行组 |
| <= 4096 | 128 | 平衡并行和缓存 |
| <= 8192 | 96 | 减少锁竞争 |
| > 8192 | 64 | 最小化锁开销 |

## 常见问题（Q&A）

**Q: LayerNorm 前向结果与 PyTorch 不一致？**

A: 检查 eps 值是否一致。NPU 上 fp16 计算可能有微小精度差异，建议 kernel 内部使用 fp32 精度。

**Q: BLOCK_SIZE 报错 "feature dim >= 64KB"？**

A: N 的值过大，超出单次融合计算的限制。需要将行分块处理，每块加载部分列。

**Q: 反向传播中 atomic_cas 死锁？**

A: NPU 上 `atomic_cas` 在循环中使用有限制。确保锁的获取和释放在同一执行路径上，避免条件分支导致锁未释放。

**Q: dw/db 梯度精度不够？**

A: 检查 `_layer_norm_bwd_dwdb` 中的累加精度。确保使用 fp32 累加后再转回原始精度。

## 相关文档

- [02-fused-softmax.md](./02-fused-softmax.md) - 融合 Softmax 模式
- [06-reduction-pattern.md](./06-reduction-pattern.md) - 归约操作模式
- 源码参考：[05-layer-norm.py (upstream)](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials/05-layer-norm.py)
