# Ascend Triton 官方测试用例模式

来源：实际测试用例目录
- `/third_party/ascend/unittest/generalization_cases/`
- `/third_party/ascend/unittest/pytest_ut/`

## 1. 矩阵乘法模式

关键代码特征：FP32 累加器、Mask、BLOCK 16 倍数对齐、1D Grid

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    acc_dtype: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**Host 侧**：

```python
kalign = 32 // dtype_bytes
BLOCK_M = min(max(M, 16), 32)
BLOCK_N = min(max(N, 16), 32)
BLOCK_K = min(max(K, kalign), 32)
acc_type = tl.int32 if dtype == "int8" else tl.float32
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
```

## 2. 逐元素操作模式（SwiGLU）

关键代码特征：升精度计算、Mask、1D Grid

```python
@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = e_row * tl.sigmoid(e_row)
    h_row = (f_row * g_row).to(g_row.dtype)

    tl.store(h + offsets, h_row, mask=mask)
```

## 3. FlashAttention 模式（make_block_ptr + advance）

关键代码特征：`make_block_ptr`、`tl.advance`、核间循环分配

```python
@triton.jit
def _attn_fwd(Q, K, V, Out, ...):
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        Q_block_ptr = tl.make_block_ptr(
            base=Q + offset, shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
        
        q = tl.load(Q_block_ptr)
        for start_n in range(lo, hi, BLOCK_N):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            trans_k = tl.trans(k)
            qk = tl.dot(q, trans_k)
            K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
```

**大 HEAD_DIM 分片处理**（≥ 256，使用 Ascend 扩展 API）：

```python
import triton.language.extra.cann.extension as extension

for i in range(4):
    offset = i * (BLOCK_M // 4)
    acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
    pv_i = extension.extract_slice(pv, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
    acc_i = acc_i * alpha_i[:, None] + pv_i
    acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
```

## 4. Rotary Embedding 模式（多维 Mask）

关键代码特征：复合 Mask、多维偏移

```python
@triton.jit
def rotary_embedding_kernel(state, cos, sin, ...,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
    token_range = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    head_range = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    state_x_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + tl.arange(0, BLOCK_D // 2)[None, None, :] * stride_state_d)

    mask = (token_range[:, None, None] < num_tokens) & \
           (head_range[None, :, None] < num_heads)

    state_x = tl.load(state + state_x_offset, mask=mask, other=0.0)
    tl.store(state + state_x_offset, out_x, mask=mask)
```

## 5. 归约操作模式

关键代码特征：dim 为 constexpr、自定义 combine 函数、多维索引

```python
@triton.jit
def _reduce_combine(a, b):
    return a + b

@triton.jit
def tt_reduce_2d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, dim: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        tl.store(out_ptr + yidx, ret)
    else:
        tl.store(out_ptr + xidx, ret)
```

## 6. 精度验证模式

```python
def torch_reference(x, dim):
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.sum(x.to(torch.float32), dim=dim).to(x.dtype)
    return torch.sum(x, dim=dim)

# 容差标准
# FP16:  rtol=1e-3, atol=1e-3
# BF16:  rtol=1e-3, atol=1e-3（转 float32 后比较）
# FP32:  rtol=1e-4, atol=1e-4
# 整数:  exact match
```
