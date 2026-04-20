# Triton 算子模板参考

> **获取核数的函数定义**：见 [hardware-architecture.md](hardware-architecture.md#核心类型选择)

---

## 模板 1：归约类算子（Reduction）

**代表**：RMSNorm、L2Norm、Softmax、CrossEntropy

**特征**：
- 需要跨元素求和、求最大值等归约操作
- 归约操作必须在 FP32 精度下进行
- grid = 物理核数，核内循环处理多行

```python
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]

@triton.jit
def l2norm_fwd_kernel(
    X,
    Y,
    eps,
    M,
    N: tl.constexpr,
    rows_per_prog: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    prog_id = tl.program_id(0)

    row_start = prog_id * rows_per_prog
    col_off = tl.arange(0, NBLOCK)
    col_mask = col_off < N

    for row_blk_id in range(0, rows_per_prog, MBLOCK):
        row_blk_id = tl.multiple_of(row_blk_id, MBLOCK)
        row_idx = row_start + row_blk_id + tl.arange(0, MBLOCK)
        row_off = row_idx * N
        row_mask = row_idx < M

        tot_off = row_off[:, None] + col_off[None, :]
        tot_mask = row_mask[:, None] & col_mask[None, :]

        xs = tl.load(X + tot_off, mask=tot_mask).to(tl.float32)

        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + tot_off, xs * rsqrt, mask=tot_mask)

def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    y = torch.empty_like(x)

    T, D = x.shape[0], x.shape[-1]
    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))

    ub_size = 196608
    core_num = get_npu_vectorcore_num()

    if T <= core_num:
        num_progs = T
        rows_per_prog = 1
        MBLOCK = 1
    else:
        num_progs = core_num
        rows_per_prog = triton.cdiv(T, core_num)
        MBLOCK = min(ub_size // (BD * 4 * 4), rows_per_prog)

    grid = (num_progs,)
    l2norm_fwd_kernel[grid](
        X=x, Y=y, eps=eps, M=T, N=D,
        rows_per_prog=rows_per_prog, MBLOCK=MBLOCK, NBLOCK=BD,
    )

    return y.view(x_shape_og)
```

---

## 模板 2：矩阵乘法类算子（GEMM）

**代表**：MatMul、Linear、ScaledMM

**特征**：
- 使用 `tl.dot()` 进行矩阵乘法
- 使用 `get_npu_aicore_num()` 获取核数
- 需要 L1 Buffer（1MB）用于矩阵分块
- 使用 `tl.compile_hint(x, "dot_pad_only_k")` 优化

```python
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    mat_a, mat_b, mat_c,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + \
                           (k_start + tl.arange(0, BLOCK_K))[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \
                         ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")

            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + \
                           (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & \
                         ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")

            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)

        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + \
                       (n_start + tl.arange(0, BLOCK_N))[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & \
                     ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)

def matmul(mat_a: torch.Tensor, mat_b: torch.Tensor):
    M, K = mat_a.shape
    N = mat_b.shape[1]
    mat_c = torch.empty(M, N, dtype=mat_a.dtype, device=mat_a.device)

    num_cores = get_npu_aicore_num()
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, M, N, K, num_cores)
    return mat_c
```

---

## 模板 3：激活函数类算子（Activation）

**代表**：SwiGLU、SiLU、GELU

**特征**：
- 逐元素或简单的逐行计算
- 使用 `get_npu_vectorcore_num()` 获取核数
- 可能需要升精度计算（sigmoid 需要 FP32）

```python
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride,
    n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = base_offsets + i
        mask = col_offsets < n_cols

        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
        c_row = silu(a_row).cast(b_row.dtype) * b_row
        tl.store(c_ptr + col_offsets, c_row, mask=mask)

def swiglu_forward(a: torch.Tensor, b: torch.Tensor):
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE = min(5120, triton.next_power_of_2(n_cols))

    swiglu_forward_kernel[(n_rows,)](
        a, b, c, c.stride(-2),
        n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return c.view(*ori_shape)
```

---

## 模板 4：损失函数类算子（Loss）

**代表**：CrossEntropy、FusedLinearCrossEntropy

**特征**：
- 复杂的多阶段计算（softmax + log + reduction）
- 使用 online softmax 算法优化
- 归约操作必须在 FP32 精度下进行

```python
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 3840

@triton.jit
def cross_entropy_kernel(
    X_ptr, X_stride,
    Y_ptr, Y_stride,
    loss_ptr,
    n_cols,
    n_non_ignore,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr).to(tl.int32)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id

    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)
    loss = lse - ori_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)

def cross_entropy_forward(_input: torch.Tensor, target: torch.Tensor,
                          ignore_index: int = -100, reduction: str = "mean"):
    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    cross_entropy_kernel[(n_rows,)](
        X_ptr=_input, X_stride=_input.stride(-2),
        Y_ptr=target, Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        n_cols=V, n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if reduction == "none":
        return loss_1d
    return torch.sum(loss_1d)
```

---

## 模板 5：索引变换类算子（Index）

**代表**：ConvertIndex、FusedRotaryEmb、Rope

**特征**：
- 涉及索引计算和变换
- 可能有条件分支逻辑
- 通常使用 vector core

```python
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]

@triton.jit
def convert_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    rows_per_prog: tl.constexpr,
    tiles_per_row: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bt_stride0, bt_stride1,
    ti_stride0, ti_stride1,
    out_stride0, out_stride1,
):
    row_blk_id = tl.program_id(0)

    for row_id in range(0, rows_per_prog):
        token_id = row_blk_id * rows_per_prog + row_id
        if token_id < num_tokens:
            req = tl.load(req_id_ptr + token_id)

            for tile_id in range(0, tiles_per_row):
                indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

                ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
                tok = tl.load(ti_ptr)

                is_invalid_tok = tok.to(tl.float32) < 0
                block_id = tok // BLOCK_SIZE
                inblock_off = tok % BLOCK_SIZE

                valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
                bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
                base = tl.load(bt_ptr, mask=valid_block, other=0)
                out_val = base * BLOCK_SIZE + inblock_off
                out_val = tl.where(is_invalid_tok, -1, out_val)

                out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
                tl.store(out_ptr_ij, out_val)

def convert_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    out = torch.empty_like(token_indices)

    bt_stride0, bt_stride1 = block_table.stride()
    ti_stride0, ti_stride1 = token_indices.stride()
    out_stride0, out_stride1 = out.stride()

    core_num = get_npu_vectorcore_num()
    rows_per_prog = triton.cdiv(num_tokens, core_num)

    grid = (core_num,)
    convert_index_kernel[grid](
        req_id, block_table, token_indices, out,
        max_num_blocks_per_req, rows_per_prog, tiles_per_row, num_tokens,
        BLOCK_SIZE, BLOCK_N,
        bt_stride0, bt_stride1, ti_stride0, ti_stride1, out_stride0, out_stride1,
    )
    return out
```

---

## 模板 6：注意力类算子（Attention）

**代表**：DSAPrefill、DSADecode、PagedAttention

**特征**：
- 使用 `tl.dot()` 进行 QK^T 和 SV 计算
- 复杂的多阶段计算（QK、Softmax、SV）
- 使用 AI Core

**关键点**：
- 使用 workspace 缓存中间结果
- 多层嵌套循环处理分块
- 使用 `tl.compile_hint` 优化 dot 操作

**核心模式**：
```python
@triton.jit
def attention_qk_kernel(
    Q_ptr, K_ptr, output_ptr,
    stride_qt, stride_qd,
    stride_kt, stride_kd,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    q_off = (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_qt + \
            tl.arange(0, D)[None, :] * stride_qd
    q = tl.load(Q_ptr + q_off)

    k_off = (n_start + tl.arange(0, BLOCK_N))[:, None] * stride_kt + \
            tl.arange(0, D)[None, :] * stride_kd
    k = tl.load(K_ptr + k_off)

    tl.compile_hint(q, "dot_pad_only_k")
    tl.compile_hint(k, "dot_pad_only_k")

    qk = tl.dot(q, k.trans())
    tl.store(output_ptr + m_start * BLOCK_N + n_start, qk)
```

---

## 模板 7：MoE 类算子

**代表**：FusedGDNgating、FusedRecurrent

**特征**：
- 门控机制计算
- 可能涉及 softplus、sigmoid 等激活函数
- 使用 vector core

```python
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]

@triton.jit
def fused_gating_kernel(
    g, beta_output, A_log, a, b, dt_bias,
    seq_len, beta: tl.constexpr, threshold: tl.constexpr,
    rows_per_prog: tl.constexpr, cols_per_row: tl.constexpr,
    NUM_BATCHES: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_BATCHES: tl.constexpr, BLOCK_HEADS: tl.constexpr,
):
    prog_id = tl.program_id(0)

    row_block_off = prog_id * rows_per_prog

    for i_d in range(0, cols_per_row):
        head_off = i_d * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        head_mask = head_off < NUM_HEADS

        blk_A_log = tl.load(A_log + head_off, mask=head_mask)
        blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

        for row_id in range(0, rows_per_prog, BLOCK_BATCHES):
            i_b = row_block_off + row_id + tl.arange(0, BLOCK_BATCHES)
            batch_off = i_b * seq_len * NUM_HEADS
            batch_mask = i_b < NUM_BATCHES

            tot_off = batch_off[:, None] + head_off[None, :]
            tot_mask = batch_mask[:, None] & head_mask[None, :]

            blk_a = tl.load(a + tot_off, mask=tot_mask)
            blk_b = tl.load(b + tot_off, mask=tot_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
            softplus_mask = beta * x <= threshold
            softplus_x = tl.where(
                softplus_mask, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
            )
            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))

            tl.store(g + tot_off, blk_g.to(g.dtype.element_ty), mask=tot_mask)
            tl.store(beta_output + tot_off, blk_beta_output.to(beta_output.dtype.element_ty), mask=tot_mask)

def fused_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    seq_len = 1
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)

    core_num = get_npu_vectorcore_num()

    if batch <= core_num:
        num_progs = batch
        rows_per_prog = 1
    else:
        num_progs = core_num
        rows_per_prog = triton.cdiv(batch, core_num)

    BLOCK_HEADS = num_heads
    BLOCK_BATCHES = min(196608 // (num_heads * 4 * 4), rows_per_prog) if rows_per_prog > 1 else 1
    cols_per_row = 1

    grid = (num_progs, seq_len)
    fused_gating_kernel[grid](
        g, beta_output, A_log, a, b, dt_bias,
        seq_len, beta, threshold, rows_per_prog, cols_per_row,
        batch, num_heads, BLOCK_BATCHES, BLOCK_HEADS,
    )
    return g, beta_output
```

---

## 模板 8：后处理类算子（Postprocess）

**代表**：ExpandBatchToTokens、RejectSample

**特征**：
- 简单的数据变换和扩展
- 通常不需要复杂的计算

```python
import torch
import triton
import triton.language as tl

@triton.jit
def expand_kernel(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)

    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)

def expand_batch_to_tokens(
    x: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
):
    batch_size = x.shape[0]
    expanded_x = x.new_empty(num_tokens)

    expand_kernel[(batch_size,)](
        expanded_x, x, cu_num_tokens,
        replace_from, replace_to,
        MAX_NUM_TOKENS=32,
    )
    return expanded_x
```

---

## 模板 9：卷积类算子（Conv）

**代表**：CausalConv1D、ConvUpdate

**特征**：
- 状态管理和更新
- 滑动窗口计算
- 可能有连续批处理支持

**关键点**：
- 状态索引计算
- 边界处理（padding）
- 条件分支（IS_CONTINUOUS_BATCHING）
