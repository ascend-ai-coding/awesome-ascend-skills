# Triton Ascend Layout-transform / Permutation 代码生成要点

当算子为 `permute`、`transpose`、`reshape`（触发 contiguous copy）等仅改变数据布局的算子时，代码生成应遵循以下原则。

## 1. 禁止退化为 PyTorch 布局操作

- `forward()` 中不能直接调用 `torch.permute`、`torch.transpose`、`x.t()` 等完成核心计算；所有数据搬移必须通过自定义 `@triton.jit` kernel 实现。

## 1.5 禁止单一 generic gather 作为主路径
- `forward()` 中禁止对所有输入调用同一个标量 gather/scatter kernel。
- 常见模式必须有独立 kernel，例如 `_transpose_2d_kernel`、`_batch_transpose_kernel`、`_swap_adjacent_kernel`。
- 通用 fallback kernel 必须是最后一条分支，且不应覆盖 2D / batch transpose 等高频情况。

## 1.6 推荐代码骨架（伪代码）

以下伪代码仅示意模式分发和专用 kernel 结构，变量名、tile 大小、边界处理需根据实际 shape 重新设计，禁止直接复制。

### forward() 模式分发

```python
def forward(self, x, dims):
    # 1. View 短路
    if dims == tuple(range(x.dim())):
        return x.view(out_shape)
    if _only_moves_size_one_dims(x.shape, dims):
        return x.view(out_shape)

    # 2. 连续维度合并，降低有效 rank
    merged_shape, merged_dims = _merge_contiguous_dims(x.shape, dims)

    # 3. 模式特化路由
    if len(merged_dims) == 2 and merged_dims == (1, 0):
        return _launch_transpose_2d(x, merged_shape, merged_dims)
    if _is_batch_transpose(merged_dims):
        return _launch_batch_transpose(x, merged_shape, merged_dims)
    if _is_swap_adjacent(merged_dims):
        return _launch_swap_adjacent(x, merged_shape, merged_dims)

    # 4. 罕见模式 fallback
    return _launch_generic_permute(x, merged_shape, merged_dims)
```

### 2D transpose 专用 kernel

```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=1),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=1),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=1),
], key=['M', 'N'])
@triton.jit
def transpose_2d_kernel(input_ptr, output_ptr, M, N,
                        in_stride_m: tl.constexpr, in_stride_n: tl.constexpr,
                        out_stride_m: tl.constexpr, out_stride_n: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 每个 block 负责一个 (BLOCK_M, BLOCK_N) 的 tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # 从输入的 (m, n) tile 连续读取
    in_ptrs = input_ptr + (offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n)
    tile = tl.load(in_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # 转置写入输出的 (n, m) tile
    out_ptrs = output_ptr + (offs_n[:, None] * out_stride_m + offs_m[None, :] * out_stride_n)
    tl.store(out_ptrs, tile.T, mask=mask_n[:, None] & mask_m[None, :])
```

### Batch transpose 专用 kernel

```python
@triton.jit
def batch_transpose_kernel(input_ptr, output_ptr, B, M, N,
                           in_stride_b: tl.constexpr, in_stride_m: tl.constexpr, in_stride_n: tl.constexpr,
                           out_stride_b: tl.constexpr, out_stride_m: tl.constexpr, out_stride_n: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    in_ptrs = (input_ptr
               + pid_b * in_stride_b
               + offs_m[:, None] * in_stride_m
               + offs_n[None, :] * in_stride_n)
    tile = tl.load(in_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    out_ptrs = (output_ptr
                + pid_b * out_stride_b
                + offs_n[:, None] * out_stride_m
                + offs_m[None, :] * out_stride_n)
    tl.store(out_ptrs, tile.T, mask=mask_n[:, None] & mask_m[None, :])
```

### 关键约束
- 专用 kernel 内部使用规则的二维 tile load/store，避免逐元素 div/mod 索引。
- `forward()` 必须优先走 view 短路和模式特化分支，generic gather 仅用于无法识别的 permutation。
- grid 大小按 `num_cores` 限制，kernel 内循环处理多 block。

## 1.7 强制要求：禁止 element-wise gather 冒充 pattern specialization

以下要求与“禁止单一 generic gather”并列，必须同时满足：

- **每个常见模式专用 kernel 内部必须使用 tile-based 连续 `tl.load`/`tl.store`**；禁止用逐元素 `div`/`mod` 或 `tl.where` 链在专用 kernel 中实现 gather/scatter。
- **`forward()` 在模式特化前必须通过 `view` 合并连续维度**，降低有效 rank 后再调用专用 kernel。
- 2D / batch transpose / swap-adjacent 等高频模式若仍使用 element-wise 索引，视为未真正特化，必须重写为 tile-based kernel。

## 2. 常见置换模式特化

- 生成多个专用 kernel，分别处理：2D transpose、batch transpose、swap adjacent dims、reverse dims 等常见模式。
- 每个专用 kernel 应使用连续/规则的 `tl.load`/`tl.store`，避免逐元素 gather/scatter。

## 3. 连续维度合并

- 在 `forward()` 中通过 reshape 把置换后仍连续的维度合并，再调用专用 kernel。例如 `[A,B,M,N] -> [A,B,N,M]` 可合并为 `view(A*B, M, N)` 后走 batch transpose。

## 4. View 短路

- identity permutation 或仅移动 size-1 维度时，直接返回 `x.view(*out_shape)`，不分配输出、不启动 kernel。

## 5. Autotune 与 Grid 配置

- 对 2D/batch transpose 类 kernel，使用 `@triton.autotune` 覆盖 `(BLOCK_M, BLOCK_N)` 组合。
- grid 大小按 `num_cores` 限制，kernel 内循环处理多 block；避免 grid 远大于物理核数导致调度开销。

## 6. 退化 shape 路由

- batch 维度为 1 时，batch transpose 应路由到更简单的 2D transpose kernel。

## 7. 通用 fallback

- 仅对罕见 permutation 模式保留 1D gather/scatter fallback，且应保证其正确性优先于性能。
