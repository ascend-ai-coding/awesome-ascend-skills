> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M4：host 装配与 launch 积木（源 L1637-1934）

> **为什么**：host 只做 tilingdata 装配与身份登记，不做任何与设备相关的猜测；tiling 值来自 [容量方程](../../optimize/3-capacity-equations.md) 的推导结果，不重复计算。
```python

class _SparseFlashAttentionLauncher:
    """Hold JIT launch configuration for native-layout tensors."""

    def __init__(
        self,
        heads,
        sparse_count,
        total_queries,
        head_chunks,
        queries_per_group,
        dtype,
        block_dim,
        use_sinks,
        return_lse,
        layout_query,
        layout_kv,
        batch,
        sparse_mode,
        has_rope=True,
    ):
        self.heads = heads
        self.sparse_count = sparse_count
        self.total_queries = total_queries
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype = dtype
        self.block_dim = block_dim
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.batch = batch
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope

    # Keep this argument order aligned with the kernel and launcher call sites.
    @host
    def launch(
        self,
        output: Tensor,
        query: Tensor,
        key: Tensor,
        indices: Tensor,
        query_rope: Tensor,
        key_rope: Tensor,
        sinks: Tensor,
        maxima: Tensor,
        sums: Tensor,
        workspace: Tensor,
        table: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        used_q: Tensor,
        used_kv: Tensor,
        scale: float,
    ):
        op = _SparseFlashAttentionKernel(
            TILE_M,
            self.sparse_count,
            self.head_chunks,
            self.queries_per_group,
            self.dtype,
            self.use_sinks,
            self.return_lse,
            self.layout_query,
            self.heads,
            self.total_queries,
            self.layout_kv,
            self.batch,
            self.sparse_mode,
            self.has_rope,
        )
        op[self.block_dim](
            output,
            query,
            key,
            indices,
            query_rope,
            key_rope,
            sinks,
            maxima,
            sums,
            workspace,
            table,
            cu_q,
            cu_kv,
            used_q,
            used_kv,
            scale,
        )


def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    scale_value=1.0,
    sparse_block_size=SUPPORTED_SPARSE_BLOCK_SIZE,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=SPARSE_MODE_DENSE,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=SUPPORTED_ATTENTION_MODE,
    return_softmax_lse=False,
):
    """Run sparse MLA attention for valid-prefix, minus-one-padded indices."""
    validate_sparse_flash_attention_args(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        block_table=block_table,
        query_rope=query_rope,
        key_rope=key_rope,
        sinks=sinks,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )

    return _native_sparse_launch(
        query,
        key,
        query_rope,
        key_rope,
        sparse_indices,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        scale_value,
        return_softmax_lse,
        layout_query,
        layout_kv,
        sparse_mode,
    )


def _calculate_core_partition(total, heads, core_num):
    """Return head-chunk count, queries per group, and launch block count."""
    head_chunks = (heads + TILE_M - 1) // TILE_M
    core_groups = core_num // head_chunks
    if core_groups == 0:
        raise ValueError(
            f"Head tiles ({head_chunks}) exceed available cube cores ({core_num})"
        )
    queries_per_group = (total + core_groups - 1) // core_groups
    used_core_groups = (total + queries_per_group - 1) // queries_per_group
    block_dim = used_core_groups * head_chunks
    return head_chunks, queries_per_group, block_dim


def _native_sparse_launch(
    query,
    key,
    query_rope,
    key_rope,
    sparse_indices,
    table,
    cu_q,
    cu_kv,
    used_q,
    used_kv,
    sinks,
    scale,
    return_lse,
    layout,
    layout_kv,
    sparse_mode,
):
    """Allocate outputs and workspace, then launch the configured kernel."""
    has_rope = query_rope is not None
    batch = query.shape[0] if layout == "BSND" else cu_q.numel() - 1
    total = query.shape[0] * query.shape[1] if layout == "BSND" else query.shape[0]
    heads = query.shape[-2]
    index_capacity = sparse_indices.shape[-1]
    indices = sparse_indices.view(total, index_capacity)
    core_num = torch.npu.get_device_properties(query.device).cube_core_num
    head_chunks, queries_per_group, block_dim = _calculate_core_partition(
        total, heads, core_num
    )

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)

    # Reserve padded max/sum output buffers for all head tiles.
    state_shape = (total, head_chunks * TILE_M, 1)
    max_out = torch.empty(state_shape, dtype=torch.float32, device=query.device)
    sum_out = torch.empty(state_shape, dtype=torch.float32, device=query.device)

    sink_view = None if sinks is None else sinks.view(heads, 1)
    # Each core reuses its three workspace slots across pipeline ticks.
    qk_dim = DIM_QK if has_rope else DIM_NOPE
    workspace = torch.empty(
        (block_dim * PIPELINE_SLOT_COUNT * TILE_S2, qk_dim),
        dtype=query.dtype,
        device=query.device,
    )
    launcher = _SparseFlashAttentionLauncher(
        heads,
        index_capacity,
        total,
        head_chunks,
        queries_per_group,
        dtypes.bfloat16 if query.dtype == torch.bfloat16 else dtypes.float16,
        block_dim,
        sinks is not None,
        return_lse,
        layout,
        layout_kv,
        batch,
        sparse_mode,
        has_rope,
    )
    launcher.launch(
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sink_view,
        max_out,
        sum_out,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    )

    if return_lse:
        max_out = max_out[:, :heads, 0]
        sum_out = sum_out[:, :heads, 0]
        if layout == "BSND":
            # Restore [B, 1, S1, H] stats from the flattened query axis.
            max_out = max_out.view(batch, query.shape[1], heads).unsqueeze(1)
            sum_out = sum_out.view(batch, query.shape[1], heads).unsqueeze(1)
        else:
            max_out = max_out.unsqueeze(0)
            sum_out = sum_out.unsqueeze(0)
    else:
        max_out = torch.empty((0,), dtype=torch.float32, device=query.device)
        sum_out = torch.empty((0,), dtype=torch.float32, device=query.device)
    return output, max_out, sum_out


def try_dispatch(query, key, value, sparse_indices, block_table, query_rope,
                 key_rope, sinks, cu_seqlens_q, cu_seqlens_kv, seqused_q,
                 seqused_kv, scale_value, sparse_block_size, layout_query,
                 layout_kv, sparse_mode, pre_tokens, next_tokens, attention_mode,
                 return_softmax_lse):
    """Dispatch the verified metadata intersection on the exact runtime SoC."""
    if not (query.dtype == key.dtype == torch.bfloat16
            and sparse_indices.dtype == torch.int32
            and tuple(query.shape) == (1, 4096, 64, 512)
            and tuple(key.shape) == (1, 8192, 1, 512)
            and tuple(sparse_indices.shape) == (1, 4096, 1, 640)
            and value is key
            and layout_query == layout_kv == "BSND"
            and sparse_mode == 0 and attention_mode == 2 and sparse_block_size == 1
            and not return_softmax_lse and pre_tokens == next_tokens == INT64_MAX
            and all(x is None for x in (block_table, query_rope, key_rope, sinks,
                    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv))):
        return None
    import ctypes
    with torch.npu.device(query.device):
        runtime = ctypes.CDLL("libascendcl.so")
        runtime.aclrtGetSocName.restype = ctypes.c_char_p
        if runtime.aclrtGetSocName() != b"Ascend950PR_957b":
            return None
        return _native_sparse_launch(query, key, query_rope, key_rope,
            sparse_indices, block_table, cu_seqlens_q, cu_seqlens_kv,
            seqused_q, seqused_kv, sinks, scale_value, return_softmax_lse,
            layout_query, layout_kv, sparse_mode)
```

