> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M3：双核主循环积木——AIC/AIV tick 主循环与跨核交接胶水（源 L757-1636，含 @kernel 入口）

> **为什么**：AIC 的 QK→PV 相位落后 1 tick，句柄必须跨 tick carry，否则 PV 读错代；AIV 侧 I→G→S→U 延迟线的 tap 深度错会读写竞速。这些跨拍约束只能住在类里（胶水随类走的根因），见 [流水建模](../../optimize/2-pipeline-modeling.md) §2/§3。
```python
@kernel(profile=cannbotdsl.ProfileSpec(
    name="SparseFlashAttention_DSL", op_type="SparseFlashAttention_DSL",
    inputs=("query",), outputs=("output",),
))
class _SparseFlashAttentionKernel:
    """Process the query rows assigned to one AIC head group."""

    def __init__(
        self,
        group_size: int,
        sparse_count: int,
        head_chunks: int,
        queries_per_group: int,
        dtype_16,
        use_sinks: bool,
        return_lse: bool,
        layout_query: str = "BSND",
        head_count: int = 64,
        total_queries: int = 1,
        layout_kv: str = "BSND",
        batch_count: int = 1,
        sparse_mode: int = 0,
        has_rope: bool = True,
    ):
        self.group_size = group_size
        self.vec_m = (group_size + 1) // 2
        self.sparse_count = sparse_count
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype_16 = dtype_16
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.batch_count = batch_count
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope
        self.qk_dim = DIM_QK if has_rope else DIM_NOPE
        self.head_count = head_count
        self.total_queries = total_queries

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.matmul = _SparseMatmul(
            group_size,
            dtype_16,
            has_rope,
        )
        self.vector = _SFAOnlineSoftmaxVector(
            self.vec_m,
            TILE_S2,
            DIM_NOPE,
            self.subblock_idx,
            dtype_16=dtype_16,
            preload_num=SOFTMAX_STATE_SLOTS,
        )

        self.qk_ub = Channel(
            MemLoc.UB,
            (self.vec_m, TILE_S2),
            dtypes.float32,
            depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.pv_halves = [Channel(
            MemLoc.UB, (self.vec_m * 2, 256), dtypes.float32,
            depth=1, kind=ChannelKind.CrossCore,
        ) for _ in range(2)]
        # Both AIV partitions form one P tile for the AIC.
        self.p_channel = Channel(
            MemLoc.L1, (group_size, TILE_S2), dtype_16,
            depth=1, kind=ChannelKind.CrossCore,
        )

        # Ping-pong sparse loads on Vec0: depth=2 lets the gather DMA (mte2) of
        # the next chunk overlap the nd2nz (vec) of the current chunk, which is
        # the AIV critical path now that nd2nz runs on the vector core.
        self.stage0 = [
            Buffer(MemLoc.UB, (GATHER_CHUNK_ROWS, self.qk_dim), dtype_16)
            for _ in range(2)
        ]

        # AIV-side ND->NZ staging: one gather chunk converted to NZ before the
        # cross-core push into the shared L1 right_root slot (ds-style, moves
        # the nd2nz transform off the cube core onto the vector core).
        self.nz_staging = [
            Buffer(
                MemLoc.UB, (GATHER_CHUNK_ROWS, DIM_NOPE), dtype_16,
                data_format="nz",
                n1_pad=NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT,
            )
            for _ in range(2)
        ]
        # Carry each consumed KV L1 handle from the QK stage to the PV stage.
        self._kv_handles = [None] * PIPELINE_SLOT_COUNT

        # Load indices one Vector tick ahead, so gather does not wait for
        # the preceding softmax/update before issuing its GM reads.
        self.idx_buf = [Buffer(
            MemLoc.UB, (1, TILE_S2), dtypes.int32
        ) for _ in range(2)]

        # Channels order GM transfers around Vec1 softmax state updates.
        self.sink_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()
        self.lse_max_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()
        self.lse_sum_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()

    def __call__(
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
        if cannbotdsl.is_aic():
            self._cube_program(output, query, key, indices, query_rope, key_rope, sinks, maxima, sums, workspace, table, cu_q, cu_kv, used_q, used_kv, scale)
        else:
            self._vector_program(output, query, key, indices, query_rope, key_rope, sinks, maxima, sums, workspace, table, cu_q, cu_kv, used_q, used_kv, scale)

    @jit
    def _cube_program(
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
        # Map this core to a query group and a head tile.
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * TILE_M

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = min(first_query + self.queries_per_group, self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = dtypes.int64(0)
        is_first_block = dtypes.int64(1)
        batch = dtypes.int64(0)
        sparse_end = dtypes.int64(0)
        is_query_active = dtypes.int64(0)

        # Nonempty tasks cover at most one KV tile; EMPTY_QUERY_ID marks a bubble.
        task_pipeline = DelayLineGroup(
            PIPELINE_STAGE_COUNT,
            "query_id",
            "valid_kv_count",
            "is_first_block",
            "is_last_block",
            "is_query_active",
        )
        # Each tick gathers a new KV tile while older tiles move through
        # QK, softmax/PV, and output. The final three ticks drain the pipeline.
        max_gather_tasks = query_count * (
            (self.sparse_count + TILE_S2 - 1) // TILE_S2
        )
        # CrossCore KV handoff: the AIV gather produces one L1 slot per tile and
        # the cube side consumes it once. QK reads the freshly consumed handle
        # and then advances to the next; PV reuses the handle QK read one tick
        # earlier (carried via kv_pv), so a single produce feeds both readers.
        if query_count > 0:
            self.matmul.load_q_wide(self._query_tile(query, first_query, head_offset), first_query % 2)
        kv_lo = self.matmul.right_lower.consume()
        kv_hi = self.matmul.right_upper.consume()
        pv_lo, pv_hi = kv_lo, kv_hi
        for tick in range(max_gather_tasks + PIPELINE_DRAIN_TICKS):
            next_lo, next_hi = pv_lo, pv_hi
            # Stage 0: gather a KV tile and register its task.
            if current_query < query_end:
                # A new Query needs its batch and valid sparse range.
                if is_first_block != 0:
                    sparse_idx = 0
                    is_query_active = 0
                    sparse_end = 0
                    (
                        batch,
                        unused_kv_length,
                        unused_kv_end,
                        planned_sparse_end,
                        is_query_active,
                    ) = self._prepare_sparse_query(
                        query, key, table, cu_q, cu_kv,
                        used_q, used_kv, current_query,
                    )
                    sparse_end = self._find_valid_end(
                        indices, current_query, planned_sparse_end,
                    )

                # Gather at most one TILE_S2-sized block.
                valid_kv_count = dtypes.int64(0)
                if sparse_idx < sparse_end:
                    valid_kv_count = min(TILE_S2, sparse_end - sparse_idx)
                    sparse_idx += valid_kv_count

                # The last block lets the gather stage move to the next Query.
                is_last_block = 1 if sparse_idx >= sparse_end else 0
                task_pipeline.push(
                    query_id=current_query,
                    valid_kv_count=valid_kv_count,
                    is_first_block=is_first_block,
                    is_last_block=is_last_block,
                    is_query_active=is_query_active,
                )
                is_first_block = 0
                if is_last_block != 0:
                    current_query += 1
                    is_first_block = 1
            else:
                # Fill unused ticks so older tasks can finish the pipeline.
                task_pipeline.push(
                    query_id=EMPTY_QUERY_ID,
                    valid_kv_count=0,
                    is_first_block=0,
                    is_last_block=0,
                    is_query_active=0,
                )

            # Stage 1: compute QK for the tile gathered one tick ago.
            if tick >= 1:
                qk_task = task_pipeline.tap(1)
                if qk_task.query_id >= 0 and qk_task.valid_kv_count > 0:
                    if qk_task.is_first_block != 0:
                        self.matmul.load_q_tail(self._query_tile(query, qk_task.query_id, head_offset))
                    self._stage_qk(kv_lo, kv_hi, qk_task.valid_kv_count, qk_task.query_id % 2)
                    next_lo, next_hi = kv_lo, kv_hi
                    kv_lo = self.matmul.right_lower.consume()
                    kv_hi = self.matmul.right_upper.consume()
                # Empty queries still prefetch the next generation, but submit
                # neither QK nor tail. A following nonempty query therefore
                # never relies on a skipped producer.
                if qk_task.query_id >= 0 and qk_task.is_first_block != 0:
                    if qk_task.query_id + 1 < query_end:
                        self.matmul.load_q_wide(self._query_tile(query, qk_task.query_id + 1, head_offset), (qk_task.query_id + 1) % 2)


            # Stage 2: update online softmax and compute PV for the tile
            # gathered two ticks ago.
            if tick >= 2:
                compute_task = task_pipeline.tap(2)
                if compute_task.query_id >= 0:
                    if compute_task.valid_kv_count > 0:
                        self._stage_pv(pv_lo, pv_hi, compute_task.valid_kv_count, compute_task.is_first_block)

            # Carry the QK handles one tick so PV reads the same published slot.
            pv_lo, pv_hi = next_lo, next_hi
            task_pipeline.advance()

    @jit
    def _vector_program(
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
        # Map this core to a query group and a head tile.
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * TILE_M

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = min(first_query + self.queries_per_group, self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = dtypes.int64(0)
        is_first_block = dtypes.int64(1)
        batch = dtypes.int64(0)
        sparse_end = dtypes.int64(0)
        is_query_active = dtypes.int64(0)

        # Nonempty tasks cover at most one KV tile; EMPTY_QUERY_ID marks a bubble.
        task_pipeline = DelayLineGroup(
            PIPELINE_STAGE_COUNT + 1,
            "query_id",
            "valid_kv_count",
            "is_first_block",
            "is_last_block",
            "is_query_active",
            "sparse_idx", "sparse_end", "batch",
        )
        # Each tick gathers a new KV tile while older tiles move through
        # QK, softmax/PV, and output. The final three ticks drain the pipeline.
        max_gather_tasks = query_count * (
            (self.sparse_count + TILE_S2 - 1) // TILE_S2
        )
        # CrossCore KV handoff: each AIV pushes its 64-row half into one of the
        # two CrossCore L1 rings; the cube consumes each slot once per tile and
        # carries the same handles from QK to the following PV.
        kv_lo = self.matmul.right_lower.produce()
        kv_hi = self.matmul.right_upper.produce()
        for tick in range(max_gather_tasks + PIPELINE_DRAIN_TICKS + 1):
            gather_n = dtypes.int64(0)
            gather_query = current_query
            gather_idx = sparse_idx
            gather_end = sparse_end
            gather_batch = batch
            # Stage 0: gather a KV tile and register its task.
            if current_query < query_end:
                # A new Query needs its batch and valid sparse range.
                if is_first_block != 0:
                    sparse_idx = 0
                    is_query_active = 0
                    sparse_end = 0
                    (
                        batch,
                        unused_kv_length,
                        unused_kv_end,
                        planned_sparse_end,
                        is_query_active,
                    ) = self._prepare_sparse_query(
                        query, key, table, cu_q, cu_kv,
                        used_q, used_kv, current_query,
                    )
                    sparse_end = self._find_valid_end(
                        indices, current_query, planned_sparse_end,
                    )

                # Gather at most one TILE_S2-sized block.
                valid_kv_count = dtypes.int64(0)
                if sparse_idx < sparse_end:
                    valid_kv_count = min(TILE_S2, sparse_end - sparse_idx)
                    gather_n = valid_kv_count
                    gather_query = current_query
                    gather_idx = sparse_idx
                    gather_end = sparse_end
                    gather_batch = batch
                    sparse_idx += valid_kv_count

                # The last block lets the gather stage move to the next Query.
                is_last_block = 1 if sparse_idx >= sparse_end else 0
                task_pipeline.push(
                    query_id=current_query,
                    valid_kv_count=valid_kv_count,
                    is_first_block=is_first_block,
                    is_last_block=is_last_block,
                    is_query_active=is_query_active,
                    sparse_idx=gather_idx, sparse_end=gather_end, batch=gather_batch,
                )
                is_first_block = 0
                if is_last_block != 0:
                    current_query += 1
                    is_first_block = 1
            else:
                # Fill unused ticks so older tasks can finish the pipeline.
                task_pipeline.push(
                    query_id=EMPTY_QUERY_ID,
                    valid_kv_count=0,
                    is_first_block=0,
                    is_last_block=0,
                    is_query_active=0,
                    sparse_idx=0, sparse_end=0, batch=0,
                )

            # I(t) is independent of A(t-1). The latter reads a descriptor
            # whose VF producer was issued before the preceding tick's E.
            if gather_n > 0:
                self._gather_setup(tick % 2, gather_query, gather_idx, gather_n, indices)
            gather_n = dtypes.int64(0)
            if tick >= 1:
                gather_task = task_pipeline.tap(1)
                if gather_task.query_id >= 0:
                    gather_n = gather_task.valid_kv_count
                    gather_query = gather_task.query_id
                    gather_idx = gather_task.sparse_idx
                    gather_end = gather_task.sparse_end
                    gather_batch = gather_task.batch

            if gather_n > 0:
                self._gather_chunks(kv_lo, kv_hi, key.view(-1), 0, 2, (tick - 1) % 2, gather_n)

            # Stage 2: update online softmax and compute PV for the tile
            # gathered two ticks ago.
            if tick >= 3:
                compute_task = task_pipeline.tap(3)
                if compute_task.query_id >= 0:
                    state_slot = compute_task.query_id % SOFTMAX_STATE_SLOTS
                    if compute_task.is_first_block != 0:
                        self._init_query_state(
                            sinks,
                            head_offset,
                            (
                                compute_task.valid_kv_count
                                if compute_task.is_query_active != 0
                                else -1
                            ),
                            state_slot,
                        )
                    if compute_task.valid_kv_count > 0:
                        self._stage_softmax(
                            compute_task.is_first_block,
                            compute_task.valid_kv_count,
                            (tick - 3) % SOFTMAX_STATE_SLOTS,
                            state_slot,
                            scale,
                        )

            if gather_n > 0:
                self._gather_chunks(kv_lo, kv_hi, key.view(-1), 2, 4, (tick - 1) % 2, gather_n)
                kv_lo = self.matmul.right_lower.produce()
                kv_hi = self.matmul.right_upper.produce()

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tick >= 4:
                output_task = task_pipeline.tap(4)
                if output_task.query_id >= 0:
                    self._stage_output(
                        output_task.is_first_block,
                        output_task.is_last_block,
                        output_task.valid_kv_count,
                        (tick - 4) % SOFTMAX_STATE_SLOTS,
                        output_task.query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

            task_pipeline.advance()

    @jit
    def _fill_softmax_sum_one(self, state_slot=0):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.vec_m, elem_bits=32)
            ones = rr.vdups(1.0, dtypes.float32, mask=mask)
            rr.vstore(self.vector.sm_sum_tb[state_slot], 0, ones, mask)

    @jit
    def _store_softmax_state(self, max_token, sum_token, state_slot=0):
        mem_copy(reinterpret(self.lse_max_state, shape=(self.vec_m, 1)),
                 self.vector.sm_max_tb[state_slot])
        mem_copy(
            max_token.partition(dim=0, part_id=self.subblock_idx),
            reinterpret(self.lse_max_state, shape=(self.vec_m, 1)),
        )
        mem_copy(reinterpret(self.lse_sum_state, shape=(self.vec_m, 1)),
                 self.vector.sm_sum_tb[state_slot])
        mem_copy(
            sum_token.partition(dim=0, part_id=self.subblock_idx),
            reinterpret(self.lse_sum_state, shape=(self.vec_m, 1)),
        )

    @jit
    def _kv_token(self, gm, token):
        """Select one physical KV row without flattening paged strides."""
        if const_expr(self.layout_kv == "TND"):
            span = gm[token, None, None]
        else:
            span = gm[token // gm.shape[1], token % gm.shape[1], None, None]
        return tile_slice(span, (1, gm.shape[-1]), (0, 0))

    @jit
    def _kv_row(self, gm, batch, kv_idx):
        """BSND fast path: index (batch, kv_idx) directly, no div/mod."""
        span = gm[batch, kv_idx, None, None]
        return tile_slice(span, (1, gm.shape[-1]), (0, 0))

    @jit
    def _build_pair_desc_from(self, n_pairs, idx_buf, desc_first, desc_second):
        """Fold raw indices into per-pair min/max *byte offsets* in UB.

        The ``idx -> byte offset`` multiply (idx * row_bytes) is done here,
        vectorized and once, so the scalar issue loop only reads the offset
        and feeds it to a base+offset DMA (ds-style, no per-row multiply).
        """
        with vf(mode="simd"):
            mask, _ = rr.update_mask(n_pairs, elem_bits=32)
            raw_indices = rr.vload(idx_buf, 0)
            even, odd = rr.vdeinterleave(raw_indices, raw_indices)
            first = rr.vmin(even, odd, mask=mask)
            second = rr.vmax(even, odd, mask=mask)
            first = rr.vmuls(first, DIM_NOPE, mask=mask)
            second = rr.vmuls(second, DIM_NOPE, mask=mask)
            rr.vstore(desc_first, 0, first, mask)
            rr.vstore(desc_second, 0, second, mask)

    @jit
    def _gather_setup(self, slot, query_id, sparse_idx, tile_rows, indices):
        """Prefetch this AIV's sparse row indices into a ping-pong slot."""
        half = TILE_S2 // AIV_COUNT
        local_start = self.subblock_idx * half
        local_end = min(local_start + half, tile_rows)
        half_rows = local_end - local_start
        if half_rows > 0:
            index_base = sparse_idx + local_start
            index_row = tile_slice(
                indices[query_id, None], (1, indices.shape[1]), (0, 0)
            )
            mem_copy(
                tile_slice(self.idx_buf[slot], (1, TILE_S2), (0, 0)),
                index_row[None, index_base:index_base + half_rows],
            )
        return half_rows

    @jit
    def _copy_kv_pair_bytes(self, stage, key_flat, off0, off1, pair_index):
        """burst-2 gather of two KV rows via GM base + prestored elem offset."""
        row0 = key_flat[off0:off0 + DIM_NOPE,].view(1, DIM_NOPE)
        row1 = key_flat[off1:off1 + DIM_NOPE,].view(1, DIM_NOPE)
        dst = tile_slice(
            stage, (SPARSE_INDEX_PAIR_SIZE, DIM_NOPE), (pair_index, 0)
        )
        mem_copy(dst, (row0, row1))

    @jit
    def _copy_kv_token_bytes(self, stage, key_flat, off, row):
        """Single-row gather via GM base + prestored elem offset."""
        rowv = key_flat[off:off + DIM_NOPE,].view(1, DIM_NOPE)
        dst = tile_slice(stage, (1, DIM_NOPE), (row, 0))
        mem_copy(dst, rowv)

    @jit
    def _workspace_tile(self, workspace, task_slot):
        """Return one slot from this block's GM workspace ring."""
        return tile_slice(workspace, (TILE_S2, self.qk_dim),
            (self.block_idx * PIPELINE_SLOT_COUNT + task_slot, 0))

    @jit
    def _find_valid_end(self, indices, query_id, planned_end):
        """Find the first padding entry in a valid-prefix sparse row."""
        lo = dtypes.int64(0)
        hi = planned_end
        # Preserve the valid-prefix contract; zero bounds never load -1.
        if planned_end > 0:
            if dtypes.int64(indices[query_id, planned_end - 1]) != INVALID_SPARSE_INDEX:
                lo = planned_end
        while lo < hi:
            mid = (lo + hi) // 2
            if dtypes.int64(indices[query_id, mid]) == INVALID_SPARSE_INDEX:
                hi = mid
            else:
                lo = mid + 1
        return lo

    @jit
    def _nd2nz_chunk(self, ready, nz):
        """Convert one 16-row ND gather chunk to NZ in UB (ds-style, vector).

        Uses the proven vstore_strided NZ-store idiom so the cube core no
        longer carries the nd2nz transform. ``nz`` is a produced depth-2
        staging slot so the next chunk's nd2nz overlaps this chunk's L1 push.
        """
        m0, s_m1, s_m0, block_stride = _nz_params(nz)
        s_n1 = block_stride * (NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT)
        with vf(mode="simd"):
            b16_full, _ = rr.update_mask(128, elem_bits=16)
            for row in range(GATHER_CHUNK_ROWS):
                nz_m = (row // m0) * s_m1 + (row % m0) * s_m0
                # ds-style 2-bank interleave: load two adjacent 128-element
                # column blocks, then store both, so the two vstore streams
                # land in different 256B UB bank groups and pipeline instead
                # of serializing on bank conflicts.
                for fb in range(0, DIM_NOPE, 256):
                    reg0 = rr.vload(ready, row * DIM_NOPE + fb)
                    reg1 = rr.vload(ready, row * DIM_NOPE + fb + 128)
                    rr.vstore_strided(
                        nz, nz_m + (fb // 16) * s_n1, reg0, b16_full,
                        block_stride=block_stride, repeat_stride=0,
                    )
                    rr.vstore_strided(
                        nz, nz_m + ((fb + 128) // 16) * s_n1, reg1, b16_full,
                        block_stride=block_stride, repeat_stride=0,
                    )

    @jit
    def _gather_chunks(self, kv_lo, kv_hi, key_flat, chunk_start, chunk_end, desc_slot, tile_rows):
        """Constexpr gather of chunk range [chunk_start, chunk_end) of this
        AIV's complete 64-row half. Splitting the range lets the caller fire
        the front half, run softmax, then fire the back half so the gather
        MTE2 loads overlap the vector-heavy softmax."""
        chunks = (TILE_S2 // AIV_COUNT) // GATHER_CHUNK_ROWS
        pairs = GATHER_CHUNK_ROWS // SPARSE_INDEX_PAIR_SIZE
        local_rows = min(TILE_S2 // AIV_COUNT, max(0,
            tile_rows - self.subblock_idx * (TILE_S2 // AIV_COUNT)))
        for chunk in range_constexpr(chunk_start, chunk_end):
            stage = self.stage0[chunk % 2]
            if local_rows >= (chunk + 1) * GATHER_CHUNK_ROWS:
                self._gather_chunk_rows(stage, key_flat, desc_slot, chunk, local_rows, True)
            else:
                with vf(mode="simd"):
                    mask = rr.full_mask(elem_bits=16)
                    zero = rr.vdups(0.0, self.dtype_16, mask=mask)
                    for elem in range(GATHER_CHUNK_ROWS * DIM_NOPE // 128):
                        rr.vstore(stage, elem * 128, zero, mask)
                self._gather_chunk_rows(stage, key_flat, desc_slot, chunk, local_rows, False)
        # Both 16-row DMA batches are in flight before VECTOR consumes either.
        for chunk in range_constexpr(chunk_start, chunk_end):
            ready = self.stage0[chunk % 2]
            nz_slot = self.nz_staging[chunk % 2]
            self._nd2nz_chunk(ready, nz_slot)
            if self.subblock_idx == 0:
                mem_copy(tile_slice(kv_lo, (GATHER_CHUNK_ROWS, DIM_NOPE), (chunk, 0)),
                         nz_slot)
            else:
                mem_copy(tile_slice(kv_hi, (GATHER_CHUNK_ROWS, DIM_NOPE), (chunk, 0)),
                         nz_slot)

    @jit
    def _gather_chunk_rows(self, stage, key_flat, desc_slot, chunk, local_rows, full):
        for pair in range_constexpr(GATHER_CHUNK_ROWS // 2):
            row = chunk * GATHER_CHUNK_ROWS + pair * 2
            if const_expr(full):
                idx0 = dtypes.int64(self.idx_buf[desc_slot][0, row])
                idx1 = dtypes.int64(self.idx_buf[desc_slot][0, row + 1])
                lo = idx0 if idx0 < idx1 else idx1
                hi = idx1 if idx0 < idx1 else idx0
                self._copy_kv_pair_bytes(stage, key_flat, lo * DIM_NOPE, hi * DIM_NOPE, pair)
            else:
                if row + 1 < local_rows:
                    idx0 = dtypes.int64(self.idx_buf[desc_slot][0, row])
                    idx1 = dtypes.int64(self.idx_buf[desc_slot][0, row + 1])
                    lo = idx0 if idx0 < idx1 else idx1
                    hi = idx1 if idx0 < idx1 else idx0
                    self._copy_kv_pair_bytes(stage, key_flat, lo * DIM_NOPE, hi * DIM_NOPE, pair)
                elif row < local_rows:
                    idx0 = dtypes.int64(self.idx_buf[desc_slot][0, row])
                    self._copy_kv_token_bytes(stage, key_flat, idx0 * DIM_NOPE, pair * 2)

    @jit
    def _stage_qk(self, kv_lo, kv_hi, valid_n, q_slot):
        """Compute QK from the AIV-published NZ KV tile (no cube-side nd2nz).

        ``kv_lo``/``kv_hi`` are the CrossCore handles consumed in the pipeline
        loop; the first data access waits on the Channel's cross-core publish.
        """
        self.matmul.compute_qk(kv_lo, kv_hi, valid_n, q_slot)
        self.matmul.store_score(self.qk_ub, valid_n)

    @jit
    def _store_p(self):
        self.vector.store_p(self.p_channel.produce())

    @jit
    def _stage_pv(self, kv_lo, kv_hi, valid_n, first):
        # Same CrossCore KV handles the QK stage read, carried one tick forward.
        self.matmul.compute_pv(kv_lo, kv_hi, self.p_channel.consume(), valid_n,
                               self.pv_halves[0].produce(), self.pv_halves[1].produce(), first)

    @jit
    def _cross_init_state(self, state_slot, valid_count):
        """Reset running softmax max and sum for one state slot."""
        with vf(mode="simd"):
            rows, _ = rr.update_mask(self.vec_m, elem_bits=32)
            zero = rr.vdups(0.0, dtypes.float32, mask=rows)
            initial = rr.vdups(
                SOFTMAX_NEG_INF if valid_count > 0 else 0.0,
                dtypes.float32,
                mask=rows,
            )
            rr.vstore(self.vector.sm_max_tb[state_slot], 0, initial, rows)
            rr.vstore(self.vector.sm_sum_tb[state_slot], 0, zero, rows)

    @jit
    def _query_tile(self, gm, query_id, head_offset):
        return _native_head_tile(gm, self.layout_query, query_id, head_offset, TILE_M)

    @jit
    def _output_tile(self, gm, query_id, head_offset, rows):
        return _native_head_tile(gm, self.layout_query, query_id, head_offset, rows)

    @jit
    def _sequence_length(self, cu, used, batch, capacity, packed):
        """Prefer the used length; otherwise use the packed length or capacity."""
        length = capacity
        if const_expr(packed):
            length = dtypes.int64(cu[batch + 1]) - dtypes.int64(cu[batch])
        if const_expr(used is not None):
            length = dtypes.int64(used[batch])
        return length

    @jit
    def _q_length(self, query, cu, used, batch):
        capacity = query.shape[1] if self.layout_query == "BSND" else query.shape[0]
        return self._sequence_length(cu, used, batch, capacity, self.layout_query == "TND")

    @jit
    def _kv_capacity(self, key, table):
        """Return the logical KV capacity for the active storage layout."""
        if const_expr(self.layout_kv == "BSND"):
            capacity = key.shape[1]
        elif const_expr(self.layout_kv == "PA_BSND"):
            capacity = table.shape[1] * key.shape[1]
        else:
            capacity = key.shape[0]
        return capacity

    @jit
    def _kv_length(self, key, table, cu, used, batch):
        capacity = self._kv_capacity(key, table)
        return self._sequence_length(cu, used, batch, capacity, self.layout_kv == "TND")

    @jit
    def _query_coords(self, query, cu, query_id):
        """Map a flattened Query ID to its batch and position."""
        batch = dtypes.int64(0)
        position = query_id
        if const_expr(self.layout_query == "BSND"):
            batch = query_id // query.shape[1]
            position = query_id % query.shape[1]
        else:
            # Repeated boundaries represent empty packed sequences.
            for b in range(self.batch_count):
                if query_id >= dtypes.int64(cu[b]) and query_id < dtypes.int64(cu[b + 1]):
                    batch = b
                    position = query_id - dtypes.int64(cu[b])
        return batch, position

    @jit
    def _kv_address(self, key, table, cu, batch, kv_idx):
        """Map one logical KV position to a physical row."""
        token = kv_idx
        if const_expr(self.layout_kv == "BSND"):
            token = batch * key.shape[1] + kv_idx
        elif const_expr(self.layout_kv == "TND"):
            token = dtypes.int64(cu[batch]) + kv_idx
        else:
            page = dtypes.int64(table[batch, kv_idx // key.shape[1]])
            token = page * key.shape[1] + kv_idx % key.shape[1]
        return token

    @jit
    def _prepare_sparse_query(self, query, key, table, cu_q, cu_kv,
                              used_q, used_kv, query_id):
        """Resolve this Query's batch, visible KV limit, and sparse scan bound."""
        batch, position = self._query_coords(query, cu_q, query_id)
        q_length = self._q_length(query, cu_q, used_q, batch)
        kv_length = self._kv_length(key, table, cu_kv, used_kv, batch)
        kv_end = kv_length
        if const_expr(self.sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            kv_end = kv_length - q_length + position + 1
        is_query_active = 1 if position < q_length and kv_end >= 0 else 0
        if const_expr(self.sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            if kv_end == 0:
                is_query_active = 0
        sparse_end = (
            min(self.sparse_count, kv_end) if is_query_active != 0 else 0
        )
        return batch, kv_length, kv_end, sparse_end, is_query_active

    @jit
    def _init_query_state(self, sinks, head_offset, count, state_slot):
        """Initialize a Query's softmax state and optional sink."""
        self._cross_init_state(state_slot, count)
        if const_expr(self.use_sinks):
            # A negative count marks an inactive Query: do not load its sink.
            if count >= 0:
                # Limit sink loads to valid rows in a partial head group.
                for part in tuple(range(self.head_chunks)):
                    for sub in (0, 1):
                        rows = min(
                            AIV_ROW_CAPACITY,
                            max(0, self.head_count - part * TILE_M - sub * AIV_ROW_CAPACITY),
                        )
                        if const_expr(rows > 0):
                            row_start = part * TILE_M + sub * AIV_ROW_CAPACITY
                            if head_offset == part * TILE_M and self.subblock_idx == sub:
                                source = tile_slice(
                                    sinks[row_start:row_start + rows, None],
                                    (rows, 1),
                                    (0, 0),
                                )
                                mem_copy(
                                    reinterpret(self.sink_state, shape=(rows, 1)),
                                    source,
                                )
                                mem_copy(
                                    reinterpret(
                                        self.vector.sm_max_tb[state_slot],
                                        shape=(rows, 1),
                                    ),
                                    reinterpret(self.sink_state, shape=(rows, 1)),
                                )
                self._fill_softmax_sum_one(state_slot)

    @jit
    def _emit_stats(self, maxima, sums, query_id, head_offset, state_slot):
        if const_expr(self.return_lse):
            mb = tile_slice(maxima[query_id, None, None], (maxima.shape[1], 1), (0, 0))
            sb = tile_slice(sums[query_id, None, None], (sums.shape[1], 1), (0, 0))
            self._store_softmax_state(
                tile_slice(mb, (TILE_M, 1), (head_offset // TILE_M, 0)),
                tile_slice(sb, (TILE_M, 1), (head_offset // TILE_M, 0)), state_slot)

    @jit
    def _write_empty_query(self, output, query_id, head_offset):
        zero_row = Channel(MemLoc.UB, (1, DIM_NOPE), self.dtype_16, depth=1).produce()
        for row in range(AIV_ROW_CAPACITY):
            head = head_offset + self.subblock_idx * AIV_ROW_CAPACITY + row
            if head < self.head_count:
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(128, elem_bits=16)
                    bits = rr.vdups(0, dtypes.uint16, mask=mask)
                    value = rr.vreinterpret(bits, self.dtype_16)
                    for col in tuple(range(0, DIM_NOPE, 128)):
                        rr.vstore(zero_row, col, value, mask)
                mem_copy(self._output_tile(output, query_id, head, 1), zero_row)

    @jit
    def _stage_softmax(self, first, valid_n, slot, state_slot, scale):
        """Update online softmax and stage unnormalized weights for PV."""
        qk = self.qk_ub.consume()
        if const_expr(not self.use_sinks):
            if first != 0:
                self.vector.softmax_first(
                    qk, valid_n, scale, state_slot
                )
            else:
                self.vector.softmax_rest(
                    qk, valid_n, scale, state_slot, slot
                )
        else:
            self.vector.softmax_rest(
                qk, valid_n, scale, state_slot, slot
            )
        self._store_p()

    @jit
    def _stage_output(self, first, last, valid_n, slot, query_id,
                       head_offset, output, maxima, sums):
        """Accumulate or finalize one PV tile, then close the Query if needed."""
        state_slot = query_id % SOFTMAX_STATE_SLOTS
        dst = self._output_tile(output, query_id, 0, self.head_count)
        if valid_n > 0:
            for half in range_constexpr(2):
                pv = self.pv_halves[half].consume()
                self.vector.output_half(pv, self.vector.sm_sum_tb[state_slot],
                    self.vector.sm_exp_tb[slot], first, last, half,
                    dst, head_offset)
        if last != 0:
            if valid_n == 0:
                if first != 0:
                    self._write_empty_query(output, query_id, head_offset)
            self._emit_stats(maxima, sums, query_id, head_offset, state_slot)

```

