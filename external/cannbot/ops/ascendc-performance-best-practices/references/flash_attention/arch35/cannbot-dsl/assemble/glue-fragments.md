# FA 胶水：关键代码片段（拼接件，按易错等级排序）

本篇是第 5 层**胶水**的逐字拼接件，与[惯用法](glue-idioms.md)配套：惯用法讲规则，本篇给可直接拼装的代码与拼接点。

> 定位：把已验证（逐位一致）的实现拆成**可拼接的片段**。每段给出：拼接点（接在哪、前置依赖）、逐字代码、易错点（含三轮盲测实测翻车模式）。
> 使用方法：baseline 保留其数值路径（softmax/gather 内核），按 F1→F6→F4→F5→F7→F2/F3 顺序拼接结构骨架；每拼一段跑一次逐位门禁 + Task Duration。
> 完整参考实现仍在 [整类投放件 M0-M4](index.md)（逐模块独立文件，出处行号见各模块标注）（本文片段均摘自它，行号一致）；表达层规则见 [实施惯用法](glue-idioms.md)。

**易错等级总表**（★ 越多越易错，来自三轮盲测）：

| 片段 | 内容 | 易错度 | 典型翻车 |
|---|---|---|---|
| F1 | 缓冲/通道声明区 | ★★★ | L0 无显式视图→拆块拷贝布局被推导改写（数值错） |
| F2 | AIC 主循环（tick 编排+handle 传递） | ★★★★★ | 消费序错→死锁/错配；handle 少 carry 一 tick→PV 读错代 |
| F3 | AIV 主循环（I/G/S/U 延迟线） | ★★★★★ | tap 深度错→读写竞速；gather 分半夹 softmax 的位置错 |
| F4 | compute_qk（L0B 双半装配） | ★★★★ | NZ offset 换算错→静默错位 |
| F5 | compute_pv（分区 fixpipe 落 acc/pv 半区） | ★★★★+ | reinterpret 后 partition→静默误译（行0对/其余偏） |
| F6 | store_score / store_p（跨核交接） | ★★ | unit_flag 链尾忘发 3 |
| F7 | output_half / finish_half（O 原地三态） | ★★★★ | 所有权 ack 缺失→下 tick 阻塞；版本链断裂 |

---

## F1 缓冲/通道声明区（★★★）

拼接点：`_SparseMatmul.__init__` 与 attention 主类的 `__init__`。前置：常量块（TILE_M/TILE_S2/MM_FEATURE_CHUNK/GATHER_CHUNK_ROWS/PIPELINE_SLOT_COUNT=3/SOFTMAX_STATE_SLOTS=2/PIPELINE_STAGE_COUNT=4/PIPELINE_DRAIN_TICKS=3）。

```python
# ---- _SparseMatmul.__init__：L1 双环 + Q 双代 + L0 显式视图 ----
# Two 64-row KV rings, each depth 3. Vector writes a future tile
# while Cube reads the current one. Cube consumes each slot once,
# then carries the same handle from QK to the following PV.
self.right_lower = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                           depth=PIPELINE_SLOT_COUNT, kind=ChannelKind.CrossCore)
self.right_upper = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                           depth=PIPELINE_SLOT_COUNT, kind=ChannelKind.CrossCore)
self.q_lower = [Buffer(MemLoc.L1, (group_size // 2, width), dtype_16,
                       data_format="nz") for width in (384, 384, 128)]
self.q_upper = [Buffer(MemLoc.L1, (group_size // 2, width), dtype_16,
                       data_format="nz") for width in (384, 384, 128)]
# Four operand slots; independent QK and per-feature PV accumulators.
# ⚠️ NZ physical_layout 的 canonical 维序是 (N1, M1, M0, N0)：前两维是"列块数, 行块数"，
#    physical_stride 首维 = 行数×16（0.7.0 包校验：`nd2nz trailing physical shape must match
#    the canonical (N1, M1, M0, N0) layout`）。l0a/l0b 两行都按此序，勿凭直觉交换。
self.l0a = Channel(
    MemLoc.L0A, (TILE_M, TILE_S2), dtype_16, depth=4,
    view=TensorViewSpec((TILE_M, TILE_S2), dtype_16,
        layout=ViewLayout("nz", physical_shape=(TILE_S2 // 16, TILE_M // 16, 16, 16),
            physical_stride=(TILE_M * 16, 256, 16, 1)))
)
self.l0b = Channel(
    MemLoc.L0B, (TILE_S2, MM_FEATURE_CHUNK), dtype_16, depth=4,
    view=TensorViewSpec((TILE_S2, MM_FEATURE_CHUNK), dtype_16,
        layout=ViewLayout("nz", physical_shape=(MM_FEATURE_CHUNK // 16, TILE_S2 // 16, 16, 16),
            physical_stride=(TILE_S2 * 16, 256, 16, 1)))
)
self.qk_l0c = Channel(MemLoc.L0C, (TILE_M, TILE_S2), dtypes.float32, depth=1)
self.pv_l0c = [Buffer(MemLoc.L0C, (TILE_M, MM_FEATURE_CHUNK), dtypes.float32)
               for _ in range(DIM_NOPE // MM_FEATURE_CHUNK)]
```

```python
# ---- 主类 __init__：跨核接力棒 + AIV 侧暂存 ----
self.qk_ub = Channel(MemLoc.UB, (self.vec_m, TILE_S2), dtypes.float32,
                     depth=2, kind=ChannelKind.CrossCore)
self.pv_halves = [Channel(MemLoc.UB, (self.vec_m * 2, 256), dtypes.float32,
                          depth=1, kind=ChannelKind.CrossCore) for _ in range(2)]
# Both AIV partitions form one P tile for the AIC.
self.p_channel = Channel(MemLoc.L1, (group_size, TILE_S2), dtype_16,
                         depth=1, kind=ChannelKind.CrossCore)
# Ping-pong sparse loads on Vec0: depth=2 lets the gather DMA (mte2) of
# the next chunk overlap the nd2nz (vec) of the current chunk.
self.stage0 = [Buffer(MemLoc.UB, (GATHER_CHUNK_ROWS, self.qk_dim), dtype_16)
               for _ in range(2)]
self.nz_staging = [Buffer(MemLoc.UB, (GATHER_CHUNK_ROWS, DIM_NOPE), dtype_16,
                          data_format="nz",
                          n1_pad=NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT)
                   for _ in range(2)]
# Carry each consumed KV L1 handle from the QK stage to the PV stage.
self._kv_handles = [None] * PIPELINE_SLOT_COUNT
# Load indices one Vector tick ahead, so gather does not wait for
# the preceding softmax/update before issuing its GM reads.
self.idx_buf = [Buffer(MemLoc.UB, (1, TILE_S2), dtypes.int32) for _ in range(2)]
```

易错点：
1. **l0a/l0b 必须带显式 `TensorViewSpec`**——省略时拆块拷贝布局被编译器推导改写（派生布局默认也是 NZ-canonical），数值错位；且 physical_shape 必须按 **(N1, M1, M0, N0) canonical 序**（见代码内注释，盲测第 4 轮 A 段实测 l0a 序写反即编译失败）。
2. **显式 pitch ⇔ matmul extent 必须全静态**：声明显式视图后，`'cannir.matmul' op explicit L0 physical pitch must be statically compatible with the matmul M/N extent`——所以 F4/F5 里 `valid_n = TILE_S2`（保距）不是优化选择而是**硬前置**；PV 的 N 也必须静态（per-chunk L0C buffer）。
3. Q 双代是**普通 Buffer 列表**（L1，NZ），不是 Channel；跨核交接才用 Channel。错误形态清单见惯用法 §3。
4. `pv_halves` 形状是 `(vec_m*2, 256)`=64 行：rows 0-31 留作 acc 区、rows 32-63 是进料区——**声明时就决定**，不是使用时临时切。
5. L1 台账自检：双环 384KB + Q 112KB + p_channel 16KB = 512KB 贴顶；多声明任何一个槽即超限（表现为分配期错误）。

## F6 store_score / store_p（★★，先拼，被 F4/F5/F7 依赖）

```python
def store_score(self, qk_ub, valid_n):
    """Split L0C score rows across AIVs while preserving TILE_S2 pitch."""
    score = _bounded_local(
        self.qk_l0c.consume(), (self.group_size, TILE_S2), (self.group_size, TILE_S2),
    )
    mem_copy(qk_ub.produce(), score.partition(dim=0), options=copy_options(unit_flag=3))

def store_p(self, p_l1_ch):
    """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
    mem_copy(
        p_l1_ch.partition(dim=0, part_id=self.subblock_idx),
        self.p_ub.consume(),
    )
```

易错点：score fixpipe 是链尾 → `unit_flag=3`；`partition(dim=0)` 直接施加在 L0C/L1 通道张量上（硬件按行劈给两个 AIV / 按 subblock 取半），**先 reinterpret 窗口再 partition 会静默误译**。

## F4 compute_qk（★★★★）

拼接点：`_SparseMatmul`。前置：F1 声明、`load_q_wide/load_q_tail`（普通 Buffer 数组写法）。

```python
@jit
def compute_qk(self, right_lower, right_upper, valid_n, q_slot):
    """Accumulate eight 64-feature slices into one score tile."""
    valid_n = TILE_S2
    m = self.group_size
    score = _bounded_local(
        self.qk_l0c.produce(), (m, valid_n), (m, TILE_S2)
    )
    set_mmad_direction("m")
    # L1 retains the TILE_S2 pitch when valid_n is smaller.
    for feature in range_constexpr(0, DIM_NOPE, MM_FEATURE_CHUNK):
        width = MM_FEATURE_CHUNK
        a = reinterpret(self.l0a.produce(), shape=(m, width))
        b = _bounded_local(self.l0b.produce(), (valid_n, width), (TILE_S2, width))
        for h in range_constexpr(2):
            q_slots = self.q_lower if h == 0 else self.q_upper
            if const_expr(feature < 384):
                q = reinterpret(q_slots[:2][q_slot], shape=(m // 2, width),
                    offset=(m // 2) * feature * BYTES_PER_16BIT_ELEMENT)
            else:
                q = reinterpret(q_slots[2], shape=(m // 2, width),
                    offset=(m // 2) * (feature - 384) * BYTES_PER_16BIT_ELEMENT)
            qa = reinterpret(a, shape=(m // 2, width),
                    offset=h * (m // 2) * 16 * BYTES_PER_16BIT_ELEMENT)
            mem_copy(qa, q)
        for half in range_constexpr(2):
            source = right_lower if half == 0 else right_upper
            k = reinterpret(source, shape=(TILE_S2 // 2, width),
                offset=(TILE_S2 // 2) * feature * BYTES_PER_16BIT_ELEMENT)
            dst = reinterpret(b, shape=(TILE_S2 // 2, width),
                offset=half * (TILE_S2 // 2) * 16 * BYTES_PER_16BIT_ELEMENT)
            mem_copy(dst, k)
        matmul(score, a, b, init=(feature == 0), unit_flag=3 if feature + MM_FEATURE_CHUNK == DIM_NOPE else 2)
    if const_expr(self.has_rope):
        ...  # rope 尾块：同构，q_rope_l1/k_rope_l1 各一次 matmul(init=False)，无 unit_flag 链
```

易错点：
1. L0B 双半装配的 dst offset：`half * (TILE_S2//2) * 16 * BYTES_PER_16BIT_ELEMENT`——这是"半帧=64 行 × 16 元素行块 × 2B"的物理网格字节偏移，**整式照抄**。
2. `init=(feature == 0)` 只清零首 chunk；链中 `unit_flag=2`、链尾 `3`（惯用法 §6）。
3. 宽/尾 Q 段分派：`feature < 384` 走双代宽段（`q_slot`），否则走尾段（固定下标 2）。
4. `valid_n` 固定为 TILE_S2（L1 保距），有效行由 V 侧清零+mask 兜底。

## F5 compute_pv（★★★★+，静默误译最高发）

```python
@jit
def compute_pv(self, right_lower, right_upper, p_l1, valid_n, result_lo, result_hi, first):
    """[G,valid_n] @ [valid_n,512], retaining static NZ storage pitch."""
    valid_n = TILE_S2
    set_mmad_direction("n")
    m = self.group_size
    p = _bounded_local(p_l1, (m, valid_n), (m, TILE_S2))
    a = _bounded_local(self.l0a.produce(), (m, valid_n), (m, TILE_S2))
    mem_copy(a, p)
    for feature in range_constexpr(0, DIM_NOPE, MM_FEATURE_CHUNK):
        b = self.l0b.produce()
        for half in range_constexpr(2):
            source = right_lower if half == 0 else right_upper
            v = reinterpret(source, shape=(TILE_S2 // 2, MM_FEATURE_CHUNK),
                offset=(TILE_S2 // 2) * feature * BYTES_PER_16BIT_ELEMENT)
            dst_b = reinterpret(b, shape=(MM_FEATURE_CHUNK, TILE_S2 // 2),
                offset=half * MM_FEATURE_CHUNK * (TILE_S2 // 2) * BYTES_PER_16BIT_ELEMENT, data_format="nz")
            mem_copy(dst_b, v.transpose(0, 1))
        dst = self.pv_l0c[feature // MM_FEATURE_CHUNK]
        matmul(dst, a, reinterpret(b, shape=(MM_FEATURE_CHUNK, TILE_S2), data_format="nz"), init=True)
        root = result_lo if feature < DIM_NOPE // 2 else result_hi
        if first != 0:
            target = tile_slice(root, (m // 2, DIM_NOPE // 2), (0, 0))
            mem_copy(tile_slice(target, (m // 2, MM_FEATURE_CHUNK),
                                (0, (feature // MM_FEATURE_CHUNK) % 4)),
                     dst.partition(dim=0))
        else:
            target = tile_slice(root, (m // 2, DIM_NOPE // 2), (1, 0))
            mem_copy(tile_slice(target, (m // 2, MM_FEATURE_CHUNK),
                                (0, (feature // MM_FEATURE_CHUNK) % 4)),
                     dst.partition(dim=0))
```

易错点（**全部来自实测翻车**）：
1. **分区 fixpipe 骨架**：`dst.partition(dim=0)` 直接施加在 L0C buffer 上；目的侧用 `tile_slice(root, ...)` 在通道根张量上选半区——`(0,0)`=acc 区（首 tile 直写累加器）、`(1,0)`=进料区。**先对 L0C 做 reinterpret 窗口再 partition → 静默误译：每 AIV 行 0 正确、行 1-31 同量级小偏差**（盲测第 2 轮 G1）。
2. 列位置：`(feature // MM_FEATURE_CHUNK) % 4`——256 宽半区内第 chunk%4 个 64 列块。
3. V 是**转置**进 B（`v.transpose(0,1)`，mmad direction n）。
4. 每个 D-chunk 的 L0C `init=True`（独立清零，不是跨 chunk 累加）。

## F7 output_half / finish_half（★★★★）

拼接点：softmax 向量类。前置：F5 的 pv_halves 双半区约定。

```python
@jit
def output_half(self, result, sm_sum_buf, sm_exp_buf, first, last, half,
                output_heads, output_head):
    width = self.tile_d // 2
    acc = tile_slice(result, (self.tile_vec_m, width), (0, 0))
    pv = tile_slice(result, (self.tile_vec_m, width), (1, 0))
    if first != 0:
        # First PV lands directly in the persistent accumulator; this access
        # acknowledges ownership without copying the entire tile in VECTOR.
        with vf(mode="simd"):
            lane, _ = rr.update_mask(1, elem_bits=32)
            value = rr.vload(acc, 0)
            rr.vstore(acc, 0, value, lane)
        if last != 0:
            self.finish_half(acc, acc, pv, sm_sum_buf, None, half, output_heads, output_head)
    elif last != 0:
        self.finish_half(pv, acc, pv, sm_sum_buf, sm_exp_buf, half, output_heads, output_head)
    else:
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                for col in range_constexpr(0, width, 64):
                    off = row * width + col
                    pre = rr.vload(acc, off)
                    cur = rr.vload(pv, off)
                    val = rr.vmadd(pre, exp_b, cur, mask=full)
                    rr.vstore(acc, off, val, full)

@jit
def finish_half(self, cur_buf, acc, out_buf, sm_sum_buf, sm_exp_buf, half,
                output_heads, output_head):
    width = self.tile_d // 2
    out = reinterpret(out_buf, dtype=self.dtype_16, shape=(self.tile_vec_m, width * 2))
    with vf(mode="simd"):
        full, _ = rr.update_mask(64, elem_bits=32)
        zero = rr.vdups(0.0, dtypes.float32, mask=full)
        one = rr.vdups(1.0, dtypes.float32, mask=full)
        for row in range(self.tile_vec_m):
            sum_b = rr.vload_broadcast(sm_sum_buf, row)
            is_zero = rr.veqs(sum_b, 0.0, mask=full)
            safe_sum = rr.vselect(one, sum_b, cond_mask=is_zero)
            if const_expr(sm_exp_buf is not None):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
            for col in range_constexpr(0, width, 64):
                off = row * width + col
                cur = rr.vload(cur_buf, off)
                if const_expr(sm_exp_buf is not None):
                    pre = rr.vload(acc, off)
                    cur = rr.vmadd(pre, exp_b, cur, mask=full)
                val = rr.vdiv(cur, safe_sum, mask=full)
                val = rr.vselect(zero, val, cond_mask=is_zero)
                packed = rr.vcast(val, self.dtype_16, mask=full)
                rr.vstore_pack(out, row * width * 2 + col,
                               packed, full, pack_mode="b32_to_b16")
    rows = min(self.tile_vec_m, max(0, output_heads.shape[0] - output_head
                                   - self.subblock_idx * self.tile_vec_m))
    row_start = output_head + self.subblock_idx * self.tile_vec_m
    if rows > 0:
        mem_copy(output_heads[row_start:row_start + rows, half * width:(half + 1) * width],
                 out[0:rows, 0:width])
```

易错点：
1. **首 tile 只做 1-lane 读写**（`update_mask(1)`）确认所有权——写整 tile 会触发跨核版本校验；不 ack 则 producer 下 tick 阻塞。
2. 三态分支次序：`first` 优先组织（编译器控制路径校验要求）；首 tile 落 acc 区是 **AIC fixpipe 直写**的（F5），AIV 不搬运。
3. 中间 tile：`acc = acc·sm_exp + cur`（vmadd，原地写 rows 0-31）；cur 读 rows 32-63。
4. 末 tile 归一化在 **pv 区原地**：`(acc·se + cur)/l`，sum=0 安全除 1，`vstore_pack` b32→b16，再 mem_copy 本 AIV 的 32 行到 GM 的 `half*width` 列偏移处。
5. sm_exp_buf=None 的分支（首 tile 即末 tile）不能读 exp——`const_expr` 编译期剪枝。

## F2 AIC 主循环（★★★★★）

拼接点：主类 `_cube_program`。前置：F1/F4/F5/F6、`_prepare_sparse_query`/`_find_valid_end`/`_query_tile`（baseline 已有）。常量：`PIPELINE_STAGE_COUNT=4`、`PIPELINE_DRAIN_TICKS=3`、`EMPTY_QUERY_ID=-1`。

```python
@jit
def _cube_program(self, output, query, key, indices, query_rope, key_rope,
                  sinks, maxima, sums, workspace, table, cu_q, cu_kv,
                  used_q, used_kv, scale):
    # Map this core to a query group and a head tile.
    core_group = self.block_idx // self.head_chunks
    head_offset = self.block_idx % self.head_chunks * TILE_M
    first_query = core_group * self.queries_per_group
    query_end = min(first_query + self.queries_per_group, self.total_queries)
    current_query = first_query
    query_count = query_end - first_query
    sparse_idx = dtypes.int64(0)
    is_first_block = dtypes.int64(1)
    batch = dtypes.int64(0)
    sparse_end = dtypes.int64(0)
    is_query_active = dtypes.int64(0)
    # Nonempty tasks cover at most one KV tile; EMPTY_QUERY_ID marks a bubble.
    task_pipeline = DelayLineGroup(
        PIPELINE_STAGE_COUNT,
        "query_id", "valid_kv_count", "is_first_block",
        "is_last_block", "is_query_active",
    )
    max_gather_tasks = query_count * ((self.sparse_count + TILE_S2 - 1) // TILE_S2)
    if query_count > 0:
        self.matmul.load_q_wide(self._query_tile(query, first_query, head_offset), first_query % 2)
    kv_lo = self.matmul.right_lower.consume()
    kv_hi = self.matmul.right_upper.consume()
    pv_lo, pv_hi = kv_lo, kv_hi
    for tick in range(max_gather_tasks + PIPELINE_DRAIN_TICKS):
        next_lo, next_hi = pv_lo, pv_hi
        # Stage 0: register this tick's gather task (no DMA on the cube side).
        if current_query < query_end:
            if is_first_block != 0:
                sparse_idx = 0
                is_query_active = 0
                sparse_end = 0
                (batch, unused_kv_length, unused_kv_end, planned_sparse_end,
                 is_query_active) = self._prepare_sparse_query(
                    query, key, table, cu_q, cu_kv, used_q, used_kv, current_query)
                sparse_end = self._find_valid_end(indices, current_query, planned_sparse_end)
            valid_kv_count = dtypes.int64(0)
            if sparse_idx < sparse_end:
                valid_kv_count = min(TILE_S2, sparse_end - sparse_idx)
                sparse_idx += valid_kv_count
            is_last_block = 1 if sparse_idx >= sparse_end else 0
            task_pipeline.push(
                query_id=current_query, valid_kv_count=valid_kv_count,
                is_first_block=is_first_block, is_last_block=is_last_block,
                is_query_active=is_query_active,
            )
            is_first_block = 0
            if is_last_block != 0:
                current_query += 1
                is_first_block = 1
        else:
            task_pipeline.push(query_id=EMPTY_QUERY_ID, valid_kv_count=0,
                               is_first_block=0, is_last_block=0, is_query_active=0)
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
                    self.matmul.load_q_wide(
                        self._query_tile(query, qk_task.query_id + 1, head_offset),
                        (qk_task.query_id + 1) % 2)
        # Stage 2: PV for the tile gathered two ticks ago.
        if tick >= 2:
            compute_task = task_pipeline.tap(2)
            if compute_task.query_id >= 0 and compute_task.valid_kv_count > 0:
                self._stage_pv(pv_lo, pv_hi, compute_task.valid_kv_count, compute_task.is_first_block)
        # Carry the QK handles one tick so PV reads the same published slot.
        pv_lo, pv_hi = next_lo, next_hi
        task_pipeline.advance()
```

易错点（★★★★★，盲测从未独立写对）：
1. **handle 传递不变量**：`pv_lo/pv_hi` 必须是 **QK 消费过的同一槽 handle**（一次 produce 喂两个读者）。顺序：tick 开头 `next = pv` → QK 成功后 `next = kv（刚消费的）+ 预 consume 下一槽` → 循环尾 `pv = next`。任何"PV 重新 consume"都会把环读穿。
2. **空 query 气泡**：`EMPTY_QUERY_ID` 任务也要 push/tap/advance；QK 跳过但 **Q 宽段预取不能跳**（注释原文：empty queries still prefetch——否则下一个非空 query 会依赖被跳过的生产者）。
3. `tap(1)`/`tap(2)` 的深度差 = QK/PV 相位差；`DelayLineGroup` 深度 = `PIPELINE_STAGE_COUNT`（4）。
4. `load_q_tail` 只在 `is_first_block` 且任务非空时发（每 query 一次）。
5. 动态分支内的 consume/produce 必须无条件配对——只在 `valid_kv_count > 0` 时 consume 会导致计数漂移（consume 在 QK 成功分支内、预 consume 下一槽也在同分支，配对关系见骨架原文）。

## F3 AIV 主循环（★★★★★）

拼接点：主类 `_vector_program`。关键差异 vs AIC：延迟线深度 **+1**（5），gather 分两半夹住 softmax。

```python
@jit
def _vector_program(self, output, query, key, indices, query_rope, key_rope,
                    sinks, maxima, sums, workspace, table, cu_q, cu_kv,
                    used_q, used_kv, scale):
    core_group = self.block_idx // self.head_chunks
    head_offset = self.block_idx % self.head_chunks * TILE_M
    first_query = core_group * self.queries_per_group
    query_end = min(first_query + self.queries_per_group, self.total_queries)
    current_query = first_query
    query_count = query_end - first_query
    sparse_idx = dtypes.int64(0)
    is_first_block = dtypes.int64(1)
    batch = dtypes.int64(0)
    sparse_end = dtypes.int64(0)
    is_query_active = dtypes.int64(0)
    task_pipeline = DelayLineGroup(
        PIPELINE_STAGE_COUNT + 1,                      # ← 5，比 AIC 深 1
        "query_id", "valid_kv_count", "is_first_block",
        "is_last_block", "is_query_active",
        "sparse_idx", "sparse_end", "batch",           # ← gather 需要的额外字段
    )
    max_gather_tasks = query_count * ((self.sparse_count + TILE_S2 - 1) // TILE_S2)
    kv_lo = self.matmul.right_lower.produce()
    kv_hi = self.matmul.right_upper.produce()
    for tick in range(max_gather_tasks + PIPELINE_DRAIN_TICKS + 1):   # ← +1 排空
        gather_n = dtypes.int64(0)
        gather_query = current_query
        gather_idx = sparse_idx
        gather_end = sparse_end
        gather_batch = batch
        # Stage 0: register the task (identical to cube side except extra fields).
        if current_query < query_end:
            if is_first_block != 0:
                sparse_idx = 0
                is_query_active = 0
                sparse_end = 0
                (batch, unused_kv_length, unused_kv_end, planned_sparse_end,
                 is_query_active) = self._prepare_sparse_query(
                    query, key, table, cu_q, cu_kv, used_q, used_kv, current_query)
                sparse_end = self._find_valid_end(indices, current_query, planned_sparse_end)
            valid_kv_count = dtypes.int64(0)
            if sparse_idx < sparse_end:
                valid_kv_count = min(TILE_S2, sparse_end - sparse_idx)
                gather_n = valid_kv_count
                gather_query = current_query
                gather_idx = sparse_idx
                gather_end = sparse_end
                gather_batch = batch
                sparse_idx += valid_kv_count
            is_last_block = 1 if sparse_idx >= sparse_end else 0
            task_pipeline.push(
                query_id=current_query, valid_kv_count=valid_kv_count,
                is_first_block=is_first_block, is_last_block=is_last_block,
                is_query_active=is_query_active,
                sparse_idx=gather_idx, sparse_end=gather_end, batch=gather_batch,
            )
            is_first_block = 0
            if is_last_block != 0:
                current_query += 1
                is_first_block = 1
        else:
            task_pipeline.push(query_id=EMPTY_QUERY_ID, valid_kv_count=0,
                               is_first_block=0, is_last_block=0, is_query_active=0,
                               sparse_idx=0, sparse_end=0, batch=0)
        # I(t) is independent of G(t-1): load indices for THIS tick's tile first.
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
        # G(t-1) 前半：chunk 0-1（DMA 与上一 tick 的 softmax 重叠）
        if gather_n > 0:
            self._gather_chunks(kv_lo, kv_hi, key.view(-1), 0, 2, (tick - 1) % 2, gather_n)
        # S(t-3)：在线 softmax，吃 qk_ub，写 p_ub → p_channel
        if tick >= 3:
            compute_task = task_pipeline.tap(3)
            if compute_task.query_id >= 0:
                state_slot = compute_task.query_id % SOFTMAX_STATE_SLOTS
                if compute_task.is_first_block != 0:
                    self._init_query_state(
                        sinks, head_offset,
                        (compute_task.valid_kv_count
                         if compute_task.is_query_active != 0 else -1),
                        state_slot)
                if compute_task.valid_kv_count > 0:
                    self._stage_softmax(
                        compute_task.is_first_block, compute_task.valid_kv_count,
                        (tick - 3) % SOFTMAX_STATE_SLOTS, state_slot, scale)
        # G(t-1) 后半：chunk 2-3，然后发布本 tile 并预取下一槽
        if gather_n > 0:
            self._gather_chunks(kv_lo, kv_hi, key.view(-1), 2, 4, (tick - 1) % 2, gather_n)
            kv_lo = self.matmul.right_lower.produce()
            kv_hi = self.matmul.right_upper.produce()
        # U(t-4)：O 原地更新/收尾
        if tick >= 4:
            output_task = task_pipeline.tap(4)
            if output_task.query_id >= 0:
                self._stage_output(
                    output_task.is_first_block, output_task.is_last_block,
                    output_task.valid_kv_count, (tick - 4) % SOFTMAX_STATE_SLOTS,
                    output_task.query_id, head_offset, output, maxima, sums)
        task_pipeline.advance()
```

易错点（★★★★★）：
1. **gather 分半夹 softmax**：`_gather_chunks(0,2)` 发出前半 DMA → softmax（吃满向量单元）→ `_gather_chunks(2,4)` 发后半 + `produce()` 下一槽。前半/后半拆在哪决定 MTE2 能否与 VF 重叠——顺序错则 AIV 串行化。
2. `produce()` 时机：在**后半 gather 完成后**才发布本 tile、预取下一槽 handle（`kv_lo/kv_hi` 是环形游标）。
3. `tap` 深度语义：`tap(1)`=G、`tap(3)`=S、`tap(4)`=U（I 在 tick 内联）；延迟线深度 = `PIPELINE_STAGE_COUNT + 1` = 5。
4. idx 槽选择：`_gather_setup(tick % 2, ...)`（本 tick 装载）、`_gather_chunks(..., (tick - 1) % 2, ...)`（上一 tick 装载的槽）——错位一格就是读旧索引。
5. 排空：`+PIPELINE_DRAIN_TICKS + 1`（AIC 是 `+PIPELINE_DRAIN_TICKS`，AIV 多 1 拍）。
6. `_stage_softmax`/`_stage_output` 内部的 qk_ub.consume/pv_halves.consume 与 AIC 侧 produce 必须一一配对（同 tick 交接的锁相点，见流水篇 §2）。

## F9 gather/nd2nz 细节（★★★，"每 AIV 写自己半环"的可验证形态）

盲测第 4 轮 A 段实证：`subblock_idx==0 推 kv_lo 否则 kv_hi` 一句话在完整 kernel 上下文会被同步校验器拒绝（`cross-core access has no provable ownership boundary` / `cross-core empty handoff is not a completed local rendezvous`），但同样模式在孤立微核中通过。**可验证的合法结构 = 以下三点同时成立**：

1. `kv_lo/kv_hi` 是在 **tick 循环顶层**取得的 `produce()` 句柄（不在任何动态子区域里取），作为函数实参传入；
2. 条件推入是**语句级 if/else**（`if self.subblock_idx == 0: ... else: ...`），不是 `with vf` 内的写、也不是两个句柄的 tensor_select；
3. 每次推入都是无条件完成的 mem_copy（16 行 chunk 粒度、tile_slice 定位），DMA 前置批与 VF/推入分两段循环。

完整骨架（逐字，与 F3 的调用点配合）：

```python
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
```

- 尾行：`local_rows` 不足一个 chunk 时先 vstore 清零整 chunk 再 gather 有效行（`full=False` 路径）。
- `_nd2nz_chunk(ready, nz)`：行循环内按 256 列两 bank 交织（`fb` 与 `fb+128` 两次 vload → 两次 `vstore_strided`），`nz_m = (row // m0) * s_m1 + (row % m0) * s_m0`。
- 两 AIV 各写半环：subblock 0 写 lower 环、subblock 1 写 upper 环，互不重叠——这是"每 AIV 单写点"的所有权边界。

## 拼接顺序与门禁

```
F1 声明 → 冒烟（应与 baseline 数值同路径，仅声明变化）
F6 交接 → F4 QK → 逐位 + Task Duration
F5 PV → 逐位（重点盯"行0对/其余偏"静默误译）→ F7 O 原地 → 逐位
F2/F3 主循环替换 → 逐位 + Task Duration
F9 按 baseline 差异最小改
```

每步 `accuracy.py` 必须逐位 True；性能只认 `profile.sh` 的 Task Duration。历史阶梯数值与前置条件见惯用法 §8。

## 替换到同类算子：片段怎么用

1. **先试整类投放**（[积木模块](index.md)），只有当目标算子与 SFA 差异集中在局部（如 softmax 变体、gather 粒度）时才退回片段级拼装——片段拼装的集成面远大于整类（三轮盲测结论）。
2. 按 F1→F6→F4→F5→F7→F2/F3 顺序拼接；每拼一段跑一次逐位门禁 + Task Duration，不连续拼多段。
3. 替换后易错点随形状变化：NZ offset 换算、tap 深度、槽位算术都要按[容量方程](../optimize/3-capacity-equations.md)的新解重算，逐字复制代码而不改数是常见翻车原因。
