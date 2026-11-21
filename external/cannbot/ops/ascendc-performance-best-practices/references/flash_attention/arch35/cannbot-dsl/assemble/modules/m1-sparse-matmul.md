> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M1：MM 积木——Cube 类 _SparseMatmul（源 L607-756）

> **为什么**：Cube 指令链必须改"真实依赖"——unit_flag/mmad 方向、D 向 64 分块、L0 四槽贴顶；纯改循环/编号（step7）实测无收益。L0B 双半装配依赖 [惯用法](../glue-idioms.md) §1 的显式 TensorViewSpec，否则拆块拷贝布局被编译器改写（数值错）。
```python
class _SparseMatmul:
    """Cube matmul stages and their L1/L0 buffers."""

    def __init__(self, group_size: int, dtype_16, has_rope: bool):
        self.group_size = group_size
        self.has_rope = has_rope

        # Two 64-row KV rings, each depth 3. Vector writes a future tile
        # while Cube reads the current one. Cube consumes each slot once,
        # then carries the same handle from QK to the following PV.
        self.right_lower = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                                   depth=PIPELINE_SLOT_COUNT, kind=ChannelKind.CrossCore)
        self.right_upper = Channel(MemLoc.L1, (TILE_S2 // 2, DIM_NOPE), dtype_16,
                                   depth=PIPELINE_SLOT_COUNT, kind=ChannelKind.CrossCore)
        self.k_rope_l1 = None
        if has_rope:
            self.k_rope_l1 = Channel(
                MemLoc.L1, (TILE_S2, DIM_ROPE), dtype_16,
                depth=1,
            ).produce()
        self.q_lower = [Buffer(MemLoc.L1, (group_size // 2, width), dtype_16,
                               data_format="nz") for width in (384, 384, 128)]
        self.q_upper = [Buffer(MemLoc.L1, (group_size // 2, width), dtype_16,
                               data_format="nz") for width in (384, 384, 128)]
        self.q_rope_l1 = None
        if has_rope:
            self.q_rope_l1 = Channel(
                MemLoc.L1, (group_size, DIM_ROPE), dtype_16,
                depth=1,
            ).produce()

        # Four operand slots; independent QK and per-feature PV accumulators.
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
        self.qk_l0c = Channel(
            MemLoc.L0C, (TILE_M, TILE_S2), dtypes.float32, depth=1
        )
        self.pv_l0c = [Buffer(MemLoc.L0C, (TILE_M, MM_FEATURE_CHUNK), dtypes.float32)
                       for _ in range(DIM_NOPE // MM_FEATURE_CHUNK)]

    @jit
    def load_q_wide(self, query_token, slot):
        mem_copy(self.q_lower[:2][slot], query_token[0:self.group_size // 2, 0:384])
        mem_copy(self.q_upper[:2][slot], query_token[self.group_size // 2:self.group_size, 0:384])

    def load_q_tail(self, query_token):
        mem_copy(self.q_lower[2], query_token[0:self.group_size // 2, 384:512])
        mem_copy(self.q_upper[2], query_token[self.group_size // 2:self.group_size, 384:512])

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
            q = reinterpret(self.q_rope_l1, shape=(m, DIM_ROPE))
            k_root = reinterpret(self.k_rope_l1, shape=(TILE_S2, DIM_ROPE))
            k = _bounded_local(
                k_root, (valid_n, DIM_ROPE), (TILE_S2, DIM_ROPE)
            )
            a = reinterpret(self.l0a.produce(), shape=(m, DIM_ROPE))
            b = _bounded_local(
                self.l0b.produce(), (valid_n, DIM_ROPE),
                (TILE_S2, DIM_ROPE)
            )
            mem_copy(a, q)
            mem_copy(b, k)
            matmul(score, a, b, init=False)

    @jit
    def compute_pv(self, right_lower, right_upper, p_l1, valid_n, result_lo, result_hi, first):
        """[G,valid_n] @ [valid_n,512], retaining static NZ storage pitch."""
        valid_n = TILE_S2
        set_mmad_direction("n")
        m = self.group_size
        p = _bounded_local(p_l1, (m, valid_n), (m, TILE_S2))
        a = _bounded_local(
            self.l0a.produce(), (m, valid_n), (m, TILE_S2)
        )
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

    def store_score(self, qk_ub, valid_n):
        """Split L0C score rows across AIVs while preserving TILE_S2 pitch."""
        score = _bounded_local(
            self.qk_l0c.consume(), (self.group_size, TILE_S2), (self.group_size, TILE_S2),
        )
        mem_copy(qk_ub.produce(), score.partition(dim=0), options=copy_options(unit_flag=3))



```

