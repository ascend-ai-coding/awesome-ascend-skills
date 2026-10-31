> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M2：softmax 积木——online softmax 双类（源 L187-585）

> **为什么**：online softmax 的状态（max/sum/alpha）只有一拍生命周期（SM→U），槽位收紧到各两槽即可覆盖；状态多槽是浪费，去写回 fold 重做实测负收益（+0.7%）。跨 tile 驻留的 O 累加器与 PV 进料共享 pv_halves 通道的上下半区——能这样做是因为 L0C 结果按行劈给 2 个 AIV，每 AIV 每 tick 只收 32 行。
```python


class _StableOnlineSoftmaxVector:
    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        dtype_16=dtypes.float16,
        preload_num=SOFTMAX_STATE_SLOTS,
    ):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx
        self.dtype_16 = dtype_16

        self.sm_max_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_sum_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_exp_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)

        # Address NZ data with its padded physical stride.
        p_n1_pad = NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT
        self.p_ub = Channel(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=self.dtype_16,
            depth=1,
            data_format="nz",
            n1_pad=p_n1_pad,
        )


    @jit
    def softmax_first_original(self, qk_ch, actual_n, scale, m_axis_triple: int):
        """Initialize running max/sum and unnormalized weights for the first tile."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = _nz_params(p_ub)
        with vf(mode="simd"):
            # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
            rows = self.tile_vec_m
            src_row_stride = self.tile_n
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    sm_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                sm_max,
                sm_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )

    @jit
    def softmax_rest_original(
        self, qk_ch, actual_n, scale, m_axis_triple: int, tile_triple: int
    ):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max)."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = _nz_params(p_ub)
        with vf(mode="simd"):
            # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
            rows = self.tile_vec_m
            src_row_stride = self.tile_n
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask
            )
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                self.tmp_new_max,
                self.tmp_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def softmax_first(self, qk_ch, actual_n, scale, m_axis_triple: int):
        if actual_n == 128:
            self.softmax_full_row_fused(qk_ch, scale, m_axis_triple, 0, True)
        else:
            self.softmax_first_original(qk_ch, actual_n, scale, m_axis_triple)

    @jit
    def softmax_rest(self, qk_ch, actual_n, scale, m_axis_triple: int, tile_triple: int):
        if actual_n == 128:
            self.softmax_full_row_fused(qk_ch, scale, m_axis_triple, tile_triple, False)
        else:
            self.softmax_rest_original(qk_ch, actual_n, scale, m_axis_triple, tile_triple)

    @jit
    def softmax_full_row_fused(self, qk_ch, scale, state_slot, alpha_slot, first: bool):
        sm_max = self.sm_max_tb[state_slot]
        sm_sum = self.sm_sum_tb[state_slot]
        sm_exp = self.sm_exp_tb[alpha_slot]
        p_ub = self.p_ub.produce()
        m0, s_m1, s_m0, block_stride = _nz_params(p_ub)
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            b16, _ = rr.update_mask(128, elem_bits=16)
            for row in range(self.tile_vec_m):
                # Same scaled FP32 values and contiguous-half max tree as baseline.
                v0 = rr.vmuls(rr.vload(qk_ch, row * 128), scale, mask=full)
                v1 = rr.vmuls(rr.vload(qk_ch, row * 128 + 64), scale, mask=full)
                rowmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full), mask=full)
                if const_expr(first):
                    zero = rr.vdups(0.0, dtypes.float32, mask=full)
                    masked = rr.vles(rowmax, EMPTY_SOFTMAX_THRESHOLD, mask=full)
                    rowmax = rr.vselect(zero, rowmax, cond_mask=masked)
                    mx = rr.vdup(rowmax, mask=full)
                else:
                    oldmax = rr.vload_broadcast(sm_max, row)
                    mx = rr.vmax(oldmax, rr.vdup(rowmax, mask=full), mask=full)
                even, odd = rr.vdeinterleave(v0, v1)
                even = rr.vexp_sub(even, mx, mask=full)
                odd = rr.vexp_sub(odd, mx, mask=full)
                he = rr.vcast(even, self.dtype_16, mask=full, reg_layout=rr.RegLayout.ZERO)
                ho = rr.vcast(odd, self.dtype_16, mask=full, reg_layout=rr.RegLayout.ONE)
                merged = rr.vbitwise_or(he, ho, mask=b16)
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                rr.vstore_strided(p_ub, nz_off, merged, b16,
                                 block_stride=block_stride, repeat_stride=0)
                # Preserve baseline add-even-odd THEN reduce, not reference's
                # separately reduced halves. Output VDIV remains unchanged.
                row_sum = rr.vreduce_sum(rr.vadd(even, odd, mask=full), mask=full)
                if const_expr(first):
                    rr.vstore_first(sm_sum, row, row_sum)
                    rr.vstore_first(sm_max, row, mx)
                else:
                    # Keep score/exp/P fusion, but collect one max and sum
                    # per row for the original batched state update.
                    rr.vstore_first(self.tmp_new_max, row, mx)
                    rr.vstore_first(self.tmp_sum, row, row_sum)
            if const_expr(not first):
                rr.vmem_bar("vst_vld")
                rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    def store_p(self, p_l1_ch):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
        mem_copy(
            p_l1_ch.partition(dim=0, part_id=self.subblock_idx),
            self.p_ub.consume(),
        )

    @jit
    def _softmax_fold_loop(
        self,
        qk_ch,
        p_ub,
        max_buf,
        sum_dst,
        actual_n,
        rows,
        src_row_stride,
        tile_n,
        m0,
        s_m1,
        s_m0,
        block_stride,
    ):
        """Write unnormalized weight rows and their exponential sums."""
        for row in range(rows):
            full, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(tile_n, elem_bits=16)
            ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
            vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
            b16, _ = rr.update_mask(actual_n, elem_bits=16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo = self._softmax_fold_row(
                qk_ch,
                p_ub,
                base,
                nz_off,
                max_buf,
                row,
                ve_mask,
                vo_mask,
                b16,
                b16_full,
                block_stride,
            )
            rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
            rr.vstore_first(sum_dst, row, rsum)
        rr.vmem_bar("vst_vld")

    def _softmax_fold_row(
        self,
        qk_ch,
        p_ub,
        base,
        nz_off,
        max_brc_buf,
        row,
        ve_mask,
        vo_mask,
        b16,
        b16_full,
        block_stride,
    ):
        """Store unnormalized NZ weights and return the FP32 exponential halves."""
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, self.dtype_16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, self.dtype_16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        rr.vstore_strided(
            p_ub,
            nz_off,
            merged,
            b16_full,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return ve, vo

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        """Online-softmax running-state update."""
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)  # exp(old-new)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)


class _SFAOnlineSoftmaxVector(_StableOnlineSoftmaxVector):
    """Online softmax with tail-safe maxima for signed scores."""

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

    def _pass_a_row(
        self,
        qk_ch,
        scale,
        sm_max_dst,
        row,
        row_stride,
        vector_lanes,
        half0_mask,
        half1_mask,
        full_mask,
    ):
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_ch, base + vector_lanes), scale, mask=half1_mask)
        rr.vstore(qk_ch, base, v0, half0_mask)
        rr.vstore(qk_ch, base + vector_lanes, v1, half1_mask)
        neg_inf = rr.vdups(SOFTMAX_NEG_INF, dtypes.float32, mask=full_mask)
        max0 = rr.vselect(v0, neg_inf, cond_mask=half0_mask)
        max1 = rr.vselect(v1, neg_inf, cond_mask=half1_mask)
        rmax = rr.vreduce_max(rr.vmax(max0, max1, mask=full_mask), mask=full_mask)
        # Initialize an empty first tile without replacing later running maxima.
        if sm_max_dst is not self.tmp_new_max:
            zeros = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            all_masked = rr.vles(rmax, EMPTY_SOFTMAX_THRESHOLD, mask=full_mask)
            rmax = rr.vselect(zeros, rmax, cond_mask=all_masked)
        rr.vstore_first(sm_max_dst, row, rmax)


```

