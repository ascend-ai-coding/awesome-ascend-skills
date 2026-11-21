### Matmul 访存布局优化（L2 命中率优化）

**问题**：大 shape Matmul 场景下（M、N、K 全部较大），即使 TileShape 配置了推荐值，固定的分核布局可能导致 L2 命中率偏低、MTE2 带宽利用率不足。

**原理**：L2 命中率由单轮次分核数 mDim、nDim 和 mL1、nL1 共同决定：

```

l2_hit_ratio = 1 - (1/(nDim·nL1) + 1/(mDim·mL1)) / (1/nL1 + 1/mL1)

```

**最优条件**：`nDim·nL1 = mDim·mL1`，即 M、N 轴分核到 L1 的数据量相等，L2 复用最大化。

例如 mL1=128、nL1=256、24 核平台时，推荐：
- mDim=6、nDim=4 → 6×128 = 4×256 = 1024 ✅
- mDim=8、nDim=3 → 8×128 ≈ 3×256 = 768 ✅

**优化方法**：在 M、N 轴外层再套一层 loop，手动控制每轮 M 和 N 的计算范围，从而控制分核数：

```python
def create_mm_with_l2_opt(M, K, N, mL1, nL1, mDim, nDim):
    @pypto.frontend.jit(...)
    def matmul_kernel(a, b):
        pypto.set_cube_tile_shapes([mL1, mL1], [64, 256], [nL1, nL1])

        m_view = mL1 * mDim
        n_view = nL1 * nDim
        m_loop = (M + m_view - 1) // m_view
        n_loop = (N + n_view - 1) // n_view
        out = pypto.Tensor([M, N], pypto.DT_FP16)

        for m_idx in pypto.loop(m_loop, name="LOOP_m", idx_name="m_idx"):
            for n_idx in pypto.loop(n_loop, name="LOOP_n", idx_name="n_idx"):
                a_block = a[m_idx*m_view : m_idx*m_view+m_view, :]
                b_block = b[:, n_idx*n_view : n_idx*n_view+n_view]
                out_block = pypto.matmul(a_block, b_block,
                    out_dtype=pypto.DT_FP16)
                out[m_idx*m_view:m_idx*m_view+m_view,
                    n_idx*n_view:n_idx*n_view+n_view] = out_block
        return out
    return matmul_kernel

```

**调试方法**：
1. 从泳道图或气泡分析中检查 MTE2 带宽利用率（是否明显低于 HBM 带宽）
2. 根据 mL1/nL1 和芯片总核数，按最优条件计算 mDim/nDim
3. 按上述模式改写，重新采集性能数据对比

**收益**：M=N=K=6144、TileShape=[128,128],[64,256],[256,256] 场景：
- 默认分核：2.1ms，等效算力约 220 TFLOPS
- L2 优化（mDim=6,nDim=4）：1.6ms，等效算力约 290 TFLOPS（+31%）
