# Layout Strategy: NDHWC for Row-Granularity Pooling

> **定位**：通用（前向 reduce 与反向 scatter-add 均适用），反向 block 语义需镜像对调（见 [backward-patterns.md](backward-patterns.md) §5）。
> **实现篇**（host permute + kernel 行地址的 AscendC 落地）在 tilelang2ascend-translator 的 Pooling 类别，
> 见 `references/pooling-patterns/references/layout-implementation.md`（下称 layout-implementation.md）。

## 核心决策

**Pooling 算子使用 NDHWC 内部布局，而非标准的 NCDHW。**

## 为什么

PyTorch/NPU 的标准卷积布局是 NCDHW（Channel-first），但 pooling 算子的计算特征是**沿空间维度滑动窗口做归约**。在 NCDHW 下，每个空间位置 (d,h,w) 的 C 个元素在 GM 中非连续（stride = D*H*W），需要一个一个加载，无法向量化。

| 布局 | 一行内存布局 | DataCopy 效率 | 向量化 |
|------|------------|-------------|--------|
| NCDHW | C 个元素跨 D*H*W stride | 需要逐元素 gather | 不支持 |
| NDHWC | W*C 个连续元素 | 单指令整行搬运 | C 维度向量化 |

在 NDHWC 下，每个 (n,d,h) 的整行 `X[n,d,h,0:W,0:C]` 在 GM 中连续存储，可以一次 `DataCopy`（`W*C` 个元素）加载到 UB，然后沿 W 维度滑动 ow 窗口做 Add(C) 累加。

## TileLang 设计表达（语义蓝图）

```python
@T.prim_func
def avg_pool3d_ndhwc(X: T.Buffer((N, D, H, W, C), dtype),
                     Y: T.Buffer((N, OD, OH, OW, C), dtype)):
    for n, od, oh in T.grid(N, OD, OH):             # 一个 block 一个输出行
        acc = T.alloc_buffer((OW, C), "float32")
        for kd, kh in T.grid(KD, KH):               # 窗口内每 (kd,kh) 加载一行
            id = od * SD - PD + kd
            ih = oh * SH - PH + kh
            if 0 <= id < D and 0 <= ih < H:
                T.copy(X[n, id, ih, 0:W, 0:C], buf)   # 整行 W*C 连续 → 单次搬运
                for ow in T.serial(OW):
                    iw = ow * SW - PW
                    if 0 <= iw < W:
                        for c in T.Parallel(C):        # C 维并行向量化
                            acc[ow, c] += buf[iw, c]
        for ow in T.serial(OW):
            for c in T.Parallel(C):
                Y[n, od, oh, ow, c] = acc[ow, c] / divisor
```

**设计要点**：布局选择决定搬运与向量化的形态——NDHWC 让每行 `W*C` 连续（一次 `T.copy` 即一次整行搬运），C 维归约用 `T.Parallel(C)` 向量化表达。转译时这直接对应 AscendC 的单次 `DataCopy` + `Add(C)`（见 layout-implementation.md）。

## 约束

- `permute().contiguous()` 在 host 侧产生额外的内存分配和拷贝
- 对于大 tensor，permute 开销可能显著，需要在性能分析中计入
- **仅在 kernel 计算收益 > permute 开销时使用**

## 适用场景

- AvgPool / MaxPool 等空间窗口归约算子
- 输入为 NCDHW 的 3D/2D/1D pooling
- C 维度足够大（≥8 且满足对齐要求）

## 不适用场景

- 自适应 pooling 的「固定行偏移滑动窗口」计算方式（窗口非固定，需改为逐输出点 StartIndex/EndIndex 推导，见 [adaptive-avg-pool3d-lessons.md](adaptive-avg-pool3d-lessons.md) §1）——**但 NDHWC 布局 + C 向量化本身对 adaptive pooling 依然适用**（adaptive_avg_pool3d 实测 6/6 精度、常规窗口 >1.1x）
- C 很小（<8）的极端情况
- 非标准布局输入（如 NHWC 原生输入）
