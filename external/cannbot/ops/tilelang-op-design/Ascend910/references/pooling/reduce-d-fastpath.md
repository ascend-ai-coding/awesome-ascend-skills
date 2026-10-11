# Reduce-D Fast Path: Depth-Only Pooling Optimization

> **定位**：前向 reduce 专属优化。反向无对应结构（反向是 gather 区间累加），见 [backward-patterns.md](backward-patterns.md)。
> **实现篇**（direct TBuf + DataCopy 的完整 AscendC 落地）在 tilelang2ascend-translator 的 Pooling 类别，
> 见 `references/pooling-patterns/references/reduce-d-fastpath-implementation.md`（下称 reduce-d-fastpath-implementation.md）。

## 问题

当 pooling 窗口只有深度维度（KH=1, KW=1）时，每个输出行 (n,od,oh) 只加载 KD 行输入数据。由于没有 KH×KW 内窗口来摊薄 TQue 开销，每行的 `AllocTensor → DataCopy → EnQue → DeQue → ... → FreeTensor` 循环成为性能瓶颈。

avg_pool3d v2 中 c2 场景（k=(3,1,1), shape=(2,32,16,32,32)）实测 TQue 开销占比约 **2/3** 的 kernel 时间。

## 解决方案

当 KH==1 && KW==1 时，跳过 TQue，直接 DataCopy 到共享 TBuf<>（完整实现见 reduce-d-fastpath-implementation.md）。

设计本质：D 维窗口的每次加载都是「一次整行（W*C）搬运 + OW 个位置 C 维累加」，没有 KH×KW 内层窗口可摊薄 TQue 的 Alloc/EnQue/DeQue/Free 开销——因此设计中直接消除 TQue，单 buffer 直连即可。

## TileLang 设计表达（语义蓝图）

```python
@T.prim_func
def avg_pool3d_reduce_d(X: T.Buffer((N, D, H, W, C), dtype),
                        Y: T.Buffer((N, OD, OH, OW, C), dtype)):
    for n, od, oh in T.grid(N, OD, OH):           # 每个 (n,od,oh) 一个工作单元
        acc = T.alloc_buffer((OW, C), "float32")  # fp32 累加器，与通用路径同
        T.copy(acc, 0)                            # 清零
        ih = oh * SH - PH                         # KH==1 → 只有 kh=0 一行
        for kd in T.serial(KD):                   # 只有 D 维窗口（KH==KW==1）
            id = od * SD - PD + kd
            if 0 <= id < D:
                # 每个 kd 一次整行 T.copy —— W*C 连续，一次搬运
                # （转译为 AscendC 单次 DataCopy 到 direct TBuf，见
                #  references/pooling-patterns/references/reduce-d-fastpath-implementation.md）
                T.copy(X[n, id, ih, 0:W, 0:C], buf)
                for ow in T.serial(OW):
                    iw = ow * SW - PW
                    if 0 <= iw < W:
                        acc[ow, :] += buf[iw, :]  # C 维向量化累加
        for ow in T.serial(OW):
            Y[n, od, oh, ow, :] = acc[ow, :] / divisor
```

**设计要点**：KD 循环是唯一的窗口维度（无 KH/KW 内层）。每 kd 加载的行在下一次覆盖前只被 OW 个 C 维累加读取，单 buffer 即可、**无需** TQue 的多 buffer 语义——这正是「跳过 TQue」在 TileLang 设计层的对应。

## 推广到其他 Pooling 算子

此模式适用于任何 KH==1 && KW==1 的 pooling：
- MaxPool3d (k=(N,1,1)): 将 Add 替换为 Max
- AvgPool2d (k=(1,1)): 退化为 global pooling over D，结构相同
- 任何 depth/spatial 分离的 pooling variant
