# Pooling Pattern Index

转译 pooling 类算子（AvgPool/MaxPool/AdaptivePool 及反向 Grad）前先读本文件，用它决定需要的模式文档。前向是「滑动窗口 reduce」，反向是「scatter-add/gather」。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的类别，再只读对应文档；只有实现细节不够时，再继续读文档后续章节。

> **设计阶段知识不在此处**：Pooling 的 TileLang 设计模式（NDHWC 布局、行粒度 tile、reduce_d 快路径、adaptive 窗口、反向闭式公式）在 `ops/tilelang-op-design/references/pooling/`，由 `tilelang-op-design` 的决策树路由。本类别只承载**转译/实现阶段**知识。

## 核心模式

| 如果看到 | 读这个文档 |
| --- | --- |
| TileLang 设计无法高效执行 / 需要逐语义转译映射（`T.copy→DataCopy`、AUTO_SYNC→`PipeBarrier`） | `@references/pooling-patterns/references/tilelang-translation.md` |
| TQue 每窗口 Alloc/EnQue/DeQue/Free 开销大 / 需要直连 TBuf | `@references/pooling-patterns/references/ub-management.md` |
| NDHWC 布局落地：host permute + kernel 行地址 + 整行 DataCopy（设计语义见 layout-strategy.md） | `@references/pooling-patterns/references/layout-implementation.md` |
| 前向 KH==1&&KW==1 深度-only 快路径落地：direct TBuf + 完整 kernel + barrier 决策（设计语义见 reduce-d-fastpath.md） | `@references/pooling-patterns/references/reduce-d-fastpath-implementation.md` |
| `UB address not aligned` 崩溃 / 输出值 512.0/nan / 需要 host `TORCH_CHECK` 对齐守卫 | `@references/pooling-patterns/references/alignment-guards.md` |
| fp32 累加 / divisor 三级策略 / bf16 下 Cast round mode | `@references/pooling-patterns/references/precision-patterns.md` |
| 反向 Grad 算子（AvgPoolGrad/MaxPoolGrad/AdaptivePoolGrad）实现：WAR 陷阱、标杆口径、output-driven transpose-scatter、`SetAtomicAdd`、六路 tiling key | `@references/pooling-patterns/references/backward-implementation.md` |
| 反向落地增量踩坑：`GetCoreNumAiv` vs `GetCoreNum`、`Axpy`、Cast 分派、data_format 四象限、延迟分桶 | `@references/pooling-patterns/references/grad-v2-lessons.md` |
| kernel 实现常见错误：TQue 过载、per-ow barrier、DeQue in-place、双重屏障缺失、固定核数 | `@references/pooling-patterns/references/pooling-anti-patterns.md` |

## 设计阶段指引（转译输入）

Pooling 的**设计语义**来自 `tilelang-op-design` 的 Pooling 领域参考（`ops/tilelang-op-design/references/pooling/`，由决策树 Pooling 分支路由），转译时按需回读：

- 前向 reduce 设计：`layout-strategy.md`（NDHWC 布局）、`row-granularity.md`（行粒度 tile）、`reduce-d-fastpath.md`（KH==1&&KW==1 快路径）
- adaptive 窗口：`adaptive-avg-pool3d-lessons.md`
- 反向语义设计：`backward-patterns.md`（scatter-add 语义、闭式公式、divisor 共享、block 语义对调）
- 反向实现（本篇）：`backward-implementation.md`（陷阱 + 高性能落地）

## 组合顺序

反向 Grad 算子实现时按这个顺序理解：

```text
1. 语义：读 design 的 backward-patterns.md §1-§8（scatter-add、闭式公式、divisor、布局、ArgMax、多核）。
2. 实现策略：读 backward-implementation.md §12.5 决策表（gather vs transpose-scatter 何时用）。
3. 基础实现：读 ub-management.md / alignment-guards.md / precision-patterns.md（buffer、对齐、精度）。
4. 落地踩坑：读 backward-implementation.md §11（WAR、标杆、串行瓶颈）+ grad-v2-lessons.md（核数、Axpy、Cast、data_format、分桶）。
5. 常见错误：读 pooling-anti-patterns.md 对照检查。

前向 reduce 算子按需读 tilelang-translation.md（转译映射）+ ub-management.md / alignment-guards.md / precision-patterns.md；NDHWC 布局落地读 layout-implementation.md，KH==1&&KW==1 深度快路径落地读 reduce-d-fastpath-implementation.md。

## 生成前问题

1. 前向还是反向？（reduce 窗口 → 单值，或 grad 单值 → 窗口）
2. 前向：KH==KW==1 吗？（→ reduce_d 快路径）窗口固定还是 adaptive？
3. 反向：AvgPool（只依赖 shape）还是 MaxPool（依赖 argmax）？
4. 布局：NCDHW 还是 NDHWC？C 维是否可向量化（`C%8`/`C%16`）？
5. 反向实现策略：input-driven gather（默认）还是 output-driven transpose-scatter（大 kernel/重叠）？
6. 是否有 `data_format` 属性、ceil_mode、count_include_pad、divisor_override？

根据答案选择上面的文档。所有 pooling kernel 的累加必须在 fp32 中进行（`precision-patterns.md`），对齐守卫是 host 侧强制 `TORCH_CHECK`（`alignment-guards.md`）。
