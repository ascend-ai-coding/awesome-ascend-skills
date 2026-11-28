---
type: "PyPTO Performance Optimization Card"
title: "三阶段顺序重排（计算发射优先 + 写出延后一拍）"
description: "双缓冲已开却仍不并行时，把消费异步计算结果的写出延后一拍、把下一轮搬入排在计算发射之后，不改计算逻辑提升流水重叠。"
status: "stable"
tags: ["pypto-pro", "pipeline", "scheduling"]
item_id: "pipe-02"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/memory-allocation.md#independent-buffers-and-slot-depth"
      scope: "重排计算、预取与延迟写出时，输入输出均要有足够独立槽位，并等最终 store 完成后才能复用。"
      boundary: "增加深度不证明当前程序序已重叠；本卡首轮、末轮补写、索引窗口和发射阻塞须在实际 trace 上核验。"
bound_hint: "PIPELINE"
applicability: "TileGroup 双缓冲已开但 trace 显示搬入-计算-写出仍串行（写回等待刚发出的计算、搬运发射反压计算发射），且每核 tile 数足够形成流水"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 group[i] 运行时索引与 auto_mutex 依赖"
---
# 技术卡片 pipe-02：三阶段顺序重排（计算发射优先 + 写出延后一拍）

- **适用 bound**：无 bound
- **一句话**：双缓冲已开却仍不并行时，每拍按"先发当前计算 → 再发下一轮搬入 → 最后写出上一拍结果"重排，写出一律晚一拍。

## 何时用（诊断特征）

- 【前置】TileGroup 深度 ≥2 已开，但搬入与计算仍不重叠（MTE2 全在前、VECTOR 全在后）。
- trace 上写回泳道 wait 比例高（写出在等刚发出的计算），或标量泳道出现大块空窗。
- 每核 tile 数足够（流水能形成）；小 shape 无收益。

## 何时不适用

- 双缓冲未开：先开双缓冲（`make_tile_group` 深度 ≥2 + `next()` 轮转），本卡不是它的替代品。
- 框架依赖图已自动重叠（多数纯 Tile API 逐 tile 循环即属此类）：手排不带来额外重叠，只增加状态维护成本。
- 每核只有一个 tile：流水未形成，重排中性。

## 原理

普通"搬入→计算→写出"循环里，写出消费的可能是**刚发出、还在跑**的异步计算结果（消费阻塞），大批量搬入发射也可能占住发射通道把计算发射反压（发射阻塞）。重排后：当前拍的计算先发射、自行在后台跑；下一轮搬入随后发射，与计算并行；写出的是**上一拍**早已完成的结果，不再等待。PyPTO-Pro 的 `auto_mutex` 依赖图已覆盖大部分重叠空间，本卡只用于 trace 证实存在上述发射/消费阻塞形态的场景；两处改动正交，须同时做。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的逐 tile `y = x * 2` kernel；尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：每拍 搬入 → 计算 → 写出
# for t in pl.range(0, TPC):
#     cur = t % 2
#     pl.load(x_g[cur], x, [(t0 + t) * TILE_M, 0])
#     pl.mul(y_g[cur], x_g[cur], 2.0)
#     pl.store(y, y_g[cur], [(t0 + t) * TILE_M, 0])

# after：计算发射优先 + 写出延后一拍
tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec)
x_g = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])
y_g = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])

with pl.section_vector():
    pl.load(x_g[0], x, [t0 * TILE_M, 0])
    for t in pl.range(0, TPC):
        cur = t % 2
        pl.mul(y_g[cur], x_g[cur], 2.0)                            # 1. 先发计算
        if t + 1 < TPC:
            pl.load(x_g[(t + 1) % 2], x, [(t0 + t + 1) * TILE_M, 0])   # 2. 再发下轮搬入
        if t > 0:
            pl.store(y, y_g[(t - 1) % 2], [(t0 + t - 1) * TILE_M, 0])  # 3. 写出上一拍
    pl.store(y, y_g[(TPC - 1) % 2], [(t0 + TPC - 1) * TILE_M, 0])      # 收尾补写最后一拍
```

## 性能与验证指标

比较写回泳道 wait 比例、重叠因子与同条件 `Task Duration`；正确性按常规精度回归，重点覆盖奇/偶 tile 数两种槽位对齐与最后一拍的收尾。以 trace 的相对占用形态判收益，不凭单点耗时。

## 技术限制与风险

- 先确认双缓冲已开且确实未重叠再手排；框架已自动重叠的算子不该走这条路。
- 输入、输出两组缓冲都须深度 ≥2：写出延后一拍后，上一拍结果与当前拍结果同时挂起。
- 越界守卫（`t + 1 < TPC`、`t > 0`）与收尾补写不能省；漏写收尾丢失最后一个 tile。
- 跨 tile 的索引/元数据窗口要能同时容纳待写、在算、预取三个 tile。
- 收益口径随 shape 变化：小 shape 中性；评估须覆盖目标 shape 区间。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`（group[i] 运行时索引）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
