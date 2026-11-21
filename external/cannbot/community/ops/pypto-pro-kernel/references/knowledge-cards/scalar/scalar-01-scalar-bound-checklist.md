---
type: "PyPTO Performance Optimization Card"
title: "Scalar bound 手段清单"
description: "标量准备挡住计算时，按清单减冗余标量、展开小循环、简化循环轴、消标量-向量往返，缩短标量关键路径。"
status: "stable"
tags: ["pypto-pro", "scalar", "checklist"]
item_id: "scalar-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#metadata-and-invariant-inputs"
      scope: "从热循环中提取真正不变的 metadata、输入与地址准备，减少标量前缀。"
      boundary: "改写坐标循环、展开或分核不是同一个优化；不变性、覆盖、host/kernel 同源及负载变化分别验证。"
bound_hint: "SCALAR"
applicability: "scalar 泳道忙而 VECTOR 空闲、总计算量极小或 shape 很小，循环内存在可外提的冗余标量计算（div/mod、重复 offset 推导）"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.range 循环标量表达式与 tile 基础算术"
---
# 技术卡片 scalar-01：Scalar bound 手段清单

- **适用 bound**：Scalar / 小 case
- **一句话**：标量准备成为关键路径时，减冗余标量、简化循环轴、消标量-向量往返、展开小定长循环，把标量关键路径缩到最短。

## 何时用（诊断特征）

- 标量泳道忙而 VECTOR 空闲；总计算量极小或 shape 很小；IPC 低。
- 循环内逐 tile 做可外提的坐标恢复（`//`、`%`、重复 stride 推导）或分支。
- SCALAR 与 VECTOR 间有不必要同步等待。

## 何时不适用

- 循环体被异步搬运/计算掩盖时（大 tile、带宽 bound），标量改写预期中性；先确认标量是关键路径。
- 除数为 2 的幂时 `//`、`%` 已是移位/掩码，不构成冗余。
- 标量-向量往返的向量化改造归 vec-04，避免重复实现。

## 原理

小 case / scalar bound 下，标量地址计算、分支、循环控制成为关键路径——后续指令发射不出去，VECTOR/MTE 干等。把逐 tile 的坐标恢复（flat 下标的 `//`、`%`）改为嵌套循环的增量 offset，把循环不变量外提，可直接缩短每个 tile 的标量前缀；小定长循环显式展开则删除循环控制本身。

## 怎么改（手段清单）

| 手段 | PyPTO-Pro 操作 | 效果 | 证据卡 |
| --- | --- | --- | --- |
| 减冗余标量 | 嵌套循环 + 增量/外提 offset，替代逐 tile `//`、`%` | 缩短标量前缀 | 本卡 |
| 简化循环轴 | tiling 侧合并可并的轴，减少循环层数 | 降标量控制 | 本卡 |
| 小循环展开 | 编译期已知小 trip count 的循环显式展开 | 删循环控制 | vec-10 |
| 外层循环下沉 | 行间无依赖的重复调用下沉进 VF | 减 setup 重复 | vec-09 |
| 消标量-向量往返 | `vf.full`/广播替代标量回读 | 减搬移与同步 | vec-04 |

以下为嵌入片段，截取自已上板验证的逐 tile `y = x * 2` kernel；尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：flat 循环逐 tile 恢复逻辑坐标
# for t in pl.range(core_id, total, num_cores):
#     i = t // j_tiles
#     j = t % j_tiles
#     xt = x_g.next(); yt = y_g.next()
#     pl.load(xt, x, [i * TILE_M, j * TILE_N])
#     pl.mul(yt, xt, 2.0)
#     pl.store(y, yt, [i * TILE_M, j * TILE_N])

# after：嵌套循环 + 增量 offset，不变量外提
with pl.section_vector():
    i_tiles = x.shape[0] // TILE_M
    j_tiles = x.shape[1] // TILE_N
    for i in pl.range(core_id, i_tiles, num_cores):
        row_off = i * TILE_M                  # 循环不变量外提
        for j in pl.range(0, j_tiles):
            xt = x_g.next()
            yt = y_g.next()
            pl.load(xt, x, [row_off, j * TILE_N])
            pl.mul(yt, xt, 2.0)
            pl.store(y, yt, [row_off, j * TILE_N])
```

## 性能与验证指标

比较标量泳道利用率、IPC 与同条件 `Task Duration`；覆盖 2 的幂与非 2 的幂分块两类 case。注意：当循环体本身被异步引擎掩盖时，此类改写预期中性，须以 trace 证实标量是关键路径后再投入。

## 技术限制与风险

- 循环轴改写与 tiling 联动，host 与 kernel 的分块计算必须同源。
- 嵌套循环改变了多核分工方式（按行块 stride 分配），负载不均的 shape 需重新核对分核。
- 标量关键路径未证实前，优先做收益明确的 mem/vec 类手段。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/controlflow/range.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
