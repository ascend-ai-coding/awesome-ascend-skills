---
type: "PyPTO Performance Optimization Card"
title: "对齐连续主路径 + 隔离尾块"
description: "在 tiling 按物理连续性与对齐把主区切为静态 shape 的完整 Tile，尾块走独立 set_validshape 路径，主循环不再逐 tile 重算有效形状。"
status: "stable"
tags: ["pypto-pro", "mem", "alignment", "tail"]
item_id: "mem-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#fixed-and-dynamic-valid-regions"
      scope: "主区静态完整 Tile 与尾区动态 valid shape 分开声明，所有 load/store 使用实际有效范围。"
      boundary: "valid shape 不保证物理连续或自动初始化 padding；尾区槽位、stride、format 与 dtype 对齐须单独核验。"
bound_hint: "MTE2、MTE3"
applicability: "同一算子同时覆盖对齐与非对齐规格，主循环每个 tile 都走动态 valid_shape/保守搬运路径，且主区可切出静态 shape 的对齐完整 Tile"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 TileType 静态 shape、valid_shape=[-1,-1] 与 set_validshape"
---
# 技术卡片 mem-01：对齐连续主路径 + 隔离尾块

- **适用 bound**：访存 / MTE
- **一句话**：主路径只处理静态 shape 的对齐完整 Tile，尾块与非对齐区间拆成独立路径，主循环不携带逐 tile 的有效形状重算与保守搬运。

## 何时用（诊断特征）

- 主循环每个 tile 都调用 `set_validshape` 或走动态有效形状搬运，而实际只有最后一小段是非对齐/尾块。
- 逻辑连续但物理行跨度、内轴长度或切分边界未满足对齐，全部规格被迫共用保守模板。
- trace 中主路径 copy 粒度过碎或携带大量 padding 处理。

## 何时不适用

- tile 已经很小时，拆出独立尾块路径会增加每轮流水断点；尾块 TileGroup 深度不足（<2）会让尾块串行化，反而退化。
- 对齐切分显著缩小 tile、降低并行度或增加无效搬运时，保留统一动态路径。
- 尾块占比大时主/尾拆分意义有限，应按整体动态路径或重新设计 tiling。

## 原理

静态 shape 的 Tile 让 `pl.load`/`pl.store` 在编译期确定搬运长度，生成连续对齐的搬运指令；动态 valid_shape 路径则按保守描述符逐次处理。把两类区间拆开：主循环只含静态 Tile，无逐 tile 标量重算；尾块用 `valid_shape=[-1,-1]` 的 TileType 加 `set_validshape` 单独处理，且尾块 TileGroup 深度不小于 2 以保持轮转。连续性判定必须基于物理 stride、format 与 dtype 字节数，不能只按逻辑 shape。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的逐块 `y = x * 2` kernel；Tile 尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：主循环逐 tile 重算有效形状，全部走保守搬运
# for j in pl.range(0, n_tiles):
#     xt = x_g.next(); yt = y_g.next()
#     valid = pl.min(TILE_N, x.shape[1] - j * TILE_N)
#     pl.set_validshape(xt, [TILE_M, valid])
#     pl.set_validshape(yt, [TILE_M, valid])
#     pl.load(xt, x, [i * TILE_M, j * TILE_N])
#     pl.mul(yt, xt, 2.0)
#     pl.store(y, yt, [i * TILE_M, j * TILE_N])

# after：主区静态 shape，尾块独立路径
main_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec)
tail_type = pl.TileType(shape=[TILE_M, TILE_N], valid_shape=[-1, -1],
                        dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
xm_g = pl.make_tile_group(type=main_type, addrs=0x00000, mutex_ids=[0, 1])
ym_g = pl.make_tile_group(type=main_type, addrs=0x10000, mutex_ids=[2, 3])
xt_g = pl.make_tile_group(type=tail_type, addrs=0x20000, mutex_ids=[4, 5])
yt_g = pl.make_tile_group(type=tail_type, addrs=0x28000, mutex_ids=[6, 7])

with pl.section_vector():
    n_main = x.shape[1] // TILE_N
    tail = x.shape[1] - n_main * TILE_N
    for i in pl.range(core_id, m_blocks, num_cores):
        for j in pl.range(0, n_main):
            xt = xm_g.next()
            yt = ym_g.next()
            pl.load(xt, x, [i * TILE_M, j * TILE_N])
            pl.mul(yt, xt, 2.0)
            pl.store(y, yt, [i * TILE_M, j * TILE_N])
        if tail > 0:
            xt = xt_g.next()
            yt = yt_g.next()
            pl.set_validshape(xt, [TILE_M, tail])
            pl.set_validshape(yt, [TILE_M, tail])
            pl.load(xt, x, [i * TILE_M, n_main * TILE_N])
            pl.mul(yt, xt, 2.0)
            pl.store(y, yt, [i * TILE_M, n_main * TILE_N])
```

## 性能与验证指标

比较统一保守路径与主/尾拆分的 MTE copy 次数、标量指令数与同条件 `Task Duration`；按对齐主区、非对齐内轴、尾块分别覆盖正确性。PyPTO-Pro 的搬运抽象会隐藏大部分模板差异，大 tile 下预期差距很小，主要收益在主路径标量与描述符开销占比较高的场景。

## 技术限制与风险

- 连续性须同时检查 format、物理 pitch、offset、转置方向与 dtype 字节数；NZ/ND 的连续内轴不同。
- 尾块的 valid_shape、写回范围与访问上界必须独立验证，不能复用主路径长度。
- 对齐阈值按当前 API/硬件契约（如 32B）与 dtype 推导，不要抽成全局常量。
- 尾块路径也要双槽轮转（mutex_ids 长度 ≥2），单槽尾块会在每个主区末尾形成串行断点。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/basic_data_structures/TileType.md`、`docs/zh/api/pro_api/SIMD-API/transpose_and_element_access/set_validshape.md`
