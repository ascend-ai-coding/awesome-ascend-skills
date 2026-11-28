---
type: "PyPTO Performance Optimization Card"
title: "无 bound 手段清单"
description: "各硬件单元利用率均不高、无单一瓶颈但吞吐未达标时，按清单用 TileGroup 双缓冲、合并搬运、VF 融合与减少同步等手段填流水气泡。"
status: "stable"
tags: ["pypto-pro", "pipeline", "checklist"]
item_id: "pipe-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:performance-counters"
      scope: "无单一高利用单元时，用实际 trace 和各 pipe、工作量、等待区间诊断重叠与固定开销。"
      boundary: "低利用率不自动证明流水编排有错；归约依赖、启动、标量发射和传输粒度须分别建立证据。"
bound_hint: "PIPELINE"
applicability: "各硬件单元利用率均不高、无明显单一瓶颈，trace 有流水气泡与等待区间、搬移-计算 overlap 不足"
target_api_gate: "仅限 Ascend 950PR 或 950DT；各手段分别依赖 make_tile_group 深度轮转、auto_mutex、整块搬运与 VF 融合等已核验能力"
---
# 技术卡片 pipe-01：无 bound 手段清单

- **适用 bound**：无 bound
- **一句话**：无单一瓶颈但吞吐未达标时，根因多在流水编排——用双缓冲重叠搬算、合并搬运降固定开销、VF 融合消中间往返，把气泡填满。

## 何时用（诊断特征）

- 各硬件单元利用率均不高、无明显单一瓶颈，但 `Task Duration` 不达预期。
- trace 有流水气泡与等待区间、搬移-计算 overlap 不足。
- 若诊断指向**阶段先后顺序不当**（写回等刚发出的计算、批量搬运发射反压计算发射），先看 pipe-02：本卡是在既定顺序内填气泡，pipe-02 改的是顺序本身。

## 何时不适用

- 存在明确单一 bound（MTE 带宽打满、VECTOR 饱和、归约依赖链）：先按对应 bound 卡片处理（vec-08 等）。
- 小 shape（每核不足一个完整 tile 流水）：流水根本没形成，编排手段中性。

## 原理

无 bound 通常是流水编排问题：搬移与计算未重叠、同步点过多、固定开销摊不薄。PyPTO-Pro 中重叠由 `make_tile_group` 深度与 `auto_mutex` 依赖图建立，不需要手工事件配对；优化重心是让依赖图允许重叠（多槽轮转）、让每次搬运摊薄固定开销（合并）、让中间结果不落 GM（融合）。

## 怎么改（手段清单）

| 手段 | PyPTO-Pro 操作 | 效果 | 证据卡 |
| --- | --- | --- | --- |
| 开双缓冲 | `make_tile_group` 深度 ≥2 + `next()` 轮转 | 搬算重叠 | — |
| 合并搬运 | 按 UB 预算一次搬多行/多块 | 摊薄描述符与同步 | mem-02、mem-04 |
| 消中间 GM 往返 | UB Tile 接力 / VF 内融合 | 减 MTE2/MTE3 流量 | vec-01 |
| 减细粒度同步开销 | matmul 用 `phase=` 细粒度流水 | 减整段串行 | cube-01 |
| MTE2 预取错拍 | K 循环消费后立刻回填下下块 | 填 MTE2 空泡 | cube-05 |
| 退化语义路由 | identity 走 pure-copy 模板 | 删元数据与小 copy | mem-05 |

以下为嵌入片段（双缓冲手段，截取自已上板验证的 kernel）：

```python
import pypto_pro.language as pl

tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec)
x_g = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])
y_g = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])

with pl.section_vector():
    for t in pl.range(0, TPC):
        xt = x_g.next()          # 双槽轮转，框架按依赖重叠搬运与计算
        yt = y_g.next()
        pl.load(xt, x, [(t0 + t) * TILE_M, 0])
        pl.mul(yt, xt, 2.0)
        pl.store(y, yt, [(t0 + t) * TILE_M, 0])
```

## 性能与验证指标

采集同条件 `Task Duration` 与各泳道 overlap 度前后对比，重点确认双缓冲是否生效（MTE2 与 VECTOR 时间线是否重叠）。逐项手段单独验证，不叠加归因。

## 技术限制与风险

- 双缓冲使每类缓冲占用翻倍，按槽位总量核对 Vec/L1 预算。
- 预取窗口（轮转深度）过大撑爆容量；窗口须匹配计算耗时。
- 归约依赖链型空泡不属"无 bound"，用 vec-08；阶段顺序型空泡用 pipe-02。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
