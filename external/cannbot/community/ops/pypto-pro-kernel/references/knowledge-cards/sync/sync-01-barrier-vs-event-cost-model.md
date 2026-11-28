---
type: "PyPTO Performance Optimization Card"
title: "核间同步选型：全局屏障与核间事件的成本模型"
description: "每轮跨核依赖用全局屏障同步、导致固定开销高或全局锁步等待时，按依赖范围改用核内事件对，降低单次成本与耦合范围。"
status: "stable"
tags: ["pypto-pro", "sync", "pipeline"]
item_id: "sync-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/crosscore-sync.md#how-to-choose"
      scope: "按实际 producer/consumer 的核关系、介质、复用和阶段边界选择自动 stage 或手工事件。"
      boundary: "局部事件不能替代全设备阶段屏障；forward/reverse 配对、ID 预算和跨 block 模式须完整核验，不从历史成本直接替换。"
bound_hint: "PIPELINE、SCALAR"
applicability: "循环体内每轮存在跨核生产/消费依赖，当前使用全局屏障，且屏障总成本或全局锁步等待在总耗时中占比高"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 set_cross_core/wait_cross_core 与 CrossCoreSyncMode.INTRA_BLOCK，须核验事件 ID 与屏障占用 ID 不冲突"
---

# 技术卡片 sync-01：核间同步选型（全局屏障与核间事件的成本模型）

- **适用 bound**：PIPELINE、SCALAR
- **一句话**：把每轮的全局屏障按依赖范围替换为核内事件对，降低单次同步成本并解除全局锁步。

## 何时用（诊断特征）

- 循环体每轮存在跨核依赖，当前用 `pl.system.sync_all(MIX)` 全局屏障。
- 屏障总成本可观：按“单次屏障成本 × 每轮屏障数 × 轮数”估算，其量级与总耗时可比。
- 全局锁步导致空闲核无法提前推进，或核间工作量不均时相互拖累。
- 依赖仅在固定核对之间（如同一逻辑 block 的 cube 与两个 AIV），不涉及全设备集合。

## 何时不适用

- 依赖确实覆盖全设备集合，或需要一次性全局可见（阶段边界）时保留全局屏障。
- 依赖方向在同一轮内形成环，或事件未按配对消费完即被复用。
- 目标版本对核内事件的支持未核验：不得以本卡替代未证的能力。

## 原理

全局屏障使本轮参与核全部到达后统一放行，单次成本随参与核数与锁步等待上升，且把互不相关的核耦合。
核内事件对以计数器就绪/消费实现点对点同步，单次成本与耦合范围都小。将每轮的环内依赖改为事件对后，
核可独立推进，均衡性提升。

## 怎么改（before / after）

以下为结构示意，展示同步点改动；所用 `pl.*` 形态（`pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`、
`pl.system.set_cross_core/wait_cross_core(pipe=..., event_id=...)`、`pl.CrossCoreSyncMode.INTRA_BLOCK`、
双槽 `beat % 2` 轮转）来自历史结构片段，不能继承为当前目标的完整验收。片段是教学级最小示意：
例题设定 cube 每轮产出 `cv_ub` 双槽中的一槽、vec 同轮消费，反向以事件释放槽位；事件 ID、配对
顺序、与全局屏障/自动流水占用 ID 的隔离须按目标版本核验。

**before（每轮两次全局屏障，全设备锁步）：**

```python
with pl.section_cube():
    for it in pl.range(0, n_iter):
        produce_cv(cv_ub[it % 2], it)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # 全设备到达后统一放行
with pl.section_vector():
    for it in pl.range(0, n_iter):
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # 与 cube 侧同一点位配对
        consume_cv(cv_ub[it % 2], it)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)  # 反向释放同样走全局屏障
```

**after（环内依赖改为核内事件对，阶段边界保留全局屏障）：**

```python
EV_C2V = 0   # cube→vec 数据就绪；与 sync_all、自动流水占用 ID 隔离
EV_V2C = 2   # vec→cube 槽位释放；实际 kernel 常按 beat % 2 再分奇偶两拍
with pl.section_cube():
    for it in pl.range(0, n_iter):
        if it > 0:
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=EV_V2C,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)  # 等 vec 消费完上一拍
        produce_cv(cv_ub[it % 2], it)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_C2V)            # 通知本 block 的 AIV
with pl.section_vector():
    for it in pl.range(0, n_iter):
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=EV_C2V,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)      # 等本 block cube 产出
        consume_cv(cv_ub[it % 2], it)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_V2C)              # 释放槽位给下一拍
# 阶段边界（如 split-K 归约的全集合可见点）仍保留：
# pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
```

**关键陷阱**：事件 ID 是稀缺资源且按方向与拍次复用——相邻拍共用同一 ID 前必须确认上一过程的
SET/WAIT 已全部配对完成，否则事件失效表现为静默错值或 aicore 超时；A5 上手写逐 tile 跨核握手有
历史失败记录；落地前按 [当前 cross-core owner](../../../../pypto-pro-kernel/references/api/crosscore-sync.md) 的实际参与者、ready/release、初始空槽和最终消费完成规则构造完整序列并上板验证。本页简图省略的事件不能当成合法协议。

## 性能与验证指标

比较屏障与等待占比、逐核推进独立性、均衡性与同条件 `Task Duration`；正确性按完整套件回归，
在算子内部与多 case 顺序运行下均须复跑，排除跨 case 残留状态。

## 技术限制与风险

- 事件 ID 与 `sync_all` 占用 ID、自动流水分配 ID 不得冲突；复用同一事件 ID 前须完成前一过程全部 SET/WAIT。
- 事件失效会导致静默错值或超时；须以多 case 顺序运行与重复运行排除残留状态，不能仅凭单 case 通过。
- 阶段边界仍须全局屏障保证全设备可见；不得以事件对替代阶段边界屏障。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/synchronization/set_cross_core.md`、`docs/zh/api/pro_api/SIMD-API/synchronization/wait_cross_core.md`、
  `docs/zh/api/pro_api/SIMD-API/synchronization/sync_all.md`。
- [当前 cross-core owner](../../../../pypto-pro-kernel/references/api/crosscore-sync.md)：同步层级、参与者、事件与槽位生命周期；旧教学片段不构成当前可复用同步验收。
