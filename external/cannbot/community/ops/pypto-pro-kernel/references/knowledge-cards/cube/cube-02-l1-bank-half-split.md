---
type: "PyPTO Performance Optimization Card"
title: "L1 bank 冲突规避（ping/pong 分居前后半 L1）"
description: "多 buffer 的 L1 tile group 用显式 addrs 把 ping buffer 与 pong buffer 分别放入 L1 的前半与后半 bank 区，使 MTE2 写与 MTE1 读落在不同 bank，消除读写同 bank 冲突。"
status: "stable"
tags: ["pypto-pro", "cube", "l1", "bank-conflict", "memory-layout"]
item_id: "cube-02"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/memory-allocation.md#independent-buffers-and-slot-depth"
      scope: "显式安排 ping/pong 物理地址并核算所有槽位和随路数据的容量与独立性。"
      boundary: "地址互不重叠不等于已消除 L1 bank 冲突；半区边界、写优先行为和收益须按目标平台单独核验。"
bound_hint: "MTE1"
applicability: "K 循环 matmul 的 L1 操作数 buffer 数 ≥2（ping/pong），profiling 显示 MTE1 bound 或 MMAD 断流，且单个 buffer 数据总量不超过半个 L1"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 addrs 列表显式指定各 buffer 地址与 auto_mutex 轮转；L1 容量与 bank 边界按目标 ini 复核"
---
# 技术卡片 cube-02：L1 bank 冲突规避（ping/pong 分居前后半 L1）

- **适用 bound**：MTE1（L1→L0 搬运带宽不足、MMAD 断流）
- **一句话**：多个 L1 buffer 的地址按半个 L1 边界对半分，读写分离到不同 bank。

## 何时用（诊断特征）

- K 循环 matmul 的 A/B L1 tile group 深度 ≥2（ping/pong 轮转）。
- 流水证据显示 MTE1 段拉长或 MMAD 断流，且 MTE2 写与 MTE1 读在时间上重叠。
- 单侧全部 buffer 数据量 ≤ 半个 L1（容量硬约束，超限先缩小 tile 或深度）。

## 何时不适用

- 只有 1 个 buffer（深度 1）：没有 ping/pong，无从分离。
- 瓶颈在 MTE2/CUBE 而非 L1 读写冲突：改地址布局无收益。
- 单侧 buffer 数据总量超半个 L1：须先收缩 tile，不要为对半而越界。

## 原理

L1 按固定粒度划分为独立 bank 区；一条搬运指令的突发横跨多个 bank group，顺序排布的 ping/pong 必然让 MTE2 写 pong 与 MTE1 读 ping 落入同 bank，芯片策略写优先导致读带宽下降。把 ping 集中放前半、 pong 集中放后半，冲突的必要条件被消除。本手段只改地址布局，不改同步结构。

## 怎么改（before / after）

以下为嵌入片段；tile shape、mutex_ids 与 K 循环上下文须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

L1_HALF = 0x40000  # 半个 L1 的字节偏移，按目标 ini 复核

# before：buffer 顺序排布（同 bank 区相邻，读写冲突）
a_l1 = pl.make_tile_group(type=..., addrs=[0x00000, 0x10000], mutex_ids=[0, 1])

# after：ping buffer 前半、pong buffer 后半（bank 边界隔离）
a_l1 = pl.make_tile_group(type=..., addrs=[0x00000, L1_HALF], mutex_ids=[0, 1])
b_l1 = pl.make_tile_group(type=..., addrs=[0x20000, L1_HALF + 0x20000], mutex_ids=[2, 3])
```

`addrs` 列表长度须与 `mutex_ids` 数量一致；两侧操作数与随路数据（如 scale）的 ping/pong 都要纳入对应半区，只迁 A/B 不迁 scale 会留下冲突。

## 性能与验证指标

观察 MTE1 段耗时、搬运指令平均周期与 MMAD 连续性；搬运次数不应变化。正确性按常规精度回归。

## 技术限制与风险

- 半区容量是硬约束：单侧所有 buffer（含 scale 等随路数据）之和不得超过半个 L1。
- 与三缓冲等更深 buffer 方案存在预算竞争，须统一核算。
- 对 MTE1 非主导的场景收益不明显，先确认瓶颈再实施。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`（addrs 列表与 buffer 数/深度）
