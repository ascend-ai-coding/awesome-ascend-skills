---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
category: operator_optimization
title: "Attention / MLA 类算子优化点库（Triton-Ascend）"
description: "MLA 类算子在 Triton-Ascend 上的瓶颈判别、仿射 store 修复、KV 共享与 split-KV、P_SPLIT 精度补偿。"
tags: [affine_addressing, tiling_optimization, precision_compensation, synchronization_bound, pipeline_optimization, bottleneck_analysis, profiling, launch_overhead]
keywords: [mla, matrix_absorption, ckv, paged_attention, page_table, BLOCK_QO, affine_store, P_SPLIT, split_kv, LOG2E, aicore_timeout]
created_at: '2026-08-13T00:00:00Z'
---

# Attention / MLA 类算子优化点库（Triton-Ascend）

> **优化点 #31 的参考文档。** 命中条件见 `SKILL.md` 索引表。
> 证据基础：BatchMLAPagedAttention 类算子的完整优化轨迹（Ascend910B2C，24 cube core / 48 vector core，
> triton-ascend 3.2.x / CANN 8.5.x）：基线时 reduce kernel 病态（平均毫秒级）→ 结构重写后精度通过但性能劣化
> → **`+ BLOCK_QO=1` 仿射化修复**（决定性一步）。
> 设计期与编码期的对应约束见 `@../../../../plugins-official/triton-op-generator/template/mla.md`。

---

## 0. 一句话结论与目标函数

MLA decode 的性能天花板几乎总是藏在**寻址/数据路径**，而不是算法结构或算术量：
Ascend 后端对**无法证明仿射（provable affine）**的 load/store 做逐元素标量化（单 `[16, Dc]` store 毫秒级、
间歇 aicore timeout / 设备 hang）。因此**目标函数不是算术量，而是「是否全部 load/store 可证仿射」**；
一切优化等价于**把行号变成标量或常量、把 KV 迭代数压到 UB 上限允许的最小值**。

---

## 1. 瓶颈判别（先答 Q0/Q1，再决定优化方向）

| 症状 | 判别 | 指向 |
|---|---|---|
| 单 store 毫秒级 / 间歇 aicore timeout | 行号由 `rows//GH` 等 `div`/`mod` 推导（即使恒为 0 也无法折叠） | **§3.1 仿射 store 修复（最高优先）** |
| `q_len > 1`（MTP/spec-decode）+ 增量因果 | 主循环二维 `[q_len, kv]`，掩码未特化 | mask constexpr 特化 + 区间收缩 |
| 长 KV 单 program 扫全部 | 迭代数 = kv_len/BLOCK_KV，UB 未用满 | BLOCK_KV 开到 UB 上限 / split-KV |
| 16-bit 输入精度失败 | 与 fp32 标杆偏差超 1-ulp | P_SPLIT 双点积 |
| 页表未 kernel 内映射 | host 重排 / BLOCK_KV 依赖 page_size | 页表 kernel 内做、BLOCK_KV 与 page_size 解耦 |
| reduce kernel 病态（ns=1 case） | 3D masked load 触发 vector core timeout | reduce 用标量循环 |

---

## 2. 有效方向（按收益排序）

1. **★★★ `BLOCK_QO=1` 仿射 store 修复**（§3.1）：消除非仿射 store 病态（核心约束）。
2. **★★ KV 复用**：一个 program 处理同一 request 的 GH 个 head，KV tile 只 load 一次。
3. **★★ `P_SPLIT` 双点积**（16-bit 输入）：`p_hi + p_lo` 双 dot 补偿，过精度闸门。
4. **★ split-KV**：`ns = max(1, min(8, max(ns_par, ns_kv)))`，长 KV 必开；reduce 标量循环。
5. **★ UB 预算自适应**：静态估算 <140KB（F7 公式），bm/bkv 取首组可编译配置。
6. 页对齐大块加载（剩余空间方向，未验证）。

---

## 3. 关键修复技巧

### 3.1 仿射 store 修复（SINGLE 开关 A/B 定位法）

**症状**：病态 case 单次 forward 毫秒级甚至间歇死循环；普通 case 的单个 `[16, 256]` store 也病态。
**定位**：同配置 A/B——`SINGLE=0`（`q_row` 标量直接仿射 store）正常 vs `SINGLE=1`
（行号 = `rows//GH`、`rows%GH` 计算，即使恒为 0 也无法折叠）毫秒级/死循环。
**根因**：无法证明仿射的 store lowering 为逐元素标量写。
**修复**：`BLOCK_QO=1`，`rows = tl.arange(0, BLOCK_M)` 直接作为 GH 个 head 的索引，
`h = h_group*GH + rows` 为仿射、`q_row` 标量 → 全部 Q/O/LSE load/store 可证仿射。

### 3.2 页表映射

虚拟 KV 位置 j 经 C 序 head 交错展平映射到物理 cache 行：
`(pages[j // (page_size*H)] * page_size + (j // H) % page_size) * H + j % H`；
页可乱序、可冗余；越界页 `other=0` 显式掩码；索引用 int32。

### 3.3 split-KV reduce

长 KV 切到多个 program，小 reduce kernel 归并 partial `(acc, m, l)`；
reduce **禁止 3D masked load**（vector core timeout），用 `for s in range(num_splits)` 标量循环。

---

## 4. 会误导的 profiling 字段

- **profiler 伪影实为真慢**：偶发 aicore timeout / 设备 hang 常被归为"伪影"，实为逐元素标量 store 的真实耗时。
- **按 Name 分组看**：病态耗时集中在 ns=1 的 fwd 和多数 case 的 reduce——
  拆 `_mla_paged_fwd_kernel` / `_mla_splitkv_reduce_kernel` 分开看，不要整 kernel 调。
- **launch_count**：一次 forward 内多次启动同一 kernel（L>1）时按**单次调用**耗时算，不要除以发射次数。

