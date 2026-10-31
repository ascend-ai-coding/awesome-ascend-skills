---
name: external-cannbot-community-model-model-recommend-optimization-loop
description: 持续优化循环（闭环工作流，贯穿条目 5→14 的迭代编排）。按「选动作 → 执行优化（双门禁）→ profiling 采集 → 深度分析
  → 收益判定 → 终止检查 → 下一步动作表更新」循环迭代，直到达到指定性能目标或连续 3 轮无收益。状态由 loop_state.json 管理。触发：优化循环、迭代优化、循环执行、性能目标、收敛终止、无收益终止、多轮优化、自动迭代。
original-name: model-recommend-optimization-loop
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 持续优化循环（闭环工作流）

## 流水位置

工作流条目 0-4 就绪后（环境/模型分析/基线/精度门禁建立），**本 skill 编排条目 5-14 的循环执行**：每一轮从下一步优化动作表取动作实施，profiling 复测 + 深度分析，更新动作表，直到终止条件满足。线性流水（0→14）是本循环的**首轮展开**；首轮收口后进入后续轮次。

```
初始化（条目 0-4 就绪 + 性能目标确认 + 噪声带测定）
  ▶ 循环轮次 k = 1, 2, 3, ...
     L1 动作选择：next_actions 顶部可执行项（前置依赖满足）
     L2 执行优化：进入对应条目 skill（6/7/8/9/11/12）实施，过「改动校验双门禁」
     L3 profiling 采集：优化后配置同口径 --prof msprof，归档 prof/{model_name}/iter{k}/
     L4 profiling 深度分析：条目 5 四项专项（[profiling-deep-analysis](../model-recommend-profiling-deep-analysis/SKILL.md)）
     L5 收益判定：主指标（目标口径）对照上轮，改善 > 噪声带？
          是 → no_gain_streak = 0，动作保留，写台账
          否 → no_gain_streak += 1（动作按双门禁结果回退或保留）
     L6 终止检查（见下）——满足即跳出
     L7 交付件与动作表同步更新（强制）：本轮全部交付件落盘，之后才可进入下一轮
  收口：条目 14 最终报告（含循环轨迹：每轮动作/收益/no_gain_streak/终止原因）
```

## Prerequisites

- 条目 0-4 已就绪（基线 + 精度门禁 + 同口径对照基准）
- **性能目标已由用户指定**（缺失时向用户确认，禁止自行设定）：`目标指标`（latency_avg_ms / tp99_ms / qps）+ `方向与阈值`（如 `avg ≤ 1.0ms`、`QPS ≥ 1500`）
- 噪声带已测定（[performance-analysis](../model-recommend-performance-analysis/SKILL.md) 改动性能校验口径；未测定则首轮先测）
- 初始动作表存在（首轮收口产物，或基线后直接由条目 5 生成）

## 目标

把「优化 → 复测 → 分析 → 再优化」固化为可终止、可追溯、每轮有账的闭环；避免无限迭代（收益枯竭即停）与盲目迭代（目标达成即停）。

## 终止条件（满足其一即终止，记录于 loop_state）

| 条件 | 判定 | 终止原因 |
|------|------|---------|
| **性能目标达成** | 目标指标连续 2 次同口径复测达标（防单次波动误判） | `target_reached`（成功） |
| **连续 3 轮无收益** | no_gain_streak ≥ 3（每轮判定口径：主指标改善 ≤ 噪声带；回退轮计入） | `converged_no_gain`（收敛） |
| 安全阀：无动作可执行 | 动作表无可执行项且无可新增候选（含用户确认被拒） | `no_actionable` |
| 安全阀：轮次/预算上限 | 达到用户指定的最大轮次或时间预算（默认上限 10 轮） | `budget_exhausted` |

- "无收益"的严格口径：该轮**所有已执行动作**（含组合）的主指标改善 ≤ 噪声带；分析类轮次（无动作执行，仅深度分析）不计入 no_gain_streak，但连续 2 轮纯分析无新候选时触发安全阀
- 回退轮（动作落地后不达标被回退）**计入** no_gain_streak

## 状态管理：loop_state.json（强制，唯一状态源）

路径：`data/{model_name}/optimization_log/loop_state.json`，每轮 L5 后更新：

```json
{
  "target": {"metric": "latency_avg_ms", "op": "<=", "value": 1.0, "confirmed_by_user": true},
  "noise_band_pct": 2.0,
  "iteration": 3,
  "no_gain_streak": 1,
  "max_iterations": 10,
  "history": [
    {"iter": 1, "action": "启用 FORCE_16", "entry": "条目6",
     "perf_before": {"latency_avg_ms": 1.802}, "perf_after": {"latency_avg_ms": 1.100},
     "gain_pct": -38.9, "landed": true, "prof_analysis": "data/.../iter1/findings.json"}
  ],
  "terminated": false,
  "termination_reason": null
}
```

- **幂等恢复**：循环中断后重启，读 loop_state 从 `iteration + 1` 继续，不重做已记录轮次
- perf_ledger.json 照常逐动作记账（`step` 字段标注循环轮次，如「循环第2轮：XXX」）；loop_state 是轮次视图，ledger 是动作视图，二者对账一致

## 每轮执行纪律

1. **L1 动作选择**：取 `next_actions.json` 中优先级最高且前置依赖满足的**单个**动作（一轮一动作，保证归因唯一；强关联的组合动作在动作表中须登记为一条）
2. **L2 执行**：按动作的 `workflow_entry` 进入对应条目 skill；**双门禁全程有效**（精度不过 → 该轮回退并计入 no_gain_streak；性能不过 → 回退同上）；需用户确认的动作（方案确认步骤）在 L1 前完成确认，被拒则该动作标记 `rejected_by_user` 并重选
3. **L3 采集**：同口径（同卡/batch/THREAD_NUM/CORE_NUM/统计策略）`--prof msprof`；归档 `prof/{model_name}/iter{k}/`（即工作流口径的 `{case}`）
4. **L4 深度分析**：四模块全做（[profiling-deep-analysis](../model-recommend-profiling-deep-analysis/SKILL.md)），优化点并入动作表
5. **L5 收益判定**：主指标 = 目标指标同口径复测（≥3 次重复或框架统计策略取稳定值）；改善 ≤ 噪声带 → 无收益
6. **L6 终止检查**：按终止条件表
7. **L7 交付件与动作表同步更新（强制，每轮必做，未完成即该轮未完成）**：
   - **台账类（实时更新）**：`perf_ledger.json` 追加本轮动作记录（`step` 标注「循环第 k 轮」）；`loop_state.json` 更新 iteration / no_gain_streak / history
   - **分析类（当轮产出）**：`data/{model_name}/profiling-deep-analysis/iter{k}/findings.json` + `docs/{model_name}/profiling-deep-analysis/iter{k}/deep_analysis_report.md`（L4 产物，当轮落盘）
   - **条目报告类（当轮完成即写，禁止推迟到收口）**：本轮动作所属条目 skill 的 docs 交付件——如精度战役/验证报告（[precision-comparison](../model-recommend-precision-comparison/SKILL.md)）、策略与选项验证报告（[ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)）、并发验收报告（[multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md)）、算子设计文档（[kernel-generation](../model-recommend-kernel-generation/SKILL.md)）等
   - **最终报告轨迹（增量追加）**：`docs/{model_name}/final_optimization_report.md` 的「循环轨迹」表**每轮追加一行**（轮次/动作/条目/before/after/收益/落地/计数/交付件指针）——收口时只补累计收益与结论，**禁止事后批量补写**
   - **动作表**：`next_actions.json` 重排（完成项标记 done；新发现并入；回退项附重试条件；`loop_status` 字段同步）
   - 判定：以上任一未更新 = 该轮未完成，不得进入 L1（下一轮）

## 收口（循环终止后，条目 14）

最终报告在既有内容外**必须追加「循环轨迹」章节**：

```
| 轮次 | 动作 | 条目 | before | after | 收益% | 落地 | 备注 |
|------|------|------|--------|-------|-------|------|------|
| 1 | ... | 条目6 | ... | ... | ... | ✅ | ... |

终止原因：target_reached / converged_no_gain / no_actionable / budget_exhausted
目标达成情况：目标 avg ≤ 1.0ms → 最终 0.98ms（达成）/ 未达成（差距 x%）
每轮深度分析报告：docs/{model_name}/profiling-deep-analysis/iter{k}/
```

## 昇腾 NPU 专属要点

- 单卡约束全程有效：循环各轮同卡执行（共享环境换 die 须整轮重采基线口径并在台账注明）
- 共享设备的干扰判定：某轮"无收益"若伴随 npu-smi 他进程占用，须复测一次再计数（防误判收敛）
- profiling 轮次成本控制：L3 采集 circle 不必与压测一致（50-100 次足够 op_summary 稳定）；正式收益判定用压测口径（[performance-analysis](../model-recommend-performance-analysis/SKILL.md)）
- 多线程场景的 profiling：THREAD_NUM>1 时深度分析的 host/多流模块价值最大（TP99 劣化归因），但注意框架已知缺陷（多线程退出挂起，M100 案例 RD-M100-01）用 timeout 保护

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——数值区间用全宽 `～` 或 `-`/`至`，约数用 `≈`；交付前 grep 确认无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **数据类交付件** → `data/{model_name}/optimization_log/`：`loop_state.json`（循环状态）+ `perf_ledger.json`（动作台账）；`data/{model_name}/profiling-deep-analysis/iter{k}/findings.json`
- **profiling 归档** → `prof/{model_name}/iter{k}/`（每轮优化后配置的同口径采集）
- **文档类交付件** → `docs/{model_name}/profiling-deep-analysis/iter{k}/`；循环轨迹并入 `docs/{model_name}/final_optimization_report.md`
- **中间产物** → `tmp/{model_name}/optimization-loop/`：轮次调度脚本、复测原始日志
- 全局约定见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- loop_state.json 与 perf_ledger.json 对账一致（每轮动作可追溯）
- 每轮有：动作记录 + 双门禁结论 + 同口径复测数据 + 深度分析 findings
- **每轮交付件同步在案**（L7 纪律）：轨迹表逐轮追加、当轮条目 docs 报告已落盘、无"事后批量补写"痕迹
- 终止时：终止原因明确（四选一）；目标达成情况量化；循环轨迹章节完整
- **连续 3 轮无收益的判定有数据支撑**（每轮的噪声带与改善量在案）
- 中断恢复不重做已完成轮次（幂等性）

## 依据

【实测】M100 首轮全链路（基线→三表→动作表）已验证循环的输入输出闭环；api_statistic/op_summary 数据支撑深度分析四模块。
【框架】`--prof msprof` 采集、`[GE PROGRESS]` 统计口径、运行参数可回退性。
【技能库经验】单变量归因（一轮一动作）、噪声带判定、收敛终止（连续 N 轮无收益）、目标终止需连续复测防波动。
