# 性能结果聊天自动输出规范

本文规定 Agent 在 Phase 6（性能采集）与 Phase 6.5（性能优化）**完成后 MUST 自动在聊天界面输出性能结果**，无需用户追问。数值 **MUST** 来自实际采集：SHMEM 输出 `[PERF] source=inline_event` 的 `e2e_us/kernel_us`，并额外输出 msprof 的 `kernel_us_msprof`；baseline 只要求 msprof `kernel_us`。**NEVER** 编造或照搬历史参考值。

性能评估唯一标准见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)。

## 1. 自动输出原则（Hard Gate）

| 原则 | 说明 |
| --- | --- |
| **自动触发** | 性能脚本执行完毕 → **同轮对话内 MUST 输出带宽表+时延表**，不得只写路径或只报时延 |
| **禁止静默** | **NEVER** 等用户说「输出性能结果」才补表 |
| **指标口径** | 通信算子达标与对比 **MUST** 用 `kernel_bus_bandwidth_GBps`；`e2e_bus_bandwidth_GBps` 仅作参考 |
| **优化轮次** | Phase 6.5 每轮结束 **MUST** 输出「本轮 vs 上轮 vs Round0」变化表 |
| **环境** | **MUST** 在 `phase0_intake.docker_container` 指定容器内 `docker exec` 执行（见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)）；禁止宿主机裸跑 perf |

---

## 2. Phase 6：首次性能采集（Round 0）

### 2.1 执行顺序（分阶段 — HCCL 与 SHMEM 不得同会话）

命令全文见 [perf-workflow.md §1](perf-workflow.md)。顺序：

```text
阶段 A → perf-workflow.md §1 阶段 A（仅 HCCL，独立 shell）
阶段 B → perf-workflow.md §1 阶段 B（仅 SHMEM，新 shell，间隔 ≥30s；同时保存 [PERF] 日志和 msprof profile）
阶段 C → perf-workflow.md §1 阶段 C（离线解析 [PERF] + msprof task_time）
```

### 2.2 聊天 MUST 输出（两块）

**块 A — S 档 + L 档性能表**（Round 0 覆盖 S+L 档）：

```text
### Round 0 性能（<op_name> · <dtype> · 8PE）

| 规模 | case_id | e2e_us(inline) | kernel_us(inline) | kernel_us(msprof) | kernel_bus_GBps(msprof) | utilization%(msprof) |
|------|---------|----------------|-------------------|------------------|-------------------------|---------------------|
| S 档 | ... | ... | ... | ... | ... GB/s | ...% |
| L 档 | ... | ... | ... | ... | ... GB/s | ...% |
```

**块 B — P2P 对比 SHMEM vs Baseline 对比**（达标判定）：

```text
### 性能对比（case=<case_id> · L 档）

| 指标 | SHMEM inline_event | SHMEM msprof | HCCL baseline msprof | SHMEM(msprof)/baseline | 达标 |
|------|--------------------|------------------------|----------------------|------------------------|------|
| kernel_bus_bandwidth_GBps | ... | ... | ... | ...% | PASS/FAIL |
| e2e_us | ... | N/A | N/A | N/A | — |
| kernel_us | ... | ... | ... | ... | — |

达标线：有 baseline 时默认 ≥ baseline 80%；无 baseline 时通信算子带宽利用率 ≥ 20%
Phase 6 结论：达标 → Phase 7
未达标 + `performance_auto_optim: true` → **MUST 立即** Phase 6.5
未达标 + `performance_auto_optim: false` → 仅报告差距，**禁止**擅自优化
```

---

## 3. Phase 6.5：每轮优化后自动输出

每完成一轮（Round 1~5）的 OptimStep 5（性能验证）后，**MUST** 在聊天输出以下三块。中间轮次以 **L 档**性能为准。

### 3.1 本轮 Step 摘要（若有多 step）

```text
### Round <N> Step 摘要（<round_label>）

| Step | 变更 | Correctness | kernel_bus_bandwidth_GBps | vs Round0 | vs 上轮最优 |
|------|------|-------------|---------------------------|-----------|-------------|
| N.1  | ...  | PASS/FAIL   | ...                       | +X%       | +Y%         |
| N.2  | ...  | PASS/FAIL   | ...                       | ...       | ...         |
```

- **vs Round0**：`(bus_bw_step / bus_bw_round0 - 1) * 100%`
- **vs 上轮最优**：相对 Phase 6.5 上一轮 keep 配置的 kernel_bus_bandwidth_GBps

### 3.2 本轮最优 vs 基线

```text
### Round <N> 最优配置性能验证

| 指标 | Round 0 | Round <N> 最优 | Δ% vs R0 | 上轮最优 | Δ% vs 上轮 |
|------|---------|---------------|----------|----------|------------|
| kernel_bus_bandwidth_GBps | ... | ... | ... | ... | ... |
| e2e_us | ... | ... | ... | ... | ... |
| kernel_us | ... | ... | ... | ... | ... |

决策：keep / revert
```

### 3.3 累计轮次一览（每轮结束后更新）

```text
### 优化进度总览（已完成 N/5 轮）

| Round | 性质 | 最优 kernel_bus_bandwidth_GBps | vs R0 | vs baseline | 决策 |
|-------|------|-------------------------------|-------|-------------|------|
| 0 | baseline | ... | — | ...% | — |
| 1 | 机制优化 | ... | +X% | ...% | keep |
| 2 | 机制优化 | ... | ... | ... | ... |
```

**NEVER** 只在 Round 5 结束才输出总览；**每轮结束 MUST 更新**上表。

---

## 4. Phase 6.5 结束 / Phase 7 交付

**MUST** 额外输出最终对比（S 档 + L 档）。对比基准始终为**外部基线**（HCCL/aclnn/metric_only），**NEVER** 以 SHMEM Round 0 作为 §6 Baseline 列。

```text
### 最终性能（SHMEM 最终 vs <baseline>）

| 规模 | 指标 | SHMEM 最终 | <baseline> | SHMEM/基线 | 达标 |
|------|------|-----------|------------|-----------|------|
| S 档 | kernel_bus_bandwidth_GBps | ... | ... | ...% | PASS/FAIL |
| L 档 | kernel_bus_bandwidth_GBps | ... | ... | ...% | PASS/FAIL |

SHMEM 最终 = 最优轮 keep 配置复采数据；未进入 Phase 6.5 时 = Round 0 SHMEM 数据。
整体结论：达标 / 未达标；主要瓶颈；推荐配置
```

若经过 Phase 6.5 优化，额外追加内部提升摘要：

```text
### 优化内部提升（Round 0 → 最优轮）
| 规模 | Round 0 | 最优轮 | Δ% |
|------|---------|--------|-----|
| L 档 kernel_bus_bandwidth_GBps | ... | ... | +X% |
```

---

## 5. 数据来源

性能数据来源与采集命令见 [perf-workflow.md](perf-workflow.md)。优化轮次内 step 扫描也必须用 msprof 包裹算子 `scripts/perf.sh`（L 档固定 `base_count`）；**Round 最优 step MUST** 与 Round0 同口径对比。

---

## 6. 反模式

- ❌ 性能脚本跑完只写文件路径
- ❌ 优化 5 轮只在最后一轮报数
- ❌ 用 e2e 带宽作 round 间对比（固定开销导致虚假「提升」）
- ❌ 每轮换不同 payload 导致不可比
- ❌ 同一 docker exec 内连续跑 HCCL baseline 与 SHMEM perf

---

## 7. 与 performance_report.md 的关系

| 输出位置 | 内容 |
| --- | --- |
| 聊天 | 上述表格（用户即时可见） |
| `docs/performance_report.md` | 完整 §1~§7（存档）；eval skill 写入，optim 不直接改 |

聊天摘要与报告 **MUST** 数值一致；报告可更详，但不得与聊天矛盾。

## HCCL 展开模式脚注（MUST）

对比表下方 **MUST** 标明：`HCCL_OP_EXPANSION_MODE=<AIV|AI_CPU>`。若为 `AI_CPU` 回退，加一句：小消息启动时延可能偏大。详见 [hccl-expansion-mode.md](hccl-expansion-mode.md)。