> **Baseline**：`HCCL_OP_EXPANSION_MODE` 优先 `AIV`，回退 `AI_CPU`（见 hccl-expansion-mode.md）。

# 性能报告：<op_name>

> **章节约束**：本模板节号 §1~§7 为固定结构，**MUST** 严格遵循，**NEVER** 新增、删除或重命名章节。采集命令、指标公式、正确性回归等辅助信息归入已有子节（如 §1 或 §2），不单独建章。

## 1. 算子信息

| 项目 | 值 |
| --- | --- |
| 算子名 | |
| Shape | |
| dtype | |
| PE 数 | |
| SoC | |
| 目标指标 | |
| baseline 来源 | (hccl / aclnn / stitched / metric_only) |

## 2. Baseline 详细信息

- Type: <hccl | aclnn | stitched | metric_only>
- Source API: <具体 API 名称>
- Search process: <逐步排查记录：HCCL 清单 → aclnn 清单 → metric_only，每步结论>
- Measurement: <baseline 测试命令和数据>

## 3. 基线数据（Round 0 — 来自 shmem-ops-performance-eval）

### 3.1 性能结果表

| case_id | shape | dtype | n_pes | e2e_us | kernel_us | kernel_bus_bandwidth_GBps | algo_GBps | e2e_bus_bandwidth_GBps | utilization% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### 3.2 通信指标表（**MUST**）

| case_id | logical_payload_bytes | bus_factor | peak_bandwidth_GBps | bandwidth_utilization% |
| --- | --- | --- | --- | --- |

### 3.3 计算指标表（通信算子标注 "N/A"）

| case_id | op_count_flops | compute_latency_us | effective_flops | peak_flops | compute_utilization% |
| --- | --- | --- | --- | --- | --- |

### 3.4 Device Frame Table

（**MUST** 存在）

| Frame | Phase | Avg us | Max core us | Count | % E2E | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- |

### 3.5 性能对比表（有 baseline 时 **MUST** — hccl / aclnn / stitched 均适用）

| metric | SHMEM kernel_bus_bandwidth_GBps | SHMEM e2e | SHMEM kernel | Baseline kernel_bus_bandwidth_GBps | delta | 达标 |
| --- | --- | --- | --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | | | +/-N% | PASS/FAIL |
| e2e_latency_us | | | | | +/-N% | |

（无 baseline 时，此表替换为 metric_only 达标判断：带宽利用率 ≥ 20% 或 compute_utilization ≥ 目标值）

---

## 4. 优化记录

> **Round 子节约束**：§4 下每个 Round 必须独占 `### Round N: ...` 子节，包含完整的 5 个子块：正确性复测 → Step 表 → 性能验证对比表 → Device Frame Table → 本轮最优配置 → 决策。**NEVER** 将所有 Round 合并为一个扁平表格。

### Round 1: 机制优化

**机制改动**：<具体代码级改动描述，如并发 peer 通信、分块 put/get>

**瓶颈假设**：<本轮瓶颈假设>

**正确性复测**：调用 shmem-ops-correctness-eval → PASS / FAIL

| Step | 变更 | Correctness | e2e_us | kernel_us | algo_bandwidth_GBps | kernel_bus_bandwidth_GBps | compute_utilization% | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | | | | | | | | |
| 1.2 | | | | | | | | |

> compute_utilization% 列仅通算融合算子填写；纯通信标 N/A

**性能验证**：调用 shmem-ops-performance-eval → 对比基线

| 指标 | Baseline | Round 1 最优 | Δ% (kernel_bus_bandwidth_GBps) |
| --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | |
| e2e_latency_us | | | |
| kernel_latency_us | | | |
| algo_bandwidth_GBps | | | |
| e2e_bus_bandwidth_GBps | | | |
| compute_utilization_pct | | | |

> compute_utilization_pct 行仅通算融合算子填写；纯通信标 N/A

**Device Frame（最优 step）**：

| Frame | Phase | Avg us | Max core us | Count | % E2E | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- |

**本轮最优配置**：

**决策**：keep / revert

---

### Round 2: (机制描述)

**机制改动**：<具体代码级改动描述>

**瓶颈假设**：<本轮瓶颈假设>

**正确性复测**：调用 shmem-ops-correctness-eval → PASS / FAIL

| Step | 变更 | Correctness | e2e_us | kernel_us | algo_bandwidth_GBps | kernel_bus_bandwidth_GBps | compute_utilization% | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 | | | | | | | | |
| 2.2 | | | | | | | | |

> compute_utilization% 列仅通算融合算子填写；纯通信标 N/A

**性能验证**：调用 shmem-ops-performance-eval → 对比基线

| 指标 | Baseline | Round 2 最优 | Δ% (kernel_bus_bandwidth_GBps) |
| --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | |
| e2e_latency_us | | | |
| kernel_latency_us | | | |
| algo_bandwidth_GBps | | | |
| e2e_bus_bandwidth_GBps | | | |
| compute_utilization_pct | | | |

> compute_utilization_pct 行仅通算融合算子填写；纯通信标 N/A

**Device Frame（最优 step）**：

| Frame | Phase | Avg us | Max core us | Count | % E2E | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- |

**本轮最优配置**：

**决策**：keep / revert

---

### Round N: (按 Round 2 格式继续追加，含 kernel_bus_bandwidth_GBps 对比表；无上限)

> 若轮次 < 5 异常停止（如需修改 SHMEM 核心库但无 gap analysis 授权），则剩余轮次标注"未执行 + 原因"。
> 若用户要求额外轮次（> 5），从 Round 6 起按 Round 2 格式继续追加，并在 §5 轮次总览中标注"用户要求的额外优化轮次"。
> Round N 中的"Device Frame（最优 step）"子节在每轮仅填最优 step，与 Round 1/2 格式完全一致。

---

## 5. 轮次总览

| Round | 性质 | 机制改动 | 最优 Step | 最优 kernel_bus_bandwidth_GBps | vs R0 Δ% | 最优 e2e_us | 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | — | — | | — | | — |
| 1 | 机制优化 | <描述> | | | | | |
| 2 | 机制优化 | | | | | | |
| ... | ... | ... | ... | ... | ... | ... | ... |

> Round 对比主指标为 **kernel_bus_bandwidth_GBps**；e2e_us 列仅参考

---

## 6. 最终对比

> SHMEM 最终列 = Phase 6.5 最优轮 keep 配置复采数据；未进入 Phase 6.5 时 = Round 0 SHMEM 数据。
> 外部基线列 = HCCL / aclnn / metric_only（与 §2 baseline type 一致）。
> 始终以**外部基线**为对比基准，**NEVER** 将 SHMEM Round 0 作为 §6 的 Baseline 列。

### 6.1 L 档对比

| 指标 | SHMEM 最终 | <baseline> | SHMEM/基线 | 达标 |
| --- | --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | | PASS/FAIL |
| e2e_latency_us | | | | — |
| kernel_latency_us | | | | — |
| algo_bandwidth_GBps | | | | — |
| e2e_bus_bandwidth_GBps | | | | — |
| bandwidth_utilization_pct | | | | — |
| compute_utilization_pct | | | | — |

> `<baseline>` 列名替换为实际基线名称（如 `HCCL`、`aclnn`、`metric_only`）。
> 达标线：有外部基线时默认 `kernel_bus_bandwidth_GBps ≥ 80% × 基线`；无外部基线时 `bandwidth_utilization_pct ≥ 20%`。
> compute_utilization_pct 行仅通算融合算子填写；纯通信标 N/A。

### 6.2 S 档对比（8PE）

| 指标 | SHMEM 最终 | <baseline> | SHMEM/基线 | 达标 |
| --- | --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | | PASS/FAIL |
| e2e_latency_us | | | | — |
| kernel_latency_us | | | | — |
| algo_bandwidth_GBps | | | | — |
| e2e_bus_bandwidth_GBps | | | | — |
| bandwidth_utilization_pct | | | | — |
| compute_utilization_pct | | | | — |

> 格式与 §6.1 一致。compute_utilization_pct 行仅通算融合算子填写；纯通信标 N/A。

---

## 7. 结论

**达标状态**：达标 / 未达标

**最终配置**：<block_dim=X, chunk_size=Y, ...>

**优化路径总结**：
- Round 1: 机制优化 → (效果)
- Round 2: (机制) → (效果)
- ...

**残留瓶颈**：<如有>

**建议后续方向**：<如有>