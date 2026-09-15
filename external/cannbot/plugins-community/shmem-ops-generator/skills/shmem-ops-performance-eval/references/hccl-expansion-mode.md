# HCCL 算子展开模式（`HCCL_OP_EXPANSION_MODE`）

> **何时执行**：Phase 6 / 6.5 采集 HCCL baseline **之前**（阶段 A）。  
> **目的**：默认 `AI_CPU` 对小消息启动时延偏大；优先切到 `AIV`，与 SHMEM（Device Vector）对比更公平。  
> **官方参考**：[CANN 社区版环境变量 — HCCL_OP_EXPANSION_MODE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/envvar/envref_07_0096.html)

---

## 合法枚举（MUST 原样使用）

| 取值 | 含义 | 本 skill 用法 |
| --- | --- | --- |
| **`AIV`** | 编排与执行均在 Device 侧 Vector Core | **优先**：与 SHMEM AIV 口径对齐 |
| **`AI_CPU`** | 编排在 Device 侧 AI CPU | **回退**（不支持 AIV / 探测失败时） |
| `HOST` | Host CPU 编排 | 不做默认 baseline 模式 |
| `HOST_TS` | Host 编排 + Task Scheduler | 不做默认 baseline 模式 |

> **NEVER** 写成 `aicpu` / `AICPU`；官方字符串是 **`AI_CPU`**（中间有下划线）。

---

## 行为契约（Hard Gate）

```text
1. 阶段 A（仅 HCCL）开始前：
   export HCCL_OP_EXPANSION_MODE="AIV"
2. 执行一次 warmup / smoke（与正式 case 同 shape，iters 可短）
3. 成功 → 全程保持 AIV，日志与报告写 expansion_mode: AIV
4. 失败（init/launch 报错、plog 提示不支持 Vector 直驱、产品/算子不在支持列表）→
   export HCCL_OP_EXPANSION_MODE="AI_CPU"
   重跑 smoke，通过后用 AI_CPU 采正式 baseline
   报告 MUST 写 expansion_mode: AI_CPU 与回退原因
5. SHMEM 采集（阶段 B）不依赖本变量；对比表脚注标明 HCCL 实际模式
```

**写入位置**：

- 运行 log 头：`[BASELINE_MODE] HCCL_OP_EXPANSION_MODE=<AIV|AI_CPU>`
- `performance_report.md` §2 Baseline 详细信息
- Phase 6 聊天对比表脚注（见 [perf-chat-output-spec.md](perf-chat-output-spec.md)）

---

## 支持边界（摘要）

以当前 CANN 文档为准（版本/产品不同支持集可能变化）：

- **常见支持算子（AIV）**：Broadcast、AllReduce、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、ReduceScatter、AllGatherV、ReduceScatterV 等
- **场景限制**：部分产品仅单机 / 单算子模式；跨超节点或特定 Box 可能不支持
- **确定性**：若 `HCCL_DETERMINISTIC=true` 或 `strict`，确定性优先级更高，**AIV 可能不生效** → 报告注明或回退 `AI_CPU`

Agent **MUST** 在该环境实测 smoke，**禁止**仅凭文档假设 AIV 一定可用。

---

## `run_baseline.sh` / 阶段 A 推荐写法

```bash
try_mode() {
  local mode="$1"
  export HCCL_OP_EXPANSION_MODE="${mode}"
  echo "[BASELINE_MODE] HCCL_OP_EXPANSION_MODE=${mode}"
  bash "${OP_DIR}/baseline/scripts/run_baseline.sh" \
    "${DEVICE_LIST}" "${SMOKE_COUNT}" "${DTYPE}" 2 \
    && return 0
  return 1
}

if try_mode "AIV"; then
  ACTIVE_MODE="AIV"
else
  echo "[BASELINE_MODE] AIV unsupported or failed; fallback AI_CPU"
  try_mode "AI_CPU" || { echo "baseline smoke failed"; exit 1; }
  ACTIVE_MODE="AI_CPU"
fi

export HCCL_OP_EXPANSION_MODE="${ACTIVE_MODE}"
# 再跑正式 ITERS 采集 …
```

生成 baseline 脚本时 **SHOULD** 内嵌 prefer-AIV / fallback；至少在 [perf-workflow.md](perf-workflow.md) 阶段 A 前 export，并记录实际模式。

---

## 报告与对比口径

| 场景 | 要求 |
| --- | --- |
| AIV 成功 | 主对比表正常；脚注「HCCL expansion=AIV」 |
| 回退 AI_CPU | **MUST** 脚注：小消息启动时延可能偏大，与 SHMEM 对比偏保守 |
| S 档 | 尤其强调本条；回退时勿仅因 HCCL e2e 偏慢判 SHMEM 达标，看 kernel 口径 |

带宽主指标用 `kernel_bus_bandwidth_GBps`（kernel 口径，达标主指标；见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)）。

---

## 关联文档

- [perf-workflow.md](perf-workflow.md)
- [baseline-selection.md](baseline-selection.md)
- [baseline-compare-workflow.md](baseline-compare-workflow.md)
- [perf-compare-spec.md](perf-compare-spec.md)
- [perf-chat-output-spec.md](perf-chat-output-spec.md)
- [../templates/performance-report.md](../templates/performance-report.md)
