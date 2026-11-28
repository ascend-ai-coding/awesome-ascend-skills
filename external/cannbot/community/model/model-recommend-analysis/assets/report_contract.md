# 分析输出契约

> 本文件由 `model-recommend-analysis` 在输出前读取；主 `SKILL.md` 只保留输出入口与关键约束。

固定两份文件，各保留一份、持续更新，不按轮次新建：

- `models/M<id>/analysis/model_analysis_report.md`：人读、简洁。
- `models/M<id>/analysis/optimization_suggestions.json`：机读、结构化。

## MD 结构

```markdown
# Model Analysis Report - M<id>
- 更新时间 / 当前阶段 / 当前配置 / 约束
## 轮次记录
| 时间 | 阶段 | 来源 | 方法 | 尝试的优化 | 结果 |
## 测量与数据质量
## 总账（wall / kernel_sum / non_kernel / ratio）
## Host / 算子 / gap
## 图结构验证
## 深化分析
- Roofline / 消融 / 动态 shape / 并发 / 精度 / 内存 / 图规范化
## 跨模型与版本对比（如涉及）
## 诊断
- 主瓶颈类别 / 证据 / 排除项 / 置信度
## 实验设计与统计
- 对照方式 / 重复次数 / 噪声带 / 效应量 / 离群点处理
## 建议摘要
| id | 来源 | 交付形态 | 改什么 | 依据 | 收益上界 | 成本风险 | 承接方 | 优先级 | 置信度 |
## 可选实验（不强制）
| 项 | 条件 | 开关 | 预期收益 | 精度风险 |
## 暂缓与越界
```

## JSON 结构（机器读契约，枚举固定）

```json
{
  "schema_version": "1.4",
  "model": "M<id>",
  "updated_at": "<ISO8601>",
  "analysis_id": "M<id>-A-<seq>",
  "constraints": {
    "tp99_ms": null,
    "target": {"metric": "qps|avg_ms|tp99_ms", "value": null, "direction": "max|min"},
    "hbm_gb": null,
    "device_id": 0
  },
  "inputs": {
    "model": "<path>",
    "weights": "<path|null>",
    "dump_graph": {"path": "<path>", "collected_at": "<ISO8601>", "sha256": "<hash>"},
    "profiling": [{"path": "<path>", "collected_at": "<ISO8601>", "sha256": "<hash>"}],
    "config": {"pass_state": "", "die": 0, "threads": 1, "cores": "", "dtype": ""},
    "metrics": {"avg_ms": null, "tp99_ms": null, "qps": null},
    "applied": ["<suggestion-id>"],
    "missing": ["<item>"]
  },
  "ledger": {
    "measurement": {"window": "2/4-3/4", "repeats": 3},
    "quality": {"first_iter_excluded": true, "outlier": false, "cv_pct": null, "removed": [], "note": ""},
    "experiment": {
      "design": "interleaved|adjacent|grouped|unknown",
      "noise_band_pct": null,
      "effect": {"delta_avg_pct": null, "delta_tp99_pct": null, "delta_qps_pct": null}
    },
    "wall_ms": null,
    "kernel_sum_ms": null,
    "non_kernel_ms": null,
    "non_kernel_ratio": null,
    "host": {"launch_ms": null, "sync_ms": null, "memcpy_ms": null, "model_execute_ms": null},
    "top_ops": [{"op_type": "", "core_type": "", "total_ms": null, "count": null, "avg_us": null, "ratio": null}],
    "gap_wait": [{"op": "", "gap_before_ms": null, "gap_after_ms": null, "wait_ms": null}],
    "roofline": {"compute_bound_ms": null, "memory_bound_ms": null, "lower_bound_ms": null, "note": ""},
    "ablation": [{"target": "", "method": "remove|replace|readonly|skip", "delta_ms": null, "note": ""}],
    "dynamic_shape": {"typical": "", "worst": "", "dynamic_dim": "", "tiling_note": ""},
    "concurrency": {"threads": null, "instances": null, "per_instance_avg_ms": null, "per_instance_tp99_ms": null, "saturated": "no|yes|unknown"},
    "bottleneck": "host|sync|dtype|compute|memory|fusion|idle|mixed",
    "confidence": "high|medium|low",
    "excluded": ["<why not other classes>"]
  },
  "rounds": [
    {
      "round": 1,
      "ts": "<ISO8601>",
      "stage": "baseline|post-opt|final",
      "source": "graph|profiling|both",
      "method": ["graph", "profiling"],
      "attempted": ["<suggestion-id>"],
      "result": {"verdict": "improved|no-change|regressed|not-run", "delta": {}},
      "note": ""
    }
  ],
  "suggestions": [
    {
      "id": "OPT-0001",
      "source": "graph|profiling|both",
      "action_type": "pass|plugin|custom-op|runtime_config",
      "needs_human_eval": false,
      "priority": "P0|P1|P2",
      "owner": "model-recommend-plugin-dev|model-recommend-fusion-pass-dev|model-recommend-custom-op-dev|model-recommend-core-limit-stress|workflow",
      "status": "suggested|accepted|rejected|done|blocked",
      "target": {"kind": "graph|op", "ref": "", "pattern": ""},
      "change": {"action": "develop|enable|disable", "value": ""},
      "evidence": [{"source": "<file>", "metric": "<field>", "value": null, "detail": ""}],
      "graph_support": {"matched": false, "detail": ""},
      "expected_gain": {
        "metric": "", "min": null, "max": null, "unit": "%",
        "basis": "evidence|heuristic",
        "confidence": "high|medium|low",
        "upper_bound_note": ""
      },
      "prerequisites": [],
      "risk": {"level": "low|medium|high", "detail": ""},
      "roi": {"cycle": "", "risk": "", "maintenance": "", "verdict": "go|hold|no-go", "note": ""},
      "precision_risk": "none|low|needs-verification",
      "verification": {"method": "", "command": ""},
      "why": "",
      "note": ""
    }
  ],
  "deferred": [
    {"id": "DEF-0001", "reason_code": "noise|not-critical-path|no-structure|needs-decision|roi|unstable|precision-unknown", "note": ""}
  ],
  "termination": {"reached": false, "reason_code": "none|no-action|constraint|missing-data|noise|not-critical-path|no-structure|needs-decision|roi|unstable|precision-unknown|unsafe", "reason": ""},
  "out_of_scope": [{"id": "OOS-0001", "reason": "", "note": ""}]
}
```

## 机器读约定

- 枚举字段只取固定值；自由文本仅放 `why`/`note`/`detail`/`upper_bound_note`；`out_of_scope.reason` 为说明文本。
- `source`：`graph`、`profiling` 或 `both`。
- `action_type`：`pass`（融合/结构优化）、`plugin`（未知/非标算子翻译）、`custom-op`（最后手段，`needs_human_eval=true`）、`runtime_config`（运行/编译开关，如 HF32，`owner=model-recommend-core-limit-stress`）。
- `graph_support.matched=true` 才可作为融合类建议；纯 dtype/算子替换需在 `detail` 说明结构依据。
- 融合类建议必须同时有 profiling 证据与图结构支持。
- `deferred` 记录在范围内但不建议/暂缓的候选；`out_of_scope` 记录超出职责或约束的项。
- `ledger.experiment` 记录对照设计、噪声带与相对效应；无重复实验时对应字段为 `null` 并在 `note` 说明。
- `ledger.roofline`/`ablation`/`dynamic_shape`/`concurrency` 为可选深化分析结果；未做则保持空值，不臆造。
- `ledger.confidence` 与建议内 `expected_gain.confidence` 取 `high|medium|low`；低置信度不得作为实施依据。
- `roi.verdict`：`go`（建议实施）、`hold`（需补充证据或等人决策）、`no-go`（不做）。
- `deferred.reason_code` 固定取值：`noise`、`not-critical-path`、`no-structure`、`needs-decision`、`roi`、`unstable`、`precision-unknown`。
- HF32 的 20% 判定与优先级由主 analysis 负责；`model-recommend-core-limit-stress` 只在限核 Step 1 执行 `--verify`、A/B 与“avg 收益 >5% 且 >噪声带”保留判定。
- 复用规则：读取旧文件，保留已有 `status` 与历史；只追加新建议、更新轮次、账本与终止结论；schema 1.3 → 1.4 仅新增 `owner` 枚举 `model-recommend-core-limit-stress`，旧字段语义不变。
- 同一问题不重复生成建议；已 `done/rejected` 的不再提出，除非证据变化并在 `evidence` 注明。
