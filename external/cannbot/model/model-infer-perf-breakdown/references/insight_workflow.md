# Step 5 — Insight Workflow

## 1. 概述

Step 5 在 Step 3 渲染和 Step 4 切片之后执行，目标是把已有事实产物整理成可审计的性能洞察。它不重新切 layer / component，不改 `network_spec.json`，不把任何未确认的语义规则写死到脚本里。

**输入**：

| 输入 | 来源 | 用途 |
|------|------|------|
| `<run_dir>/runs/<label>/metrics.json` | Step 3 | cluster / bubble / TOTAL / outlier 的统计事实；Step 5 输出跟随同一个 `<label>` |
| `<run_dir>/structure_draft.json` | Step 2 | component instance 的 `phase` / `layer_idx` / `type` / `op_indices` / `op_to_component` |
| `<run_dir>/raw_ops.json` | Step 1 | 单 step 算子序列、core type、时间、stream、shape |
| `<run_dir>/raw_ops_details.json` | Step 1 | 原 CSV 完整字段，用于需要 profiler 原始列的证据 |
| `<run_dir>/<network>_spec.json` | Step 3 | cluster 语义规则；只作为已确认语义来源 |
| `<run_dir>/splits/manifest.json` | Step 4 | evidence link：component / gap 的 csv 与 trace 子目录 |
| `op_statistic_*.csv` | 原始 prof，可选 | agent 判断数据搬运 taxonomy 时查看 OP Type 总览 |

**默认输出目录**：`<run_dir>/runs/<label>/insights/`。Step 5 必须和它消费的 `<run_dir>/runs/<label>/metrics.json` 归属同一个 label，避免多个 run 的洞察互相覆盖。只有没有 `runs/<label>/metrics.json` 的老式单 run 产物，才允许退回 `<run_dir>/insights/`。

**默认输出**只落下列文件，不再新增长期中间文件：

| 输出 | 生成者 | 内容 |
|------|--------|------|
| `<run_dir>/runs/<label>/insights/module_bubble.json` | 脚本 + agent | 模块长期 bubble 偏大、具体层/模块 bubble 异常 |
| `<run_dir>/runs/<label>/insights/operator_jitter.json` | 脚本 + agent | cluster / operator 性能抖动候选 |
| `<run_dir>/runs/<label>/insights/theoretical_deviation.json` | 脚本 + agent | 逻辑 sub-item / TOTAL / operator slot 与理论性能的偏离 |
| `<run_dir>/runs/<label>/insights/vector_sequence_candidates.json` | 脚本 + agent | 连续 vector 区间，按 pattern 聚合并给代表样本 |
| `<run_dir>/runs/<label>/insights/data_movement_ops.json` | agent + 脚本 | 冗余 Tensor Movement 消除候选：搬运 / 结构变换 taxonomy、定位、上下文、消除方向 |
| `<run_dir>/runs/<label>/insights/final_conclusions.md` | agent | 最终结论、证据链接、限制说明 |
| `<run_dir>/runs/<label>/insights/index.html` | 脚本 | 五类 insight 的可视化页面 |
| `<run_dir>/runs/<label>/insights/main_report_insights.json` | agent | 回填主报告 `insight` 列的 high / medium 洞察 |

**执行者边界**：

| 执行者 | 做什么 | 不做什么 |
|--------|--------|----------|
| 脚本 | join、group-by、排序、区间扫描、统计分布、选择代表样本、渲染 HTML | 不判断模型语义，不内置搬运/融合 op 规则，不给性能原因 |
| agent | 读取事实候选，审阅理论偏离可信度，定义本模型 taxonomy，判断是否值得展示，写语义解释和结论 | 不手工匹配全量算子，不静默接受脚本默认语义 |
| 用户 | 必要时确认 taxonomy 或解释口径 | 不需要逐层逐算子参与 |

**脚本落地**：事实提取与 HTML 渲染由 `scripts/gen_insights.py` + `scripts/render_insights.py` 实现，agent **不要**手写提取逻辑。两遍流程：

1. `gen_insights.py`（pass 1，无 `--annotations`/`--movement-taxonomy`）→ 五类 JSON（`agent_review` 字段全空）+ `_review_stub.json`（所有待审 stable key）+ `_taxonomy_candidates.json`（op×core 频次）。
2. agent 据 `_taxonomy_candidates.json` 写 `movement_taxonomy.json`（movement family + `movement_kind`），据 `_review_stub.json` 写 `annotations.json`（每条 `agent_review`：`summary` / `confidence` `high|medium|low` / `reason` / `fusion_candidate` / `elimination_direction`）。
3. `gen_insights.py`（pass 2，带 `--movement-taxonomy` + `--annotations`）→ 合并 agent 判断的五类 JSON。
4. agent 写 `final_conclusions.md`；`render_insights.py` 读五类 JSON + `final_conclusions.md` + 模板 → `index.html`（按 confidence high→medium→low 排序、Top5）。

CLI：

```bash
# pass 1：产事实候选（agent_review 留空）+ _review_stub.json + _taxonomy_candidates.json
python scripts/gen_insights.py \
  --metrics <run_dir>/runs/<label>/metrics.json \
  --raw-ops <run_dir>/raw_ops.json --structure-draft <run_dir>/structure_draft.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --out-dir <run_dir>/runs/<label>/insights

# pass 2：合并 agent 写好的 movement_taxonomy.json + annotations.json
python scripts/gen_insights.py ...（同 pass 1）\
  --movement-taxonomy <run_dir>/runs/<label>/insights/movement_taxonomy.json \
  --annotations <run_dir>/runs/<label>/insights/annotations.json

# 渲染洞察页（需 final_conclusions.md 已写）
python scripts/render_insights.py --insights-dir <run_dir>/runs/<label>/insights \
  --model-name "<name>" --label <label> -o <run_dir>/runs/<label>/insights/index.html

# 回填主报告 insight 列
python scripts/render.py -d <run_dir>/structure_draft.json -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json -s <run_dir>/<network>_spec.json \
  --insight-annotations <run_dir>/runs/<label>/insights/main_report_insights.json \
  --label <label> -o <run_dir>/runs/<label>/index.html
```

脚本按**绝对增量**(gap/delta/total_us)排序候选；`confidence` 与所有语义只来自 agent 注解，缺省留空。被 agent 注解过的候选即使不在脚本事实 Top-K 内也会强制保留（agent 可越过脚本排序选要展示项）。`bound_type` 可能为空（取决于理论 skill 版本），不据此判 compute/memory bound。排序口径、`op_records` 反查等实现细节见 `gen_insights.py` docstring。

## 2. 重要原则

1. **事实和语义分离**：脚本产出的数值必须只来自 `metrics.json` / `raw_ops.json` / `structure_draft.json` / 原始 prof；“这个 op 是搬运”“这个区间像融合候选”“这个模块在做什么”必须来自 `network_spec.json`、agent 审阅或用户确认。
2. **不写死模型结构**：不同模型的 component 名、cluster 名、op 名都可能不同。Step 5 只能使用当前 run 已经确认过的 `component_type` / cluster，不能假设一定有 attention / mlp / moe。
3. **同类比较优先**：bubble 和抖动默认只在同一 `component_type` 或同一 cluster/op 语义内比较。跨 component_type 的排序只能作为耗时排行榜，不能自动判异常。
4. **单 step 与跨 step 分清**：当前 sample-driven 分支的 `raw_ops.json` 是单 step。如果用户要求“同一算子跨 step 抖动”，必须回到完整 `kernel_details.csv` 设计多 step 对齐；不能用单 step 结果伪装成跨 step。
5. **Top5 而不是刷屏**：每类 insight 的每个数据数组都默认只输出 / 展示 Top5。理论偏离、连续 vector 和数据搬运类 insight 仍按 logical slot / pattern / family / module 聚合，不每层每 step 展开；长尾只保留在原始 JSON 的统计计数或 evidence 链接里，不在 Markdown / HTML 主视图铺开。
6. **证据可回溯**：每条 insight 至少带 `component_type`、`phase`、`layer_idx`、`op_indices`（或展示用 `op_range_envelope`）/ `org_index`、代表样本、可选 split 路径。Tensor movement 候选还必须带前后邻接算子上下文。
7. **不确定要显式**：agent 对语义解释给 `confidence: high|medium|low`，不能确定时用 `low` 并在 `needs_user_review` 或 `reason` 里说明，不要编原因。HTML 和 Markdown 展示必须按 `high -> medium -> low` 排序；同置信度内按耗时、占比或预期收益降序。
8. **不修改原始产物**：Step 5 只写 `<run_dir>/runs/<label>/insights/`（或老式 fallback `<run_dir>/insights/`），不改原 prof、`raw_ops.json`、`structure_draft.json`、`network_spec.json`。

## 3. Insight 1：模块 Bubble

本 insight 输出一个 JSON，里面包含两类榜单：

1. **模块长期 bubble 偏大**：排序对象是 `component_type`，回答“这个模块类型是不是整体 bubble 高”。
2. **具体层/模块 bubble 异常大**：排序对象是 `(phase, layer_idx, component_type)`，回答“第几层的哪个模块 bubble 特别大”。

**脚本输入**：`metrics.json`。

**脚本操作**：

- 对每个 `section.component_type` 找 `sub_items[kind=bubble].per_instance_ms`。
- 结合 `section.total.per_instance_ms` 计算 `bubble_pct_of_total`。
- 对模块类型计算 `median_bubble_us`、`p90_bubble_us`、`max_bubble_us`、`median_bubble_pct_of_total`、`high_bubble_instance_ratio`。
- 对具体 instance 只和同一 `component_type` 的 bubble 分布比较，计算 `delta_vs_type_median_us` / `ratio_vs_type_median`。
- 选择 Top5 代表样本；`selection_policy.top_k` 必须为 `5`，不要隐藏在脚本里。

**agent 操作**：

- 判断模块长期 bubble 是“整体偏高”还是“少数层拉高”。
- 对具体层/模块异常，只描述观察到的形态；如果需要解释原因，必须引用 split trace、相邻 op 或已有 cluster 证据。
- 在 `agent_review` 字段写 `summary`、`confidence`、`needs_followup`。

**输出 schema**：

```json
{
  "schema_version": "insight.module_bubble.v1",
  "selection_policy": {"top_k": 5, "baseline": "same_component_type"},
  "persistent_module_bubbles": [
    {
      "component_type": "moe",
      "instance_count": 30,
      "median_bubble_us": 12.3,
      "p90_bubble_us": 13.0,
      "median_bubble_pct_of_total": 4.2,
      "high_bubble_instance_ratio": 0.8,
      "representative_instances": [
        {"phase": "main", "layer_idx": 21, "bubble_us": 13.0, "op_indices": [1000, 1004, 1028], "op_range_envelope": [1000, 1028]}
      ],
      "agent_review": {"summary": "", "confidence": "medium"}
    }
  ],
  "instance_bubble_outliers": [
    {
      "phase": "main",
      "layer_idx": 21,
      "component_type": "moe",
      "bubble_us": 13.0,
      "bubble_pct_of_total": 4.6,
      "delta_vs_type_median_us": 1.2,
      "evidence": {"op_indices": [1000, 1004, 1028], "op_range_envelope": [1000, 1028], "split_dir": "splits/L021_main_moe"},
      "agent_review": {"summary": "", "confidence": "medium"}
    }
  ]
}
```

## 4. Insight 2：算子性能抖动

本 insight 用于发现同类 cluster / operator 中耗时异常长的候选。当前分支默认分析的是**单 step 内跨 layer/component instance 的抖动**；如果要跨 step 抖动，必须另行读取完整 `kernel_details.csv` 做多 step 对齐。

**脚本输入**：`metrics.json`，可选 `raw_ops_details.json`。

**脚本操作**：

- 从 `sections[].sub_items[kind=cluster]` 读取 `per_instance_ms`、`outlier_idx`、`kernel_outliers`。
- **cluster jitter 必须使用 `per_instance_ms`**：它来自 `render.py` 的 cluster wall，即该 cluster 内算子的 timeline interval union。禁止用 `op_records` 或 raw kernel `duration_us` 相加来计算 cluster jitter。
- `op_records` 只作为 evidence 索引用；它不能代表 cluster wall。
- cluster 级别：对 `(component_type, cluster)` 的 instance 分布计算 median / p95 / max / outlier delta。
- operator 级别：优先使用已有 `kernel_outliers`；该层级表示单 kernel duration 抖动，不是 wall。若字段不足，只输出 cluster 级候选，不伪造 operator 粒度。
- 排序使用绝对耗时增量和相对增量两类字段；排序策略写入 `selection_policy`。

**agent 操作**：

- 将候选解释为“统计抖动候选”，不要直接写成根因。
- 结合 `network_spec.json` 的 cluster 描述说明这个候选属于哪个语义步骤。
- 如果候选来自首层/尾层/特殊 layer，必须标注“可能是结构差异，不一定是异常”。

**输出 schema**：

```json
{
  "schema_version": "insight.operator_jitter.v1",
  "selection_policy": {
    "baseline": "same_component_type_and_cluster",
    "top_k": 5,
    "cluster_metric_source": "metrics.sections[].sub_items[kind=cluster].per_instance_ms",
    "cluster_metric_semantics": "cluster wall = interval union, not sum(duration_us)"
  },
  "cluster_jitter_candidates": [
    {
      "component_type": "dense_mlp",
      "cluster": "down_proj",
      "phase": "main",
      "layer_idx": 0,
      "duration_us": 38.3,
      "baseline_median_us": 22.5,
      "delta_us": 15.8,
      "ratio_vs_baseline": 1.7,
      "evidence": {"op_indices": [38, 39, 43], "op_range_envelope": [38, 43]},
      "agent_review": {"summary": "", "confidence": "medium"}
    }
  ],
  "operator_jitter_candidates": [
    {
      "component_type": "moe",
      "cluster": "expert_gate_up",
      "op_name": "GroupedMatmul",
      "occurrence": 0,
      "phase": "main",
      "layer_idx": 7,
      "duration_us": 49.6,
      "baseline_median_us": 25.6,
      "agent_review": {"summary": "", "confidence": "medium"}
    }
  ]
}
```

## 5. Insight 3：理论性能偏离

本 insight 用于发现实测耗时明显偏离理论性能的逻辑对象。主排序对象是**逻辑 slot**，不是某个单独 `(step, layer)` 实例；具体层 / 实例只作为代表证据和定位入口。

Step 1.5（理论性能注入）是**可选**步骤，默认不开（外部 `operator-theoretical-perf` skill 可能不存在）。**未启用时**：`raw_ops_details.json` 无理论列、`metrics.json` 无 `theoretical` 摘要，本 insight 直接输出空 JSON 并在 `data_limits` 注明"理论性能未启用"，**正常跳过即可**（不必回 Step 1.5）。**启用时**：`raw_ops_details.json` 含理论列、`metrics.json` 带 `theoretical` 摘要，按下文产出偏离候选。

**排序对象**：

1. **sub-item / TOTAL 逻辑 slot**：`(component_type, sub_item)`，其中 `sub_item` 可以是 cluster 名、`TOTAL` 或 renderer 已确认的模块子项。回答“这个逻辑模块整体比理论慢多少”。
2. **operator 逻辑 slot**：`(component_type, cluster, op_name, occurrence)`。回答“同一语义算子位置是否经常比理论慢”。
3. **单例实例**：`(phase, layer_idx, component_index, org_index)` 只放在 `top_locations` / `representative_instances`，不能作为主榜单行。

**脚本输入**：

- `<run_dir>/runs/<label>/metrics.json`
- `<run_dir>/raw_ops_details.json`，必须已合并 `theoretical_operator_time_us`、`duration_over_theoretical`、`duration_analysis`、`bound_type`、`theory_supported`
- `<run_dir>/raw_ops.json`
- `<run_dir>/structure_draft.json`
- `<run_dir>/<network>_spec.json`

**脚本操作**：

- 从 `metrics.sections[].sub_items[].theoretical` 和 `metrics.sections[].total.theoretical` 读取理论摘要。
- 对每个 sub-item / TOTAL 逻辑 slot 计算：
  - `actual_median_ms`
  - `theoretical_median_ms`
  - `wall_over_theoretical_median`
  - `absolute_gap_us = (actual_median_ms - theoretical_median_ms) * 1000`
  - `supported_instance_count` / `instance_count` / `support_ratio`
  - `representative_instances`，只保留 Top5 证据位置。
- 对 operator 逻辑 slot，从 `raw_ops_details.json` + `raw_ops.json` + `structure_draft.json` 反查所属 component / cluster，按 `(component_type, cluster, op_name, occurrence)` 聚合：
  - `duration_over_theoretical_median`
  - `absolute_gap_us_median`
  - `max_duration_over_theoretical`
  - `supported_count`
  - `top_locations`，只保留 Top5 单例证据。
- 脚本候选排序使用 `wall_over_theoretical_median` / `duration_over_theoretical_median` 和 `absolute_gap_us` 两类字段，并把实际排序策略写入 `selection_policy`；`confidence` 只能由 agent 审阅后填写。
- 不支持理论估算的 kernel、通信、runtime、host-bound 辅助项可以统计到 `unsupported_summary`，但不要伪造理论偏离。

**agent 操作**：

- 对每个候选判断理论数据是否可信，写入 `agent_review.summary`、`confidence`、`reason`、`needs_followup`。
- 多流 sub-item 需要检查 Step 3 的理论流选择口径：若理论列取的是覆盖范围大的流，必须在 `semantic_note` 或 `reason` 里说明“被 overlap 的流没有重复计入理论耗时”。
- 对 fused/custom kernel、shape mismatch、`theory_supported=false` 占比高、通信/host/runtime 项，不要强行写成性能问题；用 `confidence: low` 或放入 `data_limits`。
- 结论措辞必须是“偏离理论性能候选”，不能直接写成根因。需要源码、shape 或 trace 复核时，写 `needs_followup: true`。

**输出 schema**：

```json
{
  "schema_version": "insight.theoretical_deviation.v1",
  "selection_policy": {
    "top_k": 5,
    "ranking_object": "logical_slot",
    "singleton_instances": "evidence_only",
    "sub_item_sort": ["wall_over_theoretical_median", "absolute_gap_us"],
    "operator_slot_sort": ["duration_over_theoretical_median", "absolute_gap_us_median"]
  },
  "data_limits": [],
  "sub_item_deviation_candidates": [
    {
      "component_type": "moe",
      "sub_item": "expert",
      "actual_median_ms": 0.53,
      "theoretical_median_ms": 0.31,
      "wall_over_theoretical_median": 1.71,
      "absolute_gap_us": 220.0,
      "support_ratio": 0.93,
      "representative_instances": [
        {"phase": "main", "layer_idx": 4, "actual_ms": 0.55, "theoretical_ms": 0.32, "op_indices": [1200, 1210, 1260], "op_range_envelope": [1200, 1260]}
      ],
      "agent_review": {"summary": "", "confidence": "medium", "reason": "", "needs_followup": true}
    }
  ],
  "operator_slot_deviation_candidates": [
    {
      "component_type": "moe",
      "cluster": "expert",
      "op_name": "GroupedMatmul",
      "occurrence": 0,
      "duration_over_theoretical_median": 1.86,
      "absolute_gap_us_median": 42.0,
      "supported_count": 9,
      "top_locations": [
        {"phase": "main", "layer_idx": 4, "org_index": 3221, "duration_us": 118.0, "theoretical_us": 63.0, "bound_type": "compute"}
      ],
      "agent_review": {"summary": "", "confidence": "medium", "reason": "", "needs_followup": true}
    }
  ],
  "unsupported_summary": []
}
```

## 6. Insight 4：连续 Vector 区间融合候选

本 insight 只列出大段连续 vector 区间候选，并按 pattern 聚合；不每层每 step 列一条。是否称为“缺失融合候选”由 agent 根据语义审阅后写入，不由脚本默认判断。

**脚本输入**：`raw_ops.json`、`structure_draft.json`、`network_spec.json`。

**脚本操作**：

- 按 `raw_ops.operators[].index` 顺序扫描连续 `accelerator_core == "AI_VECTOR_CORE"` 的区间。
- 对每个区间记录 `op_indices`、展示用 `op_range_envelope`、`org_index_range`、`duration_us`、`stream_ids`、`normalized_name` 序列。
- 通过 `structure_draft.op_to_component` 定位所属 component；通过 `network_spec.json` 尽力映射到 cluster。映射不到时写 `cluster: "unknown"`。
- 将连续区间压成 pattern signature，例如 `RmsNorm x3 -> StridedSliceD x8 -> Mul -> ...`，按 signature 聚合。
- 只输出 Top5 patterns 和每个 pattern 的少量代表样本；`min_ops`、`min_total_us`、`top_k` 必须写入 `selection_policy`。

**agent 操作**：

- 审阅每个 pattern，写明“这个模块在做什么语义”，语义来源优先级为 `network_spec` cluster 描述 > component type > op/shape 观察。
- 判断是否是融合候选：`fusion_candidate: true|false|uncertain`。
- 对 `uncertain` 给出需要用户/工程师确认的点。

**输出 schema**：

```json
{
  "schema_version": "insight.vector_sequence_candidates.v1",
  "selection_policy": {"min_ops": 5, "min_total_us": 20.0, "top_k": 5, "group_by": "pattern_signature"},
  "patterns": [
    {
      "pattern_signature": "RmsNorm x3 -> StridedSliceD x8 -> Mul -> ...",
      "occurrences": 5,
      "total_duration_us": 365.0,
      "typical_duration_us": 73.0,
      "components": ["full_attn"],
      "clusters": ["partial_rope", "unknown"],
      "representative_samples": [
        {
          "phase": "main",
          "layer_idx": 23,
          "component_type": "full_attn",
          "cluster": "partial_rope",
          "op_indices": [857, 860, 880],
          "op_range_envelope": [857, 880],
          "duration_us": 73.6
        }
      ],
      "agent_review": {
        "semantic_summary": "",
        "fusion_candidate": "uncertain",
        "confidence": "medium"
      }
    }
  ]
}
```

## 7. Insight 5：冗余 Tensor Movement 消除候选

本 insight 的目的不是泛泛列出搬运算子，而是帮助工程师发现**可能可以消除的冗余 tensor movement / layout movement / shape movement**。输出要列出搬运 / 结构变换类算子的所属模块、耗时、前后上下文、语义解释，以及 agent 对“是否可能消除”的判断。

脚本只能给事实：哪些 op、在哪、耗时多少、前后邻接是什么。agent 才能判断它是否像冗余 movement 候选，并给出可能的消除方向。不能因为 op 名包含 concat / split / transpose / slice 就自动判为可消除。

**agent 先做 taxonomy**：

- 读取 `op_statistic_*.csv`、`raw_ops.json` 中的 OP Type / `normalized_name` 总览。
- 结合 `network_spec.json` 的 cluster 语义，选择当前模型中的 movement family，例如 slice / concat / split / transpose / gather / scatter / tensor move / transdata / reshape-like。
- 每个 family 必须写 `ops`、`reason`，并区分 `movement_kind`：`layout` / `shape` / `indexing` / `cache_update` / `communication_prep` / `unknown`。
- 不确定的 op 放进 `needs_user_review`，不要静默纳入。

**脚本再做定位统计**：

- 根据 `agent_taxonomy.families[].ops` 在 `raw_ops.json` 中定位算子。
- 用 `structure_draft.op_to_component` / `components[].op_indices` 反查所属 component；用 `network_spec.json` 尽力反查 cluster。
- 为每个命中 op 记录前后邻接上下文：同 stream 前后 op、全局序列前后 op、所在连续 vector 区间、是否靠近 cube/communication op。
- 按 `family -> op_name -> component_type -> cluster` 聚合 count、total_us、typical_us、top locations、representative contexts；主输出只保留 Top5 family/module 聚合项，每项内部代表位置也最多 Top5。

**agent 最后写冗余消除可能性判断**：

- 对每个 family + module 写一句“在该模块里大概率做什么”。
- 用 `confidence: "high" | "medium" | "low"` 表示这个 tensor movement **作为可消除冗余的可能性**：
  - `high`：强候选，比较像可通过融合、layout 传递或消除中间张量去掉。
  - `medium`：可能可消除，但需要看源码 / trace / shape 约束确认。
  - `low`：不太像可消除，或证据不足；例如 cache 写入、通信准备、采样后处理等可能是必要语义的 movement。
- 标注 `elimination_direction`：`fuse_with_previous` / `fuse_with_next` / `fuse_sequence` / `layout_propagation` / `remove_intermediate` / `needs_code_check` / `none`。
- 若只能确认是结构变换，但不知道语义，写 `semantic_summary: "unknown"`，并将 `confidence` 置为 `low`。

**输出 schema**：

```json
{
  "schema_version": "insight.data_movement_ops.v1",
  "agent_taxonomy": {
    "families": [
      {
        "family": "slice",
        "ops": ["StridedSliceD"],
        "reason": "observed as repeated shape/index slicing in attention preparation",
        "movement_kind": "indexing"
      }
    ],
    "needs_user_review": []
  },
  "families": [
    {
      "family": "slice",
      "ops": ["StridedSliceD"],
      "total_duration_us": 180.0,
      "occurrences": 40,
      "by_module": [
        {
          "component_type": "full_attn",
          "cluster": "partial_rope",
          "total_duration_us": 175.0,
          "occurrences": 40,
          "top_locations": [
            {
              "phase": "main",
              "layer_idx": 23,
              "duration_us": 35.6,
              "op_indices": [857, 860, 880],
              "op_range_envelope": [857, 880],
              "neighbor_context": {
                "prev_same_stream": "RmsNorm",
                "next_same_stream": "Mul",
                "prev_global": "Fill",
                "next_global": "Mul",
                "vector_sequence_signature": "RmsNorm x3 -> StridedSliceD x8 -> Mul"
              }
            }
          ],
          "agent_review": {
            "semantic_summary": "",
            "elimination_direction": "needs_code_check",
            "reason": "",
            "confidence": "medium"
          }
        }
      ]
    }
  ]
}
```

## 8. HTML 渲染

Step 5 HTML 只渲染已落盘的五个 JSON 和 `final_conclusions.md`，不在前端重新计算隐含规则。HTML 主视图和 Markdown 结论都只展示每个数据数组的 Top5，避免长表刷屏。

**输入**：

- `<run_dir>/runs/<label>/insights/module_bubble.json`
- `<run_dir>/runs/<label>/insights/operator_jitter.json`
- `<run_dir>/runs/<label>/insights/theoretical_deviation.json`
- `<run_dir>/runs/<label>/insights/vector_sequence_candidates.json`
- `<run_dir>/runs/<label>/insights/data_movement_ops.json`
- `<run_dir>/runs/<label>/insights/final_conclusions.md`

**输出**：`<run_dir>/runs/<label>/insights/index.html`。

**回填 Step 3 主报告**：

Step 3 首次生成的 `<run_dir>/runs/<label>/index.html` 必须已经包含最后一列 `insight`，但默认留空。Step 5 最后由 agent 把五类 insight 的 high / medium 条目回填主报告，并标注 insight 类别，写：

```text
<run_dir>/runs/<label>/insights/main_report_insights.json
```

schema：

```json
{
  "schema_version": "main_report_insights.v1",
  "items": [
    {
      "category": "operator_jitter",
      "target": {
        "component_type": "moe",
        "sub_item": "expert",
        "mapping_type": "direct",
        "mapping_note": "",
        "related_targets": []
      },
      "confidence": "high",
      "summary": "MoE expert cluster 在 MoE-4 上持续偏慢，建议优先打开对应 split trace。",
      "source": "operator_jitter.json",
      "evidence": "main#4 / op_indices [1200, 1210, 1260] / envelope [1200, 1260]"
    }
  ]
}
```

约束：

- `category` 必须标注 insight 类别，只允许：`module_bubble` / `operator_jitter` / `theoretical_deviation` / `vector_sequence` / `tensor_movement`。
- `target.component_type` 必须匹配主报告 section。
- `target.sub_item` 必须匹配主报告行名：cluster 名、`bubble` 或 `TOTAL`。
- `target.mapping_type` 只允许 `direct` / `agent_primary`。能精准落到单个 row 时用 `direct`；无法精准映射到单个模块时，由 agent 判断最能代表该 insight 的主要模块 / 主要 row，用 `agent_primary`，并在 `mapping_note` 说明为什么放在这里。
- `TOTAL` 只是可选主归属之一，不是兜底默认值。只有 insight 描述的是整个 component、整体耗时、整体 bubble/overlap，或无法合理归入任一具体 cluster 时，才把 `target.sub_item` 设为 `TOTAL`；否则优先选择最相关的具体 cluster / `bubble` 行。
- `target.related_targets` 可列出被该 insight 影响但不作为主归属的其它 row；渲染器只在主 row 展示，不自动复制到相关 row。
- 同一 `target` 可以对应多条 high / medium insight；渲染器必须全部展示，并按 high -> medium、类别、summary 排序，不把多条 insight 合并成一条。
- `confidence` 只允许 `high` / `medium`；`low` 不回填到主报告。
- `summary` 必须由 agent 填写，脚本不得从五个 insight JSON 自动推断。

写完后重跑主报告渲染：

```bash
python scripts/render.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --insight-annotations <run_dir>/runs/<label>/insights/main_report_insights.json \
  -s <run_dir>/<network>_spec.json \
  --label <label> \
  -o <run_dir>/runs/<label>/index.html
```

**页面结构**：

1. 常驻通用结论区：显示最终结论、数据来源、限制说明、证据索引；始终可见。
2. 五个选项卡页面：`Module Bubble` / `Operator Jitter` / `Theoretical Deviation` / `Vector Sequences` / `Tensor Movement`。
3. Module Bubble 页：分成“模块长期 bubble 偏大 Top5”和“具体层 / 模块异常 bubble Top5”两个表。
4. Operator Jitter 页：展示 cluster / operator 抖动候选 Top5，强调这是统计候选，不是根因结论。
5. Theoretical Deviation 页：按 sub-item / TOTAL logical slot 和 operator logical slot 展示理论性能偏离 Top5，单例只作为 evidence。
6. Vector Sequences 页：按 pattern 聚合展示连续 vector 区间 Top5，列代表样本与 agent 语义解释。
7. Tensor Movement 页：按 family / module 展示搬运与结构变换算子 Top5，列 taxonomy 来源、前后上下文、冗余消除可能性和方向。

**HTML 模板约束**：

完整 HTML/CSS/JS 骨架放在 `references/insight_html_template.html`，不要 inline 到本文档。渲染脚本 `scripts/render_insights.py` 以该模板为基础、服务端预填各表生成 `<run_dir>/runs/<label>/insights/index.html`（模板 JS 只切 tab，不在前端 fetch/算数）。

模板固定以下结构：

- 常驻通用结论区：核心结论、数据限制、证据索引，始终可见。
- 五个选项卡页面：`Module Bubble` / `Operator Jitter` / `Theoretical Deviation` / `Vector Sequences` / `Tensor Movement`。
- 每个选项卡使用当前 JSON 产物填充表格；每个表默认只渲染 Top5 行，可以动态隐藏空内容，但不能改掉五个页面的含义。
- `component_type` / cluster / family 的显示名来自当前产物，不做固定枚举。
- 所有“可优化 / 可消除”措辞必须来自 agent review 字段，HTML 渲染层不能自行推断。

**渲染弹性**：

- 某个 JSON 为空时，对应选项卡页面保留标题并显示“本次未发现候选”，不要删除 tab。
- 表格列可以增减，但每行必须保留：对象名称、所属模块、耗时/比例、代表位置、agent 解释、confidence、evidence。
- 所有 HTML 表格和 Markdown 结论都按 `high -> medium -> low` 显示；同一 confidence 内按耗时、占比、delta 或预期收益降序。
- 所有 HTML 表格和 Markdown Top5 表都必须标注“Top5”。如果原 JSON 候选不足 5 条，展示实际条数并保留空状态说明；如果原 JSON 超过 5 条，不在主视图继续展示第 6 条以后。
- 模型没有某类结构时，不要补假数据；例如没有 MoE 就不能渲染 MoE 专属卡片。
- `component_type` / cluster / family 的显示名来自当前产物，不做固定枚举。
- 所有“可优化 / 可消除”措辞必须来自 agent review 字段，脚本渲染层不能自行推断。

**`final_conclusions.md` 模板**：

```markdown
# 性能洞察结论

## 核心结论

> 本节必须按 high -> medium -> low 分组展示；组内按耗时、占比、delta 或预期收益降序。

### High

- [high] <一句话结论>
  Evidence: `<json file>` / component / layer / op_indices or op_range_envelope / split dir

### Medium

- [medium] <一句话结论>
  Evidence: `<json file>` / component / layer / op_indices or op_range_envelope / split dir

### Low

- [low] <一句话结论>
  Evidence: `<json file>` / component / layer / op_indices or op_range_envelope / split dir

## 数据限制

- 当前是否仅基于单 step。
- 哪些 taxonomy 或语义解释需要用户确认。
- 哪些候选只是统计异常，不代表根因。

## 建议后续动作

- 优先查看的 split trace。
- 建议补充的多 step / 多 rank 数据。
- 值得进一步验证的融合或搬运优化候选。

## Top5 证据表

> 本节不是自由发挥摘要。agent 必须按下列数据数组逐项列 Top5；每张表按 `high -> medium -> low`、同置信度内按对应数值降序。候选不足 5 条则写实际条数；没有候选则写“本次未发现候选”。

### Module Bubble / persistent_module_bubbles Top5

| rank | component_type | metric | representative location | confidence | agent summary | evidence |
|------|----------------|--------|-------------------------|------------|---------------|----------|

### Module Bubble / instance_bubble_outliers Top5

| rank | phase/layer/component | bubble_us / pct | delta vs baseline | confidence | agent summary | evidence |
|------|-----------------------|-----------------|-------------------|------------|---------------|----------|

### Operator Jitter / cluster_jitter_candidates Top5

| rank | component_type / cluster | wall_us | delta / ratio | confidence | agent summary | evidence |
|------|--------------------------|---------|---------------|------------|---------------|----------|

### Operator Jitter / operator_jitter_candidates Top5

| rank | component_type / cluster / op | duration_us | delta / ratio | confidence | agent summary | evidence |
|------|-------------------------------|-------------|---------------|------------|---------------|----------|

### Theoretical Deviation / sub_item_deviation_candidates Top5

| rank | component_type / sub_item | actual median / theory median | ratio / gap | confidence | agent summary | representative evidence |
|------|---------------------------|-------------------------------|-------------|------------|---------------|-------------------------|

### Theoretical Deviation / operator_slot_deviation_candidates Top5

| rank | component_type / cluster / op | duration/theory | gap | confidence | agent summary | representative evidence |
|------|-------------------------------|-----------------|-----|------------|---------------|-------------------------|

### Vector Sequences / patterns Top5

| rank | pattern_signature | occurrences / total_us | modules / clusters | confidence | semantic summary | representative evidence |
|------|-------------------|------------------------|--------------------|------------|------------------|-------------------------|

### Tensor Movement / families Top5

| rank | family / module | count / total_us | movement_kind | confidence | redundancy possibility / direction | representative evidence |
|------|-----------------|------------------|---------------|------------|------------------------------------|-------------------------|
```
