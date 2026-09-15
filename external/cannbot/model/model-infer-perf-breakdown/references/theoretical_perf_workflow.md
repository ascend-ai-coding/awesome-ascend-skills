# Step 1.5 — 理论性能注入（可选，默认不开）

**本步骤可选，默认不跑**：它依赖外部 `operator-theoretical-perf` skill，该 skill 可能不存在。不启用时无需任何额外动作——`render.py` 的 theory median 列自动留空（`—`）、不阻塞，Step 5 理论偏离 insight 自动跳过。仅当用户要理论对比、且外部 skill 可用时按本文启用。

启用后，本步骤把逐 kernel 理论性能接入当前 sample-driven breakdown。当前 skill **不实现**理论性能估算公式，也不复制 `operator-theoretical-perf` skill 的脚本或 TokenSim 内容；agent 只用自然语言调用该 skill，让它按原生方式给 `kernel_details.csv` 加理论性能列。

## 1. 入口位置

```text
Step 1: kernel_details.csv -> raw_ops.json + raw_ops_details.json
Step 1.5: agent 调用 operator-theoretical-perf -> operator_analysis.csv
Step 1.5: merge_theoretical_columns.py -> raw_ops_details.json 增加理论列
Step 3: render.py --raw-ops-details ... -> index.html 增加 theory median 列
```

Step 1.5 可选；启用后 Step 3 渲染会从 `raw_ops_details.json` 读取已合并的理论列。

## 2. Agent 如何调用理论分析 skill

agent 确认 chip 配置后，调用外部 `operator-theoretical-perf` skill，让它以当前 run 对应的原始 `kernel_details.csv` 为输入，输出：

```text
<run_dir>/operator_analysis.csv
```

agent 提示中必须说明：

- 输入是当前 breakdown 使用的同一份 `kernel_details.csv`。
- 外部 skill 按它自己的原生流程生成 `operator_analysis.csv`，不要要求它额外输出 JSON。
- `duration / theoretical` 的方向必须是实测耗时除以理论耗时。
- 不支持估算的 kernel 不要编造理论耗时。
- 当前 skill 后续会按原 CSV 行序 / `org_index` 把理论列合并回 `raw_ops_details.json`。

## 3. 合并理论列

`operator_analysis.csv` 是完整 CSV，不直接进入渲染。当前 skill 用轻量脚本把理论列合并到 per-kernel detail carrier：

```bash
python scripts/merge_theoretical_columns.py \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --operator-analysis-csv <run_dir>/operator_analysis.csv \
  -o <run_dir>/raw_ops_details.json
```

合并规则：

- 默认假设 `operator_analysis.csv` 保留原始 `kernel_details.csv` 数据行顺序。
- `raw_ops_details.operators[].org_index` 是原 CSV 的 0-based 数据行号。
- 若 `operator_analysis.csv` 有显式 `org_index` 类列，则优先用该列；否则按 CSV body row number 对齐。
- 合并后的字段包括 `theoretical_operator_time_us`、`theoretical_compute_time_us`、`theoretical_memory_time_us`、`fixed_overhead_us`、`bound_type`、`duration_over_theoretical`、`duration_analysis`、`theory_supported`。

agent 需要检查脚本输出的 matched / supported / missing 数量；missing 异常时回查是否用了不同 CSV、AI_CPU 过滤口径或 step 选择。

## 4. 渲染与多流口径

Step 3 渲染时仍只传 `raw_ops_details.json`：

```bash
python scripts/render.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  -s <run_dir>/<network>_spec.json \
  --label <label> \
  -o <run_dir>/runs/<label>/index.html
```

HTML 主表会在 `median (ms)` 后增加 `theory median (ms)`：

- cluster 行：该 cluster 的 critical-stream 理论耗时中位数。
- TOTAL 行：该 component instance 内所有 kernel 按 stream 分组后，critical-stream 理论耗时中位数。
- bubble 行：显示 `—`，因为 bubble 是 idle / host / scheduling gap，不是 kernel 理论耗时。

多流默认口径：

- 对每个 component instance + sub-item，脚本按 `stream_id` 分组。
- 每个流计算 observed timeline union、supported kernel 数、理论耗时和。
- 默认选择 observed timeline union 最大的流，作为该 sub-item 的 critical stream。
- 未选中的流视为被覆盖或旁路，不计入 theory median。
- 只要出现多流，agent 应填写语义说明；若第二大流的 observed union 达到最大流的 80% 以上，HTML/metrics 还会标 `review`，agent 需要判断是否改选。

agent 不需要重新计算理论耗时，只负责填清楚多流语义，必要时覆盖脚本选择。`selected_stream` 可省略，省略时沿用脚本默认选择：

```json
{
  "decisions": [
    {
      "component_type": "moe",
      "sub_item": "expert",
      "phase": "main",
      "layer_idx": 4,
      "selected_stream": "203",
      "semantic_note": "stream 203 承载 expert GEMM 主路径；stream 186 是 dispatch/route 旁路搬运，被主计算覆盖",
      "stream_semantics": {
        "203": "expert GEMM critical path",
        "186": "covered dispatch / route movement"
      },
      "reason": "选择覆盖范围最大且承载主要 GEMM 的流作为理论耗时口径"
    }
  ]
}
```

然后重跑 render：

```bash
python scripts/render.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --theory-decisions <run_dir>/theory_stream_decisions.json \
  -s <run_dir>/<network>_spec.json \
  --label <label> \
  -o <run_dir>/runs/<label>/index.html
```

`wall/theory` 只表示当前实际 wall median 相对 critical-stream 理论 median 的倍数；它不是全量 kernel 理论和，也不能直接解释被其他流覆盖的 kernel。

## 5. 输出文件

| 文件 | 生成者 | 作用 |
|------|--------|------|
| `<run_dir>/operator_analysis.csv` | 外部 `operator-theoretical-perf` skill | 原始 CSV 加理论性能列 |
| `<run_dir>/raw_ops_details.json` | 当前 skill Step 1；Step 1.5 合并理论列 | 当前 run 的 per-kernel 事实载体 |
| `<run_dir>/theory_stream_decisions.json` | agent，多流场景必填 | 多流语义说明；必要时覆盖 critical stream 选择 |
