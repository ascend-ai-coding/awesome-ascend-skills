# Step 3 — cluster 规则对话

`structure_draft.json` 提供 component 实例边界，但每个 component 内部还要进一步分桶才能看出抖动来源。这一步由 agent 主动和用户对话，产出 `<run_dir>/<network>_spec.json`——每个 `component_type` 一个 cluster 列表，每个 cluster 一条或多条匹配规则。

---

## agent 工作流程

1. **回贴算子序列**：对每个 component_type，从 `structure_draft.components[*]` 取一个代表实例（avoid block 0 / 最后一层 / phase 边界），按 `op_indices[]` 从 `raw_ops.json` 取算子序列，并按 `stream_id` 分组展示 `normalized_name` + input/output shapes。不要用 `min(op_indices)..max(op_indices)` 的连续 envelope 代替真实 membership。
2. **逐 op 引导分类**：按用户描述把 op 划进语义 cluster（如 `input_norm` / `q_compress` / `q_proj` / `flash_attn` / `output_proj` / `residual` …）。
3. **匹配规则字段**：
   - `op_name` 精确匹配
   - `op_name_regex`
   - `input_shapes_contains` / `input_shapes_regex`
   - `output_shapes_contains` / `output_shapes_regex`
   - `catch_all: true`（必须放最后，保证不漏算子）
4. **复用 component_type**：ffn / moe 同理；MTP 头若复用主干同名 component（如 `moe`），cluster 规则可共用——同一 component_type 一套规则覆盖所有 phase。
5. **逐 component_type ack 后写入** `network_spec.json`。
6. **必走收尾**：跑一次 `render.py` 后看输出里的 `unmatched ops` 面板——若有未匹配算子，回 2-3 继续细化规则直到 unmatched = 0 或用户明确说"剩下的不关心"。

---

## `network_spec.json` schema

```json
{
  "model_name": "<network>",
  "outlier": {"method": "iqr", "k": 1.5},
  "component_clusters": {
    "csa": [
      {"cluster": "input_norm",
       "description": "Pre-attn RMS norm",
       "rules": [{"op_name": "RmsNorm"}]},
      {"cluster": "q_compress",
       "description": "Q projection: hidden → q_rank latent",
       "rules": [{"op_name": "MatMulV3", "input_shapes_contains": "<dim>"}]},
      {"cluster": "other", "rules": [{"catch_all": true}]}
    ],
    "moe": [...]
  }
}
```

`outlier` 可选：
- `{"method": "iqr", "k": 1.5}`（默认）
- `{"method": "z", "k": 2}`

每个 sub-item 独立检测 outlier：cluster 在 cluster wall 上检测、bubble 在 layer bubble 上检测、TOTAL 在 layer end-to-end span 上检测。

---

## 数值定义

- **cluster wall** = 该 cluster 算子的时间并集
- **bubble** = 层 span − 总 wall 并集（层内算子之间的空隙）
- **TOTAL** = 层 end-to-end span（`max_op_end − min_op_start`，**含 bubble**），等价于 trace_view 里框选一层量到的端到端耗时

---

## 网络 spec 存放位置 / 复用

写到 `<run_dir>/<network>_spec.json`。同一模型在不同 prof 上复用——新 run 进入 `<run_dir>` 后优先查同名文件，存在就直接读，不必重跑 cluster 对话。

模型迭代时：
- 不引入新算子 → 直接复用
- 新增算子 → 只需补一条 rule
- 算子大改 → 重新走一遍 cluster 对话

cluster / bubble / TOTAL 定义、outlier 检测、overlap 度量、规则求值顺序见 `render.py` docstring。
