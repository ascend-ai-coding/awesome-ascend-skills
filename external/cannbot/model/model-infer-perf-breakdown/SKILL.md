---
name: external-cannbot-model-model-infer-perf-breakdown
description: 'NPU 性能数据拆解技能。把 kernel_details.csv 按用户描述的模型结构切成 component

  实例（每层 attn / ffn / moe…），再按用户给的 cluster 规则把 component 内部算子

  分桶，生成 wall_ms / bubble_ms 中位数 + 异常 layer 的单页 HTML。

  触发场景：分析 NPU prof / 拆解大模型性能 / 找抖动 layer。'
original-name: model-infer-perf-breakdown
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# NPU 性能数据拆解技能

输入一份 prof（`kernel_details.csv` + `trace_view.json`），结合用户对模型结构的口头描述，先把 op 序列切成 component 实例，再按 cluster 规则统计每类计算的 wall_ms + bubble_ms 中位数 / 抖动 / 异常 layer，输出一页可读的 HTML。内部指标字段用 `*_ms`，HTML 展示按量级自动使用 `µs` 或 `ms`。

---

## 工作流

```
Step 0: 收齐 prof 路径 + 模型脚本路径 (硬约束)  →  <run_dir>/inputs.json
Step 1: kernel_details.csv  →  raw_ops.json (单 step 提取)
Step 1.5: (可选,默认不开) 调用 operator-theoretical-perf → operator_analysis.csv → 合并理论列到 raw_ops_details.json
Step 2: Phase 0a 用户描述模型 → structure_spec.json
        Phase 0b 敲定每个 component 的 stream sample 范围 (必须用户 ack)
                → sample_ack.json
        Phase 1  跑 detect_structure → structure_draft.json (component 实例表)
        Phase 1.5 复读 warnings 让用户 ack
Step 3: 用户和 agent 对话定 cluster 规则  →  network_spec.json
        → index.html（含 theory median 列）
Step 4: 按 component 切 csv / trace  →  splits/ (单层下钻)
Step 5: 洞察归纳（agent 语义判断 + 脚本统计） → insights/
```

每个 Step 各自的脚本 docstring 里有完整 schema 与算法描述；本文档只列契约与 CLI。

---

## 工作目录约定

跑本 skill 的 agent **第一步**（早于 Step 1）必须先和用户敲定一个 **`<run_dir>`**——本次 skill 所有产物的落盘根，下文 CLI 里出现的 `<run_dir>/` 都指这个目录。默认建议 `./perf_breakdown/<label>/`，`<label>` 由用户给（如 `<network>_baseline` / `<network>_variant`）；用户也可以直接指定绝对路径。不存在就 `mkdir -p`。

**进 `<run_dir>` 先查历史产物，再判档复用**：每次跑 skill，agent 在做新的对话 / 脚本调用前，必须先列 `<run_dir>` 已有文件 + 读 `HISTORY.md`（如果有），然后跑 Step 1 出新 `raw_ops.json`（落到 `raw_ops.next.json` 以免覆盖），和上一 run 做自动 diff，判出 **小改 / 中改 / 大改** 三档之一，再按对应档位复用产物：

| 档 | 触发信号 | 复用范围 |
|----|---------|----------|
| **小改** | kernel 集合完全相同；anchor count 偏差 = 0；上一 sample 的 stream anchors 在相近位置命中 | `structure_spec` / `structure_draft`（sample 平移） / `<network>_spec` 全复用；跳过所有对话 |
| **中改** | 出现新 normalized_name；anchor count 偏差 ≤ 10%；sample 偏移 > 5 row；shape 分布漂移 | `structure_spec` 复用 + 回显问；**重走 Phase 0b** 重出 sample；`<network>_spec` 增量补 rule |
| **大改** | anchor count 偏差 > 10%；总 op 数偏差 > 30%；关键 anchor 消失 | 重走 Phase 0a / 0b；`<network>_spec` 当参考但逐条审 |

判档后 agent 复述一句结论给用户 ack（如"prof 相对 `<prev_label>` 是中改：MoE 新增 `AddRmsNormCast`，sample 偏移约 80 行。打算 B 档复用，对吗？"），用户不同意直接按用户给的档位走。

首次跑（`<run_dir>` 空）跳过 diff 走全量流程。完整信号定义与判档规则见 `references/reuse_policy.md`。

---

## 反模式

**禁止修改用户提供的任何原始文件**（prof / modeling 源码 / config 等）；产物只写到 `<run_dir>/`。

---

## 前置物料 checklist（Step 1 之前必走）

跑本 skill 在做任何 prof 解析之前，agent **必须**和用户敲齐下列物料；任一缺失或未确认即 abort，不要带着空缺/默认值往下走。

| 物料 | 强度 | 缺失/未确认行为 |
|------|------|----------|
| **prof 路径**（`kernel_details.csv` + `trace_view.json`） | 硬约束 | 文件不存在直接 abort |
| **模型脚本路径**（`modeling*.py` + 相关 module 文件） | **硬约束** | 用户说"没有/不方便"也 abort——stream sample ack 流程依赖它；用户应当先去拿到再回来 |
| **`<run_dir>` 目录** | 硬约束 | 不存在则 `mkdir -p` |
| **phase + device 选择**（多卡 / 多采集会话时） | **硬约束** | 必须列出可选项让用户挑；不允许 agent 自作主张默认选第一个 |

Ascend prof 有两种主流目录布局（A 类 `torch_npu.profiler` / B 类原生 `msprof`），各自的 csv / json 落在哪、多卡怎么分目录见 `references/prof_layouts.md`。agent 看到 prof 目录后**先识别是哪一类**，再按对应位置取 `kernel_csv` / `trace_json`；多卡 / 多采集 / 多 phase 任一维度组合数 > 1 时枚举所有 `(phase, rank/device)` 让用户挑，每个组合各起一个 sub `<run_dir>` 独立跑。

每个子 run 的 `inputs.json`（agent 写）：

```json
{
  "prof_dir": "/abs/path/to/<prof 根目录>",
  "kernel_csv": "/abs/path/to/.../kernel_details.csv 或 op_summary_*.csv",
  "trace_json": "/abs/path/to/.../trace_view.json 或 msprof_*.json",
  "model_script_paths": ["/abs/path/to/modeling_xxx.py", "/abs/path/to/moe_module.py"],
  "phase": "decode",
  "rank": 0
}
```

`rank` 一般 = `device_id`（每 process 绑一张卡时），分布式训练 rank ≠ device 时由用户给。

后续 `detect_structure.py` 启动时会校验 `inputs.json` 存在、`model_script_paths` 非空且文件存在、`phase` / `rank` 字段已填，否则 exit 1。

---

## 历史记录（`<run_dir>/HISTORY.md`）

agent 维护的人类可读运行日志，用途：

- 下一个 agent / 用户进 `<run_dir>` 时，一眼看清"这个模型之前是怎么拆的、为什么这么拆"
- 跨 run 归因：某次性能变化是因为换了 sample？改了 cluster 规则？还是模型本身改了？（与 `compare_runs.py` 输出的 `history.html` 定量对比互补）

**契约**：
- 进 `<run_dir>` 时若文件存在 → 必读
- 每次完成 `render.py`（Step 3 Phase 2 出 `runs/<label>/`）后 → 必 append 一条
- append-only，按时间从老到新；禁止回头改旧条目

条目字段、写法、撤回规则见 `references/history_template.md`。

---

### Step 1 — 单 step 提取

```bash
python scripts/analyze_kernels.py -f kernel_details.csv -s 2 \
  -o <run_dir>/raw_ops.json -d <run_dir>/raw_ops_details.json
```

**作用**：从 csv 抽一个稳定 step（默认跳过 step 0 warmup），后续所有分析以单 step 为单位。

**输入**：`kernel_details.csv`（同目录通常还有 `trace_view.json`，Step 4 用，Step 1 不消费）。

**输出**：
- `raw_ops.json` — 单 step `operators[]`，每条含 `index` / `org_index`（csv 0-based 数据行号）/ `normalized_name` / `duration_us` / `start_time_us` / `stream_id` / `input_shapes` / `output_shapes`。Step 2、3 消费。
- `raw_ops_details.json` — 单 step 详情，含 csv 全部字段。Step 4 切片时用。

**行号 domain**（Excel / `org_index` / `index`，三者各自的语义和换算关系）+ AI_CPU drift 注意事项见 `references/phase0b_workflow.md` 末尾的"行号 domain 速查"。

**Fallback 行为**：`accelerator_core ∈ {AI_CPU, AICPU}` 默认剔除；sample 模式严禁加 `--keep-ai-cpu`（破坏 row-order）。若 profiler 同时记录 HCCL/COMMUNICATION 的 collective summary 行和 `Name == AivKernel` 的 task fragment 行，Step 1 只保留 summary 行、丢弃可匹配 summary 的 fragment；没有 summary 证据的通信 fragment 必须保留，不能静默丢失。AI_VECTOR_CORE 中的 Gather/Scatter/Reduce/Slice 等仍按普通算子保留。

---

### Step 1.5 — 理论性能注入（可选，默认不启用）

**可选步骤**——依赖外部 `operator-theoretical-perf` skill。不启用时 `render.py` 的 theory median 列自动留空（`—`）、不阻塞；Step 5 的理论偏离 insight 也自动跳过（输出空 JSON + `data_limits` 说明）。仅当用户要理论对比、且外部 skill 可用时才启用。

**启用时**：chip 配置提问可并入 Step 0 前置物料，或推迟到 Step 2 之后、`render.py` 之前（Phase 0a 的「不要先于此发问」只约束结构类提问，chip 不受限）；理论列合并必须在 `render.py` 之前完成。

调用外部 `operator-theoretical-perf` skill（或直接跑其 `analyze_operators.py --chip-config <chip>.json`），用原始 `kernel_details.csv` + 用户确认的 chip 生成 `<run_dir>/operator_analysis.csv`，再把理论列按原 CSV 行序 / `org_index` 合并回 `<run_dir>/raw_ops_details.json`：

```bash
python scripts/merge_theoretical_columns.py \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --operator-analysis-csv <run_dir>/operator_analysis.csv \
  -o <run_dir>/raw_ops_details.json
```

理论列不参与 Step 2 结构拆解；sub-item / TOTAL 粒度理论耗时在 Step 3 渲染时从合并后的 `raw_ops_details.json` 计算。调用契约、多流选择口径见 `references/theoretical_perf_workflow.md`。

---

### Step 2 — sample-driven 层检测

**Phase 0a：问用户模型结构（必走）**

agent **逐字**把 `references/phase0a_prompt.md` 里的 prompt 发给用户，解析答案后自动起 phase 名（①→`main`、②→`mtp_head`、③→用户类别名），写 `<run_dir>/structure_spec.json` 并回显 JSON 让用户确认。schema：

```json
{
  "model_hint": "<自然语言>",
  "phases": [
    {"name": "main", "layers": 6,
     "layer_compositions": [
       {"layer_range": [0, 2], "components": ["mla", "dense"]},
       {"layer_range": [3, 5], "components": ["mla", "moe"]}
     ]},
    {"name": "mtp_head", "layers": 3,
     "layer_compositions": [{"layer_range": [0, 2], "components": ["mtp", "moe"]}]}
  ],
  "expected_components": ["mla", "dense", "moe", "mtp"],
  "expected_layer_count": 9
}
```

**Phase 0b：定 stream sample 范围（必走，必须用户 ack）**

**核心契约**：按 `expected_components` 去重列表**逐项**敲定每个 component 的 sample 范围。脚本会把 sample 转成 per-stream `stream_samples`；每个 component 必须有且只有一个 `primary` stream，其他确认属于该 component 的流标 `auxiliary`。**归属看语义、不看时间是否重叠**：语义上属于该 component 的流，即使与本层时间脱节也写 `auxiliary`——`detect_structure` 会发 hard warning `auxiliary_stream_temporally_displaced`，`render` 自动把它从 cluster/bubble/TOTAL 剔除并标注、op 仍 matched 保留在 component（`op_indices` 全集不破、Step 4 split 照切），判定是流级全有全无；只有**不属于任何 component** 的流（全局通信/采样/IO 等）才留 `unmatched`。用户 ack 后写 `<run_dir>/sample_ack.json`；agent 不能跳过 ack 自己往下推。pre / post / inter 不需要给。

两条路径（sample 范围是硬需求，二选一）：

- **A 档**：用户直接给 row 区间或算子名，agent 对照 `raw_ops.json` 校验后回贴 ack
- **B 档**：用户给不出 → agent 读 Step 0 收齐的 `model_script_paths` 推典型 sample 范围，标注来源后让用户核对

详细步骤、回贴格式、`sample_ack.json` schema 见 `references/phase0b_workflow.md`。`expected_components` 中任一缺失即 ack 未完成，`detect_structure.py` 启动时 exit 1。

**Phase 1：跑 detect_structure**

```bash
python scripts/detect_structure.py \
  -r <run_dir>/raw_ops.json \
  --structure-spec <run_dir>/structure_spec.json \
  --inputs <run_dir>/inputs.json \
  --sample-ack <run_dir>/sample_ack.json \
  -o <run_dir>/structure_draft.json
```

`--inputs` 和 `--sample-ack` 在 sample 模式下**必传**：
- `--inputs <run_dir>/inputs.json`：校验 `model_script_paths` 非空（Step 0 物料）
- `--sample-ack <run_dir>/sample_ack.json`：使用 `schema_version: "stream_sample_ack.v1"`，校验每个 component 都有 `stream_samples[]` 且只有一个 `primary`
- 任一校验失败 exit 1，agent 必须回 Phase 0b 补齐

新写入的 `sample_ack.json` 必须保存为 stream sample schema；Step 2 输入须为此 schema。

hard / ambiguous warnings（含 `auxiliary_stream_temporally_displaced`）默认 exit 1；agent 必须先逐条复读给用户，用户明确接受后才可加 `--accept-warnings` 重跑放行。

**输出 `structure_draft.json`**（下游 Step 3 / Step 4 直接消费）：
- `mode: "stream_sample_driven"`
- `schema_version: "structure_draft.stream.v1"`
- `structure_spec` — 原样回写
- `samples_used[]` — 每 sample 的 `stream_samples[]`
- `components[]` — 扁平表，每条 = 一个 component instance（`component_id` / `layer_idx` / `phase` / `type` / `op_indices[]` / `stream_segments[]` / stream-aware `scores`；某条 aux 流被判时间脱节时还含 `displaced_op_indices[]`——该流的 op，render 据此从 cluster/bubble/TOTAL 剔除，但仍在 `op_indices` 全集内、partition 不破）
- `op_to_component` — op index 到 component_id 的反查表
- `unmatched_op_indices[]` / `unmatched_stream_segments[]` — 未归属 op，不能静默丢弃
- `warnings[]` — primary/auxiliary stream、op membership、composition schedule、count/shape/ambiguity、temporally-displaced aux stream 类问题，每条含 `code` + `message`
- `validation` — declared vs detected 层数 / missing_samples / ambiguous_matches

warning codes、validation 字段、匹配算法细节见 `detect_structure.py` / `sample_matching.py` docstring。

**Phase 1.5：复读 warnings 等用户 ack（必走）**

逐条把 `warnings[]` 复读给用户，列 layer_idx / component / 期望 vs 实测。用户选项："忽略" / "改 0a 重跑" / "改 0b 重跑"。无 ack 视为 Step 2 未结束。

---

### Step 3 — cluster 规则对话 + 渲染

**Phase 1：agent 引导用户给 cluster 规则**

`structure_draft.json` 给出 component 实例边界，但每个 component 内部还要进一步分桶才能看出抖动来源。agent 主动和用户对话产出 `<run_dir>/<network>_spec.json`——每个 `component_type` 一个 cluster 列表，每个 cluster 一条或多条匹配规则（`op_name` / `op_name_regex` / `input_shapes_contains` / `output_shapes_regex` / `catch_all` 等），逐 component_type 让用户确认。

跑完 `render.py` 后看 `unmatched ops` 面板，若有未匹配算子回头细化规则直到 unmatched = 0 或用户明确说"剩下的不关心"。

详细工作步骤、规则字段、`network_spec.json` schema、`outlier` 选项、cluster / bubble / TOTAL 数值定义、spec 跨 prof 复用规则见 `references/cluster_dialogue.md`。

**网络 spec 存放位置**：`<run_dir>/<network>_spec.json`。同一模型在不同 prof 上复用——新 run 进入工作目录后优先查同名文件，存在就直接读，不必重跑 cluster 对话。

**Phase 2：渲染**

**启用理论时**（Step 1.5 已把逐 kernel 理论列合并进 `raw_ops_details.json`），渲染脚本在 cluster / TOTAL 粒度按 critical stream 聚合理论耗时；**未启用则 theory median 列整列留空（`—`）**，本段其余多流口径不适用。多流时默认取 observed timeline union 最大的 stream，未选中流视为被覆盖或旁路，不计入该 sub-item 的理论耗时。若某 sub-item / TOTAL 出现多流，agent 应在 `<run_dir>/theory_stream_decisions.json` 填 `semantic_note` 说明这些流的语义；若候选流覆盖范围接近，还要同时判断是否覆盖 `selected_stream`。没有语义说明的多流项会在理论列标 `semantic note required`。

```bash
python scripts/render.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  -s <run_dir>/<network>_spec.json \
  --label <label> \
  -o <run_dir>/runs/<label>/index.html
```

（首次渲染也走 `runs/<label>/`，与输出文件表 / Step 5 回填 / `compare_runs` 一致；`metrics.json` 落到同目录。）

如需填写多流语义或覆盖 critical stream 选择，再追加：

```bash
--theory-decisions <run_dir>/theory_stream_decisions.json
```

HTML 主表在 `median` 后展示 `theory median` 列（数据来自 Step 1.5 合并进 `raw_ops_details.json` 的 `theoretical_operator_time_us`；**未跑 Step 1.5 时整列留空 `—`**）；两列都按量级自动显示 `µs` 或 `ms`。bubble 行没有 kernel 理论耗时，显示 `—`。

主报告最后一列固定为 `insight`：Step 3 初次渲染时保留为空；Step 5 结束后由 agent 写 `<run_dir>/runs/<label>/insights/main_report_insights.json`，再用 `--insight-annotations` 重跑 `render.py` 回填。该列只显示 `confidence = high | medium` 的洞察，`low` 不写入主报告。每条回填必须带 `category`；若不能精准映射到单个模块，由 agent 选择主要归属 `target`，用 `mapping_type: agent_primary` 和 `mapping_note` 说明依据。`TOTAL` 只是可选主归属之一，只有洞察确实是整体性的才放到 `TOTAL`，否则优先放到最相关的具体 cluster / `bubble` 行。同一主报告 row 可以显示多条 insight。

`--raw-ops-details` **必传**：缺了 chip 退化成静态 `<span>`，无法下钻 per-op 明细。

**unmatched gate**：`render.py` 在 **compute-core unmatched 比例 > `--unmatched-limit`（默认 5%）** 时 exit 1（通信类 core 不计入该比例）。报错打印按 `accelerator_core` 的 unmatched 分解：若主要是通信 / 采样 / embedding / IO，加 `--accept-unmatched` 放行；若有大段 compute 算子漏匹配，回 Phase 0a/0b 补 sample 或声明 component。`--unmatched-limit` 可调阈值。

**输出**：
- `index.html` — 单页报告，按 spec 中 component_type 顺序排，每 component_type 一张 cluster / bubble / TOTAL 抖动表，含 `unmatched ops` 面板。若 cluster 桶之间存在 timeline overlap 或覆盖率异常，HTML 顶部按下列规则插提示面板：
  - **cluster 覆盖率红字**：Σ cluster_wall.median / TOTAL.median < 80% 时插红字 banner，提示大量算子落进 bubble / 未匹配 / 漏算子，查 unmatched 面板与 cluster 规则。
  - **overlap 黄字警告**：median gap ≥ 5% TOTAL，提示"分桶不能直接反映真实性能分布"，列主要重叠桶对 + 占比。
  - **overlap 浅蓝 info**：median gap 0 < x < 5% TOTAL，列出仍存在跨 stream 并发的桶对（过滤掉 < 1µs 的噪声项，最多 8 对），方便定位并发来源，但不影响主表读数。
  - **辅流时间脱节提示（浅蓝）**：某 aux 流被判 displaced 时，列其 op_count / median wall / %TOTAL，标注"已从 TOTAL/bubble 排除"、op 仍 matched（逐层明细见 splits/）。
- `metrics.json` — `compare_runs.py` 消费的指标数据；每个 section 含 `overlap: {median_gap_ms, median_gap_pct, max_gap_ms, top_pairs[]}`、`cluster_sum_pct_of_total`、`displaced`（per-stream 脱节摘要）字段。字段名保持 `*_ms`，HTML 展示层会把小于 1ms 的值显示成 `µs`。
- render 完成时若 `<run_dir>/HISTORY.md` 不存在，stdout 打 reminder（不 exit 1）。

cluster / bubble / TOTAL 定义、outlier 检测、overlap 度量、规则求值顺序见 `render.py` docstring。

---

### Step 4 — 按 component 切 csv / trace_view

**必跑**。skill 的交付契约要求 op 序列被完整切片：每个 component 实例与每段 unmatched op 集合各占一个文件夹，整体严格覆盖 `[0, N-1]`、无重无漏。新 draft 以 `op_indices` 为准，component 可以是非连续 op 集合；`op_range` 只作为 manifest 里的展示 envelope。脚本会自检 partition，draft 与 raw_ops 不匹配时直接 abort。

```bash
python scripts/split_artifacts.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  -f /path/to/kernel_details.csv \
  -t /path/to/trace_view.json \
  -o <run_dir>/splits/
```

文件夹命名（直接对应 layer_id 与 component 类型）：

| kind | 命名 |
|------|------|
| component 实例 | `L<idx:03d>_<phase>_<type>` |
| pre/post 头尾 | `pre` / `post` |
| 相邻 component 间的 gap | `gap_L<prev>_<phase>_<type>_to_L<next>_<phase>_<type>` |
| 头尾外侧 gap | `gap_before_L<next>_…` / `gap_after_L<prev>_…` |
| 疑似未声明的 component | `suspected_…`（其余结构同 gap） |

每个文件夹内只放两份产物：
- `kernels.csv` — csv 切片（保留 header，首列 `org_index`）
- `trace_view.json` — trace 证据；传 `-t` 时是该目标 op 的时间包络上下文窗口，可能包含同时间并行的其他目标 op；不传时从 raw_ops 的 `op_indices` 精确合成

顶层 `manifest.json` 是 folder → `{kind, op_range, op_indices?, ops_count, files}` 索引，给下游 / 用户导航用；当目标 op 非连续时才写 `op_indices`。`files.trace_scope` 标明 trace 是 `op_membership_exact` 还是 `time_envelope_context`。

`-f` / `-t` 在标准 prof 包下必传；缺 csv 跳过 csv 切片，缺 trace 时 trace_view.json 从 raw_ops 合成。`--dry-run` 只打印目标清单不落盘。

### Step 5 — 洞察归纳

Step 5 在 Step 3 / Step 4 之后执行，只做洞察层归纳与展示，不重新定义模型结构、不修改 cluster 规则。事实提取与渲染由 `scripts/gen_insights.py`（两遍：先产事实候选，agent 写 `movement_taxonomy.json` + `annotations.json` 后带 `--movement-taxonomy`/`--annotations` 再跑）+ `scripts/render_insights.py` 完成，agent 不手写提取逻辑；候选可靠性 / 是否展示由 agent 用 `confidence`（high|medium|low）标注。CLI、脚本/agent 边界、五类 insight、HTML 渲染与落盘契约见 `references/insight_workflow.md`。

默认输出目录：`<run_dir>/runs/<label>/insights/`，和该 run 的 `index.html` / `metrics.json` 放在同一目录下。只有未使用 `runs/<label>/` 的老式单 run 输出，才退回 `<run_dir>/insights/`。Step 5 最后还要生成 `main_report_insights.json`，并重跑主报告渲染（`render.py --insight-annotations`），把 high / medium 洞察连同类别和主映射说明填回 `runs/<label>/index.html` 最后一列。

---

## 增量分析（模型迭代 + 历史对比）

模型局部改动后重新 prof 时，**不要**从零再走一遍 Step 2 对话。复用规则：

- **`network_spec.json` 跨 prof 复用**：cluster 规则是 op_name + shape pattern，与 layer 位置 / 数量无关，模型改动只要不引入新算子就直接复用。新增算子时只需补一条 rule。
- **`structure_spec.json` 通常微调**：layer 数 / composition 不变就照搬；改了层数 / 加了 MTP 改 0a 重写。
- **sample 行号需重出**：op 在 csv 中的位置随改动平移，Phase 0b 每次都要给。

**每个 run 落盘到独立目录**：`render.py` 的 `--label <label>` + `-o <run_dir>/runs/<label>/index.html`（首次起就这样，见 Step 3 Phase 2）让每个 run 独占 `runs/<label>/`；增量 run 只换 `<label>` 值。同目录同时产 `metrics.json`（含 `label` / `model_name` / `outlier` / 每个 `(component_type, cluster)` 的 wall + bubble 指标 + outlier 列表，`compare_runs.py` 消费）。

**对比表**：

```bash
python scripts/compare_runs.py \
  -r <run_dir>/runs/ \
  -o <run_dir>/history.html \
  [--metric median_ms]    # 默认 median_ms；可选 mean_ms / p95_ms / max_ms / std_ms
  [--baseline <label>]    # 默认第一个 run 当 baseline
```

`-r` 给目录时自动扫所有 `*/metrics.json`，按 mtime 升序排（早的当 baseline）。也可以列出多个 `metrics.json` 路径显式指定顺序。

输出 `history.html` — 多 run 跨版本对比表（Δ% vs baseline + outlier diff），列布局见 `compare_runs.py` docstring。

---

## 输出文件

| 文件 | 阶段 | 说明 |
|------|------|------|
| `<run_dir>/raw_ops.json` | Step 1 | 单 step kernel 概要，下游所有脚本消费 |
| `<run_dir>/raw_ops_details.json` | Step 1；Step 1.5 可加列 | 单 step kernel 详情，Step 4 切片用；启用理论性能时合并理论列 |
| `<run_dir>/operator_analysis.csv` | Step 1.5 | 外部 `operator-theoretical-perf` skill 对原 CSV 加列后的原生产物 |
| `<run_dir>/theory_stream_decisions.json` | Step 3 / 2 | agent 对多流语义的说明（多流场景必填）；必要时覆盖 critical stream 选择 |
| `<run_dir>/structure_spec.json` | Step 2 / 0a | 用户描述的模型结构 |
| `<run_dir>/structure_draft.json` | Step 2 / 1 | component 实例表，Step 3 / 4 输入 |
| `<run_dir>/<network>_spec.json` | Step 3 / 1 | 用户的 cluster 规则，新 run 优先复用同名文件 |
| `<run_dir>/runs/<label>/index.html` | Step 3 / 2 | 单 run 的单页 cluster 抖动报告 |
| `<run_dir>/runs/<label>/metrics.json` | Step 3 / 2 | 单 run 指标，`compare_runs.py` 消费 |
| `<run_dir>/runs/<label>/insights/module_bubble.json` | Step 5 | 模块长期 bubble 与具体层/模块 bubble 异常 |
| `<run_dir>/runs/<label>/insights/operator_jitter.json` | Step 5 | 算子 / cluster 性能抖动候选与证据 |
| `<run_dir>/runs/<label>/insights/theoretical_deviation.json` | Step 5 | 逻辑 sub-item / TOTAL / operator slot 与理论性能偏离 Top5 |
| `<run_dir>/runs/<label>/insights/vector_sequence_candidates.json` | Step 5 | 连续 vector 区间融合候选，按 pattern 聚合 |
| `<run_dir>/runs/<label>/insights/data_movement_ops.json` | Step 5 | 冗余 Tensor Movement 消除候选：搬运 / 结构变换算子定位、上下文与消除方向 |
| `<run_dir>/runs/<label>/insights/final_conclusions.md` | Step 5 | agent 基于五类 insight 汇总的最终结论与证据索引 |
| `<run_dir>/runs/<label>/insights/index.html` | Step 5 | 五类 insight 的可视化页面 |
| `<run_dir>/runs/<label>/insights/main_report_insights.json` | Step 5 | agent 选择回填到主报告 `insight` 列的 high / medium 洞察 |
| `<run_dir>/history.html` | 增量 | 多 run 对比表（Δ% vs baseline） |
| `<run_dir>/HISTORY.md` | 每 run 收尾 | agent append 的运行日志，记录复用 / 变更 / 观察 |
| `<run_dir>/splits/` | Step 4 | 按 component 切的 csv / trace_view，含 manifest.json |

---

## 参考

- `references/kernel_data_guide.md` — `kernel_details.csv` 字段说明
- `references/prof_layouts.md` — Ascend prof 两类目录布局（torch_npu / msprof）+ 多卡识别要点
- `references/phase0a_prompt.md` — Step 2 Phase 0a 用户结构描述逐字 prompt
- `references/phase0b_workflow.md` — Step 2 Phase 0b stream sample ack 流程 + `sample_ack.json` schema + 行号 domain
- `references/cluster_dialogue.md` — Step 3 cluster 规则对话步骤 + `network_spec.json` schema + 数值定义
- `references/history_template.md` — `HISTORY.md` 条目字段与模板
- `references/reuse_policy.md` — prof 变更检测信号 + 三档复用规则
- `references/theoretical_perf_workflow.md` — Step 1.5 理论性能注入：外部 skill 调用、CSV 理论列合并、多流选择口径
- `references/insight_workflow.md` — Step 5 洞察归纳：五类 insight、脚本/agent 边界、HTML 渲染
- `references/insight_html_template.html` — Step 5 洞察页 HTML/CSS/JS 骨架
- 各脚本 docstring — 内部算法、阈值推导、edge case 处理
