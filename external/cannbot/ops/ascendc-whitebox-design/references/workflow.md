# White-box Test Case Generation Workflow

> ⚠️ **全流程约束规则的唯一定义在 SKILL.md「执行约束（强制）」节。** 主 Agent 和子 agent 执行前必须核对。

---


## Step 1：输入收集

> **前置条件**：无（流程起始步）。

主 Agent 必须先 Read `references/S1-input-collection.md`，并严格按该文件的 1.0 → 1.1 → 1.2 → 1.3 → 1.4 执行。`workflow.md` 不重复定义 Step 1 子步骤；Step 1 的唯一执行规则以 `S1-input-collection.md` 为准。

> Step 1 的串行要求、question 工具规范、路径查找限制和确认摘要格式均见 `references/S1-input-collection.md`。

**输出数据**：

| 数据 | 来源 | 用途 |
|------|------|------|
| `npu_arch` / `soc_version` / `chip_model` | 1.0 平台检测 | 全流程平台判断 |
| `platform`（核数 / UB 大小） | 1.3 参数映射 | 子 agent 上下文 |
| `op_name` / `op_path` / `gate_status` | 1.1 用户输入，1.2 两阶段查找（目录名 + OP_ADD 反查） | 流程控制和产物路径 |

---

## Step 2：分析源码

> **前置条件**：Step 1.4 用户已确认摘要（"确认，开始分析"）；算子路径、平台参数均已确定。

**输入文件**：

| 文件 | 用途 |
|------|------|
| 算子源码路径 | Step 1.2 定位，tiling / kernel / 接口分析 |
| 平台参数（npu_arch / soc_version / core_count / ub_size） | Step 1.0-1.3，平台分支判断 |

**执行概要**：

主 Agent 按阶段完成源码分析：Phase 0 调用 `scripts/run_s2p0.py` 一次性生成源码读取边界；Phase 1 并行调度子 agent。Phase 1 全部子 agent 返回后，由 `scripts/check_s2p1.py` 对 Phase 1 产出做文件级粗验收并给出失败归属。Step 2 产物需覆盖 route，并让 `S2P1_cases.json` 中已进入 `case_values` 的字段满足 route model 声明的多样性目标。多样性目标必须通过分支约束域随机采样和 evaluator route 稳定性复验实现，不得通过固定 witness 值表或添加内部推导字段实现。Step 2 的分析目标、Phase 分工和源码读取策略见 `references/source-analysis/00-execution-order.md`。

**输出文件**：

| 文件 | 说明 |
|------|------|
| `S2P0_scout_t.md` | Phase 0 wrapper 生成的 tiling 侦察报告 |
| `S2P0_scout_t.json` | Phase 0 wrapper 生成的 tiling 侦察结构化数据 |
| `S2P0_scout_k.md` | Phase 0 wrapper 生成的 kernel 侦察报告 |
| `S2P0_scout_k.json` | Phase 0 wrapper 生成的 kernel 侦察结构化数据（含 `reachable_kernel_templates`） |
| `S2P0_file_manifest.json` | 源码文件清单（tiling/kernel 优先级 + 排除列表；kernel file_list 仅包含 P0/P1，P2 在 excluded） |
| `S2P0_source_scope.md` | 源码读取最大边界（后续任务按自身规则读取；P2 禁止读取） |
| `S2P1_route_dispatch_model.md` | evaluator 指导书 |
| `S2P1_route_extract.py` | route evaluator 脚本 |
| `S2P1_case_generator.py` | route case 生成器 |
| `S2P1_route_cases.json` | route cases（case generator 输出） |
| `S2P1_cases.json` | 用例文件（evaluator 附带输出，flat array，仅含 reachable 用例） |
| `S2P1_path_list.json` | 代码路径清单（path graph） |
| `S2P1_tiling_glossary.md` | tiling 变量含义表（tiling 源码变量名 → 语义名映射） |
| `S2P1_operator_model.json` | 算子接口模型（inputs/outputs/attributes） |
| `S2P1_low_configs.json` | 常见网络 shape 配置（语义参数名） |
| `S2P1_postcheck_report.json` | Step 2 Phase 1 postcheck 报告 |

---

## Step 3：多样性分析

> **前置条件**：Step 2 全部 Phase（0/1）已完成；`S2P1_cases.json` 已生成。

主 Agent 直接运行多样性分析脚本，生成覆盖率报告。

```bash
python3 {skill_base}/scripts/s3_diversity_report.py \
  --cases {op_path}/tests/whitebox/S2P1_cases.json \
  --path-list {op_path}/tests/whitebox/S2P1_path_list.json \
  --output {op_path}/tests/whitebox/S3_diversity_report.md \
  --operator {op_name}
```

输出：`S3_diversity_report.md`。Step 3 为统计步骤，脚本退出码为 0 后直接进入 Step 4。主 Agent 只读取报告顶部 `Agent Summary`，不根据 WARN 回退 Step 2。

---

## Step 4：用户确认

> **前置条件**：Step 3 已完成，`S3_diversity_report.md` 已生成。

展示 `S3_diversity_report.md` 顶部 `Agent Summary` 后停下来等待确认。

| 选项 | 说明 |
|------|------|
| 确认，继续 | 进入 Step 5（case mapper） |
| 需要调整 | 回 Step 2 修改 |

IF Step 1 选择「跳过 Step 4 闸门」→ 跳过此步直接进入 Step 5；ELSE 用户确认后才能进入 Step 5。

### Step 4 闸门（强制）

收到用户确认之前，**禁止**：运行 Step 5（case mapper）、生成 `S5_cases_low.json` / `S5_cases_high.json`。

**允许继续条件（满足其一）**：用户在对话中明确确认（如「确认」「继续生成 cases」）或用户写明「跳过 Step4 确认」/「一次跑完全流程」。

---

## Step 5：映射参数组合（子 agent）

> **前置条件**：Step 4 用户已确认；`S2P1_cases.json`、`S2P1_operator_model.json`、`S2P1_low_configs.json`、`S2P1_tiling_glossary.md`、`S2P1_route_dispatch_model.md` 均已产出。

派 1 个独立子 agent。主 Agent prompt：

```text
在 {repo_root} 中为 {whitebox_dir} 从空状态执行完整 Step 5。
首先读取并严格遵守：
{skill_base}/references/case-mapper/00-execution-order.md

完成后报告文件、命令结果、case 数量、Step 5.3 empty 生成结果、Step 5.4 high 生成结果、Step 5.5 final low/high schema 验收结果和阻塞项。
```

- low 用于路径覆盖、网络 shape 和空 tensor 的常规白盒用例集
- high 用于 data_range 展开的信息性白盒用例集

---

## TTK 模块：CSV 生成与 kernel 验收

TTK 模块用于将 Step 5 final case 文件转换为 TTK CSV，并执行环境 precheck 与 low/high 单用例 TTK kernel 验收。当前执行规则、模式支持范围和内部验收行为由入口大纲统一定义。

入口大纲：

```text
{skill_base}/references/ttk-converter/00-execution-order.md
```

核心产物：

| 文件 | 说明 |
|------|------|
| `ttk_{op_name}_cases_low.csv` | low 档位 TTK CSV |
| `ttk_{op_name}_cases_high.csv` | high 档位 TTK CSV |
| `ttk_precheck_report.json` | TTK 环境和 kernel gate 报告 |
| `ttk_module_report.json` | TTK 模块统一结构化报告 |

Agent 对 TTK 模块给结论时，以 `ttk_module_report.json` 的 `acceptance.accepted` / `acceptance.status` 为准。

---

## 最终产物

```
{算子源码路径}/tests/whitebox/
├── S2P0_scout_t.md
├── S2P0_scout_t.json
├── S2P0_scout_k.md
├── S2P0_scout_k.json
├── S2P0_file_manifest.json
├── S2P0_source_scope.md
├── S2P1_route_dispatch_model.md
├── S2P1_route_extract.py
├── S2P1_case_generator.py
├── S2P1_route_cases.json
├── S2P1_cases.json
├── S2P1_path_list.json
├── S2P1_tiling_glossary.md
├── S2P1_low_configs.json
├── S2P1_operator_model.json
├── S2P1_postcheck_report.json
├── S3_diversity_report.md
├── S5_variable_semantics.md
├── S5_data_range_policy.json
├── S5_case_mapper.py
├── S5_mapped_cases_path.json
├── S5_mapped_cases_network.json
├── S5_mapped_cases_low_shape.json
├── S5_cases_low.json
├── S5_cases_high.json
├── ttk_precheck_report.json
├── ttk_{op_name}_cases_low.csv
├── ttk_{op_name}_cases_high.csv
├── ttk_{op_name}_cases_low_one_result.csv
├── ttk_{op_name}_cases_high_one_result.csv
└── ttk_module_report.json
```

其中 `ttk_{op_name}_cases_low/high_one_result.csv` 仅在 kernel 验收实际执行（`kernel_gate` passed）时生成；环境缺失导致验收跳过时不产出。

## 参考提示词索引

| Step | 提示词文件 | 执行方 | 执行顺序节 |
|------|-----------|--------|-----------|
| 1 | `S1-input-collection.md` | 主 Agent | `全局约束（强制）` |
| 2 | `source-analysis/00-execution-order.md` | 主 Agent | `Step 2：源码分析任务` |
| 3 | `scripts/s3_diversity_report.py` | 主 Agent | 多样性分析 |
| 5 | `case-mapper/00-execution-order.md` | 子 agent | `执行顺序约束（强制）` — 5 步（5.1 语义 → 5.2 mapper → 5.3 empty → 5.4 high → 5.5 final schema check） |
| TTK模块 | `scripts/run_ttk_kernel_module.py` | 主 Agent（Step 5.5 通过后必须执行） | wrapper 串行执行 CSV 生成/校验/precheck/单用例验收 |
