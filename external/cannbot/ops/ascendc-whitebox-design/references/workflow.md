# White-box Test Case Generation Workflow

> ⚠️ **全流程约束规则的唯一定义在 SKILL.md「执行约束（强制）」节。** 主 Agent 和子 agent 执行前必须核对。

---


## Step 1：输入收集

> **前置条件**：无（流程起始步）。

主 Agent 直接执行。按 1.0 → 1.1 → 1.2 → 1.3 → 1.4 严格逐步收集平台参数和用户输入，最后展示摘要并等待用户确认。详细规则见 `references/S1-input-collection.md`。

> **关键串行要求**：1.2（定位算子路径）必须在 1.1（收集用户输入）完成后才能执行。1.1 会询问用户选择"自动查找"还是"手动输入路径"，只有确认后才能决定是否执行 `find` 搜索。question 工具返回前，禁止执行任何路径搜索操作。

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

主 Agent 按阶段调度子 agent 完成源码分析。执行顺序、各 Phase 的详细规则和文件索引见 `references/source-analysis/00-execution-order.md`。

**输出文件**：

| 文件 | 说明 |
|------|------|
| `S2P0_scout_t.md` | tiling 侦察报告（入口函数 + P0/P1/P2 文件清单） |
| `S2P0_scout_t.json` | tiling 侦察结构化数据（供 verify 消费） |
| `S2P0_scout_k.md` | kernel 侦察报告（dispatch 类型 + key 列表） |
| `S2P0_scout_k.json` | kernel 侦察结构化数据（供 verify 消费） |
| `S2P0_file_manifest.json` | 源码文件清单（tiling/kernel 优先级 + 排除列表） |
| `S2P0_source_scope.md` | 源码读取范围（供 Task A/D 直接读取） |
| `S2P1_path_list.json` | 代码路径清单 + 分支树 |
| `S2P1_tiling_glossary.md` | tiling 变量含义表（tiling 源码变量名 → 语义名映射） |
| `S2P1_operator_model.json` | 算子接口模型（inputs/outputs/attributes） |
| `S2P1_low_configs.json` | 常见网络 shape 配置（语义参数名） |
| `S2P2_analysis_data.json` | 紧凑格式分析数据（LLM 生成） |
| `S2P2_dim_spec.json` | 维度范围规格（由 assemble_dim_spec.py 生成） |
| `S2P2_param_def_groups.json` | 推导数据文件（由 pick_dims.py 自动生成） |
| `S2P2_reachability_data.json` | 可达性 + 分组中间数据（update_path_list.py 输入） |
| `S2P2_param_def.json` | 参数定义 + 约束 + 分组 + tiling_keys（由 builder 自动生成） |
| `S2P2_gen_cases.py` | 参数组合枚举脚本 |
| `S2P2_cases.json` | 参数组合枚举结果 |
| `S2P2_traceability.md` | 推导追溯文档 |
| `S2P3_test_design.md` | 测试设计文档 |

---

## Step 3：Task D Contract Gate

> **前置条件**：Step 2 全部 Phase（0/1/2/3）已完成；`S2P3_test_design.md` 已生成；Phase 3a 的 disputed 路径已由用户确认（无 disputed 则自动满足）。

主 Agent 直接运行 Task D Contract Gate 脚本；Step 3 只做最终用例 JSON 的覆盖率校验，不设置第二层 LLM soft review，不再复验 Task A/B/C 的源码事实。执行规则见 `{skill_base}/references/design-verifier/00-execution-order.md`。

```bash
python3 {skill_base}/scripts/s3_task_d_gate.py \
  --output-dir {op_path}/tests/whitebox
```

**输入文件**：每个步骤按需读取所需文件（详见 `design-verifier/00-execution-order.md` 执行顺序约束表），不提前读取后续步骤的文件。

输出：`S3_verification_report.md` 和 `S3_verification_report.json`。fail 项 → 回 Step 2 Task D 修正参数定义或重新生成 case 文件（Phase 0/1 不需重跑，最多 3 轮），仍 fail → 触发轮次耗尽协议；pass 项 → 无需处理。

---

## Step 4：用户确认

> **前置条件**：Step 3 已完成，`S3_verification_report.md` 已生成；验证结论已更新到 `S2P3_test_design.md`。

将验证结论更新到 S2P3_test_design.md `## 8. Step 3 验证结论（原 §9 验证结论）` 后停下来等待用户确认。提示语："源码分析和 Task D 契约门禁已完成：{N} 个测试 group，S2P2 参数组合 ~{M} 条（最终用例数含网络和空 tensor 变体可能更多），验证状态 {status}。"

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

> **前置条件**：Step 4 用户已确认；`S2P2_cases.json`、`S2P1_operator_model.json`、`S2P1_low_configs.json` 和 `S2P1_tiling_glossary.md` 均已产出。

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

## TTK模块：TTK CSV 生成

TTK 模块用于将 Step 5 final case 文件转换为 TTK CSV。当前执行规则、模式支持范围和内部验收行为由入口大纲统一定义。

入口大纲：

```text
{skill_base}/references/ttk-converter/00-execution-order.md
```

核心产物：

| 文件 | 说明 |
|------|------|
| `ttk_{op_name}_cases_low.csv` | low 档位 TTK CSV |
| `ttk_{op_name}_cases_high.csv` | high 档位 TTK CSV |
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
├── S2P1_path_config.json          ← Task A 中间文件（build_path_list.py 输入）
├── S2P1_path_list.json
├── S2P1_tiling_glossary.md
├── S2P1_low_configs.json
├── S2P1_operator_model.json
├── S2P2_analysis_data.json      ← LLM 生成的紧凑格式分析数据
├── S2P2_dim_spec.json            ← 由 assemble_dim_spec.py 生成的维度范围规格
├── S2P2_param_def_groups.json    ← 由 pick_dims.py 从 dim_spec 生成
├── S2P2_reachability_data.json  ← 可达性 + 分组中间数据（update_path_list.py 输入）
├── S2P2_param_def.json          ← 由 build_param_def.py 从 groups 文件生成
├── S2P2_gen_cases.py
├── S2P2_cases.json
├── S2P2_traceability.md
├── S2P3_test_design.md
├── S3_verification_report.md
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
├── ttk_module_report.json
```

## 参考提示词索引

| Step | 提示词文件 | 执行方 | 执行顺序节 |
|------|-----------|--------|-----------|
| 1 | `S1-input-collection.md` | 主 Agent | — |
| 2 | `source-analysis/00-execution-order.md` | 主 Agent | `执行顺序约束（强制）` — 10 行 |
| 3 | `design-verifier/00-execution-order.md` + `scripts/s3_task_d_gate.py` | 主 Agent | Task D Contract Gate（D1 cases coverage + 输出汇总） |
| 5 | `case-mapper/00-execution-order.md` | 子 agent | `执行顺序约束（强制）` — 5 步（5.1 语义 → 5.2 mapper → 5.3 empty → 5.4 high → 5.5 final schema check） |
| TTK模块 | `scripts/run_ttk_kernel_module.py` | 主 Agent（Step 5.5 通过后必须执行） | wrapper 串行执行 CSV 生成/校验/precheck/单用例验收 |
