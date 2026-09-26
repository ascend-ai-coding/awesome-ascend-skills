# TTK Converter 执行入口

## 角色

将 Step 5 final case 文件转换为 TTK Kernel CSV，并执行 low/high 单用例 TTK kernel 验收。

常规流程只执行 `scripts/run_ttk_kernel_module.py`，不派发 TTK 子 agent，不读取 `01-csv-common.md`、`02-kernel-fields.md` 或 `03-kernel-mode.md`。`01`-`03` 仅作为 kernel 模式内部实现细节、固定脚本维护资料和失败诊断参考。

## 输入

| 参数 | 说明 |
|------|------|
| `{skill_base}` | 技能根目录绝对路径 |
| `{whitebox_dir}` | 白盒测试产出目录，需包含已通过 Step 5.5 校验的 `S5_cases_low.json`、`S5_cases_high.json` |
| `{op_name}` | 规范算子名，如 `add_rms_norm` |
| `{op_path}` | 算子源码根目录，如 `.../ops-nn/norm/add_rms_norm` |
| `{ops_test_kit_path}` | 主 Agent 已在当前工作目录 `$PWD` 下定位到的 `ops-test-kit/` 路径 |

## 常规执行命令

主 Agent 直接执行：

```bash
python3 {skill_base}/scripts/run_ttk_kernel_module.py \
  --op-name {op_name} \
  --whitebox-dir {whitebox_dir} \
  --op-path {op_path} \
  --ops-test-kit-path {ops_test_kit_path} \
  --skill-base {skill_base}
```

如用户提供了非默认 CANN 环境脚本，可追加：

```bash
  --setenv-path {setenv_path}
```

若只需生成/校验 CSV 和 precheck，不执行 kernel 单用例验收，可追加：

```bash
  --skip-kernel-run
```

`--skip-kernel-run` 仅用于脚本调试、CSV 生成链路冒烟或环境问题隔离。正常验收必须执行 TTK kernel 单用例，禁止追加 `--skip-kernel-run`。

## wrapper 内部流程

`run_ttk_kernel_module.py` 串行执行以下固定步骤：

1. 调用 `scripts/ttk_generate_kernel_csv.py` 生成 low/high CSV。
2. 调用 `scripts/ttk_validate_csv.py` 校验 low/high CSV，并对账 CSV 行数与 low/high case 数量。
3. 调用 `scripts/ttk_precheck_env.py` 生成 `ttk_precheck_report.json`。
4. `kernel_gate.status == "passed"` 时，串行执行 low/high 各 1 个 `python3 -m ttk kernel --golden-mode Disable` 用例。
5. 内联分析 result CSV，确认目标 testcase 行存在、`dyn_precision` 为 `SUPPRESSED` 且 `memory_oob_status` 为空或 `PASS`，生成 `ttk_module_report.json`。

## 输出

| 文件 | 说明 |
|------|------|
| `ttk_{op_name}_cases_low.csv` | low 档位 TTK Kernel CSV |
| `ttk_{op_name}_cases_high.csv` | high 档位 TTK Kernel CSV |
| `ttk_precheck_report.json` | TTK 环境和 kernel gate 报告 |
| `ttk_{op_name}_cases_low_one_result.csv` | low 单用例验收结果 |
| `ttk_{op_name}_cases_high_one_result.csv` | high 单用例验收结果 |
| `ttk_module_report.json` | wrapper 统一结构化报告，含步骤状态、耗时和 `acceptance` 最终验收结论 |

## 状态判断

Agent 对 TTK 模块给结论时，只读取 `ttk_module_report.json` 的 `acceptance` 字段：

| acceptance 字段 | 结论 |
|---|---|
| `accepted=true` 且 `status="passed"` | TTK 模块验收通过 |
| `accepted=false` 且 `status="failed"` | TTK 模块验收失败 |
| `accepted=false` 且 `status="skipped"` | TTK 模块验收未完成，不能作为通过结论 |

禁止直接使用 `kernel_gate.status` 作为最终验收结论。`kernel_gate` 只表示 TTK/CANN 环境门禁是否允许进入 kernel 执行阶段；最终结论必须以 `acceptance.accepted` / `acceptance.status` 为准。

`acceptance.accepted=true` 仅表示 low/high result CSV 内联分析均通过。当前内联分析通过条件为：目标 `testcase_name` 行存在，必需字段 `testcase_name` / `dyn_precision` / `memory_oob_status` 存在，`dyn_precision` 为 `SUPPRESSED`，且 `memory_oob_status` 为空或 `PASS`。`dyn_precision` 对应 ttk-info.log 中的 `DYN_GOLD`，在 `--golden-mode Disable` 下用于确认 DYN 链路未出现 `INVALID_TILING` 等前置错误。

### 输入分级

wrapper 将输入分为两类：

| 类别 | 路径 | 缺失处理 |
|------|------|----------|
| CSV 必需 | `S5_cases_low.json`、`S5_cases_high.json`、三个 `ttk_*.py` 固定脚本 | `status=failed`，无法生成 CSV |
| kernel 环境 | `ops-test-kit/`、CANN 环境 | CSV 继续生成；precheck 记录原因；kernel 验收 `skipped` |

### 常见 reason

| reason | 说明 |
|--------|------|
| `missing_required_for_csv` | CSV 必需输入缺失 |
| `csv_row_count_mismatch` / `csv_validation_failed` | CSV 生成结果不满足要求 |
| `skip_kernel_run_requested` | 调试/冒烟场景主动跳过 kernel 验收；正常验收禁止使用 |
| `env_unavailable` | `ops-test-kit`、TTK kernel 或 CANN 环境不可用 |
| `low_kernel_execution_failed` / `high_kernel_execution_failed` | kernel 命令返回非 0，或 result CSV 缺失/空，或目标行缺失，或 `dyn_precision` 非 `SUPPRESSED`，或 `memory_oob_status` 非 PASS/空 |

若 precheck 报告 `checks.cann_env.status == "passed_after_source"`，wrapper 会使用报告中的 `setenv_path` 执行 low/high TTK kernel 命令。

## 失败诊断

仅当 `ttk_module_report.json` 显示 wrapper 失败、CSV validator 失败、TTK kernel 执行失败，或用户要求分析 TTK 字段/规则时，再读取以下内部参考文档：

- `03-kernel-mode.md`：Kernel CSV 生成、字段来源、TTK kernel 验收规则。
- `02-kernel-fields.md`：Kernel CSV 字段定义。
- `01-csv-common.md`：公共 CSV 字段和格式规则。
