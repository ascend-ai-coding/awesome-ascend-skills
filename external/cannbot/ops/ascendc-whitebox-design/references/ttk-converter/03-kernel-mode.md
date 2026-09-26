# Kernel 模式脚本规则

Kernel 模式由 `run_ttk_kernel_module.py` 固定执行。常规流程不要求 LLM 手写 CSV、逐字段复核 CSV 或手动拼接 `python3 -m ttk kernel` 命令。

TTK 模块默认信任已通过 Mapper-v1 Step 5.5 校验的 `S5_cases_low.json` / `S5_cases_high.json`。Kernel 模式不重新校验 Mapper-v1 schema，仅在 CSV 生成失败、CSV 行数不一致、CSV 格式校验失败或 TTK kernel 基本执行链路失败时报告失败。

## 输入

| 输入 | 用途 |
|---|---|
| `S5_cases_low.json` | Step 5 final low cases，生成 low CSV。 |
| `S5_cases_high.json` | Step 5 final high cases，生成 high CSV。 |
| `{op_name}` | 作为 CSV `op_name` 列。 |
| `{ops_test_kit_path}` | 执行 `python3 -m ttk kernel` 的 TTK 工具路径。 |

Step 5 final case 的 schema 由 Mapper-v1 Step 5 文档定义，并已由 Step 5.5 最终校验脚本检查字段完整性。Kernel CSV 生成脚本直接读取 `attributes`、`const_inputs`、`inputs`、`outputs` 和 `meta`。

## 输出

| 输出 | 说明 |
|---|---|
| `ttk_{op_name}_cases_low.csv` | low 档位 Kernel CSV。 |
| `ttk_{op_name}_cases_high.csv` | high 档位 Kernel CSV。 |
| `ttk_precheck_report.json` | TTK 环境和 kernel gate 报告。 |
| `ttk_{op_name}_cases_low_one_result.csv` | low 单用例 TTK kernel 验收结果。 |
| `ttk_{op_name}_cases_high_one_result.csv` | high 单用例 TTK kernel 验收结果。 |
| `ttk_module_report.json` | wrapper 统一结构化报告。 |

## Wrapper 流程

1. 调用 `scripts/ttk_generate_kernel_csv.py` 生成 low/high CSV。
2. 调用 `scripts/ttk_validate_csv.py` 校验 low/high CSV，并对账 CSV 行数与 low/high case 数量。
3. 调用 `scripts/ttk_precheck_env.py` 生成 `ttk_precheck_report.json`。
4. `kernel_gate.status == "passed"` 时，串行执行 low/high 各 1 个 `python3 -m ttk kernel --golden-mode Disable` 用例。
5. 解析 result CSV 和日志关键状态，生成 `ttk_module_report.json`。

`ttk_precheck_env.py` 只检查 TTK/CANN 环境是否能启动 kernel 命令，不检查 case JSON schema，不检查额外函数注册。

TTK kernel 验收固定使用 `--golden-mode Disable`，禁用参考计算函数调用，只验证 TTK 能消费 CSV、完成 kernel 编译并启动基本执行链路。

## CSV 生成规则

CSV 列顺序必须与 `02-kernel-fields.md` 的 Kernel CSV 列顺序完全一致。CSV 的 26 个字段全部由 `ttk_generate_kernel_csv.py` 生成，不要求 LLM 手写或复核字段提取脚本。

| 规则 | 说明 |
|---|---|
| `attributes` | 直接合并 Step 5 final case 的 `attributes` 和 `const_inputs`。 |
| `inputs` | 从 Step 5 final case 的 input descriptor 生成 shape、dtype、format 和 data range。 |
| `outputs` | 从 Step 5 final case 的 output descriptor 全量生成 shape、dtype、format 和 tolerance。 |
| `shape = null` | JSON `null` 写为 Python literal `None`，并保留 dtype、format 和位置结构。 |
| `output_inplace_indexes` | 固定写 `()`；必要的同名 inplace 由 TTK op_info 自动推导。 |
| `output_shape_unknown_indexes` | 固定写 `()`。 |
| ori 系列字段 | 留空，交由 TTK 回退。 |
| `manual_input_binaries` / `manual_golden_binaries` | 固定写 `()`，保持 TTK Kernel CSV 标准列。 |

`attributes` 生成形式：

```text
attributes = {
  **case.attributes,
  **case.const_inputs,
}
```

`output_inplace_indexes = ()` 会触发 TTK 的 op_info 自动推导。TTK 根据 op_info 中 input/output 名称同名关系推导 inplace 输入索引；CSV 生成器不根据算子名、shape 或 case 分布自行推断。

## TensorList 规则

TensorList 嵌套格式以 `02-kernel-fields.md` 为准。固定脚本负责将 Step 5 final case 中的 TensorList descriptor 转换为 TTK Kernel CSV 需要的嵌套 tuple：

- `input_shapes` / `output_shapes`：展开到每个子 tensor。
- `input_dtypes` / `output_dtypes`：使用压缩嵌套 dtype。
- `input_data_ranges` / `precision_tolerances`：展开到每个子 tensor。

## data_range 映射

固定脚本按每个输入 descriptor 的 `data_range` 标签生成 TTK `input_data_ranges`。

| data_range | `input_data_ranges` 生成方式 |
|---|---|
| `normal` | 固定 seed 随机范围。 |
| `zero` | `(0, 0)`。 |
| `extreme` | `(dtype_max * 0.9, dtype_max)`。 |
| `negative` | 固定 seed 负数随机范围。 |
| `tiny_pos` | `(1e-7, 1e-5)`。 |
| `all_ones` | `(1, 1)`。 |
| `near_zero` | `(-0.01, 0.01)`。 |
| `with_inf` | `(1, float('inf'))`。 |
| `with_nan` | `(nan, nan)`。 |

固定 seed 为 `42`，保证 CSV 生成可复现。

## tolerance 映射

固定脚本按输出 dtype 生成 tolerance：

| dtype | `(rtol, atol)` |
|---|---|
| `float16` | `(0.001, 0.001)` |
| `bfloat16` | `(0.001, 0.001)` |
| `float32` | `(0.0001, 0.0001)` |
| 其他 dtype | `(0.001, 0.001)` |

## Kernel Gate

`kernel_gate.status` 只由 TTK/CANN 环境决定：

| status | 含义 |
|---|---|
| `passed` | `ops-test-kit` 可用，且 `python3 -m ttk kernel --help` 在当前环境或 source setenv 后可运行。 |
| `skipped` | `ops-test-kit` 不可用，或 TTK/CANN kernel 环境不可用。 |

若 precheck 报告 `checks.cann_env.status == "passed_after_source"`，wrapper 使用报告中的 `setenv_path` 执行 low/high TTK kernel 命令。

## 结果判定

常规流程只读取 `ttk_module_report.json` 判断模块结果。

| status | 含义 |
|---|---|
| `passed` | CSV 生成/校验通过，TTK 能消费 CSV，low/high 单用例在禁用参考计算模式下 `dyn_precision=SUPPRESSED`，且 result CSV 内联分析通过。 |
| `skipped` | CSV 生成/校验通过，但 kernel 验收被跳过。 |
| `failed` | CSV 阶段失败，或 TTK 无法消费 CSV/完成基本执行链路，或 result CSV 内联分析不通过。 |

TTK kernel low/high 单用例串行执行，禁止并行执行，避免 TTK profiling/profiler 目录争用。

TTK kernel 验收不写单用例日志文件。执行结果以命令返回码和 result CSV 为准，命令 stdout/stderr tail 保存在 `ttk_module_report.json` 对应 step 中。

`--golden-mode Disable` 下，result CSV 内联分析要求目标 `testcase_name` 行存在，`dyn_precision` 为 `SUPPRESSED`，且 `memory_oob_status` 为空或 `PASS`。`dyn_precision` 对应 ttk-info.log 中的 `DYN_GOLD`，用于确认 DYN 链路未出现 `INVALID_TILING` 等前置错误；`precision_status` 不参与功能验收门禁。

## 失败处理

| 失败点 | 处理 |
|---|---|
| CSV 生成失败 | 检查 `ttk_generate_kernel_csv.py` 和 Step 5 final case。 |
| CSV validator 失败 | 检查 CSV 字段格式和 TensorList 嵌套格式。 |
| CSV 行数不一致 | 检查 low/high case 文件是否被并发修改，重新生成 CSV。 |
| `env_unavailable` | 检查 `ops-test-kit` 路径、CANN 环境变量和 setenv 脚本。 |
| kernel 命令返回非 0、无 result CSV 或 `dyn_precision` 非 `SUPPRESSED` | 核查 case shape/dtype 是否违反算子约束，并查看 `ttk_module_report.json` 中对应 step 的 stdout/stderr tail。 |
| 目标 testcase 行缺失 | 检查 wrapper 选择的 testcase_name 与 TTK result CSV 输出是否一致。 |
| `memory_oob_status` 非 PASS/空 | 单 case 复现，必要时进入内存/崩溃调试流程。 |
| profiler 目录争用或 `Directory not empty` 等并发痕迹 | 按 low → high 顺序重跑单用例，不直接判为 case 失败。 |
