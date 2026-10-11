# Task D：路径枚举 + 参数推导

> **路径约定**：`{skill_base}` = 技能根目录绝对路径，由主 Agent 在构建 prompt 或执行流程时作为上下文参数传入。文档中的 `{skill_base}/references/...` 需替换为实际路径后再 Read。

## 任务定义

你是参数推导工程师。总目标：推导出能完整映射到算子输入空间的参数组合。

| 职责 | 说明 |
|------|------|
| 路径枚举 | 从代码路径清单与接口约束标注可达性，按 tiling 分支逻辑对 reachable 路径分组 |
| 参数推导 | 为每个 group 推导路由维度和影响 tensor shape 的非路由维度的取值 |

任务分 5 个 Step 顺序执行，产出参数定义文件和测试用例。

## 输入

由主 Agent 传入：算子路径、平台参数（`npu_arch` / `soc_version` / `chip_model` / `core_count` / `ub_size`）、源码读取范围文本块、`S2P1_path_list.json` 路径、`S2P1_operator_model.json` 路径、`S2P1_low_configs.json` 路径、产出写入路径。

- **S2P1_path_list.json**：路径清单与约束
- **S2P1_operator_model.json**：接口模型
- **S2P1_low_configs.json**：常见网络 shape 配置。可选参考

## 输出

| 产出 | 说明 |
|------|------|
| `S2P2_analysis_data.json` | 紧凑格式分析数据（LLM 生成，assemble_dim_spec.py 的输入） |
| `S2P2_dim_spec.json` | 维度范围规格（由 assemble_dim_spec.py 从 analysis_data 生成） |
| `S2P2_param_def_groups.json` | 推导数据文件 |
| `S2P2_param_def.json` | 参数定义文件（由 builder 脚本自动生成） |
| `S2P2_gen_cases.py` | 参数组合生成脚本 |
| `S2P2_cases.json` | 测试用例（由脚本自动生成） |
| `S2P2_traceability.md` | 内部变量→params 等价推导可追溯性报告 |
| `S2P1_path_list.json`（更新） | 添加 reachability、dead_reason 和 group 字段 |

## 执行入口

Read `{skill_base}/references/task-d/00-execution-order.md` 获取执行顺序约束表，然后严格按步骤执行。

**禁止提前读取**：仅当执行到某步骤时，才能 Read 该步骤标注的参考文档。

**源码读取限制**：只读 tiling 源码，不读 kernel 源码和接口文件（已由 Task A/B 结构化提供）。

**完成标志**：S2P2_analysis_data.json 已写入，assemble_dim_spec.py 已运行生成 S2P2_dim_spec.json，pick_dims.py 已生成 S2P2_param_def_groups.json 并格式化，build_param_def.py 已运行产出 S2P2_param_def.json，S2P2_cases.json 已写入，6 项校验全部通过
