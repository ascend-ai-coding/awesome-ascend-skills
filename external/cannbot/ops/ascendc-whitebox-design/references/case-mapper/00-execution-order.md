# Case Mapper v1 执行总纲

> **目标**：按顺序将 Step 2 最终事实产物中的 path case 和 network config 翻译为下游可消费的 Mapper-v1 固定 schema cases，并生成 final low/high whitebox cases。

## 读取规则（强制）

1. 读完本文件后，按“执行顺序约束”表逐 Step Read 并执行。**每完成一个 Step 的状态判断后，才能读取下一个 Step 的指导文件、模板或固定脚本。**
2. 禁止在 Step 5.1 之前读取 `01-variable-semantics.md` 以外的 Step 子文件、`s5_case_mapper_template.py` 或 Step 5.3/5.4/5.5 固定脚本；禁止一次性 Read 全部流程文档和模板。
3. 当前 Step 只读取当前行列出的指导文件、模板、固定命令、前置产物及其运行所需的最小输入。不得为了预先规划、补充背景或重复理解而读取后续 Step 内容。
4. 后续 Step 所需语义必须优先通过前序产物传递，例如 `S5_variable_semantics.md`、mapped JSON、`S5_cases_low.json` 和 `S5_cases_high.json`。不得预读后续 Step 或回读 Step 2 推理链补齐。
5. 如果当前 Step 的允许输入和前序产物无法解释必需语义，必须报告 Step 2 语义产物不完整；不得读取禁止输入、源码或后续 Step 内容补齐。

## 执行顺序约束（强制）

以下 Steps 必须按编号顺序逐步骤执行，禁止跳步或抢跑。

| Step | 角色 | 核心产物 | 进入下一步条件 |
|------|------|----------|----------------|
| Step 5.1：理解变量语义 | LLM 分析阶段，整理 mapper 所需语义、`attributes` / `const_inputs` policy 和 data_range policy。 | `S5_variable_semantics.md`、`S5_data_range_policy.json` | 两个产物已写入，mapper 必需字段已有明确语义，且字段去向符合 Mapper-v1 schema。 |
| Step 5.2：生成 Mapper-v1 Low Shape Cases | LLM 代码生成阶段，基于模板生成 `S5_case_mapper.py` 并产出符合 Mapper-v1 固定 schema 的 shape low 中间用例。 | `S5_case_mapper.py`、`S5_mapped_cases_path.json`、`S5_mapped_cases_network.json`、`S5_mapped_cases_low_shape.json` | Step 5.2 产物已写入，脚本无模板残留，输出 JSON 符合 `S5_case_json_schema.md`。 |
| Step 5.3：追加 Empty Low Cases | 固定脚本阶段，读取 shape low 中间用例，机械追加 empty cases，生成最终 low 用例文件。 | `S5_cases_low.json` | 命令退出码为 0，并输出 `PASS: appended empty cases`。 |
| Step 5.4：生成 High Data Range Cases | 固定脚本阶段，读取最终 low 用例和 data_range policy，生成 high 档用例。 | `S5_cases_high.json` | 命令退出码为 0，并输出 `PASS: expanded high data_range cases`。 |
| Step 5.5：Final Low/High Schema 验证 | 固定脚本验收阶段，只检查最终 `S5_cases_low.json` 和 `S5_cases_high.json` 的 Mapper-v1 结构、字段完整性和基础类型。 | 验收结果 | 命令退出码为 0，并输出 `PASS: mapper schema accepted`。 |

固定命令：

```bash
python -m py_compile .opencode/skills/ascendc-whitebox-design/scripts/s5_append_empty.py
python .opencode/skills/ascendc-whitebox-design/scripts/s5_append_empty.py --whitebox-dir <operator>/tests/whitebox
python -m py_compile .opencode/skills/ascendc-whitebox-design/scripts/s5_expand_high.py
python .opencode/skills/ascendc-whitebox-design/scripts/s5_expand_high.py --whitebox-dir <operator>/tests/whitebox
python .opencode/skills/ascendc-whitebox-design/scripts/s5_check_mapper_outputs.py --whitebox-dir <operator>/tests/whitebox
```

Step 5.3 固定脚本位于 `.opencode/skills/ascendc-whitebox-design/scripts/s5_append_empty.py`，Step 5.4 固定脚本位于 `.opencode/skills/ascendc-whitebox-design/scripts/s5_expand_high.py`，Step 5.5 固定脚本位于 `.opencode/skills/ascendc-whitebox-design/scripts/s5_check_mapper_outputs.py`。这些固定脚本都不需要 LLM 修改，也不需要在 whitebox 目录生成本地展开脚本。

**完成标志**：Step 5.1 产出语义与 policy；Step 5.2 产出 path/network 审计产物和 `S5_mapped_cases_low_shape.json`；Step 5.3 产出最终 `S5_cases_low.json`；Step 5.4 产出最终 `S5_cases_high.json`；Step 5.5 最终 low/high schema 验证通过。最终 `S5_case_mapper.py` 无模板残留，Step 5.3/5.4/5.5 固定脚本可直接执行。

## 通用规则

- 前置条件表中标明的条件未全部满足时，禁止启动该 Step 的任何操作。
- 完成当前 Step 的全部子步骤并确认状态判断满足后，才能进入下一 Step。
- 自检失败时，回到对应 Step 修复；修复完成并重新满足状态判断后方可继续。
- Step 5 只允许写入 `S5*` 命名产物；禁止写入、覆盖或新建 `S1*`、`S2*`、`S3*`、`S4*` 文件，这些前缀保留给上游 Step。
- 禁止读取 `S2P2_traceability.md`、`S2P2_param_def.json`、`S2P2_dim_spec.json`、`S2P1_path_list.json` 补字段语义；这些属于 Step 2 推理链、生成规格或路径枚举，不是 Mapper 输入。
- 禁止读取 tiling、kernel、`_def.cpp` 或注册源码重新推导 path、tiling key、mode、接口、shape 或 case 字段语义。
- `S5_variable_semantics.md` 和 `S5_data_range_policy.json` 是 Step 5.1 的产物，不是上游事实来源；事实来源仍是 Step 2 允许读取的最终产物。`S5_variable_semantics.md` 承载审查说明和 mapper 语义，`S5_data_range_policy.json` 承载唯一 data_range policy。
- Mapper-v1 输出 schema 以 `S5_case_json_schema.md` 为准；流程文档不得重新定义或放宽字段结构。

## ID 命名规范

| 类型 | 格式 |
|------|------|
| path audit | `case{index:05d}` |
| network audit | `network{index:05d}` |
| shape low case | `low_case_{index:02d}` |
| empty low case | `low_case_empty_{index:02d}` |
| range high case | `{low_case_id}_range{index:02d}` |
