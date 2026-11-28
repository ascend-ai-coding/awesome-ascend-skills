# Step 5.2：根据 Mapper 语义说明生成 Mapper

## 1. 目的

Step 5.2 根据当前算子的 `S5_variable_semantics.md` 编写并运行 `S5_case_mapper.py`。

Step 5.2 的任务是把 `S2P1_cases.json` 中的 path seed case 和 `S2P1_low_configs.json` 中的 network config 转换为 Mapper-v1 cases。具体的变量映射、rank/shape 泛化、shape 组装、输入输出派生和 network 处理方式，以当前算子的 `S5_variable_semantics.md` 为准。

Step 5.2 不为所有算子规定统一的 mapper 实现方式。函数划分、rank 遍历方式、shape 展开方式、variant 组织方式和审计信息组织方式，由当前算子决定。

## 2. 输入

必需输入：

- `S5_variable_semantics.md`：当前算子的唯一 mapper 语义说明。
- `S2P1_cases.json`：path seed cases 及其 `case_values`。
- `S2P1_low_configs.json`：network config。

实现参考：

- `references/case-mapper/s5_case_mapper_template.py`：可复用的 Mapper-v1 通用工具和校验骨架。是否使用模板中的具体函数结构，由当前算子实际情况决定。

Step 5.2 不读取其他上游文件补充语义。若 `S5_variable_semantics.md` 缺少实现所需规则，必须回到 Step 5.1 补充，不得在 Step 5.2 自行猜测。

## 3. 输出

必须生成：

- `S5_case_mapper.py`
- `S5_mapped_cases_path.json`
- `S5_mapped_cases_network.json`
- `S5_mapped_cases_low_shape.json`

其中：

- path mapped cases 应逐个对应 `S2P1_cases.json` 的 seed case。
- network mapped cases 应逐个对应 `S2P1_low_configs.json` 的 network config。
- low shape cases 应按照当前算子的语义说明生成一个或多个合法 variant。
- 所有输出必须符合 Mapper-v1 schema。

Step 5.2 不生成或修改：

- `S5_data_range_policy.json`
- `S5_cases_low.json`
- `S5_cases_high.json`
- 任何 `S1*` / `S2*` / `S3*` / `S4*` 文件

## 4. 实现要求

Step 5.2 必须完整实现当前算子的 `S5_variable_semantics.md`，包括：

- `case_values` 到 `inputs`、`outputs`、`attributes`、`const_inputs` 和 `meta` 的映射。
- 当前算子声明的整体全合法 rank 覆盖、均匀 rank 分布、shape 泛化和合法性判断。
- shape 变量的展开、拼接、同步、broadcast 或派生。
- dependent input 和 output descriptor 的构造。
- 当前算子的 attribute、const input、dtype、format 和 network config 映射。
- 元素数固定的输入 Tensor 按普通 Tensor 输入生成：descriptor 写入 `inputs`（`kind=tensor`、`shape=[n]`，n 为当前算子 `S5_variable_semantics.md` 声明的固定元素数，含 `data_range`）；仅元素数固定、元素值并不固定，取值由 `data_range` 生成，与普通输入一致；不得把值写入 `const_inputs` 或 `attributes`。
- 必要的变量反推或一致性校验。

如果一个 seed case 对应多个合法 rank/shape variants，必须按 `S5_variable_semantics.md` 的说明生成这些 variants；不能擅自只保留一个，也不能擅自增加文档未声明的 variants。Step 5.2 不得破坏 `S5_variable_semantics.md` 声明的整体合法 rank 覆盖和均匀分布目标；不要求每个 seed case 生成所有合法 rank variants。rank 泛化时应遵守其中的均衡拆分要求，避免用多个 1 凑 rank。

实现可以使用模板提供的通用 descriptor 工具，但不得让模板的默认函数结构限制当前算子的实际语义。一个 seed case 可以生成一个或多个输出 case，当前算子应选择能准确表达其语义的组织方式。

### 易错点：输出 tensor 为空 / empty tensor 不是 None

Step 5.2 翻译 `S5_variable_semantics.md` 时，empty tensor、`shape = null` 与 TensorList 数量的唯一权威为 `S5_case_json_schema.md` §5（TL1–TL3）与 §6，本节不重复定义。易错提示：输出 tensor 为空时，REQUIRED 输出不得用 `null`，必须写 `[0]`（TTK `(0,)`）；empty tensor 不能写成 `null`。

- 元素数固定的输入是普通 Tensor：descriptor 进入 `inputs`（`shape=[n]`），不得写入 `const_inputs` 做值注入、不得用 `attributes` 覆盖、不得只写 `const_inputs` 而遗漏 `inputs` 中的同名 descriptor。

## 5. 执行与验证

至少执行：

```bash
python -m py_compile S5_case_mapper.py
python S5_case_mapper.py
```

执行后确认以下文件已生成：

- `S5_mapped_cases_path.json`
- `S5_mapped_cases_network.json`
- `S5_mapped_cases_low_shape.json`

Step 5.2 生成的 input/output descriptor 必须满足当前算子的语义约束；low 档 input 的 `data_range` 使用 `normal`。最终 low/high 文件由后续固定步骤生成。

## 6. 基本边界

- 语义来源是当前算子的 `S5_variable_semantics.md`。
- 不重新分析源码或 Step 2 推理链。
- 不修改 `S2P1_*` 产物。
- 不修改 `S5_variable_semantics.md` 或 `S5_data_range_policy.json`。
- 不为了增加 case 数量生成无法由当前算子约束解释的 case。
- 不输出不符合 Mapper-v1 schema 的 descriptor。
