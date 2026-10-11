# Step 5.2：翻译 Mapper 指导书生成 S5_case_mapper.py

> **职责**：复制 `references/case-mapper/s5_case_mapper_template.py` 为 `S5_case_mapper.py`，只实现模板动态区函数，将 Step 5.1 产出的 `S5_variable_semantics.md` mapper 代码指导书翻译为可执行 Python 代码，并运行脚本生成 path/network 审计产物和 shape low mapped cases。

Step 5.2 是代码翻译执行阶段，不是语义设计阶段。Step 5.2 不重新设计变量语义、`attributes` / `const_inputs` 构造、format 规则、rank 覆盖策略、shape assembly plan、network mapping 或 output 派生规则。

`S5_variable_semantics.md` 是 Step 5.2 的唯一语义来源。Step 5.2 必须按该指导书实现动态区；若指导书缺少可编码规则，必须停止并回到 Step 5.1 补充，不得在 Step 5.2 中猜测或补写语义。

## 输入输出

输入文件：

| 文件 | 用途 |
|------|------|
| `S5_variable_semantics.md` | 唯一语义输入；Step 5.2 必须逐项翻译其中规则，不得从其他文件补语义。 |
| `S2P2_cases.json` | path case 运行输入，由 `S5_case_mapper.py` 静态区完整读取；只作为字段取值来源。 |
| `S2P1_low_configs.json` | network config 运行输入，由 `S5_case_mapper.py` 静态区完整读取；只作为字段取值来源。 |
| `references/case-mapper/s5_case_mapper_template.py` | Step 5.2 脚本模板，复制为 `S5_case_mapper.py` 后只实现动态区。 |

输出文件：

| 文件 | 类型 | 要求 |
|------|------|------|
| `S5_case_mapper.py` | 生成脚本 | 由模板复制并实现动态区形成；静态区不得修改。 |
| `S5_mapped_cases_path.json` | 审计产物 | path base cases，与 `S2P2_cases.json` 一一对应。 |
| `S5_mapped_cases_network.json` | 审计产物 | network base cases，由 `S2P1_low_configs.json` 直接映射得到，并与 `S2P1_low_configs.json` 一一对应。 |
| `S5_mapped_cases_low_shape.json` | 中间产物 | shape low cases，由 Step 5.3 固定脚本补充 empty 后生成最终 `S5_cases_low.json`。 |

Step 5.2 不读取、生成或修改 `S5_data_range_policy.json`。该文件由 Step 5.1 生成、Step 5.4 消费；Step 5.2 只保证 low 档 input `data_range = "normal"`。

Step 5.2 不生成 `S5_cases_low.json` 或 `S5_cases_high.json`。`S5_cases_low.json` 由 Step 5.3 固定脚本生成，`S5_cases_high.json` 由 Step 5.4 固定脚本生成。

## 总体流程

1. 读取 `S5_variable_semantics.md`，确认其中 `dtype`、`format`、`execution values`、`input shapes`、`shape coverage plan`、`outputs`、`network mapping`、`mapper notes` 章节足以直接编码。
2. 复制 `references/case-mapper/s5_case_mapper_template.py` 为当前算子 whitebox 目录下的 `S5_case_mapper.py`。
3. 只编辑 `S5_case_mapper.py` 的动态区，不修改静态区。
4. 将 `S5_variable_semantics.md` 的规则逐项翻译到动态区函数中。
5. 运行 `python -m py_compile S5_case_mapper.py`，确保脚本语法正确。
6. 运行 `python S5_case_mapper.py`，生成 Step 5.2 产物。
7. 确认 `S5_mapped_cases_path.json`、`S5_mapped_cases_network.json` 和 `S5_mapped_cases_low_shape.json` 已生成。最终 low/high schema 验收在 Step 5.5 执行。

如果任一动态区函数需要自行决定变量语义、execution value 字段、format 字段、rank 选择、axis group、shape 拆分、network 字段映射或 output 派生规则，说明 `S5_variable_semantics.md` 尚未满足 Step 5.2 输入要求，必须回到 Step 5.1 补充指导书后再继续。

## 语义来源边界

`S5_variable_semantics.md` 是 Step 5.2 的唯一语义来源。动态区代码必须逐项翻译该文件中的规则，不得从其他输入文件、源码或字段名补充语义。

`S2P2_cases.json` 和 `S2P1_low_configs.json` 只作为运行输入和字段取值来源：

- 可以读取其中的字段值，按 `S5_variable_semantics.md` 指定的规则填充 `attributes`、`const_inputs`、`inputs`、`outputs` 和 `meta`。
- 可以使用其中已有的 shape、dtype、format、id 或网络配置字段，但字段去向必须由 `S5_variable_semantics.md` 明确说明。
- 不得根据字段名、字段值形态、样例分布或 case 数量推断未声明的变量语义、shape 关系、rank 覆盖策略或 output 规则。

动态区实现不得读取以下内容补充语义：

- Step 2 推理链或中间设计产物。
- tiling、kernel、`_def.cpp`、注册源码或 infershape 源码。
- operator model、glossary、path list、dim spec 或 param def 等上游事实文件。

如果 `S5_variable_semantics.md` 缺少实现动态区所需的信息，必须停止并回到 Step 5.1 补充指导书。Step 5.2 不得在 `S5_case_mapper.py` 中临时写入猜测规则，也不得修改或补写 `S5_variable_semantics.md`。

缺失信息包括但不限于：

- 某个字段应进入 `attributes`、`const_inputs`、`inputs`、`outputs` 还是 `meta`。
- dtype 名称如何归一化，input/output dtype 如何同步。
- format 如何构造，默认 `ND` 还是按 path/config 约束写入特定 format。
- path case 的 rank 选择、axis group、product 拆分或 dependent input shape 规则。
- network config 如何映射到 Mapper-v1 schema。
- output 的 dtype、format、param_type、shape 或 optional placeholder 规则。

## 模板边界

`references/case-mapper/s5_case_mapper_template.py` 分为静态区和动态区。Step 5.2 生成 `S5_case_mapper.py` 时，必须复制模板并只实现动态区；静态区不得修改。

静态区固定承担通用流程：

1. 读取 `S2P2_cases.json` 和 `S2P1_low_configs.json`。
2. 遍历 path cases，调用动态区生成 path base cases，并写入 `S5_mapped_cases_path.json`。
3. 遍历 network configs，调用动态区生成 network base cases，并写入 `S5_mapped_cases_network.json`。
4. 将 path/network base cases 分派给动态区 shape hook，生成 shape low cases。
5. 汇总 shape low cases，写入 `S5_mapped_cases_low_shape.json`。
6. 在上述流程中统一完成通用校验、ID 规范化和 low 档 `data_range = "normal"` 归一化。

动态区只承担算子特异逻辑：

- 按 `S5_variable_semantics.md` 构造 base case。
- 按 `S5_variable_semantics.md` 处理 path shape case。
- 按 `S5_variable_semantics.md` 处理 network shape case。
- 按 `S5_variable_semantics.md` 派生完整 V1 outputs descriptor。

动态区不得复制、绕过或改写静态区流程。动态区不自行落盘最终产物，不生成 empty variants，不读取 `S5_data_range_policy.json`，不生成 high/data_range variants。

如果动态区实现需要改变静态区调用链、文件落盘方式、ID 生成规则或 validator 行为，必须先修改模板设计，而不是在算子特异代码中绕开模板。

## 动态区函数翻译要求

动态区只实现模板中已有的函数。每个函数都必须把 `S5_variable_semantics.md` 中对应章节翻译为 Python 代码；不得新增指导书未声明的 `attributes`、`const_inputs`、format、shape、rank、meta、output 或 network mapping 规则。

| 函数 | 翻译来源 | 职责 |
|------|----------|------|
| `build_low_base_case(record, source, index)` | `dtype`、`format`、`execution values`、`input shapes`、`outputs`、`network mapping`、`mapper notes` | 将一条 path case 或 network config 映射为 base case。 |
| `make_path_shape_case(case)` | `format`、`input shapes`、`shape coverage plan`、`outputs`、`mapper notes` | 将一条 path base case 转换为 shape low case。 |
| `make_network_shape_case(case)` | `format`、`network mapping`、`outputs`、`mapper notes` | 将一条 network base case 转换为 shape low case。 |
| `derive_outputs(inputs, attributes, const_inputs, meta)` | `outputs` | 从完整 input 空间、execution values 和 meta 派生完整 V1 outputs descriptor。 |

### Output Descriptor Rule

`derive_outputs(...)` 的返回值就是完整 `case["outputs"]`，不得返回 shape-only map。

- Output Tensor 必须包含 `kind/dtype/format/shape/param_type`。
- Output TensorList 外层必须包含 `kind/dtype/format/param_type/tensor_count/tensors`。
- Output TensorList child 必须包含 `kind/dtype/format/shape`。
- outputs 任意层级不得包含 `data_range`。
- optional output 不输出时保留 descriptor，并使用 `shape = null`。

### Base Case Schema

Step 5.2 生成的所有 mapped cases 必须遵守 `S5_case_json_schema.md` 和 `S5_mapped_case_schema.md` 定义的 Mapper-v1 schema。

`build_low_base_case(record, source, index)` 输出 base case，其中 `source` 为 `"path"` 或 `"network"`。`make_path_shape_case(case)` 和 `make_network_shape_case(case)` 输出 shape low case，由静态区纳入 `S5_mapped_cases_low_shape.json`。

path 和 network 的最终 schema 一致，区别只在字段取值来源和 shape/dtype/format 映射规则。动态区返回的 case 必须符合 `S5_case_json_schema.md`。

### `build_low_base_case(record, source, index)`

输入：一条原始运行输入记录。`source` 由模板静态区传入，动态区不得自行推断或改写：

- `source = "path"` 时，`record` 来自 `S2P2_cases.json`。
- `source = "network"` 时，`record` 来自 `S2P1_low_configs.json`。

输出：一条完整的 base case，必须遵守 Mapper-v1 schema。path 和 network 的输出 schema 一致。

该函数负责翻译 `S5_variable_semantics.md` 中 base mapping 相关规则：

- 构造 `id`、`source`、`attributes`、`const_inputs`、`inputs`、`outputs`、`meta`。
- 按 `dtype`、`format`、`execution values`、`input shapes`、`outputs`、`network mapping`、`mapper notes` 放置字段。
- 保证 low 档 input `data_range = "normal"`。
- 对 optional input/output 使用 `shape = null` 占位，不删除 descriptor。

该函数不生成 shape low case，不生成 empty/high/data_range variants，不写文件，不根据字段名猜测字段去向。

### `make_path_shape_case(case)`

输入：一条 `source = "path"` 的 base case，由 `build_low_base_case(record, "path", index)` 生成。

输出：一条完整的 shape low case，必须遵守 Mapper-v1 schema。返回完整 case，不返回 patch/diff。

该函数负责翻译 `S5_variable_semantics.md` 中 path 分支的 shape 规则：

- 按 `input shapes` 和 `shape coverage plan` 执行 path shape assembly。
- 按指导书派生 dependent input shape。
- 按指导书记录必要的 shape variant meta。
- 按最终 `inputs` 调用 `derive_outputs(inputs, attributes, const_inputs, meta)`，重新写入完整 `outputs` descriptor。

该函数不生成 empty/high/data_range variants，不写文件，不设计指导书未声明的 shape fallback。

### `make_network_shape_case(case)`

输入：一条 `source = "network"` 的 base case，由 `build_low_base_case(record, "network", index)` 生成。

输出：一条完整的 shape low case，必须遵守 Mapper-v1 schema。返回完整 case，不返回 patch/diff。

该函数负责翻译 `S5_variable_semantics.md` 中 network 分支的映射规则：

- 复用或规范化 base case 中已经映射好的 input shape/dtype/format。
- 按 `network mapping` 和 `mapper notes` 补齐必要 meta。
- 按最终 `inputs` 调用 `derive_outputs(inputs, attributes, const_inputs, meta)`，重新写入完整 `outputs` descriptor。

该函数不执行 path-style shape assembly，不选择 rank、不拆 product、不重组 axis group，不生成 empty/high/data_range variants，不写文件。

### `derive_outputs(inputs, attributes, const_inputs, meta)`

输入：完整 `inputs`、`attributes`、`const_inputs` 和 `meta`。

输出：完整 V1 `outputs` descriptor，key 为 output 名，value 为 Output Tensor 或 Output TensorList descriptor。

该函数只翻译 `S5_variable_semantics.md` 的 `outputs` 章节。它不修改 case，不读取外部文件，不区分 path/network 来源。

## 验证与完成条件

Step 5.2 完成前必须执行：

```bash
python -m py_compile S5_case_mapper.py
python S5_case_mapper.py
```

脚本运行后必须生成：

- `S5_mapped_cases_path.json`
- `S5_mapped_cases_network.json`
- `S5_mapped_cases_low_shape.json`

完成条件：

- `S5_case_mapper.py` 不包含 `TODO(operator-specific)`、`<TODO` 或 `NotImplementedError`。
- `S5_mapped_cases_path.json` 与 `S2P2_cases.json` 一一对应。
- `S5_mapped_cases_network.json` 与 `S2P1_low_configs.json` 一一对应。
- `S5_mapped_cases_low_shape.json` 包含 shape low cases。
- 所有 mapped cases 遵守 `S5_case_json_schema.md` 和 `S5_mapped_case_schema.md`。
- 所有 input/output descriptor 均包含非空字符串 `dtype` 和 `format`。
- 所有 low 档 input `data_range` 均为 `normal`。
- outputs 由 mapper 完整生成，后续脚本无需推导任何 output descriptor。
- 最终 low/high schema 验收由 Step 5.5 使用 `s5_check_mapper_outputs.py --whitebox-dir <operator>/tests/whitebox` 执行。

## 禁止事项

- 在执行单个算子的 Step 5.2 时，不修改 `S5_case_mapper.py` 中由模板复制而来的静态区逻辑。
- 动态区实现不得读取源码、Step 2 推理链、operator model、glossary、path list、dim spec 或 param def 补语义。
- 不根据字段名、字段值形态、样例分布或 case 数量猜测变量语义。
- 不在 Step 5.2 修改或补写 `S5_variable_semantics.md`。
- 不读取、生成或修改 `S5_data_range_policy.json`。
- 不生成 `S5_cases_low.json` 或 `S5_cases_high.json`。
- 不生成 high/data_range variants。
- 不生成“输入输出”章节未列出的额外中间产物。
- 不把 path-style shape assembly 用到 network 分支。
- 不把 `S5_variable_semantics.md` 未声明的字段写入 `attributes`、`const_inputs`、`inputs`、`outputs` 或 `meta`。
