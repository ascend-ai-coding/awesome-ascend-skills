# Step 5.1：Mapper 语义说明生成

## 1. 目的

Step 5.1 为当前算子生成 `S5_variable_semantics.md`，供 Step 5.2 编写 `S5_case_mapper.py`。

Step 5.1 只需要完成三件事：

1. 梳理 `S2P1_cases.json` 中每个 case 的变量如何映射到 Mapper-v1 的 `inputs`、`outputs`、`attributes`、`const_inputs` 和 `meta`。
2. 梳理当前算子、当前 branch 和当前 case 在 Step 5.2 中应生成的 rank/shape 泛化结果及边界；rank 泛化应在当前算子、branch 或 case class 的整体 case 集合中覆盖所有合法 rank，并使 rank 分布尽量均匀。
3. 生成 `S5_data_range_policy.json`，说明 high 档 data_range 扩展策略。

Step 5.1 只规定 `S5_variable_semantics.md` 的预期输出，不规定 Step 5.2 的实现流程或固定函数结构。`S5_variable_semantics.md` 必须足以让 Step 5.2 直接实现 mapper，包括 dtype、format、output 派生和必要的反推校验。具体映射、rank 选择、shape 展开、shape 拼接、输入输出派生和 network case 处理方式，均由当前算子的实际情况决定，不使用全局固定方案。

## 2. 输入

Step 5.1 读取 Step 2 的最终事实产物：

- `S2P1_cases.json`：path seed cases 及其 `case_values`。
- `S2P1_tiling_glossary.md`：`case_values` 中变量的语义和 shape 关系。
- `S2P1_route_dispatch_model.md`：route、tiling key、branch 和 kernel 可达性事实。
- `S2P1_operator_model.json`：inputs、outputs、attributes、const_inputs 以及 dtype/rank/shape 约束。
- `S2P1_low_configs.json`：network config 及其 shape/dtype 等字段。

必要时可以读取 `{op_path}/README.md` 辅助理解算子语义；只有在 output shape 信息不足时，才可以读取 `infershape.cpp` 的 output shape 片段。

Step 5.1 不读取其他源码、Step 2 推理链或 `{op_path}/docs/aclnn*.md` 补充 mapper 语义。若允许的 Step 2 产物不足以确定规则，应报告 Step 2 语义产物不完整，不得猜测。

## 3. 输出

Step 5.1 输出：

- `S5_variable_semantics.md`：当前算子的 mapper 语义说明，供 Step 5.2 翻译代码。
- `S5_data_range_policy.json`：输入 tensor 的 high 档 data_range 扩展策略。

Step 5.1 不输出 `S5_case_mapper.py`、mapped cases 或最终 low/high cases。

## 4. `S5_variable_semantics.md` 应说明的内容

文档应覆盖 `S2P1_cases.json` 的每个 seed case，可以按 branch、case class 或共享规则分组组织，但每个 seed case 都必须落到明确的映射和泛化规则上。文档应简明说明：

- `case_values` 中各字段的含义，以及字段进入 `inputs`、`outputs`、`attributes`、`const_inputs` 还是 `meta`。
- tiling 变量如何映射到真实输入/输出空间；如果由 shape 表达，说明其维度展开、拼接、同步、broadcast 或派生关系。
- `S2P1_operator_model.json` 中的 rank/shape 约束如何作用于生成的 descriptor。
- 当前算子、当前 branch、当前 case 的合法 rank 集合，以及 Step 5.2 应生成的 rank 覆盖目标；rank 覆盖应在当前算子、branch 或 case class 的整体 case 集合中覆盖所有合法 rank，并使 rank 分布尽量均匀，不要求每个 seed case 覆盖所有合法 rank。
- 每个合法 rank 下 descriptor 应满足的 shape 语义，包括 product、shape_size、outer dims、suffix dims、broadcast、同步或派生关系。
- 每个 seed case 至少生成一个合法 descriptor，并说明每个 seed case 如何选择合法 rank，或如何按 branch / case class 分配 rank，使整体 rank 覆盖完整且分布均匀。
- rank 泛化时的维度拆分目标：在保持 `case_values` 可反推和算子约束合法的前提下，尽量将元素量均衡分配到各轴，避免使用多个 1 仅为凑 rank。若某个 seed 的 `case_values` 不适合承载被分配 rank，应选择该 seed 可支持的合法 rank，或在同 branch / case class 内换由其他 seed 承载该 rank。
- 若某个 rank 不能生成，必须说明该 rank 被当前算子约束排除的依据。
- 泛化后的 descriptor 如何保持原 seed case 的变量语义，并能够反推必要的 `case_values`。
- output、attribute、const input 和 network config 的当前算子映射方式。
- 必须区分输出的「接口位置不存在」与「空输出」，并显式声明：`param_type == REQUIRED` 的输出禁止使用 `shape = null`；当某 branch 不产生该输出数据时，必须用空 tensor `shape = [0]`（TTK `(0,)`）表达；只有接口位置本身不存在（非 `REQUIRED`）时才能用 `shape = null`。判定依据见 `S5_case_json_schema.md` §6。
- 元素数固定的输入 Tensor 的处理规则：接口声明为 Tensor 输入且元素数量固定时，按普通 Tensor 输入处理。descriptor 必须写入 `inputs`（`kind=tensor`、`shape=[n]`，n 为固定元素数，即一维 Tensor），`param_type` 按接口声明、`dtype`/`format` 照常填写；仅元素数固定、元素值并不固定，其取值与其他输入一样由 `input_data_ranges` 生成；禁止通过 `attributes` 覆盖该输入的值；禁止把该输入的值写入 `const_inputs`；禁止只写入 `const_inputs` 而遗漏 `inputs` 中的同名 descriptor。
- 术语统一：此类输入称为「一维 Tensor（shape=[n]）」，不使用「标量 Tensor」。

泛化规则只描述当前算子的预期结果，不规定所有算子统一采用某个 rank 遍历、shape 拆分或 padding 方式。Step 5.1 必须给出整体覆盖所有合法 rank 且分布尽量均匀的覆盖目标；shape 如何拆分由当前算子语义决定，但输出 descriptor 必须保持 `case_values` 可反推。若存在多个合法泛化结果，应说明这些结果是否都需要生成；Step 5.2 按该说明实现。

## 5. `S5_data_range_policy.json`

该文件只描述各 input 是否参与 high 档 data_range 扩展及支持的标准范围。它不得改变或重新定义 shape、rank、dtype、format、route、tiling key、attributes、const_inputs 或 outputs。

标准范围为：`zero`、`extreme`、`negative`、`tiny_pos`、`all_ones`、`near_zero`、`with_inf`、`with_nan`。`normal` 仅作为 low/base case 的默认范围，不写入 `supported`。

元素数固定的输入 Tensor 与其他输入同等参与 data_range policy；除非有算子约束依据，不得因「元素数固定」将其排除在 `participates`/`supported` 之外。

格式：

```json
{
  "version": 1,
  "mode": "per_input_cyclic",
  "inputs": {
    "input_name": {
      "participates": true,
      "supported": ["zero", "extreme", "negative", "tiny_pos", "all_ones", "near_zero", "with_inf", "with_nan"]
    }
  }
}
```

## 6. 基本边界

- Step 5.1 不重新发现 Step 2 已覆盖的 tiling/kernel 分支。
- 不修改任何 Step 2 产物。
- 参与 descriptor 或 route 语义的变量不能只保存在 `meta`；`meta` 仅保存审计信息。
- 生成的 descriptor 必须符合 Mapper-v1 schema，并保持当前算子的变量语义。
- 不得为了追求 rank/shape 数量生成无法由当前算子约束解释的 case。
- Step 5.1 必须区分 tiling/kernel route 覆盖和 descriptor rank/shape 语义覆盖；不得仅以 route 已覆盖、network shape 已覆盖、case 数量控制或 seed case 已覆盖作为不生成合法 rank variants 的理由。
- 对于 Step 2 事实产物中存在 product、shape_size、outer dims、suffix dims、broadcast 或类似 shape 关系的算子，Step 5.1 必须把这些自由度落实为整体覆盖所有合法 rank 且分布尽量均匀的预期 descriptor 结果。
- 不得把所有 product/shape_size 变量无依据压缩为单一扁平 shape，除非当前算子约束只允许该 rank/shape。
- rank 泛化不得主要依赖插入多个 size=1 的维度来凑 rank。除非当前算子约束要求某些轴为 1，否则应优先把 product 或 shape_size 对应的元素量拆分到多个非 1 维度，并尽量让各轴规模相对均衡。
- 任何 Tensor/TensorList 输入的取值需要由 `data_range` 表达，不得依赖 `attributes` 覆盖或 `const_inputs` 注入输入值。
- `const_inputs` 仅承载框架 ValueDepend 语义，不承载 Tensor/TensorList 输入（含元素数固定的输入）的值。
- Step 5.1 产物中的自然语言说明、规则描述和人工可读信息统一使用中文；代码标识符、字段名、枚举值、路径、宏名、平台名及其他机器消费内容保持原文。
