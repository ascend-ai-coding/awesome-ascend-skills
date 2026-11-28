# 模块接口

DESIGN.md 说明模块如何划分及其原因，`custom/<op>/eval/module_interfaces.yaml` 记录每个模块的输入来源和输出。模块数等于 `modules` 中的条目数，并与 DESIGN.md 一致。

## 输入资料

读取 DESIGN.md 的模块边界、SPEC.md 的规格和 golden 的实际签名及返回值。
核对每个模块的职责、必须保持的条件、依赖哪些模块、输出供谁使用，以及跨迭代状态、归约、布局和对齐要求。API 签名从目标版本文档获取；通用规则引用原文，算子特有条件写入设计。

## 字段

| 字段 | 内容 |
|---|---|
| `schema_version` | 接口版本，当前为 1 |
| `op` | 算子名称 |
| `primary_inputs` | 按 golden 签名顺序列出 name、shape、dtype，包含必要标量参数 |
| `modules` | 每个模块的 id、name、description、inputs、outputs |
| `final_outputs` | 每个最终返回结果的名称及来源模块 |
| `composition_verification` | SPEC 中的容差、代表性 shape 和验证使用的种子 |

模块输入用 `{name, source}` 引用算子输入或前面模块的输出，shape/dtype 沿用来源定义。
模块输出用 `{name, shape, dtype}` 声明，不能只写“中间结果”。

## 输入来源与输出检查

- 模块 id 从 1 开始连续编号；`source: primary` 引用同名原始输入。
- `source: module_j` 引用前面模块的输出，j 小于当前模块 id，且输出名称匹配。
- 最终输出覆盖 golden 的全部返回值，包括较早模块直接产生的最终结果。
- 模块要求的输入 shape、dtype、布局和含义必须与来源张量一致；需要转换时，明确由哪个模块完成。
- 符号维度使用输入规格中的名称；维度表达式采用读取该文件的工具支持的写法。
- 单模块也需完整声明输入输出，不为满足多模块形式而增加空模块。

## 辅助生成

可使用[接口生成器](../scripts/gen_module_interfaces.py)提取签名并生成草稿：

```bash
python <skills目录>/pypto-op-design/scripts/gen_module_interfaces.py \
  custom/<op>/<op>_golden.py --spec custom/<op>/SPEC.md --op <op> \
  > custom/<op>/eval/module_interfaces.yaml
```

先确保输出目录存在。生成器的默认 dtype、shape 和容差只是草稿值，尤其不能用统一 dtype 替代混合类型输入；按 SPEC 逐项确认，再填写模块边界和最终输出。

使用[接口检查工具](../scripts/validate_yaml.py)检查文件结构和输入来源是否存在。还需组合各模块的 golden，与原始 golden 比较结果。
