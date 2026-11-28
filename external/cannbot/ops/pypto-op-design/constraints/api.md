# API 约束

```yaml
- id: C-API-01
  level: must
  rule: "sum 的输入 dtype 按目标版本和设备文档选择；需要提高累加精度时显式转换为 FP32。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-sum.md"
  consequence: "不支持的类型会调用失败，低精度累加可能引入误差。"

- id: C-API-02
  level: must
  rule: "matmul 的两侧输入满足目标 API 的 dtype 配对要求，转换位置在计算图中明确。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-matmul.md"
  consequence: "不支持的 dtype 配对会编译失败。"

- id: C-API-03
  level: must
  rule: "amax 的输入使用其 API 文档支持的 dtype，并单独评估低精度输入的误差。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-amax.md"
  consequence: "调用失败或误差超出要求。"

- id: C-API-04
  level: must
  rule: "exp 的输入使用 FP16、BF16 或 FP32。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-exp.md"
  consequence: "整数输入不受支持。"

- id: C-API-05
  level: should
  rule: "精度敏感的归约和跨循环累加优先使用 FP32；转换位置保持与参考计算的数值要求一致。"
  consequence: "舍入误差可能随迭代积累。"

- id: C-API-06
  level: must
  rule: "广播按具体二元 API 的 shape 规则处理；不支持的多轴组合先显式扩展或变形。"
  consequence: "输入 shape 不兼容。"
```
