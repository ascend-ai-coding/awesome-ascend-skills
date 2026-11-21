# 数据流约束

```yaml
- id: C-DATA-01
  level: must
  rule: "同图内读出张量、计算后写回该张量时检查依赖是否成环；必要时使用独立输出或分图。"
  consequence: "成环的计算图无法完成拓扑排序。"

- id: C-DATA-02
  level: must
  rule: "写回的输入和目标秩一致，偏移及写回范围合法；assemble 前确保输出有效形状正确。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-assemble.md"
  consequence: "写回区域错误或有效数据丢失。"

- id: C-DATA-03
  level: should
  rule: "输出参数的 inplace reshape 需要验证别名、布局和写回行为；不能仅因变形前后元素数相同就认定安全。"
  consequence: "别名或布局处理错误影响输出。"

- id: C-DATA-04
  level: must
  rule: "框架无法推导有效范围时显式设置 view 的 valid_shape；分别处理算术掩码和输出有效区域，不把 valid_shape 当成所有计算的通用掩码。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-view.md"
  consequence: "尾块或无效元素参与计算导致错误。"
```
