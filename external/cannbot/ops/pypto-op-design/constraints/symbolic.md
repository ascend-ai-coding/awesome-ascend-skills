# 符号值约束

```yaml
- id: C-SYM-01
  level: must_not
  rule: "不假定 SymbolicScalar 支持 Python 幂运算；静态指数计算尽量在编译前完成，动态表达式核对目标版本。"
  consequence: "不支持的运算导致构图失败。"

- id: C-SYM-02
  level: must_not
  rule: "不假定 SymbolicScalar 支持取模；使用替代表达式前确认除法、负数及除数非零的语义。"
  consequence: "结果可能与参考运算不同。"

- id: C-SYM-03
  level: must_not
  rule: "不使用符号值索引 Python list 等容器；Tensor 索引支持范围以 Tensor API 为准。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop.md"
  consequence: "Python 容器要求具体整数。"

- id: C-SYM-04
  level: must_not
  rule: "不对符号值执行 Python if 判断；运行时分支使用框架控制流。"
  consequence: "符号没有可供 Python 判断的运行时真值。"

- id: C-SYM-05
  level: must
  rule: "符号最小值和最大值使用 sym.min(x)、sym.max(x)，不用 Python min/max。"
  source: "PyPTO docs/zh/api/tensor_api/symbolic/pypto-SymbolicScalar-min.md"
  consequence: "Python 比较可能要求符号真值。"

- id: C-SYM-06
  level: must_not
  rule: "不将符号值作为 Python range 的边界，使用框架循环。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop.md"
  consequence: "无法转换为具体整数。"

- id: C-SYM-07
  level: must
  rule: "pypto.view 的 shape 使用 List[int]，符号值用于 offsets 和 valid_shape。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-view.md"
  consequence: "符号 shape 不受该 API 支持。"
```
