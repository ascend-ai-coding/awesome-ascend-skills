# Tiling 约束

```yaml
- id: C-TILE-01
  level: must
  rule: "TileShape 的秩按具体 API 设置：例如 sum 对齐输入秩，exp 对齐输出秩；不能一律按输出判断。"
  source: "PyPTO docs/zh/api/tensor_api/operation/pypto-sum.md"
  consequence: "归约移除维度后沿用配置可能导致秩不匹配。"

- id: C-TILE-02
  level: must
  rule: "要求尾轴 32B 对齐的操作满足 tile_last × dtype_bytes % 32 = 0；FP32 为 8 个元素，FP16/BF16 为 16 个元素。"
  consequence: "对齐检查失败。"

- id: C-TILE-03
  level: must
  rule: "按目标设备可用 UB 和实际驻留张量估算容量；计入中间结果、缓冲副本及数据类型。"
  consequence: "资源不足或额外搬运。"

- id: C-TILE-04
  level: should
  rule: "估算张量切分后的 tile 数与展开规模，结合实际编译结果控制图大小；不把固定表达式数量当作跨版本上限。"
  consequence: "编译耗时和内存占用上升。"

- id: C-TILE-05
  level: must
  rule: "矩阵乘的 m/k/n 各轴使用 [L0, L1] 配置，满足 0 < L0 <= L1 且 L1 % L0 == 0；Vector 配置不能替代 Cube 配置。"
  source: "PyPTO docs/zh/api/tensor_api/config/pypto-set_cube_tile_shapes.md"
  consequence: "矩阵乘配置不符合 API 要求。"

- id: C-TILE-06
  level: must
  rule: "在对应操作前设置 TileShape；参数使用编译期整数或可解析为整数的常量，不能来自运行时 shape、kernel 参数或 SymbolicScalar。"
  source: "PyPTO docs/zh/api/tensor_api/config/pypto-set_vec_tile_shapes.md"
  consequence: "切分配置不正确。"

- id: C-TILE-07
  level: should
  rule: "初始 tile 依据形状、对齐与容量选择；不把所有轴固定限制在 16 到 64。"
  consequence: "不合适的固定值可能增加切分开销或浪费资源。"

- id: C-TILE-08
  level: should
  rule: "计算形状发生变化时重新核对当前 Vector TileShape，必要时局部调整。"
  consequence: "旧配置可能不适合后续操作。"
```
