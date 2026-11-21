# 约束

约束描述 PyPTO 设计和实现必须遵守的边界。设计文档引用稳定 ID，不复制规则。

```yaml
- id: C-AREA-01
  level: must | must_not | should
  rule: 规则本身
  when: 适用条件（可选）
  source: 事实来源（可选）
  consequence: 违反后的结果
```

规则按主题分布在 `api.md`、`tiling.md`、`loop.md`、`symbolic.md`、`dataflow.md`。
