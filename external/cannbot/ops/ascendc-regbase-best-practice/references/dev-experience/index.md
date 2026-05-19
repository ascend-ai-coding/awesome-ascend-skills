# 开发经验索引

该目录沉淀 RegBase 开发过程中的工程经验。需要判断路线、避免重复踩坑、准备构建或解释失败案例时，先读这里。

## 选择表

| 文档 | 用途 | 何时阅读 |
|---|---|---|
| [RegBase 构建笔记](./regbase_build_notes.md) | 处理 build/header/路径问题 | 编译前或构建失败时 |
| [RegBase 性能实践](./regbase_performance_practices.md) | 评估性能瓶颈和优化优先级 | 性能审查或调优 |
| [RegBase 编程笔记](./regbase_programming_notes.md) | 编码前的写法提醒，覆盖参考实现阅读、数据流、buffer 分域和同步边界 | 开发者开始实现前 |
| [Tiling 审查笔记](./tiling_review_notes.md) | 设计和审查 tiling packet、tail、dtype route 和 UB 大小 | 设计串讲或 review |
| [DAV_3510 成功/失败案例](./working_vs_failed_dav_3510_cases.md) | 对照成功和失败路径 | 修复或判断路线时 |

## 使用原则

- 这些文档是工程经验，不替代 API 文档或真实源码。
- 如果经验来自兼容路径，应在设计中标注适用范围。
- 经验只能缩短判断路径，不能省略验证。

## 相关文档

- [[../regbase_development_guide]]
- [[../pitfalls/index]]
