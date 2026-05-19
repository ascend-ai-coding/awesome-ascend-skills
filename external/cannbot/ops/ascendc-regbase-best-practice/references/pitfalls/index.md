# 陷阱索引

该目录用于定位 RegBase 设计、实现和审查中的常见错误。出现精度异常、构建失败、API 不确定、tail 错误或 MemBase / RegBase 混淆时，先读这里。

## 选择表

| 文档 | 用途 | 何时阅读 |
|---|---|---|
| [常见陷阱](./common_traps.md) | 快速扫查高频错误 | 设计或代码看起来不稳 |
| [API 误用](./api_misuse.md) | 定位 API 家族、签名、层级误用 | 构建失败或审查 API |
| [精度指南](./precision_guide.md) | 设计精度策略 | 需要 dtype/cast/mask 判断 |
| [精度失败](./precision_failures.md) | 从结果异常定位原因 | 输出错误、NaN、tail 错 |
| [RegBase vs MemBase 混淆](./regbase_vs_membase_confusions.md) | 区分两类数据流和 API | 代码混用 LocalTensor 与 RegTensor |
| [症状到原因](./symptom_to_cause.md) | 快速从症状找检查路径 | 修复循环中使用 |

## 使用原则

- 先按症状定位，再回到对应 API、主开发指南或开发经验文档验证。
- 不要只修表面错误；RegBase 问题通常来自分层或 mask 设计不清。
- 审查中发现阻塞问题时，要指向具体层级和修复建议。

## 相关文档

- [[../api/index]]
- [[../dev-experience/index]]
