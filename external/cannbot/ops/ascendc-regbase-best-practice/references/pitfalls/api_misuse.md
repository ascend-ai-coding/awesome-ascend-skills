---
title: API 误用
purpose: 定位 RegBase API 家族、签名、层级和 dtype 误用。
read_when:
  - 编译失败指向 API。
  - 审查怀疑 API 不存在或不适用于 VF。
keywords:
  - API
  - misuse
  - signature
next_reads:
  - ../api/regbase_api_whitelist.md
  - ../api/regbase_api_reference.md
  - regbase_vs_membase_confusions.md
depth: foundation
topic_type: pitfall
---

# API 误用

## 1. 常见误用

| 误用 | 风险 | 修复 |
|---|---|---|
| 未验证 API 名称 | 构建失败或幻觉 API | 查 SDK 文档、header 或参考实现 |
| MemBase API 写进 VF 层 | 层级错误，无法编译或语义错 | 回到 RegBase API 白名单 |
| 忽略模板参数 | dtype route 不匹配 | 明确 `RegTensor<T>` 和 API 模板参数 |
| mask 参数缺失 | tail 通道错误 | 为每个 VF API 明确 `MaskReg` |
| load/store dist 错 | UB/寄存器数据布局错 | 检查 `LoadDist` / `StoreDist` |

## 2. 验证步骤

1. 在白名单或 SDK 文档中确认 API 家族。
2. 查真实参考实现是否使用相同模式。
3. 确认参数顺序、模板参数、dtype 限制。
4. 确认 API 所在层：Host、UB pipeline 还是 VF/寄存器。
5. 在 DESIGN.md 的 API 映射表中记录依据。

## 3. 构建失败时不要做

- 不要把函数名改成“看起来类似”的名字。
- 不要删掉 mask 让编译先过。
- 不要把 RegBase 路径改成 LocalTensor 路径而不回到设计。
- 不要把参考实现中不同 dtype 的调用直接搬过来。

## 相关文档

- [[../api/regbase_api_whitelist]]
- [[../api/regbase_api_reference]]
- [[regbase_vs_membase_confusions]]
