---
name: external-cannbot-ops-ascendc-regbase-best-practice
description: 当需要为 DAV_3510 RegBase 算子确认 API 约束、实现结构、排查常见陷阱或选择真实参考算子时使用。
original-name: ascendc-regbase-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# Ascend C RegBase 开发

该能力解决 RegBase 路线下的四类问题：API 约束确认、实现结构参考、常见陷阱排查、真实参考算子选择。

`reference-ops` 子树是轻量参考资产。算子名称或基础类型匹配只表示“值得查看的候选参考”，不能证明目标任务可以直接复用、融合或复制该实现。是否复用必须重新做 RegBase 专项设计判断。

## 何时使用

满足以下任一条件时使用该能力：

- Architect 在方案决策中把 RegBase 作为候选路线。
- 用户、DESIGN.md 或代码明确出现 RegBase / RegTensor / asc_vf_call / __simd_vf__ 等信号。
- Developer 或 Reviewer 需要确认 RegBase 专属 API、实现结构、陷阱或参考实现。

不要把该能力当成默认算子开发路径的通用替代品。技术路线未决时，由 Architect 完成 SIMD/MemBase 与 RegBase 的方案决策。

## 阅读入口

- 端到端开发主线：`references/regbase_development_guide.md`
- API 约束和签名：`references/api/index.md`
- 陷阱和精度风险：`references/pitfalls/index.md`
- 工程经验和路线判断：`references/dev-experience/index.md`
- 真实参考实现：`references/reference-ops/open_source_operator_table.md`

## 按阶段选择

- 方案决策：`references/regbase_development_guide.md`、`references/dev-experience/index.md`
- 设计：`references/regbase_development_guide.md`、`references/api/index.md`
- 实现：`references/api/index.md`、`references/dev-experience/regbase_programming_notes.md`、`references/reference-ops/open_source_operator_table.md`
- 审查：`references/pitfalls/index.md`、`references/dev-experience/index.md`
- 修复和调试：`references/pitfalls/symptom_to_cause.md`、`references/dev-experience/index.md`

进入 `api/`、`pitfalls/` 或 `dev-experience/` 后，先读该目录的 `index.md`，再打开叶子文档。针对一个子问题，最多展开 5 篇叶子文档；仍不够时拆分问题，而不是一次加载整个子树。

## 约束

- 引用 API 前必须检查 API 白名单、API reference 或官方文档；不要凭函数名猜测。
- 设计伪代码不能直接当作可编译实现；写代码时必须回到真实工程模板和 API 签名。
- 编写新 RegBase 算子前，必须通过 `open_source_operator_table.md` 定位候选目录，并阅读至少一个经确认包含 RegBase 证据的真实实现。
- 看到 `RegTensor` / `MaskReg` / `asc_vf_call` / `__simd_vf__` 等关键词时，优先检查 API 层和 pitfalls 层。
- 架构判断必须显式说明。如果某条经验来自兼容路径而不是主路径 `DAV_3510 / RegBase`，需要说清楚。
