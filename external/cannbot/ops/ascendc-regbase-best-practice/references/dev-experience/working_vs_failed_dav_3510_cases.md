---
title: DAV_3510 成功/失败案例对照
purpose: 用 DAV_3510 RegBase 的成功与失败信号帮助判断设计是否走在正确路径上。
read_when:
  - 修复中不确定问题来自设计、API、tiling 还是同步。
  - 需要判断当前实现是否符合 RegBase 主路径。
keywords:
  - DAV_3510
  - success
  - failure
next_reads:
  - regbase_programming_notes.md
  - ../pitfalls/symptom_to_cause.md
  - tiling_review_notes.md
depth: practice
topic_type: experience
---

# DAV_3510 成功/失败案例对照

## 成功信号

- Host route、tiling、kernel branch 清楚。
- `CopyIn -> Compute -> CopyOut` 分层稳定。
- VF body 明确使用 `RegTensor`、`MaskReg`、`LoadDist`、`StoreDist`。
- tail mask、dtype cast 和 store 长度一致。
- API 来自白名单、SDK 文档或真实参考实现。

## 失败信号

- 设计只说“使用 RegBase”，没有分层。
- `LocalTensor` 计算和 `RegTensor` 计算混在同一层。
- tail 只在 copy 层处理，VF store 没有 mask。
- 直接复制参考实现，shape/dtype 不匹配。
- 为了修复构建问题随意替换 API。
- 使用 `SyncAll` 或 cross-core flag 解决本地 stage ordering 问题。

## 修复优先级

1. 先确认 route 和 tiling。
2. 再确认数据流和 buffer ownership。
3. 再确认 VF API 与 mask。
4. 最后处理性能和高级同步。

## 相关文档

- [[regbase_programming_notes]]
- [[../pitfalls/symptom_to_cause]]
- [[tiling_review_notes]]
