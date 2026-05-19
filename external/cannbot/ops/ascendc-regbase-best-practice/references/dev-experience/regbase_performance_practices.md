---
title: RegBase 性能实践
purpose: 给出 RegBase 性能审查和优化的优先级。
read_when:
  - RegBase 算子功能正确但性能不达标。
  - 需要评估融合链路是否真正减少 UB/GM 往返。
keywords:
  - performance
  - fusion
  - register
next_reads:
  - regbase_programming_notes.md
  - tiling_review_notes.md
  - working_vs_failed_dav_3510_cases.md
depth: practice
topic_type: experience
---

# RegBase 性能实践

RegBase 性能收益通常来自减少 UB/GM 往返、保持中间值在 register、减少不必要同步以及让 tail/mask 简洁。

## 1. 优先检查

- 融合链路是否真的减少中间 UB 写回。
- `RegTensor` 临时变量是否服务于真实数学链。
- UB buffer 是否过度分配。
- `CopyIn` / `CopyOut` 是否因为非对齐产生大量碎片 copy。
- 是否引入了不必要 `PipeBarrier`、`SyncAll` 或 cross-core sync。

## 2. 优化方向

| 问题 | 方向 |
|---|---|
| GM 往返多 | 合并 VF chain，减少中间输出 |
| UB 使用高 | 把短生命周期中间值留在 register |
| tail 复杂 | 简化 tile 形状或集中 mask 逻辑 |
| dtype cast 多 | 合并 cast 点，避免来回 cast |
| 同步过重 | 回到分层同步模型，删除无依据 barrier |

## 3. 不要为了性能牺牲可验证性

- 不要删掉 tail mask。
- 不要用未验证 API 替换已验证 API。
- 不要把 Host shape 假设写死进 kernel。
- 不要为了少一个 buffer 破坏数据 ownership。

## 相关文档

- [[regbase_programming_notes]]
- [[tiling_review_notes]]
- [[working_vs_failed_dav_3510_cases]]
