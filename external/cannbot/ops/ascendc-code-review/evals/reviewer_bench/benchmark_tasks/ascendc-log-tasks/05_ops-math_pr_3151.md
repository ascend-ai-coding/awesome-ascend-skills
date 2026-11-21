# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 3151
- **代码文件**: math/kl_div_v2/op_host/arch35/kl_div_v2_tiling_arch35.cpp
- **代码侧别**: Host 侧（1 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 1 条（LOG-1 ×1 / LOG-2 ×0）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 1 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 0 |

---

## 发现问题

### 文件: math/kl_div_v2/op_host/arch35/kl_div_v2_tiling_arch35.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: zhou-qilong
- **文件**: math/kl_div_v2/op_host/arch35/kl_div_v2_tiling_arch35.cpp
- **行号**: 171
- **评论时间**: 2026-06-09
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-1
- **问题描述**:

  > 这里校验的是用户传入的 reduction 属性值是否在合法范围内（none/mean/sum/batchmean），属于用户可感知的参数校验，应该迁移到 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON`。当前仍使用旧的 `OP_LOGE`，迁移不完整。
- **代码片段**（行171）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
