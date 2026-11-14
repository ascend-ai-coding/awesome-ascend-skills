# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 5768
- **代码文件**: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_a2a3.cpp；mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp
- **代码侧别**: Host 侧（2 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 3 条（LOG-1 ×3 / LOG-2 ×0）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 3 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 0 |

---

## 发现问题

### 文件: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_a2a3.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: wang-minbo
- **文件**: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_a2a3.cpp
- **行号**: 31
- **评论时间**: 2026-05-23
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-1
- **问题描述**:

  > 应该用OP_LOGE_WITH_INVALID_ATTR宏去整改
- **代码片段**（行31）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [2] 人工检视意见
- **提出人**: wang-minbo
- **文件**: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_a2a3.cpp
- **行号**: 34
- **评论时间**: 2026-05-23
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-1
- **问题描述**:

  > 应该用OP_LOGE_WITH_INVALID_ATTR宏去整改
- **代码片段**（行34）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

### 文件: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp（Tiling侧（Host））

#### [3] 人工检视意见
- **提出人**: wang-minbo
- **文件**: mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp
- **行号**: 159
- **评论时间**: 2026-05-23
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-1
- **问题描述**:

  > 应该用OP_LOGE_WITH_INVALID_ATTR宏去整改
- **代码片段**（行159）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
