# 代码检视报告

## 检视概览
- **仓库**: ops-nn
- **PR编号**: 4780
- **代码文件**: matmul/dual_level_quant_batch_matmul/op_host/dual_level_quant_batch_matmul_infershape.cpp；matmul/dual_level_quant_batch_matmul/op_host/op_tiling/dual_level_quant_batch_matmul_tiling_base.cpp
- **代码侧别**: Host 侧（2 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 2 条（LOG-1 ×0 / LOG-2 ×2）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 0 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 2 |

---

## 发现问题

### 文件: matmul/dual_level_quant_batch_matmul/op_host/op_tiling/dual_level_quant_batch_matmul_tiling_base.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: maqijun
- **文件**: matmul/dual_level_quant_batch_matmul/op_host/op_tiling/dual_level_quant_batch_matmul_tiling_base.cpp
- **行号**: 169
- **评论时间**: 2026-05-26
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-2
- **问题描述**:

  > OP_LOGE第一个入参是字符串，需要传入算子名
- **代码片段**（行169）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

### 文件: matmul/dual_level_quant_batch_matmul/op_host/dual_level_quant_batch_matmul_infershape.cpp（Host侧）

#### [2] 人工检视意见
- **提出人**: chenbinbin199309
- **文件**: matmul/dual_level_quant_batch_matmul/op_host/dual_level_quant_batch_matmul_infershape.cpp
- **行号**: 70
- **评论时间**: 2026-06-02
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-2
- **问题描述**:

  > 这里用常量NODE_NAME，不要硬编码算子名了
- **代码片段**（行70）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
