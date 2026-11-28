# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 4089
- **代码文件**: mc2/common/op_host/op_tiling/matmul_performance.cpp
- **代码侧别**: Host 侧（1 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 1 条（LOG-1 ×0 / LOG-2 ×1）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 0 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 1 |

---

## 发现问题

### 文件: mc2/common/op_host/op_tiling/matmul_performance.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: Allan_Yu
- **文件**: mc2/common/op_host/op_tiling/matmul_performance.cpp
- **行号**: 263
- **评论时间**: 2026-04-14
- **Commit**: 待查（raw 源无 commit 字段，可经 gen_proposer_sample 按 PR ref + 评论时间定位补齐）
- **关联条款**: LOG-2
- **问题描述**:

  > 这行 OP_LOGI 用了硬编码的 "alltoallmatmul-test" 作为算子名，看起来是调试阶段留下的日志。生产代码中不应该用 test 后缀的标识符做日志tag，要么删掉这行，要么改成使用实际的 opName 参数。
- **代码片段**（行263）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
