# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 2652
- **代码文件**: random/resize_bilinear_v2_grad/op_graph/resize_bilinear_v2_grad_graph_infer.cpp
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

### 文件: random/resize_bilinear_v2_grad/op_graph/resize_bilinear_v2_grad_graph_infer.cpp（图推断侧（Host））

#### [1] 人工检视意见
- **提出人**: llimwang
- **文件**: random/resize_bilinear_v2_grad/op_graph/resize_bilinear_v2_grad_graph_infer.cpp
- **行号**: 19
- **评论时间**: 2026-05-11
- **Commit**: 98750b2381a0
- **关联条款**: LOG-2
- **问题描述**:

  > 其它 infer 都用 OP_LOGD(context->GetNodeName(), ...)（带 node 名 + Debug 级别）。这里改用 OP_LOGI 且没传 context，日志风格不一致且 INFO 级别在生产环境会刷屏。建议跟其它算子保持一致。
- **代码片段**（行19）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
