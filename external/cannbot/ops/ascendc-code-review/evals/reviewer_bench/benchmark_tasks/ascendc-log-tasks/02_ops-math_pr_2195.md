# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 2195
- **代码文件**: math/histogram_v2/op_host/histogram_v2_simt_tiling.cpp；math/non_finite_check/op_host/non_finite_check_tiling.cpp；math/pow/op_host/arch35/pow_tiling_arch35.cpp
- **代码侧别**: Host 侧（3 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 5 条（LOG-1 ×4 / LOG-2 ×1）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 4 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 1 |

---

## 发现问题

### 文件: math/histogram_v2/op_host/histogram_v2_simt_tiling.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: zl_hw
- **文件**: math/histogram_v2/op_host/histogram_v2_simt_tiling.cpp
- **行号**: 148
- **评论时间**: 2026-04-14
- **Commit**: 57a4c500cb21
- **关联条款**: LOG-1
- **问题描述**:

  > 使用OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON打印当前tensor的dtype，reason中详细说明
- **代码片段**（行148）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

### 文件: math/non_finite_check/op_host/non_finite_check_tiling.cpp（Tiling侧（Host））

#### [2] 人工检视意见
- **提出人**: zl_hw
- **文件**: math/non_finite_check/op_host/non_finite_check_tiling.cpp
- **行号**: 98
- **评论时间**: 2026-04-14
- **Commit**: 57a4c500cb21
- **关联条款**: LOG-1
- **问题描述**:

  > 改成OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON打印tensorlist的当前tensor，参数名参照评审的结果，采用 %s %ldth tensor
- **代码片段**（行98）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [3] 人工检视意见
- **提出人**: zl_hw
- **文件**: math/non_finite_check/op_host/non_finite_check_tiling.cpp
- **行号**: 101
- **评论时间**: 2026-04-17
- **Commit**: 57a4c500cb21
- **关联条款**: LOG-1
- **问题描述**:

  > 使用OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON，参照： std::string paramMsg = "x " + std::to_string(i) + "th tensor"; std::string reasonMsg = "DataType of all input x should be the same, but x " + std::to_string(i) + "th tensor's dtype is not same with x 0th tensor's dtype " + Ops::Base::ToString(dataType_);
- **代码片段**（行101）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [4] 人工检视意见
- **提出人**: zl_hw
- **文件**: math/non_finite_check/op_host/non_finite_check_tiling.cpp
- **行号**: 110
- **评论时间**: 2026-04-14
- **Commit**: 57a4c500cb21
- **关联条款**: LOG-1
- **问题描述**:

  > OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON使用OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON打印当前tensor的异常情况，reason要明确当前tensor的异常场景
- **代码片段**（行110）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

### 文件: math/pow/op_host/arch35/pow_tiling_arch35.cpp（Tiling侧（Host））

#### [5] 人工检视意见
- **提出人**: llimwang
- **文件**: math/pow/op_host/arch35/pow_tiling_arch35.cpp
- **行号**: 87
- **评论时间**: 2026-04-14
- **Commit**: 57a4c500cb21
- **关联条款**: LOG-2
- **问题描述**:

  > 第一个参数用了硬编码字符串 "Pow"，其他所有文件统一用的是 context_->GetNodeName()。建议改为 context_->GetNodeName() 保持一致，也避免算子名变更后日志不匹配。
- **代码片段**（行87）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
