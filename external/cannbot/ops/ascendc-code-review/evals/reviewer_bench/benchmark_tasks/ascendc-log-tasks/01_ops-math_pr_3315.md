# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 3315
- **代码文件**: conversion/slice_last_dim/op_host/arch35/slice_last_dim_tiling_arch35.cpp；conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp
- **代码侧别**: Host 侧（2 个文件）
- **检视来源**: 人工检视评论（GitCode PR）
- **所属档位**: ascendc-log-tasks（LOG-1/LOG-2 规则专项，筛选口径见 evals/reviewer_bench/README.md）
- **总评论数**: 5 条（LOG-1 ×5 / LOG-2 ×0）
- **检视时间**: 2026-09-17

## 检视统计

| 关联条款 | 条数 |
|---------|------|
| LOG-1 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 5 |
| LOG-2 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 0 |

---

## 发现问题

### 文件: conversion/slice_last_dim/op_host/arch35/slice_last_dim_tiling_arch35.cpp（Tiling侧（Host））

#### [1] 人工检视意见
- **提出人**: zhanw_coding
- **文件**: conversion/slice_last_dim/op_host/arch35/slice_last_dim_tiling_arch35.cpp
- **行号**: 86
- **评论时间**: 2026-06-16
- **Commit**: ffb05f712361
- **关联条款**: LOG-1
- **问题描述**:

  > 输入错误的场景，不要使用 OP_LOGE，这里可以使用 OP_LOGE_FOR_INVALID_SHAPEDIM
- **代码片段**（行86）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

### 文件: conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp（Tiling侧（Host））

#### [2] 人工检视意见
- **提出人**: zhanw_coding
- **文件**: conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp
- **行号**: 82
- **评论时间**: 2026-06-16
- **Commit**: ffb05f712361
- **关联条款**: LOG-1
- **问题描述**:

  > 使用 OP_LOGE_FOR_INVALID_DTYPE，参考 `conversion\batch_to_space_nd\op_host\arch35\batch_to_space_nd_tiling_arch35.cpp`
- **代码片段**（行82）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [3] 人工检视意见
- **提出人**: zhanw_coding
- **文件**: conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp
- **行号**: 89
- **评论时间**: 2026-06-16
- **Commit**: ffb05f712361
- **关联条款**: LOG-1
- **问题描述**:

  > 改用 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
- **代码片段**（行89）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [4] 人工检视意见
- **提出人**: zhanw_coding
- **文件**: conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp
- **行号**: 113
- **评论时间**: 2026-06-16
- **Commit**: ffb05f712361
- **关联条款**: LOG-1
- **问题描述**:

  > 改用 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
- **代码片段**（行113）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---

#### [5] 人工检视意见
- **提出人**: zhanw_coding
- **文件**: conversion/slice_with_axes/op_host/arch35/slice_with_axes_tiling_arch35.cpp
- **行号**: 125
- **评论时间**: 2026-06-16
- **Commit**: ffb05f712361
- **关联条款**: LOG-1
- **问题描述**:

  > 改用 OP_LOGE_FOR_INVALID_SHAPE
- **代码片段**（行125）: 待补（评测仅依赖文件/行号/问题描述，见 README「扩充 benchmark_tasks」说明）

---
