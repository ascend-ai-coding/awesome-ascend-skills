---
name: external-gitcode-ascend-triton-operator-dev
description: 昇腾 Triton 算子全流程开发编排。当用户需要从零开发 Triton 算子、进行端到端开发流程、或不确定该用哪个子 skill 时使用。自动编排：环境配置→需求设计→代码生成→静态检视→精度验证→性能评估→性能优化。关键词：全流程、开发编排、端到端、workflow
  orchestration。
original-name: triton-operator-dev
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Triton 算子全流程开发

## 工作流概览

构建 Triton 算子分 7 个阶段（含 1 个条件阶段）：

| # | 阶段 | 产出 | Skill | 是否可跳过 |
|---|------|------|-------|-----------|
| 1 | 环境配置 | 环境验证报告 | `triton-operator-env-config` | 是：torch/torch_npu/triton 已可用 |
| 2 | 需求设计 | 设计文档 | `triton-operator-design` | 是：用户已提供完整设计文档 |
| 3 | 代码生成 | kernel + smoke test | `triton-operator-code-gen` | **否** |
| 4 | 静态检视 | 检视报告 | `triton-operator-code-review` | **否** |
| 5 | 精度验证 | 精度报告 | `triton-operator-precision-eval` | **否** |
| 6 | 性能评估 | 性能报告 + ratio | `triton-operator-performance-eval` | **否** |
| 7 | 性能优化 | 优化后代码 | `triton-operator-performance-optim` | 条件：ratio ≥ 目标则跳过 |

## ⚠️ 核心约束

1. **必须走完全流程**：阶段 3-6 不可跳过，不可在代码生成后停止
2. **用 TaskCreate 跟踪进度**：每阶段一个 Task，进入时 `in_progress`，完成后 `completed`
3. **精度通过前不做性能优化**（precision-eval 的核心原则）
4. **输出最终报告**：精度结果 + 性能 ratio + 优化历史 + 结论

## 场景路由

| 用户意图 | 正确做法 |
|---------|---------|
| "开发一个算子"、"从零开发" | 本 skill 全流程 |
| "帮我写算子代码" | **不用本 skill**，直接 `triton-operator-code-gen` |
| "优化算子性能" | **不用本 skill**，直接 `triton-operator-performance-optim` |
| "检查算子精度" | **不用本 skill**，直接 `triton-operator-precision-eval` |

## 阶段执行要点

### 阶段 1-2：环境 + 设计（可跳过）

**跳过判断**：
- 阶段 1 跳过条件：能 `import torch, torch_npu, triton` 且 `torch.npu.is_available()`
- 阶段 2 跳过条件：用户已提供设计文档（如 `docs/context/*.md`）或直接给出算子 API 规格

### 阶段 3：代码生成

调用 `Skill(triton-operator-code-gen)`。此阶段**只生成代码和 smoke test，不运行**。

**进入下一阶段前**：kernel 代码 + 测试文件已写入磁盘

### 阶段 4：静态检视

调用 `Skill(triton-operator-code-review)`。此阶段**只静态分析，不运行**。

**进入下一阶段前**：P0/P1 问题已修复

### 阶段 5：精度验证（MANDATORY 运行）

调用 `Skill(triton-operator-precision-eval)`。

**此阶段必须在 NPU 上运行测试。** 关键产出：
- 多 shape × dtype 的误差指标
- 通过/失败判定

**进入下一阶段前**：所有精度测试通过

### 阶段 6：性能评估（MANDATORY 运行）

调用 `Skill(triton-operator-performance-eval)`。

**此阶段必须在 NPU 上运行 benchmark。** 关键产出：
- Triton vs torch_npu 的 ratio
- 瓶颈类型诊断

### 阶段 7：性能优化（条件执行）

**触发条件**：阶段 6 的 ratio < 用户要求的性能目标。

调用 `Skill(triton-operator-performance-optim)`。优化后**必须重新运行精度验证**确认无回归。

## 常见陷阱

| 陷阱 | 症状 | 正确做法 |
|------|------|---------|
| 代码生成后停止 | 用户以为开发完成但无验证 | 强制执行阶段 5-6 |
| 精度未通过就优化 | 优化了错误的代码 | 精度通过是优化的前提 |
| 跳过 TaskCreate | 阶段遗漏无法追溯 | 每阶段创建 Task |
| 混淆"生成测试"和"运行测试" | 只有测试文件但从未执行 | 阶段 5-6 必须实际运行 |

## 反模式清单（NEVER）

- ❌ 代码生成后停止，不执行精度验证和性能评估
- ❌ 精度未通过就做性能优化
- ❌ 不创建任务跟踪就开始执行
- ❌ 跳过阶段但不更新任务状态
- ❌ 只生成代码就说"开发完成"
- ❌ 用"只需代码"的借口跳过验证流程

## 检查清单（最终输出前确认）

- [ ] 代码已生成并通过静态检视（P0/P1 清零）
- [ ] **精度测试已在 NPU 上运行**，所有 dtype/shape 通过
- [ ] **性能测试已在 NPU 上运行**，有 ratio 数据
- [ ] ratio < 目标时已尝试优化并重新验证精度
- [ ] **性能报告已生成并写入磁盘**（包含 ratio 数据、瓶颈诊断、优化历史、结论）
- [ ] **精度报告已生成并写入磁盘**（包含多 shape × dtype 误差指标、通过/失败判定）
- [ ] 向用户输出完整最终报告（精度 + 性能 + 结论）

## 最终交付物

全流程完成后，**必须**在算子目录下输出以下文件：

| 文件 | 内容 | 必须 |
|------|------|------|
| `{算子名}.py` | Kernel 代码 + Host 接口 | 是 |
| `test_{算子名}.py` | Smoke test | 是 |
| `precision_eval.py` | 精度评估脚本 | 是 |
| `precision_report.md` | 精度报告 | 是 |
| `performance_eval.py` | 性能评估脚本 | 是 |
| `performance_report.md` | 性能报告 | 是 |