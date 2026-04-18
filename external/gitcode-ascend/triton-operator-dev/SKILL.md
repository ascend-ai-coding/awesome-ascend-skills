---
name: external-gitcode-ascend-triton-operator-dev
description: 昇腾Triton 算子全流程开发任务编排。当用户需要开发 Triton 算子时使用，覆盖环境配置→需求设计→代码生成→静态检视→精度验证→性能评估→文档生成→性能优化完整流程。
original-name: triton-operator-dev
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Triton 算子全流程开发

## 任务编排

| 阶段 | Skill | 产出 |
|------|-------|------|
| 1 | [triton-operator-env-config](triton-operator-env-config/SKILL.md) | 可用的开发环境 |
| 2 | [triton-operator-design](triton-operator-design/SKILL.md) | 算子需求文档 |
| 3 | [triton-operator-code-gen](triton-operator-code-gen/SKILL.md) | 可执行代码 |
| 4 | [triton-operator-code-review](triton-operator-code-review/SKILL.md) | 代码检视报告 |
| 5 | [triton-operator-precision-eval](triton-operator-precision-eval/SKILL.md) | 精度验证报告 |
| 6 | [triton-operator-performance-eval](triton-operator-performance-eval/SKILL.md) | 性能评估报告 |
| 7 | [triton-operator-doc-gen](triton-operator-doc-gen/SKILL.md) | 接口文档 |
| 8 | [triton-operator-performance-optim](triton-operator-performance-optim/SKILL.md) | 优化后代码 |

## 子 Skill 概览

### 1. triton-operator-env-config
- **触发**: 首次开发或环境异常
- **核心**: 依次检查 CANN → Python → torch → triton-ascend
- **验证**: 运行 `01-vector-add.py`

### 2. triton-operator-design
- **触发**: 需要设计新算子
- **核心**: 需求分析 → 原型设计 → 规格约束 → 特性实现
- **关键**: 必须包含 Tiling 策略具体计算方法

### 3. triton-operator-code-gen
- **触发**: 已有需求文档，需要生成代码
- **流程**: 确认计算逻辑 → 设计 Tiling → 生成 Kernel → 生成测试
- **依赖**: 必须先阅读 `references/hardware-architecture.md` 和 `references/templates.md`

### 4. triton-operator-code-review
- **触发**: 代码生成完成后，进入精度验证前
- **核心**: Host 侧检视 → Device 侧检视 → 性能隐患检视
- **关键**: 静态分析 Ascend API 约束合规性、Mask 完整性、精度处理

### 5. triton-operator-precision-eval
- **触发**: 代码检视通过后，进入性能评估前
- **核心**: 与 PyTorch 参考实现对比 → 计算误差指标 → 生成精度报告
- **关键**: 归约操作必须使用 FP32，确保 rtol/atol 满足阈值

### 6. triton-operator-performance-eval
- **触发**: 精度验证通过后，进入性能优化前
- **核心**: msprof 性能采集 → 瓶颈诊断 → 硬件利用率分析 → 性能报告
- **关键**: 必须使用 msprof/msprof op，不接受其他计时方式

### 7. triton-operator-doc-gen
- **触发**: 需要生成接口文档
- **产出**: 标准化的昇腾 NPU 接口文档（产品支持表、参数说明、调用示例）

### 8. triton-operator-performance-optim
- **触发**: 性能不达标
- **流程**: 性能诊断 → 基础调优 → 硬件特化 → 高级优化
- **关键**: 必须先用 msprof 定位瓶颈，优化后重新验证精度

## 快速决策

| 场景 | 跳过阶段 |
|------|----------|
| 环境已配置 | 1 |
| 已有设计文档 | 2 |
| 只需文档 | 1,2,3,4,5,6,8 |
| 只需代码 | 1,2,4,5,6,7,8 |
| 只需优化 | 1,2,3,4,5,6,7 |
| 跳过检视和验证 | 4,5,6 |

## 通用反模式

- ❌ 忽略 UB 大小（192KB）
- ❌ 归约操作不使用 FP32
- ❌ BLOCK 非 16 倍数（Cube 单元）
- ❌ 忘记 Mask（Ascend 零容错）
- ❌ 混淆 Vector Core 和 Cube Core 用途
