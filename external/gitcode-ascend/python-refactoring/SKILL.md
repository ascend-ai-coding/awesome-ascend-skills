---
name: external-gitcode-ascend-python-refactoring
description: Python 代码重构技能，覆盖代码坏味道识别、设计模式应用、可读性改进和实战经验。当用户要求"重构代码"、"refactor"、"代码优化"、"改善代码质量"、"code
  smell review"、"应用设计模式"、"提升可读性"，或提交代码审查请求时使用此技能。支持在重构完成后输出结构化重构文档（"输出重构文档"、"生成重构报告"）。包含基于
  vllm-ascend 仓库 20+ 个真实重构 PR 提炼的实战模式。
keywords:
- 重构
- refactor
- code smell
- 代码坏味道
- 设计模式
- design pattern
- 可读性
- 可维护性
- clean code
- Pythonic
- 代码审查
- code review
- 重构文档
- 重构报告
- refactoring report
original-name: python-refactoring
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Python 代码重构

对 Python 代码进行系统性重构，在不改变外部行为的前提下改善内部结构，提升可读性、可维护性和可扩展性。

## Overview

- **适用场景**：代码重构、代码审查、代码质量改善、技术债务清理
- **核心原则**：重构 ≠ 重写；保持接口兼容；有测试才重构；最小改动
- **参考文档**：基于 vllm-ascend 仓库 20+ 个真实重构 PR 提炼
- **约束配置**：所有数值阈值（函数行数、参数个数等）集中在 [`refactoring-config.json`](references/refactoring-config.json)，用户可自定义或禁用

## Quick Reference

| 任务 | 参考文档 |
|------|----------|
| 自定义约束阈值 | [refactoring-config.json](references/refactoring-config.json) |
| 识别代码问题 | [references/code-smells.md](references/code-smells.md) |
| 选择设计模式 | [references/design-patterns.md](references/design-patterns.md) |
| 改善可读性 | [references/readability.md](references/readability.md) |
| 实战模式与案例 | [references/refactoring-in-practice.md](references/refactoring-in-practice.md) |
| 生成重构文档 | [references/refactoring-report.md](references/refactoring-report.md) |

---

## 坏味道 → 重构模式 速查表

发现问题后，用此表快速定位解决方案：

| 坏味道 | 推荐模式 | 详细参考 |
|--------|----------|----------|
| 过长函数 | Extract Method、Guard Clause | [code-smells.md §1.1](references/code-smells.md) |
| 过长参数列表 | dataclass 参数对象、Builder | [design-patterns.md §1.3](references/design-patterns.md) |
| 重复代码（横切关注点） | 装饰器模式 | [design-patterns.md §2.1](references/design-patterns.md) |
| 重复代码（流程骨架相同） | 模板方法 / 策略模式 | [practice §模式三](references/refactoring-in-practice.md) |
| 上帝类 | 按职责拆分、组合替代继承 | [practice §模式二、五](references/refactoring-in-practice.md) |
| 过度继承 | 组合 + Protocol | [design-patterns.md §3.1](references/design-patterns.md) |
| if-elif 创建实例 | 工厂模式（注册表） | [practice §模式一](references/refactoring-in-practice.md) |
| if-elif 类型分发 | 多态 / singledispatch | [design-patterns.md §3.4](references/design-patterns.md) |
| 多个 `xxx_with_yyy` 函数变体 | 策略模式（Protocol） | [practice §模式三](references/refactoring-in-practice.md) |
| dict/tuple 传递复合数据 | dataclass 类型形式化 | [practice §模式七](references/refactoring-in-practice.md) |
| 魔法数字/字符串 | 枚举（Enum / IntEnum） | [code-smells.md §3.3](references/code-smells.md) |
| 自定义实现与上游重复 | 对齐上游 / 继承基类 | [practice §模式四](references/refactoring-in-practice.md) |
| 过大模块（>500 行） | 分离关注点、包化 | [practice §模式二](references/refactoring-in-practice.md) |
| 基准测试证明的劣势路径 | 删除死路径 | [practice §模式六](references/refactoring-in-practice.md) |

---

## 重构执行流程

```
┌─────────────────────────────────────────────────────────┐
│  0. 前置检查                                              │
│     - 确认有测试覆盖（无则先补关键路径测试）                  │
│     - 确认代码可正常运行                                    │
│     - 评估重构规模 → refactoring-in-practice.md 规模矩阵   │
├─────────────────────────────────────────────────────────┤
│  1. 识别问题  →  references/code-smells.md               │
│     - 通读代码，按分类逐项检查                               │
│     - 输出问题清单，按严重程度排序                            │
├─────────────────────────────────────────────────────────┤
│  2. 选择策略  →  references/refactoring-in-practice.md   │
│     - 匹配七大实战模式，选择最合适的重构路径                  │
│     - 大型重构制定分阶段计划（参考 Quantization/MoE 案例）    │
├─────────────────────────────────────────────────────────┤
│  3. 结构重构  →  references/design-patterns.md           │
│     - 对结构性问题选择合适的模式                             │
│     - 实施重构，确保接口兼容                                 │
├─────────────────────────────────────────────────────────┤
│  4. 打磨优化  →  references/readability.md               │
│     - 改善命名、结构、类型标注                               │
│     - 补充必要注释和文档                                    │
├─────────────────────────────────────────────────────────┤
│  5. 验证                                                 │
│     - 运行全部已有测试，确认无回归                            │
│     - 运行 linter / type checker                          │
│     - 对比重构前后，确认行为一致                             │
├─────────────────────────────────────────────────────────┤
│  6. 输出重构文档（可选，用户请求时生成）                       │
│     - 汇总本次重构的完整记录                                 │
│     - 生成结构化文档，可用于 PR 描述 / 团队分享 / 项目归档     │
│     - 参见 references/refactoring-report.md                  │
└─────────────────────────────────────────────────────────┘
```

---

## 重构决策指南

### 何时重构

- 添加新功能前，先重构相关代码使其易于扩展
- 修复 Bug 时，顺便改善周边代码结构
- Code Review 中发现的问题
- 代码难以理解或修改时

### 何时不重构

- 即将废弃的代码
- 没有测试覆盖且无法快速补充测试的核心代码
- 时间紧迫的紧急修复（先修复，后重构）
- 代码虽不完美但足够清晰且稳定运行

### 重构粒度控制

| 规模 | 范围 | 建议 | 参考案例 |
|------|------|------|----------|
| 小型 | 单个函数/方法 | 直接修改，单次提交 | MoE #5189 复用上游 all_reduce（-38 行） |
| 中型 | 单个类或模块 | 拆分为 2-3 次提交 | MoE #5481 dict/tuple → dataclass（6 文件） |
| 大型 | 跨模块/包 | 制定计划，分阶段执行 | Quantization 4 阶段重构（36 文件） |

---

## 重构文档模板

当用户请求生成重构文档（如 "输出重构文档"、"生成重构报告"、"写重构总结"）时，在重构完成并验证后，按 [references/refactoring-report.md](references/refactoring-report.md) 中的模板生成文档。

文档支持按用途裁剪（PR 描述 / 团队分享 / 项目归档），小型重构可使用精简格式。详见参考文档。

---

## 输出格式

对每个重构建议，按以下格式输出：

```
### [问题类型] 问题简述

**位置：** `file_path:line_number`
**问题：** 描述当前代码的问题
**方案：** 描述修复方案
**实战参考：** 匹配的实战模式（如适用）
**修复前：**
（代码片段）
**修复后：**
（代码片段）
**依据：** 引用的参考文档和具体规则
```

---

## References

详细参考文档：

- **[code-smells.md](references/code-smells.md)** — 代码坏味道识别与修复（函数/类/逻辑/模块四个层级，含 isinstance 链分发）
- **[design-patterns.md](references/design-patterns.md)** — Pythonic 设计模式应用（创建型/结构型/行为型 + singledispatch + 反模式警示）
- **[readability.md](references/readability.md)** — 可读性与可维护性改进（命名/结构/类型标注/注释/模块组织/Pythonic 惯用法）
- **[refactoring-in-practice.md](references/refactoring-in-practice.md)** — 基于 vllm-ascend 真实 PR 的重构实战经验（七大模式 + 安全守则 + 规模评估矩阵）
- **[refactoring-report.md](references/refactoring-report.md)** — 重构文档生成（模板 + 各节填写指南 + 场景适配 + 小型重构精简格式）
