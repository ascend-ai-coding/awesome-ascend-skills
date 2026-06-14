---
name: external-gitcode-ascend-atb-golden-developer
description: 'ATB CSV 测试 Golden 参考实现开发指南。覆盖 DataGen 类的 customize/golden/case_preprocess
  三件套开发模式、hostData 注入机制（hosttensor binder）、NO_ERROR vs I:NO_ERROR 区别、 kernel 对齐原则、精度调试完整流程。当用户需要为算子编写
  CSV 正例 golden 参考实现时调用此技能。

  '
keywords:
- golden
- datagen
- customize
- case_preprocess
- hosttensor
- hostData
- binder
- NO_ERROR
- precision
- kernel
- 参考实现
- 正例
- csv
- test
- 算子测试
metadata:
  author: ascend-transformer-boost-team + Claude Code + Cursor + deepseek-v4-pro
  version: 1.3.2
  created: '2026-04-30'
  updated: '2026-05-12'
  skill-type: test
  allowed-tools: Bash(*) Read(*) Edit(*) Write(*)
original-name: atb-golden-developer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-06-14'
synced-commit: 62f55373f87e10e1d64f3f82f369be625c56b9fc
license: UNKNOWN
---

# ATB Golden Developer

## Quick Reference

**用途**：为 ATB 算子编写 CSV 正例所需的 golden 参考实现，覆盖 `customize/golden/case_preprocess` 三件套与精度对齐。

**渐进式披露阅读顺序**：

1. 完成「执行前 Checklist」与「前置条件」
2. 阅读 **[references/golden-developer-playbook.md](references/golden-developer-playbook.md)**（NO_ERROR 对比、Kernel 对齐、三件套、hostData、调试、精度、完整示例）

## 执行前 Checklist

- [ ] 已确认目标是 `NO_ERROR` 正例，而非仅 `I:NO_ERROR`
- [ ] 已确认可访问 ATB 测试框架与参考实现
- [ ] 已确认需要对齐 kernel 实际行为（非理论假设）
- [ ] 已确认本轮交付物（golden 逻辑、验证结果、问题记录）

## 功能概述

为 ATB 算子编写 CSV 正例所需的 golden 参考实现。正例的完整流程为 `Create → InferShape → Setup → Execute → Golden Compare`，缺一不可。

Golden 开发是 CSV 测试中最复杂的环节，涉及 3 个紧密配合的静态方法 + hostData 注入机制。

## 调用时机

- 需要为算子编写完整 CSV 正例（`ExpectedError: NO_ERROR`）
- `I:NO_ERROR` 不足够，需要验证执行结果的数值精度
- 测试报 `S:ERROR_INVALID_PARAM` + `hostData is null`
- 测试报 golden 维度不匹配、精度不达标

## 资源目录（按需读取）

| 资源 | 说明 |
|------|------|
| [`references/golden-developer-playbook.md`](references/golden-developer-playbook.md) | 正文详解：NO_ERROR、Kernel 对齐、三件套、hostData、调试、精度、示例代码 |
| [`references/case-study-mla-paged-attention-kernel-alignment.md`](references/case-study-mla-paged-attention-kernel-alignment.md) | MLA/PagedAttention kernel 对齐案例（CSV / Kernel） |
| [`../atb-atk-testcase-generator/references/pagedattention-case-study.md`](../atb-atk-testcase-generator/references/pagedattention-case-study.md) | **ATK** PagedAttention 精度与输入规范化复盘（CPU Golden vs ATB、`execute_*.py`） |
| [`../atb-atk-testcase-generator/references/common-faq.md`](../atb-atk-testcase-generator/references/common-faq.md) | **ATK** 常见问题（ranges/hash、contextLen、Golden 解耦等） |
| [`../atb-atk-testcase-generator/checks/gate-3-golden-implementation.md`](../atb-atk-testcase-generator/checks/gate-3-golden-implementation.md) | **Gate 3**：Golden 与 data_generation、executor 规范化 |

## 前置条件

- ATB 测试框架已编译
- 已理解算子的输入/输出 tensor 布局
- 已参考仓库内同类型算子的现有 golden 实现

---

## 相关文档

- [ATB CSV Tester](../atb-csv-tester/SKILL.md) — CSV 测试运行
- [ATB ATK Testcase Generator](../atb-atk-testcase-generator/SKILL.md) — ATK 格式 Golden（**独立 `execute_*.py`**）；详见该技能 **`references/pagedattention-case-study.md`**、**`common-faq.md`** 与 **Gate 3**（对齐 ref、可内联解耦、executor 输入规范化）。与本技能 CSV **DataGen 三件套**路径不同，按需选读
- [ATB Debug Guide](../atb-debug-guide/SKILL.md) — 调试指南
- [ATB Skills 索引](../../SKILL.md)
