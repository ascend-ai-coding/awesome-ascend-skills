---
name: external-cannbot-ops-ascendc-docs-gen
description: Ascend C 算子文档写作参考。提供需求分析、详细设计、迭代计划、aclnnAPI接口文档、算子README的标准模板。当用户需要生成算子文档、aclnnAPI文档、算子README、参考文档模板或了解算子文档规范时触发此技能。
original-name: ascendc-docs-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Ascend C 算子文档写作指南

## 概述

本技能提供算子开发文档的写作标准和模板参考，覆盖开发过程文档和用户文档。

## 文档类型

| 文档类型 | 模板路径 | 输出位置 |
|---------|---------|---------|
| 需求分析 | [requirement-analysis-template.md](references/requirement-analysis-template.md) | `operators/{operator_name}/docs/REQUIREMENTS.md` |
| 详细设计 | [detailed-design-template.md](references/detailed-design-template.md) | `operators/{operator_name}/docs/DESIGN.md` |
| 迭代计划 | [iteration-plan-template.md](references/iteration-plan-template.md) | `operators/{operator_name}/docs/PLAN.md` |
| aclnnAPI 接口文档 | [aclnn-api-doc-template.md](references/aclnn-api-doc-template.md) | `operators/{operator_name}/docs/aclnn{OperatorName}.md` |
| 算子 README | [operator-readme-template.md](references/operator-readme-template.md) | `operators/{operator_name}/README.md` |

## 使用方式

- 生成文档时参考对应模板的结构和必填项
- 根据实际算子特性调整内容

## 文档关系

```
需求分析文档 ──[确认]──▶ 详细设计文档 ──▶ 迭代执行计划
     │
     ├──▶ aclnnAPI 接口文档（数据来源：需求文档的算子规格、API定义、约束部分）
     │
     └──▶ 算子 README（数据来源：需求文档+设计文档+代码）
```

## 核心规范

### 文档命名

- 需求分析：`REQUIREMENTS.md`
- 详细设计：`DESIGN.md`
- 迭代计划：`PLAN.md`
- aclnnAPI 接口文档：`aclnn{OperatorName}.md`（如 `aclnnAdd.md`）
- 算子 README：`README.md`

### 版本管理

- 文档需包含修订记录
- 版本变更需记录修改人和修改内容

### 设计锁定

- 需求分析确认后锁定
- 详细设计确认后锁定
- 变更需经确认并更新文档
