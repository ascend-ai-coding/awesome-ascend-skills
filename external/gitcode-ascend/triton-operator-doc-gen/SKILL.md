---
name: external-gitcode-ascend-triton-operator-doc-gen
description: 为昇腾 NPU Triton 算子生成标准化接口文档。当用户需要为算子创建 README、生成 API 文档、编写产品支持表、整理参数说明时使用。关键词：文档生成、doc
  generation、README、接口文档、API documentation。
original-name: triton-operator-doc-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Triton 算子接口文档生成

## 工作流

1. 收集算子信息：名称、功能、公式、接口、参数、约束
2. **MANDATORY**：完整阅读 [`triton_operator_template.md`](references/triton_operator_template.md) 获取输出格式
3. 按模板生成文档，确保包含：
   - 产品支持情况表（Ascend 全系列产品）
   - 功能说明 + LaTeX 公式
   - 函数原型
   - 参数说明表
   - 约束条件（各平台数据类型支持）
   - 调用示例
