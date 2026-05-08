---
name: external-gitcode-ascend-triton-operator-doc-gen
description: 为昇腾 NPU Triton 算子生成标准化接口文档。当用户需要为算子创建 README、生成 API 文档、编写产品支持表、整理参数说明时使用。关键词：文档生成、doc
  generation、README、接口文档、API documentation。
original-name: triton-operator-doc-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Triton 算子接口文档生成

## 工作流

1. 收集算子信息：名称、功能、公式、接口、参数、约束
2. **MANDATORY**：完整阅读 [`triton_operator_template.md`](references/triton_operator_template.md) 获取输出格式
3. 参考示例：完整阅读 [`layer-norm.md`](examples/layer-norm.md) 理解完整文档结构
4. 需要确认 API 签名时，完整阅读 [`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md)
5. 按模板生成文档，确保包含：
   - 产品支持情况表（Ascend 全系列产品）
   - 功能说明 + LaTeX 公式
   - 函数原型
   - 参数说明表
   - 约束条件（各平台数据类型支持）
   - 调用示例

## 常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| 产品支持表不全 | 文档缺平台行 | 覆盖所有 Ascend 产品线（A3/A2/950/Kirin） |
| 约束表缺少 dtype 组合 | 用户按文档用不支持的类型 | 从 code 的 BLOCK_SIZE/dtype 限制反推，列出所有合法组合 |
| 公式与实现不一致 | 文档说的和代码算的不一样 | 从 kernel 代码逐行验证公式 |
| 调用示例不可运行 | 用户复制粘贴报错 | 示例必须完整可执行，包含 import 和数据构造 |

## 反模式清单（NEVER）

- ❌ 生成空模板占位符（如 `{公式}`、`{参数描述}` 未填写）——所有占位符必须替换为实际内容
- ❌ 调用示例缺少 import 语句或数据构造
- ❌ 约束表缺少 dtype × 平台组合
- ❌ 函数原型与实际 kernel 签名不一致
- ❌ 公式使用纯文本而非 LaTeX 语法
- ❌ 产品支持表包含不支持的型号标记为 √
