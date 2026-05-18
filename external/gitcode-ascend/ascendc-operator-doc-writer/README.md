# AscendC Operator Doc Writer 使用说明

## 1. 技能简介

`ascendc-operator-doc-writer` 用于基于现有模板文档和 AscendC 源码，生成结构化的算子技术文档（README 风格）。

该技能强调：

- 以大模型理解源码为主
- 以源码事实为依据
- 不依赖 Python 解析脚本自动抽取语义

## 2. 适用场景

当你有以下需求时使用本技能：

- 为 AscendC 算子补充或新建 README 文档
- 将一个仓库中的文档模板迁移到另一种目录结构
- 基于 `op_host` + `op_kernel` 的实现整理输入输出、tiling、kernel 流程
- 批量生成多个算子的文档（可配合外部 API）

## 3. 需要准备的输入

至少提供以下材料：

1. 一个参考模板 README（用于对齐结构和风格）
2. 目标算子相关源码文件，通常包括：
   - `op_host/<op_name>.cpp`
   - `op_host/<op_name>_tiling.h`
   - `op_kernel/<op_name>.cpp`
3. 可选：构建入口文件（如 `CMakeLists.txt`）

## 4. 快速使用

可直接在对话中发出类似请求：

```text
请使用模板 README 和以下源码，为 ascendc_dist_int8_flat_cos 生成文档：
- 模板：<template_readme_path>
- host：<op_host_cpp_path>
- tiling：<op_host_tiling_h_path>
- kernel：<op_kernel_cpp_path>
输出到：<target_readme_path>
```

如果你希望通过外部 API 生成，可使用：

- `references/model-prompt-template.md` 中的提示词骨架
- 将本地提取的关键源码片段填入后调用 API

## 5. 推荐工作流程

1. 读取模板 README，确定章节结构
2. 审查源码并提取事实信息
3. 按模板结构撰写目标 README
4. 区分“确认事实”和“推断信息”
5. 回查源码，完成一致性校验

对应参考文件：

- 文档章节建议：`references/readme-outline.md`
- 源码审查清单：`references/source-review-checklist.md`
- 提示词模板：`references/model-prompt-template.md`

## 6. 输出要求

生成文档时建议满足：

- 章节清晰，适合工程师快速阅读
- 输入输出、运行参数、tiling、kernel 流程有明确说明
- 不虚构调用示例、测试命令或运行结论
- 对无法从源码直接确认的内容明确标注为推断或未知

## 7. 常见问题

### Q1：目标仓库目录和模板仓库不一致怎么办？

按算子名组织文档单元，并增加 `File Layout Mapping` 章节解释目录映射关系。

### Q2：是否可以完全自动从代码提取语义？

本技能不推荐纯脚本语义抽取。脚本可用于文件定位和片段检索，语义归纳由模型完成。

### Q3：可以批量处理多个算子吗？

可以。建议按“一个算子一份输入材料、一份输出 README”的方式循环执行，保证质量与可追溯性。
