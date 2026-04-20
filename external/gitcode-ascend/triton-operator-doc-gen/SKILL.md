---
name: external-gitcode-ascend-triton-operator-doc-gen
description: 生成昇腾NPU的Triton算子接口文档。当用户需要为昇腾NPU的Triton算子创建或更新接口文档时使用。核心能力：(1)根据模板生成标准化文档
  (2)支持昇腾NPU产品型号列表 (3)提供算子参数说明规范 (4)生成调用示例框架。
original-name: triton-npu-operator-doc-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Triton NPU 算子接口文档生成器

## 功能说明

该Skill用于为昇腾NPU平台的Triton算子生成标准化的接口文档。生成的文档遵循昇腾官方文档格式，包含产品支持情况、功能说明、函数原型、参数说明、约束条件和调用示例等内容。

## 使用方法

### 基础用法

1. 提供算子名称和基本信息
2. 输入算子的功能描述和计算公式
3. 提供函数原型和参数说明
4. 定义约束条件和支持的数据类型
5. 提供调用示例代码

### 文档结构模板

生成的文档将包含以下章节：

```markdown
# {算子名称}

[📄 查看源码]({源码链接})

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     {是否支持}    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     {是否支持}    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     {是否支持}    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     {是否支持}    |
|  <term>Atlas 推理系列产品</term>    |     {是否支持}    |
|  <term>Atlas 训练系列产品</term>    |     {是否支持}    |

## 功能说明

- 接口功能：{算子功能详细描述}
- 计算公式：

  $$
  {LaTeX格式的计算公式}
  $$

## 函数原型

  {函数原型}

## 参数说明

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>{参数名}</td>
      <td>{输入/输出}</td>
      <td>{参数描述}</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 各平台支持数据类型说明：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
    | `{参数1}`数据类型 | `{参数2}`数据类型 | `{参数3}`数据类型 | `{参数4}`数据类型 |
    | -------- | -------- | -------- | -------- |
    | {数据类型1} | {数据类型2} | {数据类型3} | {数据类型4} |

## 调用示例

```python
{Python调用示例代码}
```
```