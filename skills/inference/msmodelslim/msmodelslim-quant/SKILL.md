---
name: msmodelslim-quant
description: 华为昇腾 NPU 服务器模型quant工具技能，专为在昇腾 NPU 卡上进行大语言模型quant加速，聚焦于msmodelslim已验证模型的快速量化流程，支持量化流程和量化配置的轻微修改。支持 INT4/INT8/INT16 quant，提供完整的quant工作流，包括环境检查、动态方案query、容器deployment、quant执行等。
keywords:
  - msmodelslim
  - quant
  - model quantization
  - 华为昇腾
  - Ascend
  - NPU
  - 模型 compress
  - quant加速
  - INT4quant
  - INT8quant
  - 大模型 deployment
  - 昇腾NPU
  - 华为AI
  - Atlas
  - Davinci
  - 模型加速
  - 权重量化
  - ascend
  - 昇腾卡
  - npu-smi
  - 模型 deployment
  - LLMquant
---

# msmodelslim-quant

华为 msmodelslim quant工具技能 - 动态 query + 容器化deployment

## 技能描述

**本技能专为华为昇腾 NPU 服务器定制**，用于在昇腾 NPU 卡上进行大语言模型的quant加速。msmodelslim 是华为昇腾生态的开源模型 compress工具，支持在昇腾 NPU 上对大模型进行 INT4/INT8/INT16 quant，显著降低模型显存占用并提升推理性能。本技能提供完整的quant工作流，包括环境检查、动态方案query、容器deployment、quant执行等。

**适用场景**：
- 在昇腾 NPU 服务器上进行模型quant
- 将大模型quant后deployment到昇腾 NPU 进行推理
- 优化模型性能，降低显存占用

**支持的quant类型**：W4A8, W4A8C8, W8A16, W8A8, W8A8S, W8A8C8, W16A16S

## 关键词

```
msmodelslim, quant, model quantization, 华为昇腾, Ascend, NPU, 
模型compress, quant加速, INT4quant, INT8quant, 大模型deployment,
昇腾NPU, 华为AI, Atlas, Davinci, 模型加速, 权重量化,
ascend, 昇腾卡, npu-smi, 模型deployment, LLMquant
```

## 触发条件

当用户询问以下内容时触发：
- "quant XXX 模型"
- "XXX model quantization"
- "msmodelslim 支持 XXX"
- "XXX quant"
- 使用 msmodelslim 进行模型quant

## 核心原则

**不要硬编码模型支持列表** - 始终动态 query 实际项目来确定可用功能

## 远程服务器访问

> ⚠️ **重要提示**：如果用户的指令是在**远程服务器**上执行quant操作，则优先使用**非交互式**工具访问服务器，例如：
> - `sshpass -p 'password' ssh user@server` 
> - `sshpass -f passwordfile ssh user@server`
> - 配置 SSH Key 并使用 `ssh -i keyfile user@server`
>
> **避免直接使用 `ssh user@server` 命令**，因为这会导致交互式会话，可能造成：
> - 无法自动化执行quant脚本
> - 模型文件传输困难
> - 与远程服务器交互复杂化
>
> 建议在本地完成所有准备工作（模型下载、配置检查等），然后通过 `scp` 或 `rsync` 批量传输文件到服务器。

## 模型命名规范

quant后模型命名格式：`{original_name}-{quant_type}-{features}`

示例：
- `qwen2-7b-w8a8` (Qwen2-7B, W8A8量化)
- `qwen3-32b-w8a8s-QuaRot` (Qwen3-32B, W8A8稀疏量化, QuaRot特征)
- `deepseek-v3-w4a8c8` (DeepSeek-V3, W4A8C8量化)

## 昇腾 NPU 服务器环境

**本技能专为华为昇腾 NPU 服务器定制**。在开始quant之前，请确保了解以下信息：

### 查看昇腾 NPU 卡信息

@scripts/npu-commands.sh

### 昇腾 NPU 型号参考

@references/npu-models.md

### 华为昇腾生态组件

@references/ecosystem.md

### 昇腾 NPU quant前置检查

@scripts/env-check.sh

## 工作流程

### Step 1: 信息提取与路径确认

从用户描述中提取以下信息：
- **工作目录** (work_path): 项目文件和脚本的存放位置
- **原始模型路径** (weights_path): 原始模型权重文件的位置
- **模型名称** (model_name): 要量化的模型名称
- **quant格式** (quant_format): 如 W8A8, W4A8, W8A8S 等
- **特征** (features): 可选，如 QuaRot, AWQ 等

**必须与用户确认**：
1. 工作目录 (work_path)
2. 保存路径 (save_path) - quant后模型的输出位置

### Step 2: 检查/创建 msmodelslim 项目

@scripts/msmodelslim-setup.sh

### Step 3: 动态query量化解决方案

@scripts/quant-queries.sh

### Step 4: 容器化deployment（可选）

询问用户是否需要使用容器。如果使用：

@scripts/docker-commands.sh

### Step 5: 检查模型特定环境要求

**重要**：安装完 msmodelslim 后，必须检查该模型是否有特殊的环境要求。这些要求通常记录在 `example/{model_name}/README.md` 中。

**重要**：在使用pip安装依赖的过程中，如果出现pip安装超时的情况，可以询问用户是否更换pip源。可以选择预置pip源：`pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && pip config set install.trusted-host mirrors.aliyun.com`，也可以让客户输入特定的合适的源。以加速环境依赖的安装。

#### 5.1 检查模型特定的 README

@scripts/env-check.sh

#### 5.2 检查 README 中的环境要求

@references/quant-types.md

#### 5.3 常见的环境检查项

@scripts/env-check.sh

#### 5.4 应用环境更改

@scripts/env-check.sh

#### 5.5 更新预量化报告

@templates/quant-report-template.md

### Step 6: 预量化报告

@templates/quant-report-template.md

### ⚠️ Step 6.5: 用户交互确认（必须）

**在执行quant之前，必须与用户进行交互式确认**：

1. 向用户展示完整的预量化报告
2. 明确告知用户以下信息：
   - quant命令和所有参数
   - 预计耗时
   - 输出目录位置
   - 可能的风险
3. 询问用户：**"报告信息是否正确？量化方案是否可以执行？"**
4. **只有用户明确确认后**，才能进入 Step 7 执行quant
5. 如果用户有任何疑问或要求修改，必须停下来解答或调整方案

> ⚠️ **严禁跳过此确认步骤直接执行quant**，必须得到用户明确答复后才能继续。

### Step 7: 执行quant

用户确认后，执行quant命令：

@scripts/quant-execute.sh

### Step 8: 完成摘要

quant完成后，返回摘要：

@templates/quant-report-template.md

## quant类型参考

@references/quant-types.md

## 动态query命令参考

### query支持的模型组

### query支持的quant类型

### query模型适配器

@scripts/quant-queries.sh

## 官方资源

@references/official-resources.md

## 注意事项

1. **动态query优先**：不要假设某个模型一定支持或不支持，必须实际query项目
2. **路径确认**：必须与用户确认 work_path 和 save_path 后再执行
3. **容器化建议**：生产环境推荐使用容器，确保环境一致性
4. **版本兼容性**：不同版本的 msmodelslim 可能支持不同的模型和quant类型
5. **硬件要求**：部分quant可能需要特定的 NPU 或 GPU 硬件
6. **预量化报告**：执行前必须生成报告供用户确认
