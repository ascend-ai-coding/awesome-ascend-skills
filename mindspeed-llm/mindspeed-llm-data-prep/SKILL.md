---
name: mindspeed-llm-data-prep
description: MindSpeed-LLM 数据预处理指南，用于华为昇腾 NPU。覆盖预训练数据、指令微调数据（Alpaca/ShareGPT 格式）、偏好对齐数据（Pairwise）的预处理流程。包含分词、打包、输出文件结构说明和 prompt 模板配置。当用户需要为 MindSpeed-LLM 训练准备数据时使用。
keywords:
    - mindspeed
    - mindspeed-llm
    - data preprocessing
    - 数据预处理
    - alpaca
    - sharegpt
    - tokenizer
    - instruction data
    - 指令数据
    - pretrain data
    - 预训练数据
---

# MindSpeed-LLM 数据预处理

本 Skill 指导用户为 MindSpeed-LLM 训练准备数据，覆盖预训练、指令微调和偏好对齐三种数据格式。

## 快速开始

### 指令微调数据（Alpaca 格式）

```bash
python preprocess_data.py \
    --input ./dataset/alpaca.parquet \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --output-prefix ./finetune_dataset/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --prompt-type qwen
```

### 预训练数据

```bash
python preprocess_data.py \
    --input ./dataset/raw_text.parquet \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B/ \
    --output-prefix ./pretrain_dataset/data \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --json-keys text \
    --workers 4
```

## 数据处理器

| Handler | 输入格式 | 用途 |
|---------|---------|------|
| `GeneralPretrainHandler` | `{"text": "..."}` | 预训练 |
| `AlpacaStyleInstructionHandler` | `{"instruction": "...", "output": "..."}` | 单轮指令微调 |
| `SharegptStyleInstructionHandler` | `{"conversations": [...]}` | 多轮对话微调 |
| `AlpacaStylePairwiseHandler` | `{"question": "...", "chosen": "...", "rejected": "..."}` | DPO 偏好对齐（Alpaca 风格） |
| `SharegptStylePairwiseHandler` | `{"conversations": [...], "chosen": "...", "rejected": "..."}` | DPO 偏好对齐（ShareGPT 风格） |

## 输入数据格式

### Alpaca 格式

```json
{
    "instruction": "描述一下量子计算的基本原理",
    "input": "",
    "output": "量子计算利用量子比特..."
}
```

支持的字段：`instruction`（必需）、`input`（可选）、`output`（必需）。

### ShareGPT 格式

```json
{
    "conversations": [
        {"from": "human", "value": "你好"},
        {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"},
        {"from": "human", "value": "请介绍一下深度学习"},
        {"from": "gpt", "value": "深度学习是机器学习的一个分支..."}
    ]
}
```

角色标识：`human` / `gpt`（或 `user` / `assistant`）。

### Pairwise 格式（DPO）

```json
{
    "system": "You are a helpful assistant.",
    "question": "解释什么是机器学习",
    "chosen": "机器学习是人工智能的一个子领域...",
    "rejected": "机器学习就是让机器学习。"
}
```

### 预训练数据

```json
{"text": "这是一段用于预训练的长文本..."}
```

支持 `.jsonl`、`.json`、`.parquet`、`.csv`、`.txt`、`.arrow` 输入格式。

## 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input` | 输入数据文件路径 | `./dataset/alpaca.parquet` |
| `--output-prefix` | 输出文件前缀 | `./finetune_dataset/alpaca` |
| `--tokenizer-name-or-path` | 分词器路径（HF 模型目录） | `./model_from_hf/Qwen2.5-7B-Instruct/` |
| `--tokenizer-type` | 分词器类型 | `PretrainedFromHF` |
| `--handler-name` | 数据处理器 | `AlpacaStyleInstructionHandler` |
| `--prompt-type` | 对话模板类型 | `qwen`、`llama3`、`glm4` |
| `--json-keys` | 预训练文本字段名 | `text`（预训练数据必需） |
| `--map-keys` | 字段映射（非标准数据格式） | `'{"prompt":"question"}'` |
| `--workers` | 并行处理线程数 | `4` |
| `--seq-length` | 序列长度（打包时使用） | `4096` |
| `--log-interval` | 日志打印间隔 | `1000` |

### Prompt 模板

| 模板名 | 适用模型 |
|--------|---------|
| `qwen` | Qwen、Qwen2.5 |
| `qwen3` | Qwen3 |
| `llama3` | LLaMA 3/3.1 |
| `llama2` | LLaMA 2 |
| `glm4` | GLM-4 |
| `deepseek3` | DeepSeek-V3 |
| `deepseek2` | DeepSeek-V2 |
| `baichuan2` | Baichuan2 |
| `mistral` | Mistral |
| `chatml` | ChatML 通用格式 |

## 输出文件结构

### 指令微调数据输出

预处理生成 packed 格式文件：

```
<prefix>_packed_input_ids_document.bin / .idx
<prefix>_packed_labels_document.bin / .idx
<prefix>_packed_attention_mask_document.bin / .idx
```

### Pairwise 数据输出（DPO）

```
<prefix>_packed_chosen_input_ids_document.bin / .idx
<prefix>_packed_chosen_labels_document.bin / .idx
<prefix>_packed_rejected_input_ids_document.bin / .idx
<prefix>_packed_rejected_labels_document.bin / .idx
```

### 预训练数据输出

```
<prefix>_text_document.bin / .idx
```

> **注意**：预训练 `--data-path` 需包含 `_text_document` 后缀（如 `./dataset/data_text_document`），微调则使用前缀即可。

## 数据路径约定

> **重要**：训练时 `--data-path` 应设为输出前缀（如 `./finetune_dataset/alpaca`），**不要**包含 `_packed`。数据加载器 `get_packed_indexed_dataset()` 会自动拼接 `_packed_*_document` 后缀。

```bash
# 正确
--data-path ./finetune_dataset/alpaca

# 错误
--data-path ./finetune_dataset/alpaca_packed
```

## 多文件数据集

支持多个数据文件按权重混合：

```bash
--data-path 0.7 ./dataset/pretrain_zh 0.3 ./dataset/pretrain_en
```

## 常见问题

**Q: 预处理报错 tokenizer 找不到**

确保 `--tokenizer-name-or-path` 指向包含 `tokenizer.json` 或 `tokenizer.model` 的目录。

**Q: 处理速度慢**

增加 `--workers` 参数（建议不超过 CPU 核心数）。

**Q: ShareGPT 格式的角色标识不匹配**

确保使用 `human` / `gpt` 或 `user` / `assistant` 作为角色标识。

## 使用顺序

数据预处理完成后：

1. **权重转换** → 使用 [mindspeed-llm-weight-prep](../mindspeed-llm-weight-prep/SKILL.md)
2. **训练启动** → 使用 [mindspeed-llm-training](../mindspeed-llm-training/SKILL.md)

## 参考资源

- [详细数据格式说明](references/data-formats.md) - 各格式详细字段说明和高级用法
- [MindSpeed-LLM 仓库](https://gitcode.com/ascend/MindSpeed-LLM)
