# MindSpeed-LLM 数据格式详解

## Alpaca 格式详解

### 标准字段

```json
{
    "instruction": "将以下句子翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is nice today."
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `instruction` | 是 | 指令/问题 |
| `input` | 否 | 附加上下文（为空时不拼入 prompt） |
| `output` | 是 | 期望输出 |

### 处理流程

1. 根据 `--prompt-type` 选择对话模板
2. 将 instruction + input 拼为用户消息
3. output 作为助手回复
4. 分词后打包为 packed 格式

## ShareGPT 格式详解

### 标准字段

```json
{
    "conversations": [
        {"from": "human", "value": "你好，请介绍一下自己"},
        {"from": "gpt", "value": "我是一个 AI 助手..."},
        {"from": "human", "value": "你能做什么？"},
        {"from": "gpt", "value": "我可以帮助你..."}
    ]
}
```

| 字段 | 说明 |
|------|------|
| `from` | 角色：`human`/`user` 或 `gpt`/`assistant` |
| `value` | 消息内容 |

### 多轮对话处理

- 每轮对话的 human 消息作为输入，gpt 消息作为标签
- 自动添加对应模型的特殊 token（如 Qwen 的 `<|im_start|>`/`<|im_end|>`）

## Pairwise 格式详解（DPO）

```json
{
    "system": "You are a helpful assistant.",
    "question": "解释什么是深度学习",
    "chosen": "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示...",
    "rejected": "深度学习就是让计算机学习。"
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `question` | 是 | 指令/问题（通过 `--map-keys` 映射） |
| `system` | 否 | 系统提示（通过 `--map-keys` 映射） |

> 完整 map-keys 示例：`--map-keys '{"prompt":"question", "query":"", "system":"system"}'`
| `chosen` | 是 | 偏好的回复 |
| `rejected` | 是 | 不偏好的回复 |

> Pairwise 数据输出为 8 个文件：`chosen_input_ids`、`chosen_labels`、`rejected_input_ids`、`rejected_labels`（各含 `.bin` + `.idx`）。

## 预训练数据格式

```json
{"text": "这是一段很长的预训练文本，可以包含多个段落..."}
```

每行一个 JSON 对象，仅需 `text` 字段。

## 输入文件格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| JSON Lines | `.jsonl` | 每行一个 JSON 对象 |
| JSON Array | `.json` | 整个文件是一个 JSON 数组 |
| Parquet | `.parquet` | Apache Parquet 列式格式 |
| CSV | `.csv` | 逗号分隔值 |
| Text | `.txt` | 纯文本 |
| Arrow | `.arrow` | Apache Arrow 格式 |

## 高级预处理选项

### 序列打包

指令数据默认使用序列打包（packing），将多个短序列拼接为一个长序列以提高训练效率：

```bash
--seq-length 4096    # 打包后的目标序列长度
```

### 多文件混合

```bash
# 按权重混合多个数据源
--data-path 0.7 ./data/zh_pretrain 0.3 ./data/en_pretrain
```

### EOD Token

预训练数据中，每个文档末尾自动添加 EOD (End of Document) token 以分隔文档。

### 自定义 Handler

如需支持自定义数据格式，可在 `mindspeed_llm/tasks/preprocess/` 下添加新的 Handler 类。

## 输出文件详解

### 指令微调输出（Packed 格式）

```
<prefix>_packed_input_ids_document.bin    # 输入 token IDs
<prefix>_packed_input_ids_document.idx    # 索引文件
<prefix>_packed_labels_document.bin       # 标签
<prefix>_packed_labels_document.idx
<prefix>_packed_attention_mask_document.bin  # 注意力掩码
<prefix>_packed_attention_mask_document.idx
```

### 预训练输出

```
<prefix>_text_document.bin    # Token IDs
<prefix>_text_document.idx   # 索引文件
```

## 官方参考

- [Alpaca 数据集文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/finetune/datasets/alpaca_dataset.md)
- [ShareGPT 数据集文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/finetune/datasets/sharegpt_dataset.md)
- [Pairwise 数据集文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/finetune/datasets/pairwise_dataset.md)
- [预训练数据文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/pretrain/pretrain_dataset.md)
