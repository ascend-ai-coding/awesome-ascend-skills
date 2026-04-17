---
name: mindspeed-llm-weight-prep
description: MindSpeed-LLM 权重转换指南，用于华为昇腾 NPU。覆盖 HuggingFace 到 Megatron 格式转换、Megatron 到 HuggingFace 反向转换、LoRA 权重合并导出、TP/PP/EP 并行切分配置。支持 v1 和 v2 转换器，适用于 Qwen、LLaMA、DeepSeek、Mixtral 等模型。当用户需要转换模型权重格式时使用。
keywords:
    - mindspeed
    - mindspeed-llm
    - weight conversion
    - 权重转换
    - checkpoint
    - hf2megatron
    - megatron2hf
    - lora merge
    - LoRA 合并
    - tensor parallel
    - pipeline parallel
---

# MindSpeed-LLM 权重转换

本 Skill 指导用户在 HuggingFace 和 Megatron 格式之间转换模型权重，包括 LoRA 权重合并和并行切分配置。

## 快速开始

### HuggingFace → Megatron

```bash
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --add-qkv-bias \
    --load-dir ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --save-dir ./model_weights/qwen25_7b_mcore/ \
    --tokenizer-model ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

### Megatron → HuggingFace

```bash
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/qwen25_7b_mcore/ \
    --save-dir ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --tokenizer-model ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

> MG→HF 时使用 v1 需设 TP=PP=1；v2 无需设置并行参数（自动处理）。v1 的 `--save-dir` 必须指向原始 HF 模型路径（含 config.json 和 tokenizer）。输出保存在 `--save-dir` 下的 `mg2hf/` 子目录。

## 转换方向

| 方向 | 脚本 | 用途 |
|------|------|------|
| HF → Megatron | `convert_ckpt.py` / `convert_ckpt_v2.py` | 训练前：将 HF 权重转为 Megatron 格式 |
| Megatron → HF | `convert_ckpt.py` / `convert_ckpt_v2.py` | 训练后：将 Megatron 权重转回 HF 格式 |
| LoRA 合并导出 | `convert_ckpt.py`（仅 v1） | LoRA + 基础权重合并后转换 |

## 关键参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `--load-model-type` | 源格式：`hf` 或 `mg` | 是 |
| `--save-model-type` | 目标格式：`hf` 或 `mg` | 是 |
| `--target-tensor-parallel-size` | 张量并行度（TP） | 是 |
| `--target-pipeline-parallel-size` | 流水线并行度（PP） | 是 |
| `--target-expert-parallel-size` | 专家并行度（EP，MoE 模型） | 否（默认 1） |
| `--use-mcore-models` | 启用 Megatron-Mcore 格式 | 是（仅 v1） |
| `--load-dir` | 源模型路径 | 是 |
| `--save-dir` | 输出路径 | 是 |
| `--model-type-hf` | HF 模型类型 | 否（v1 默认 `llama2`，v2 默认 `qwen3`） |
| `--tokenizer-model` | 分词器文件路径 | 是 |
| `--params-dtype` | 权重精度：`fp16`、`bf16` | 是（仅 v1） |
| `--add-qkv-bias` | 添加 QKV 偏置（部分模型需要） | 视模型而定 |

## TP/PP 配置规则

| 规则 | 说明 |
|------|------|
| TP × PP ≤ NPU 数量 | 并行度不能超过可用 NPU |
| 层数可被 PP 整除 | 否则需使用 `--num-layer-list` 动态分配 |
| MG→HF 时 v1 需 TP=PP=1 | v2 无需设置（自动处理） |
| 训练时 TP/PP 必须匹配 | 转换时的 TP/PP 必须与训练脚本一致 |

### 动态 Pipeline 并行

层数不能被 PP 整除时：

```bash
# 14 层模型，PP=4：按 3,4,4,3 分配
--target-pipeline-parallel-size 4 \
--num-layer-list 3,4,4,3
```

## LoRA 权重处理

### LoRA 合并后转换（v1 专属）

```bash
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/base_mcore/ \
    --save-dir ./model_from_hf/merged_output/ \
    --lora-load ./lora_output/ \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --tokenizer-model ./model_from_hf/tokenizer.json
```

### LoRA 参数匹配

LoRA 转换时的 `--lora-r`、`--lora-alpha`、`--lora-target-modules` **必须与训练时一致**。

| 参数 | 说明 |
|------|------|
| `--lora-r` | LoRA 秩（如 8、16） |
| `--lora-alpha` | LoRA 缩放因子（如 16、32） |
| `--lora-target-modules` | 目标模块：`linear_qkv linear_proj linear_fc1 linear_fc2` |

## v1 vs v2 对比

| 特性 | v1 (`convert_ckpt.py`) | v2 (`convert_ckpt_v2.py`) |
|------|----------------------|--------------------------|
| LoRA 转换 | 支持 | **不支持** |
| 大模型内存优化 | 一般 | 流式加载，TB 级参数不崩溃 |
| MoE 高级功能 | 基础 | `--moe-grouped-gemm`、`--moe-tp-extend-ep` |
| 默认 HF 类型 | `llama2` | `qwen3` |
| MTP 层 | 不支持 | `--mtp-num-layers` |
| 缩减层调试 | 不支持 | `--num-layers` + `--first-k-dense-replace` |

> **注意**：v1 和 v2 是独立系统，不要混用（如 v2 做 HF→MG，然后 v1 做 MG→HF）。

## 示例脚本

脚本位于 `examples/mcore/<model_name>/`：

| 脚本模式 | 用途 |
|----------|------|
| `ckpt_convert_<model>_hf2mcore.sh` | HF → Megatron |
| `ckpt_convert_<model>_mcore2hf.sh` | Megatron → HF |

```bash
# 示例：Qwen2.5-7B HF→Megatron
bash examples/mcore/qwen25/ckpt_convert_qwen25_7b_hf2mcore.sh

# 示例：LLaMA-3-8B HF→Megatron
bash examples/mcore/llama3/ckpt_convert_llama3_8b_hf2mcore.sh
```

## 常见问题

**Q: 转换后训练报错权重加载失败**

确认转换时的 TP/PP 与训练脚本中的设置完全一致。

**Q: 大模型（100B+）转换 OOM**

使用 v2 转换器：`python convert_ckpt_v2.py`，支持流式加载。

**Q: MG→HF 输出缺少 config.json 和 tokenizer**

MG→HF 只输出权重文件（保存在 `mg2hf/` 子目录）。config.json、tokenizer 等需从原始 HF 模型目录复制。

**Q: LoRA 转换后精度异常**

Qwen 模型可能存在 dtype 不匹配（bf16 模型合并后生成 fp16 权重）。检查 config.json 中的 `torch_dtype` 字段。

## 使用顺序

权重转换完成后：

1. **训练启动** → 使用 [mindspeed-llm-training](../mindspeed-llm-training/SKILL.md)

## 参考资源

- [详细转换指南](references/conversion-guide.md) - v1/v2 完整参数、MoE 转换、高级用法
- [MindSpeed-LLM 仓库](https://gitcode.com/ascend/MindSpeed-LLM)
