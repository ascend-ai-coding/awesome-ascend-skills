# MindSpeed-LLM 权重转换详细指南

## v1 完整参数（convert_ckpt.py）

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-type` | 模型类型 | `GPT` |
| `--load-model-type` | 源格式 | `hf` |
| `--save-model-type` | 目标格式 | `mg` |
| `--use-mcore-models` | 启用 Mcore 格式 | 必需 |
| `--load-dir` | 源模型路径 | 必需 |
| `--save-dir` | 输出路径 | 必需 |
| `--model-type-hf` | HF 模型类型 | `llama2` |
| `--tokenizer-model` | 分词器路径 | 必需 |
| `--params-dtype` | 权重精度（v1 必需） | 无（推荐 `bf16`） |

### 并行参数

| 参数 | 说明 |
|------|------|
| `--target-tensor-parallel-size` | 张量并行度（TP） |
| `--target-pipeline-parallel-size` | 流水线并行度（PP） |
| `--target-expert-parallel-size` | 专家并行度（EP，默认 1） |
| `--num-layer-list` | 动态 PP：每阶段层数列表 |
| `--num-layers-per-virtual-pipeline-stage` | VPP 每阶段层数 |

> 动态 PP 和 VPP 不能同时使用。

### LoRA 参数

| 参数 | 说明 |
|------|------|
| `--lora-load` | LoRA 权重路径 |
| `--lora-r` | LoRA 秩 |
| `--lora-alpha` | LoRA 缩放因子 |
| `--lora-target-modules` | 目标模块列表 |
| `--save-lora-to-hf` | 仅导出 LoRA 到 HF 格式 |
| `--load-checkpoint-loosely` | 松散加载（LoRA-only 导出时需要） |

## v2 完整参数（convert_ckpt_v2.py）

### v2 新增参数

| 参数 | 说明 |
|------|------|
| `--expert-tensor-parallel-size` | 专家 TP（ETP，目前仅支持 1） |
| `--moe-grouped-gemm` | MoE 分组 GEMM 优化 |
| `--moe-tp-extend-ep` | TP 扩展 EP |
| `--mla-mm-split` | MLA 维度扩展（compressed q/kv） |
| `--mtp-num-layers` | MTP 层数 |
| `--schedules-method` | DualPipeV 调度：`dualpipev` |
| `--noop-layers` | 插入空操作层位置（逗号分隔） |
| `--num-layers` | 缩减层调试模式的层数 |
| `--first-k-dense-replace` | 缩减层模式中 MoE 前的 dense 层数 |

> v2 **不支持** LoRA/QLoRA 转换。

## MoE 模型转换

### Qwen3-MoE 235B 示例（v2）

```bash
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 32 \
    --num-layers-per-virtual-pipeline-stage 8 \
    --noop-layers 94,95 \
    --load-dir ./model_from_hf/qwen3_moe_hf/ \
    --save-dir ./model_weights/qwen3_moe_mcore/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```

### MoE 转换注意事项

- `--moe-grouped-gemm` 与 LoRA 转换不兼容
- MG→HF 时如果源权重使用了 ETP=1，必须添加 `--expert-tensor-parallel-size 1`
- noop-layers 在反向转换（MG→HF）时必须指定相同配置

## 常见转换场景

### 1. 标准 HF→MG（训练前）

```bash
python convert_ckpt.py \
    --use-mcore-models --model-type GPT \
    --load-model-type hf --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --load-dir ./model_from_hf/llama3-8b/ \
    --save-dir ./model_weights/llama3_8b_mcore/ \
    --tokenizer-model ./model_from_hf/llama3-8b/tokenizer.model \
    --model-type-hf llama2
```

### 2. 标准 MG→HF（训练后导出）

```bash
python convert_ckpt.py \
    --use-mcore-models --model-type GPT \
    --load-model-type mg --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/llama3_8b_mcore/ \
    --save-dir ./model_from_hf/llama3-8b/ \
    --tokenizer-model ./model_from_hf/llama3-8b/tokenizer.model
```

### 3. LoRA 合并 + 导出 HF

```bash
python convert_ckpt.py \
    --use-mcore-models --model-type GPT \
    --load-model-type mg --save-model-type hf \
    --load-dir ./model_weights/base_mcore/ \
    --lora-load ./lora_output/ \
    --lora-r 8 --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/merged/ \
    --tokenizer-model ./model_from_hf/tokenizer.json
```

### 4. LoRA-only 导出 HF（v1）

```bash
python convert_ckpt.py \
    --use-mcore-models --model-type GPT \
    --load-model-type mg --save-model-type hf \
    --save-lora-to-hf \
    --load-checkpoint-loosely \
    --load-dir ./lora_output/ \
    --lora-r 16 --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 \
    --save-dir ./lora_hf/ \
    --tokenizer-model ./model_from_hf/tokenizer.json
```

> LoRA 权重需使用 `--lora-ckpt-filter` 训练保存。

## 错误预防清单

- [ ] 确认训练 TP/PP/EP 配置后再转换
- [ ] 验证模型层数可被 PP 整除（否则用 `--num-layer-list`）
- [ ] 匹配分词器格式（`.model`、`.json`、`.tiktoken`）
- [ ] HF→MG 时添加 `--use-mcore-models`（v1）
- [ ] MG→HF 时：v1 设 TP=PP=1，v2 无需设置并行参数
- [ ] LoRA 的 rank/alpha/target-modules 与训练时一致
- [ ] 大模型（TB 级）用 v2
- [ ] 不要混用 v1 和 v2

## 官方参考

- [权重转换 v1 文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/checkpoint/checkpoint_convert.md)
- [权重转换 v2 文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/checkpoint/checkpoint_convert_v2.md)
