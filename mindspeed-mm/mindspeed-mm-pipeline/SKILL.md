---
name: mindspeed-mm-pipeline
description: MindSpeed-MM skill router and model index for Huawei Ascend NPU. Use when the user is uncertain which MindSpeed-MM skill to use, needs to choose between understanding/generative/omni/audio model categories, or wants an overview of the full training pipeline. Routes to the appropriate leaf skill based on model type.
keywords:
    - mindspeed-mm
    - router
    - model-index
    - skill-selection
    - overview
---

# MindSpeed-MM End-to-End Multimodal Training Pipeline

This Skill is the **routing entry point** for all MindSpeed-MM Skills. It determines the model type based on user intent, routes to the corresponding Skill, and provides a complete pipeline overview.

## Model Type Router

```
User Intent → Model Type Detection → Target Skill
  "Train understanding model / VLM"       → mindspeed-mm-vlm
  "Train generative model / video / image" → mindspeed-mm-generative
  "Train omni model"                       → See examples/qwen2.5omni/README.md
  "Train speech / TTS model"               → See examples/whisper/ or examples/cosyvoice3/README.md
```

**Routing Criteria**:

| Keywords | Model Type | Target |
|----------|-----------|--------|
| VLM, vision-language, image-text understanding, OCR, Qwen2VL, InternVL, GLM4V | Understanding (VLM) | mindspeed-mm-vlm |
| Video generation, image generation, t2v, t2i, i2v, Wan, CogVideoX, FLUX | Generative | mindspeed-mm-generative |
| Omni, speech + vision + text | Omni | examples/qwen2.5omni/ |
| Speech recognition, TTS, ASR, Whisper, CosyVoice | Audio | examples/whisper/ or examples/cosyvoice3/ |
| DPO, GRPO, preference alignment, reinforcement learning | Post-training | See Post-training section |

## Complete Workflow Overview

### VLM Workflow

```
1. Environment Setup (mindspeed-mm-env-setup)
→ 2. Model Dependency Installation (mindspeed-mm-vlm Step 0)
→ 3. Weight Download + HF→MM Conversion (mindspeed-mm-weight-prep)
→ 4. Data Preprocessing (MLLM JSON)
→ 5. Training (pretrain_vlm.py)
→ 6. Inference Validation (inference_vlm.py)
→ 7. Evaluation (evaluate_vlm.py)
→ 8. Weight Export MM→HF (optional)
```

**Inter-Stage Data Flow**:

```
model_from_hf/Qwen2.5-VL-7B-Instruct/   ← Step 3 download
    ↓ mm-convert hf_to_mm
ckpt/mm_path/Qwen2.5-VL-7B-Instruct/    ← Step 3 output
    ↓ Used as the load path in model.json
    ↓
dataset/train.json + images/             ← Step 4 input (MLLM JSON format)
    ↓ Used directly, no binary preprocessing needed
    ↓
saved_ckpt/                              ← Step 5 output
    ↓ mm-convert mm_to_hf (optional)
model_from_hf/.../converted/             ← Step 8 output
```

### Generative Model Workflow

```
1. Environment Setup (mindspeed-mm-env-setup)
→ 2. Model Dependency Installation (mindspeed-mm-generative Step 0)
→ 3. Weight Download + HF→MM Conversion (mindspeed-mm-weight-prep)
→ 4. Data Preprocessing (video/image + caption JSON)
→ 5. Feature Extraction (VAE + TextEncoder)  ← VLM does not have this step
→ 6. Training (pretrain_sora.py)
→ 7. Inference Generation (inference_sora.py)
→ 8. Weight Export MM→HF (optional)
```

**Inter-Stage Data Flow**:

```
weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/  ← Step 3 download
    ↓ mm-convert WanConverter hf_to_mm
weights/.../transformer/                     ← Step 3 output (in-place conversion)
    ↓
dataset/videos/ + dataset/train.json         ← Step 4 input
    ↓ Feature extraction script
dataset/features/                            ← Step 5 output (VAE latents + text embeddings)
    ↓ Used as training data input
    ↓
saved_ckpt/                                  ← Step 6 output
    ↓ mm-convert WanConverter mm_to_hf (optional)
converted_weights/                           ← Step 8 output
```

> **Key difference between VLM and generative models**: Generative models require an additional **feature extraction** step before training (VAE encodes video/images into latents, TextEncoder encodes text into embeddings). VLM does not have this step.

## Full Model Index

### Understanding Models (VLM)

| Model | Specs | Entry Script | Status |
|-------|-------|-------------|--------|
| Qwen2VL | 2B/7B/72B | pretrain_vlm.py | Released |
| Qwen2.5VL | 3B/7B/32B/72B | pretrain_vlm.py | Released |
| Qwen3VL | 8B/30B/235B | pretrain_transformers.py | Released |
| InternVL2.5 | 4B/78B | pretrain_internvl.py | Released |
| InternVL3 | 8B/78B | pretrain_vlm.py | Released |
| InternVL3.5 | 30B | pretrain_transformers.py | Released |
| GLM4.1V | 9B | pretrain_vlm.py | Released |
| GLM4.5V | -- | pretrain_transformers.py | Prototype |
| DeepSeekVL2 | -- | pretrain_deepseekvl.py | Released |
| DeepSeekOCR | -- | finetune_ocr.py (custom) | Prototype |
| DeepSeekOCR2 | -- | finetune_ocr2.py (custom) | Prototype |
| JanusPro | -- | -- | -- |
| Ming | -- | finetune_vl.py (custom) | -- |
| Bagel | -- | pretrain_omni.py | -- |

### Generative Models

| Model | Subtask | Entry Script | Status |
|-------|---------|-------------|--------|
| Wan2.1 | t2v/i2v/v2v/flf2v | pretrain_sora.py | Released |
| Wan2.2 | t2v/i2v | pretrain_sora.py | Released |
| HunyuanVideo | t2v | pretrain_sora.py | Prototype |
| HunyuanVideo 1.5 | t2v | pretrain_sora.py | Prototype |
| CogVideoX | t2v | pretrain_sora.py | Released |
| FLUX | t2i | train_dreambooth_flux.py (diffusers) | Prototype |
| OpenSoraPlan 1.3 | t2v | pretrain_sora.py | Released |
| OpenSoraPlan 1.5 | t2v | pretrain_sora.py | Released |
| StepVideo | t2v | pretrain_sora.py | Prototype |
| LTX2 | t2v | mindspeed_mm/fsdp/train/trainer.py | -- |
| Lumina-mGPT | -- | pretrain_lumina.py | Released |

### Omni Models

| Model | Entry Script | Status |
|-------|-------------|--------|
| Qwen2.5Omni | pretrain_vlm.py | Released |
| Qwen3Omni | pretrain_transformers.py | Released |

### Audio Models

| Model | Entry Script | Status |
|-------|-------------|--------|
| Whisper | pretrain_whisper.py | -- |
| CosyVoice3 | mindspeed_mm/fsdp/tasks/cosyvoice3/train.py | -- |
| Qwen3TTS | mindspeed_mm/fsdp/train/trainer.py | -- |
| FunASR | mindspeed_mm/fsdp/tasks/funasr/trainer.py | -- |

### Post-training

| Task | Script | Applicable Models |
|------|--------|-------------------|
| DPO | posttrain_qwen2vl_dpo.py | Qwen2VL |
| DPO | posttrain_sora_dpo.py | Wan, Sora-like |
| GRPO | posttrain_flux_dancegrpo.py | FLUX |
| GRPO (verl) | verl_plugin/ | Qwen2.5VL |

## Entry Script Selection Rules

MindSpeed-MM has three entry script patterns:

1. **Megatron-based unified entry**: `pretrain_vlm.py` (VLM), `pretrain_sora.py` (generative) — most models use these
2. **Megatron-based model-specific entry**: `pretrain_internvl.py`, `pretrain_deepseekvl.py`, `pretrain_whisper.py`, `pretrain_lumina.py` — dedicated scripts for specific models
3. **FSDP2-based entry**: `pretrain_transformers.py` or `mindspeed_mm/fsdp/train/trainer.py` or `mindspeed_mm/fsdp/tasks/<model>/train.py` — newer models (Qwen3VL, Qwen3Omni, LTX2, CosyVoice3, Qwen3TTS, FunASR)

> **Always check the actual shell script** in `examples/<model_name>/` — do not assume from the model name.

> New models should use the unified entry. Legacy models still use model-specific entries and are being migrated gradually.

## Common Training Args Quick Reference

The following parameters apply to all model types. For full parameter descriptions, see [references/common-args.md](references/common-args.md).

### Parallelism Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--tensor-model-parallel-size` | Tensor parallelism degree (TP) | 1/2/4/8 |
| `--pipeline-model-parallel-size` | Pipeline parallelism degree (PP) | 1/2/4/8 |
| `--context-parallel-size` | Context parallelism degree (CP) | 1/2 |
| `--expert-model-parallel-size` | Expert parallelism degree (EP, for MoE models) | 1/2/4 |

### Batch and Sequence Parameters

| Parameter | Description |
|-----------|-------------|
| `--micro-batch-size` | Number of samples per device per step |
| `--global-batch-size` | Global batch size (= micro * DP * gradient_accum) |
| `--seq-length` | Training sequence length |

### Memory Optimization Parameters

| Parameter | Description |
|-----------|-------------|
| `--recompute-granularity` | Recomputation granularity: `full` / `selective` |
| `--recompute-method` | Recomputation method: `uniform` / `block` |
| `--use-distributed-optimizer` | Use ZeRO-1 distributed optimizer |
| `--sequence-parallel` | Sequence parallelism (reduces activation memory) |

### Training Control Parameters

| Parameter | Description |
|-----------|-------------|
| `--train-iters` | Total training steps |
| `--lr` | Initial learning rate |
| `--min-lr` | Minimum learning rate |
| `--lr-decay-style` | Learning rate decay strategy: `cosine` / `linear` |
| `--weight-decay` | Weight decay |
| `--bf16` | Use BF16 mixed precision |
| `--use-flash-attn` | Enable FlashAttention |

### Docker Runtime

| Setting | Recommendation |
|---|---|
| `--ipc=host` | Required for DataLoader shared memory |
| `--privileged` | Required for NPU device access |
| `--num-workers` | Set to 0 if Docker shm is insufficient |
| `MASTER_PORT` | Change if port conflict with stale processes |

## FSDP2 vs Megatron Backend Selection

MindSpeed-MM supports two distributed training backends:

| Feature | Megatron | FSDP2 |
|---------|----------|-------|
| Maturity | Mature and stable | Newer |
| Parallelism | Fine-grained TP/PP/CP/EP control | Automatic sharding |
| Configuration | Command-line arguments | `--fsdp2-config-path` specifies YAML |
| Supported Models | All models | Select models (Qwen3.5, CosyVoice3, Kimi-K2.5, etc.) |
| Advantage | Flexible and tunable | Simple configuration, easy to get started |

**Selection Guidelines**:
- Use the **Megatron** backend for most scenarios (better documentation and examples)
- If the model's official examples provide FSDP2 configuration and fine-grained parallelism tuning is not needed, FSDP2 is an option
- FSDP2 uses `--fsdp2-config-path` to specify the configuration file, replacing Megatron's TP/PP/CP parameters

## Parameter Consistency Rules

The following parameters **must** be consistent between weight conversion and training:

| Parameter | Weight Conversion (mm-convert) | Training Script |
|-----------|:------------------------------:|:---------------:|
| TP (`tensor-model-parallel-size` / `tp_size`) | Set | Must match |
| PP (`pipeline-model-parallel-size` / `pp_layers`) | Set | Must match |
| Model architecture | Determined by HF config | Must match |

> Inconsistent parameters will cause weight loading failures or shape mismatch errors.

## Pre-flight Checklist

Verify each item before starting deployment:

- [ ] Docker container created with `--privileged --ipc=host` (or `--shm-size=16g`)
- [ ] Model-specific dependencies installed (diffusers version, decord for video models)
- [ ] No stale torchrun processes holding MASTER_PORT
- [ ] NPU available: `python -c "import torch_npu; print(torch.npu.is_available())"`
- [ ] CANN environment activated: `npu-smi info`
- [ ] MindSpeed-MM installed: `pip show mindspeed-mm`
- [ ] Megatron module copied: `ls MindSpeed-MM/megatron/`
- [ ] Model type determined (VLM / Generative / Omni / Audio)
- [ ] Target model and specs confirmed
- [ ] HF weights fully downloaded
- [ ] TP/PP configuration determined and documented
- [ ] Training data prepared

## FAQ

**Q: How do I determine which Skill to use?**

Choose based on model type: use mindspeed-mm-vlm for VLM models, mindspeed-mm-generative for generative models. When in doubt, refer to the model index table above.

**Q: What if different models have conflicting dependency versions?**

MindSpeed-MM models have vastly different version requirements for transformers/diffusers/peft. It is strongly recommended to create a separate Docker container for each model. See the dependency conflict section in [mindspeed-mm-env-setup](../mindspeed-mm-env-setup/SKILL.md).

**Q: Where can I find training scripts and configurations for a specific model?**

Example scripts and YAML configurations for each model are located in the `MindSpeed-MM/examples/<model_name>/` directory.

**Q: What is the difference between pretrain_vlm.py and pretrain_qwen2vl.py?**

`pretrain_vlm.py` is the new unified entry point that differentiates models via YAML configuration. `pretrain_qwen2vl.py` is the legacy model-specific entry point. New models should use the unified entry; legacy models still use their dedicated entry points.

**Q: Why do generative models need a feature extraction step?**

Generative models (e.g., Wan, CogVideoX) do not directly ingest raw video/images during training. Instead, a VAE first encodes video into latent features, and a TextEncoder encodes text into embeddings. Training then loads these pre-extracted features directly. This avoids redundant encoding during training and significantly improves training efficiency.

**Q: Training fails with `Communication_Error_Bind_IP_Port`**

Stale process holding the port from a previous run. Kill zombie processes or change MASTER_PORT in the training script.

```bash
ps aux | grep torchrun | grep -v grep | awk '{print $2}' | xargs kill -9
```

## Related Skills

- [mindspeed-mm-env-setup](../mindspeed-mm-env-setup/SKILL.md) - Base environment setup
- [mindspeed-mm-weight-prep](../mindspeed-mm-weight-prep/SKILL.md) - Weight conversion (HF<->MM)
- mindspeed-mm-vlm - VLM model training (understanding)
- mindspeed-mm-generative - Generative model training (video/image)

## Reference Resources

- [Model Registry](references/model-registry.md) - Complete lookup table: model → backend, entry script, converter, feature extraction, requirements
- [Common Training Args Reference](references/common-args.md) - GPT_ARGS, MOE_ARGS, OUTPUT_ARGS, environment variables
- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
- [MindSpeed-MM Training Args Documentation](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/args_readme.md)
