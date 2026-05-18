---
name: mindspeed-mm-weight-prep
description: "MindSpeed-MM weight conversion guide using mm-convert CLI tool. Covers HuggingFace to MindSpeed-MM format conversion, reverse conversion, and PP weight resplitting. Supports Qwen2VLConverter, Qwen2_5_VLConverter, InternVLConverter, WanConverter and more. Use when converting multimodal model weights on Ascend NPU."
keywords:
    - mindspeed-mm
    - weight
    - weight conversion
    - mm-convert
    - hf_to_mm
    - mm_to_hf
    - checkpoint
    - converter
---

# MindSpeed-MM Weight Conversion

This Skill guides users through converting multimodal model weights between HuggingFace and MindSpeed-MM formats using the `mm-convert` CLI tool, including VLM (Vision-Language Models) and generative models (video generation, etc.).

## mm-convert CLI Overview

```bash
mm-convert -h
# Available subcommands: Qwen2VLConverter, Qwen2_5_VLConverter, InternVLConverter, WanConverter, ...
```

Converters support different operations depending on the model. Common operations include:

| Operation | Description |
|-----------|-------------|
| `hf_to_mm` | HuggingFace weights -> MindSpeed-MM format (Megatron models) |
| `mm_to_hf` | MindSpeed-MM weights -> HuggingFace format (export after training) |
| `hf_to_dcp` / `dcp_to_hf` | DCP format conversion (FSDP2 models, e.g., Qwen3VL, InternVL3.5) |
| `source_to_mm` | Source weights -> MM format (e.g., OpenSoraPlan v1.5) |
| `resplit` | PP weight resplitting (adjust parallelism strategy without re-converting) |

> **Note**: Not all converters support all operations. Check the [Model Registry](../mindspeed-mm-pipeline/references/model-registry.md) for the exact operations supported by each converter.

## Converter Selection

| Model Family | Converter Class | Key Parameters |
|--------------|----------------|----------------|
| Qwen2VL | `Qwen2VLConverter` | `llm_pp_layers`, `vit_pp_layers`, `tp_size` |
| Qwen2.5VL | `Qwen2_5_VLConverter` | `llm_pp_layers`, `vit_pp_layers`, `tp_size` |
| InternVL2.5, InternVL3 | `InternVLConverter` | `llm_pp_layers`, `vit_pp_layers`, `tp_size` |
| Wan2.1, Wan2.2 | `WanConverter` | `source_path`, `target_path`, `target_parallel_config.pp_layers` |

> **Selection rule**: Match the Converter directly based on the model name. Qwen2VL and Qwen2.5VL use different Converters and must not be mixed.

## Quick Start

### VLM: HuggingFace -> MindSpeed-MM

Using Qwen2.5-VL-3B-Instruct as an example (single device, no PP splitting):

```bash
mm-convert Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1
```

### VLM: MindSpeed-MM -> HuggingFace

```bash
mm-convert Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-3B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [36] \
  --cfg.parallel_config.vit_pp_layers [32] \
  --cfg.parallel_config.tp_size 1
```

> **Note**: For `mm_to_hf`, `llm_pp_layers` and `vit_pp_layers` use a 1D list (e.g., `[36]`), not a nested list. `--cfg.hf_config.hf_dir` must point to the original HF weights directory (containing config.json).

### Generative Model: Wan2.1 hf_to_mm

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

### Generative Model: Wan mm_to_hf

```bash
mm-convert WanConverter mm_to_hf \
  --cfg.source_path <saved_weight_path>/ \
  --cfg.target_path ./converted_weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.hf_dir <original_hf_weights>/transformer/
```

> WanConverter uses a different parameter system from VLM Converters: it uses `source_path` / `target_path` instead of `mm_dir` / `hf_config.hf_dir`.

## Parameter Passing Methods

`mm-convert` is based on jsonargparse and supports three parameter passing methods:

### 1. Command-line Arguments

Pass parameters directly on the command line with `--cfg.xxx` (as shown in the examples above).

### 2. YAML Configuration File

First generate an annotated template, then modify as needed:

```bash
# Generate configuration template
mm-convert Qwen2_5_VLConverter hf_to_mm --print_config=comments > config.yaml

# Use configuration file
mm-convert Qwen2_5_VLConverter hf_to_mm --config config.yaml
```

Configuration file example:

```yaml
cfg:
  mm_dir: "ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
  hf_config:
    hf_dir: "ckpt/hf_path/Qwen2.5-VL-3B-Instruct"
  parallel_config:
    llm_pp_layers: [[36]]
    vit_pp_layers: [[32]]
    tp_size: 1
```

### 3. Environment Variables

```bash
export JSONARGPARSE_DEFAULT_ENV=true
# Then set parameters via environment variables (refer to jsonargparse documentation for variable naming conventions)
```

## PP Splitting Guide

### PP Splitting for VLM Models

VLM models have two components that need splitting: LLM and ViT.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `llm_pp_layers` | Number of layers per PP stage for LLM | `[[1,10,10,7]]` means 4 stages |
| `vit_pp_layers` | Number of layers per PP stage for ViT | `[[32,0,0,0]]` means all ViT in stage 0 |
| `tp_size` | Tensor parallelism degree | Must match the TP value in the training script |

**Splitting Rules**:

- Sum of elements in `llm_pp_layers` = total number of LLM layers
- Sum of elements in `vit_pp_layers` = total number of ViT layers
- Both lists must have the same length (= number of PP stages)
- Must be consistent with the `pipeline_num_layers` configuration in `model.json`
- ViT layers are typically concentrated in stage 0 (e.g., `[[32,0,0,0]]`), but can also be distributed

**Example: 4-device PP (Qwen2.5-VL-7B, 28-layer LLM + 32-layer ViT)**:

```bash
--cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
--cfg.parallel_config.vit_pp_layers [[32,0,0,0]]
```

### PP Splitting for Wan Models

Wan models use the `target_parallel_config.pp_layers` parameter:

```bash
--cfg.target_parallel_config.pp_layers [[6,6,6,6]]  # 24-layer model, evenly split across 4 stages
```

### resplit: PP Weight Resplitting

When already-converted MM weights need a different PP strategy, use `resplit` instead of re-converting from HF:

```bash
mm-convert Qwen2_5_VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/old_pp/" \
  --cfg.target_dir "ckpt/mm_path/new_pp/" \
  --cfg.source_parallel_config.llm_pp_layers [[28]] \
  --cfg.source_parallel_config.vit_pp_layers [[32]] \
  --cfg.target_parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.target_parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.target_parallel_config.tp_size 1
```

## FAQ

**Q: Conversion fails due to transformers version mismatch**

Error messages like `KeyError` or weight key mismatches. This is caused by the local transformers version not matching the version corresponding to the model weights. Solution: re-download the latest weights from HuggingFace, or install a matching transformers version.

**Q: Weights saved from LayerZero training cannot be directly converted with mm_to_hf**

Weights saved from LayerZero training have a different format from the standard MM format. They need post-processing (merging or reordering) before running `mm_to_hf` conversion.

**Q: hf_to_mm and mm_to_hf use different pp_layers formats**

`hf_to_mm` uses nested lists `[[36]]`, while `mm_to_hf` uses 1D lists `[36]`. Be careful to distinguish between them.

**Q: Loading fails due to tp_size mismatch**

The `tp_size` during conversion must exactly match the TP value in the training script. If training uses TP=2, conversion must also set `tp_size 2`.

**Q: llm and vit PP layer list lengths are inconsistent**

`llm_pp_layers` and `vit_pp_layers` must have the same number of elements (i.e., the same number of PP stages).

## Usage Order

After weight conversion is complete:

1. **Start training** -> Refer to the MindSpeed-MM Training Skill

## References

- [Detailed Conversion Guide](references/conversion-guide.md) - Complete parameter tables for each Converter, multi-model multi-scale examples
- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
