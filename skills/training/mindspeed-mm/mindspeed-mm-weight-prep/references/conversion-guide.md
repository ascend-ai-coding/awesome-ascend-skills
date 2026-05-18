# MindSpeed-MM Weight Conversion Detailed Guide

## Converter Overview

| Converter | Applicable Models | Supported Operations |
|-----------|-------------------|----------------------|
| `Qwen2VLConverter` | Qwen2-VL series | hf_to_mm, mm_to_hf, resplit |
| `Qwen2_5_VLConverter` | Qwen2.5-VL series | hf_to_mm, mm_to_hf, resplit |
| `InternVLConverter` | InternVL2.5, InternVL3 series | hf_to_mm, mm_to_hf, resplit |
| `WanConverter` | Wan2.1, Wan2.2 series | hf_to_mm, mm_to_hf, resplit |

## Qwen2_5_VLConverter Full Parameters

### hf_to_mm Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.mm_dir` | MindSpeed-MM format output path | Yes |
| `--cfg.hf_config.hf_dir` | HuggingFace source weights path | Yes |
| `--cfg.parallel_config.llm_pp_layers` | LLM PP splitting (nested list) | Yes |
| `--cfg.parallel_config.vit_pp_layers` | ViT PP splitting (nested list) | Yes |
| `--cfg.parallel_config.tp_size` | Tensor parallelism degree | Yes |

### mm_to_hf Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.save_hf_dir` | HuggingFace format output path | Yes |
| `--cfg.mm_dir` | MindSpeed-MM source weights path | Yes |
| `--cfg.hf_config.hf_dir` | Original HF weights path (with config.json) | Yes |
| `--cfg.parallel_config.llm_pp_layers` | LLM PP splitting (1D list) | Yes |
| `--cfg.parallel_config.vit_pp_layers` | ViT PP splitting (1D list) | Yes |
| `--cfg.parallel_config.tp_size` | Tensor parallelism degree | Yes |

### resplit Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.source_dir` | Source MM weights path | Yes |
| `--cfg.target_dir` | Target MM weights path | Yes |
| `--cfg.source_parallel_config.llm_pp_layers` | Source LLM PP splitting | Yes |
| `--cfg.source_parallel_config.vit_pp_layers` | Source ViT PP splitting | Yes |
| `--cfg.target_parallel_config.llm_pp_layers` | Target LLM PP splitting | Yes |
| `--cfg.target_parallel_config.vit_pp_layers` | Target ViT PP splitting | Yes |
| `--cfg.target_parallel_config.tp_size` | Target tensor parallelism degree | Yes |

## InternVLConverter Full Parameters

### hf_to_mm Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.mm_dir` | MindSpeed-MM format output path | Yes |
| `--cfg.hf_config.hf_dir` | HuggingFace source weights path | Yes |
| `--cfg.parallel_config.llm_pp_layers` | LLM PP splitting (nested list) | Yes |
| `--cfg.parallel_config.vit_pp_layers` | ViT PP splitting (nested list) | Yes |
| `--cfg.parallel_config.tp_size` | Tensor parallelism degree | Yes |

### mm_to_hf Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.save_hf_dir` | HuggingFace format output path | Yes |
| `--cfg.mm_dir` | MindSpeed-MM source weights path | Yes |
| `--cfg.hf_config.hf_dir` | Original HF weights path (with config.json) | Yes |
| `--cfg.parallel_config.llm_pp_layers` | LLM PP splitting (1D list) | Yes |
| `--cfg.parallel_config.vit_pp_layers` | ViT PP splitting (1D list) | Yes |
| `--cfg.parallel_config.tp_size` | Tensor parallelism degree | Yes |

## WanConverter Full Parameters

### hf_to_mm Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.source_path` | HuggingFace source weights path (transformer subdirectory) | Yes |
| `--cfg.target_path` | MindSpeed-MM format output path | Yes |
| `--cfg.target_parallel_config.pp_layers` | PP splitting (nested list, optional) | No |

### mm_to_hf Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--cfg.source_path` | MindSpeed-MM source weights path | Yes |
| `--cfg.target_path` | HuggingFace format output path | Yes |
| `--cfg.hf_dir` | Original HF weights path (used for structure reference) | Yes |

> WanConverter parameter naming differs significantly from VLM Converters: it uses `source_path`/`target_path` instead of `mm_dir`/`hf_config.hf_dir`, and has no `parallel_config` nesting.

## Multi-Model Conversion Examples

### Qwen2.5-VL-7B-Instruct (PP=4, TP=1)

**hf_to_mm**:

```bash
mm-convert Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1
```

- Qwen2.5-VL-7B LLM has 28 layers total, distributed as 1+10+10+7 across 4 PP stages
- ViT has 32 layers total, all placed in stage 0

**mm_to_hf**:

```bash
mm-convert Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
```

### Qwen2.5-VL-72B-Instruct (PP=4, TP=4)

**hf_to_mm**:

```bash
mm-convert Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,26,26,27]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 4
```

- Qwen2.5-VL-72B LLM has 80 layers total, distributed as 1+26+26+27 across 4 PP stages
- ViT has 32 layers total, all in stage 0
- TP=4 means at least 4*4=16 NPUs are required

**mm_to_hf**:

```bash
mm-convert Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-72B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,26,26,27] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 4
```

### Qwen2-VL-7B-Instruct (PP=2, TP=2)

**hf_to_mm** (note: use `Qwen2VLConverter`, not `Qwen2_5_VLConverter`):

```bash
mm-convert Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[14,14]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 2
```

### InternVL3-78B (PP=4, TP=2)

**hf_to_mm**:

```bash
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/InternVL3-78B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/InternVL3-78B" \
  --cfg.parallel_config.llm_pp_layers [[1,26,26,27]] \
  --cfg.parallel_config.vit_pp_layers [[45,0,0,0]] \
  --cfg.parallel_config.tp_size 2
```

- InternVL3-78B LLM part is based on Qwen2.5-72B, with 80 layers total
- InternViT has 45 layers
- TP=2, PP=4 requires 8 NPUs

**mm_to_hf**:

```bash
mm-convert InternVLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/InternVL3-78B" \
  --cfg.mm_dir "ckpt/mm_path/InternVL3-78B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/InternVL3-78B" \
  --cfg.parallel_config.llm_pp_layers [1,26,26,27] \
  --cfg.parallel_config.vit_pp_layers [45,0,0,0] \
  --cfg.parallel_config.tp_size 2
```

### InternVL2.5-8B (PP=2, TP=1)

**hf_to_mm**:

```bash
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/InternVL2_5-8B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/InternVL2_5-8B" \
  --cfg.parallel_config.llm_pp_layers [[16,16]] \
  --cfg.parallel_config.vit_pp_layers [[25,0]] \
  --cfg.parallel_config.tp_size 1
```

- InternVL2.5-8B LLM has 32 layers total, evenly split across 2 stages
- InternViT has 25 layers

### Wan2.1-T2V-1.3B (Single Device)

**hf_to_mm**:

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

**mm_to_hf**:

```bash
mm-convert WanConverter mm_to_hf \
  --cfg.source_path ./training_output/Wan2.1-T2V-1.3B/ \
  --cfg.target_path ./converted_weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.hf_dir ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

### Wan2.1-T2V-14B (PP=4)

**hf_to_mm**:

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/ \
  --cfg.target_parallel_config.pp_layers [[10,10,10,10]]
```

- Wan2.1-T2V-14B transformer has 40 layers total, evenly split across 4 PP stages

**mm_to_hf**:

```bash
mm-convert WanConverter mm_to_hf \
  --cfg.source_path ./training_output/Wan2.1-T2V-14B/ \
  --cfg.target_path ./converted_weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/ \
  --cfg.hf_dir ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/
```

### Wan2.1-I2V-14B (PP=4)

**hf_to_mm**:

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/transformer/ \
  --cfg.target_parallel_config.pp_layers [[10,10,10,10]]
```

## Common PP Splitting Configurations Reference

### Qwen2.5-VL Series

| Model | LLM Layers | ViT Layers | PP=1 | PP=2 | PP=4 |
|-------|------------|------------|------|------|------|
| Qwen2.5-VL-3B | 36 | 32 | `[[36]]` / `[[32]]` | `[[18,18]]` / `[[32,0]]` | `[[9,9,9,9]]` / `[[32,0,0,0]]` |
| Qwen2.5-VL-7B | 28 | 32 | `[[28]]` / `[[32]]` | `[[14,14]]` / `[[32,0]]` | `[[1,10,10,7]]` / `[[32,0,0,0]]` |
| Qwen2.5-VL-72B | 80 | 32 | `[[80]]` / `[[32]]` | `[[40,40]]` / `[[32,0]]` | `[[1,26,26,27]]` / `[[32,0,0,0]]` |

> Table format: `llm_pp_layers` / `vit_pp_layers`

### InternVL Series

| Model | LLM Layers | ViT Layers | PP=2 | PP=4 |
|-------|------------|------------|------|------|
| InternVL2.5-8B | 32 | 25 | `[[16,16]]` / `[[25,0]]` | `[[8,8,8,8]]` / `[[25,0,0,0]]` |
| InternVL3-78B | 80 | 45 | `[[40,40]]` / `[[45,0]]` | `[[1,26,26,27]]` / `[[45,0,0,0]]` |

### Wan Series

| Model | Transformer Layers | PP=1 | PP=2 | PP=4 |
|-------|--------------------|------|------|------|
| Wan2.1-T2V-1.3B | 24 | Not required | `[[12,12]]` | `[[6,6,6,6]]` |
| Wan2.1-T2V-14B | 40 | Not required | `[[20,20]]` | `[[10,10,10,10]]` |

## resplit Examples

### PP=1 -> PP=4 (Qwen2.5-VL-7B)

```bash
mm-convert Qwen2_5_VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/Qwen2.5-VL-7B-pp1" \
  --cfg.target_dir "ckpt/mm_path/Qwen2.5-VL-7B-pp4" \
  --cfg.source_parallel_config.llm_pp_layers [[28]] \
  --cfg.source_parallel_config.vit_pp_layers [[32]] \
  --cfg.target_parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.target_parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.target_parallel_config.tp_size 1
```

### PP=2 -> PP=4 (InternVL3-78B)

```bash
mm-convert InternVLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/InternVL3-78B-pp2" \
  --cfg.target_dir "ckpt/mm_path/InternVL3-78B-pp4" \
  --cfg.source_parallel_config.llm_pp_layers [[40,40]] \
  --cfg.source_parallel_config.vit_pp_layers [[45,0]] \
  --cfg.target_parallel_config.llm_pp_layers [[1,26,26,27]] \
  --cfg.target_parallel_config.vit_pp_layers [[45,0,0,0]] \
  --cfg.target_parallel_config.tp_size 2
```

## YAML Configuration File Examples

### Qwen2.5-VL-7B hf_to_mm Configuration

```yaml
cfg:
  mm_dir: "ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
  hf_config:
    hf_dir: "ckpt/hf_path/Qwen2.5-VL-7B-Instruct"
  parallel_config:
    llm_pp_layers: [[1,10,10,7]]
    vit_pp_layers: [[32,0,0,0]]
    tp_size: 1
```

### InternVL3-78B hf_to_mm Configuration

```yaml
cfg:
  mm_dir: "ckpt/mm_path/InternVL3-78B"
  hf_config:
    hf_dir: "ckpt/hf_path/InternVL3-78B"
  parallel_config:
    llm_pp_layers: [[1,26,26,27]]
    vit_pp_layers: [[45,0,0,0]]
    tp_size: 2
```

### WanConverter hf_to_mm Configuration

```yaml
cfg:
  source_path: "./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/"
  target_path: "./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/"
  target_parallel_config:
    pp_layers: [[10,10,10,10]]
```

## Error Prevention Checklist

- [ ] Confirm model family matches the Converter class (Qwen2VL vs Qwen2_5_VL)
- [ ] hf_to_mm uses nested lists `[[...]]`, mm_to_hf uses 1D lists `[...]`
- [ ] Sum of llm_pp_layers elements = total number of LLM layers
- [ ] Sum of vit_pp_layers elements = total number of ViT layers
- [ ] tp_size matches the TP value in the training script
- [ ] PP splitting is consistent with pipeline_num_layers in model.json
- [ ] For mm_to_hf, hf_config.hf_dir points to the original HF weights (with config.json)
- [ ] WanConverter uses source_path/target_path parameters, not mm_dir
- [ ] transformers version matches the model weights
- [ ] LayerZero training weights need post-processing before conversion

## Official References

- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
- [mm-convert Documentation](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/features/mm_convert.md)
