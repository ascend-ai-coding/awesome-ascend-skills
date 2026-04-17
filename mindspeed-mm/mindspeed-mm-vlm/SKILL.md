---
name: mindspeed-mm-vlm
description: Universal VLM (vision-language understanding model) training guide for Huawei Ascend NPU using MindSpeed-MM. Covers all three framework patterns (Megatron, FSDP2, Custom trainers), weight conversion, dataset preparation (MLLM JSON format), fine-tuning, inference, and evaluation. Supports Qwen2.5VL, Qwen2VL, Qwen3VL, InternVL2.5/3/3.5, GLM4.1V, GLM4.5V, DeepSeekVL2, DeepSeekOCR, Ming, and more. Use when training or fine-tuning any multimodal understanding model on Ascend NPU.
keywords:
    - mindspeed-mm
    - vlm
    - multimodal understanding
    - qwen2.5vl
    - qwen3vl
    - internvl
    - glm4v
    - deepseekvl
    - finetune
    - fine-tuning
    - vision-language
    - megatron
    - fsdp2
---

# MindSpeed-MM VLM (Vision-Language Model) Training

This Skill guides users through training multimodal understanding (VLM) models on Huawei Ascend NPU using MindSpeed-MM. It uses Qwen2.5VL-3B as the flagship example and covers the end-to-end fine-tuning workflow.

## Prerequisites

> **Critical**: For most VLMs (Qwen2.5VL, Qwen2VL, InternVL, GLM4V, DeepSeekVL2), follow the **manual install** flow below. Do NOT use `bash scripts/install.sh` — official MindSpeed-MM docs state it only fully supports Qwen3/Qwen3.5. (For Qwen3VL / Qwen3.5, use one-click install + `bash examples/qwen3_5/install_extensions.sh`.)

### Step P1: Clone repositories

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git /root/workspace/MindSpeed-MM
git clone https://github.com/NVIDIA/Megatron-LM.git /root/workspace/Megatron-LM
cd /root/workspace/Megatron-LM && git checkout core_v0.12.1
cp -r megatron /root/workspace/MindSpeed-MM/
cd /root/workspace/MindSpeed-MM
```

### Step P2: Install PyTorch + torch_npu

**Always use pre-built wheels** from [Ascend PyTorch Releases](https://gitcode.com/Ascend/pytorch/releases). Do **not** use `pip install torch==2.7.1` — aarch64 wheels on PyPI are unreliable.

```bash
# Download matching wheels from https://gitcode.com/Ascend/pytorch/releases
# Replace cp310 with cp311 if using Python 3.11
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
# For x86_64: same naming pattern with _x86_64 suffix

# Required by torch_npu and other components:
pip install numpy pyyaml scipy attrs decorator psutil
```

### Step P3: Install MindSpeed

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git /root/workspace/MindSpeed
cd /root/workspace/MindSpeed
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -r requirements.txt
pip install -e .
cd /root/workspace/MindSpeed-MM
```

### Step P4: Install MindSpeed-MM base dependencies

```bash
# Installs base stack from pyproject.toml: transformers==4.57.0, diffusers==0.30.3, peft==0.7.1, etc.
pip install -e .
```

### Step P5: Activate CANN environment

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### Step P6: Smoke test

```bash
python3 -c "
import torch, torch_npu
assert torch.npu.is_available(), 'NPU not available'
print(f'NPU count: {torch_npu.npu.device_count()}')
import transformers, diffusers, mindspeed
print(f'transformers={transformers.__version__}, diffusers={diffusers.__version__}')
"
```

**Then proceed to Step 0** below for model-specific dependencies. For Qwen2.5VL the base is sufficient; other models need overlays (see Step 0 table).

For detailed troubleshooting and alternative install paths, see [mindspeed-mm-env-setup](../mindspeed-mm-env-setup/SKILL.md).

## Supported VLM Models

| Model | Framework | Entry Script | Sizes | Status |
|-------|-----------|-------------|-------|--------|
| Qwen2.5VL | Megatron | pretrain_vlm.py | 3B/7B/32B/72B | Released |
| Qwen2VL | Megatron | pretrain_vlm.py | 2B/7B/72B | Released |
| Qwen3VL | FSDP2 | pretrain_transformers.py | 8B/30B/32B/235B | Released |
| InternVL2.5 | Megatron | pretrain_internvl.py | 4B/78B | Released |
| InternVL3 | Megatron | pretrain_vlm.py | 8B/78B | Released |
| InternVL3.5 | FSDP2 | pretrain_transformers.py | -- | Released |
| GLM4.1V | Megatron | pretrain_vlm.py | 9B | Released |
| GLM4.5V | FSDP2 | pretrain_transformers.py | -- | Released |
| DeepSeekVL2 | Megatron | pretrain_deepseekvl.py | -- | Released |
| DeepSeekOCR | Custom | finetune_ocr.py | -- | Prototype |
| DeepSeekOCR2 | Custom | finetune_ocr2.py | -- | Prototype |
| Ming | Custom | finetune_vl.py | -- | Prototype |

> **Entry Script Note**: VLM models use different entry scripts. Check the shell script in `examples/<model_name>/` for the exact command — do not assume from the model name.

## How to Train Any VLM Model

The workflow for training **any** VLM model in MindSpeed-MM follows a universal pattern:

1. **Find the example**: Look in `examples/<model_name>/` for available shell scripts (finetune, pretrain, inference, evaluate)
2. **Read the shell script**: Identify the entry script, config files, and parallelism settings
3. **Check weight conversion**: Determine if the model needs `mm-convert` (Megatron models), DCP conversion (FSDP2 models), or uses HF weights directly (custom trainer models)
4. **Modify config files**: Update data paths and weight paths in the config files (JSON or YAML depending on the framework)
5. **Launch the shell script**: `bash examples/<model_name>/finetune_<model>_<size>.sh`

For other models, adapt the Qwen2.5VL quick start below. Use the [Model Registry](../mindspeed-mm-pipeline/references/model-registry.md) to look up the exact entry script, converter, and backend for your target model.

### Framework Patterns

All VLM models in MindSpeed-MM follow one of three framework patterns:

**Pattern 1: Megatron** (Qwen2.5VL, Qwen2VL, InternVL2.5, InternVL3, GLM4.1V, DeepSeekVL2)
```bash
# Uses JSON config files: data.json + model.json
# Weight conversion: mm-convert <Converter> hf_to_mm
torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
  --mm-data examples/<model>/data.json \
  --mm-model examples/<model>/model.json \
  --load $MM_WEIGHT_PATH ...
```

**Pattern 2: FSDP2** (Qwen3VL, InternVL3.5, GLM4.5V)
```bash
# Uses YAML config file; requires CUDA_DEVICE_MAX_CONNECTIONS=2
# Weight conversion: mm-convert <Converter> hf_to_dcp
# MoE models may need: --init-model-with-meta-device
CUDA_DEVICE_MAX_CONNECTIONS=2 torchrun $DISTRIBUTED_ARGS pretrain_transformers.py \
  --fsdp2-config-path examples/<model>/fsdp2_config.yaml ...
```

**Pattern 3: Custom Trainers** (DeepSeekOCR, DeepSeekOCR2, Ming)
```bash
# Standalone entry scripts; uses HF weights directly (no conversion needed)
torchrun $DISTRIBUTED_ARGS examples/<model>/finetune_<model>.py <args>
```

## Quick Start: Qwen2.5VL-3B Fine-Tuning (End-to-End)

> **This is the flagship Megatron-pattern example.** For other models, follow the same workflow structure but use configs and scripts from `examples/<model_name>/`. FSDP2 models use YAML configs instead of JSON; Custom trainer models skip weight conversion entirely. See the "Framework Patterns" section above.

### Step 0: Model-Specific Dependencies

Different VLM models have different version requirements for libraries such as transformers. Some of these requirements conflict, so environment isolation is necessary.

| Model | Additional Dependencies | Notes |
|-------|------------------------|-------|
| Qwen2.5VL | None required | Base environment is sufficient |
| Qwen3VL | `pip install -r examples/qwen3vl/requirements.txt` | Requires latest transformers; may conflict with other models |
| DeepSeekOCR | `pip install -r examples/deepseekocr/requirements.txt` | Downgrades transformers to 4.46.3; **must use isolated environment** |
| InternVL3 | None required | Base environment is sufficient (directory is `examples/internvl3/`, no extra requirements.txt) |

> **Important**: Qwen3VL and DeepSeekOCR have conflicting transformers version requirements. If you need to support both, use separate Docker containers or conda environments. See [per-model-deps.md](references/per-model-deps.md) for details.

### Step 1: Weight Download and Conversion

#### 1.1 Download HuggingFace Weights

```bash
# Download from HuggingFace (requires proxy)
http_proxy=http://127.0.0.1:58232 https_proxy=http://127.0.0.1:58232 \
  huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ckpt/hf_path/Qwen2.5-VL-3B-Instruct
```

#### 1.2 HF to MM Format Conversion

MindSpeed-MM uses its own weight format (MM format). Weights must be converted before training.

```bash
mm-convert Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1
```

Key parameter descriptions:

| Parameter | Description |
|-----------|-------------|
| `--cfg.mm_dir` | Output directory for converted MM format weights |
| `--cfg.hf_config.hf_dir` | Source HF weight directory |
| `--cfg.parallel_config.llm_pp_layers` | PP partitioning for LLM layers. `[[36]]` means all 36 layers on one device when PP=1 |
| `--cfg.parallel_config.vit_pp_layers` | PP partitioning for ViT layers. `[[32]]` means all 32 layers on one device when PP=1 |
| `--cfg.parallel_config.tp_size` | Tensor parallelism degree; must match the training configuration |

> **Important**: `llm_pp_layers` and `vit_pp_layers` must match the `pipeline_num_layers` in model.json. When PP > 1, you need to split the layer counts, e.g., `[[18,18]]` for PP=2.

#### 1.3 Converters for Different Models

| Model | Converter Name | Direction | Notes |
|-------|---------------|-----------|-------|
| Qwen2.5VL | `Qwen2_5_VLConverter` | hf_to_mm / mm_to_hf | See example above |
| Qwen2VL | `Qwen2VLConverter` | hf_to_mm / mm_to_hf | Same pattern as Qwen2.5VL |
| Qwen3VL | `Qwen3VLConverter` | hf_to_dcp / dcp_to_hf | **FSDP2**: uses DCP format, not mm format |
| Qwen3VL (Megatron) | `Qwen3VLMegatronConverter` | hf_to_mm / mm_to_hf | Alternative Megatron-path converter |
| InternVL2.5/3 | `InternVLConverter` | hf_to_mm / mm_to_hf | `--cfg.parallel_config.vit_pp_layers [[45]]` |
| InternVL3.5 | `ExpertMergeDcpConverter` | hf_to_dcp / dcp_to_hf | FSDP2 path; MoE model uses expert merge converter |
| DeepSeekVL2 | `DeepSeekVLConverter` | hf_to_mm only | `mm_to_hf` is unimplemented (stub). Note MoE expert layers |
| GLM4.1V | `GlmConverter` | hf_to_mm / mm_to_hf | Generic Megatron converter |
| GLM4.5V | `ExpertMergeDcpConverter` | hf_to_dcp / dcp_to_hf | FSDP2 path; MoE model uses expert merge converter |
| MoE models | `ExpertMergeDcpConverter` | merge / split | Merge/split MoE expert weights for DCP checkpoints |
| DeepSeekOCR/OCR2 | *None* | -- | Uses HF weights directly; no conversion needed |
| Ming | *None* | -- | Uses HF weights directly; no conversion needed |

### Step 2: Dataset Preparation

VLM training uses the **MLLM JSON format**. This example uses COCO2017 + LLaVA-Instruct-150K.

#### 2.1 Download Data

```bash
# 1. Download COCO2017 training images
mkdir -p data/COCO2017/train2017
# Extract to data/COCO2017/train2017/

# 2. Download LLaVA-Instruct-150K
# Download llava_instruct_150k.json to data/
```

#### 2.2 Convert to MLLM Format

> **Important**: The conversion script uses hardcoded relative paths. It expects `./data/llava_instruct_150k.json` as input and `./data/COCO2017/train2017/` for image lookup. Run it from the MindSpeed-MM root directory, and ensure data is at `./data/` (or create symlinks).

```bash
cd /root/workspace/MindSpeed-MM  # Must run from repo root
python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py
```

Output: `data/mllm_format_llava_instruct_data.json`

The script converts LLaVA format (`conversations`/`image`) to MLLM format (`messages`/`images`), which is what MindSpeed-MM expects.

#### 2.3 Data Directory Structure

```
data/
├── COCO2017/train2017/          # Image files
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── llava_instruct_150k.json     # Original data (for conversion)
└── mllm_format_llava_instruct_data.json  # MLLM format (for training)
```

> **Field name matching**: The field names in your data JSON must match the `attr` section in data.json config. Default mapping: `messages` for conversations, `images` for image paths. Using different field names (e.g., `conversations` instead of `messages`) will cause `KeyError`.

> For detailed MLLM JSON format specifications, see [data-format.md](references/data-format.md).

### Step 3: Configuration Files

Qwen2.5VL-3B fine-tuning requires two JSON configuration files:

#### 3.1 data.json (Data Configuration)

Path: `examples/qwen2.5vl/data_3b.json`

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-3B-Instruct"
        },
        "basic_parameters": {
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir"
        }
    }
}
```

**Must edit before running** — update these 3 paths to match your environment:

| Field | Description | What to set |
|-------|-------------|-------------|
| `model_name_or_path` | **Original HF weights** (not MM-converted) | e.g., `/home/weights/Qwen2.5-VL-3B-Instruct` |
| `dataset` | MLLM format JSON path | e.g., `./data/mllm_format_llava_instruct_data.json` |
| `cache_dir` | Data preprocessing cache (LOCAL path) | e.g., `./data/cache_dir` — **not NFS/shared mount** |

> **Multi-node Note**: `cache_dir` stores preprocessing cache. During multi-node training, each node must use a local path and must not point to an NFS shared directory, otherwise concurrent write conflicts will occur.

#### 3.2 model.json (Model Configuration)

Path: `examples/qwen2.5vl/model_3b.json`

model.json defines three components of the model architecture:

| Component | Type | Layers | Description |
|-----------|------|--------|-------------|
| `image_encoder` | qwen2vit | 32 | Vision encoder (ViT) |
| `vision_projector` | lnmlp | -- | Vision-to-text projection layer |
| `text_decoder` | qwen2_5_lm | 36 | Text decoder (LLM) |

> **Important**: The `pipeline_num_layers` field defines the PP partitioning scheme and must exactly match the `llm_pp_layers` / `vit_pp_layers` used during weight conversion.

### Step 4: Training Script Configuration

The training launch script is located at `examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`.

#### 4.1 Key Variables

```bash
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"   # Converted MM weights
SAVE_PATH="save_dir"                                 # Training output save path
DATA_CONFIG="examples/qwen2.5vl/data_3b.json"        # Data configuration
MODEL_CONFIG="examples/qwen2.5vl/model_3b.json"      # Model configuration
TP=1; PP=1; CP=1                                      # Parallelism configuration
MBS=1; GRAD_ACC_STEP=32                               # Batch size configuration
```

#### 4.2 Important Training Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--tensor-model-parallel-size` | Tensor parallelism degree (TP) | 1-8 |
| `--pipeline-model-parallel-size` | Pipeline parallelism degree (PP) | 1-4 |
| `--context-parallel-size` | Context parallelism degree (CP) | 1-2 |
| `--micro-batch-size` | Micro-batch size per NPU | 1 |
| `--global-batch-size` | Global batch size | 32 |
| `--lr` | Learning rate | 1e-5 to 2e-5 |
| `--train-iters` | Number of training iterations | 1000-5000 |
| `--bf16` | BFloat16 precision | Always recommended |
| `--use-distributed-optimizer` | Distributed optimizer | Recommended for multi-device |
| `--no-load-optim` | Do not load optimizer state | Required for fine-tuning |
| `--no-load-rng` | Do not load RNG state | Required for fine-tuning |

#### 4.3 Docker Users

**Docker users**: If your container lacks `--ipc=host` or `--shm-size=16g`, set `--num-workers 0` in the training script to avoid `Bus error` from DataLoader workers.

#### 4.4 Launch Command

```bash
torchrun $DISTRIBUTED_ARGS pretrain_vlm.py $GPT_ARGS $MM_ARGS $OUTPUT_ARGS
```

> **Note**: Qwen2.5VL uses `pretrain_qwen2vl.py` or `pretrain_vlm.py` as the entry point. Different versions of the script may vary; refer to the actual script in your installation.

#### 4.5 Single-Node vs Multi-Node

```bash
# Single node, 8 NPUs
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

# Multi-node (2 nodes)
# Node 0 (master node)
MASTER_ADDR=192.168.1.100  # Master node IP
NNODES=2
NODE_RANK=0

# Node 1
MASTER_ADDR=192.168.1.100  # Same as above; points to the master node
NNODES=2
NODE_RANK=1
```

### Step 5: Launch Training

```bash
bash examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
```

Key metrics to monitor in training logs:
- `lm loss`: Language model loss; should decrease steadily
- `grad norm`: Gradient norm; excessively large values may indicate the need to adjust the learning rate
- `iteration time`: Time per training iteration

### Step 6: Inference Verification (Optional)

After training, you can run inference verification. Note: no 3B-specific inference script exists; use the 7B script and adjust the model/config paths:

```bash
# Edit inference_qwen2_5_vl_7b.sh to point to your 3B checkpoint, then:
bash examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh
```

Inference requires an inference JSON configuration (`inference_qwen2_5_vl_7b.json`) specifying the model path and test images. The inference script uses the `inference_vlm.py` entry point.

### Step 7: Evaluation (Optional)

```bash
# Similarly, use the 7B evaluation script with 3B paths:
bash examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh
```

> **Note**: Only 7B and 72B inference/evaluation scripts are provided. For 3B or 32B, copy and adapt the 7B scripts with the appropriate model config and checkpoint paths.

## Recommended Parallelism Configurations by Model Size

| Model Size | NPU Count | Recommended TP | Recommended PP | MBS |
|------------|-----------|----------------|----------------|-----|
| 3B | 1-8 | 1 | 1 | 1 |
| 7B | 8 | 1-2 | 1-2 | 1 |
| 32B | 8-16 | 4 | 2 | 1 |
| 72B-78B | 16-32 | 8 | 2-4 | 1 |

## Post-Training: DPO Preference Alignment

Qwen2.5VL supports DPO post-training using the `posttrain_qwen2vl_dpo.py` entry point:

```bash
bash examples/qwen2vl/finetune_qwen2vl_72b_dpo.sh
```

See the relevant files in the `examples/qwen2vl/` directory for DPO configuration and data format details.

## FAQ

**Q: Training fails with `shape mismatch` after weight conversion**

`pipeline_num_layers` does not match the `llm_pp_layers` / `vit_pp_layers` used during weight conversion. Check that the layer count configuration in model.json matches the parameters passed to the `mm-convert` command.

**Q: Data loading hangs during multi-node training**

`cache_dir` is set to an NFS shared mount path. Change it to a local path (e.g., `/tmp/cache_dir`) so each node preprocesses data independently.

**Q: `transformers` version conflict**

Different VLM models require different versions of transformers. Qwen3VL needs the latest version, while DeepSeekOCR needs 4.46.3. Use separate containers to isolate environments.

**Q: Images fail to load during inference**

Check that the image paths in the inference JSON are correct and that the image files exist in a supported format (JPEG/PNG).

**Q: Training OOM (out of memory)**

Reduce `--micro-batch-size` to 1, enable `--use-distributed-optimizer`, or increase TP/PP parallelism.

**Q: Training fails with `Communication_Error_Bind_IP_Port`**

A stale process from a previous run is holding the port. Fix:
```bash
ps aux | grep torchrun | grep -v grep | awk '{print $2}' | xargs kill -9
# Or change MASTER_PORT in the training script
```

## Related Skills

- [mindspeed-mm-env-setup](../mindspeed-mm-env-setup/SKILL.md) - MindSpeed-MM Environment Setup
- [ascend-docker](../../ascend-docker/SKILL.md) - Ascend Docker Container Configuration
- [hccl-test](../../hccl-test/SKILL.md) - Multi-Device Communication Testing

## Reference Resources

- [Model Delta Cards](references/model-delta-cards.md) - Execution checklists for non-flagship models (Qwen3VL, InternVL3.5, GLM4.5V, DeepSeekVL2, DeepSeekOCR, Ming)
- [Model Dependency Configuration](references/per-model-deps.md) - Dependencies and version requirements for each VLM model
- [Data Format Specification](references/data-format.md) - Detailed MLLM JSON format documentation
- [Model Registry](../mindspeed-mm-pipeline/references/model-registry.md) - Complete lookup table for all models
- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
