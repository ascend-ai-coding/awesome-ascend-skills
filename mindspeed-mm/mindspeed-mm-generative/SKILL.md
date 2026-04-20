---
name: mindspeed-mm-generative
description: Universal MindSpeed-MM generative model training guide for Huawei Ascend NPU. Covers all backend patterns (Megatron, Megatron+FSDP2, FSDP2-native, Accelerate+DeepSpeed), feature extraction, weight conversion, and training for ALL supported generative models. Supports Wan2.1/2.2, HunyuanVideo/1.5, CogVideoX, OpenSoraPlan, VACE, LTX2, FLUX, SD3, SDXL, Sana, HiDream, StepVideo, Lumina and more. Use when training multimodal generative models on Ascend NPU.
keywords:
    - mindspeed-mm
    - generative
    - video generation
    - image generation
    - wan
    - t2v
    - t2i
    - i2v
    - feature extraction
    - sora
    - diffusion
    - flux
    - hunyuanvideo
    - cogvideox
    - accelerate
    - fsdp2
---

# MindSpeed-MM Generative Model Training

This Skill guides users through the end-to-end training pipeline for multimodal generative models (text-to-video, text-to-image) on Huawei Ascend NPU.

## Prerequisites

> **Critical**: Do NOT use `bash scripts/install.sh` for generative models. Official MindSpeed-MM docs state `install.sh` only fully supports Qwen3/Qwen3.5. For Wan2.1 and other generative models, follow the **manual install** flow below (matches `examples/wan2.1/README.md`).

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

**Then proceed to Step 0** below for model-specific dependencies (diffusers upgrade, decord, etc.) — Prerequisites alone is NOT enough for any generative model.

For detailed troubleshooting and alternative install paths, see [mindspeed-mm-env-setup](../mindspeed-mm-env-setup/SKILL.md).

## Supported Models

| Model | Sub-tasks | Backend | Status |
|---|---|---|---|
| Wan2.1 | t2v, i2v, v2v, flf2v | Megatron | Released |
| Wan2.2 | t2v, i2v | Megatron+FSDP2 | Released |
| HunyuanVideo | t2v, i2v | Megatron | Prototype |
| HunyuanVideo 1.5 | t2v, i2v | Megatron+FSDP2 | Prototype |
| CogVideoX | t2v | Megatron | Released |
| OpenSoraPlan 1.3/1.5 | t2v | Megatron | Released |
| OpenSora 2.0 | t2v | Megatron | Released |
| StepVideo | t2v | Megatron | Prototype |
| VACE | t2v | Megatron+FSDP2 | Prototype |
| LTX2 | t2v | FSDP2-native | Prototype |
| FLUX | t2i | Accelerate+DeepSpeed | Prototype |
| SD3 | t2i | Accelerate+DeepSpeed | Prototype |
| SDXL | t2i | Accelerate+DeepSpeed | Prototype |
| Sana | t2i | Accelerate+DeepSpeed | Prototype |
| HiDream | t2i | Accelerate+DeepSpeed | Prototype |
| Lumina | t2v | Megatron | Prototype |

**Sub-task descriptions**:
- **t2v** (text-to-video): Generate video from text, the most fundamental generation task
- **i2v** (image-to-video): Generate video from image + text, requires a reference image and different model weights
- **v2v** (video-to-video): Video style transfer
- **flf2v** (first-last-frame-to-video): Generate intermediate video from the first and last frames

## How to Train Any Generative Model

For ANY generative model, use the [Model Registry](../mindspeed-mm-pipeline/references/model-registry.md) to look up the exact backend, entry script, converter, and feature extraction script. Then:

1. Check `examples/<model_name>/README.md` for model-specific instructions
2. Read the shell script to identify: entry script, config files, backend pattern
3. Check if the model needs feature extraction (not all do — see registry)
4. Check if the model needs weight conversion (diffusers models use HF weights directly)
5. Modify config files with your data/weight paths
6. Launch the shell script

### Backend Patterns

MindSpeed-MM uses four different training backends depending on the model:

| Backend | Entry Script | Models | Notes |
|---|---|---|---|
| Megatron | `pretrain_sora.py` | wan2.1, hunyuanvideo, cogvideox, opensoraplan1.3/1.5, opensora2.0, stepvideo | Standard Megatron distributed training |
| Megatron+FSDP2 | `pretrain_sora.py` + `--use-torch-fsdp2` | wan2.2, hunyuanvideo_1.5, vace | Requires `CUDA_DEVICE_MAX_CONNECTIONS=2` |
| FSDP2-native | `mindspeed_mm/fsdp/train/trainer.py` | ltx2 | Standalone FSDP2 trainer, no Megatron |
| Accelerate+DeepSpeed | `accelerate launch` | flux, sd3, sdxl, sana, hidream (all diffusers models) | Uses HF accelerate with DeepSpeed, completely different from Megatron |

### Feature Extraction Matrix

Not all models require feature extraction. Models that train on raw data skip this step entirely.

| Script | Models | Notes |
|---|---|---|
| `get_wan_feature.py` | Wan2.1 | VAE + text encoder pre-encoding |
| `get_hunyuan_feature.py` | HunyuanVideo | HunyuanVideo-specific VAE |
| `get_sora_feature.py` | CogVideoX, StepVideo, OpenSoraPlan 1.3 | Shared extraction script |
| `get_lumina_feature.py` | Lumina | Lumina-specific |
| `get_vace_feature.py` | VACE | VACE-specific |
| **Not needed** | wan2.2, OpenSoraPlan 1.5, ltx2, all diffusers models (flux, sd3, sdxl, sana, hidream) | Train directly on raw data |

### Weight Converter Reference

| Converter | Models |
|---|---|
| `WanConverter` | wan2.1, wan2.2 |
| `HunyuanVideoConverter` | hunyuanvideo, hunyuanvideo_1.5 |
| `CogVideoConverter` | cogvideox |
| `OpenSoraPlanConverter` | opensoraplan1.3/1.5 |
| `StepVideoConverter` | stepvideo |
| `LuminaConverter` | lumina |
| `VACEConverter` | vace |
| **No converter needed** | diffusers models (flux, sd3, sdxl, sana, hidream) -- use HF weights directly |

## Key Difference from VLM

The generative model training pipeline includes an additional **feature extraction** step (for models that need it):

```
Data Preparation → Feature Extraction (VAE + TextEncoder) → Training → Inference
```

VLMs (Vision-Language Models) train directly from raw image/video inputs, whereas generative models first use a VAE to encode video into latent space features and a text encoder to encode text into vectors, then train a diffusion model in the feature space. Some newer models (wan2.2, ltx2, diffusers models) skip feature extraction and train on raw data directly.

## Quick Start: Wan2.1-1.3B T2V (End-to-End, Flagship Example)

Using Wan2.1-1.3B text-to-video as the flagship example (Megatron backend with feature extraction). This section walks through the complete training pipeline. To adapt for other models, refer to the "How to Train Any Generative Model" section above and the "Adaptation Notes" section below.

### Step 0: Install Model-Specific Dependencies

> **MANDATORY**: Wan2.1 requires `diffusers==0.33.1`. The base environment ships `diffusers==0.30.3`, which does **NOT** contain `AutoencoderKLWan` and will fail at feature extraction. You **must** upgrade before proceeding.

```bash
# Wan2.1 — upgrade diffusers (MANDATORY)
pip install diffusers==0.33.1

# IMPORTANT: After upgrading diffusers, verify core dependencies haven't been broken
pip install numpy==1.26.0 pandas==2.0.3  # Pin back if upgraded
python3 -c "import numpy, pandas, sklearn; print('Dependencies OK')"

# Video decoder (DecordVideo is the official default for Wan2.1)
# x86 architecture:
pip install decord==0.6.0
# ARM (aarch64) — build decord from source (apt-get may need retries due to network):
apt-get update || (sleep 5 && apt-get update)  # retry if first attempt times out
apt-get install -y cmake build-essential \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev
git clone --recursive https://github.com/dmlc/decord.git /tmp/decord
cd /tmp/decord && mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd /tmp/decord/python && pip install .
# Note: libavfilter-dev is required — without it cmake fails with "FFMPEG_LIBAVFILTER-NOTFOUND"
# Verify: python3 -c "import decord; print(decord.__version__)"
```

> Different models have significantly different dependency versions. See [references/per-model-deps.md](references/per-model-deps.md) for details.
> It is strongly recommended to create a separate container for each model to avoid dependency conflicts.

### Step 1: Weight Download and Conversion

Download the `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` weights from HuggingFace.

**Weight conversion** (HuggingFace format -> MindSpeed-MM format):

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

**When using Pipeline Parallelism (PP)**, you need to additionally specify the layer splitting scheme:

```bash
mm-convert WanConverter hf_to_mm \
  --cfg.source_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.target_path ./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
  --cfg.target_parallel_config.pp_layers '[[7,8,8,7]]'
```

> **Why `LOAD_PATH` uses `transformer/` (parent dir)**: `mm-convert hf_to_mm` writes `latest_checkpointed_iteration.txt` in `transformer/` and stores weights in `transformer/release/`. Megatron checkpoint loading reads `latest_checkpointed_iteration.txt` to resolve the actual subdirectory automatically. Set `LOAD_PATH` to `.../transformer/`, **not** `.../transformer/release/`.
> For detailed information on weight conversion, refer to the `mindspeed-mm-weight-prep` Skill.

### Step 2: Dataset Preparation

Dataset directory structure:

```
<dataset>/
├── data.json       # Video-text pair metadata
└── videos/
    ├── video0001.mp4
    └── video0002.mp4
```

`data.json` format:

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "A cat playing with a ball in a garden.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {"height": 480, "width": 832}
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Ocean waves hitting the shore at sunset.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {"height": 480, "width": 832}
    }
]
```

**Field descriptions**:
- `path`: Relative path to the video file (relative to the dataset root directory)
- `cap`: Video description text
- `num_frames`: Number of frames (Wan2.1 typically uses 81 frames)
- `fps`: Frame rate
- `resolution`: Resolution (height x width)

Then edit `examples/wan2.1/feature_extract/data.txt`, with each line in the format:

```
<dataset_root_directory>,<data.json_path>
```

Example:

```
/home/dataset/wan_training,/home/dataset/wan_training/data.json
```

### Step 3: Feature Extraction

> **Docker users**: If your container was not created with `--ipc=host` or `--shm-size=16g`, DataLoader workers will crash with `Bus error`. Set `--num-workers 0` in the script as a workaround.

Feature extraction is a step unique to generative models. This step runs the VAE and text encoder on NPU to pre-encode raw video and text into feature vectors, so that training can directly use features without repeated encoding/decoding.

**Configuration files** (3 files need to be modified):

1. **`examples/wan2.1/feature_extract/model_t2v.json`**: Set the `from_pretrained` paths for the VAE and text encoder
2. **`examples/wan2.1/feature_extract/data.json`**: Set `num_frames`, `max_height`, `max_width`, and the tokenizer's `from_pretrained` path
3. **`mindspeed_mm/tools/tools.json`**: Set `sorafeature.save_path`, the output directory for extracted features

**Launch feature extraction**:

```bash
bash examples/wan2.1/feature_extract/feature_extraction.sh
```

Underlying call: `torchrun ... mindspeed_mm/tools/feature_extraction/get_wan_feature.py`

After extraction completes, the output structure is:

```
./sora_features/               ← sorafeature.save_path (feature dataset root)
├── data.jsonl                 ← file paths are RELATIVE: "features/test_0000.pt"
└── features/
    ├── test_0000.pt
    └── test_0001.pt
```

> **Path rule (important)**: `data_folder` in `feature_data.json` must point to the `save_path` root (e.g., `./sora_features/`), **NOT** `./sora_features/features/`. The `data.jsonl` already includes `features/` in its relative paths. Setting `data_folder` to the `features/` subdirectory causes double-nesting (`features/features/...`) and data loading failures.

> For detailed feature extraction configuration, see [references/feature-extraction.md](references/feature-extraction.md).

### Step 4: Training Configuration

After feature extraction is complete, configure the training script.

**Update data.txt**: Edit `examples/wan2.1/1.3b/t2v/data.txt` to point to the feature dataset root (e.g., `./sora_features`).

**Key configuration files**:

- **`feature_data.json`**: Dataset configuration. Set `data_folder` to `sorafeature.save_path` root (e.g., `./sora_features/`), `dataset_type: "feature"`, and tokenizer path
- **`pretrain_model.json`**: Model architecture configuration, defines the diffusion scheduler and predictor (wandit) architecture parameters

**Training script** `examples/wan2.1/1.3b/t2v/pretrain.sh` key parameters:

```bash
# Path configuration
LOAD_PATH="./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/"
SAVE_PATH="./output/wan2.1-1.3b-t2v/"

# Parallelism strategy
TP=1   # Tensor Parallelism
PP=1   # Pipeline Parallelism
VP=1   # Virtual Pipeline
CP=1   # Context Parallelism
MBS=1  # Micro Batch Size

# Memory optimization (key parameters)
--recompute-granularity full
--recompute-method block
--recompute-num-layers 20

# Distributed optimizer
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

### Step 5: Launch Training

> **Docker users**: If your container was not created with `--ipc=host` or `--shm-size=16g`, DataLoader workers will crash with `Bus error`. Set `--num-workers 0` in the script as a workaround.

```bash
bash examples/wan2.1/1.3b/t2v/pretrain.sh
```

Entry script: `pretrain_sora.py`

Training checkpoints are saved in the directory configured by `SAVE_PATH`.

### Step 6: Inference Generation (Optional)

After training completes, you can use the inference script to generate video from text prompts:

```bash
bash examples/wan2.1/1.3b/t2v/inference.sh
```

Entry script: `inference_sora.py`

## Adaptation Notes for Other Models

### Megatron Backend Models

These models follow the same pipeline as Wan2.1 (feature extraction -> training via `pretrain_sora.py`):

- **CogVideoX**: `examples/cogvideox/`, uses `CogVideoConverter`, feature extraction via `get_sora_feature.py`
- **OpenSoraPlan 1.3**: `examples/opensoraplan1.3/`, uses `OpenSoraPlanConverter` (hf_to_mm), feature extraction via `get_sora_feature.py`
- **OpenSoraPlan 1.5**: `examples/opensoraplan1.5/`, uses `OpenSoraPlanConverter` (source_to_mm), **no feature extraction** (trains on raw data)
- **OpenSora 2.0**: `examples/opensora2.0/`
- **HunyuanVideo**: `examples/hunyuanvideo/`, uses `HunyuanVideoConverter`, feature extraction via `get_hunyuan_feature.py`
- **StepVideo**: `examples/stepvideo/`, uses `StepVideoConverter`, feature extraction via `get_sora_feature.py`
- **Lumina**: uses `LuminaConverter`, feature extraction via `get_lumina_feature.py`

### Megatron+FSDP2 Backend Models

These models use `pretrain_sora.py` with `--use-torch-fsdp2` and require `CUDA_DEVICE_MAX_CONNECTIONS=2`:

- **Wan2.2**: `examples/wan2.2/`, uses `WanConverter`, **NO feature extraction needed** (trains on raw data). Additional deps: `pip install -r examples/wan2.2/requirements.txt` (diffusers==0.35.1, peft==0.17.1)
- **HunyuanVideo 1.5**: `examples/hunyuanvideo_1.5/`, uses `HunyuanVideoConverter`. Additional deps: `pip install -r examples/hunyuanvideo_1.5/requirements.txt` (transformers==4.57.1, diffusers==0.35.0)
- **VACE**: `examples/vace/`, uses `VACEConverter`, feature extraction via `get_vace_feature.py`

### FSDP2-Native Backend Models

- **LTX2**: Uses `mindspeed_mm/fsdp/train/trainer.py` directly, **NO feature extraction needed**, no Megatron dependency

### Accelerate+DeepSpeed Backend Models (Diffusers)

These models use `accelerate launch` with DeepSpeed, **NOT** `pretrain_sora.py`. They use HuggingFace weights directly (no converter needed) and do NOT need feature extraction:

- **FLUX**: `examples/diffusers/flux/`, text-to-image
- **SD3**: `examples/diffusers/sd3/`, text-to-image
- **SDXL**: `examples/diffusers/sdxl/`, text-to-image
- **Sana**: `examples/diffusers/sana/`, text-to-image
- **HiDream**: `examples/diffusers/hidream/`, text-to-image

> Diffusers models have a completely different training flow from Megatron models. Read the shell scripts in `examples/diffusers/<model>/` to understand the specific launch command and config structure.

## Post-Training

MindSpeed-MM supports the following post-training optimization methods:

| Method | Entry Script | Description |
|---|---|---|
| DPO | `posttrain_sora_dpo.py` | Direct Preference Optimization, used for aligning with human preferences |
| GRPO | `posttrain_flux_dancegrpo.py` | Group Relative Policy Optimization |

## Directory Structure Reference

```
MindSpeed-MM/
├── pretrain_sora.py                 # Megatron/FSDP2 training entry point
├── inference_sora.py                # Inference entry point
├── posttrain_sora_dpo.py            # DPO post-training entry point
├── posttrain_flux_dancegrpo.py      # GRPO post-training entry point
├── mindspeed_mm/
│   ├── fsdp/train/trainer.py        # FSDP2-native trainer (used by ltx2)
│   └── tools/
│       ├── tools.json               # Global config (sorafeature.save_path for feature output)
│       └── feature_extraction/
│           ├── get_wan_feature.py        # Wan2.1 feature extraction
│           ├── get_hunyuan_feature.py    # HunyuanVideo feature extraction
│           ├── get_sora_feature.py       # CogVideoX, StepVideo, OpenSoraPlan
│           ├── get_lumina_feature.py     # Lumina feature extraction
│           └── get_vace_feature.py       # VACE feature extraction
└── examples/
    ├── wan2.1/                      # Megatron backend
    │   ├── feature_extract/
    │   │   ├── feature_extraction.sh
    │   │   ├── model_t2v.json
    │   │   ├── data.json
    │   │   └── data.txt
    │   ├── 1.3b/t2v/
    │   │   ├── pretrain.sh
    │   │   ├── inference.sh
    │   │   ├── data.txt
    │   │   ├── feature_data.json
    │   │   └── pretrain_model.json
    │   └── 14b/t2v/
    ├── wan2.2/                      # Megatron+FSDP2 backend
    ├── hunyuanvideo/                # Megatron backend
    ├── hunyuanvideo_1.5/            # Megatron+FSDP2 backend
    ├── cogvideox/                   # Megatron backend
    ├── opensoraplan1.3/             # Megatron backend
    ├── opensoraplan1.5/             # Megatron backend
    ├── opensora2.0/                 # Megatron backend
    ├── stepvideo/                   # Megatron backend
    ├── vace/                        # Megatron+FSDP2 backend
    ├── ltx2/                        # FSDP2-native backend
    └── diffusers/                   # Accelerate+DeepSpeed backend
        ├── flux/
        ├── sd3/
        ├── sdxl/
        ├── sana/
        └── hidream/
```

## FAQ

**Q: Feature extraction reports OOM** -- Reduce `max_height`/`max_width` or `num_frames` in `data.json`, or reduce `--nproc_per_node` in torchrun.

**Q: Training loss does not converge** -- Check that `LOAD_PATH` points to `transformer/` (not model root), weight conversion completed successfully, and `dataset_type` in `feature_data.json` is `"feature"`.

**Q: Inference generates all-black or noisy video** -- Confirm the inference script loads the correct checkpoint path and sufficient training steps have been completed.

**Q: `mm-convert` command not found** -- Run `pip install -e .` to install MindSpeed-MM. `mm-convert` is a console entry point registered during installation.
**Q: Feature extraction cannot read videos** -- Install decord: `pip install decord==0.6.0` on x86, or compile from source on ARM.
**Q: `Communication_Error_Bind_IP_Port`** -- Stale process holding port. Kill with `lsof -i :29500` + `kill -9 <PID>`, or change `MASTER_PORT`.

## References

- [Model Delta Cards](references/model-delta-cards.md) - Execution checklists for non-flagship models (Wan2.2, HunyuanVideo, CogVideoX, Diffusers/FLUX)
- [Per-Model Dependencies](references/per-model-deps.md) - Dependency versions and installation instructions for each generative model
- [Feature Extraction Guide](references/feature-extraction.md) - Feature extraction configuration files, workflow, and troubleshooting
- [Model Registry](../mindspeed-mm-pipeline/references/model-registry.md) - Complete lookup table for all models
- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
