---
name: mindspeed-mm-env-setup
description: MindSpeed-MM multimodal model suite environment setup guide for Huawei Ascend NPU. Covers CANN activation, PyTorch + torch_npu installation, MindSpeed acceleration library, Megatron-LM core module integration, and MindSpeed-MM installation. Use when setting up MindSpeed-MM multimodal training environment on Ascend NPU.
keywords:
    - mindspeed
    - mindspeed-mm
    - multimodal
    - environment
    - installation
    - cann
    - torch_npu
    - megatron
    - ascend npu
---

# MindSpeed-MM Ascend NPU Base Environment Setup

This skill guides users through setting up the **base** environment for MindSpeed-MM multimodal training on Huawei Ascend NPU.

> **Important**: This guide only covers the base environment. Different multimodal models (qwen3vl, wan2.2, hunyuanvideo, etc.) have vastly different dependency version requirements that may conflict with each other. Model-specific dependencies must be installed on top of the base environment. After completing this guide, refer to the corresponding model's SKILL for additional configuration.

## Component Relationship

```
Megatron-LM (NVIDIA)     <- Distributed training core (TP/PP), uses core_v0.12.1 branch
    |
MindSpeed (Huawei)       <- Ascend adaptation layer, monkey-patches Megatron kernels
    |
MindSpeed-MM (Huawei)    <- Multimodal application layer: VLM/generation/omni-modal training
```

MindSpeed-MM shares the underlying dependency stack (CANN, torch_npu, MindSpeed, Megatron-LM) with MindSpeed-LLM, but targets multimodal scenarios at the application level (vision-language models, video generation, speech synthesis, etc.).

## Quick Start -- 6 Steps to Complete the Base Environment

```bash
# 1. Activate CANN environment (CANN 8.5.0+ path; for older versions use /usr/local/Ascend/ascend-toolkit/set_env.sh)
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. Install PyTorch + torch_npu — USE PRE-BUILT WHEELS from https://gitcode.com/Ascend/pytorch/releases
# Do NOT use `pip install torch==2.7.1` — aarch64 wheels on PyPI are unreliable
# For Python 3.10 + aarch64:
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install numpy pyyaml scipy attrs decorator psutil

# 3. Clone and install MindSpeed (pinned to a specific commit)
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -r requirements.txt && pip install -e .
cd ..

# 4. Clone Megatron-LM and copy the core module into MindSpeed-MM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout core_v0.12.1 && cd ..
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cp -r Megatron-LM/megatron MindSpeed-MM/

# 5. Install MindSpeed-MM
cd MindSpeed-MM
pip install -e .

# 6. Verify the environment
python -c "
import torch
import torch_npu
print(f'NPUs available: {torch_npu.npu.device_count()}')
print(f'NPU ready: {torch.npu.is_available()}')
"
```

## One-Click Install Script (Qwen3/Qwen3.5 ONLY)

MindSpeed-MM provides `scripts/install.sh`, but **official docs state it only fully supports Qwen3 and Qwen3.5 models**. For all other models (Qwen2.5VL, Wan2.1/2.2, InternVL, HunyuanVideo, etc.), use the manual install above.

```bash
# ONLY for Qwen3 / Qwen3.5:
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
```

> **Warning**: On ARM (aarch64), `install.sh` uses `pip install torch torchvision torchaudio` from PyPI, which frequently fails because aarch64 wheels are not consistently available. For ARM, prefer the manual install method with pre-built wheels from [Ascend PyTorch Releases](https://gitcode.com/Ascend/pytorch/releases).

### install.sh Parameter Reference

| Flag | Long Flag | Description |
|------|-----------|-------------|
| `-t` | `--torchversion` | Specify PyTorch version string (any user-provided version, default `2.7.1`) |
| `-m` | `--msid` | **Required**. MindSpeed commit ID specifying the MindSpeed version to install |
| `-y` | `--yes` | Auto-confirm installation, skipping interactive prompts |
| `-n` | `--no` | Auto-skip all reinstallation prompts |
| `-mt` | `--megatron` | Install Megatron-LM (boolean toggle, default: SKIP; only installs when explicitly set) |
| `-ic` | `--install-cann` | Also install the CANN environment (not installed by default; assumes CANN is already present) |

**Example: Specify PyTorch 2.6.0 and auto-confirm**

```bash
bash scripts/install.sh --msid eb10b92 -t 2.6.0 -y
```

## Version Compatibility Matrix

| MindSpeed-MM | MindSpeed | Megatron-LM | PyTorch | torch_npu | CANN |
|---|---|---|---|---|---|
| master | master | core_v0.12.1 | 2.6.0, 2.7.1 | in-development | in-development |

> The above reflects the current master branch. Both MindSpeed-MM and MindSpeed are under rapid iteration; it is recommended to use a pinned commit rather than master HEAD.
> Check CANN version: `cat /usr/local/Ascend/cann/latest/aarch64-linux/ascend_toolkit_install.info` (CANN 8.5.0+) or `cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info` (older versions).

## Docker Container Creation

```bash
docker run -it --name mindspeed-mm \
    --privileged \
    --network host \
    --ipc=host \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/workspace:/home/workspace \
    <cann-image> bash
```

> **Warning**: If `--ipc=host` is not available, set `--shm-size=16g`. Without proper shared memory configuration, DataLoader workers will crash with `Bus error`. As a workaround, set `--num-workers 0` in all training and feature extraction scripts.

> Note: The CANN container image does not include ML dependencies; after entering the container you must manually install PyTorch, torch_npu, etc.
> **It is strongly recommended to create a separate container for each multimodal model** to avoid dependency version conflicts.

## Directory Structure

After installation, the workspace structure should look like this:

```
workspace/
├── MindSpeed/                    # Acceleration library
├── MindSpeed-MM/                 # Main project
│   ├── megatron/                 # Core module copied from Megatron-LM
│   ├── mindspeed_mm/             # Multimodal core code
│   ├── examples/                 # Example scripts and configs for each model
│   ├── scripts/
│   │   └── install.sh            # One-click install script
│   └── pyproject.toml            # Base dependency definitions
├── Megatron-LM/                  # NVIDIA original repo (used only to copy megatron/)
├── model_from_hf/                # Model weights
└── dataset/                      # Training datasets
```

## Base Dependency Versions (pyproject.toml)

The base installation (`pip install -e .`) installs the following core dependencies:

| Package | Base Version |
|---|---|
| torch | 2.7.1 |
| transformers | 4.57.0 |
| diffusers | 0.30.3 |
| peft | 0.7.1 |
| accelerate | 0.32.1 |

## Model-Specific Dependency Conflict Warning

> **Critical Warning**: The multimodal models supported by MindSpeed-MM have vastly different dependency version requirements that may conflict with the base environment.

**Known conflict examples**:
- **qwen3vl**: Requires installing transformers from git source (specific commit c0dbe09), incompatible with base version 4.57.0
- **wan2.2**: Requires diffusers==0.35.1, peft==0.17.1, overriding base versions
- **hunyuanvideo_1.5**: Requires transformers==4.57.1, diffusers==0.35.0, also incompatible with wan2.2
- **deepseekocr**: Requires transformers==4.46.3 (**downgrade!**), conflicts with all other models

**Docker shared memory**: When running in Docker containers, ensure adequate shared memory is configured (`--ipc=host` or `--shm-size=16g`). Default Docker shm (64MB) will cause `Bus error` in DataLoader workers. If you cannot change the container settings, set `--num-workers 0` in all training scripts.

**Isolation recommendations**:
- Create a separate Docker container or conda/venv environment for each model
- Complete the base environment setup first, then install model-specific dependencies in an isolated environment
- See [references/version-matrix.md](references/version-matrix.md) for the full conflict matrix

## Environment Verification Checklist

| Check Item | Command | Expected Result |
|------------|---------|-----------------|
| CANN environment | `npu-smi info` | Displays NPU device information |
| PyTorch | `python -c "import torch; print(torch.__version__)"` | `2.7.1` |
| torch_npu | `python -c "import torch_npu; print(torch_npu.__version__)"` | `2.7.1rc1` |
| NPU available | `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` | `True` |
| NPU count | `python -c "import torch_npu; print(torch_npu.npu.device_count())"` | >= 1 (matching hardware) |
| MindSpeed | `pip show mindspeed` | Displays package information |
| megatron module | `ls MindSpeed-MM/megatron/` | Directory exists and is non-empty |
| MindSpeed-MM | `pip show mindspeed-mm` | Displays package information |
| transformers | `python -c "import transformers; print(transformers.__version__)"` | `4.57.0` (base version) |
| diffusers | `python -c "import diffusers; print(diffusers.__version__)"` | `0.30.3` (base version) |

## Smoke Test

Run this quick preflight check before starting any training job:

```bash
python3 -c "
import torch, torch_npu
assert torch.npu.is_available(), 'NPU not available'
print(f'NPU count: {torch_npu.npu.device_count()}')
import transformers, diffusers, mindspeed
print(f'transformers={transformers.__version__}, diffusers={diffusers.__version__}')
print('Smoke test passed')
"
```

If any import fails or the assertion triggers, revisit the corresponding installation step above.

## FAQ

**Q: `ModuleNotFoundError: No module named 'yaml'` when importing torch_npu**

The CANN Docker image is missing basic Python packages:

```bash
pip install numpy pyyaml scipy attrs decorator psutil
```

**Q: pip install or git clone fails due to network issues**

Use a proxy:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

**Q: Cannot find the Megatron-LM checkout branch**

Make sure you are using the correct branch name (note the `core_v` prefix):

```bash
cd Megatron-LM
git branch -a | grep core
git checkout core_v0.12.1
```

**Q: MindSpeed commit does not exist**

Make sure you are using the full commit hash:

```bash
cd MindSpeed
git log --oneline | head -20
git checkout 93c45456c7044bacddebc5072316c01006c938f9
```

**Q: `pip install -e .` fails when installing MindSpeed-MM**

Check that the megatron module has been copied into the MindSpeed-MM directory and that pyproject.toml exists:

```bash
ls MindSpeed-MM/megatron/
ls MindSpeed-MM/pyproject.toml
```

**Q: Training fails with `Communication_Error_Bind_IP_Port`**

A stale process from a previous run is holding the port. Kill it and use a different port:

```bash
# Find and kill stale processes
ps aux | grep torchrun | grep -v grep | awk '{print $2}' | xargs kill -9
# Or use a different port in the training script
export MASTER_PORT=6100
```

**Q: Base environment is broken after installing model-specific dependencies**

This is expected behavior. Different models require conflicting versions of transformers/diffusers/peft. Solutions:

```bash
# Option 1: Separate Docker containers (recommended)
docker run -it --name mindspeed-mm-qwen3vl ...

# Option 2: conda virtual environments
conda create -n mm-qwen3vl python=3.10
conda activate mm-qwen3vl
# Re-run base installation + model-specific dependencies
```

## Next Steps

The base environment setup is complete. Next, install additional dependencies for the specific model you need. Refer to the corresponding model's SKILL.

## References

- [Version Conflict Matrix](references/version-matrix.md) - Dependency version conflict details for each model
- [MindSpeed-MM Repository](https://gitcode.com/ascend/MindSpeed-MM)
- [MindSpeed Repository](https://gitcode.com/ascend/MindSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Ascend Documentation](https://www.hiascend.com/document)
