# MindSpeed-MM Model Dependency Version Conflict Matrix

This document lists the version requirements for key dependency packages across MindSpeed-MM multimodal models, along with their conflicts with the base environment (pyproject.toml).

## Version Conflict Matrix

| Model | transformers | diffusers | peft | Additional Dependencies |
|---|---|---|---|---|
| **Base (pyproject.toml)** | 4.57.0 | 0.30.3 | 0.7.1 | -- |
| **qwen3vl** | from git (c0dbe09) | -- | -- | triton-ascend, accelerate==1.2.0 |
| **qwen3_5** | from source | -- | -- | triton-ascend, accelerate==1.2.0 |
| **wan2.2** | -- | 0.35.1 | 0.17.1 | -- |
| **hunyuanvideo_1.5** | 4.57.1 | 0.35.0 | 0.17.0 | omegaconf, modelscope, angelslim |
| **deepseekocr** | 4.46.3 (downgrade!) | -- | -- | PyMuPDF, img2pdf |
| **cosyvoice3** | -- | -- | -- | torchaudio, openai-whisper, pyworld, librosa |
| **qwen3tts** | 4.57.3 | -- | -- | torchaudio, librosa, sox, gradio |

> `--` indicates that the model does not override the base version and uses the default from pyproject.toml.

## Conflict Analysis

### transformers Version Conflicts

transformers has the most severe conflicts:

- **Base version**: 4.57.0
- **qwen3vl / qwen3_5**: Requires installation from git source at a specific commit (c0dbe09), incompatible with any PyPI version
- **hunyuanvideo_1.5**: Requires 4.57.1 (minor version upgrade, relatively low risk)
- **qwen3tts**: Requires 4.57.3 (minor version upgrade, relatively low risk)
- **deepseekocr**: Requires 4.46.3 (**major downgrade**, conflicts with all other models)

### diffusers Version Conflicts

- **Base version**: 0.30.3
- **wan2.2**: Requires 0.35.1 (major version upgrade)
- **hunyuanvideo_1.5**: Requires 0.35.0 (major version upgrade, but also incompatible with wan2.2's 0.35.1)

### peft Version Conflicts

- **Base version**: 0.7.1
- **wan2.2**: Requires 0.17.1 (major version upgrade)
- **hunyuanvideo_1.5**: Requires 0.17.0 (major version upgrade, incompatible with wan2.2's 0.17.1)

## Isolation Recommendations

Due to severe version conflicts, **it is strongly recommended to use an isolated environment for each model**:

### Option 1: Separate Docker Containers (Recommended)

Create a separate Docker container for each model, installing dependencies independently based on the CANN base image:

```bash
# Example: Create a separate container for qwen3vl
docker run -it --name mindspeed-mm-qwen3vl \
    --device /dev/davinci0 \
    ... \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10 \
    bash

# After completing the base installation inside the container, install qwen3vl-specific dependencies
```

### Option 2: conda Virtual Environments

```bash
# Create a separate environment for each model
conda create -n mm-qwen3vl python=3.10
conda create -n mm-wan22 python=3.10
conda create -n mm-hunyuanvideo python=3.10

# Complete base installation + model-specific dependencies in each environment
conda activate mm-qwen3vl
# ... installation steps ...
```

### Option 3: Python venv

```bash
python -m venv /opt/envs/mm-qwen3vl
source /opt/envs/mm-qwen3vl/bin/activate
# ... installation steps ...
```

## Model Grouping Recommendations

The following models can share the same environment (no dependency conflicts):

| Environment Group | Shareable Models | Notes |
|---|---|---|
| Base group | cosyvoice3 | Does not override base dependency versions; only adds extra dependencies |
| qwen group | qwen3vl, qwen3_5 | Both require git-source transformers + triton-ascend |
| Video generation group | -- | wan2.2 and hunyuanvideo_1.5 have incompatible diffusers/peft versions and cannot share an environment |

> Note: Even for models marked as shareable, it is still recommended to use separate environments in practice to avoid potential hidden conflicts.
