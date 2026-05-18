# VLM Model Dependency Configuration

This document lists the dependency requirements and installation methods for each VLM model supported by MindSpeed-MM.

## Dependency Conflict Overview

The primary conflict between VLM models lies in the `transformers` library version:

| Model | transformers Version | Conflict Risk |
|-------|---------------------|---------------|
| Qwen2.5VL | >= 4.45.0 | Low |
| Qwen3VL | Latest development version (bleeding-edge) | High; may break other models |
| InternVL2.5/3 | >= 4.44.0 | Low |
| GLM4.1V | >= 4.44.0 | Low |
| DeepSeekVL2 | >= 4.44.0 | Low |
| DeepSeekOCR | == 4.46.3 (downgrade) | High; conflicts with Qwen3VL |

> **Conclusion**: Qwen3VL and DeepSeekOCR must use separate environments and cannot be installed together.

## Per-Model Dependency Details

### Qwen2.5VL

**Additional dependencies**: None. The MindSpeed-MM base environment is sufficient.

```bash
# No additional installation required
# The base environment already includes all needed dependencies
```

Base environment requirements:
- torch >= 2.1
- torch_npu (matching the CANN version)
- transformers >= 4.45.0
- accelerate
- Pillow

### Qwen3VL

**Additional dependencies**: Requires bleeding-edge transformers.

```bash
pip install -r examples/qwen3vl/requirements.txt
```

Key dependency changes:
- transformers: Install the latest development version (may require git+https installation)
- May require updating related libraries such as tokenizers and safetensors

> **Warning**: Installation will overwrite the system transformers version, affecting models like Qwen2.5VL. Using a separate container is recommended.

### InternVL2.5 / InternVL3

**Additional dependencies**: None. The MindSpeed-MM base environment is sufficient.

```bash
# No additional installation required
# The actual directory is examples/internvl3/ (not examples/internvl/) and it has no requirements.txt
```

Key notes:
- transformers >= 4.44.0 (already satisfied by the base environment)
- timm and einops may be needed depending on usage, but are not listed in a requirements.txt

### GLM4.1V

**Additional dependencies**:

```bash
pip install -r examples/glm4v/requirements.txt
```

Generally compatible with the base environment; low conflict risk.

### DeepSeekVL2

**Additional dependencies**: None. The MindSpeed-MM base environment is sufficient.

```bash
# No additional installation required
# The actual directory is examples/deepseekvl2/ (not examples/deepseekvl/) and it has no requirements.txt
```

Note MoE-related dependencies. Generally compatible with the base environment.

### DeepSeekOCR

**Additional dependencies**:

```bash
pip install -r examples/deepseekocr/requirements.txt
```

Key changes:
- **transformers is downgraded to 4.46.3**
- This is a significant downgrade from the base 4.57.0 with different API/behavior, and will break Qwen2.5VL and Qwen3VL

> **Strongly recommended**: Use a separate Docker container for DeepSeekOCR-related tasks.

## Environment Isolation Strategies

### Option 1: Separate Docker Containers (Recommended)

Create a separate container for each model with conflicting dependencies:

```bash
# Qwen2.5VL / InternVL / GLM4V / DeepSeekVL2 share the base container
docker run -it --name mm-vlm-base ...

# Qwen3VL in a separate container
docker run -it --name mm-vlm-qwen3vl ...
pip install -r examples/qwen3vl/requirements.txt

# DeepSeekOCR in a separate container
docker run -it --name mm-vlm-deepseekocr ...
pip install -r examples/deepseekocr/requirements.txt
```

### Option 2: Conda Environments

```bash
# Base environment
conda create -n mm-vlm-base python=3.10
conda activate mm-vlm-base
# ... install base dependencies

# Qwen3VL environment
conda create -n mm-vlm-qwen3vl python=3.10
conda activate mm-vlm-qwen3vl
pip install -r examples/qwen3vl/requirements.txt

# DeepSeekOCR environment
conda create -n mm-vlm-deepseekocr python=3.10
conda activate mm-vlm-deepseekocr
pip install -r examples/deepseekocr/requirements.txt
```

## Compatibility Matrix

| Environment | Qwen2.5VL | Qwen3VL | InternVL | GLM4V | DeepSeekVL2 | DeepSeekOCR |
|-------------|-----------|---------|----------|-------|-------------|-------------|
| Base environment | OK | X | OK | OK | OK | X |
| Qwen3VL environment | Needs verification | OK | Needs verification | Needs verification | Needs verification | X |
| DeepSeekOCR environment | X | X | X | X | X | OK |

- **OK**: Verified to work correctly
- **X**: Incompatible or not verified
- **Needs verification**: Theoretically possible, but requires actual testing to confirm
