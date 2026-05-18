# Per-Model Dependencies for Generative Models

This document lists the model-specific dependency requirements for each generative model supported by MindSpeed-MM. The base environment is covered by the `mindspeed-mm-env-setup` Skill.

> **Key principle**: Different models have severe dependency version conflicts. It is strongly recommended to create a separate Docker container or virtual environment for each model.

## Base Environment Dependencies (installed by default via pyproject.toml)

| Package | Base Version |
|---|---|
| torch | 2.7.1 |
| torch_npu | 2.7.1rc1 |
| transformers | 4.57.0 |
| diffusers | 0.30.3 |
| peft | 0.7.1 |
| accelerate | (base) |

## Wan2.1

**Override installation required**:

> **MANDATORY**: `diffusers==0.33.1` is a mandatory override — the base `diffusers==0.30.3` does **NOT** contain `AutoencoderKLWan` and will fail at feature extraction. You must upgrade before any Wan2.1 work.

```bash
pip install diffusers==0.33.1
pip install decord==0.6.0       # x86 architecture
# ARM architecture: see "Common Dependency: decord" section below for full build instructions
```

| Package | Version | Difference from Base |
|---|---|---|
| diffusers | 0.33.1 | Mandatory override (base 0.30.3 incompatible) |
| decord | 0.6.0 | New |

**Notes**: Wan2.1 has the smallest dependency override footprint and is the easiest generative model to deploy.

## Wan2.2

**Override installation required**:

```bash
pip install -r examples/wan2.2/requirements.txt
```

| Package | Version | Difference from Base |
|---|---|---|
| diffusers | 0.35.1 | Upgraded |
| peft | 0.17.1 | Upgraded (base 0.7.1) |
| decord | 0.6.0 | New |

**Conflict warnings**:
- `diffusers==0.35.1` is incompatible with Wan2.1's `0.33.1`
- `peft==0.17.1` differs significantly from base version `0.7.1`

## HunyuanVideo

**Override installation required**:

```bash
# Check dependency requirements under examples/hunyuanvideo/
pip install diffusers==0.30.3   # Same as base version
```

**Notes**: The base HunyuanVideo version has mild dependency requirements. Prototype stage.

## HunyuanVideo 1.5

**Override installation required**:

```bash
pip install -r examples/hunyuanvideo_1.5/requirements.txt
```

| Package | Version | Difference from Base |
|---|---|---|
| transformers | 4.57.1 | Minor upgrade |
| diffusers | 0.35.0 | Upgraded |

**Conflict warnings**:
- `diffusers==0.35.0` differs from Wan2.2's `0.35.1` (minor version difference may be compatible but is not guaranteed)
- `transformers==4.57.1` has a small difference from base `4.57.0`

## CogVideoX

**Override installation required**:

```bash
# Check dependency requirements under examples/cogvideox/
pip install diffusers==0.30.3   # Typically compatible with base version
```

**Notes**: CogVideoX 1.5 is friendly to the base environment dependencies. Released status.

## FLUX (Text-to-Image)

**Override installation required**:

```bash
# Check dependency requirements under examples/flux/
```

**Notes**: FLUX is the only t2i (text-to-image) model. Prototype stage. No video frame-related dependencies.

## OpenSoraPlan 1.3 / 1.5

**Override installation required**:

```bash
# Check dependency requirements under examples/opensoraplan1.3/ or examples/opensoraplan1.5/
pip install decord==0.6.0       # Video decoder
```

**Notes**: Released status, dependencies are relatively stable.

## StepVideo

**Override installation required**:

```bash
# Check dependency requirements under examples/stepvideo/
```

**Notes**: Prototype stage.

## Dependency Conflict Matrix

| Dependency | Base | Wan2.1 | Wan2.2 | HunyuanVideo 1.5 | CogVideoX |
|---|---|---|---|---|---|
| diffusers | 0.30.3 | 0.33.1 | 0.35.1 | 0.35.0 | 0.30.3 |
| peft | 0.7.1 | 0.7.1 | 0.17.1 | 0.7.1 | 0.7.1 |
| transformers | 4.57.0 | 4.57.0 | 4.57.0 | 4.57.1 | 4.57.0 |

**Conclusion**: Wan2.1 and CogVideoX can coexist in the same environment; Wan2.2 and HunyuanVideo 1.5 must have separate environments.

## Common Dependency: decord (Video Decoder)

All video generation models require decord to decode training videos.

**x86 architecture**:

```bash
pip install decord==0.6.0
```

**ARM architecture** (e.g., Kunpeng processors commonly used in Ascend servers):

decord does not provide pre-built ARM packages and must be compiled from source:

```bash
# Install build dependencies (libavfilter-dev is required — without it cmake fails with "FFMPEG_LIBAVFILTER-NOTFOUND")
apt-get update && apt-get install -y cmake build-essential \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev

git clone --recursive https://github.com/dmlc/decord.git /tmp/decord
cd /tmp/decord && mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd /tmp/decord/python && pip install .
```

## Post-Install Verification

After installing or upgrading model-specific dependencies (especially `diffusers`), core packages such as `numpy` and `pandas` may be silently upgraded to incompatible versions. Pin them back and verify:

```bash
pip install numpy==1.26.0 pandas==2.0.3
python3 -c "import numpy, pandas, sklearn; print('Dependencies OK')"
```

Run this check after every `pip install diffusers==...` or `pip install -r requirements.txt` to catch breakage early.

## Installation Recommendations

1. **Prefer the `requirements.txt` in the model directory**: If `examples/<model>/requirements.txt` exists, use it first
2. **Verify installed versions**: After installation, confirm versions with `pip list | grep -E "diffusers|peft|transformers"`
3. **Do not mix installations**: Installing dependencies for different models in the same environment will cause version overrides -- the later-installed model may work, but the earlier-installed model will break
