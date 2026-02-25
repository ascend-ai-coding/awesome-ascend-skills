# Installation Guide

Complete installation guide for msModelSlim on Huawei Ascend AI processors.

---

## Prerequisites

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.8+ (3.9+ recommended) |
| **CANN** | 8.2.RC1+ (8.3.RC1 or 8.5.0 recommended) |
| **PyTorch** | Ascend Extension for PyTorch (torch-npu) |

> **Note**: Python 3.8 may have issues with `accelerate` dependency. Upgrade to Python 3.9 if installation fails.

---

## Step 1: Install CANN

Download and install CANN from [Huawei Ascend](https://www.hiascend.com/developer/download/community/result?module=cann).

Select the appropriate version for your system:
- **aarch64** for ARM systems
- **x86_64** for x86 systems

### Installation Guide

Follow the official [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html).

### Environment Setup

```bash
# For CANN 8.3.RC1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# For CANN 8.5.0+
source /usr/local/Ascend/cann/set_env.sh
```

---

## Step 2: Install PyTorch Ascend

Install Ascend Extension for PyTorch (torch-npu).

### Installation Guide

Follow the official [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html) guide.

### Quick Install

```bash
# Install torch-npu (adjust version as needed)
pip install torch-npu
```

---

## Step 3: Install msModelSlim

### Standard Installation (Atlas A2/A3 Training/Inference Products)

```bash
# 1. Clone repository
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim

# 2. Run installation script
bash install.sh
```

### Atlas 300I Duo Series Installation

For Atlas 300I Duo cards with sparse quantization support:

```bash
# 1. Standard installation
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim
bash install.sh

# 2. Build weight compression component
# Navigate to site-packages location
cd ${PYTHON_SITE_PACKAGES}/msmodelslim/pytorch/weight_compression/compress_graph/

# Example path (adjust for your Python version):
# cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 3. Build with CANN path
sudo bash build.sh ${CANN_INSTALL_PATH}/ascend-toolkit/latest
# Example: sudo bash build.sh /usr/local/Ascend/ascend-toolkit/latest

# 4. Set permissions
chmod -R 550 build
```

> **Note**: Atlas 300I Duo only supports single-card, single-chip quantization.

---

## Step 4: Verify Installation

```bash
# Check installation
python3 -c "import msmodelslim; print(msmodelslim.__version__)"

# Check CLI
msmodelslim --help
```

---

## Post-Installation

### Environment Variables

For NPU multi-card quantization:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

### Docker Environment

If using vLLM-Ascend Docker image:

```bash
# Pull image
docker pull quay.io/ascend/vllm-ascend:latest

# Install msModelSlim inside container
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim && bash install.sh
```

---

## Uninstallation

```bash
pip uninstall msmodelslim -y
```

---

## Troubleshooting

### accelerate Installation Fails

**Problem**: `accelerate` dependency fails on Python 3.8.

**Solution**: Upgrade to Python 3.9+.

```bash
conda create -n msmodelslim python=3.9 -y
conda activate msmodelslim
```

### Module Import Conflicts

**Problem**: Import errors when running from source directory.

**Solution**: Do NOT run `msmodelslim` commands from within the cloned source directory.

```bash
# WRONG
cd msmodelslim
msmodelslim quant ...

# CORRECT
cd /some/other/directory
msmodelslim quant ...
```

### CANN Not Found

**Problem**: CANN environment not set up.

**Solution**: Source the environment script.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## Next Steps

- [Quick Start](../SKILL.md#quick-start) - Try quantization
- [Model Support](model-support.md) - Check supported models
- [Example Scripts](example-scripts.md) - Explore examples
