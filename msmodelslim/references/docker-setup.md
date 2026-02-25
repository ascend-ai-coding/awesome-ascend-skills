# Docker Setup for msmodelslim

This guide provides msmodelslim-specific Docker container configuration. For general Ascend Docker setup, see [ascend-docker](../../ascend-docker/SKILL.md).

## Recommended Images

For msmodelslim quantization tasks, use one of these images:

```bash
# vLLM-Ascend (recommended for vLLM deployment)
quay.io/ascend/vllm-ascend:v0.14.0rc1

# PyTorch Ascend (general purpose)
ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC1
```

## Container Setup

### Privileged Mode (Recommended)

```bash
docker run -d --name msmodelslim-test --network host \
  -v /home/weights:/home/weights \
  -v /root/msmodelslim-output:/root/output \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 \
  --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  <IMAGE> \
  sleep infinity
```

### Key Volume Mounts

| Volume | Purpose |
|--------|---------|
| `/home/weights` | Model weights directory |
| `/root/output` | Quantization output directory |
| `/usr/local/dcmi` | DCMI libraries |
| `/usr/local/bin/npu-smi` | NPU management tool |
| `/usr/local/Ascend/driver` | Ascend driver |

### Key Devices

| Device | Purpose |
|--------|---------|
| `/dev/davinci0-7` | NPU devices (adjust based on available devices) |
| `/dev/davinci_manager` | NPU device manager |
| `/dev/devmm_svm` | Device memory management |
| `/dev/hisi_hdc` | HDC communication |

## Install msmodelslim in Container

```bash
# Enter container
docker exec -it msmodelslim-test bash

# Install msmodelslim
cd /tmp
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim
bash install.sh
```

## Verify Environment

After container setup, verify the environment works correctly:

```bash
# Test NPU access
python3 -c "import torch; import torch_npu; a = torch.tensor(1).npu(); print('NPU OK')"

# Check msmodelslim installation
msmodelslim --help
```

## Troubleshooting

### BFLOAT16 Support Error

If you see `AclNN_Parameter_Error(EZ1001): Tensor self not implemented for DT_BFLOAT16`:

1. **Verify torch_npu works**:
   ```bash
   python3 -c "import torch; import torch_npu; a = torch.tensor(1).npu(); print('OK')"
   ```

2. **If failed**, the Docker image has compatibility issues:
   - Try a different/updated image
   - Reinstall torch_npu matching your CANN version
   - Ensure CANN 8.3.RC1+ for BF16 support

### Device Not Found

```bash
# Check available NPU devices on host
ls /dev/davinci* | grep -oE 'davinci[0-9]+$'

# Adjust --device flags accordingly
```

## Related References

- [ascend-docker skill](../../ascend-docker/SKILL.md) - Full Docker setup guide
- [installation.md](installation.md) - msmodelslim installation details
- [Ascend Docker Guide](https://www.hiascend.com/document/detail/zh/300Vtest/300VG/300V_0032.html)
