# vLLM-Ascend Troubleshooting Guide

This guide covers common issues and their solutions when running vLLM on Huawei Ascend NPUs.

---

## 1. Common Errors and Solutions

### Error: `RuntimeError: ACL error: 507015`

**Cause**: Device initialization failed. Usually due to missing environment variables or incorrect CANN installation.

**Solution**:
```bash
# Source CANN environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Verify installation
npu-smi info
```

### Error: `ImportError: cannot import name 'Platform' from 'vllm.platforms'`

**Cause**: vLLM version mismatch with vllm-ascend plugin.

**Solution**:
```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Install compatible versions
pip install vllm==0.11.4 vllm-ascend==0.11.4
```

### Error: `ModuleNotFoundError: No module named 'torch_npu'`

**Cause**: torch-npu not installed or incorrect version.

**Solution**:
```bash
# Install torch-npu for CANN 8.5.0
pip install torch-npu==2.5.1

# Verify installation
python -c "import torch_npu; print(torch_npu.__version__)"
```

### Error: `ValueError: Unsupported model architecture`

**Cause**: Model architecture not yet supported by vLLM-Ascend.

**Solution**:
- Check the [vLLM-Ascend model support list](https://docs.vllm.ai/projects/ascend/en/latest/supported_models.html)
- Try using a compatible model format (e.g., convert to Safetensors)
- Update to the latest vLLM-Ascend version

### Error: `RuntimeError: ACL error: 107002`

**Cause**: Invalid parameter or configuration for NPU operation.

**Solution**:
- Check model path is correct
- Verify tokenizer is accessible
- Reduce `--max-model-len` if value exceeds NPU capability
### Error: `RuntimeError: An attempt has been made to start a new process...`

**Cause**: Python multiprocessing spawn mode requires code protection.

**Solution**:
```python
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def main():
    from vllm import LLM
    llm = LLM(...)

if __name__ == "__main__":
    main()
```

### Error: `RuntimeError: Cannot re-initialize NPU in forked subprocess`

**Cause**: vLLM-Ascend requires spawn multiprocessing method.

**Solution**:
```bash
# Set before running Python
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Or in Python before importing vllm
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

---

## 2. CANN Version Issues

### Checking CANN Version

```bash
# Check toolkit version
cat /usr/local/Ascend/ascend-toolkit/version.info

# Check firmware version
npu-smi info -t board

# Check driver version
cat /var/log/npu/slog/host-0/*.log 2>/dev/null | head -20
```

### Version Mismatch Symptoms

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ACL error: 507011` | CANN version incompatible | Upgrade to CANN 8.5.0+ |
| Slow initialization | Driver/CANN version mismatch | Update both driver and CANN |
| Random crashes during inference | Firmware too old | Upgrade firmware |
| Missing operators | CANN toolkit incomplete | Reinstall CANN with all components |

### Fixing Version Issues

**Step 1**: Verify current versions
```bash
# CANN toolkit
cat /usr/local/Ascend/ascend-toolkit/version.info 2>/dev/null || echo "CANN not found"

# Driver
npu-smi info -t board | grep "Driver Version"

# Python packages
pip list | grep -E "(torch|vllm|ascend)"
```

**Step 2**: Update CANN (if needed)
```bash
# Download CANN 8.5.0 from Huawei website
# Extract and install
cd /path/to/cann-package
chmod +x Ascend-cann-toolkit*.run
./Ascend-cann-toolkit*.run --install --quiet

# Set environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Step 3**: Reinstall compatible Python packages
```bash
pip uninstall torch-npu vllm-ascend -y

# Reinstall with correct versions
pip install torch-npu==2.5.1
pip install vllm-ascend
```

---

## 3. Memory Issues

### Out of Memory Errors

**Error**: `RuntimeError: NPU out of memory` or `ACL error: 507011`

#### Quick Fixes

```bash
# Reduce maximum sequence length
vllm serve <model> --max-model-len 4096

# Reduce concurrent sequences
vllm serve <model> --max-num-seqs 128

# Use smaller batch size
vllm serve <model> --max-num-batched-tokens 2048
```

#### Memory Optimization Tips

**1. Large Models (30B+ MoE)**

For MoE models like Qwen3-30B-A3B, use conservative settings:
```python
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM

llm = LLM(
    model="Qwen3-30B-A3B",
    tensor_parallel_size=2,  # Use multiple NPUs
    gpu_memory_utilization=0.5,  # Lower for MoE models
    max_model_len=2048,  # Reduce context length
    max_num_seqs=128     # Reduce concurrent sequences
)
```

**2. Enable Quantization**
```bash
# INT8 quantization (recommended)
vllm serve <model> --quantization w8a8

# INT4 quantization (highest compression)
vllm serve <model> --quantization w4a16
```

**3. Use Chunked Prefill**
```bash
vllm serve <model> --enable-chunked-prefill
```

**4. Limit GPU Memory Utilization**
```bash
# Default is 0.9, reduce if OOM during warmup
vllm serve <model> --gpu-memory-utilization 0.85

# For 30B+ MoE models, use 0.5 or lower
vllm serve <model> --gpu-memory-utilization 0.5 --max-model-len 2048
```

**5. Enable Prefix Caching**
```bash
vllm serve <model> --enable-prefix-caching
```
### Checking NPU Memory Usage

```bash
# Real-time monitoring
npu-smi info -t memory

# Detailed per-process info
npu-smi info -t processes

# Python check
python -c "
import torch
import torch_npu
print(f'Total memory: {torch_npu.npu.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'Allocated: {torch_npu.npu.memory_allocated() / 1e9:.2f} GB')
print(f'Cached: {torch_npu.npu.memory_reserved() / 1e9:.2f} GB')
"
```

### Memory Leak Detection

If memory grows over time:

```python
# Add to your code for debugging
import torch_npu

def print_memory_stats():
    allocated = torch_npu.npu.memory_allocated() / 1e9
    reserved = torch_npu.npu.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Call periodically
torch_npu.npu.empty_cache()  # Force cache cleanup
```

---

## 4. Network Issues

### Multi-Node Connectivity Problems

**Symptom**: Distributed training/inference hangs or fails to connect between nodes.

#### Pre-Flight Checks

```bash
# 1. Check network connectivity between nodes
ping <other-node-ip>

# 2. Verify SSH access
ssh <other-node> "npu-smi info"

# 3. Check firewall settings
iptables -L | grep -i drop

# 4. Test NCCL/HCCL connectivity
# Run on all nodes simultaneously
hccl_test -b 8M -e 1G -f 2 -p 8 -c
```

#### Environment Setup for Multi-Node

```bash
# Node 0 (master)
export MASTER_ADDR=<node0-ip>
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
export VLLM_HOST_IP=<node0-ip>

# Node 1
export MASTER_ADDR=<node0-ip>
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1
export VLLM_HOST_IP=<node1-ip>
```

### HCCL Errors

**Error**: `HCCL error: hccl communication failure`

**Causes and Solutions**:

1. **Network Interface Mismatch**
   ```bash
   # Find correct network interface
   ip addr show
   
   # Set HCCL to use correct interface
   export HCCL_SOCKET_IFNAME=eth0
   export HCCL_INTRA_ROCE_ENABLE=0  # Disable RoCE if not available
   ```

2. **Port Conflicts**
   ```bash
   # Check if port is in use
   netstat -tuln | grep 29500
   
   # Use different port
   export MASTER_PORT=29501
   ```

3. **Inconsistent CANN Versions**
   ```bash
   # Verify all nodes have same CANN version
   for node in node1 node2 node3; do
       ssh $node "cat /usr/local/Ascend/ascend-toolkit/version.info"
   done
   ```

**Error**: `HCCL error: rank timeout`

**Solution**:
```bash
# Increase timeout
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600

# Check all NPUs are healthy
npu-smi info
```

### HuggingFace Connection Issues

**Error**: `Connection refused` or `Cannot reach HuggingFace`

**Solution**: Use ModelScope mirror
```bash
export VLLM_USE_MODELSCOPE=true

# Or set mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com
```

### Slow Inter-Node Communication

**Symptoms**: Low bandwidth between nodes, slower than expected distributed performance.

**Diagnosis**:
```bash
# Test bandwidth with HCCL test
mpirun -np 8 -H node1:4,node2:4 ./hccl_test -b 1G -e 1G -f 2

# Expected: ~100 Gbps for RoCE, ~10 Gbps for TCP
```

**Solutions**:
1. Enable RoCE if available: `export HCCL_INTRA_ROCE_ENABLE=1`
2. Use RDMA-capable network interfaces
3. Check switch configuration for jumbo frames
4. Verify NIC driver versions are consistent

---

## Additional Resources

- [vLLM-Ascend FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)
- [Huawei Ascend Documentation](https://www.hiascend.com/document)
- [HCCL User Guide](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/hccluserguide/hcclug_0010.html)
