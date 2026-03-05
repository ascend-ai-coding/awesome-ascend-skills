# Troubleshooting Guide for Wan-Ascend Adaptation

## Device Management Issues

### Issue: `ImportError: cannot import name 'torch_npu'`
**Cause**: torch_npu not installed or incompatible torch version

**Solutions**:
1. Install torch_npu compatible with your torch version:
   ```bash
   # For torch 2.1.0
   pip install torch-npu==2.1.0
   
   # For torch 2.6.0
   pip install torch-npu==2.6.0
   ```
2. Verify CANN installation:
   ```bash
   cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
   ```

### Issue: `RuntimeError: No NPU device found`
**Cause**: NPU not detected or driver issues

**Solutions**:
1. Check NPU visibility:
   ```bash
   npu-smi info  # Should list available NPUs
   ```
2. Verify driver installation:
   ```bash
   cat /usr/local/Ascend/driver/version.info
   ```
3. Check environment variables:
   ```bash
   echo $ASCEND_DEVICE_ID  # Should be set
   ```

### Issue: Device string mismatch (`cuda:` vs `npu:`)
**Cause**: Code still using "npu:" but torch_npu expects "cuda:"

**Solution**: torch_npu provides compatibility layer - "cuda:" strings work:
```python
# This is fine with torch_npu
device = torch.device("cuda:0")  
model.to(device)  # Will use NPU if torch_npu is loaded
```

## Memory Issues

### Issue: `RuntimeError: NPU out of memory`
**Cause**: Insufficient NPU memory for model/data

**Solutions**:
1. **Enable gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(model.forward, x)
   ```

2. **Reduce batch size or sequence length**:
   ```python
   # In config
   batch_size = 1  # Reduce from 2
   max_seq_len = 1024  # Reduce from 2048
   ```

3. **Use Rainfusion sparse attention** (see `rainfusion_sparse_attn.md`):
   ```bash
   export SPARSITY=0.5  # 50% sparsity
   ```

4. **Apply W8A8 quantization** (see `quantization.md`):
   ```bash
   python quant_wan22.py --model_path ./model --is_dynamic True
   ```

5. **Tune memory allocator**:
   ```bash
   export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
   ```

### Issue: Slow memory allocation, fragmentation
**Cause**: Suboptimal NPU memory allocation

**Solution**: Configure allocator settings:
```bash
# Option 1: Expandable segments (recommended)
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'

# Option 2: Limit split size
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
```

## Performance Issues

### Issue: Slow attention computation
**Cause**: Using default attention algorithm

**Solutions**:
1. **Switch to high-performance algorithm**:
   ```bash
   export ALGO=1  # ascend_laser_attention (fastest)
   ```

2. **Enable communication/computation overlap**:
   ```bash
   export OVERLAP=1
   ```

3. **Use sequence parallel for long sequences** (>1024 tokens):
   ```python
   # In generate.py
   --ulysses_size 4  # Split across 4 NPUs
   ```

### Issue: Slow layer normalization
**Cause**: Using PyTorch native layer norm

**Solution**: Enable MindIE optimized layer norm:
```bash
export FAST_LAYERNORM=1
```

### Issue: Low NPU utilization
**Cause**: CPU bottlenecks, poor scheduling

**Solutions**:
1. **Enable task queue optimization**:
   ```bash
   export TASK_QUEUE_ENABLE=2
   ```

2. **Configure CPU affinity** for NUMA-aware scheduling:
   ```bash
   # For 8 NPUs on 4-socket system (24 cores each)
   export CPU_AFFINITY_CONF=0-23:24-47:48-71:72-95
   ```

## Distributed Training Issues

### Issue: Distributed training hangs
**Cause**: HCCL communication issues

**Solutions**:
1. **Verify HCCL backend**:
   ```python
   # In parallel_mgr.py
   backend = "hccl"  # Not "nccl"
   ```

2. **Check device mapping**:
   ```python
   # Each rank should use correct device
   torch.npu.set_device(local_rank)
   ```

3. **Add synchronization** after communication ops:
   ```python
   dist.all_to_all_single(output, input_t, group=group)
   torch.npu.synchronize()  # Critical for NPU
   ```

4. **Increase HCCL timeout**:
   ```bash
   export HCCL_CONNECT_TIMEOUT=7200
   export HCCL_EXEC_TIMEOUT=7200
   ```

### Issue: `RuntimeError: HCCL error`
**Cause**: Network or configuration issues

**Solutions**:
1. **Check network connectivity**:
   ```bash
   # Test between all nodes
   ping <other-node-ip>
   ```

2. **Verify rank table file** (if using):
   ```bash
   cat /usr/local/Ascend/ascend-toolkit/latest/hi_hccl.json
   ```

3. **Check environment variables**:
   ```bash
   echo $RANK_TABLE_FILE
   echo $RANK_ID
   echo $WORLD_SIZE
   ```

## Attention Mechanism Issues

### Issue: `RuntimeError: ALGO=1 failed`
**Cause**: ascend_laser_attention not supported for certain shapes

**Solutions**:
1. **Try different algorithm**:
   ```bash
   export ALGO=3  # npu_fused_infer_attention_score
   # or
   export ALGO=0  # Default fused_attn_score
   ```

2. **Check tensor layout** (must be BNSD):
   ```python
   # Transpose if needed
   q = q.transpose(1, 2)  # BHSD -> BNSD
   ```

### Issue: Attention output mismatch with original
**Cause**: Numerical precision differences

**Solutions**:
1. **Use same dtype** (bfloat16):
   ```python
   q = q.to(torch.bfloat16)
   k = k.to(torch.bfloat16)
   v = v.to(torch.bfloat16)
   ```

2. **Disable autocast** for RoPE:
   ```python
   @torch.amp.autocast('npu', enabled=False)
   def rope_apply(...):
       # RoPE computation
   ```

## Quantization Issues

### Issue: Quantization calibration fails
**Cause**: Unsupported layer or configuration

**Solutions**:
1. **Check quantization config**:
   ```python
   quant_config = QuantConfig(
       w_bit=8,
       a_bit=8,
       act_method=3,  # Use method 3 for dynamic
       is_dynamic=True
   )
   ```

2. **Verify device setting**:
   ```python
   # In quant_wan22.py
   dev_type="npu",  # Not "cpu"
   dev_id=0
   ```

3. **Ensure msmodelslim version compatibility**:
   ```bash
   pip install msmodelslim>=1.1.0
   ```

### Issue: Accuracy degradation after quantization
**Cause**: Aggressive quantization settings

**Solutions**:
1. **Use anti-outlier preprocessing**:
   ```python
   from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier
   anti_outlier = AntiOutlier(model, calib_data)
   anti_outlier.process()
   ```

2. **Try higher bit widths**:
   ```python
   quant_config = QuantConfig(
       w_bit=16,  # Use W16A16 instead of W8A8
       a_bit=16,
       is_dynamic=False
   )
   ```

## Rainfusion Sparse Attention Issues

### Issue: Rainfusion produces NaN values
**Cause**: Invalid sparsity or timestep settings

**Solutions**:
1. **Reduce sparsity**:
   ```python
   sparsity = 0.3  # Lower than 0.5
   ```

2. **Increase skip_timesteps**:
   ```python
   skip_timesteps = 20  # Start sparsity later
   ```

3. **Check bandwidth calculation**:
   ```python
   bandwidth = 1 - math.sqrt(sparsity)
   # Should be between 0 and 1
   ```

### Issue: Rainfusion slower than dense attention
**Cause**: Overhead for short sequences

**Solution**: Use only for long sequences (>1024 tokens):
```python
if seq_len > 1024:
    use_rainfusion = True
```

## Environment Setup Issues

### Issue: CANN version mismatch
**Cause**: Incompatible CANN/torch_npu versions

**Solution**: Check compatibility matrix:
```bash
# CANN 8.3.RC3 -> torch_npu 2.1.0
# CANN 8.5.0+ -> torch_npu 2.2.0+
```

### Issue: MindIE import error
**Cause**: MindIE not in PYTHONPATH

**Solution**:
```bash
export PYTHONPATH=/usr/local/Ascend/mindie/latest/lib:$PYTHONPATH
```

## Verification Checklist

Before debugging, verify:
- [ ] NPU visible: `npu-smi info` shows devices
- [ ] torch_npu installed: `python -c "import torch_npu; print(torch_npu.__version__)"`
- [ ] CANN installed: `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
- [ ] MindIE available: `python -c "import mindiesd; print(mindiesd.__version__)"`
- [ ] Environment variables set: `env | grep -E '(ALGO|FAST_LAYERNORM|PYTORCH_NPU)'`

## Getting Help

1. **Check logs**:
   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   export ASCEND_GLOBAL_LOG_LEVEL=3  # DEBUG level
   ```

2. **Collect system info**:
   ```bash
   npu-smi info > npu_info.txt
   cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg > cann_version.txt
   python -c "import torch; print(torch.__version__)" > torch_version.txt
   ```

3. **Minimal reproduction**: Create smallest possible script demonstrating the issue

4. **Contact support** with:
   - System info (npu-smi info output)
   - CANN version
   - torch/torch_npu versions
   - Minimal reproduction script
   - Full error traceback
