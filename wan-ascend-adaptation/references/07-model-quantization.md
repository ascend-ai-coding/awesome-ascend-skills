# Domain 7: Model Quantization

## Overview

The Ascend version supports W8A8 (8-bit weight, 8-bit activation) dynamic quantization using Huawei's `msmodelslim` library for offline quantization and `mindiesd` for runtime loading.

## 7.1 Offline Quantization Tool (`quant_wan22.py`)

### Usage

```bash
python quant_wan22.py \
    --task t2v-A14B \
    --ckpt_dir /path/to/model/weights \
    --quant_type W8A8 \
    --is_dynamic \
    --device_type cpu
```

### Quantization Flow

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# 1. Load model
model = WanModel.from_pretrained(ckpt_dir, subfolder=checkpoint_name)

# 2. Patch model attributes (required by msmodelslim)
model.config = lambda: None
model.config.num_hidden_layers = len(model.blocks)
model.dtype = next(model.parameters()).dtype

# 3. Configure quantization
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    w_sym=True,
    disable_names=[],    # No layers disabled
    dev_type=device_type,
    act_method=3,        # Dynamic activation quantization
    pr_method=0,
    is_dynamic=True,
    disable_level='L0',  # Auto-fallback for incompatible layers
)

# 4. Run calibration (no calibration data needed for dynamic quant)
calibrator = Calibrator(model, quant_config)
calibrator.run()

# 5. Save quantized model
model.save_pretrained(output_dir, safe_serialization=True)
```

### Task-Specific Handling

For **MoE models** (T2V-A14B, I2V-A14B), two sub-models must be quantized separately:

```python
if task in ['t2v-A14B', 'i2v-A14B']:
    # Quantize low_noise_model
    quantize_model(config.low_noise_checkpoint)
    # Quantize high_noise_model
    quantize_model(config.high_noise_checkpoint)
elif task == 'ti2v-5B':
    # Single model
    quantize_model(config.checkpoint)
```

## 7.2 Runtime Quantization Loading

In the pipeline `__init__`, load quantized weights via `mindiesd.quantize()`:

```python
from mindiesd import quantize
from .utils.utils import find_quant_config_file

if quant_dit_path:
    quant_dit_path = os.path.abspath(quant_dit_path)
    quant_model_path = os.path.join(quant_dit_path, config.low_noise_checkpoint)

    # Find quantization config file
    quant_desc_path, use_nz = find_quant_config_file(quant_model_path)

    if not os.path.exists(quant_desc_path):
        raise FileNotFoundError(f"Quantization config not found: {quant_desc_path}")

    # Apply quantization to model
    quantize(
        model=self.low_noise_model,
        quant_des_path=quant_desc_path,
        use_nz=use_nz,
    )
```

### Quantization Config File Discovery

```python
def find_quant_config_file(quant_config_path):
    """Search for quantization description file in priority order."""
    # Priority 1: W8A8 dynamic quantization (NZ format)
    quant_desc = os.path.join(quant_config_path, "quant_model_description_w8a8_dynamic.json")
    use_nz = True

    if not os.path.exists(quant_desc):
        # Priority 2: MXFP8 quantization (non-NZ format)
        quant_desc = os.path.join(quant_config_path, "quant_model_description_w8a8_mxfp8.json")
        use_nz = False

    return quant_desc, use_nz
```

**NZ format** is a Huawei-specific memory layout optimized for NPU computation.

## 7.3 FSDP + Float8 Compatibility

When using both FSDP and quantization, float8 buffers cause issues during FSDP buffer synchronization:

```python
def patch_cast_buffers_for_float8():
    """Monkey-patch FSDP to handle float8 buffers correctly."""
    import torch.distributed.fsdp._runtime_utils as runtime_utils
    original_fn = runtime_utils._cast_buffers_to_dtype_and_device

    def patched_fn(buffers, buffer_dtypes, device, *args, **kwargs):
        new_dtypes = []
        for buf, dtype in zip(buffers, buffer_dtypes):
            if buf.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                new_dtypes.append(buf.dtype)  # Keep original float8 dtype
            else:
                new_dtypes.append(dtype)
        return original_fn(buffers, new_dtypes, device, *args, **kwargs)

    runtime_utils._cast_buffers_to_dtype_and_device = patched_fn
```

**Must be called before model creation** when using FSDP + quantization together.

## 7.4 Quantized Attention (ALGO=3)

When quantization is active and ALGO=3, the attention computation uses a quantized FA path:

```python
def _attention_op(self, q, k, v, **kwargs):
    if ALGO == 3 and q.shape[1] == k.shape[1]:
        if hasattr(self, 'fa_quant') and self.fa_quant:
            return self.fa_quant(q, k, v)  # Quantized flash attention
        else:
            return torch_npu.npu_fused_infer_attention_score(q, k, v, ...)
```

The `fa_quant` attribute is injected by `mindiesd.quantize()` onto attention modules.

## 7.5 Model Loading with dtype

Ascend version adds `torch_dtype` parameter during model loading:

```python
# Original
model = WanModel.from_pretrained(ckpt_dir, subfolder=checkpoint_name)

# Ascend
model = WanModel.from_pretrained(ckpt_dir, subfolder=checkpoint_name,
                                  torch_dtype=self.param_dtype)
```

This avoids a separate `.to(dtype)` call after loading, which is especially important for quantized models where type conversion could break quantization parameters.

## Pitfalls

1. **`disable_level='L0'`**: This auto-disables quantization for layers where INT8 would cause significant accuracy loss. Do not change without validation.
2. **Float8 buffer patch must come first**: Call `patch_cast_buffers_for_float8()` before `shard_model()` in FSDP scenarios.
3. **Two separate quant configs for MoE**: Low-noise and high-noise models may have different quantization configurations.
4. **`use_nz` flag**: NZ format provides better NPU performance but requires NZ-compatible hardware. MXFP8 is the fallback.
