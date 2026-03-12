# Quantization Guide for Ascend Diffusion Models

How to quantize diffusion models (DiT, UNet, VAE) for Ascend NPU deployment using mindiesd.

## Table of Contents
1. [Overview](#1-overview)
2. [Supported Schemes](#2-supported-schemes)
3. [Basic Quantization](#3-basic-quantization)
4. [Quant Config File Format](#4-quant-config-file-format)
5. [Config File Discovery](#5-config-file-discovery)
6. [FSDP Integration](#6-fsdp-integration)
7. [Quantized Attention](#7-quantized-attention)
8. [CLI Integration](#8-cli-integration)

---

## 1. Overview

mindiesd provides post-training quantization (PTQ) for diffusion models. The quantization flow:

1. Prepare a quant description JSON (specifies per-layer quantization parameters)
2. Discover the config file via `find_quant_config_file()` — returns `(path, use_nz)` tuple
3. Call `quantize(model=model, quant_des_path=path, use_nz=use_nz)` to apply quantization
4. Run inference as normal — quantized ops are transparent

Quantization reduces memory usage and increases throughput, with minimal quality loss for well-calibrated configs.

---

## 2. Supported Schemes

| Scheme | Description | Memory Savings | Quality Impact |
|--------|-------------|---------------|----------------|
| W8A8 (dynamic) | 8-bit weights, 8-bit activations | ~2x | Minimal |
| W8A8 (MXFP8) | Microscaling FP8 format | ~2x | Minimal |
| W4A8 | 4-bit weights, 8-bit activations | ~3x | Low-moderate |
| W4A4 (MXFP4) | 4-bit weights, 4-bit activations, dual-scale | ~4x | Moderate |

Start with W8A8 — it's the safest. Move to W4A8 only if memory is the bottleneck and you've verified quality.

---

## 3. Basic Quantization

```python
import torch
from mindiesd import quantize

# Load model normally
model = load_your_diffusion_model(checkpoint_path)
model = model.to("npu")

# Discover quant config (returns tuple)
quant_des_path, use_nz = find_quant_config_file(quant_dir)

# Apply quantization
# NOTE: parameter is quant_des_path (description path), NOT quant_config_path
# use_nz controls NZ compression format (True for w8a8_dynamic, w4a4; False for mxfp8)
quantize(model=model, quant_des_path=quant_des_path, use_nz=use_nz)

# Inference as normal — quantized ops are transparent
output = model(input_tensor)
```

The `quantize()` call:
- Reads the description file to determine which layers to quantize and how
- Replaces linear layers with quantized equivalents
- Handles calibration internally if needed
- Modifies the model in-place

---

## 4. Quant Config File Format

The quant config JSON specifies per-layer quantization parameters:

### W8A8 Dynamic Example
```json
{
  "quant_type": "w8a8_dynamic",
  "model_type": "dit",
  "layers": {
    "transformer_blocks.*.attn.to_q": {"weight_bits": 8, "act_bits": 8},
    "transformer_blocks.*.attn.to_k": {"weight_bits": 8, "act_bits": 8},
    "transformer_blocks.*.attn.to_v": {"weight_bits": 8, "act_bits": 8},
    "transformer_blocks.*.attn.to_out": {"weight_bits": 8, "act_bits": 8},
    "transformer_blocks.*.ffn.fc1": {"weight_bits": 8, "act_bits": 8},
    "transformer_blocks.*.ffn.fc2": {"weight_bits": 8, "act_bits": 8}
  },
  "exclude_layers": [
    "final_layer",
    "time_embed"
  ]
}
```

### Key Config Fields
- `quant_type`: One of `w8a8_dynamic`, `w8a8_mxfp8`, `w4a8`, `w4a4_mxfp4_dualscale`
- `model_type`: Model architecture hint (`dit`, `unet`, `vae`)
- `layers`: Glob patterns mapping layer names to quantization config
- `exclude_layers`: Layers to keep in full precision (typically embedding, final projection, time embedding)

### Sensitive Layers

Some layers are sensitive to quantization and should be excluded:
- **Time embedding** — Precision here affects all denoising steps
- **Final projection** — Direct impact on output quality
- **First/last transformer block** — Often more sensitive than middle blocks
- **Cross-attention QKV** — Text conditioning is precision-sensitive

---

## 5. Config File Discovery

The Ascend adaptation uses a utility to find quant configs. **Returns a `(path, use_nz)` tuple**, not just a path.

```python
import os

def find_quant_config_file(quant_dit_path):
    """
    Search for quantization description files in the given directory.
    
    Returns:
        tuple: (config_path: str, use_nz: bool)
            - config_path: Full path to the quant description JSON
            - use_nz: Whether to use NZ compression format
    
    Priority order (with corresponding use_nz values):
    1. quant_model_description_w8a8_dynamic.json  → use_nz=True
    2. quant_model_description_w8a8_mxfp8.json    → use_nz=False
    3. quant_model_description_w4a4_mxfp4_dualscale.json → use_nz=True
    """
    candidates = [
        ("quant_model_description_w8a8_dynamic.json", True),
        ("quant_model_description_w8a8_mxfp8.json", False),
        ("quant_model_description_w4a4_mxfp4_dualscale.json", True),
    ]
    
    for name, use_nz in candidates:
        path = os.path.join(quant_dit_path, name)
        if os.path.exists(path):
            return path, use_nz
    
    raise FileNotFoundError(
        f"No quant config found in {quant_dit_path}. "
        f"Expected one of: {[c[0] for c in candidates]}"
    )
```

**IMPORTANT**: The file naming convention is `quant_model_description_<scheme>.json`. The `use_nz` flag varies by scheme — W8A8 dynamic and W4A4 use NZ format, while MXFP8 does not.

---

## 6. FSDP Integration

When combining quantization with FSDP, float8 buffer types need special handling:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from mindiesd import quantize

def setup_quantized_fsdp_model(model, quant_des_path, use_nz, device_id):
    """
    Quantize first, then wrap with FSDP.
    Order matters: quantize → FSDP, not the reverse.
    """
    # Step 1: Quantize (note: quant_des_path and use_nz, not quant_config_path)
    quantize(model=model, quant_des_path=quant_des_path, use_nz=use_nz)
    
    # Step 2: Patch FSDP for float8 compatibility
    patch_cast_buffers_for_float8(model)
    
    # Step 3: Wrap with FSDP
    model = FSDP(
        model,
        device_id=device_id,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
    )
    
    return model


def patch_cast_buffers_for_float8(model):
    """
    FSDP's internal buffer casting doesn't handle float8 dtypes.
    This patches _cast_buffers to skip float8 buffers.
    """
    float8_types = {torch.float8_e4m3fn, torch.float8_e5m2}
    
    original_cast = FSDP._cast_buffers
    
    def safe_cast(self, *args, **kwargs):
        saved = {}
        for name, buf in self.named_buffers():
            if buf is not None and buf.dtype in float8_types:
                saved[name] = buf.clone()
        
        original_cast(self, *args, **kwargs)
        
        for name, buf in saved.items():
            parts = name.split('.')
            mod = self
            for p in parts[:-1]:
                mod = getattr(mod, p)
            setattr(mod, parts[-1], buf)
    
    FSDP._cast_buffers = safe_cast
```

---

## 7. Quantized Attention

mindiesd also supports quantized flash attention:

```python
# When model is quantized, attention can use quantized kernels
class QuantizedAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
        self.fa_quant = None  # Set by mindiesd.quantize() if applicable
    
    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim)
        
        if self.fa_quant is not None:
            # Use quantized attention kernel
            output = self.fa_quant(q, k, v)
        else:
            # Standard attention
            output = attention_op(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            output = output.transpose(1, 2)
        
        output = output.reshape(*output.shape[:-2], -1)
        return self.to_out(output)
```

---

## 8. CLI Integration

Typical CLI pattern for enabling quantization:

```python
import argparse
from mindiesd import quantize

parser = argparse.ArgumentParser()
parser.add_argument("--quant_dit_path", type=str, default=None,
                    help="Path to directory containing quant description JSON")

args = parser.parse_args()

# Load model
model = load_model(checkpoint)

# Optionally quantize
if args.quant_dit_path:
    quant_des_path, use_nz = find_quant_config_file(args.quant_dit_path)
    quantize(model=model, quant_des_path=quant_des_path, use_nz=use_nz)
    print(f"Model quantized with config: {quant_des_path} (use_nz={use_nz})")

# Continue with FSDP, inference, etc.
```

### Running Quantized Inference
```bash
# W8A8 quantized inference
torchrun --nproc_per_node=8 generate.py \
    --quant_dit_path /path/to/quant_configs/ \
    --cfg_size 2 \
    --ulysses_size 4 \
    --prompt "..."
```
