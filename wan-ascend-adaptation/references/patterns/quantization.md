# W8A8 Quantization on Ascend NPU

## Overview

W8A8 quantization reduces model weight and activation precision from FP16/FP32 to 8-bit integers. This delivers:

- **50% memory reduction** - Weights stored as INT8 instead of FP16
- **Faster inference** - INT8 operations run faster on Ascend NPU cores
- **Lower power consumption** - Reduced data movement and compute
- **Minimal accuracy loss** - Dynamic calibration preserves model quality

This pattern uses `msmodelslim` (Huawei's model compression toolkit) for Wan2.1 video generation models.

## Prerequisites

```bash
# 1. CANN toolkit installed (8.0.RC3+)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. Install msmodelslim
pip install msmodelslim

# 3. Verify installation
python -c "import msmodelslim; print(msmodelslim.__version__)"
```

**Supported Models:**
- Wan2.1-T2V-A14B (text-to-video)
- Wan2.1-I2V-A14B (image-to-video)
- Wan2.1-TI2V-5B (text+image-to-video)

## Step-by-Step Process

### 1. Load Pretrained Model

```python
import torch
from wan import WanT2V, WanI2V

def load_model(model_type="t2v", device="npu"):
    """Load Wan2.1 model for quantization."""
    model_path = f"./Wan2.1-{model_type.upper()}-A14B"
    
    if model_type == "t2v":
        model = WanT2V.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif model_type == "i2v":
        model = WanI2V.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()
    return model
```

### 2. Configure QuantConfig

```python
from msmodelslim.pytorch.quant import QuantConfig

def create_quant_config(
    w_bit=8,
    a_bit=8,
    act_method=3,
    is_dynamic=True
):
    """
    Create quantization configuration.
    
    Args:
        w_bit: Weight bit-width (4, 8, or 16)
        a_bit: Activation bit-width (4, 8, or 16)
        act_method: Activation quantization method
            1: Min-Max quantization
            2: KL divergence
            3: Moving average (recommended for W8A8)
        is_dynamic: Use dynamic quantization scales
    """
    config = QuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        act_method=act_method,
        is_dynamic=is_dynamic,
        # Optional: Layer-specific settings
        excluded_layers=["embeddings", "norm", "head"]
    )
    return config
```

### 3. Run Calibration

```python
from msmodelslim.pytorch.quant import quantize_model

def calibrate_model(model, config, num_samples=32):
    """
    Calibrate quantized model without training data.
    
    Uses forward passes to collect activation statistics.
    """
    # Prepare dummy calibration data
    # For diffusion models, use random noise + text embeddings
    dummy_input = {
        "latents": torch.randn(1, 16, 16, 60, 104),  # B, C, F, H, W
        "timestep": torch.tensor([500]),
        "encoder_hidden_states": torch.randn(1, 512, 4096),
    }
    
    # Quantize and calibrate
    quantized_model = quantize_model(
        model=model,
        config=config,
        calib_data=[dummy_input] * num_samples,
        device="npu"
    )
    
    return quantized_model
```

### 4. Save Quantized Model

```python
def save_quantized_model(model, output_path):
    """Save quantized model with metadata."""
    import json
    
    # Save model weights
    torch.save(model.state_dict(), f"{output_path}/model.pt")
    
    # Save quantization config
    config = {
        "w_bit": 8,
        "a_bit": 8,
        "act_method": 3,
        "is_dynamic": True,
        "model_type": "wan2.1-t2v-a14b",
        "quantization": "w8a8_dynamic"
    }
    
    with open(f"{output_path}/quant_config.json", "w") as f:
        json.dump(config, f, indent=2)
```

## Complete Quantization Pipeline

```python
#!/usr/bin/env python3
"""
W8A8 Quantization Pipeline for Wan2.1 on Ascend NPU

Usage:
    python quant_wan22.py --model t2v --output ./quantized_model
"""

import argparse
import torch
from pathlib import Path
from wan import WanT2V, WanI2V
from msmodelslim.pytorch.quant import QuantConfig, quantize_model


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Wan2.1 models")
    parser.add_argument(
        "--model",
        choices=["t2v", "i2v", "ti2v"],
        default="t2v",
        help="Model type to quantize"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_model",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--w-bit",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Weight bit-width"
    )
    parser.add_argument(
        "--a-bit",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Activation bit-width"
    )
    parser.add_argument(
        "--act-method",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Activation quantization method"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization (default: dynamic)"
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=32,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "npu"],
        default="npu",
        help="Device for quantization"
    )
    return parser.parse_args()


def create_calibration_data(model_type, num_samples, device):
    """Generate dummy calibration data for diffusion models."""
    calib_data = []
    
    for _ in range(num_samples):
        if model_type == "i2v":
            # Image + text conditioning
            sample = {
                "latents": torch.randn(1, 16, 16, 60, 104).to(device),
                "timestep": torch.tensor([500]).to(device),
                "encoder_hidden_states": torch.randn(1, 512, 4096).to(device),
                "image_embeds": torch.randn(1, 1280).to(device),
            }
        else:
            # Text conditioning only
            sample = {
                "latents": torch.randn(1, 16, 16, 60, 104).to(device),
                "timestep": torch.tensor([500]).to(device),
                "encoder_hidden_states": torch.randn(1, 512, 4096).to(device),
            }
        calib_data.append(sample)
    
    return calib_data


def main():
    args = parse_args()
    
    print(f"Loading {args.model} model from {args.model_path}...")
    
    # Load model
    if args.model == "t2v":
        model = WanT2V.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device
        )
    elif args.model == "i2v":
        model = WanI2V.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Create quantization config
    config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        act_method=args.act_method,
        is_dynamic=not args.static,
        excluded_layers=[
            "embeddings",
            "norm",
            "head",
            "proj_out"  # Output projection
        ]
    )
    
    print(f"\nQuantization config:")
    print(f"  Weight bits: {args.w_bit}")
    print(f"  Activation bits: {args.a_bit}")
    print(f"  Activation method: {args.act_method}")
    print(f"  Dynamic: {not args.static}")
    
    # Generate calibration data
    print(f"\nGenerating {args.calib_samples} calibration samples...")
    calib_data = create_calibration_data(args.model, args.calib_samples, args.device)
    
    # Quantize model
    print("\nRunning quantization...")
    quantized_model = quantize_model(
        model=model,
        config=config,
        calib_data=calib_data,
        device=args.device
    )
    
    # Save quantized model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving quantized model to {output_dir}...")
    torch.save(quantized_model.state_dict(), output_dir / "model.pt")
    
    # Save config
    import json
    quant_config = {
        "model_type": args.model,
        "w_bit": args.w_bit,
        "a_bit": args.a_bit,
        "act_method": args.act_method,
        "is_dynamic": not args.static,
        "calib_samples": args.calib_samples,
        "device": args.device
    }
    
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    
    print("\nQuantization complete!")
    print(f"Output: {output_dir}")
    
    # Verify model size
    import os
    model_size = os.path.getsize(output_dir / "model.pt") / (1024**3)
    print(f"Model size: {model_size:.2f} GB")


if __name__ == "__main__":
    main()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_bit` | int | 8 | Weight bit-width: 4, 8, or 16 |
| `a_bit` | int | 8 | Activation bit-width: 4, 8, or 16 |
| `act_method` | int | 3 | Activation quantization method: 1 (min-max), 2 (KL), 3 (moving avg) |
| `is_dynamic` | bool | True | Dynamic quantization scales per batch |
| `excluded_layers` | list | [] | Layer names to exclude from quantization |

### Recommended Configurations

```python
# W8A8 Dynamic (default) - Best balance
config = QuantConfig(w_bit=8, a_bit=8, act_method=3, is_dynamic=True)

# W8A8 Static - Faster inference, slightly lower accuracy
config = QuantConfig(w_bit=8, a_bit=8, act_method=1, is_dynamic=False)

# W4A8 - Higher compression, more accuracy loss
config = QuantConfig(w_bit=4, a_bit=8, act_method=3, is_dynamic=True)

# FP16 weights, INT8 activations - Minimal accuracy loss
config = QuantConfig(w_bit=16, a_bit=8, act_method=3, is_dynamic=True)
```

## Accuracy vs Performance Trade-off

| Config | Model Size | Relative Speed | FVD Loss | Use Case |
|--------|-----------|----------------|----------|----------|
| FP16 (baseline) | 100% | 1.0x | 0% | Training, high-quality inference |
| W8A8 Dynamic | 50% | 1.3-1.5x | <1% | Production inference (recommended) |
| W8A8 Static | 50% | 1.4-1.6x | 1-2% | Latency-critical applications |
| W4A8 Dynamic | 25% | 1.5-1.8x | 3-5% | Edge deployment, memory-constrained |
| W4A16 | 25% | 1.2-1.3x | 2-3% | Large batch inference |

**Note:** FVD (Frechet Video Distance) measured on Wan2.1-T2V-A14B with 1000 test videos.

## Best Practices

### When to Quantize

**Quantize when:**
- Deploying to production with latency requirements
- Running on memory-constrained devices (edge NPU)
- Serving multiple models simultaneously
- Cost optimization is priority

**Avoid quantization when:**
- Maximum video quality is required
- Training/fine-tuning models
- Debug mode or model development

### Which Layers to Exclude

```python
# Always exclude these for diffusion models
excluded = [
    "embeddings",      # Vocab embeddings - sensitive to precision
    "norm",            # LayerNorm - affects training stability
    "head",            # Output head - final predictions
    "proj_out",        # Final projection layer
    "time_embed",      # Timestep embeddings
]

# Consider excluding for W4A8
excluded.extend([
    "attn.to_q",       # Query projections in attention
    "attn.to_k",       # Key projections in attention
])
```

### Calibration Tips

1. **Sample count:** 32-64 samples sufficient for W8A8, 128+ for W4A8
2. **Sample diversity:** Use varied prompts and noise levels
3. **Device consistency:** Calibrate on same device type as inference
4. **Batch size:** Use batch size 1 for calibration to match inference

```python
# Better calibration with diverse samples
def create_diverse_calib_data(num_samples):
    samples = []
    for i in range(num_samples):
        # Vary timestep (diffusion process)
        t = int(1000 * (i / num_samples))
        # Vary latent shape slightly
        f = 16 + (i % 4) * 4
        
        sample = {
            "latents": torch.randn(1, 16, f, 60, 104),
            "timestep": torch.tensor([t]),
            "encoder_hidden_states": torch.randn(1, 512, 4096),
        }
        samples.append(sample)
    return samples
```

### Verification

```python
def verify_quantization(original_model, quantized_model, test_input):
    """Compare outputs between FP16 and quantized models."""
    with torch.no_grad():
        orig_out = original_model(**test_input)
        quant_out = quantized_model(**test_input)
    
    # Compute MSE
    mse = torch.nn.functional.mse_loss(orig_out, quant_out)
    print(f"Quantization MSE: {mse.item():.6f}")
    
    # Relative error
    rel_error = mse / (orig_out.var() + 1e-8)
    print(f"Relative error: {rel_error.item():.4%}")
    
    return mse.item() < 0.01  # Threshold for acceptable quantization
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CANN not found` | CANN not installed | Run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `OOM during calibration` | Batch size too large | Reduce `calib_samples` or use `device="cpu"` |
| `Accuracy degradation >5%` | Excluded layers too few | Add attention layers to `excluded_layers` |
| `Slow quantization` | Running on CPU | Use `device="npu"` for calibration |
| `Model size not reduced` | Not saving quantized weights | Check `torch.save()` uses quantized model state |

## References

- [msmodelslim Documentation](https://www.hiascend.com/document/detail/en/canncommercial/80RC3/ptmoddevg/ptmsmodelslim/introduction/introduction_0001.html)
- [CANN Quantization Guide](https://www.hiascend.com/document/detail/en/canncommercial/80RC3/inferapplicationdev/atctool/atctool_0001.html)
- Wan2.1 Paper: "Wan: High-Quality Video Generation with Autoregressive Models"
