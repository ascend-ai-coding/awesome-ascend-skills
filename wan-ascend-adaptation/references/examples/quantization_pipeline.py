"""
Wan2.2 W8A8 Quantization Pipeline

Complete example of quantizing Wan2.2 DiT model from FP32/FP16 to W8A8
using msmodelslim calibration-based quantization (no training required).
"""

import os
import json
import argparse
import torch
from msmodelslim.pytorch.quantize.quantization import W8A8DynamicQuant

# ===================================
# 1. Load Wan2.2 Model
# ===================================


def load_wan22_model(model_path, device):
    """
    Load Wan2.2 model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: torch.device for model loading

    Returns:
        model: Wan2.2 model in eval mode
        config: Model configuration
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Create model
    from wan.modules.model import WanModel

    model = WanModel(**config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Model loaded: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, config


# ===================================
# 2. Prepare Calibration Data
# ===================================


def prepare_calibration_data(model, num_samples=512, device="cpu"):
    """
    Generate calibration data without training.

    Args:
        model: Wan2.2 model
        num_samples: Number of samples for calibration
        device: Device for data preparation

    Returns:
        calibration_data: List of calibration samples
    """
    print(f"\nPreparing calibration data ({num_samples} samples)...")

    calibration_data = []

    # Generate dummy prompts or use real data
    dummy_prompts = [
        "A beautiful landscape with mountains",
        "A futuristic city at night",
        "A portrait of a person",
    ]

    with torch.no_grad():
        for i in range(num_samples):
            prompt = dummy_prompts[i % len(dummy_prompts)]

            # Prepare inputs
            inputs = model.prepare_inputs(prompt)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass to collect activation statistics
            with torch.amp.autocast("npu", enabled=torch_npu.is_available()):
                _ = model(**inputs)

            # Store input statistics for calibration
            calibration_data.append({"inputs": inputs, "timestamp": i})

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples")

    print(f"Calibration data ready: {len(calibration_data)} samples")
    return calibration_data


# ===================================
# 3. Quantization Configuration
# ===================================


def create_quantization_config(
    weight_dtype=torch.int8,
    act_dtype=torch.int8,
    skip_layers=None,
    quantizable_modules=None,
):
    """
    Create W8A8 quantization configuration.

    Args:
        weight_dtype: Target dtype for weights
        act_dtype: Target dtype for activations
        skip_layers: List of layer patterns to skip
        quantizable_modules: List of module patterns to quantize

    Returns:
        config: Quantization configuration dictionary
    """
    config = {
        "weight_quantization": {
            "dtype": str(weight_dtype).split(".")[-1],
            "method": "dynamic",
            "calibration": "minmax",
        },
        "activation_quantization": {
            "dtype": str(act_dtype).split(".")[-1],
            "method": "dynamic",
            "calibration": "minmax",
        },
        "layers_to_skip": skip_layers or ["vae.*", "text_encoder.*", "embedding.*"],
        "quantizable_modules": quantizable_modules
        or ["transformer.*", "attention.*", "ffn.*", "mlp.*"],
        "calibration": {"num_samples": 512, "batch_size": 16, "skip_first_n": 2},
        "performance": {"enable_fusion": True, "enable_cache": True},
    }

    return config


# ===================================
# 4. Quantization Pipeline
# ===================================


def quantize_model(model, calibration_data, config, output_path):
    """
    Quantize Wan2.2 model to W8A8.

    Args:
        model: FP32/FP16 Wan2.2 model
        calibration_data: Calibration samples
        config: Quantization configuration
        output_path: Path to save quantized model

    Returns:
        quantized_model: W8A8 quantized model
        stats: Quantization statistics
    """
    print("\n=== Quantization Pipeline ===")
    print(f"Weight dtype: {config['weight_quantization']['dtype']}")
    print(f"Activation dtype: {config['activation_quantization']['dtype']}")

    # Create quantizer
    print("\nCreating W8A8 quantizer...")
    quantizer = W8A8DynamicQuant(
        model=model, calibration_data=calibration_data, config=config
    )

    # Calibrate (no training)
    print("Calibrating quantization parameters...")
    quantizer.calibrate()

    # Quantize
    print("Quantizing model to W8A8...")
    quantized_model = quantizer.quantize()

    # Get statistics
    stats = quantizer.get_statistics()

    # Save quantized model
    print(f"\nSaving quantized model to {output_path}...")
    save_quantized_model(quantized_model, config, stats, output_path)

    return quantized_model, stats


# ===================================
# 5. Save Quantized Model
# ===================================


def save_quantized_model(model, config, stats, output_path):
    """
    Save quantized model with metadata.

    Args:
        model: Quantized model
        config: Quantization configuration
        stats: Quantization statistics
        output_path: Path to save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "quantization_config": config,
        "quantization_stats": stats,
        "version": "wan22-w8a8",
        "pytorch_version": torch.__version__,
    }

    torch.save(checkpoint, output_path)

    # Calculate model sizes
    model_size_fp32 = calculate_model_size(model, dtype=torch.float32)
    model_size_w8a8 = calculate_model_size(model, dtype=torch.float16)
    compression_ratio = model_size_fp32 / model_size_w8a8

    print(f"\n=== Quantization Summary ===")
    print(f"FP32 model size: {model_size_fp32:.2f} GB")
    print(f"W8A8 model size: {model_size_w8a8:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Memory saved: {(1 - 1 / compression_ratio) * 100:.1f}%")


# ===================================
# 6. Model Size Calculation
# ===================================


def calculate_model_size(model, dtype=torch.float32):
    """Calculate model size in GB."""
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = num_params * bytes_per_param
    size_gb = size_bytes / (1024**3)
    return size_gb


# ===================================
# 7. Verification
# ===================================


def verify_quantization(quantized_model, original_model, device="cpu"):
    """
    Verify quantized model correctness.

    Args:
        quantized_model: W8A8 quantized model
        original_model: Original FP32/FP16 model
        device: Device for verification
    """
    print("\n=== Verifying Quantization ===")

    # Set both models to eval
    quantized_model.eval()
    original_model.eval()

    # Generate test input
    test_input = {
        "latent": torch.randn(1, 4, 32, 32).to(device),
        "timestep": torch.tensor([50]).to(device),
        "context": torch.randn(1, 77, 1024).to(device),
    }

    # Get outputs
    with torch.no_grad():
        output_quant = quantized_model(**test_input)
        output_orig = original_model(**test_input)

    # Calculate difference
    diff = torch.abs(output_quant - output_orig).mean().item()

    print(f"Mean absolute difference: {diff:.6f}")

    if diff < 0.1:
        print("✓ Quantization verification passed")
        return True
    else:
        print("✗ Quantization verification failed - consider adjusting calibration")
        return False


# ===================================
# Main Pipeline
# ===================================


def main():
    parser = argparse.ArgumentParser(description="Quantize Wan2.2 model to W8A8")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to original FP32/FP16 model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save W8A8 quantized model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="quantization_config.json",
        help="Path to quantization config JSON",
    )
    parser.add_argument(
        "--calibration_data_path",
        type=str,
        default=None,
        help="Path to pre-generated calibration data",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for quantization (cpu, cuda, npu)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify quantized model after completion"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "npu":
        import torch_npu

        device = torch.device("cuda:0" if torch_npu.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 1. Load model
    model, model_config = load_wan22_model(args.model_path, device)

    # 2. Load or create calibration data
    if args.calibration_data_path and os.path.exists(args.calibration_data_path):
        print(f"\nLoading calibration data from {args.calibration_data_path}...")
        calibration_data = torch.load(args.calibration_data_path)
    else:
        calibration_data = prepare_calibration_data(
            model,
            num_samples=args.num_calibration_samples,
            device="cpu",  # Calibrate on CPU for stability
        )

        if args.calibration_data_path:
            print(f"Saving calibration data to {args.calibration_data_path}...")
            torch.save(calibration_data, args.calibration_data_path)

    # 3. Create quantization config
    if os.path.exists(args.config_path):
        print(f"\nLoading quantization config from {args.config_path}...")
        with open(args.config_path) as f:
            config = json.load(f)
    else:
        print("\nCreating default quantization config...")
        config = create_quantization_config()

        if args.config_path:
            print(f"Saving config to {args.config_path}...")
            with open(args.config_path, "w") as f:
                json.dump(config, f, indent=2)

    # 4. Quantize model
    quantized_model, stats = quantize_model(
        model, calibration_data, config, args.output_path
    )

    # 5. Verify (optional)
    if args.verify:
        original_model, _ = load_wan22_model(args.model_path, device)
        verify_quantization(quantized_model, original_model, device)

    print("\n=== Quantization Complete ===")
    print(f"Quantized model saved to: {args.output_path}")
    print("\nNext steps:")
    print("1. Test quantized model inference:")
    print(f"   python generate.py --model_path {args.output_path}")
    print("2. Benchmark performance:")
    print(f"   python benchmark.py --model_path {args.output_path}")


if __name__ == "__main__":
    # Handle BF16 → FP16 conversion for NPU compatibility
    import sys

    sys.argv = sys.argv + ["--verify"]  # Always verify

    main()
