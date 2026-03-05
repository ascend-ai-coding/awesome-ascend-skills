#!/usr/bin/env python3
"""
Minimal Wan model migration from CUDA to Ascend NPU
Demonstrates basic torch_npu integration patterns
"""

import torch
import torch_npu

# Original CUDA code
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# NPU-adapted code
def get_npu_device(local_rank=0):
    """Get NPU device with compatibility check"""
    if torch.npu.is_available():
        # torch_npu allows using "cuda:" strings for compatibility
        device = torch.device(f"cuda:{local_rank}")
        torch.npu.set_device(local_rank)
        return device
    else:
        return torch.device("cpu")


def initialize_npu():
    """Initialize NPU with optimal settings"""
    import torch_npu

    # Disable JIT compilation for NPU compatibility
    torch.npu.set_compile_mode(jit_compile=False)

    # Configure NPU internal format
    torch.npu.config.allow_internal_format = False

    # Transfer tensors to NPU
    from torch_npu.contrib import transfer_to_npu

    return transfer_to_npu


# Example usage
if __name__ == "__main__":
    # Initialize NPU
    transfer_to_npu = initialize_npu()
    device = get_npu_device(local_rank=0)

    # Create model
    model = torch.nn.Linear(768, 768).to(device)

    # Transfer data
    x = torch.randn(1, 768)
    x = transfer_to_npu(x)  # Transfers to NPU

    # Forward pass
    output = model(x)
    print(f"Output device: {output.device}")
    print("✓ NPU migration successful!")
