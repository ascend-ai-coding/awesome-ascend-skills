#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for avg_pool3d NPU operator.

This script tests the ascend_kernel avg_pool3d operator by comparing
its output with PyTorch's native avg_pool3d implementation.
"""

import torch
import torch_npu
import pytest
import numpy as np

# Load the ascend_kernel library
try:
    import ascend_kernel
except ImportError:
    # If not installed, try to load the library directly
    import os
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Find and load the .so file
    lib_pattern = os.path.join(project_root, "python/ascend_kernel/ascend_kernel/lib/*.so")
    lib_files = glob.glob(lib_pattern)
    if lib_files:
        torch.ops.load_library(lib_files[0])
    else:
        raise ImportError("Could not find ascend_kernel library")


def is_npu_available():
    """Check if NPU is available."""
    try:
        return torch.npu.is_available()
    except:
        return False


@pytest.fixture(scope="module")
def device():
    """Fixture to provide NPU device if available, otherwise skip tests."""
    if not is_npu_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


class TestAvgPool3d:
    """Test cases for avg_pool3d operator."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_basic_forward(self, device, batch_size, channels, dtype):
        """Test basic forward pass with default parameters."""
        # Input: (N, C, D, H, W)
        x = torch.randn(batch_size, channels, 8, 16, 16, dtype=dtype, device=device)
        kernel_size = (2, 2, 2)

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=[],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )

        # CPU reference result
        x_cpu = x.cpu()
        cpu_result = torch.nn.functional.avg_pool3d(
            x_cpu,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            ceil_mode=False,
            count_include_pad=True
        )

        # Compare
        npu_result_cpu = npu_result.cpu()
        assert npu_result_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_result_cpu.shape} vs CPU {cpu_result.shape}"

        if dtype == torch.float32:
            assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-5, atol=1e-5), \
                f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"
        else:  # float16
            assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-3, atol=1e-3), \
                f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"

    @pytest.mark.parametrize("kernel_size,padding", [
        ((2, 2, 2), (0, 0, 0)),
        ((3, 3, 3), (1, 1, 1)),
        ((2, 3, 4), (0, 1, 1)),
    ])
    def test_different_kernel_and_padding(self, device, kernel_size, padding):
        """Test with different kernel sizes and padding."""
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=list(kernel_size),
            stride=[],
            padding=list(padding),
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )

        # CPU reference result
        x_cpu = x.cpu()
        cpu_result = torch.nn.functional.avg_pool3d(
            x_cpu,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True
        )

        # Compare
        npu_result_cpu = npu_result.cpu()
        assert npu_result_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_result_cpu.shape} vs CPU {cpu_result.shape}"
        assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-4, atol=1e-4), \
            f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"

    @pytest.mark.parametrize("stride", [
        [1, 1, 1],
        [2, 2, 2],
        [1, 2, 3],
    ])
    def test_different_strides(self, device, stride):
        """Test with different stride values."""
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)
        kernel_size = [2, 2, 2]

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )

        # CPU reference result
        x_cpu = x.cpu()
        cpu_result = torch.nn.functional.avg_pool3d(
            x_cpu,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            ceil_mode=False,
            count_include_pad=True
        )

        # Compare
        npu_result_cpu = npu_result.cpu()
        assert npu_result_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_result_cpu.shape} vs CPU {cpu_result.shape}"
        assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-4, atol=1e-4), \
            f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"

    @pytest.mark.parametrize("ceil_mode", [False, True])
    def test_ceil_mode(self, device, ceil_mode):
        """Test ceil_mode parameter."""
        x = torch.randn(1, 3, 7, 15, 15, dtype=torch.float32, device=device)
        kernel_size = [2, 2, 2]
        stride = [2, 2, 2]

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=[0, 0, 0],
            ceil_mode=ceil_mode,
            count_include_pad=True,
            divisor_override=None
        )

        # CPU reference result
        x_cpu = x.cpu()
        cpu_result = torch.nn.functional.avg_pool3d(
            x_cpu,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=True
        )

        # Compare
        npu_result_cpu = npu_result.cpu()
        assert npu_result_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_result_cpu.shape} vs CPU {cpu_result.shape}"
        assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-4, atol=1e-4), \
            f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"

    @pytest.mark.parametrize("count_include_pad", [True, False])
    def test_count_include_pad(self, device, count_include_pad):
        """Test count_include_pad parameter."""
        x = torch.randn(1, 3, 8, 8, 8, dtype=torch.float32, device=device)
        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=count_include_pad,
            divisor_override=None
        )

        # CPU reference result
        x_cpu = x.cpu()
        cpu_result = torch.nn.functional.avg_pool3d(
            x_cpu,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=count_include_pad
        )

        # Compare
        npu_result_cpu = npu_result.cpu()
        assert npu_result_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_result_cpu.shape} vs CPU {cpu_result.shape}"
        assert torch.allclose(npu_result_cpu, cpu_result, rtol=1e-4, atol=1e-4), \
            f"Results differ: max diff = {(npu_result_cpu - cpu_result).abs().max()}"

    def test_output_shape_4d_input(self, device):
        """Test with 4D input (without channel dimension)."""
        # Note: avg_pool3d typically expects 5D input, but the implementation
        # supports 4D as well (treats as single channel)
        x = torch.randn(2, 8, 16, 16, dtype=torch.float32, device=device)
        kernel_size = [2, 2, 2]

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=[],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )

        # Check output shape
        expected_shape = (2, 4, 8, 8)  # (N, D//2, H//2, W//2)
        assert npu_result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {npu_result.shape}"

    def test_output_shape_5d_input(self, device):
        """Test with 5D input (standard NCDHW format)."""
        x = torch.randn(2, 3, 8, 16, 16, dtype=torch.float32, device=device)
        kernel_size = [2, 2, 2]

        # NPU result
        npu_result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=[],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )

        # Check output shape
        expected_shape = (2, 3, 4, 8, 8)  # (N, C, D//2, H//2, W//2)
        assert npu_result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {npu_result.shape}"


def run_simple_test():
    """Run a simple test without pytest framework.

    This test only checks if the operator can run successfully,
    without verifying numerical accuracy.
    """
    if not is_npu_available():
        print("NPU not available, skipping test")
        return False

    device = torch.device("npu:0")
    print(f"Running simple test on {device}...")

    try:
        # Create input tensor
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)
        print(f"Input shape: {x.shape}")

        # Run avg_pool3d
        result = torch.ops.npu.avg_pool3d(
            x,
            kernel_size=[2, 2, 2],
            stride=[],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None
        )
        print(f"Output shape: {result.shape}")

        print("Test PASSED! Operator runs successfully.")
        return True

    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run simple test if executed directly
    run_simple_test()
