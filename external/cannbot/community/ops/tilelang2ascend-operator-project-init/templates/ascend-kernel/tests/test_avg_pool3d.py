#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""
Unit tests for avg_pool3d NPU operator.

This script tests the ascend_kernel avg_pool3d operator by comparing
its output with PyTorch's native avg_pool3d implementation.
"""

from dataclasses import dataclass

import logging
import torch
import torch_npu
import pytest
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K222 = [2, 2, 2]  # default 3D kernel size


@dataclass
class PoolConfig:
    kernel_size = None
    stride = None
    padding = None
    ceil_mode: bool = False
    count_include_pad: bool = True
    divisor_override = None


def _npu_avg_pool3d(x, pool_cfg):
    """Call torch.ops.npu.avg_pool3d with sensible defaults."""
    return torch.ops.npu.avg_pool3d(
        x,
        kernel_size=K222 if pool_cfg.kernel_size is None else pool_cfg.kernel_size,
        stride=[] if pool_cfg.stride is None else pool_cfg.stride,
        padding=[0, 0, 0] if pool_cfg.padding is None else pool_cfg.padding,
        ceil_mode=pool_cfg.ceil_mode,
        count_include_pad=pool_cfg.count_include_pad,
        divisor_override=pool_cfg.divisor_override,
    )

# Load the ascend_kernel library
try:
    import ascend_kernel
except ImportError as e:
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
        raise ImportError("Could not find ascend_kernel library") from e


def is_npu_available():
    """Check if NPU is available."""
    try:
        return torch.npu.is_available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def device():
    """Fixture to provide NPU device if available, otherwise skip tests."""
    if not is_npu_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


class TestAvgPool3d:
    """Test cases for avg_pool3d operator."""

    @staticmethod
    def test_output_shape_4d_input(device):
        """Test with 4D input (without channel dimension)."""
        # Note: avg_pool3d typically expects 5D input, but the implementation
        # supports 4D as well (treats as single channel)
        x = torch.randn(2, 8, 16, 16, dtype=torch.float32, device=device)

        npu_result = _npu_avg_pool3d(x, PoolConfig())

        # Check output shape
        expected_shape = (2, 4, 8, 8)  # (N, D//2, H//2, W//2)
        assert npu_result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {npu_result.shape}"

    @staticmethod
    def test_output_shape_5d_input(device):
        """Test with 5D input (standard NCDHW format)."""
        x = torch.randn(2, 3, 8, 16, 16, dtype=torch.float32, device=device)

        npu_result = _npu_avg_pool3d(x, PoolConfig())

        # Check output shape
        expected_shape = (2, 3, 4, 8, 8)  # (N, C, D//2, H//2, W//2)
        assert npu_result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {npu_result.shape}"

    @staticmethod
    def _cpu_ref_avg_pool3d(x, pool_cfg):
        """Compute CPU reference result using torch.nn.functional.avg_pool3d."""
        return torch.nn.functional.avg_pool3d(
            x.cpu(),
            kernel_size=pool_cfg.kernel_size,
            stride=pool_cfg.stride if pool_cfg.stride is not None else pool_cfg.kernel_size,
            padding=pool_cfg.padding,
            ceil_mode=pool_cfg.ceil_mode,
            count_include_pad=pool_cfg.count_include_pad,
        )

    @staticmethod
    def _assert_close(npu_result, cpu_result, rtol=1e-4, atol=1e-4):
        """Compare NPU and CPU results with shape and value checks."""
        npu_cpu = npu_result.cpu()
        assert npu_cpu.shape == cpu_result.shape, \
            f"Shape mismatch: NPU {npu_cpu.shape} vs CPU {cpu_result.shape}"
        assert torch.allclose(npu_cpu, cpu_result, rtol=rtol, atol=atol), \
            f"Results differ: max diff = {(npu_cpu - cpu_result).abs().max()}"

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_basic_forward(self, device, batch_size, channels, dtype):
        """Test basic forward pass with default parameters.

        Input shape: (N, C, D, H, W).
        """
        x = torch.randn(batch_size, channels, 8, 16, 16, dtype=dtype, device=device)

        npu_result = _npu_avg_pool3d(x, PoolConfig())
        cpu_result = TestAvgPool3d._cpu_ref_avg_pool3d(x, PoolConfig(kernel_size=K222))
        if dtype == torch.float32:
            TestAvgPool3d._assert_close(npu_result, cpu_result, rtol=1e-5, atol=1e-5)
        else:
            TestAvgPool3d._assert_close(npu_result, cpu_result, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("kernel_size,padding", [
        (K222, (0, 0, 0)),
        ((3, 3, 3), (1, 1, 1)),
        ((2, 3, 4), (0, 1, 1)),
    ])
    def test_different_kernel_and_padding(self, device, kernel_size, padding):
        """Test with different kernel sizes and padding."""
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)

        npu_result = _npu_avg_pool3d(
            x, PoolConfig(kernel_size=list(kernel_size), padding=list(padding)))
        cpu_result = TestAvgPool3d._cpu_ref_avg_pool3d(
            x, PoolConfig(kernel_size=kernel_size, padding=padding))
        TestAvgPool3d._assert_close(npu_result, cpu_result)

    @pytest.mark.parametrize("stride", [
        [1, 1, 1],
        [2, 2, 2],
        [1, 2, 3],
    ])
    def test_different_strides(self, device, stride):
        """Test with different stride values."""
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)

        npu_result = _npu_avg_pool3d(x, PoolConfig(stride=stride))
        cpu_result = TestAvgPool3d._cpu_ref_avg_pool3d(
            x, PoolConfig(kernel_size=K222, stride=stride))
        TestAvgPool3d._assert_close(npu_result, cpu_result)

    @pytest.mark.parametrize("ceil_mode", [False, True])
    def test_ceil_mode(self, device, ceil_mode):
        """Test ceil_mode parameter."""
        x = torch.randn(1, 3, 7, 15, 15, dtype=torch.float32, device=device)
        stride = [2, 2, 2]

        npu_result = _npu_avg_pool3d(x, PoolConfig(stride=stride, ceil_mode=ceil_mode))
        cpu_result = TestAvgPool3d._cpu_ref_avg_pool3d(
            x, PoolConfig(kernel_size=K222, stride=stride, ceil_mode=ceil_mode))
        TestAvgPool3d._assert_close(npu_result, cpu_result)

    @pytest.mark.parametrize("count_include_pad", [True, False])
    def test_count_include_pad(self, device, count_include_pad):
        """Test count_include_pad parameter."""
        x = torch.randn(1, 3, 8, 8, 8, dtype=torch.float32, device=device)
        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]

        npu_result = _npu_avg_pool3d(
            x, PoolConfig(kernel_size=kernel_size, stride=stride,
                          padding=padding, count_include_pad=count_include_pad))
        cpu_result = TestAvgPool3d._cpu_ref_avg_pool3d(
            x, PoolConfig(kernel_size=kernel_size, stride=stride, padding=padding,
                          count_include_pad=count_include_pad))
        TestAvgPool3d._assert_close(npu_result, cpu_result)


def run_simple_test():
    """Run a simple test without pytest framework.

    This test only checks if the operator can run successfully,
    without verifying numerical accuracy.
    """
    if not is_npu_available():
        logger.warning("NPU not available, skipping test")
        return False

    device = torch.device("npu:0")
    logger.info("Running simple test on %s...", device)

    try:
        # Create input tensor
        x = torch.randn(1, 3, 8, 16, 16, dtype=torch.float32, device=device)
        logger.info("Input shape: %s", x.shape)

        # Run avg_pool3d
        result = _npu_avg_pool3d(x, PoolConfig())
        logger.info("Output shape: %s", result.shape)

        logger.info("Test PASSED! Operator runs successfully.")
        return True

    except Exception as exc:
        logger.error("Test FAILED with error: %s", exc)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run simple test if executed directly
    run_simple_test()
