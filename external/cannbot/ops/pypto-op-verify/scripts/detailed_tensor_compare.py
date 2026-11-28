# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Bundled helper for golden vs PyPTO output comparison.
Import from kernel validation runners: see skills/pypto-op-verify/SKILL.md.
"""

import logging
from dataclasses import dataclass, replace

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorCompareOptions:
    """Optional controls for :func:`detailed_tensor_compare`."""

    rtol: float = 1e-3
    atol: float = 1e-3
    verbose: bool = True
    max_outliers_display: int = 20


_LEGACY_OPTION_NAMES = ("rtol", "atol", "verbose", "max_outliers_display")


def _collect_tensor_leaf_pairs(actual, expected, path, pairs):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        pairs.append((path, actual, expected))
        return
    if type(actual) is not type(expected):
        raise AssertionError(
            f"output structure mismatch at {path}: "
            f"actual={type(actual).__name__}, expected={type(expected).__name__}"
        )
    if isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(
                f"output count mismatch at {path}: "
                f"actual={len(actual)}, expected={len(expected)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _collect_tensor_leaf_pairs(
                actual_item, expected_item, f"{path}[{index}]", pairs
            )
        return
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"output keys mismatch at {path}: "
                f"actual={list(actual)}, expected={list(expected)}"
            )
        for key in expected:
            _collect_tensor_leaf_pairs(
                actual[key], expected[key], f"{path}[{key!r}]", pairs
            )
        return
    raise TypeError(
        f"output leaf at {path} must be torch.Tensor, got {type(actual).__name__}"
    )


def tensor_leaf_pairs(actual, expected, path="output"):
    """Pair tensor leaves while validating tuple/list/dict output structure."""
    pairs = []
    _collect_tensor_leaf_pairs(actual, expected, path, pairs)
    if not pairs:
        raise ValueError(f"output structure at {path} contains no tensor leaves")
    return pairs


def _resolve_options(options, legacy_values, legacy_options):
    if len(legacy_values) > len(_LEGACY_OPTION_NAMES):
        raise TypeError("too many positional arguments")
    merged_options = dict(legacy_options)
    for name, value in zip(_LEGACY_OPTION_NAMES, legacy_values):
        if name in merged_options:
            raise TypeError(f"multiple values for argument {name!r}")
        merged_options[name] = value
    if options is None:
        options = TensorCompareOptions()
    if merged_options:
        options = replace(options, **merged_options)
    return options


def _sorted_outliers(tensor1, tensor2, diff, relative_diff, mask):
    if not mask.any().item():
        return {
            "outlier_indices": None,
            "outlier_values1": None,
            "outlier_values2": None,
            "outlier_diffs": None,
            "outlier_relative_diffs": None,
        }

    outlier_diffs = diff[mask]
    order = torch.argsort(outlier_diffs, descending=True)
    indices = torch.nonzero(mask, as_tuple=True)
    return {
        "outlier_indices": tuple(index[order] for index in indices),
        "outlier_values1": tensor1[mask][order],
        "outlier_values2": tensor2[mask][order],
        "outlier_diffs": outlier_diffs[order],
        "outlier_relative_diffs": relative_diff[mask][order],
    }


def _non_elementwise_result(tensor1, tensor2, *, shape_match, all_close):
    total = max(tensor1.numel(), tensor2.numel())
    mismatch_count = 0 if all_close else total
    mismatch_value = 0.0 if all_close else float("inf")
    return {
        "total_elements": total,
        "out_of_tolerance_count": mismatch_count,
        "out_of_tolerance_ratio": 0.0 if total == 0 else mismatch_count / total,
        "max_diff": mismatch_value,
        "mean_diff": mismatch_value,
        "std_diff": 0.0,
        "max_out_of_tolerance_diff": mismatch_value,
        "mean_out_of_tolerance_diff": mismatch_value,
        "all_close": all_close,
        "shape_match": shape_match,
        "tensor1_shape": tuple(tensor1.shape),
        "tensor2_shape": tuple(tensor2.shape),
        "tolerance_mask": None,
        "diff_tensor": None,
        "outlier_indices": None,
        "outlier_values1": None,
        "outlier_values2": None,
        "outlier_diffs": None,
        "outlier_relative_diffs": None,
    }


def _build_result(tensor1, tensor2, options):
    if tensor1.shape != tensor2.shape:
        return _non_elementwise_result(
            tensor1, tensor2, shape_match=False, all_close=False
        )
    if tensor1.numel() == 0:
        return _non_elementwise_result(
            tensor1, tensor2, shape_match=True, all_close=True
        )
    finite_pair = torch.isfinite(tensor1) & torch.isfinite(tensor2)
    same_inf = (
        torch.isinf(tensor1)
        & torch.isinf(tensor2)
        & (torch.signbit(tensor1) == torch.signbit(tensor2))
    )
    invalid_mask = ~(finite_pair | same_inf)
    safe_tensor1 = torch.where(finite_pair, tensor1, 0.0)
    safe_tensor2 = torch.where(finite_pair, tensor2, 0.0)
    finite_diff = torch.abs(safe_tensor1 - safe_tensor2)
    diff = torch.where(invalid_mask, float("inf"), finite_diff)
    finite_relative = finite_diff / (torch.abs(safe_tensor2) + 1e-8)
    relative_diff = torch.where(invalid_mask, float("inf"), finite_relative)
    tolerance_mask = same_inf | (
        finite_pair
        & (finite_diff <= options.atol + options.rtol * torch.abs(safe_tensor2))
    )
    outlier_mask = ~tolerance_mask
    outlier_count = outlier_mask.sum().item()
    outlier_diff = diff[outlier_mask]
    result = {
        "total_elements": tensor1.numel(),
        "out_of_tolerance_count": outlier_count,
        "out_of_tolerance_ratio": outlier_count / tensor1.numel(),
        "max_diff": torch.max(diff).item(),
        "mean_diff": torch.mean(diff).item(),
        "std_diff": torch.std(diff, unbiased=False).item(),
        "max_out_of_tolerance_diff": torch.max(outlier_diff).item()
        if outlier_count
        else 0.0,
        "mean_out_of_tolerance_diff": torch.mean(outlier_diff).item()
        if outlier_count
        else 0.0,
        "all_close": outlier_count == 0,
        "shape_match": True,
        "tensor1_shape": tuple(tensor1.shape),
        "tensor2_shape": tuple(tensor2.shape),
        "tolerance_mask": tolerance_mask,
        "diff_tensor": diff,
    }
    result.update(_sorted_outliers(tensor1, tensor2, diff, relative_diff, outlier_mask))
    return result


def _log_outliers(result, max_display):
    count = result["out_of_tolerance_count"]
    logger.info(
        "Maximum deviation exceeding tolerance: %.6f",
        result["max_out_of_tolerance_diff"],
    )
    logger.info(
        "Average deviation exceeding tolerance: %.6f",
        result["mean_out_of_tolerance_diff"],
    )
    logger.info(
        "\n🔍 Details of elements exceeding tolerance limits (showing first %d of %d):",
        min(max_display, count),
        count,
    )
    logger.info("-" * 80)
    logger.info(
        "%-20s %-15s %-15s %-12s %-12s",
        "Index",
        "Tensor1 value",
        "Tensor2 value",
        "Absolute difference",
        "Relative difference",
    )
    logger.info("-" * 80)
    indices = result["outlier_indices"]
    for index in range(min(max_display, count)):
        index_text = str(tuple(axis[index].item() for axis in indices))
        logger.info(
            "%-20s %-15.6f %-15.6f %-12.6f %-12.6f",
            index_text,
            result["outlier_values1"][index].item(),
            result["outlier_values2"][index].item(),
            result["outlier_diffs"][index].item(),
            result["outlier_relative_diffs"][index].item(),
        )
    if count > max_display:
        logger.info(
            "... And %d more elements exceeding tolerance are not displayed.",
            count - max_display,
        )


def _log_result(tensor_name, result, options):
    logger.info("\n%s", "=" * 60)
    logger.info("📊 Tensor Detailed Comparison Report")
    logger.info("name: %s", tensor_name)
    logger.info("=" * 60)
    if not result["shape_match"]:
        logger.info(
            "Shape mismatch: tensor1=%s, tensor2=%s",
            result["tensor1_shape"],
            result["tensor2_shape"],
        )
        logger.info("\n✅ Tensor Matching: False")
        logger.info("=" * 60)
        return
    logger.info("Total number of elements: %s", f"{result['total_elements']:,}")
    logger.info(
        "Number of elements exceeding tolerance: %s",
        f"{result['out_of_tolerance_count']:,}",
    )
    logger.info(
        "Out of Tolerance Ratio: %.6f (%.4f%%)",
        result["out_of_tolerance_ratio"],
        result["out_of_tolerance_ratio"] * 100,
    )
    logger.info("Maximum difference: %.6f", result["max_diff"])
    logger.info("Average difference: %.6f", result["mean_diff"])
    logger.info("Difference Standard Deviation: %.6f", result["std_diff"])
    logger.info("Tolerance Settings: rtol=%s, atol=%s", options.rtol, options.atol)
    if result["out_of_tolerance_count"]:
        _log_outliers(result, options.max_outliers_display)
    logger.info("\n✅ Tensor Matching: %s", result["all_close"])
    logger.info("=" * 60)


def detailed_tensor_compare(
    tensor1, tensor2, tensor_name=None, *legacy_values, options=None, **legacy_options
):
    """Compare tensors and return detailed tolerance statistics.

    ``options`` is the preferred API. Existing callers may continue to pass
    ``rtol``, ``atol``, ``verbose`` and ``max_outliers_display`` positionally
    or by keyword.
    """
    name_alias = legacy_options.pop("name", None)
    if tensor_name is not None and name_alias is not None:
        raise TypeError("multiple values for tensor name")
    tensor_name = tensor_name or name_alias or "tensor"
    resolved_options = _resolve_options(options, legacy_values, legacy_options)
    normalized1 = tensor1.cpu().float()
    normalized2 = tensor2.cpu().float()
    result = _build_result(normalized1, normalized2, resolved_options)
    if resolved_options.verbose:
        _log_result(tensor_name, result, resolved_options)
    return result
