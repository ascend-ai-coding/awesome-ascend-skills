# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Bundled helper for golden vs PyPTO output comparison.
Import from kernel validation runners: see skills/pypto-op-verify/SKILL.md.
"""
import logging

import torch

logger = logging.getLogger(__name__)

# Maximum number of out-of-tolerance elements to display in the report.
MAX_OUTLIERS_DISPLAY = 20


def _summarize_outliers(diff, relative_diff, t1, t2, out_of_tolerance_mask):
    """Return per-outlier tensors (indices/values/diffs), sorted by largest
    absolute difference first."""
    outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
    outlier_diffs = diff[out_of_tolerance_mask]
    sorted_indices = torch.argsort(outlier_diffs, descending=True)
    return {
        "indices": tuple(ind[sorted_indices] for ind in outlier_indices),
        "values1": t1[out_of_tolerance_mask][sorted_indices],
        "values2": t2[out_of_tolerance_mask][sorted_indices],
        "diffs": outlier_diffs[sorted_indices],
        "relative_diffs": relative_diff[out_of_tolerance_mask][sorted_indices],
    }


def _log_outlier_rows(result):
    """Log the per-element table of out-of-tolerance values."""
    count = result["out_of_tolerance_count"]
    logger.info("Maximum deviation exceeding tolerance: %.6f", result["max_out_of_tolerance_diff"])
    logger.info("Average deviation exceeding tolerance: %.6f", result["mean_out_of_tolerance_diff"])
    logger.info(
        "\n🔍 Details of elements exceeding tolerance limits (Before Displaying%d):",
        min(MAX_OUTLIERS_DISPLAY, count),
    )
    logger.info("-" * 80)
    logger.info(
        "%-20s %-15s %-15s %-12s %-12s",
        "Index", "Tensor1 value", "Tensor2 value",
        "Absolute difference", "Relative difference",
    )
    logger.info("-" * 80)
    indices = result["outlier_indices"]
    for i in range(min(MAX_OUTLIERS_DISPLAY, count)):
        idx_str = str(tuple(indices[j][i].item() for j in range(len(indices))))
        logger.info(
            "%-20s %-15.6f %-15.6f %-12.6f %-12.6f",
            idx_str,
            result["outlier_values1"][i].item(),
            result["outlier_values2"][i].item(),
            result["outlier_diffs"][i].item(),
            result["outlier_relative_diffs"][i].item(),
        )
    if count > MAX_OUTLIERS_DISPLAY:
        logger.info(
            "... And also %d An element exceeding the tolerance is not displayed.",
            count - MAX_OUTLIERS_DISPLAY,
        )


def _log_report(result, tensor_name, rtol, atol):
    """Log the detailed comparison report for one tensor pair."""
    logger.info("\n%s", "=" * 60)
    logger.info("📊 Tensor Detailed Comparison Report")
    logger.info("name: %s", tensor_name)
    logger.info("=" * 60)
    logger.info("Total number of elements: %s", f"{result['total_elements']:,}")
    logger.info("Number of elements exceeding tolerance: %s", f"{result['out_of_tolerance_count']:,}")
    logger.info(
        "Out of Tolerance Ratio: %.6f (%.4f%%)",
        result["out_of_tolerance_ratio"], result["out_of_tolerance_ratio"] * 100,
    )
    logger.info("Maximum difference: %.6f", result["max_diff"])
    logger.info("Average difference: %.6f", result["mean_diff"])
    logger.info("Difference Standard Deviation: %.6f", result["std_diff"])
    logger.info("Tolerance Settings: rtol=%s, atol=%s", rtol, atol)
    if result["out_of_tolerance_count"] > 0:
        _log_outlier_rows(result)
    logger.info("\n✅ Tensor Matching: %s", result["all_close"])
    logger.info("=" * 60)


def detailed_tensor_compare(tensor1, tensor2, tensor_name, rtol=1e-3, atol=1e-3):
    """
    Detailed tensor comparison, analyzing the proportion of elements that are out of tolerance,
    and displaying specific information about those that exceed the tolerance.

    Args:
    tensor1: The first tensor.
    tensor2: The second tensor.
    tensor_name: Name used in the printed report.
    rtol: Relative tolerance.
    atol: Absolute tolerance.

    Returns:
    dict: A dictionary containing the comparison results.
    """
    # Ensure tensors are comparable
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # Calculate the difference
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)

    # Tolerance Check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # Statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # Difference Statistics
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()
        outliers = _summarize_outliers(diff, relative_diff, t1, t2, out_of_tolerance_mask)
    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        outliers = {"indices": None, "values1": None, "values2": None,
                    "diffs": None, "relative_diffs": None}

    result = {
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': outliers["indices"],
        'outlier_values1': outliers["values1"],
        'outlier_values2': outliers["values2"],
        'outlier_diffs': outliers["diffs"],
        'outlier_relative_diffs': outliers["relative_diffs"]
    }

    _log_report(result, tensor_name, rtol, atol)
    return result
