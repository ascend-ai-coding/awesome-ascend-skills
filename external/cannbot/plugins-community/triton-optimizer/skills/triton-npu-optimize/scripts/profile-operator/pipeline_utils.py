# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

from models import KernelInvocation, PipelineStage
from parser_base import parse_optional_float


PIPELINE_FIELDS = (
    "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
    "aic_mte3_ratio", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)


def has_pipeline_data(fieldnames: Sequence[str], row: dict[str, str]) -> bool:
    return any(column in fieldnames and row.get(column, "").strip() for column in PIPELINE_FIELDS)


def build_pipeline_stage(fieldnames: Sequence[str], row: dict[str, str]) -> PipelineStage | None:
    if not has_pipeline_data(fieldnames, row):
        return None
    values = {
        field: parse_optional_float(row.get(field)) or 0.0
        for field in PIPELINE_FIELDS
    }
    values["cube_utilization"] = parse_optional_float(
        row.get("cube_utilization(%)", row.get("cube_utilization"))
    ) or 0.0
    values["block_dim"] = int(parse_optional_float(row.get("Block Dim", "0")) or 0)
    return PipelineStage(**values)


def build_kernel_invocation(
    op_name: str,
    duration: float | None,
    wait_time: float | None,
    block_dim: int,
    pipeline: PipelineStage | None,
) -> KernelInvocation:
    return KernelInvocation(
        op_name=op_name,
        duration_us=duration or 0.0,
        wait_time_us=wait_time or 0.0,
        block_dim=block_dim,
        pipeline=pipeline,
    )
