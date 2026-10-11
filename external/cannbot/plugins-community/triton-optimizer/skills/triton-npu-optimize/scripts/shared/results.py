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

from typing import Literal

from shared.models import OptimizeCheckResult


def append_pass_issues_to_summary(summary: str, issues: tuple[str, ...]) -> str:
    if not issues:
        return summary
    return f"{summary} Notes: {'; '.join(issues)}"


def build_check_result(
    *,
    kind: Literal["baseline", "round"],
    status: Literal["pass", "fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
    next_option: str | None = None,
) -> OptimizeCheckResult:
    if summary is None:
        summary = (
            append_pass_issues_to_summary(f"{kind} check passed", issues)
            if status == "pass"
            else f"{kind} check requires fixes: {'; '.join(issues)}"
        )
    return OptimizeCheckResult(
        kind=kind,
        status=status,
        issues=issues,
        summary=summary,
        next_option=next_option,
    )
