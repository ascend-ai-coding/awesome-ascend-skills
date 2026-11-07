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

from typing import TypedDict


class ResultPayload(TypedDict):
    return_code: int
    stdout: str
    stderr: str
    stalled: bool
    session_id: str | None


def make_result(
    *,
    return_code: int,
    stdout: str,
    stderr: str,
    stalled: bool = False,
    session_id: str | None = None,
) -> ResultPayload:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stalled": stalled,
        "session_id": session_id,
    }
