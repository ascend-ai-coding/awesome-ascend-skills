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

import json
from pathlib import Path


_SKILL_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = _SKILL_ROOT / "references" / "triton-npu-optimize-state" / "round-contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
ROUND_STATE_REQUIRED_FIELDS = tuple(CONTRACT_DATA["round_state_required_fields"])
ROUND_STATE_OPTIONAL_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["round_state_optional_fields"].items()
}
