# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CI 守门：registries/*.yaml 与 schemas/op-spec.json 必须同步。

新增 enum 时漏改一边会让本测炸。仓库一次绿，后续每次 CI 都跑。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
SYNC_TOOL = SKILL_ROOT / "scripts" / "check_registry_schema_sync.py"


def test_registry_schema_in_sync():
    r = subprocess.run(
        [sys.executable, str(SYNC_TOOL)],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        "registry/schema enum 漂移，请改 schemas/op-spec.json 或 registries/*.yaml 让两边一致：\n"
        + r.stderr + r.stdout
    )
