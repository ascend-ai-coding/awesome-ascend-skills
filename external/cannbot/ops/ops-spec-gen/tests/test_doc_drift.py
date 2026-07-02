# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""references/error-codes.md 必须与源码同步。

新增 rule_id / DslError code 后必须重跑：
    python3 scripts/dump_rule_ids.py --write
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
TOOL = SKILL_ROOT / "scripts" / "dump_rule_ids.py"


def test_error_codes_doc_in_sync():
    r = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        "references/error-codes.md 与源码 rule_id 漂移；跑：\n"
        "    python3 scripts/dump_rule_ids.py --write\n\n"
        + r.stderr + r.stdout
    )
