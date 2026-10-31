# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Skill resources and default companion checkouts, independent of the caller's cwd."""

import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
REFERENCES_ROOT = SKILL_ROOT / "references"
EXAMPLES_ROOT = SKILL_ROOT / "examples"
PYPTO_ROOT = (
    Path(os.environ.get("PYPTO_SRC") or SKILL_ROOT.parent / "pypto")
    .expanduser()
    .resolve()
)
COVERAGE_DOC = REFERENCES_ROOT / "tooling/example-coverage.md"
CATALOG_DOC = REFERENCES_ROOT / "tooling/examples.md"
DOC_ROOTS = (SKILL_ROOT,)
