# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression test: every example spec.yaml must pass stage 1 + stage 2.

Run:
    pytest tests/test_examples.py -v

These examples double as living documentation. If a schema/registry change breaks
them, either the change is wrong, or the examples need updating to match — never
suppress the failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = SKILL_DIR / "examples"
VALIDATOR = SKILL_DIR / "scripts" / "validate_spec.py"


def discover_examples() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*/spec.yaml"))


@pytest.mark.parametrize("spec_path", discover_examples(), ids=lambda p: p.parent.name)
def test_example_passes_validation(spec_path: Path) -> None:
    """Every example must validate cleanly (errors → fail; warnings tolerated)."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(spec_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"\n{spec_path.parent.name}/spec.yaml failed validation\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_at_least_three_examples() -> None:
    """Ensure we did not accidentally lose example files."""
    examples = discover_examples()
    assert len(examples) >= 4, f"Expected ≥4 examples, found {len(examples)}: {examples}"
