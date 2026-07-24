# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""stage 2 字段单测：supported_chips / op.error_codes。

走子进程 validate_spec.py，验证 stage 2 校验逻辑：
  G1: dtype_chip_mismatch（fp8/fp4/hf8/int4 在不支持的芯片上）
  G2: unknown_chip
  J:  error_code_not_declared（machine_check.error_type 不在 op.error_codes 内）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = SKILL_ROOT / "scripts" / "validate_spec.py"


def _run(spec_yaml: str, tmp_path: Path) -> tuple[int, dict]:
    p = tmp_path / "spec.yaml"
    p.write_text(spec_yaml, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), str(p), "--json", "--stage", "2"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode, json.loads(r.stdout) if r.stdout else {}


# 最小可校验的 add example，便于通过 string.replace 制造负向 fixture
_BASE_ADD = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")


class TestSupportedChips:
    def test_baseline_passes(self, tmp_path):
        # add example 已声明 supported_chips=[Ascend910B]，本身 PASS
        rc, out = _run(_BASE_ADD, tmp_path)
        assert rc == 0
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "PASS"

    def test_unknown_chip(self, tmp_path):
        bad = _BASE_ADD.replace(
            "supported_chips: [Ascend910B]",
            "supported_chips: [atlas_unicorn, Ascend910B]",
        )
        rc, out = _run(bad, tmp_path)
        # stage 1 schema enum 也会报，但 stage 2 也独立报 unknown_chip
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        # schema 拦下 unicorn 时 stage 2 不一定再跑（依赖实现）；只断言整体不通过
        assert rc != 0

    def test_dtype_chip_mismatch(self, tmp_path):
        # 把 supported_chips 收窄到不支持 bfloat16 的 Ascend310，应该报 dtype_chip_mismatch
        bad = _BASE_ADD.replace(
            "supported_chips: [Ascend910B]",
            "supported_chips: [Ascend310]",
        )
        rc, out = _run(bad, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        assert any("dtype_chip_mismatch" in f["rule_id"] for f in s2["findings"])


class TestErrorCodes:
    def test_error_code_not_declared(self, tmp_path):
        # add example 的 op.error_codes 声明了 shape_mismatch；删掉它后，引用该类别的
        # boundary case 会触发 paradigm_constraint.error_code_not_declared
        bad = _BASE_ADD.replace(
            "    - shape_mismatch\n",
            "",
            1,
        )
        rc, out = _run(bad, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        assert any("error_code_not_declared" in f["rule_id"] for f in s2["findings"])
