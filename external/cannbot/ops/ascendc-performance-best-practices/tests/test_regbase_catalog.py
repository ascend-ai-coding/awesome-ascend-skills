# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REGBASE_ROOT = SKILL_ROOT / "reference" / "common" / "arch35" / "regbase"
CATALOG = json.loads((REGBASE_ROOT / "catalog.json").read_text(encoding="utf-8"))
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_vf_examples.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validate_vf_examples", VALIDATOR_PATH)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def test_regbase_library_is_self_contained() -> None:
    files = [path for path in REGBASE_ROOT.rglob("*") if path.is_file()]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    example_text = "\n".join(
        path.read_text(encoding="utf-8") for path in files if path.name != "cann_version_compat_faq.md"
    )
    assert "http://" not in text
    assert "https://" not in text
    assert "MicroAPI::" not in example_text
    assert "Reg::DataCopy" not in example_text
    assert "DataCopyUnAlign" not in example_text
    assert "Reg::ReduceMax" not in example_text
    assert "Reg::ReduceMin" not in example_text
    assert not (REGBASE_ROOT / "evidence").exists()

    for example in CATALOG["examples"]:
        assert "source" not in example
        assert (SKILL_ROOT / example["implementation_path"]).is_file()
        assert (SKILL_ROOT / example["guide_path"]).is_file()


def test_semantic_routes_cover_rope_and_logsumexp_loss() -> None:
    examples = {example["id"]: example for example in CATALOG["examples"]}
    rope_hints = {hint.lower() for hint in examples["paired-segment-rotate"]["match_hints"]}
    assert {"rope", "rotary", "x*cos + rotate(x)*sin"} <= rope_hints

    reduction_hints = " ".join(
        hint.lower()
        for example_id in ("reduce-row-full-load", "reduce-online-max-sum")
        for hint in examples[example_id]["match_hints"]
    )
    assert "logsumexp" in reduction_hints
    assert "cross entropy" in reduction_hints


def test_quantization_routes_are_distinct_and_cover_common_signals() -> None:
    examples = {example["id"]: example for example in CATALOG["examples"]}
    two_pass = " ".join(examples["two-pass-amax-cast"]["match_hints"]).lower()
    shared = " ".join(examples["shared-exponent-pack"]["match_hints"]).lower()

    assert {"dynamic quant", "amax then cast", "smooth quant"} <= {
        hint.lower() for hint in examples["two-pass-amax-cast"]["match_hints"]
    }
    assert {"mx quant", "shared exponent", "fp4 pack", "fp8 pack"} <= {
        hint.lower() for hint in examples["shared-exponent-pack"]["match_hints"]
    }
    assert "shared exponent" not in two_pass
    assert "amax then cast" not in shared


def test_checked_in_catalog_passes_its_documented_validator() -> None:
    assert VALIDATOR.validate(REGBASE_ROOT / "catalog.json") == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("verification", {"compile_status": "passed", "sdk_version": "9.2.0-beta.2"}),
        ("readiness", "annotated_excerpt"),
        ("example_build_contract", {"type": "device_only", "probe_path": "missing.cpp"}),
        ("sha256", "0" * 64),
    ],
)
def test_catalog_rejects_stale_metadata_and_broken_integration_contracts(
    tmp_path: Path, field: str, value: object
) -> None:
    catalog = json.loads(json.dumps(CATALOG))
    catalog["examples"][0][field] = value
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert VALIDATOR.validate(path)


def test_every_example_explains_why_and_how() -> None:
    for example in CATALOG["examples"]:
        guide = (SKILL_ROOT / example["guide_path"]).read_text(encoding="utf-8")
        assert "## 为什么这样写" in guide
        assert "## 怎么使用" in guide
