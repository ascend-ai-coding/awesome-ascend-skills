# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Pytest fixtures for anchor_construction tests."""

import json
import os
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader


GOLDEN_REF_DIR = Path(__file__).parent / "golden_ref"

DOC_PART_FILES = {
    "constraint": "constraint.json",
    "support": "support.json",
    "summary": "summary.json",
    "function_proto": "function_proto.json",
    "sub_function_hash": "sub_function_hash.json",
}


@pytest.fixture
def golden_dir() -> Path:
    """Path to golden reference directory."""
    return GOLDEN_REF_DIR


@pytest.fixture
def golden_data() -> dict[str, dict]:
    """Load all 5 golden reference JSON files.

    Returns dict keyed by doc_part name, each value is the parsed JSON.
    """
    data = {}
    for doc_part, filename in DOC_PART_FILES.items():
        filepath = GOLDEN_REF_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            data[doc_part] = json.load(f)
    return data


@pytest.fixture
def path_config() -> PathConfig:
    """PathConfig instance for test operator."""
    operator_path = os.environ.get("TEST_OPERATOR_PATH")
    if not operator_path:
        ops_root = os.environ.get("CANN_OPS_ROOT")
        if ops_root:
            operator_path = str(Path(ops_root) / "math" / "abs")
    if not operator_path:
        pytest.skip("需要 TEST_OPERATOR_PATH 或 CANN_OPS_ROOT 指向本机 ops 检出")
    out_path = "/tmp/anchor_test_output"
    return PathConfig(operator_path, out_path)


@pytest.fixture
def data_loader(path_config) -> DataLoader:
    """DataLoader instance from path_config fixture."""
    return DataLoader(path_config)
