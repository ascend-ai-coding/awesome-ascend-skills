#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for the standalone fault-PC command-line tools."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run_cli(script, arguments, cwd):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "script", ["decode_err_info_pc.py", "disassemble_pc.py", "locate_kernel.py"]
)
def test_standalone_help_from_another_directory(script, tmp_path):
    result = run_cli(script, ["--help"], tmp_path)
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "script, arguments",
    [
        ("decode_err_info_pc.py", ["--error-type", "MTE", "--err-info"]),
        ("disassemble_pc.py", ["--obj", "unused.o", "--pc"]),
        ("locate_kernel.py", ["--operator", "Add", "--tiling-key"]),
    ],
)
@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_invalid_numeric_arguments_use_argparse_errors(
    script, arguments, value, tmp_path
):
    result = run_cli(script, [*arguments, value], tmp_path)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "integer" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "unit, register", [("CCU", "0x32b510506c8000e9"), ("MTE", "0xd9000000e9")]
)
def test_decoder_preserves_json_output_and_same_domain_comparison(
    unit, register, tmp_path
):
    result = run_cli(
        "decode_err_info_pc.py",
        [
            "--error-type",
            unit,
            "--err-info",
            register,
            "--pc-base",
            "0x36000",
            "--direct-offset",
            "0x7a4",
        ],
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["decodedPcBytes"] == "0x367a4"
    assert report["normalization"]["decodedOffsetBytes"] == "0x7a4"
    assert report["comparison"]["equal"] is True


def test_decoder_validation_diagnostic_goes_to_stderr(tmp_path):
    result = run_cli(
        "decode_err_info_pc.py",
        [
            "--error-type",
            "MTE",
            "--err-info",
            "0xd9000000e9",
            "--current-pc",
            "0x367a4",
        ],
        tmp_path,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "--current-pc and --start-pc must be provided together" in result.stderr


def test_locator_preserves_json_hash_and_tiling_key_filter(tmp_path):
    opp_root = tmp_path / "opp"
    kernel_dir = opp_root / "ascend910b" / "flash_attention_score_grad"
    kernel_dir.mkdir(parents=True)
    object_bytes = b"test kernel object"
    (kernel_dir / "kernel.o").write_bytes(object_bytes)
    metadata = {
        "kernelList": [
            {"tilingKey": "0x2", "kernelName": "kernel", "kernelType": "MIX"},
            {"tilingKey": "0x3", "kernelName": "other", "kernelType": "AIV"},
        ]
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(metadata), encoding="utf-8")
    result = run_cli(
        "locate_kernel.py",
        [
            "--opp-root",
            str(opp_root),
            "--operator",
            "FlashAttentionScoreGrad",
            "--tiling-key",
            "2",
            "--soc",
            "ascend910b",
        ],
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    records = json.loads(result.stdout)
    assert len(records) == 1
    assert records[0]["objectSha256"] == hashlib.sha256(object_bytes).hexdigest()
    assert records[0]["entries"][0]["symbolCandidates"] == [
        "kernel_mix_aic",
        "kernel_mix_aiv",
    ]
    assert len(records[0]["entries"]) == 1
