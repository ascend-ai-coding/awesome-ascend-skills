# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Probe and sampling honor the same device selection, without NPU hardware."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("environment,expected", [
    ({}, 0),
    ({"NPU_DEVICE_ID": "1"}, 1),
    ({"ASCEND_DEVICE_ID": "2"}, 2),
    ({"NPU_DEVICE_ID": "0", "ASCEND_DEVICE_ID": "2"}, 0),
])
def test_device_is_selected_before_any_work(tmp_path, monkeypatch, probe, environment, expected):
    device = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=device))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=device))
    for name in ("NPU_DEVICE_ID", "ASCEND_DEVICE_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    path = Path(__file__).resolve().parents[1] / "assets/app_runner/torch/torch_app_runner.py"
    spec = importlib.util.spec_from_file_location("torch_app_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def observe(*args):
        device.set_device.assert_called_once_with(expected)

    warmup = Mock(side_effect=observe)
    target = Mock(side_effect=observe)
    monkeypatch.setattr(runner, "make_ops_warmup_tensor", observe)
    monkeypatch.setattr(runner, "run_ops_warmup", warmup)
    monkeypatch.setattr(runner, "make_case_inputs", observe)
    monkeypatch.setattr(runner, "invoke_target", target)
    device.synchronize.side_effect = observe
    cases = tmp_path / "cases.csv"
    cases.write_text("case_id,attrs\n1,m=1;n=1\n")
    argv = [str(path), "--probe-warmup-op", "ArgMax"] if probe else [
        str(path), "--cases", str(cases), "--case-ids", "1", "--samples", "10",
        "--execution-manifest", str(tmp_path / "execution.json"),
        "--warmup-mode", "framework_ops", "--warmup-op", "ArgMax",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    device.set_device.assert_called_once_with(expected)
    assert warmup.call_count == (1 if probe else 10)
    assert target.call_count == (0 if probe else 10)
