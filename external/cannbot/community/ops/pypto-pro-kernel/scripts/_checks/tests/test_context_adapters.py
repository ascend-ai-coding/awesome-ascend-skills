# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CLI task/batch semantics survive literal host adaptation."""

import json
import shlex
import subprocess
import sys

import pytest

from _checks import host_hooks
from _checks.paths import SKILL_ROOT


def policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(tmp_path)],
                "limits": {"max_output_bytes": 4096},
            }
        )
    )
    return path


def test_mixed_literal_reads_keep_ranges_and_one_total_budget(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("one\ntwo\nthree\n")
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
        "tool_input": {
            "command": "cat a.txt && sed -n '2,2p' a.txt && head -n 1 a.txt"
        },
    }
    result = host_hooks.pre_tool(payload, policy_path=policy(tmp_path))
    command = result["hookSpecificOutput"]["updatedInput"]["command"]
    run = subprocess.run(
        shlex.split(command), cwd=tmp_path, capture_output=True, text=True, check=True
    )
    data = json.loads(run.stdout)
    assert [item["text"] for item in data["items"]] == [
        source.read_text(),
        "two\n",
        "one\n",
    ]
    assert len(run.stdout.encode()) == data["budget"]["result_bytes"] <= 4096


@pytest.mark.parametrize("suffix", ["--task task.json", "--task=task.json"])
def test_task_query_receives_fixed_host_policy(tmp_path, suffix):
    fixed = policy(tmp_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
        "tool_input": {
            "command": f"{sys.executable} {SKILL_ROOT / 'scripts/check.py'} query {suffix} --policy ignored.json"
        },
    }
    command = host_hooks.pre_tool(payload, policy_path=fixed)["hookSpecificOutput"][
        "updatedInput"
    ]["command"]
    words = shlex.split(command)
    assert words.count("--policy") == 1 and words[-1] == str(fixed.resolve())
    assert "ignored.json" not in words
