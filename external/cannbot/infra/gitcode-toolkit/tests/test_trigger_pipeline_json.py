# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Offline CLI regression tests; curl is replaced and no real token is used."""

import json
import os
import subprocess
import uuid
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trigger_pr_pipeline.sh"


def invoke(tmp_path, *, body, status="201", curl_exit="0", comment="compile"):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    curl = fake_bin / "curl"
    curl.write_text(
        '#!/bin/sh\nprintf \'%s\\n%s\' "$FAKE_BODY" "$FAKE_STATUS"\nexit "$FAKE_EXIT"\n'
    )
    curl.chmod(0o755)
    env = dict(
        os.environ,
        GITCODE_TOKEN=uuid.uuid4().hex,
        FAKE_BODY=json.dumps(body),
        FAKE_STATUS=status,
        FAKE_EXIT=curl_exit,
        PATH=str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
    )
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo",
            "cann/fixture",
            "--pr",
            "123",
            "--comment",
            comment,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_string_comment_id_is_valid_json(tmp_path):
    result = invoke(tmp_path, body={"id": "7cfd7ecc6d9363ebe746be083d8086925c32ab32"})
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["comment_id"] == "7cfd7ecc6d9363ebe746be083d8086925c32ab32"
    assert data["status"] == "ok"


def test_custom_comment_round_trips_quotes_newlines_and_backslashes(tmp_path):
    comment = 'compile "fixture"\nC:\\temp\\result'
    result = invoke(tmp_path, body={"id": 123}, comment=comment)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["comment"] == comment


def test_numeric_comment_id_remains_numeric(tmp_path):
    result = invoke(tmp_path, body={"id": 123})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["comment_id"] == 123


def test_missing_comment_id_still_produces_valid_json(tmp_path):
    result = invoke(tmp_path, body={})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["comment_id"] == "?"
