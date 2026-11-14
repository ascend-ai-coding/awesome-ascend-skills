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


def invoke(
    tmp_path,
    *,
    body,
    status="201",
    curl_exit="0",
    comment="compile",
    fixture_token=None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    curl = fake_bin / "curl"
    curl.write_text(
        '#!/bin/sh\nprintf \'%s\\n%s\' "$FAKE_BODY" "$FAKE_STATUS"\nexit "$FAKE_EXIT"\n'
    )
    curl.write_text(
        curl.read_text().replace(
            'exit "$FAKE_EXIT"',
            'if [ "$FAKE_EXIT" != 0 ]; then printf "%s\\n" "$@" >&2; fi\nexit "$FAKE_EXIT"',
        )
    )
    curl.chmod(0o755)
    # Runtime-only dummy data; never an API credential or a network request.
    fixture_token = fixture_token or uuid.uuid4().hex
    env = dict(
        os.environ,
        GITCODE_TOKEN=fixture_token,
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


def test_connection_failure_has_structured_output(tmp_path):
    result = invoke(tmp_path, body={}, status="000", curl_exit="7")
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "status": "fail",
        "http_code": 0,
        "curl_exit": 7,
        "detail": "curl transport failed",
    }


def test_timeout_has_structured_output(tmp_path):
    fixture_token = uuid.uuid4().hex
    result = invoke(
        tmp_path, body={}, status="000", curl_exit="28", fixture_token=fixture_token
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["curl_exit"] == 28
    assert fixture_token not in result.stdout + result.stderr


def test_http_error_remains_a_failure(tmp_path):
    result = invoke(tmp_path, body={"error": "forbidden"}, status="403")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["http_code"] == 403
    assert data["status"] == "fail"


def test_successful_transport_keeps_the_success_path(tmp_path):
    result = invoke(tmp_path, body={"id": 123})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ok"
