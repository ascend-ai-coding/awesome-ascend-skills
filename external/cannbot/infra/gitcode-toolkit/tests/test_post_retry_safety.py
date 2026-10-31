# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Non-idempotent writes must not be replayed after an ambiguous failure."""

import importlib.util
from pathlib import Path
from http.client import parse_headers
from socketserver import StreamRequestHandler, ThreadingTCPServer
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gitcode_client.py"
SPEC = importlib.util.spec_from_file_location("post_retry_safety_client", SCRIPT)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class WriteSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.writes = []

    def post(self, url, **kwargs):
        self.writes.append(kwargs.get("data", kwargs.get("json")))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(status):
    return SimpleNamespace(
        status_code=status,
        headers={},
        text="",
        url="https://api.example.test/comments",
        request=SimpleNamespace(method="POST"),
    )


@pytest.mark.parametrize("helper", ["form", "json"])
@pytest.mark.parametrize("failure", ["timeout", "connection", "server"])
def test_post_is_not_replayed_after_server_may_have_committed(helper, failure):
    first = {
        "timeout": client.requests.ReadTimeout("response lost after commit"),
        "connection": client.requests.ConnectionError("response lost after commit"),
        "server": response(500),
    }[failure]
    session = WriteSession([first, response(201)])
    expected = client.requests.HTTPError if failure == "server" else type(first)
    with patch.object(client.time, "sleep"), pytest.raises(expected):
        if helper == "form":
            client.api_post(
                session,
                "https://api.example.test/comments",
                "test-token",
                data={"body": "comment"},
            )
        else:
            client.api_post_json(
                session,
                "https://api.example.test/comments",
                "test-token",
                json_data={"body": "comment"},
            )
    assert len(session.writes) == 1


@pytest.mark.parametrize("helper", ["form", "json"])
def test_explicit_rate_limit_rejection_can_still_be_retried(helper):
    session = WriteSession([response(429), response(201)])
    with patch.object(client.time, "sleep"):
        if helper == "form":
            result = client.api_post(
                session,
                "https://api.example.test/comments",
                "test-token",
                data={"body": "comment"},
            )
        else:
            result = client.api_post_json(
                session,
                "https://api.example.test/comments",
                "test-token",
                json_data={"body": "comment"},
            )
    assert result.status_code == 201
    assert len(session.writes) == 2


@pytest.mark.parametrize("helper", ["form", "json"])
def test_committed_local_http_write_is_not_repeated_when_response_is_lost(helper):
    writes = []

    class Handler(StreamRequestHandler):
        def handle(self):
            self.rfile.readline()
            headers = parse_headers(self.rfile)
            writes.append(self.rfile.read(int(headers["Content-Length"])))
            # Commit the HTTP request, then close without returning a response.

    server = ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/comments"
        with client.requests.Session() as session, patch.object(client.time, "sleep"):
            with pytest.raises(client.requests.ConnectionError):
                if helper == "form":
                    client.api_post(
                        session, url, "test-token", data={"body": "comment"}, timeout=2
                    )
                else:
                    client.api_post_json(
                        session,
                        url,
                        "test-token",
                        json_data={"body": "comment"},
                        timeout=2,
                    )
        assert len(writes) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
