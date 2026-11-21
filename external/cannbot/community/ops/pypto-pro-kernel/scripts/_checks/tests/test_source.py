# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Scoped parse reuse must preserve fresh reads and isolate independent scans."""

import os

import pytest

from _checks import source


def test_scan_reuses_parses_but_reads_same_timestamp_edits(tmp_path, monkeypatch):
    path = tmp_path / "kernel.py"
    path.write_text("import pypto_pro.language as pl\n")
    original = source.ast.parse
    calls = []

    def parse(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(source.ast, "parse", parse)
    with source.reuse_parsed_sources():
        first = source.read(path)
        assert source.read(path).pl == "pl"
        assert len(calls) == 1
        stat = path.stat()
        path.write_text(path.read_text().replace("as pl", "as pp"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert source.read(path).pl == "pp"
        assert first.pl == "pl" and len(calls) == 2
    assert source.read(path).pl == "pp" and len(calls) == 3


def test_failed_and_nested_scans_release_their_parsed_snapshots(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text("value = 1\n")
    with pytest.raises(RuntimeError), source.reuse_parsed_sources():
        outer = source.read(path)
        with source.reuse_parsed_sources():
            assert source.read(path) is not outer
        assert source.read(path) is outer
        raise RuntimeError("abort scan")
    assert source.read(path) is not outer
