# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Missing group slots are gated only for resolved declarations and known parsers."""

import json
import os
import subprocess
import sys

import pytest

from _checks import group_depth, query
from _checks.cli import rules_result
from _checks.paths import PYPTO_ROOT, SKILL_ROOT


def source(tmp_path, body, alias="pl", args=""):
    path = tmp_path / "candidate.py"
    path.write_text(
        f"import pypto_pro.language as {alias}\n@{alias}.jit()\ndef kernel({args}):\n"
        + body
        + "\n"
    )
    return path


@pytest.mark.parametrize("mutex", ['"auto"', "None", "[]", "()"])
@pytest.mark.parametrize("depth", ["", ", depth=None"])
def test_required_depth_is_rejected_for_resolved_modes(tmp_path, mutex, depth):
    path = source(
        tmp_path,
        f"    g = pl.make_tile_group(type=object(), addrs=0, mutex_ids={mutex}{depth})",
    )
    found, counts = group_depth.inspect(path)
    assert counts == {"checked": 1, "unverified": 0, "guard_verified": True}
    assert len(found) == 1 and found[0].lineno == 4 and not found[0].advisory
    report = rules_result([path], pypto_root=PYPTO_ROOT)
    assert report["status"] == "failed"
    assert [c["id"] for c in report["context"]] == ["tile-group-depth"]


@pytest.mark.parametrize(
    "arguments",
    [
        'mutex_ids="auto", depth=1',
        "mutex_ids=None, depth=2",
        "mutex_ids=[], depth=slots",
        "mutex_ids=[0]",
        "mutex_ids=(1, 2)",
    ],
)
def test_present_depth_and_manual_inference_are_not_rejected(tmp_path, arguments):
    path = source(
        tmp_path, f"    g = pl.make_tile_group(type=object(), addrs=0, {arguments})"
    )
    found, counts = group_depth.inspect(path)
    assert not found and counts["checked"] == 1 and counts["unverified"] == 0


@pytest.mark.parametrize(
    "body",
    [
        '    if False:\n        g = pl.make_tile_group(mutex_ids="auto")',
        '    for i in range(0):\n        g = pl.make_tile_group(mutex_ids="auto")',
        '    def helper():\n        g = pl.make_tile_group(mutex_ids="auto")',
        "    g = pl.make_tile_group(**options)",
        "    g = pl.make_tile_group(mutex_ids=ids)",
        "    g = pl.make_tile_group(type=object(), addrs=0)",
        '    g = pl.make_tile_group(mutex_ids="auto") if active else None',
        '    with unknown_scope():\n        g = pl.make_tile_group(mutex_ids="auto")',
        '    pl = other\n    g = pl.make_tile_group(mutex_ids="auto")',
        '    import other as pl\n    g = pl.make_tile_group(mutex_ids="auto")',
        '    pl.make_tile_group = other\n    g = pl.make_tile_group(mutex_ids="auto")',
        '    return\n    g = pl.make_tile_group(mutex_ids="auto")',
    ],
)
def test_unresolved_calls_do_not_become_hard_errors(tmp_path, body):
    path = source(tmp_path, body)
    found, counts = group_depth.inspect(path)
    assert not found and counts["checked"] == 0 and counts["unverified"] == 1


def test_native_section_alias_and_parameter_shadowing(tmp_path):
    path = source(
        tmp_path,
        '    with native.section_vector():\n        g = native.make_tile_group(mutex_ids="auto")',
        alias="native",
    )
    found, counts = group_depth.inspect(path)
    assert len(found) == 1 and found[0].lineno == 5 and counts["checked"] == 1
    path = source(tmp_path, '    g = pl.make_tile_group(mutex_ids="auto")', args="pl")
    assert group_depth.inspect(path)[1]["unverified"] == 1


def test_missing_or_changed_parser_only_advises_and_refreshes_same_size_content(
    tmp_path,
):
    path = source(tmp_path, '    g = pl.make_tile_group(mutex_ids="auto")')
    root = tmp_path / "framework"
    assert group_depth.inspect(path, pypto_root=root)[0][0].advisory
    parser = root / group_depth.PARSER
    parser.parent.mkdir(parents=True)
    parser.write_bytes((PYPTO_ROOT / group_depth.PARSER).read_bytes())
    assert not group_depth.inspect(path, pypto_root=root)[0][0].advisory
    stamp = parser.stat()
    parser.write_text(
        parser.read_text().replace("if depth is None:", "if depth is True:", 1)
    )
    os.utime(parser, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    found, counts = group_depth.inspect(path, pypto_root=root)
    assert found[0].advisory and not counts["guard_verified"]


def test_source_gate_prevents_native_command_for_missing_depth(tmp_path):
    path = source(tmp_path, '    g = pl.make_tile_group(mutex_ids="auto")')
    marker = tmp_path / "started"
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').write('ran')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert not json.loads(result.stdout)["command"]["started"] and not marker.exists()


def test_scenario_query_surfaces_missing_slots_before_execution(tmp_path):
    path = source(tmp_path, '    g = pl.make_tile_group(mutex_ids="auto")')
    result = query.report("UB to L1", source=path)
    assert result["status"] == "failed"
    assert {c["id"] for c in result["context"]} == {"ub-l1", "tile-group-depth"}
    assert result["coverage"]["group_depth"]["checked"] == 1
    result = query.report("unknown scenario", source=path)
    assert result["status"] == "failed"
    assert {row["id"] for row in result["context"]} == {"tile-group-depth"}
    assert "tile-group-depth-required" in {d["rule"] for d in result["diagnostics"]}
    assert not result["signatures"] and not result["related_contracts"]
    assert all(row["kind"] == "document" for row in result["knowledge"]["items"])
    result = query.report("unknown scenario", kinds=["contract"], source=path)
    assert "query-no-match" in {d["rule"] for d in result["diagnostics"]}
