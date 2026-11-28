# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source roles, document locators and bounded original-text reads."""

import os
import json
import shlex
from pathlib import Path

import pytest

from _checks import cli, feedback, host_io, links, query


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "pypto"
    references = tmp_path / "references"
    files = {
        "python/pypto_pro/language/__init__.py": """raise RuntimeError("do not import")
__all__ = ["insert", "move", "pto_assert"]
from ._api import insert, move, pto_assert
""",
        "python/pypto_pro/language/_api.py": '''raise RuntimeError("do not execute declarations")
def insert(dst, src, offset):
    """Insert at a two-dimensional offset.

    Keep the producer address map and consumer synchronization explicit.
    """
    raise RuntimeError("never execute")
def move(dst, src):
    """Move source data."""
    raise RuntimeError("never execute")
def pto_assert(condition):
    """Abort the kernel on failure."""
    raise RuntimeError("never execute")
''',
        "docs/zh/api/pro_api/insert.md": (
            "# pypto_pro.language.insert\r\n\r\n## 约束说明\r\nKeep the offset in"
            " bounds.\r\n\r\n## 调用示例\r\n```python\r\ncarry ="
            " vf.create_mask()\r\npl.insert(dst, src, [0, 0])\r\n```\r\n\r\n## 返回值\r\nNone\r\n"
        ),
        "docs/zh/api/pro_api/move.md": "# pypto_pro.language.move\n\nMove data between compatible tiles.\n",
        "docs/zh/api/pro_api/pto_assert.md": "# pypto_pro.language.pto_assert\n\nLogs only; execution continues.\n",
        "docs/zh/guide/programming_guide/pro/handover.md": (
            "# 跨核交接\n\n## 跨核中间数据传递\nUse pl.insert with explicit ownership.\n\n## 同步\nOrder the producer and consumer.\n"
        ),
        "docs/zh/guide/quick_start/pro/start.md": "# First kernel\n\nUse pl.move for this example.\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    references.mkdir()
    (references / "handover.md").write_text("""# Handover

<!-- kernel-query: {"id":"ub-l1","aliases":["UB to L1"],"symbols":["pl.insert","pl.move"],"facets":{"source_memory":["Vec"],"destination_memory":["Mat"]}} -->
## Handover contract
Use pl.insert after checking the physical address map.
Official owner: [insert](../pypto/docs/zh/api/pro_api/insert.md#调用示例).
""")
    monkeypatch.setattr(query, "REFERENCES_ROOT", references)
    return root, references


def content(result):
    return "".join(item["text"] for item in result["knowledge"]["content"]["items"])


def test_literal_framework_references_follow_the_selected_checkout(corpus):
    root, references = corpus
    document = root / "docs/zh/api/pro_api/physical-map.md"
    document.write_text("# Physical address map\n\nCoordinate layout requirements.\n")
    owner = references / "handover.md"
    owner.write_text(
        owner.read_text() + "\nReview `pypto/docs/zh/api/pro_api/physical-map.md`.\n"
    )
    view = query.report("UB to L1", kinds=["document"], pypto_root=root)["knowledge"]
    found = next(
        row for row in view["items"] if row["id"] == "document:pro-api/physical-map.md"
    )
    assert found["reference"]["path"] == str(document.resolve())
    assert found["reference"]["sha256"] == feedback.file_identity(document)["sha256"]


def test_literal_source_examples_and_escaped_paths_do_not_add_document_relations(
    corpus,
):
    root, references = corpus
    document = root / "docs/zh/api/pro_api/physical-map.md"
    document.write_text("# Physical address map\n\nCoordinate layout requirements.\n")
    owner = references / "handover.md"
    owner.write_text(
        owner.read_text()
        + "\n```text\n`pypto/docs/zh/api/pro_api/physical-map.md`\n```\n"
        + "\n`pypto/../outside.md` is outside the selected checkout.\n"
    )
    view = query.report("UB to L1", kinds=["document"], pypto_root=root)["knowledge"]
    assert "document:pro-api/physical-map.md" not in {
        row["id"] for row in view["items"]
    }


def test_compound_symbol_question_prioritizes_its_owner_over_cross_references(corpus):
    root, _ = corpus
    for index in range(8):
        path = root / f"docs/zh/api/pro_api/another-{index}.md"
        path.write_text(
            f"# Another operation {index}\n\nThis example also calls pl.insert.\n"
        )
    view = query.report(
        "pl.insert offset constraints", kinds=["document"], limit=5, pypto_root=root
    )["knowledge"]
    assert [row["id"] for row in view["items"][:2]] == [
        "document:pro-api/insert.md",
        "docstring:pl.insert",
    ]
    assert all(
        row["match"]["reason"] == "document_primary_symbol" for row in view["items"][:2]
    )
    assert view["page"]["total"] > 5
    assert any(
        row["match"]["reason"] == "document_mention" for row in view["items"][2:]
    )
    assert all(row["evidence"]["native"] == "not_established" for row in view["items"])


def test_namespace_and_bare_public_name_find_original_roles_without_support_inference(
    corpus,
):
    root, _ = corpus
    view = query.report(
        "PL insert output target declaration",
        limit=3,
        kinds=["api", "document"],
        filters={"dtype": "FP16"},
        pypto_root=root,
    )["knowledge"]
    assert [row["id"] for row in view["items"]] == [
        "symbol:pl.insert",
        "document:pro-api/insert.md",
        "docstring:pl.insert",
    ]
    assert all(row["match"]["reason"] == "public_name_hint" for row in view["items"])
    assert all(row["applicability"]["status"] == "unknown" for row in view["items"])
    assert all(row["evidence"]["native"] == "not_established" for row in view["items"])
    assert [row["evidence"]["kind"] for row in view["items"]] == [
        "source_declaration",
        "official_documentation",
        "source_docstring",
    ]
    original = view["items"][1]
    assert "carry = vf.create_mask()" in content(
        query.report(**original["resolve"], pypto_root=root)
    )


@pytest.mark.parametrize(
    "text",
    [
        "insert output target declaration",
        "PL inserted output target declaration",
        "VF insert output target declaration",
        "PL unknown.insert target declaration",
        "PL insert L1 to UB",
    ],
)
def test_public_name_hints_do_not_guess_namespace_substrings_or_reverse_routes(
    corpus, text
):
    root, _ = corpus
    view = query.report(text, kinds=["api", "document"], pypto_root=root)["knowledge"]
    assert not view["items"]


def test_known_document_reads_its_selected_source_after_broad_scan_limit(
    corpus, monkeypatch
):
    root, _ = corpus
    monkeypatch.setitem(host_io.CEILINGS, "max_files", 2)
    monkeypatch.setitem(host_io.DEFAULTS, "max_files", 2)
    broad = query.report("pl.insert", kinds=["document"], pypto_root=root)["knowledge"]
    assert not broad["coverage"]["documents"]["complete"]
    exact = query.report(
        "document:pro-guide/handover.md#跨核中间数据传递", pypto_root=root
    )
    view = exact["knowledge"]
    assert view["lookup"]["status"] == "matched" and "explicit ownership" in content(
        exact
    )
    assert view["request"]["kinds"] == ["document"]
    assert view["coverage"]["documents"]["files_read"] == 1
    assert view["coverage"]["documents"]["selection"] == {
        "mode": "exact_document",
        "id": "document:pro-guide/handover.md",
    }
    assert view["coverage"]["documents"]["scopes"].keys() == {"pro-guide"}
    assert view["coverage"]["examples"] == {"status": "not_requested"}


def test_known_docstring_does_not_require_markdown_roots(corpus):
    root, _ = corpus
    for document in (root / "docs").rglob("*.md"):
        document.unlink()
    exact = query.report("docstring:pl.insert", pypto_root=root)
    assert exact["knowledge"]["lookup"][
        "status"
    ] == "matched" and "two-dimensional offset" in content(exact)
    scope = exact["knowledge"]["coverage"]["documents"]
    assert scope["complete"] and scope["files_read"] == 0 and scope["docstrings"] == 1
    assert scope["selection"]["mode"] == "exact_docstring" and scope["scopes"] == {}


def test_exact_document_keeps_scope_explicit_when_an_unrelated_source_is_corrupt(
    corpus,
):
    root, _ = corpus
    (root / "docs/zh/api/pro_api/corrupt.md").write_bytes(b"\xff\xfe")
    broad = query.report("pl.insert", kinds=["document"], pypto_root=root)["knowledge"]
    assert not broad["coverage"]["documents"]["complete"]
    exact = query.report("document:pro-guide/handover.md", pypto_root=root)["knowledge"]
    assert exact["lookup"]["status"] == "matched"
    assert exact["coverage"]["documents"]["complete"]
    assert exact["coverage"]["documents"]["selection"] == {
        "mode": "exact_document",
        "id": "document:pro-guide/handover.md",
    }


def test_exact_document_keeps_existing_file_limit(corpus, monkeypatch):
    root, _ = corpus
    monkeypatch.setitem(host_io.CEILINGS, "max_file_bytes", 32)
    exact = query.report("document:pro-api/insert.md", pypto_root=root)["knowledge"]
    assert exact["lookup"]["status"] == "source_unavailable"
    assert not exact["coverage"]["documents"]["complete"] and "content" not in exact


def test_api_discovery_federates_original_owners_and_marks_literal_mentions(corpus):
    root, _ = corpus
    result = query.report("pl.insert", kinds=["document"], pypto_root=root)
    records = {row["id"]: row for row in result["knowledge"]["items"]}
    assert {
        "document:pro-api/insert.md",
        "docstring:pl.insert",
        "document:pro-guide/handover.md",
        "document:skill/handover.md",
    } <= set(records)
    assert (
        records["document:pro-api/insert.md"]["evidence"]["kind"]
        == "official_documentation"
    )
    assert records["docstring:pl.insert"]["evidence"]["kind"] == "source_docstring"
    assert (
        records["document:skill/handover.md"]["evidence"]["kind"]
        == "maintained_reference"
    )
    assert (
        records["document:pro-guide/handover.md"]["match"]["reason"]
        == "document_mention"
    )
    assert result["knowledge"]["coverage"]["documents"]["complete"]
    assert all("text" not in row and row["missing_facts"] for row in records.values())
    assert "content" not in result["knowledge"]


def test_symbol_and_contract_relations_link_sources_without_copying_support(corpus):
    root, _ = corpus
    result = query.report("pl.insert", pypto_root=root)
    declaration = next(
        row for row in result["knowledge"]["items"] if row["id"] == "symbol:pl.insert"
    )
    assert {item["target"] for item in declaration["relations"]} >= {
        "docstring:pl.insert",
        "document:pro-api/insert.md",
    }
    filtered = query.report(
        kinds=["document"],
        filters={"source_memory": "UB", "destination_memory": "L1", "dtype": "FP16"},
        pypto_root=root,
    )
    assert filtered["knowledge"]["items"]
    assert all(
        row["applicability"]["status"] == "unknown"
        and not row["applicability"]["matched_filters"]
        for row in filtered["knowledge"]["items"]
    )


def test_problem_lookup_resolves_original_guide_heading(corpus):
    root, _ = corpus
    row = query.report("跨核中间数据传递", kinds=["document"], pypto_root=root)[
        "knowledge"
    ]["items"][0]
    assert row["id"] == "document:pro-guide/handover.md"
    assert row["resolve"]["text"].endswith("#跨核中间数据传递")
    result = query.report(**row["resolve"], pypto_root=root)
    assert "explicit ownership" in content(result) and "## 同步" not in content(result)
    assert result["knowledge"]["content"]["items"][0]["section"]["start_line"] == 3


def test_official_example_fragment_preserves_raw_crlf_and_reference_hash(corpus):
    root, _ = corpus
    result = query.report(
        "document:pro-api/insert.md#%E8%B0%83%E7%94%A8%E7%A4%BA%E4%BE%8B",
        pypto_root=root,
    )
    expected = links.fragment_range(
        (root / "docs/zh/api/pro_api/insert.md").read_bytes().decode(), "调用示例"
    )
    lines = (
        (root / "docs/zh/api/pro_api/insert.md").read_bytes().splitlines(keepends=True)
    )
    assert content(result).encode() == b"".join(
        lines[expected["start_line"] - 1: expected["end_line"]]
    )
    assert "carry = vf.create_mask()" in content(result)
    assert (
        result["knowledge"]["content"]["items"][0]["sha256"]
        == feedback.file_identity(root / "docs/zh/api/pro_api/insert.md")["sha256"]
    )


def test_docstring_read_is_bound_to_its_literal_and_never_executes_the_module(corpus):
    root, _ = corpus
    result = query.report("docstring:pl.insert", pypto_root=root)
    assert '"""Insert at a two-dimensional offset.' in content(result)
    assert "never execute" not in content(result) and "def move" not in content(result)
    item = result["knowledge"]["items"][0]
    assert item["reference"]["end_line"] > item["reference"]["line"]
    assert (
        result["knowledge"]["content"]["items"][0]["start_line"]
        == item["reference"]["line"]
    )


def test_partial_document_text_output_supplies_a_working_cli_continuation(
    corpus, capsys
):
    root, _ = corpus
    owner = root / "docs/zh/api/pro_api/long example.md"
    owner.write_text(
        "# Long example\n## Usage\n"
        + "Original line.\n" * 200
        + "## Other\nOutside the request.\n"
    )
    result = query.report("document:pro-api/long%20example.md#usage", pypto_root=root)
    assert result["knowledge"]["content"]["status"] == "partial"
    feedback.print_result(result, json_output=False)
    prefix = "Content is partial; continue with "
    command = next(
        line.removeprefix(prefix)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    )
    assert cli.main(["--pypto-root", str(root), *shlex.split(command)]) == 0
    continued = json.loads(capsys.readouterr().out)
    assert (
        continued["items"][0]["offset_bytes"]
        == result["knowledge"]["content"]["next"]["offset_bytes"]
    )
    assert "Outside the request" not in continued["items"][0]["text"]


def test_conflicting_statements_remain_separate_unreconciled_evidence(corpus):
    root, _ = corpus
    result = query.report("pl.pto_assert", kinds=["document"], pypto_root=root)
    records = {row["document_role"]: row for row in result["knowledge"]["items"]}
    official = query.report(**records["official_pro_api"]["resolve"], pypto_root=root)
    source = query.report(**records["source_docstring"]["resolve"], pypto_root=root)
    assert "execution continues" in content(official) and "Abort the kernel" in content(
        source
    )
    assert result["knowledge"]["source_reconciliation"]["status"] == "not_assessed"
    assert (
        records["official_pro_api"]["reference"]
        != records["source_docstring"]["reference"]
    )
    assert all(
        row["evidence"]["native"]
        == row["evidence"]["numerical"]
        == row["evidence"]["performance"]
        == "not_established"
        for row in records.values()
    )


def test_documented_symbol_does_not_require_a_matching_source_declaration(corpus):
    root, _ = corpus
    (root / "docs/zh/api/pro_api/future.md").write_text(
        "# pypto_pro.language.future_api\n\nA documented interface.\n"
    )
    result = query.report("pl.future_api", kinds=["document"], pypto_root=root)
    assert (
        result["knowledge"]["items"][0]["relations"][0]["declaration"]
        == "not_established"
    )
    assert (
        not result["signatures"]
        and result["knowledge"]["lookup"]["knowledge_absence"] == "not_established"
    )


@pytest.mark.parametrize(
    "name", ["with space.md", "literal%20name.md", "heading#name.md", "中文.md"]
)
def test_derived_document_ids_roundtrip_special_path_characters(corpus, name):
    root, _ = corpus
    path = root / "docs/zh/api/pro_api" / name
    path.write_text("# pypto_pro.language.path_probe\n\nOriginal source.\n")
    row = query.report("pl.path_probe", kinds=["document"], pypto_root=root)[
        "knowledge"
    ]["items"][0]
    result = query.report(**row["resolve"], pypto_root=root)
    assert content(result) == path.read_text()
    assert result["knowledge"]["items"][0]["reference"]["path"] == str(path.resolve())


@pytest.mark.parametrize("change", ["body", "docstring", "rename", "delete"])
def test_document_browsing_cursors_reject_changed_sources(corpus, change):
    root, _ = corpus
    cursor = query.report(kinds=["document"], limit=1, pypto_root=root)["knowledge"][
        "page"
    ]["next_cursor"]
    path = root / (
        "python/pypto_pro/language/_api.py"
        if change == "docstring"
        else "docs/zh/api/pro_api/insert.md"
    )
    stamp = path.stat()
    if change in {"body", "docstring"}:
        original = path.read_bytes()
        path.write_bytes(original.replace(b"Keep", b"Read"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == "rename":
        path.rename(path.with_name("renamed.md"))
    else:
        path.unlink()
    with pytest.raises(ValueError, match="cursor"):
        query.report(kinds=["document"], cursor=cursor, pypto_root=root)


def test_a_pinned_resolve_cannot_silently_read_a_new_document(corpus):
    root, _ = corpus
    row = query.report("pl.insert", kinds=["document"], pypto_root=root)["knowledge"][
        "items"
    ][0]
    path = Path(row["reference"]["path"])
    stamp = path.stat()
    path.write_bytes(path.read_bytes().replace(b"Keep", b"Read"))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    result = query.report(**row["resolve"], pypto_root=root)
    assert (
        result["status"] == "unverified"
        and result["knowledge"]["lookup"]["status"] == "source_changed"
    )
    assert "content" not in result["knowledge"]


def test_missing_fragment_and_explicit_ambiguous_anchors_are_not_whole_file_fallbacks(
    corpus,
):
    root, _ = corpus
    result = query.report("document:pro-api/insert.md#gone", pypto_root=root)
    assert (
        result["knowledge"]["lookup"]["status"] == "fragment_missing"
        and result["status"] == "unverified"
    )
    assert "content" not in result["knowledge"] and result["knowledge"]["sections"]
    path = root / "docs/zh/api/pro_api/ambiguous.md"
    path.write_text('# Repeated\n<a id="same"></a>\nOne\n<a id="same"></a>\nTwo\n')
    result = query.report("document:pro-api/ambiguous.md#same", pypto_root=root)
    assert (
        result["knowledge"]["lookup"]["status"] == "fragment_ambiguous"
        and "content" not in result["knowledge"]
    )


def test_unreadable_missing_and_binary_sources_leave_other_documents_available(
    corpus, monkeypatch
):
    root, _ = corpus
    blocked = root / "docs/zh/api/pro_api/insert.md"
    read_bytes = host_io._bytes

    def guarded(path, limit):
        if path == blocked:
            raise PermissionError("blocked test source")
        return read_bytes(path, limit)

    monkeypatch.setattr(host_io, "_bytes", guarded)
    (root / "docs/zh/api/pro_api/binary.md").write_bytes(b"\xff\0")
    available = query.report("pl.move", kinds=["document"], pypto_root=root)[
        "knowledge"
    ]
    assert any(row["id"] == "document:pro-api/move.md" for row in available["items"])
    assert not available["coverage"]["documents"]["complete"]
    assert {row["reason"] for row in available["coverage"]["documents"]["errors"]} >= {
        "PermissionError",
        "UnicodeDecodeError",
    }
    result = query.report("document:pro-api/insert.md", pypto_root=root)
    assert result["knowledge"]["lookup"]["status"] == "source_unavailable"
    result = query.report("document:pro-api/missing.md", pypto_root=root)
    assert (
        result["knowledge"]["lookup"]["status"] == "source_unavailable"
        and result["knowledge"]["lookup"]["reason"] == "FileNotFoundError"
    )


def test_missing_provider_root_is_not_a_complete_empty_result(corpus):
    root, _ = corpus
    guide = root / "docs/zh/guide/programming_guide/pro"
    guide.rename(guide.with_name("unavailable"))
    view = query.report("跨核中间数据传递", kinds=["document"], pypto_root=root)[
        "knowledge"
    ]
    assert view["lookup"]["status"] == "source_unavailable"
    assert view["coverage"]["documents"]["scopes"]["pro-guide"]["complete"] is False


def test_document_symlink_escape_is_not_read_or_resolved(corpus, tmp_path):
    root, _ = corpus
    outside = tmp_path / "outside.md"
    outside.write_text("# Private source\n")
    (root / "docs/zh/api/pro_api/escape.md").symlink_to(outside)
    view = query.report(kinds=["document"], pypto_root=root)["knowledge"]
    assert not view["coverage"]["documents"]["complete"] and all(
        row["title"] != "Private source" for row in view["items"]
    )
    with pytest.raises(ValueError, match="escapes"):
        query.report("document:pro-api/escape.md", pypto_root=root)


def test_document_scan_limit_is_explicit_and_does_not_claim_absence(
    corpus, monkeypatch
):
    root, _ = corpus
    monkeypatch.setitem(host_io.CEILINGS, "max_files", 1)
    view = query.report(kinds=["document"], pypto_root=root)["knowledge"]
    assert not view["coverage"]["documents"]["complete"]
    assert any(
        row["reason"] == "document_scan_limit"
        for row in view["coverage"]["documents"]["errors"]
    )
    assert view["lookup"]["knowledge_absence"] == "not_established"


@pytest.mark.parametrize(
    "text",
    [
        "document:outside/a.md",
        "document:pro-api/../a.md",
        "document:pro-api/%2e%2e/a.md",
        "document:pro-api//absolute.md",
        "document:pro-api/a.py",
        "document:pro-api/a.md#",
        "docstring:arbitrary_name",
    ],
)
def test_invalid_locators_do_not_turn_into_arbitrary_file_reads(corpus, text):
    root, _ = corpus
    with pytest.raises(ValueError):
        query.report(text, pypto_root=root)


def test_real_official_addc_example_is_distinct_from_the_source_docstring():
    official = query.report(
        "document:pro-api/SIMD-API/Reg-API/basic_arithmetic/addc.md#调用示例"
    )
    source = query.report("docstring:vf.addc")
    assert "carry = vf.create_mask(" in content(official)
    assert "carry = vf.create_mask(" not in content(source)
    assert official["knowledge"]["items"][0]["document_role"] == "official_pro_api"
    assert source["knowledge"]["items"][0]["document_role"] == "source_docstring"
    assert (
        official["knowledge"]["content"]["items"][0]["sha256"]
        != source["knowledge"]["content"]["items"][0]["sha256"]
    )
