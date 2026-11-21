# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Owner schemas, partial reads and ambiguous source identities."""

import json
import os

import pytest

from _checks import feedback, host_io, query, query_metadata
from _checks.tests.knowledge_fixtures import write


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"id":"a","id":"b"}',
        '{"id":"a","facets":{"dtype":["FP16"],"dtype":["FP32"]}}',
        '{"id":"a","unknown":true}',
        '{"id":1}',
        '{"aliases":["x"]}',
        '{"id":"a:b"}',
        '{"id":"a","aliases":"x"}',
        '{"id":"a","aliases":[" x"]}',
        '{"id":"a","term_groups":[[]]}',
        '{"id":"a","symbols":["not_a_public_symbol"]}',
        '{"id":"a","facets":{"dtype":"FP16"}}',
        '{"id":"a","facets":{"dtype":[true]}}',
        '{"id":"a","review_symbols":[["pl.move"]]}',
        '{"id":"a","example":"../outside/kernel.py"}',
        '{"id":"a","example":"algorithms/_private/kernel.py"}',
        '{"id":"a","candidate":{"id":"x"}}',
        '{"id":"a","aliases":[NaN]}',
    ],
)
def test_invalid_metadata_is_located_and_does_not_discard_other_owners(
    tmp_path, payload
):
    bad = write(
        tmp_path / "bad.md",
        "# Source\n\n<!-- kernel-query: " + payload + " -->\n## Broken\nBody\n",
    )
    good = write(
        tmp_path / "good.md",
        '<!-- kernel-query: {"id":"good"} -->\n## Valid\nOriginal body.\n',
    )
    entries, report = query_metadata.collect(tmp_path)
    assert [meta["id"] for meta, _, _ in entries] == ["good"]
    assert entries[0][1] == good
    assert not report["complete"] and report["errors"][0]["line"] == 3
    assert report["errors"][0]["path"] == str(bad)
    assert report["errors"][0]["sha256"] == feedback.file_identity(bad)["sha256"]


def test_fenced_markers_are_examples_and_body_heading_rules_are_shared(tmp_path):
    write(
        tmp_path / "owner.md",
        """# Reference
~~~markdown
<!-- kernel-query: {broken} -->
## Fenced
~~~
<!-- kernel-query: {"id":"actual"} -->
## Actual
Original text.
```markdown
## Literal
```
### Detail
Keep this subtree.
## Next
Other body.
""",
    )
    entries, report = query_metadata.collect(tmp_path)
    assert report["complete"] and report["markers"] == 1
    assert entries[0][0]["line"] == 7
    assert "Keep this subtree." in entries[0][2] and "Other body." not in entries[0][2]


@pytest.mark.parametrize(
    "body",
    [
        '<!-- kernel-query: {"id":"a"}\n## A\n',
        '<!-- kernel-query: {"id":"a"} -->\nNo heading\n',
    ],
)
def test_incomplete_markers_are_not_silently_ignored(tmp_path, body):
    write(tmp_path / "broken.md", body)
    entries, report = query_metadata.collect(tmp_path)
    assert not entries and not report["complete"] and report["error_count"] == 1


def test_duplicate_ids_quarantine_every_owner_and_preserve_source_locations(tmp_path):
    for name in ("one", "two"):
        write(
            tmp_path / (name + ".md"),
            '<!-- kernel-query: {"id":"same"} -->\n## ' + name + "\nBody\n",
        )
    entries, report = query_metadata.collect(tmp_path)
    assert not entries and report["objects"] == 0
    assert report["error_count"] == 2
    assert {row["reason"] for row in report["errors"]} == {"duplicate_id"}
    assert len({row["path"] for row in report["errors"]}) == 2


def test_candidate_id_collisions_do_not_choose_the_first_recipe(tmp_path):
    candidate = {
        "id": "same",
        "title": "Comparison",
        "comparison": "Hold the workload fixed.",
        **{
            key: ["Condition"]
            for key in ("conditions", "decrease", "increase", "reject_if")
        },
    }
    for identity in ("one", "two"):
        write(
            tmp_path / (identity + ".md"),
            "<!-- kernel-query: "
            + json.dumps({"id": identity, "candidate": candidate})
            + " -->\n## Owner\nBody\n",
        )
    entries, report = query_metadata.collect(tmp_path)
    assert not entries and report["error_count"] == 2
    assert {row["reason"] for row in report["errors"]} == {"duplicate_candidate_id"}


def test_vf_spellings_normalize_without_modifying_the_source_object():
    meta = {
        "id": "convert",
        "symbols": ["pl.Vf.astype"],
        "review_symbols": [["pl.Vf.astype"]],
        "when": "Conversion consumer.",
    }
    parsed = query_metadata.validate(meta, kind="contract")
    assert parsed["symbols"] == ["vf.astype"] and parsed["review_symbols"] == [
        ["vf.astype"]
    ]
    assert meta["symbols"] == ["pl.Vf.astype"]


@pytest.mark.parametrize("problem", ["invalid_utf8", "escaped_path", "scan_limit"])
def test_reference_errors_retain_partial_coverage_and_current_bytes(
    tmp_path, problem, monkeypatch
):
    root = tmp_path / "references"
    good = write(root / "a.md", '<!-- kernel-query: {"id":"good"} -->\n## Good\nBody\n')
    bad = root / "z.md"
    if problem == "invalid_utf8":
        bad.write_bytes(b"\xff\xfe")
    elif problem == "escaped_path":
        bad.symlink_to(write(tmp_path / "outside.md", "private text"))
    else:
        write(bad, "# Extra\n")
        monkeypatch.setitem(host_io.CEILINGS, "max_files", 1)
    entries, report = query_metadata.collect(root)
    assert entries[0][1] == good and not report["complete"]
    assert str(tmp_path / "outside.md") not in report["_files"]
    if problem == "invalid_utf8":
        assert str(bad) in report["_files"]


def test_same_timestamp_bad_metadata_edits_change_the_source_manifest(tmp_path):
    path = write(tmp_path / "owner.md", '<!-- kernel-query: {"id":1} -->\n## Body\n')
    before = query_metadata.collect(tmp_path)[1]
    stamp = path.stat()
    path.write_text(path.read_text().replace('"id":1', '"id":2'))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = query_metadata.collect(tmp_path)[1]
    assert before["manifest_sha256"] != after["manifest_sha256"]


def test_reference_failure_keeps_valid_queries_and_marks_empty_results_partial(
    tmp_path, monkeypatch
):
    write(tmp_path / "good.md", '<!-- kernel-query: {"id":"good"} -->\n## Good\nBody\n')
    write(tmp_path / "bad.md", "<!-- kernel-query: {broken} -->\n## Broken\n")
    monkeypatch.setattr(query, "REFERENCES_ROOT", tmp_path)
    found = query.report("contract:good", kinds=["contract"])["knowledge"]
    assert (
        found["lookup"]["status"] == "matched"
        and found["lookup"]["coverage"] == "partial"
    )
    assert found["items"][0]["id"] == "contract:good"
    missing = query.report("contract:missing", kinds=["contract"])["knowledge"]
    assert missing["lookup"]["status"] == "source_unavailable"
    assert missing["lookup"]["knowledge_absence"] == "not_established"
