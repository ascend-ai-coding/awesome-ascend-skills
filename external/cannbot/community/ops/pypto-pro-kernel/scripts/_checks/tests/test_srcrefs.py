# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Criteria for `src-line-drift`.

This module differs from the other rules: **it reads `../pypto/`**. The tests therefore create a small source tree
(`tmp_path`) rather than depending on whether a real PyPTO checkout exists or which commit it is on; otherwise, changes
to that tree could break the tests.

The criteria are “which references were reported and which were checked.” **False positives are part of the contract
too**: this is a report, not a gate, and its roughly 60% precision is a documented fact. Tests must pin the cases that
can produce false positives so nobody wires it into `rules` as a gate.
"""

from pathlib import Path

import pytest

from _checks import srcrefs
from _checks.cli import DOC_ROOTS, report_srcrefs
from _checks.paths import PYPTO_ROOT


def _tree(root: Path, files: dict[str, str]) -> Path:
    """Create a source tree. Keys are relative paths and values are file contents."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _doc(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    return path


def _lines(count: int, **marks: int) -> str:
    """Create a `count`-line file; `marks` specifies what content to put on each line (1-based)."""
    body = [f"# filler {i}" for i in range(1, count + 1)]
    for text, lineno in marks.items():
        body[lineno - 1] = f"def {text}(self):"
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Non-reported cases
# ---------------------------------------------------------------------------


def test_an_identifier_at_the_target_line_is_clean(tmp_path):
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, _static_dim=50)})
    doc = _doc(tmp_path, "helper `_static_dim` 在 `block_ops.py:50`\n")
    suspects, checked, skipped, waived = srcrefs.collect([doc], src)
    assert suspects == []
    assert (checked, skipped, waived) == (1, 0, 0)


def test_the_window_reaches_fifteen_lines_out(tmp_path):
    """`WINDOW` is part of the contract: this distance usually spans from a function header to its first check."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, _check_mx=50)})
    doc = _doc(tmp_path, f"`_check_mx` 在 `block_ops.py:{50 + srcrefs.WINDOW}`\n")
    assert srcrefs.collect([doc], src)[0] == []


def test_a_dotted_reference_matches_on_its_last_segment(tmp_path):
    """The document says `ir.BlockOps.emit`, while the source contains only `emit`."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, emit=50)})
    doc = _doc(tmp_path, "`ir.BlockOps.emit` 在 `block_ops.py:50`\n")
    assert srcrefs.collect([doc], src)[0] == []


def test_cpp_files_are_checked_too(tmp_path):
    """Line-number drift is not unique to Python; `backend_cce_block_out_ops.cpp` has also been cited."""
    src = _tree(tmp_path / "src", {"out_ops.cpp": _lines(600, TEXPANDS=548)})
    doc = _doc(tmp_path, "`TEXPANDS` 在 `out_ops.cpp:548`\n")
    assert srcrefs.collect([doc], src)[0] == []


def test_the_backtick_form_is_recognised(tmp_path):
    """The docs use both forms: `` `f.py`:12 `` and `` `f.py:12` ``."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, _static_dim=88)})
    doc = _doc(tmp_path, "`_static_dim`（`block_ops.py`:88）\n")
    suspects, checked, _, _ = srcrefs.collect([doc], src)
    assert (suspects, checked) == ([], 1)


# ---------------------------------------------------------------------------
# Reported cases
# ---------------------------------------------------------------------------


def test_a_drifted_line_is_reported(tmp_path):
    """The core case: all five line numbers in the `check-coverage.md` table were stale in exactly this way."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _static_dim=308)})
    doc = _doc(tmp_path, "helper `_static_dim` 在 `block_ops.py:3234`\n")
    suspects, checked, skipped, waived = srcrefs.collect([doc], src)
    assert [(s.doc, s.lineno, s.target, s.target_line) for s in suspects] == [
        (doc, 1, "block_ops.py", 3234)
    ]
    assert (checked, skipped, waived) == (1, 0, 0)


def test_just_past_the_window_is_reported(tmp_path):
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, _check_mx=50)})
    doc = _doc(tmp_path, f"`_check_mx` 在 `block_ops.py:{50 + srcrefs.WINDOW + 1}`\n")
    assert len(srcrefs.collect([doc], src)[0]) == 1


def test_a_line_past_the_end_of_the_file_says_so(tmp_path):
    """The clearest case: **the identifier is irrelevant** because the file is not that long."""
    src = _tree(tmp_path / "src", {"_errors.py": _lines(120, __str__=60)})
    doc = _doc(tmp_path, "`__str__` 在 `_errors.py:428`\n")
    suspects, checked, _, _ = srcrefs.collect([doc], src)
    assert len(suspects) == 1
    assert "只有 120 行" in suspects[0].reason
    assert checked == 1


def test_a_line_past_the_end_is_reported_without_any_identifier(tmp_path):
    """References without identifiers are usually skipped, but “past the end of the file” is an exception that needs no
    identifier.
    """
    src = _tree(tmp_path / "src", {"_errors.py": _lines(120)})
    doc = _doc(tmp_path, "见 _errors.py:428\n")
    suspects, checked, skipped, _ = srcrefs.collect([doc], src)
    assert len(suspects) == 1 and (checked, skipped) == (1, 0)


def test_the_message_carries_the_rule_name_and_both_locations(tmp_path):
    """The full message can change, but **rule name + document location + target location** are the contract for people
    and grep.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _static_dim=308)})
    doc = _doc(tmp_path, "helper `_static_dim` 在 `block_ops.py:3234`\n")
    rendered = str(srcrefs.collect([doc], src)[0][0])
    assert srcrefs.RULE in rendered
    assert f"{doc}:1" in rendered
    assert "`block_ops.py:3234`" in rendered
    assert "_static_dim" in rendered


# ---------------------------------------------------------------------------
# Indeterminate cases: skip instead of guessing
# ---------------------------------------------------------------------------


def test_two_files_with_the_same_name_are_skipped(tmp_path):
    """If more than one file has the same name, the reference cannot be checked—choosing either could produce the
    **opposite** conclusion.
    """
    src = _tree(
        tmp_path / "src",
        {"a/block_ops.py": _lines(100, _static_dim=50), "b/block_ops.py": _lines(100)},
    )
    doc = _doc(tmp_path, "`_static_dim` 在 `block_ops.py:50`\n")
    assert srcrefs.collect([doc], src) == ([], 0, 1, 0)


def test_a_file_that_is_not_in_the_tree_is_skipped(tmp_path):
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100)})
    doc = _doc(tmp_path, "`_gone` 在 `not_here.py:50`\n")
    assert srcrefs.collect([doc], src) == ([], 0, 1, 0)


def test_a_reference_with_no_identifier_is_skipped(tmp_path):
    """Without a checkable anchor, do not guess: missing a report is better than reporting the wrong thing."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100)})
    doc = _doc(tmp_path, "见 `block_ops.py:50`\n")
    assert srcrefs.collect([doc], src) == ([], 0, 1, 0)


def test_dsl_names_do_not_count_as_identifiers(tmp_path):
    """`pl.load` is a DSL name, not a source identifier; searching for it would match everywhere."""
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100)})
    doc = _doc(tmp_path, "`pl.load` 见 `block_ops.py:50`\n")
    assert srcrefs.collect([doc], src) == ([], 0, 1, 0)


def test_the_filename_itself_is_not_used_as_the_anchor(tmp_path):
    """When the entire `` `block_ops.py:50` `` reference is inside backticks, do not treat the filename as an identifier
    to search for.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100)})
    doc = _doc(tmp_path, "见 `block_ops.py:50` 那一行\n")
    assert srcrefs.collect([doc], src) == ([], 0, 1, 0)


def test_build_directories_are_not_scanned(tmp_path):
    """Generated copies under `build/` would make every file appear to have a duplicate basename."""
    src = _tree(
        tmp_path / "src",
        {
            "block_ops.py": _lines(100, _static_dim=50),
            "build/block_ops.py": _lines(100),
        },
    )
    assert (
        srcrefs.collect([_doc(tmp_path, "`_static_dim` `block_ops.py:50`\n")], src)[0]
        == []
    )


# ---------------------------------------------------------------------------
# Known boundaries: document them so they are not “fixed” as bugs
# ---------------------------------------------------------------------------


def test_single_digit_line_numbers_are_not_matched(tmp_path):
    """`\\d{2,5}` starts at two-digit numbers. **This is a tradeoff, not an oversight**: a single digit could match
    version numbers, dimensions, or forms like `b8`. The cost is that references to lines 1–9 are not checked; those
    lines usually contain imports and are rarely cited.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100)})
    doc = _doc(tmp_path, "`_static_dim` 在 `block_ops.py:7`\n")
    assert srcrefs.collect([doc], src) == ([], 0, 0, 0)


def test_an_identifier_elsewhere_in_the_sentence_gives_a_false_positive(tmp_path):
    """**Known false-positive rate: about 40%.** A document may say “check `Y` at `f.py:N` for `X`,” where `Y` is the
    line's content and `X` is far away. The heuristic cannot tell which identifier corresponds to the line, so it
    searches for all of them and clears the reference if any one matches. Neither matches here, so it reports the
    reference. This is why it cannot be a gate.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _real_name=1159)})
    doc = _doc(tmp_path, "`_documented_name` 在 `block_ops.py:1159`\n")
    assert len(srcrefs.collect([doc], src)[0]) == 1


def test_one_matching_identifier_clears_the_whole_line(tmp_path):
    """Multiple identifiers on one line are treated as OR: one match clears the reference. Prefer a missed report to
    noise in the report.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(100, _check_mx=50)})
    doc = _doc(tmp_path, "`_unrelated_name` 与 `_check_mx` 见 `block_ops.py:50`\n")
    assert srcrefs.collect([doc], src)[0] == []


# ---------------------------------------------------------------------------
# When no source tree is available
# ---------------------------------------------------------------------------


def test_a_missing_source_tree_fails_loudly(tmp_path):
    """**Do not silently return “zero suspicious references”**—that would be read as “everything was checked and is
    fine.”
    """
    doc = _doc(tmp_path, "`_static_dim` 在 `block_ops.py:50`\n")
    with pytest.raises(SystemExit, match="srcrefs"):
        srcrefs.collect([doc], tmp_path / "nope")


def test_source_references_use_the_configured_pypto_checkout():
    """Source-reference checks share the tools' external framework source selection."""
    assert srcrefs.PYPTO.is_absolute()
    assert srcrefs.PYPTO == PYPTO_ROOT


# ---------------------------------------------------------------------------
# `VERIFIED`: manual waivers. **The line number in the key** is the whole point.
# ---------------------------------------------------------------------------


def test_a_verified_entry_is_waived_not_reported(tmp_path):
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _real_name=308)})
    doc = _doc(tmp_path, "`_documented_name` 在 `block_ops.py:3234`\n")
    ok = {(doc.as_posix(), "block_ops.py", 3234)}
    suspects, checked, _, waived = srcrefs.collect([doc], src, ok)
    assert (suspects, checked, waived) == ([], 1, 1)


def test_a_verified_entry_stops_applying_when_the_line_number_changes(tmp_path):
    """**This is why `VERIFIED` exists.** A waiver is tied to the line number; changing it returns the reference to the
    report instead of permanently silencing a citation that has drifted.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _real_name=308)})
    doc = _doc(
        tmp_path, "`_documented_name` 在 `block_ops.py:3300`\n"
    )  # The line number changed.
    ok = {(doc.as_posix(), "block_ops.py", 3234)}  # The old line number was waived.
    assert len(srcrefs.collect([doc], src, ok)[0]) == 1


def test_a_verified_entry_does_not_leak_to_another_doc(tmp_path):
    """The key includes the document path: the same line number cited in another document is a different reference and
    needs its own check.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(4000, _real_name=308)})
    doc = _doc(tmp_path, "`_documented_name` 在 `block_ops.py:3234`\n")
    ok = {("docs/somewhere-else.md", "block_ops.py", 3234)}
    assert len(srcrefs.collect([doc], src, ok)[0]) == 1


def test_a_verified_entry_cannot_waive_a_line_past_the_end_of_the_file(tmp_path):
    """A reference past the end of the file is a **hard fact** that can be judged without identifiers, so it cannot be
    waived. If it is waived, the `VERIFIED` entry itself has expired.
    """
    src = _tree(tmp_path / "src", {"block_ops.py": _lines(120)})
    doc = _doc(tmp_path, "`_documented_name` 在 `block_ops.py:3234`\n")
    ok = {(doc.as_posix(), "block_ops.py", 3234)}
    suspects, _, _, waived = srcrefs.collect([doc], src, ok)
    assert len(suspects) == 1 and waived == 0


def test_the_shipped_verified_table_still_matches_the_real_docs():
    """**The waiver table can go stale.** Every entry must still be a candidate in the report; if a citation is changed
    to a symbol name, its waiver becomes dead, and the next person may assume it is still active.

    This test requires `../pypto/`; skip it when the checkout is absent so the remaining tests can run without PyPTO
    installed locally.
    """
    if not srcrefs.PYPTO.is_dir():
        pytest.skip("没有 ../pypto，这条核不了")
    docs = sorted(p for r in DOC_ROOTS for p in r.rglob("*.md"))
    without = {s for s in srcrefs.collect(docs, verified=set())[0]}
    live = {(srcrefs.document_key(s.doc), s.target, s.target_line) for s in without}
    dead = srcrefs.VERIFIED - live
    assert not dead, f"`VERIFIED` 里这些已经不是候选了，删掉：{sorted(dead)}"


def test_cli_checks_references_inside_the_skill(tmp_path, monkeypatch, capsys):
    repo = _tree(
        tmp_path / "skill",
        {
            "references/api/probe.md": "helper `missing_rule` 在 `block_ops.py:50`\n",
        },
    )
    _tree(tmp_path / "pypto", {"python/block_ops.py": _lines(100, actual_rule=50)})
    monkeypatch.chdir(repo)

    assert report_srcrefs((repo,), tmp_path / "pypto")
    output = capsys.readouterr().out
    assert "核了 1 处行号引用，1 处可疑" in output
    assert "references/api/probe.md" in output
