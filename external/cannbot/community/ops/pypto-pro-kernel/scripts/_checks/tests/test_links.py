# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise link integrity, false-positive controls and the standalone CLI."""

import shutil
import subprocess
import sys

import pytest

from _checks import cli, links
from _checks.paths import SKILL_ROOT


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_renamed_targets_and_headings_fail_until_incoming_links_are_updated(tmp_path):
    source = write(tmp_path, "start.md", "# Start\n[guide](guide.md#entry)\n")
    target = write(tmp_path, "guide.md", "## Entry\n")
    assert not links.collect([source]).findings
    renamed = target.with_name("renamed.md")
    target.rename(renamed)
    result = links.collect([source])
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert (finding.path, finding.lineno, finding.rule) == (source, 2, links.RULE)
    assert "Missing target" in finding.message
    source.write_text("# Start\n[guide](renamed.md#entry)\n")
    renamed.write_text("## New entry\n")
    assert "Missing Markdown anchor" in links.collect([source]).findings[0].message
    source.write_text("# Start\n[guide](renamed.md#new-entry)\n")
    assert not links.collect([source]).findings


def test_heading_text_duplicate_collisions_and_html_anchors(tmp_path):
    document = write(
        tmp_path,
        "guide.md",
        """# Guide
## `pl.gemm()` — 参数 ###
## Repeat
## Repeat
## Repeat-1
## See [the guide](#guide)
<a id="named-anchor"></a>
<h3 id='explicit-heading'>Label</h3>
[unicode](#plgemm--%E5%8F%82%E6%95%B0)
[first](#repeat)
[second](#repeat-1)
[collision](#repeat-1-1)
[label](#see-the-guide)
[named](#named-anchor)
[html](#explicit-heading)
""",
    )
    result = links.collect([document])
    assert not result.findings
    assert result.checked_fragments == 8


def test_code_examples_comments_and_indented_code_are_not_links(tmp_path):
    document = write(
        tmp_path,
        "guide.md",
        """# Guide
`kernel[grid](missing.py)` and ``[fake](missing.md)``
```markdown
[fake](missing.md)
```python
[still fenced](missing.md)
```
~~~~markdown
[fake](missing.md)
~~~
[still fenced](missing.md)
~~~~
<!-- [fake](missing.md)
[fake](missing.md) -->
    [indented example](missing.md)
[real](#guide)
""",
    )
    result = links.collect([document])
    assert not result.findings
    assert (result.local_links, result.checked_fragments) == (1, 1)
    document.write_text(document.read_text() + "[broken](missing.md)\n")
    assert len(links.collect([document]).findings) == 1


def test_fenced_or_commented_headings_do_not_create_anchors(tmp_path):
    document = write(
        tmp_path,
        "guide.md",
        """# Guide
```
## Hidden
```
<!-- <a id="also-hidden"></a> -->
[fake heading](#hidden)
[fake anchor](#also-hidden)
""",
    )
    result = links.collect([document])
    assert len(result.findings) == 2
    assert all(
        "Missing Markdown anchor" in finding.message for finding in result.findings
    )


def test_encoded_paths_titles_and_code_in_link_labels(tmp_path):
    write(tmp_path, "refs/a (draft).md", "## Ready & checked\n")
    write(tmp_path, "refs/x.py", "raise RuntimeError('must not import')\n")
    document = write(
        tmp_path,
        "guide.md",
        """[`example`](<refs/a (draft).md#ready--checked> "Guide title")
[encoded](refs/a%20%28draft%29.md#ready--checked)
[source](refs/x.py#L1)
[directory](refs/)
""",
    )
    result = links.collect([document])
    assert not result.findings
    assert (
        result.local_links,
        result.checked_fragments,
        result.unverified_fragments,
    ) == (4, 2, 1)


def test_external_urls_are_counted_without_fetching_them(tmp_path):
    document = write(
        tmp_path,
        "guide.md",
        """[web](https://example.invalid/gone#missing)
[mail](mailto:nobody@example.invalid)
[host](//example.invalid/path)
""",
    )
    result = links.collect([document])
    assert not result.findings
    assert (result.local_links, result.external_links) == (0, 3)


def test_overlapping_scan_roots_do_not_duplicate_findings(tmp_path):
    document = write(tmp_path, "guide.md", "[broken](missing.md)\n")
    result = links.collect([tmp_path, document])
    assert result.documents == (document,)
    assert result.local_links == len(result.findings) == 1


def test_empty_and_missing_inputs_are_not_clean_scans(tmp_path):
    assert any(
        "No Markdown" in finding.message
        for finding in links.collect([tmp_path]).findings
    )
    assert any(
        "Expected a Markdown" in finding.message
        for finding in links.collect([tmp_path / "absent"]).findings
    )


def test_cli_resolves_explicit_paths_from_caller_and_reports_failure(
    tmp_path, monkeypatch, capsys
):
    document = write(tmp_path, "guide.md", "# Entry\n[valid](#entry)\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["links", "guide.md"]) == 0
    document.write_text("# Entry\n[broken](#missing)\n")
    assert cli.main(["links", "guide.md"]) == 1
    output = capsys.readouterr().out
    assert f"{document}:2: {links.RULE}" in output
    with pytest.raises(SystemExit) as error:
        cli.main(["links", "missing.md"])
    assert error.value.code == 2


def test_relocated_public_entry_checks_its_own_docs_from_another_cwd(tmp_path):
    bundle = tmp_path / "relocated skill"
    shutil.copytree(
        SKILL_ROOT / "scripts",
        bundle / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    write(bundle, "SKILL.md", "[guide](references/guide.md#entry)\n")
    guide = write(
        bundle, "references/guide.md", "# Entry\n[companion](../../companion.py)\n"
    )
    write(tmp_path, "companion.py", "raise RuntimeError('must not import')\n")
    caller = tmp_path / "caller"
    caller.mkdir()
    command = [sys.executable, "-E", str(bundle / "scripts/check.py"), "links"]
    passed = subprocess.run(
        command, cwd=caller, text=True, capture_output=True, timeout=30
    )
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "2 Markdown files" in passed.stdout
    guide.write_text("# Renamed\n")
    failed = subprocess.run(
        command, cwd=caller, text=True, capture_output=True, timeout=30
    )
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "references/guide.md#entry" in failed.stdout
