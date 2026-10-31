# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Catalog gates cover additions, removals and nested examples without importing kernels."""

from pathlib import Path
import shutil

import pytest

from _checks import catalog

DOC = """| Example | Mechanism |
| --- | --- |
| [alpha](../../examples/algorithms/alpha/) | Projection |
| [product](../../examples/matmul/fundamental_usage/product/) | Product |
"""


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "examples"
    for relative in ("algorithms/alpha", "matmul/fundamental_usage/product"):
        folder = root / relative
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text("raise RuntimeError('must not import')\n")
    document = tmp_path / "catalog.md"
    document.write_text(DOC)
    return root, document


def test_complete_catalog_including_nested_examples_and_kernel_links(library):
    root, document = library
    document.write_text(
        DOC.replace("algorithms/alpha/)", "algorithms/alpha/kernel.py#entry)")
    )
    helper = root / "algorithms/alpha/_kernels/kernel.py"
    helper.parent.mkdir()
    helper.write_text("raise RuntimeError('must not import')\n")
    result = catalog.collect(root, document=document)
    assert result.examples == {
        Path("algorithms/alpha"),
        Path("matmul/fundamental_usage/product"),
    }
    assert result.indexed == result.examples
    assert not result.findings


@pytest.mark.parametrize(
    "extra",
    [
        "[alpha](../../examples/algorithms/alpha/)",
        "python examples/algorithms/alpha/kernel.py",
        "```markdown\n| [alpha](../../examples/algorithms/alpha/) | Fake table |\n```",
        "```markdown\n```python\n| [alpha](../../examples/algorithms/alpha/) | Fake table |\n```",
        "~~~markdown\n| [alpha](../../examples/algorithms/alpha/) | Fake table |\n~~~",
    ],
)
def test_prose_commands_and_fenced_tables_do_not_replace_a_selection_entry(
    library, extra
):
    root, document = library
    document.write_text(DOC.replace(DOC.splitlines()[2] + "\n", "") + extra)
    findings = catalog.collect(root, document=document).findings
    assert len(findings) == 1
    assert "Example not indexed: algorithms/alpha" in findings[0].message


def test_a_group_link_does_not_cover_unlisted_children(library):
    root, document = library
    document.write_text(DOC.replace("fundamental_usage/product/", "fundamental_usage/"))
    findings = catalog.collect(root, document=document).findings
    assert len(findings) == 1
    assert (
        "Example not indexed: matmul/fundamental_usage/product" in findings[0].message
    )


@pytest.mark.parametrize(
    "target", ["algorithms/missing/", "algorithms/alpha/missing.py"]
)
def test_dangling_links_fail_even_when_all_examples_are_indexed(library, target):
    root, document = library
    document.write_text(DOC + f"| [stale](../../examples/{target}) | Stale |\n")
    findings = catalog.collect(root, document=document).findings
    assert len(findings) == 1
    assert (
        "Broken example link" in findings[0].message and target in findings[0].message
    )
    assert findings[0].path == document and findings[0].lineno == 5


def test_an_existing_directory_without_an_example_is_not_a_valid_entry(library):
    root, document = library
    (root / "algorithms/empty").mkdir()
    document.write_text(DOC + "| [empty](../../examples/algorithms/empty/) | Empty |\n")
    assert any(
        "has no example kernel" in f.message
        for f in catalog.collect(root, document=document).findings
    )


@pytest.mark.parametrize("renamed", [False, True])
def test_removal_or_rename_requires_the_matching_index_change(library, renamed):
    root, document = library
    original = root / "algorithms/alpha"
    if renamed:
        original.rename(root / "algorithms/renamed")
    else:
        shutil.rmtree(original)
    result = catalog.collect(root, document=document)
    assert any("Broken example link" in f.message for f in result.findings)
    assert any("Example not indexed" in f.message for f in result.findings) == renamed
    document.write_text(
        DOC.replace("algorithms/alpha/", "algorithms/renamed/")
        if renamed
        else DOC.replace(DOC.splitlines()[2] + "\n", "")
    )
    assert not catalog.collect(root, document=document).findings


def test_missing_catalog_or_groups_and_empty_scans_fail(library):
    root, document = library
    document.unlink()
    assert any(
        "Cannot read catalog" in f.message
        for f in catalog.collect(root, document=document).findings
    )
    document.write_text("")
    shutil.rmtree(root / "algorithms")
    assert any(
        "Missing example group" in f.message
        for f in catalog.collect(root, document=document).findings
    )
    (root / "algorithms").mkdir()
    shutil.rmtree(root / "matmul/fundamental_usage")
    assert any(
        "No public kernel.py examples" in f.message
        for f in catalog.collect(root, document=document).findings
    )


def test_repository_catalog_is_complete():
    result = catalog.collect()
    assert result.examples
    assert not result.findings, "\n".join(map(str, result.findings))
