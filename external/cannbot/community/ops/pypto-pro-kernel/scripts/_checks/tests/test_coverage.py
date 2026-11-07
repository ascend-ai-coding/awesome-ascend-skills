# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Criteria for `api-coverage` and `surface`.

Test the mechanism against a fabricated small repository; on the real repository, check only that the table parses and
its declared item count matches its row count. **Do not encode the current four gaps as criteria**—they need to be
fixed, and pinning them would make the tests fail as soon as they are fixed.
"""

from pathlib import Path

import pytest

from _checks import coverage

DOC = """\
# 假的设计文档

### 八点一、别的小节

| # | `vf.` API | 样例 |
| --- | --- | --- |
| 1 | `不该被读到` | `别处` |

### 八点二、3 项逐项

| # | `vf.` API | 样例 | 批 |
| --- | --- | --- | --- |
| 1 | `alpha` | `written` | 1 |
| 2 | `beta` | `written` | 1 |
| 3 | `gamma` | `unwritten` | 5 |

### 八点三、后面的小节

| 1 | `也不该被读到` | `别处` |
"""

KERNEL = """\
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


@pl.vector_function
def _body(src):
    return vf.alpha(src), pl.make_tile(src)
"""


def _repo(tmp_path: Path, examples: dict[str, str] | None = None) -> Path:
    (tmp_path / "coverage.md").write_text(DOC, encoding="utf-8")
    for name, kernel in (examples or {"written": KERNEL}).items():
        folder = tmp_path / "examples/basic_api/chapter" / name
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text(kernel, encoding="utf-8")
    return tmp_path


def _collect(root: Path):
    return coverage.collect(root / "examples", document=root / "coverage.md")


def test_only_the_named_section_is_parsed(tmp_path):
    """The preceding and following sections both contain rows with the same shape. Anchor parsing to the section
    heading, or two extra rows will be included.
    """
    table = _collect(_repo(tmp_path))
    assert table.expected == 3
    assert table.assigned == {
        "alpha": ("written",),
        "beta": ("written",),
        "gamma": ("unwritten",),
    }


def test_an_assigned_api_the_example_never_calls_is_a_problem(tmp_path):
    problems = _collect(_repo(tmp_path)).problems()
    assert len(problems) == 1
    assert "`vf.beta`" in problems[0] and "全仓没有任何样例调它" in problems[0]


def test_an_api_assigned_here_but_called_there_names_where(tmp_path):
    """When the assigned example lacks a call, the diagnostic should name other examples that use the API."""
    root = _repo(
        tmp_path,
        {
            "written": KERNEL.replace("vf.alpha(src)", "vf.beta(src)"),
            "elsewhere": KERNEL,
        },
    )
    problems = _collect(root).problems()
    assert len(problems) == 1
    assert "`vf.alpha`" in problems[0] and "其他调用样例：elsewhere" in problems[0]


def test_an_api_assigned_to_an_unwritten_example_is_pending_not_a_problem(tmp_path):
    table = _collect(_repo(tmp_path))
    assert table.pending() == {"unwritten": ["gamma"]}
    assert all("gamma" not in line for line in table.problems())


def test_a_row_count_that_contradicts_the_heading_is_a_problem(tmp_path):
    root = _repo(tmp_path)
    doc = root / "coverage.md"
    doc.write_text(
        DOC.replace("### 八点二、3 项逐项", "### 八点二、9 项逐项"), encoding="utf-8"
    )
    problems = _collect(root).problems()
    assert any("自报 9 项" in line and "解析到 3 行" in line for line in problems)


@pytest.mark.parametrize(
    "cell",
    [
        "`written`（分布模式）",
        "`written`（调用 `alpha`） + `elsewhere`（位宽绑定）",
    ],
)
def test_example_annotations_do_not_hide_rows_or_become_assignments(tmp_path, cell):
    kernel = KERNEL.replace("pl.make_tile(src)", "vf.beta(src)")
    root = _repo(tmp_path, {"written": kernel, "elsewhere": kernel})
    (root / "coverage.md").write_text(
        DOC.replace("| `written` | 1 |", f"| {cell} | `not_an_example` |"),
        encoding="utf-8",
    )
    table = _collect(root)
    assert len(table.assigned) == table.expected == 3
    assert table.problems() == []
    assert table.pending() == {"unwritten": ["gamma"]}


@pytest.mark.parametrize("missing_call", ["written", "elsewhere"])
def test_every_assigned_example_must_call_the_api(tmp_path, missing_call):
    kernel = KERNEL.replace("pl.make_tile(src)", "vf.beta(src)")
    examples = {"written": kernel, "elsewhere": kernel}
    examples[missing_call] = kernel.replace("vf.alpha(src)", "vf.full(src)")
    root = _repo(tmp_path, examples)
    (root / "coverage.md").write_text(
        DOC.replace(
            "| `alpha` | `written` |",
            "| `alpha` | `written`（主样例） + `elsewhere`（补充样例） |",
        ),
        encoding="utf-8",
    )
    problems = _collect(root).problems()
    assert len(problems) == 1
    assert "`vf.alpha`" in problems[0] and f"`{missing_call}`" in problems[0]


def test_an_unwritten_additional_example_is_reported_as_pending(tmp_path):
    kernel = KERNEL.replace("pl.make_tile(src)", "vf.beta(src)")
    root = _repo(tmp_path, {"written": kernel})
    (root / "coverage.md").write_text(
        DOC.replace(
            "| `alpha` | `written` |",
            "| `alpha` | `written`（已有） + `planned`（待写） |",
        ),
        encoding="utf-8",
    )
    table = _collect(root)
    assert table.problems() == []
    assert table.pending() == {"planned": ["alpha"], "unwritten": ["gamma"]}


@pytest.mark.parametrize("cell", ["", "待补样例"])
def test_a_row_without_an_example_name_cannot_pass(tmp_path, cell):
    kernel = KERNEL.replace("pl.make_tile(src)", "vf.beta(src)")
    root = _repo(tmp_path, {"written": kernel})
    (root / "coverage.md").write_text(
        DOC.replace("| `alpha` | `written` |", f"| `alpha` | {cell} |"),
        encoding="utf-8",
    )
    table = _collect(root)
    assert len(table.assigned) == table.expected == 3
    problems = table.problems()
    assert len(problems) == 1
    assert "`vf.alpha`" in problems[0] and "样例名" in problems[0]


def test_surface_indexes_both_namespaces(tmp_path):
    index = coverage.surface(_repo(tmp_path) / "examples")
    assert index["vf"] == {"alpha": ["written"]}
    assert index["pl"]["make_tile"] == ["written"]


def test_a_bare_decorator_counts_as_usage(tmp_path):
    """`@pl.vector_function` has no parentheses, so the AST contains an `Attribute`, not a `Call`; accepting only `Call`
    would miss it. `@pl.jit()` has parentheses and was already included. This test pins that fix.
    """
    index = coverage.surface(_repo(tmp_path) / "examples")
    assert index["pl"]["vector_function"] == ["written"]


def test_constants_do_not_enter_the_index(tmp_path):
    """Do not count every `pl.<attr>` as an API call: constants such as `pl.DT_FP32` would swamp the index."""
    kernel = KERNEL.replace("pl.make_tile(src)", "pl.make_tile(src, dtype=pl.DT_FP32)")
    index = coverage.surface(_repo(tmp_path, {"written": kernel}) / "examples")
    assert "DT_FP32" not in index["pl"]


def test_the_real_table_parses_and_agrees_with_its_own_heading():
    table = coverage.collect()
    assert len(table.assigned) == table.expected == 81
    assert len(table.written) >= 21


def test_the_real_surface_index_answers_the_reverse_question():
    """`surface` exists because the design document maps examples to APIs, while agents ask for the reverse mapping."""
    index = coverage.surface()
    # `squeeze` appears in three places: `vf_unalign` uses the STORE_REG chain, `vf_lane_layout:109` uses the default
    # NO_STORE_REG, and `support/spr_ctrl` uses it as an AR writer to probe `pl.get_spr()`.
    # **One API appearing in multiple examples is normal, but the forward table does not show this**—the third example
    # (in the support chapter, across chapter boundaries) was caught by this criterion; the failure revealed that the
    # index needed updating.
    assert index["vf"]["squeeze"] == ["spr_ctrl", "vf_lane_layout", "vf_unalign"]
    assert "vf_reduce" in index["vf"]["reduce_sum"]
    assert len(index["vf"]) >= 60


def test_two_level_namespaces_are_indexed_separately(tmp_path):
    """`pl.system.*` must be indexed separately. Examples contain 56 such calls, all of which were initially missed."""
    kernel = KERNEL.replace(
        "pl.make_tile(src)", "pl.make_tile(src) and pl.system.bar_all()"
    )
    index = coverage.surface(_repo(tmp_path, {"written": kernel}) / "examples")
    assert index["pl.system"] == {"bar_all": ["written"]}
    # The two-level namespace must not be swallowed by the one-level namespace: `pl` has no API named `system`.
    assert "system" not in index["pl"]
    assert index["pl"]["make_tile"] == ["written"]


def test_the_real_index_sees_the_system_namespace():
    index = coverage.surface()
    assert {"bar_all", "sync_src", "sync_dst"} <= set(index["pl.system"])
    assert "system" not in index["pl"]
