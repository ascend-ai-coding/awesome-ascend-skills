# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the public script from other working directories and a relocated skill."""

import re
import shutil
import subprocess
import sys

import pytest

from _checks.paths import PYPTO_ROOT, SKILL_ROOT
from _checks.srcrefs import VERIFIED

ENTRY = SKILL_ROOT / "scripts/check.py"
SAMPLE = '''"""写法：原始。Static-check fixture."""
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def fill():
    value = vf.full(1.0, dtype=pl.DT_FP32)
'''


def run(entry, *args, cwd):
    # -E ignores PYTHONPATH: the public entry must find its own implementation.
    return subprocess.run(
        [sys.executable, "-E", str(entry), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def bundle(tmp_path):
    target = tmp_path / "bundle" / "skill with spaces"
    shutil.copytree(
        SKILL_ROOT,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    return target


@pytest.mark.parametrize("shape,status", [("[1, 64]", 0), ("[64]", 1)])
def test_relative_rule_targets_keep_caller_cwd(tmp_path, shape, status):
    (tmp_path / "probe.py").write_text(
        "import pypto_pro.language as pl\n"
        f"def kernel():\n    tile = pl.TileType(shape={shape})\n"
    )
    result = run(ENTRY, "rules", "probe.py", cwd=tmp_path)
    assert result.returncode == status, result.stdout + result.stderr
    assert "1 个文件" in result.stdout
    if status:
        assert "tile-shape-must-be-2d" in result.stdout


def test_default_rules_are_independent_of_cwd(tmp_path):
    outputs = []
    for directory in (SKILL_ROOT.parent, SKILL_ROOT, tmp_path):
        result = run(ENTRY, "rules", cwd=directory)
        assert result.returncode in (0, 1), result.stdout + result.stderr
        assert int(re.search(r"rules：(\d+) 个文件", result.stdout)[1]) > 0
        outputs.append((result.returncode, result.stdout))
    assert outputs[0] == outputs[1] == outputs[2]


def test_ordinary_vf_is_checked_when_host_ast_has_no_match_node(tmp_path):
    source = tmp_path / "probe.py"
    source.write_text(SAMPLE)
    code = """import ast, runpy, sys
if hasattr(ast, "Match"):
    del ast.Match
entry, source = sys.argv[1:]
sys.path.insert(0, str(__import__("pathlib").Path(entry).parent))
sys.argv = [entry, "rules", source]
runpy.run_path(entry, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ENTRY), str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 个文件，0 处违规" in result.stdout


def test_a_relocated_skill_checks_explicit_files_without_companion_repos(
    bundle, tmp_path
):
    source = tmp_path / "probe.py"
    source.write_text(SAMPLE)
    result = run(bundle / "scripts/check.py", "rules", source.name, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 个文件，0 处违规" in result.stdout


def test_a_missing_default_scope_is_not_reported_as_a_clean_scan(bundle, tmp_path):
    shutil.rmtree(bundle / "examples")
    result = run(bundle / "scripts/check.py", "rules", cwd=tmp_path)
    assert result.returncode == 2
    assert "--examples-root" in result.stderr
    assert "0 个文件" not in result.stdout


@pytest.mark.parametrize("command", ["rules", "coverage", "surface"])
def test_a_relocated_skill_uses_its_bundled_examples(bundle, tmp_path, command):
    result = run(bundle / "scripts/check.py", command, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    if command == "rules":
        assert int(re.search(r"rules：(\d+) 个文件", result.stdout)[1]) > 0
    elif command == "coverage":
        assert "0 处对不上" in result.stdout
    else:
        assert "vf.full\t" in result.stdout


def test_relocated_skill_uses_its_contract_and_the_selected_examples(bundle, tmp_path):
    examples = tmp_path / "custom library"
    sample = examples / "basic_api" / "chapter" / "sample" / "kernel.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(SAMPLE)
    (bundle / "references/tooling/example-coverage.md").write_text(
        "### 八点二、1 项逐项\n\n| 1 | `full` | `sample` |\n"
    )
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "example-coverage.md").write_text("This is not the skill's contract.")

    for command in ("surface", "coverage"):
        result = run(
            bundle / "scripts/check.py",
            "--examples-root",
            "../custom library",
            command,
            cwd=caller,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if command == "surface":
            assert "vf.full\tsample" in result.stdout
        else:
            assert "1 项 ↔ 1 个样例，0 处对不上" in result.stdout


def test_default_command_gates_the_bundled_catalog_against_selected_examples(
    bundle, tmp_path
):
    examples = tmp_path / "custom library"
    for relative in ("basic_api/chapter/sample", "algorithms/alpha"):
        folder = examples / relative
        folder.mkdir(parents=True)
        (folder / "kernel.py").write_text(SAMPLE)
    (examples / "matmul").mkdir()
    (bundle / "references/tooling/example-coverage.md").write_text(
        "### 八点二、1 项逐项\n\n| 1 | `full` | `sample` |\n"
    )
    document = bundle / "references/tooling/examples.md"
    document.write_text("")
    caller = tmp_path / "caller"
    caller.mkdir()
    row = "| [alpha](../../examples/algorithms/alpha/) | Projection |\n"
    (caller / "examples.md").write_text(row)
    entry = bundle / "scripts/check.py"
    for body, status in (("", 1), (row, 0)):
        document.write_text(body)
        for command in ((), ("catalog",)):
            result = run(
                entry, "--examples-root", "../custom library", *command, cwd=caller
            )
            assert result.returncode == status, result.stdout + result.stderr
            assert "catalog：1 个样例" in result.stdout
            assert ("Example not indexed: algorithms/alpha" in result.stdout) == bool(
                status
            )


def test_relocated_skill_reads_an_explicit_stub_tree_without_importing_it(
    bundle, tmp_path
):
    source = tmp_path / "framework source"
    stub = source / "python/pypto_pro/language/_api.py"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        "raise RuntimeError('must not import')\n\ndef marker(x: int) -> int:\n    return x\n"
    )
    entry = bundle / "scripts/check.py"
    result = run(entry, "--pypto-root", source, "api", "pl", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pl.marker(x: int) -> int" in result.stdout
    missing = run(
        entry, "--pypto-root", tmp_path / "missing", "api", "pl", cwd=tmp_path
    )
    assert missing.returncode == 1
    assert "--pypto-root" in missing.stdout


def test_relocated_skill_checks_and_regenerates_its_pyi_reference(bundle, tmp_path):
    entry = bundle / "scripts/check.py"
    reference = bundle / "references/api/pypto_pro_language.pyi"
    reference.write_text("# stale reference\n")
    args = ("--pypto-root", PYPTO_ROOT, "api")
    result = run(entry, *args, "--check-pyi", cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert reference.read_text() == "# stale reference\n"
    result = run(entry, *args, "--write-pyi", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    result = run(entry, *args, "--check-pyi", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rejected = run(entry, *args, "vf", "--write-pyi", cwd=tmp_path)
    assert rejected.returncode == 2


def test_relocation_preserves_verified_source_reference_waivers(bundle, tmp_path):
    if not PYPTO_ROOT.is_dir():
        pytest.skip("PyPTO source checkout is unavailable")
    result = run(
        bundle / "scripts/check.py", "--pypto-root", PYPTO_ROOT, "srcrefs", cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(VERIFIED)} 处按 `VERIFIED` 手核放行" in result.stdout
