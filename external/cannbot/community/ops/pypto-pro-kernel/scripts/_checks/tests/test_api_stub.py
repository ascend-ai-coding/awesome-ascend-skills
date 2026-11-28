# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public-surface fidelity and freshness gates for the generated agent reference."""

import ast
import os
import shutil
import tokenize
from pathlib import Path

import pytest

from _checks import api_stub, feedback, query
from _checks.paths import PYPTO_ROOT


@pytest.fixture
def source(tmp_path):
    files = {
        "python/pypto_pro/language/__init__.py": """
raise RuntimeError("must not import framework code")
__all__ = ["load", "Token", "Vf", "simt", "system", "mutex", "pipeline", "Descriptor", "renamed", "DEFAULT"]
from ._api import load, original as renamed
from ._vf_api import Vf
from ._simt_api import Simt as simt
from pypto.pypto_impl.ir import Token
from pypto_pro.ir.op import system_ops as system
from pypto_pro.ir.op.system_ops import lock as _lock
from pypto_pro.ir.op.block_ops import Descriptor
from pypto_pro.runtime import pipeline
mutex = SimpleNamespace(lock=_lock)
DEFAULT = Token.ON
""",
        "python/pypto_pro/language/_api.py": """
from typing import Any, List, Union
Tile = Any
TileGroup = Any
DType = Any
Offset = Union[List[int], int]
def load(dst: Tile, src, /, offsets: Offset, *, order=None) -> None:
    raise RuntimeError("declaration only")
def original(value: int = 3):
    return value
""",
        "python/pypto_pro/language/_vf_api.py": """
class Vf:
    @staticmethod
    @_api_decl
    def load(src, *, mode=Token.ON):
        raise RuntimeError("declaration only")
""",
        "python/pypto_pro/language/_simt_api.py": """
class Simt:
    @staticmethod
    def atomic_add(ptr, value, /):
        raise RuntimeError("declaration only")
""",
        "python/pypto_pro/ir/op/__init__.py": "",
        "python/pypto_pro/ir/op/system_ops.py": """
MAX_ID = 7
def lock(*, pipe, event_id: int = 0):
    raise RuntimeError("builder only")
""",
        "python/pypto_pro/ir/op/block_ops.py": """
from dataclasses import dataclass
@dataclass(frozen=True)
class Descriptor:
    shape: list[int]
    compact: bool = False
    def _validate(self):
        raise RuntimeError("private")
""",
        "python/pypto_pro/runtime/__init__.py": "from . import pipeline\n",
        "python/pypto_pro/runtime/pipeline/__init__.py": """
__all__ = ["Config", "stage"]
from .config import Config
from ._stage import stage
""",
        "python/pypto_pro/runtime/pipeline/config.py": """
from dataclasses import dataclass
@dataclass(frozen=True)
class Config:
    preload: int = 2
""",
        "python/pypto_pro/runtime/pipeline/_stage.py": "def stage(fn):\n    return fn\n",
        "python/src/bindings/ir/ir.cpp": """
py::enum_<Token>(ir, "Token", R"doc(Contains a semicolon; and a // marker.)doc")
    .value("ON", Token::On, "Enabled; all lanes")
    // A comment cannot add .value("FAKE", Token::Fake).
    .value("OFF", Token::Off);
""",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def named(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def test_freshness_distinguishes_missing_sources_from_stale_generated_output(
    source, tmp_path
):
    path = tmp_path / "language.pyi"
    path.write_text(api_stub.generate(source))
    current = api_stub.inspect_reference(pypto_root=source, path=path)
    assert current["status"] == "passed"
    assert current["reference"]["sha256"] == feedback.file_identity(path)["sha256"]
    path.write_text(path.read_text() + "\n# stale artifact\n")
    stale = api_stub.inspect_reference(pypto_root=source, path=path)
    assert stale["status"] == "failed" and stale["reason"] == "reference_stale"
    unavailable = api_stub.inspect_reference(pypto_root=tmp_path / "missing", path=path)
    assert (
        unavailable["status"] == "unverified"
        and unavailable["reason"] == "selected_source_unavailable"
    )
    api = source / "python/pypto_pro/language/_api.py"
    api.write_text("invalid !!!")
    broken = api_stub.inspect_reference(pypto_root=source, path=path)
    assert (
        broken["status"] == "failed" and broken["reason"] == "selected_source_invalid"
    )


def test_generation_and_rewrite_retain_repository_license(source, tmp_path):
    generated = api_stub.generate(source)
    assert generated.startswith("# Copyright")
    assert (
        "CANN Open Software License Agreement Version 2.0"
        in generated.split("# Generated", 1)[0]
    )
    output = tmp_path / "language.pyi"
    assert api_stub.report(write=True, pypto_root=source, path=output)
    assert output.read_text() == generated
    assert api_stub.report(pypto_root=source, path=output)


def test_reference_formatting_preserves_identity_but_stale_sources_and_signatures_fail(
    source, tmp_path
):
    generated = api_stub.generate(source)
    comments = "\n".join(
        line for line in generated.splitlines() if line.startswith("#")
    )
    formatted = comments + "\n\n" + ast.unparse(ast.parse(generated)) + "\n"
    output = tmp_path / "formatted.pyi"
    output.write_text(formatted)
    assert api_stub.report(pypto_root=source, path=output)
    output.write_text(formatted.replace("offsets", "changed_offsets"))
    assert not api_stub.report(pypto_root=source, path=output)
    output.write_text(formatted.replace("# Source SHA256", "# Stale Source SHA256"))
    assert not api_stub.report(pypto_root=source, path=output)
    output.write_text(formatted.removeprefix("# Copyright"))
    assert not api_stub.report(pypto_root=source, path=output)


def test_generation_keeps_signatures_aliases_namespaces_fields_and_binding_members(
    source,
):
    generated = api_stub.generate(source)
    tree = ast.parse(generated)
    declaration = named(tree, "load")
    original = named(
        ast.parse((source / "python/pypto_pro/language/_api.py").read_text()), "load"
    )
    assert ast.dump(declaration.args) == ast.dump(original.args)
    assert ast.dump(declaration.returns) == ast.dump(original.returns)
    assert declaration.body[0].value.value is Ellipsis
    assert named(tree, "renamed").args.defaults[0].value == 3
    assert {node.target.id for node in named(tree, "Token").body} == {"ON", "OFF"}
    for namespace, member in (
        ("Vf", "load"),
        ("simt", "atomic_add"),
        ("system", "lock"),
        ("mutex", "lock"),
    ):
        function = named(named(tree, namespace), member)
        assert [ast.unparse(d) for d in function.decorator_list] == ["staticmethod"]
    descriptor = named(tree, "Descriptor")
    assert [node.target.id for node in descriptor.body] == ["shape", "compact"]
    assert descriptor.body[-1].value.value is False
    config = named(named(tree, "pipeline"), "Config")
    assert config.body[0].value.value == 2
    assert str(source) not in generated
    assert generated == api_stub.generate(source)


@pytest.mark.parametrize("class_namespace", [False, True])
def test_namespace_member_aliases_preserve_the_actual_declaration(
    source, class_namespace
):
    initializer = source / "python/pypto_pro/language/__init__.py"
    text = initializer.read_text().replace(
        "SimpleNamespace(lock=_lock)", "SimpleNamespace(lock=system.lock)"
    )
    if class_namespace:
        text = text.replace(
            "from pypto_pro.ir.op import system_ops as system",
            "from ._system_api import System as system",
        )
        (source / "python/pypto_pro/language/_system_api.py").write_text(
            "class System:\n    @staticmethod\n    def lock(*, pipe, event_id: int ="
            " 0):\n        raise RuntimeError('must not execute')\n"
        )
    initializer.write_text(text)
    tree = ast.parse(api_stub.generate(source))
    assert ast.dump(named(named(tree, "mutex"), "lock")) == ast.dump(
        named(named(tree, "system"), "lock")
    )
    rows, coverage = api_stub.public_symbols(source)
    assert coverage["complete"]
    rows = {row["symbol"]: row for row in rows}
    assert rows["pl.system"]["kind"] == "namespace"
    assert rows["pl.mutex.lock"]["reference"] == rows["pl.system.lock"]["reference"]
    path = Path(rows["pl.mutex.lock"]["reference"]["path"])
    assert (
        path.read_text()
        .splitlines()[rows["pl.mutex.lock"]["reference"]["line"] - 1]
        .lstrip()
        .startswith("def lock(")
    )


@pytest.mark.parametrize(
    "value", ["system.missing", "getattr(system, 'lock')", "system.lock.extra"]
)
def test_unknown_namespace_members_are_not_executed_or_invented(source, value):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace("lock=_lock", "lock=" + value)
    )
    with pytest.raises(ValueError):
        api_stub.generate(source)
    rows, coverage = api_stub.public_symbols(source)
    assert (
        not coverage["complete"] and coverage["errors"][0]["symbol"] == "pl.mutex.lock"
    )
    assert "pl.load" in {row["symbol"] for row in rows}
    assert "pl.mutex.lock" not in {row["symbol"] for row in rows}


def test_only_complete_named_stub_declarations_survive_a_later_syntax_error(source):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace('"DEFAULT"]', '"DEFAULT", "Good"]')
        + "\nfrom .broken import Good\n"
    )
    stub = source / "python/pypto_pro/language/broken.pyi"
    stub.write_text(
        "from typing import Final\nclass Good:\n    value: Final[int]\n\nclass"
        " Broken:\n    def incomplete(self):\n    ...\n"
    )
    generated = api_stub.generate(source)
    assert named(ast.parse(generated), "Good").body[0].target.id == "value"
    rows, coverage = api_stub.public_symbols(source)
    assert (
        coverage["complete"]
        and coverage["partial_stubs"][0]["parsed_through_line"] == 4
    )
    good = next(row for row in rows if row["symbol"] == "pl.Good")
    assert good["reference"]["line"] == 2
    assert good["reference"]["sha256"] == feedback.file_identity(stub)["sha256"]
    initializer.write_text(initializer.read_text().replace("Good", "Broken"))
    with pytest.raises(ValueError, match="outside the complete stub prefix"):
        api_stub.generate(source)
    assert not api_stub.public_symbols(source)[1]["complete"]


def test_invalid_executable_source_and_unclosed_stub_tokens_are_not_recovered(source):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace('"DEFAULT"]', '"DEFAULT", "Good"]')
        + "\nfrom .broken import Good\n"
    )
    path = source / "python/pypto_pro/language/broken.py"
    path.write_text(
        "class Good:\n    value: int\nclass Broken:\n    def incomplete(self):\n    ...\n"
    )
    with pytest.raises(SyntaxError):
        api_stub.generate(source)
    path.unlink()
    path.with_suffix(".pyi").write_text(
        "class Good:\n    value: int\nclass Broken:\n    value = (\n"
    )
    with pytest.raises(SyntaxError) as rejected:
        api_stub.generate(source)
    assert isinstance(rejected.value.__cause__, tokenize.TokenError)


def test_partial_stub_cannot_supply_a_complete_namespace(source):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace(
            "from pypto_pro.ir.op import system_ops as system",
            "from . import broken as system",
        )
    )
    (source / "python/pypto_pro/language/broken.pyi").write_text(
        "def lock(*, pipe): ...\nclass Broken:\n    def incomplete(self):\n    ...\n"
    )
    with pytest.raises(ValueError, match="incomplete stub namespace"):
        api_stub.generate(source)
    rows, coverage = api_stub.public_symbols(source)
    assert not coverage["complete"]
    assert "pl.load" in {row["symbol"] for row in rows}
    assert "pl.system.lock" not in {row["symbol"] for row in rows}


@pytest.mark.parametrize(
    "relative,before,after",
    [
        ("python/pypto_pro/language/_api.py", "order=None", "order=[1, 0]"),
        ("python/src/bindings/ir/ir.cpp", '.value("OFF",', '.value("DISABLED",'),
    ],
)
def test_source_drift_fails_check_and_regeneration_repairs_it(
    source, tmp_path, relative, before, after
):
    path = tmp_path / "reference.pyi"
    assert api_stub.report(write=True, pypto_root=source, path=path)
    original = path.read_text()
    changed = source / relative
    changed.write_text(changed.read_text().replace(before, after))
    assert not api_stub.report(pypto_root=source, path=path)
    assert path.read_text() == original
    assert api_stub.report(write=True, pypto_root=source, path=path)
    assert api_stub.report(pypto_root=source, path=path)


def test_missing_or_edited_reference_fails_without_overwriting(source, tmp_path):
    path = tmp_path / "reference.pyi"
    assert not api_stub.report(pypto_root=source, path=path)
    assert not path.exists()
    path.write_text(api_stub.generate(source) + "# manual edit\n")
    assert not api_stub.report(pypto_root=source, path=path)
    assert path.read_text().endswith("# manual edit\n")


def test_an_unresolved_export_fails_before_writing(source, tmp_path):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace('"load", "Token",', '"missing_api", "Token",')
    )
    path = tmp_path / "reference.pyi"
    path.write_text("previous reference\n")
    assert not api_stub.report(write=True, pypto_root=source, path=path)
    assert path.read_text() == "previous reference\n"


def test_current_source_public_exports_and_declared_signatures_are_covered():
    tree = ast.parse(api_stub.generate())
    initializer = ast.parse(
        (PYPTO_ROOT / "python/pypto_pro/language/__init__.py").read_text()
    )
    exports = next(
        ast.literal_eval(node.value)
        for node in initializer.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    names |= {node.targets[0].id for node in tree.body if isinstance(node, ast.Assign)}
    names |= {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert set(exports) <= names
    api_exports = {
        alias.name: alias.asname or alias.name
        for node in initializer.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "_api"
        for alias in node.names
        if (alias.asname or alias.name) in exports
    }
    for filename, source_class, target in (
        ("_api.py", None, tree),
        ("_vf_api.py", "Vf", named(tree, "Vf")),
        ("_simt_api.py", "Simt", named(tree, "simt")),
    ):
        source = ast.parse(
            (PYPTO_ROOT / "python/pypto_pro/language" / filename).read_text()
        )
        body = named(source, source_class).body if source_class else source.body
        for node in body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if source_class is None and node.name not in api_exports:
                    continue
                signature = named(
                    target,
                    api_exports[node.name] if source_class is None else node.name,
                )
                assert ast.dump(signature.args) == ast.dump(node.args), node.name
                assert (ast.dump(signature.returns) if signature.returns else None) == (
                    ast.dump(node.returns) if node.returns else None
                ), node.name


def test_public_objects_preserve_aliases_kinds_and_source_identity_without_importing(
    source,
):
    rows, coverage = api_stub.public_symbols(source)
    assert coverage["complete"] and not coverage["errors"]
    records = {row["symbol"]: row for row in rows}
    assert len(records) == len(rows)
    assert (
        records["pl.system.lock"]["declaration"]
        == "pl.system.lock(*, pipe, event_id: int=0)"
    )
    assert records["vf.load"]["aliases"] == ["pl.Vf.load"]
    assert records["vf"]["kind"] == records["pl.simt"]["kind"] == "namespace"
    assert (
        records["pl.Descriptor"]["kind"] == "type"
        and records["pl.Descriptor"]["callable"] is None
    )
    assert records["pl.Descriptor.shape"]["kind"] == "field"
    assert (
        records["pl.DEFAULT"]["kind"] == "constant"
        and records["pl.DEFAULT"]["callable"] is False
    )
    assert (
        records["pl.Token"]["kind"] == "enum"
        and records["pl.Token.ON"]["kind"] == "enum_member"
    )
    assert not any("FAKE" in name or "_validate" in name for name in records)
    reference = records["pl.Token.ON"]["reference"]
    assert (
        '.value("ON",'
        in Path(reference["path"]).read_text().splitlines()[reference["line"] - 1]
    )
    assert records["pl.Token.ON"]["declaration"] == "ON: Token"
    for row in rows:
        assert (
            row["reference"]["sha256"]
            == feedback.file_identity(Path(row["reference"]["path"]))["sha256"]
        )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_binding_references_use_original_lines_across_multiline_non_code(
    source, newline
):
    text = """/* Header with a fake .value("BAD", Token::Bad);
continued comment. */
py::enum_<Token>(ir, "Token", R"doc(Multiline documentation;
py::enum_<Fake>(ir, "Bogus").value("FAKE", Token::Bad);
)doc")
    .value("ON", Token::On, R"help(Help;
// Not a binding comment.
)help")
    /* Another multiline comment;
    .value("HIDDEN", Token::Bad); */
    .value("OFF", Token::Off);
"""
    path = source / api_stub.BINDINGS
    path.write_bytes(text.replace("\n", newline).encode())
    rows, coverage = api_stub.public_symbols(source)
    assert coverage["complete"]
    records = {row["symbol"]: row for row in rows}
    assert {name for name in records if name.startswith("pl.Token.")} == {
        "pl.Token.ON",
        "pl.Token.OFF",
    }
    lines = path.read_text().splitlines()
    for symbol, declaration in [
        ("pl.Token", "py::enum_<Token>"),
        ("pl.Token.ON", '.value("ON",'),
        ("pl.Token.OFF", '.value("OFF",'),
    ]:
        reference = records[symbol]["reference"]
        assert declaration in lines[reference["line"] - 1]
        assert reference["sha256"] == feedback.file_identity(path)["sha256"]


def test_public_properties_are_fields_and_do_not_look_like_functions(source):
    path = source / "python/pypto_pro/ir/op/block_ops.py"
    path.write_text(
        path.read_text()
        + '\n    @property\n    def rank(self) -> int:\n        raise RuntimeError("must not execute")\n'
    )
    rows, coverage = api_stub.public_symbols(source)
    row = next(row for row in rows if row["symbol"] == "pl.Descriptor.rank")
    assert coverage["complete"] and row["kind"] == "field" and row["callable"] is False
    assert row["declaration"] == "pl.Descriptor.rank: int"


def test_unresolved_and_cyclic_exports_keep_independent_source_results(source):
    initializer = source / "python/pypto_pro/language/__init__.py"
    initializer.write_text(
        initializer.read_text().replace(
            '"load", "Token",', '"missing_api", "loop", "Token",'
        )
        + "\nfrom ._cycle import loop\n"
    )
    (source / "python/pypto_pro/language/_cycle.py").write_text(
        "from ._cycle_other import loop\n"
    )
    (source / "python/pypto_pro/language/_cycle_other.py").write_text(
        "from ._cycle import loop\n"
    )
    rows, coverage = api_stub.public_symbols(source)
    assert not coverage["complete"]
    assert {error["symbol"] for error in coverage["errors"]} == {
        "pl.missing_api",
        "pl.loop",
    }
    assert any("Cyclic" in error["reason"] for error in coverage["errors"])
    assert "pl.Token.ON" in {row["symbol"] for row in rows}


def test_unreadable_binding_enum_does_not_turn_other_exports_into_missing_apis(source):
    (source / api_stub.BINDINGS).write_text('py::enum_<Token>(ir, "Token");\n')
    rows, coverage = api_stub.public_symbols(source)
    assert not coverage["complete"] and any(
        "C++ enum" in item["reason"] for item in coverage["errors"]
    )
    assert "pl.load" in {row["symbol"] for row in rows}


@pytest.mark.parametrize("change", ["same_timestamp", "root"])
def test_cursor_is_bound_to_fresh_declarations_and_selected_root(
    source, tmp_path, change
):
    page = query.report(kinds=["api"], limit=1, pypto_root=source)["knowledge"]
    assert page["page"]["next_cursor"]
    selected = source
    if change == "root":
        selected = tmp_path / "copied"
        shutil.copytree(source / "python", selected / "python")
    else:
        path = source / "python/pypto_pro/language/_api.py"
        stamp = path.stat()
        path.write_text(path.read_text().replace("value: int = 3", "value: int = 4"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        rows, _ = api_stub.public_symbols(source)
        assert (
            next(row for row in rows if row["symbol"] == "pl.renamed")["declaration"]
            == "pl.renamed(value: int=4)"
        )
    with pytest.raises(ValueError, match="cursor"):
        query.report(
            kinds=["api"], cursor=page["page"]["next_cursor"], pypto_root=selected
        )


def test_current_nested_symbols_retain_real_declaration_lines():
    rows, coverage = api_stub.public_symbols()
    assert coverage["complete"]
    row = next(row for row in rows if row["symbol"] == "pl.system.bar_all")
    text = (
        Path(row["reference"]["path"])
        .read_text()
        .splitlines()[row["reference"]["line"] - 1]
    )
    assert (
        text.lstrip().startswith("def bar_all(")
        and "def bar_all(" not in row["declaration"]
    )
