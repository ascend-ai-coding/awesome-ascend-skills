# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public-surface fidelity and freshness gates for the generated agent reference."""

import ast

import pytest

from _checks import api_stub
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
                signature = named(target, node.name)
                assert ast.dump(signature.args) == ast.dump(node.args), node.name
                assert (ast.dump(signature.returns) if signature.returns else None) == (
                    ast.dump(node.returns) if node.returns else None
                ), node.name
