# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate the agent-readable public language reference without importing PyPTO."""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from pathlib import Path

from .paths import PYPTO_ROOT, REFERENCES_ROOT

STUB_PATH = REFERENCES_ROOT / "api/pypto_pro_language.pyi"
LANGUAGE = "pypto_pro.language"
BINDINGS = "python/src/bindings/ir/ir.cpp"
SAFE_DECORATORS = {
    "staticmethod",
    "classmethod",
    "property",
    "overload",
    "dataclass",
    "contextmanager",
}


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    return []


class _Sources:
    def __init__(self, root: Path):
        self.root = root
        self.payloads: dict[str, bytes] = {}
        self.trees: dict[str, ast.Module] = {}
        self.enums: dict[str, ast.ClassDef] | None = None

    def read(self, relative: str) -> str:
        if relative not in self.payloads:
            self.payloads[relative] = (self.root / relative).read_bytes()
        return self.payloads[relative].decode("utf-8")

    def path(self, module: str) -> str:
        stem = "python/" + module.replace(".", "/")
        for relative in (stem + ".py", stem + "/__init__.py", stem + ".pyi"):
            if (self.root / relative).is_file():
                return relative
        raise ValueError(
            f"No source for {module}; select a complete PyPTO source tree with --pypto-root"
        )

    def tree(self, relative: str) -> ast.Module:
        if relative not in self.trees:
            self.trees[relative] = ast.parse(self.read(relative), filename=relative)
        return self.trees[relative]

    @staticmethod
    def import_module(module: str, relative: str, node: ast.ImportFrom) -> str:
        package = (
            module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        )
        if node.level:
            prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
            return ".".join(prefix + ([node.module] if node.module else []))
        return node.module or ""

    def binding_enums(self) -> dict[str, ast.ClassDef]:
        if self.enums is None:
            text = self.read(BINDINGS)
            # Raw docstrings and comments must not terminate the binding statement.
            text = re.sub(r'R"(\w*)\([\s\S]*?\)\1"', '""', text)
            text = re.sub(
                r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                lambda m: m[0].replace(";", " ") if m[0].startswith('"') else "",
                text,
            )
            self.enums = {}
            for match in re.finditer(r'py::enum_<[^>]+>\s*\(\s*\w+,\s*"(\w+)"', text):
                name = match[1]
                body = text[slice(match.end(), None)].split(";", 1)[0]
                members = re.findall(r'\.value\(\s*"(\w+)"\s*,', body)
                if not members:
                    raise ValueError(f"Cannot read C++ enum members for {name}")
                self.enums[name] = ast.parse(
                    "class "
                    + name
                    + ":\n"
                    + "".join(f"    {member}: {name}\n" for member in members)
                ).body[0]
        return self.enums

    def resolve(self, module: str, name: str) -> tuple[str, str, ast.AST]:
        if module == "pypto.pypto_impl.ir" and name in self.binding_enums():
            return module, BINDINGS, self.binding_enums()[name]
        relative = self.path(module)
        tree = self.tree(relative)
        for node in tree.body:
            if name in _names(node):
                return module, relative, node
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            origin = self.import_module(module, relative, node)
            for alias in node.names:
                if (alias.asname or alias.name) == name or alias.name == "*":
                    if node.module is None:
                        child = origin + "." + alias.name
                        relative = self.path(child)
                        return child, relative, self.tree(relative)
                    return self.resolve(
                        origin, name if alias.name == "*" else alias.name
                    )
        child = module + "." + name
        relative = self.path(child)
        return child, relative, self.tree(relative)


def _declaration(
    node: ast.AST, name: str | None = None, *, static: bool = False
) -> ast.AST:
    """Keep declared parameters, defaults and fields; discard executable bodies."""
    node = copy.deepcopy(node)
    if name is not None and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        node.name = name
    elif name is not None and isinstance(node, ast.AnnAssign):
        node.target = ast.Name(id=name, ctx=ast.Store())
    elif name is not None and isinstance(node, ast.Assign):
        node.targets = [ast.Name(id=name, ctx=ast.Store())]
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        node.decorator_list = [
            decorator
            for decorator in node.decorator_list
            if (ast.unparse(decorator).split("(", 1)[0] in SAFE_DECORATORS)
        ]
    if isinstance(node, ast.FunctionDef):
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        if static and not any(
            ast.unparse(d) == "staticmethod" for d in node.decorator_list
        ):
            node.decorator_list.insert(0, ast.Name(id="staticmethod", ctx=ast.Load()))
    elif isinstance(node, ast.ClassDef):
        node.body = [
            _declaration(member)
            for member in node.body
            if _names(member)
            and any(not n.startswith("_") or n.startswith("__") for n in _names(member))
        ]
        node.body = node.body or [ast.Expr(value=ast.Constant(value=Ellipsis))]
    return ast.fix_missing_locations(node)


def _exports(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if "__all__" in _names(node):
            return ast.literal_eval(node.value)
    return [
        name for node in tree.body for name in _names(node) if not name.startswith("_")
    ]


def generate(pypto_root: Path = PYPTO_ROOT) -> str:
    sources = _Sources(pypto_root)
    root = sources.path(LANGUAGE)
    exports = _exports(sources.tree(root))
    if len(exports) != len(set(exports)):
        raise ValueError("pypto_pro.language.__all__ contains duplicate exports")
    declarations: list[tuple[str, ast.AST]] = []

    def add(
        module: str, name: str, public_name: str | None = None, *, static: bool = False
    ) -> ast.AST:
        owner, relative, node = sources.resolve(module, name)
        target = public_name or name
        if isinstance(node, ast.Module):
            members = [add(owner, member, static=True) for member in _exports(node)]
            return ast.ClassDef(
                name=target, bases=[], keywords=[], body=members, decorator_list=[]
            )
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "SimpleNamespace"
        ):
            members = [
                add(owner, keyword.value.id, keyword.arg, static=True)
                for keyword in node.value.keywords
            ]
            return ast.ClassDef(
                name=target, bases=[], keywords=[], body=members, decorator_list=[]
            )
        return _declaration(node, target, static=static)

    # Source aliases used in signatures remain as weak as the declarations.
    for name in ("Tile", "TileGroup", "DType", "Offset"):
        owner, relative, _ = sources.resolve(LANGUAGE + "._api", name)
        declarations.append((relative, add(owner, name)))
    for name in exports:
        if name == "parser":
            declarations.append(
                (
                    root,
                    ast.parse("from pypto_pro.language import parser as parser").body[
                        0
                    ],
                )
            )
            continue
        _, relative, _ = sources.resolve(LANGUAGE, name)
        declarations.append((relative, add(LANGUAGE, name)))

    # Import only dependencies referenced by the retained signatures/defaults.
    tree = ast.fix_missing_locations(
        ast.Module(body=[node for _, node in declarations], type_ignores=[])
    )
    defined = set(exports) | {"Tile", "TileGroup", "DType", "Offset"}
    needed = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - defined
    imports = {}
    for relative, source_tree in sources.trees.items():
        module = (
            relative.removeprefix("python/").removesuffix(".pyi").removesuffix(".py")
        )
        module = module.removesuffix("/__init__").replace("/", ".")
        for node in source_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound not in needed:
                        continue
                    retained = copy.deepcopy(node)
                    retained.names = [alias]
                    if isinstance(retained, ast.ImportFrom):
                        retained.module = sources.import_module(
                            module, relative, retained
                        )
                        retained.level = 0
                    imports.setdefault(bound, ast.unparse(retained))
            else:
                for name in set(_names(node)) & needed:
                    imports.setdefault(name, f"from {module} import {name}")

    fingerprint = hashlib.sha256()
    for relative, payload in sorted(sources.payloads.items()):
        fingerprint.update(relative.encode() + b"\0" + payload + b"\0")
    header = [
        "# Generated agent reference for import pypto_pro.language as pl.",
        "# Read top-level names as pl.<name>; Vf is pl.Vf (often imported as vf).",
        "# Snapshot of source declarations; Any and missing annotations are retained.",
        "# This file is for lookup, not installed IDE stubs or proof of device support.",
        "# Source SHA256 (ordered paths and contents): " + fingerprint.hexdigest(),
        "# Regenerate: python scripts/check.py api --write-pyi",
        "# Verify against the selected source: python scripts/check.py api --check-pyi",
        "# Select a different source with --pypto-root before the api subcommand.",
        "# Contracts: ../index.md, constraints.md, memory-allocation.md, vf.md, simt.md,",
        "# gemm.md, datamove.md, crosscore-sync.md, fixpipe.md and support.md.",
        "# Enum member names come from Python bindings; numeric encodings are omitted.",
        "",
        *sorted(imports.values()),
        "",
    ]
    previous = None
    for relative, node in declarations:
        if relative != previous:
            header.extend(["", "# Source: " + relative])
            previous = relative
        text = ast.unparse(ast.fix_missing_locations(node))
        text = re.sub(r"(?m)^(\s*def [^\n]+:)\n\s+\.\.\.$", r"\1 ...", text)
        header.append("\n".join(line for line in text.splitlines() if line.strip()))
    result = "\n".join(header) + "\n"
    ast.parse(result, filename=str(STUB_PATH))
    return result


def report(
    *, write: bool = False, pypto_root: Path = PYPTO_ROOT, path: Path = STUB_PATH
) -> bool:
    try:
        generated = generate(pypto_root)
        if write:
            path.write_text(generated, encoding="utf-8")
            print(f"api：已生成 {path}")
            return True
        if path.read_text(encoding="utf-8") == generated:
            print(f"api：{path.name} 与所选源码一致")
            return True
        print(f"api：{path.name} 已过期，请运行 api --write-pyi")
    except (OSError, SyntaxError, ValueError) as error:
        print(f"api：{error}")
    return False
