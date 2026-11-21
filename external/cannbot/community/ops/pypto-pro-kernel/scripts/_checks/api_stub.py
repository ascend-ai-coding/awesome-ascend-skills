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
import io
import re
import tokenize
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


class SourceUnavailable(ValueError):
    """A required file is outside the selected source tree's available scope."""


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
        self.resolved: dict[tuple[str, str], tuple[str, str, ast.AST]] = {}
        self.resolving: set[tuple[str, str]] = set()
        self.partial_stubs: dict[str, dict] = {}

    @staticmethod
    def import_module(module: str, relative: str, node: ast.ImportFrom) -> str:
        package = (
            module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        )
        if node.level:
            prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
            return ".".join(prefix + ([node.module] if node.module else []))
        return node.module or ""

    def read(self, relative: str) -> str:
        if relative not in self.payloads:
            self.payloads[relative] = (self.root / relative).read_bytes()
        return self.payloads[relative].decode("utf-8")

    def path(self, module: str) -> str:
        stem = "python/" + module.replace(".", "/")
        for relative in (stem + ".py", stem + "/__init__.py", stem + ".pyi"):
            if (self.root / relative).is_file():
                return relative
        raise SourceUnavailable(
            f"No source for {module}; select a complete PyPTO source tree with --pypto-root"
        )

    def tree(self, relative: str) -> ast.Module:
        if relative not in self.trees:
            text = self.read(relative)
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError as error:
                if not relative.endswith(".pyi"):
                    raise
                tree = self._stub_prefix(relative, text, error)
            self.trees[relative] = tree
        return self.trees[relative]

    def binding_enums(self) -> dict[str, ast.ClassDef]:
        if self.enums is None:
            text = self.read(BINDINGS)

            # Mask non-code without moving the original binding lines.
            def mask(raw_match):
                return re.sub(r"[^\n]", " ", raw_match[0])

            text = re.sub(r'R"(\w*)\([\s\S]*?\)\1"', mask, text)
            text = re.sub(
                r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                lambda m: m[0].replace(";", " ") if m[0].startswith('"') else mask(m),
                text,
            )
            self.enums = {}
            for match in re.finditer(r'py::enum_<[^>]+>\s*\(\s*\w+,\s*"(\w+)"', text):
                name = match[1]
                body = text[match.end():].split(";", 1)[0]
                members = list(re.finditer(r'\.value\(\s*"(\w+)"\s*,', body))
                if not members:
                    raise ValueError(f"Cannot read C++ enum members for {name}")
                node = ast.parse(
                    "class "
                    + name
                    + ":\n"
                    + "".join(f"    {member[1]}: {name}\n" for member in members)
                ).body[0]
                node._binding_line = text.count("\n", 0, match.start()) + 1
                for field, member in zip(node.body, members):
                    field._binding_line = (
                        text.count("\n", 0, match.end() + member.start()) + 1
                    )
                self.enums[name] = node
        return self.enums

    def resolve(self, module: str, name: str) -> tuple[str, str, ast.AST]:
        key = module, name
        if key in self.resolving:
            raise ValueError(f"Cyclic source export: {module}.{name}")
        if key not in self.resolved:
            self.resolving.add(key)
            try:
                self.resolved[key] = self._resolve(module, name)
            finally:
                self.resolving.remove(key)
        return self.resolved[key]

    def resolve_reference(
        self, module: str, node: ast.expr
    ) -> tuple[str, str, ast.AST]:
        """Resolve a declared name or module/class member without evaluating it."""
        if isinstance(node, ast.Name):
            return self.resolve(module, node.id)
        if isinstance(node, ast.Attribute):
            owner, relative, parent = self.resolve_reference(module, node.value)
            if isinstance(parent, ast.Module):
                return self.resolve(owner, node.attr)
            if isinstance(parent, ast.ClassDef):
                members = [member for member in parent.body if node.attr in _names(member)]
                if len(members) == 1:
                    return owner, relative, members[0]
        raise ValueError(f"Unresolved declaration reference: {ast.unparse(node)}")

    def _stub_prefix(self, relative, text, error):
        """Keep only complete declarations before a malformed top-level stub block."""
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError) as token_error:
            raise error from token_error
        cuts = {
            token.start[0]
            for token in tokens
            if token.start[1] == 0
            and token.start[0] < (error.lineno or 0)
            and (
                (token.type == tokenize.NAME and token.string in {"class", "def"})
                or (token.type == tokenize.OP and token.string == "@")
            )
        }
        lines = text.splitlines(keepends=True)
        for cut in sorted(cuts, reverse=True):
            try:
                prefix = ast.parse("".join(lines[: cut - 1]), filename=relative)
            except SyntaxError:
                continue
            self.partial_stubs[relative] = {
                "path": relative,
                "parsed_through_line": cut - 1,
                "reason": str(error),
            }
            return prefix
        raise error

    def _resolve(self, module: str, name: str) -> tuple[str, str, ast.AST]:
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
        if relative in self.partial_stubs:
            raise SourceUnavailable(
                f"{module}.{name} is outside the complete stub prefix: "
                + self.partial_stubs[relative]["reason"]
            )
        child = module + "." + name
        relative = self.path(child)
        return child, relative, self.tree(relative)


def public_symbols(pypto_root: Path = PYPTO_ROOT):
    """Read exported declarations and members using the snapshot's resolver.

    No framework imports or expression evaluation are performed. C++ enum
    records retain original binding-file lines, not synthetic AST line numbers.
    Unresolved exports are reported independently of the resolved declarations.
    """
    from .api import _signature

    sources = _Sources(Path(pypto_root))
    records, errors = [], []

    def record(public, relative, node, kind, *, parent=None, declaration=None):
        payload = sources.payloads[relative]
        reference = {
            "path": str((sources.root / relative).resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        line = (
            getattr(node, "_binding_line", None)
            if relative == BINDINGS
            else getattr(node, "lineno", None)
        )
        if line is not None:
            reference["line"] = line
        aliases = (
            ["pl.Vf" + public[2:]] if public == "vf" or public.startswith("vf.") else []
        )
        doc = (
            ast.get_docstring(node)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
            else None
        )
        row = {
            "symbol": public,
            "kind": kind,
            "aliases": aliases,
            "reference": reference,
            "declaration": declaration,
            "summary": doc.splitlines()[0] if doc else "",
            "callable": kind == "api" if kind != "type" and kind != "enum" else None,
            "parent": parent,
        }
        if doc:
            literal = node.body[0]
            row["_docstring"] = {
                "text": ast.get_source_segment(payload.decode("utf-8"), literal),
                "reference": {
                    **reference,
                    "line": literal.lineno,
                    "end_line": literal.end_lineno,
                },
            }
        records.append(row)

    def walk(public, owner, relative, node, *, parent=None, ancestors=(), enum=False):
        identity = owner, relative, id(node)
        if identity in ancestors:
            raise ValueError(f"Cyclic public namespace: {public}")
        ancestors = (*ancestors, identity)
        leaf = public.rsplit(".", 1)[-1]
        if isinstance(node, ast.Module):
            if relative in sources.partial_stubs:
                raise SourceUnavailable(
                    f"Cannot enumerate an incomplete stub namespace: {relative}"
                )
            record(public, relative, node, "namespace", parent=parent)
            if public == "pl.parser":
                return
            for member in _exports(node):
                expand(
                    public + "." + member,
                    owner,
                    member,
                    parent=public,
                    ancestors=ancestors,
                )
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "SimpleNamespace"
        ):
            record(public, relative, node, "namespace", parent=parent)
            for keyword in node.value.keywords:
                if keyword.arg is None:
                    raise ValueError(f"Unresolved namespace member in {public}")
                expand(
                    public + "." + keyword.arg,
                    owner,
                    keyword.value,
                    parent=public,
                    ancestors=ancestors,
                )
        elif isinstance(node, ast.ClassDef):
            bound_enum = relative == BINDINGS
            kind = (
                "enum"
                if bound_enum
                else "namespace"
                if public in {"vf", "pl.simt", "pl.system"}
                else "type"
            )
            declaration = (
                "class "
                + leaf
                + (
                    "(" + ", ".join(map(ast.unparse, node.bases)) + ")"
                    if node.bases
                    else ""
                )
            )
            record(public, relative, node, kind, parent=parent, declaration=declaration)
            for member in node.body:
                for name in _names(member):
                    if not name.startswith("_"):
                        walk(
                            public + "." + name,
                            owner,
                            relative,
                            member,
                            parent=public,
                            ancestors=ancestors,
                            enum=bound_enum,
                        )
        elif isinstance(node, ast.FunctionDef):
            property_field = any(
                ast.unparse(d) == "property" for d in node.decorator_list
            )
            signature = (
                public + (": " + ast.unparse(node.returns) if node.returns else "")
                if property_field
                else public.rpartition(".")[0]
                + "."
                + _signature(_declaration(node, leaf))
            )
            record(
                public,
                relative,
                node,
                "field" if property_field else "api",
                parent=parent,
                declaration=signature,
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            kind = (
                "enum_member"
                if enum
                else "field"
                if parent
                and parent not in {"pl", "vf", "pl.simt"}
                and isinstance(node, ast.AnnAssign)
                else "constant"
            )
            record(
                public,
                relative,
                node,
                kind,
                parent=parent,
                declaration=ast.unparse(_declaration(node, leaf)),
            )
        else:
            raise ValueError(f"Unsupported public declaration: {public}")

    def expand(public, module, name, **kwargs):
        try:
            owner, relative, node = (
                sources.resolve_reference(module, name)
                if isinstance(name, ast.expr)
                else sources.resolve(module, name)
            )
            walk(public, owner, relative, node, **kwargs)
        except (
            OSError,
            SyntaxError,
            ValueError,
            RecursionError,
        ) as error:
            errors.append({"symbol": public, "reason": str(error)})

    try:
        root = sources.path(LANGUAGE)
        exports = _exports(sources.tree(root))
        if root in sources.partial_stubs:
            raise SourceUnavailable(
                f"Cannot enumerate an incomplete language stub: {root}"
            )
        if (
            not isinstance(exports, list)
            or not all(isinstance(name, str) for name in exports)
            or len(set(exports)) != len(exports)
        ):
            raise ValueError("Public exports must be unique strings")
        record("pl", root, sources.tree(root), "namespace")
        for name in exports:
            expand("vf" if name == "Vf" else "pl." + name, LANGUAGE, name, parent="pl")
    except (OSError, SyntaxError, ValueError) as error:
        errors.append({"symbol": "pl", "reason": str(error)})
    files = {
        relative: hashlib.sha256(data).hexdigest()
        for relative, data in sorted(sources.payloads.items())
    }
    try:
        changed = any(
            hashlib.sha256((sources.root / relative).read_bytes()).hexdigest() != digest
            for relative, digest in files.items()
        )
    except OSError:
        changed = True
    if changed:
        return [], {
            "files": files,
            "complete": False,
            "errors": [
                {
                    "symbol": "pl",
                    "reason": "Selected API sources changed during discovery",
                }
            ],
        }
    return sorted(records, key=lambda item: item["symbol"]), {
        "files": files,
        "complete": not errors,
        "errors": errors,
        "partial_stubs": list(sources.partial_stubs.values()),
    }


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
    if root in sources.partial_stubs:
        raise SourceUnavailable(f"Cannot enumerate an incomplete language stub: {root}")
    if len(exports) != len(set(exports)):
        raise ValueError("pypto_pro.language.__all__ contains duplicate exports")
    declarations: list[tuple[str, ast.AST]] = []

    def add(
        module: str,
        name: str | ast.expr,
        public_name: str | None = None,
        *,
        static: bool = False,
    ) -> ast.AST:
        owner, relative, node = (
            sources.resolve_reference(module, name)
            if isinstance(name, ast.expr)
            else sources.resolve(module, name)
        )
        target = public_name or name
        if isinstance(node, ast.Module):
            if relative in sources.partial_stubs:
                raise SourceUnavailable(
                    f"Cannot enumerate an incomplete stub namespace: {relative}"
                )
            members = [add(owner, member, static=True) for member in _exports(node)]
            return ast.ClassDef(
                name=target, bases=[], keywords=[], body=members, decorator_list=[]
            )
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "SimpleNamespace"
        ):
            if any(keyword.arg is None for keyword in node.value.keywords):
                raise ValueError(f"Unresolved namespace member in {target}")
            members = [
                add(owner, keyword.value, keyword.arg, static=True)
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
    license_header = []
    for line in Path(__file__).read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        license_header.append(line)
    header = [
        *license_header,
        "",
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
        *[
            f"# Partial stub: {item['path']}; complete prefix through line"
            f" {item['parsed_through_line']}; later declarations unverified."
            for item in sources.partial_stubs.values()
        ],
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


def _reference_matches(current: str, generated: str) -> bool:
    license_notice = generated.split("# Generated agent reference", 1)[0].strip()

    def comments(text):
        return [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT
        ]

    try:
        return (
            current.startswith(license_notice)
            and comments(current) == comments(generated)
            and ast.dump(ast.parse(current), include_attributes=False)
            == ast.dump(ast.parse(generated), include_attributes=False)
        )
    except (SyntaxError, tokenize.TokenError):
        return False


def inspect_reference(*, pypto_root: Path = PYPTO_ROOT, path: Path = STUB_PATH):
    """Share freshness checks and consumed identities without confusing source failures with stale output."""
    result = {"status": "unverified", "source": str(path.resolve())}
    try:
        generated = generate(pypto_root)
    except (OSError, SourceUnavailable) as error:
        return {
            **result,
            "reason": "selected_source_unavailable",
            "message": str(error),
        }
    except (SyntaxError, ValueError) as error:
        return {
            **result,
            "status": "failed",
            "reason": "selected_source_invalid",
            "message": str(error),
        }
    result["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        return {
            **result,
            "status": "failed",
            "reason": "reference_missing",
            "message": str(error),
        }
    except OSError as error:
        return {**result, "reason": "reference_unavailable", "message": str(error)}
    result["reference"] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        current = data.decode("utf-8")
    except UnicodeError as error:
        return {
            **result,
            "status": "failed",
            "reason": "reference_invalid",
            "message": str(error),
        }
    if _reference_matches(current, generated):
        return {
            **result,
            "status": "passed",
            "reason": "source_matches",
            "message": f"{path.name} 与所选源码一致",
        }
    return {
        **result,
        "status": "failed",
        "reason": "reference_stale",
        "message": f"{path.name} 已过期，请运行 api --write-pyi",
    }


def report(
    *, write: bool = False, pypto_root: Path = PYPTO_ROOT, path: Path = STUB_PATH
) -> bool:
    if not write:
        result = inspect_reference(pypto_root=pypto_root, path=path)
        print("api：" + result["message"])
        return result["status"] == "passed"
    try:
        generated = generate(pypto_root)
        path.write_text(generated, encoding="utf-8")
        print(f"api：已生成 {path}")
        return True
    except (OSError, SyntaxError, ValueError) as error:
        print(f"api：{error}")
    return False
