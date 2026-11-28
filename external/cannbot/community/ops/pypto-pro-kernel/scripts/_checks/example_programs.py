# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Discover native example programs from the catalog and static source snapshots."""

from __future__ import annotations

import ast
from collections import Counter
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from . import (
    api,
    catalog,
    coverage,
    feedback,
    host_io,
    knowledge,
    links,
    query_metadata,
    source,
)

GROUPS = ("basic_api", *catalog.GROUPS)
ENTRIES = ("kernel", "reference", "test")
_META = re.compile(r"^[ \t]*example-query:[ \t]*(.*)$", re.MULTILINE)
_DISCOVERY_WORDS = {
    "find",
    "show",
    "locate",
    "example",
    "examples",
    "program",
    "programs",
    "kernel",
    "implementation",
    "independent",
    "reference",
    "references",
    "test",
    "tests",
    "source",
    "code",
    "please",
}


def locator(text, root):
    if not text.startswith("example:"):
        return None
    value = text.removeprefix("example:")
    relative = unquote(value)
    parts = PurePosixPath(relative)
    if (
        not relative
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in relative
        or "\x00" in relative
        or parts.parts[0] not in GROUPS
        or "#" in value
    ):
        raise ValueError(
            "Example IDs require a public group and a relative program directory or kernel.py"
        )
    if parts.name != "kernel.py":
        if parts.suffix == ".py":
            raise ValueError(
                "Resolve a program by its kernel.py ID; use the returned read requests for other roles"
            )
        parts /= "kernel.py"
    if any(part.startswith(("_", ".")) for part in parts.parts):
        raise ValueError("Example IDs cannot select private support packages")
    path = (root / parts.as_posix()).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Example locator escapes its selected root")
    return "example:" + quote(parts.as_posix(), safe="/"), path


class Snapshots:
    """One bounded, query-local cache; identities always derive from current bytes."""

    def __init__(self, root):
        self.root = root.resolve()
        self.files, self.failed, self.errors, self.imports, self.identities = (
            {},
            set(),
            [],
            {},
            {},
        )
        self.scanned = 0
        self.limits = {
            name: host_io.CEILINGS[name]
            for name in ("max_files", "max_scan_bytes", "max_file_bytes")
        }

    def error(self, path, reason, detail="", *, line=1):
        item = {
            "path": str(path),
            "line": line,
            "reason": reason,
            "detail": detail[:200],
        }
        if item not in self.errors:
            self.errors.append(item)

    def read(self, path, *, within=None):
        path = Path(path)
        try:
            path = path.resolve()
            if path in self.failed:
                return None
            if within and not path.resolve().is_relative_to(within.resolve()):
                raise ValueError("Example source escapes its selected root")
            if path in self.files:
                return self.files[path]
            if len(set(self.identities) | self.failed) >= self.limits["max_files"]:
                raise ValueError("example_file_scan_limit")
            data = host_io._bytes(
                path,
                min(
                    self.limits["max_file_bytes"],
                    self.limits["max_scan_bytes"] - self.scanned,
                ),
            )
            self.scanned += len(data)
            self.identities[path] = feedback.digest(data)
            text = data.decode("utf-8")
            if "\x00" in text:
                raise ValueError("Example source contains binary data")
            row = {
                "reference": {
                    "path": str(path.resolve()),
                    "sha256": feedback.digest(data),
                    "line": 1,
                },
                "text": text,
            }
            if path.suffix == ".py":
                row["source"] = source.parse(path, text)
            self.files[path] = row
            return row
        except (OSError, RuntimeError, SyntaxError, ValueError) as error:
            self.failed.add(path)
            self.error(path, type(error).__name__, str(error))
            return None


def _literal(node):
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _reference(snapshot, node=None):
    return {
        **snapshot["reference"],
        **({"line": node.lineno, "end_line": node.end_lineno} if node else {}),
    }


def _docstring(snapshot):
    tree = snapshot["source"].tree
    text = ast.get_docstring(tree, clean=False) or ""
    return (
        {
            "text": _META.sub("", text).strip(),
            "reference": _reference(snapshot, tree.body[0]),
        }
        if text
        else None
    )


def _metadata(snapshot):
    text = ast.get_docstring(snapshot["source"].tree, clean=False) or ""
    matches = _META.findall(text)
    if len(matches) > 1:
        raise ValueError("A kernel module may declare only one example-query object")
    return query_metadata.validate(
        query_metadata.loads(matches[0]) if matches else {}, kind="example"
    )


def _module_path(base):
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imports(snapshot, root):
    """Conservative local import closure and dependency observations; never load a module."""
    local, external, environment, unresolved = set(), set(), set(), []
    tree, path = snapshot["source"].tree, Path(snapshot["reference"]["path"])
    aliases = {}

    def absolute(name):
        for prefix in ("pro_agentkit.examples.", "examples."):
            if name.startswith(prefix):
                return root.joinpath(*name[len(prefix):].split("."))
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
                base = absolute(item.name)
                if base:
                    resolved = _module_path(base)
                    if resolved:
                        local.add(resolved)
                    else:
                        unresolved.append(
                            {
                                "module": item.name,
                                "reason": "local_import_unresolved",
                                "reference": _reference(snapshot, node),
                            }
                        )
                else:
                    external.add(item.name)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            for item in node.names:
                aliases[item.asname or item.name] = name + "." + item.name
            base = (
                path.parent.joinpath(*([".."] * (node.level - 1)), *name.split("."))
                if node.level
                else absolute(name)
            )
            if base is None:
                external.add(name)
                continue
            if not base.resolve().is_relative_to(root):
                unresolved.append(
                    {
                        "module": "." * node.level + name,
                        "reason": "outside_examples_root",
                        "reference": _reference(snapshot, node),
                    }
                )
                continue
            resolved = _module_path(base)
            children = {
                _module_path(base / item.name)
                for item in node.names
                if item.name != "*"
            }
            local.update(child for child in children if child)
            if resolved:
                local.add(resolved)
            elif not any(children):
                unresolved.append(
                    {
                        "module": "." * node.level + name,
                        "reason": "local_import_unresolved",
                        "reference": _reference(snapshot, node),
                    }
                )

    def canonical(node):
        value = source.attribute_path(node)
        for alias in sorted(aliases, key=len, reverse=True):
            if value == alias or value and value.startswith(alias + "."):
                return aliases[alias] + value[len(alias):]
        return value

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and canonical(node.value) == "os.environ":
            key = _literal(node.slice)
            if key:
                environment.add(key)
        if not isinstance(node, ast.Call):
            continue
        name = canonical(node.func)
        value = _literal(node.args[0]) if node.args else None
        if name in {"os.getenv", "os.environ.get"} and value:
            environment.add(value)
        if name == "pytest.importorskip" and value:
            external.add(value)
        if name in {"importlib.import_module", "__import__"}:
            base = absolute(value) if value else None
            resolved = _module_path(base) if base else None
            if resolved:
                local.add(resolved)
            elif value and not value.startswith("."):
                external.add(value)
            else:
                unresolved.append(
                    {
                        "module": value,
                        "reason": "dynamic_import",
                        "reference": _reference(snapshot, node),
                    }
                )
        elif name in {"importlib.util.spec_from_file_location", "exec", "eval"}:
            unresolved.append(
                {
                    "module": None,
                    "reason": "dynamic_source",
                    "reference": _reference(snapshot, node),
                }
            )
    return local, external, environment, unresolved


def _parents(path, root):
    current = path.parent
    while current.is_relative_to(root):
        if (current / "__init__.py").is_file():
            yield current / "__init__.py"
        if current == root:
            break
        current = current.parent


def _program(folder, snapshots):
    entries, role_files, external, environment, unresolved = {}, {}, set(), set(), []
    for role in ENTRIES:
        path = folder / (role + ".py")
        snapshot = snapshots.read(path, within=snapshots.root)
        if snapshot:
            entries[role] = snapshot
        pending, seen = [path], set()
        if role == "test":
            pending.extend(
                parent / "conftest.py"
                for parent in (folder, *folder.parents)
                if parent.is_relative_to(snapshots.root)
                and (parent / "conftest.py").is_file()
            )
        while pending:
            path = pending.pop().resolve()
            if path in seen:
                continue
            seen.add(path)
            snapshot = snapshots.read(path, within=snapshots.root)
            if not snapshot:
                continue
            role_files.setdefault(path, set()).add(role)
            if path not in snapshots.imports:
                snapshots.imports[path] = _imports(snapshot, snapshots.root)
            local, modules, variables, unknown = snapshots.imports[path]
            external.update(modules)
            environment.update(variables)
            unresolved.extend(unknown)
            for observation in unknown:
                if observation["reason"] in {
                    "local_import_unresolved",
                    "outside_examples_root",
                }:
                    snapshots.error(
                        path, observation["reason"], str(observation["module"])
                    )
            pending.extend(sorted(local | set(_parents(path, snapshots.root))))
    return entries, role_files, external, environment, unresolved


def collect(root, references_root, entries, document_records):
    standard_modules = getattr(sys, "stdlib_module_names", None)
    if standard_modules is None:
        raise knowledge.QueryError(
            "source_unavailable",
            "example_runtime_unavailable",
            "Complete example discovery requires sys.stdlib_module_names (Python 3.10+). "
            "Repeat this read-only query with a capable interpreter; API absence was not established.",
        )
    root, references_root = root.resolve(), references_root.resolve()
    snapshots, records, inventory = Snapshots(root), [], []
    for group in GROUPS:
        if not (root / group).is_dir() or not (root / group).resolve().is_relative_to(
            root
        ):
            snapshots.error(root / group, "missing_or_escaped_example_group")
            continue
        for path in catalog.public_kernels(
            root,
            (group,),
            onerror=lambda error: snapshots.error(error.filename, type(error).__name__),
        ):
            if len(inventory) >= snapshots.limits["max_files"]:
                snapshots.error(root, "example_inventory_limit")
                break
            inventory.append(path)
    if not inventory:
        snapshots.error(root, "no_public_example_programs")
    catalog_path = references_root / "tooling/examples.md"
    catalog_source = snapshots.read(catalog_path, within=references_root)
    selection = {}
    if catalog_source:
        selection, findings = catalog.selection(
            root,
            catalog_path,
            catalog_source["text"],
            {
                path.parent.relative_to(root)
                for path in inventory
                if path.relative_to(root).parts[0] in catalog.GROUPS
            },
        )
        for finding in findings:
            snapshots.error(finding.path, finding.rule, finding.message)
    test_config = (
        snapshots.read(root / "pytest.ini", within=root)
        if (root / "pytest.ini").is_file()
        else None
    )
    contract_paths = {}
    for meta, _, _ in entries:
        if meta.get("example"):
            contract_paths.setdefault((root / meta["example"]).resolve(), []).append(
                "contract:" + meta["id"]
            )
    documentation = {}
    for document in document_records:
        if document.get("document_role") != "skill_reference":
            continue
        for link in links._LINK.finditer(document["_text"]):
            target = urlsplit(link["target"].strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            destination = (
                Path(document["reference"]["path"]).parent / unquote(target.path)
            ).resolve()
            if destination.is_relative_to(root):
                line = document["_text"].count("\n", 0, link.start()) + 1
                heading = next(
                    (
                        heading
                        for heading in reversed(document["_headings"])
                        if heading["line"] <= line
                    ),
                    None,
                )
                documentation.setdefault(destination, []).append(
                    {
                        "kind": "documentation",
                        "target": document["id"]
                        + ("#" + heading["fragment"] if heading else ""),
                    }
                )
    for path in sorted(inventory):
        initial = snapshots.read(path, within=root)
        if not initial:
            continue
        try:
            meta = _metadata(initial)
        except (ValueError, TypeError) as error:
            line = next(
                (
                    number
                    for number, value in enumerate(initial["text"].splitlines(), 1)
                    if _META.match(value)
                ),
                1,
            )
            snapshots.error(path, "invalid_example_metadata", str(error), line=line)
            continue
        program_entries, role_files, external, environment, unresolved = _program(
            path.parent, snapshots
        )
        relative = path.relative_to(root).as_posix()
        identity = "example:" + quote(relative, safe="/")
        group = path.relative_to(root).parts[0]
        roles = [
            {
                "basic_api": "minimal_api",
                "matmul": "mechanism",
                "algorithms": "algorithm",
            }[group]
        ]
        uses, ordinary_uses, negative = {}, set(), []
        files, guards = [], []
        for file, owners in sorted(role_files.items()):
            snapshot = snapshots.files[file]
            files.append(
                {
                    **snapshot["reference"],
                    "roles": sorted(owners),
                    "read": read_request(snapshot["reference"]),
                }
            )
            for node in ast.walk(snapshot["source"].tree):
                if isinstance(node, ast.If) and any(
                    isinstance(item, ast.Raise) for item in (*node.body, *node.orelse)
                ):
                    guards.append(
                        {
                            "condition": ast.get_source_segment(
                                snapshot["text"], node.test
                            ),
                            "reference": _reference(snapshot, node),
                            "roles": sorted(owners),
                        }
                    )
            if "kernel" not in owners:
                continue
            negative_ranges = []
            for node in ast.walk(snapshot["source"].tree):
                claims = (
                    source.counterexample_claims(node)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else {}
                )
                if claims:
                    negative.append(
                        {
                            "name": node.name,
                            "claims": claims,
                            "reference": _reference(snapshot, node),
                        }
                    )
                    negative_ranges.append((node.lineno, node.end_lineno))
            for name, lines in coverage.api_uses(snapshot["source"]).items():
                uses.setdefault(name, []).append(
                    {**snapshot["reference"], "line": lines[0]}
                )
                if any(
                    not any(begin <= line <= end for begin, end in negative_ranges)
                    for line in lines
                ):
                    ordinary_uses.add(name)
        if negative:
            roles.append("negative_control")
        scope = _docstring(initial)
        source_statements = [
            {"role": role, **doc}
            for role, snapshot in program_entries.items()
            if (doc := _docstring(snapshot))
        ]
        entry_refs = {role: snapshot["reference"] for role, snapshot in program_entries.items()}
        read_requests = {role: read_request(reference) for role, reference in entry_refs.items()}
        selected_rows = selection.get(path.parent.relative_to(root), [])
        relations = [{"kind": "uses", "target": "symbol:" + name} for name in uses]
        contracts = contract_paths.get(path.resolve(), [])
        relations.extend({"kind": "contract", "target": owner} for owner in contracts)
        selection_refs = [
            {**catalog_source["reference"], "line": row["line"]}
            for row in selected_rows
        ]
        if selected_rows:
            relations.append(
                {
                    "kind": "documentation",
                    "target": "document:skill/tooling/examples.md",
                }
            )
        for destination in sorted(
            {path.parent.resolve(), *(Path(ref["path"]) for ref in entry_refs.values())}
        ):
            for relation in documentation.get(destination, []):
                if relation not in relations:
                    relations.append(relation)
        dependencies = {
            "external_modules": sorted(
                name
                for name in external
                if name.split(".")[0] not in standard_modules and name != "__future__"
            ),
            "stdlib_modules": sorted(
                name for name in external if name.split(".")[0] in standard_modules
            ),
            "environment_variables": sorted(environment),
            "unresolved": list(
                {json.dumps(row, sort_keys=True): row for row in unresolved}.values()
            ),
            "status": "static_only",
            "instruction": (
                "Imports may be conditional; installed versions, optional evaluator"
                " requirements and dynamic paths must be checked in the linked source."
            ),
        }
        manifest = {file["path"]: file["sha256"] for file in files}
        if test_config:
            manifest[test_config["reference"]["path"]] = test_config["reference"][
                "sha256"
            ]
        records.append(
            knowledge._object(
                identity,
                "example",
                path.parent.name,
                initial["reference"],
                evidence="native_example_source",
                example_group=group,
                example_roles=roles,
                aliases=meta.get("aliases", []),
                facets=knowledge.facets({"id": identity, **meta}),
                summary=scope["text"].splitlines()[0] if scope else path.parent.name,
                conditions=[scope["text"]] if scope else [],
                condition_sources=[
                    {
                        "role": row["role"],
                        "reference": row["reference"],
                        "read": {
                            **read_request(row["reference"]),
                            "start_line": row["reference"]["line"],
                            "max_lines": row["reference"]["end_line"]
                            - row["reference"]["line"]
                            + 1,
                        },
                    }
                    for row in source_statements
                ],
                missing_facts=[
                    "Match the complete program's shapes, layouts, dtype, host"
                    " guards and device/test conditions before adapting it.",
                    "Test files and source statements do not establish current"
                    " native, numerical or performance acceptance.",
                ],
                relations=relations,
                program={
                    "entries": entry_refs,
                    "read_requests": read_requests,
                    "support_file_count": len(files) - len(entry_refs),
                    "missing_roles": sorted(set(ENTRIES) - set(entry_refs)),
                    "program_sha256": feedback.digest(
                        json.dumps(manifest, sort_keys=True).encode()
                    ),
                },
                dependencies={
                    key: dependencies[key]
                    for key in ("external_modules", "environment_variables", "status")
                },
                selection=selection_refs,
                negative_controls=negative,
                validation={
                    "status": "not_established",
                    "test_argv": (
                        [
                            "python",
                            "-m",
                            "pytest",
                            str(path.parent / "test.py"),
                            "-q",
                        ]
                        if "test" in program_entries
                        else None
                    ),
                    "source_statements": (
                        "Resolve the program to inspect separately identified"
                        " kernel/reference/test statements; no execution manifest is inferred."
                    ),
                },
                _uses=uses,
                _negative_only=set(uses) - ordinary_uses,
                _contracts=contracts,
                _meta=meta,
                _terms=" ".join(
                    [
                        relative.replace("_", " "),
                        *(row["text"] for row in selected_rows),
                        *(row["text"] for row in source_statements),
                    ]
                ),
                _identity_terms=" ".join(
                    [
                        path.parent.name.replace("_", " "),
                        *(row["text"] for row in selected_rows),
                        scope["text"].splitlines()[0] if scope else "",
                    ]
                ),
                _details={
                    "files": files,
                    "dependencies": dependencies,
                    "source_statements": source_statements,
                    "source_guards": guards,
                    "test_config": test_config["reference"] if test_config else None,
                },
            )
        )
    manifest = {str(path): digest for path, digest in snapshots.identities.items()}
    counts = {
        group: sum(row["example_group"] == group for row in records) for group in GROUPS
    }
    provider = {
        "root": str(root),
        "groups": counts,
        "programs": len(records),
        "files_read": len(manifest),
        "scan_bytes": snapshots.scanned,
        "complete": not snapshots.errors,
        "limits": snapshots.limits,
        "errors": snapshots.errors[:8],
        "error_count": len(snapshots.errors),
        "omitted_errors": max(0, len(snapshots.errors) - 8),
        "manifest_sha256": feedback.digest(
            json.dumps(
                {
                    "files": manifest,
                    "inventory": list(map(str, inventory)),
                    "errors": snapshots.errors,
                },
                sort_keys=True,
            ).encode()
        ),
    }
    provider["_files"] = manifest
    provider["_errors"] = snapshots.errors
    return records, provider


def read_request(reference):
    return {"path": reference["path"], "source_sha256": reference["sha256"]}


def match(row, text, exact, routes):
    if exact:
        score = {"basic_api": 650, "matmul": 600, "algorithms": 550}[
            row["example_group"]
        ]
        return (
            ("negative_control_api", 150)
            if exact in row["_negative_only"]
            else ("example_api_call", score)
            if exact in row["_uses"]
            else None
        )
    if text.casefold() in {alias.casefold() for alias in row["aliases"]}:
        return "example_alias", 900
    if re.sub(r"[\s_-]+", "", text.casefold()) == re.sub(
        r"[\s_-]+", "", row["title"].casefold()
    ):
        return "example_name", 850
    groups = row["_meta"].get("term_groups", [])
    if groups and knowledge.match_groups(groups, text.casefold()):
        return "example_scenario", 800
    if set(row["_contracts"]) & set(routes):
        return "contract_example", 450
    tokens = knowledge.words(text) - _DISCOVERY_WORDS
    if (
        tokens
        and not knowledge.directional(text)
        and tokens <= knowledge.words(row["_terms"])
    ):
        return "example_source_terms", 300
    return None


def _recovery_word(word):
    if word == "indices":
        return "index"
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "ous")):
        word = word[:-1]
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def _recovery_terms(text):
    terms = {}
    for word in knowledge.words(text) - _DISCOVERY_WORDS:
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            terms.update(
                (word[index: index + 2], word[index: index + 2])
                for index in range(len(word) - 1)
            )
        elif not word.isdecimal():
            terms[word] = _recovery_word(word)
    return terms


def recovery_programs(records, text, filters, *, limit=3):
    """Offer source-bound word-overlap hints without changing lookup acceptance."""
    # Namespace spellings select lookup context without establishing API availability.
    namespaces = set(api.STUBS) | {
        row["symbol"] for row in records if row["kind"] == "namespace"
    }
    qualified = {
        knowledge.symbol(name)
        for name in re.findall(
            r"(?<![A-Za-z0-9_.])(?:pl|vf)(?:\.[A-Za-z_][A-Za-z_0-9]*)+(?![A-Za-z_0-9.])",
            text,
        )
    }
    if (
        knowledge.directional(text)
        or knowledge.symbol(text)
        or qualified - namespaces
        or knowledge.public_name_hints(text, records)
    ):
        return []
    technical_context = bool(
        re.search(
            r"(?<![a-z0-9_.])(?:pl|vf|pypto_pro\.language)(?![a-z0-9_])",
            text,
            re.IGNORECASE,
        )
    )
    requested = _recovery_terms(text)
    if len(requested) < 4:
        return []
    programs = []
    for row in records:
        if row["kind"] != "example":
            continue
        declared = " ".join(
            [
                *row["aliases"],
                *(
                    term
                    for group in row["_meta"].get("term_groups", [])
                    for term in group
                ),
            ]
        )
        words = set(_recovery_terms(row["_terms"] + " " + declared).values())
        identity = set(
            _recovery_terms(row["_identity_terms"] + " " + declared).values()
        )
        programs.append((row, words, identity))
    frequency = Counter(word for _, words, _ in programs for word in words)
    distinctive = {
        word
        for word, count in frequency.items()
        if count <= max(2, len(programs) * 0.1)
    }
    requested_words = set(requested.values())
    ranked = []
    for row, words, identity in programs:
        state = knowledge.applicability(row["facets"], filters)
        if state["conflicting_filters"] or conflicts(row, text):
            continue
        identity_overlap = requested_words & identity
        identity_clues = identity_overlap & distinctive
        scenario_clues = sum(
            knowledge.match_groups([group], text.casefold())
            for group in row["_meta"].get("term_groups", [])
        )
        coherent = len(identity_clues) >= 2 or scenario_clues >= 2
        if technical_context and not coherent:
            continue
        overlap = {word for word, normalized in requested.items() if normalized in words}
        normalized = {requested[word] for word in overlap}
        if not coherent and not (
            len(identity_overlap) >= 2
            and identity_clues
            and len(normalized & distinctive) >= 2
        ):
            continue
        minimum_coverage = 0.25 if any(not word.isascii() for word in requested) else 0.4
        if (
            len(overlap) < 3
            or not coherent
            and len(overlap) < minimum_coverage * len(requested)
        ):
            continue
        score = sum(
            math.log1p(len(programs) / (frequency[word] + 1))
            for word in sorted(normalized)
        )
        score /= 1 + 0.15 * math.log1p(len(words))
        ranked.append(
            (scenario_clues >= 2, len(identity_clues), score, row, state, overlap)
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]["id"]))
    return [
        {
            "id": row["id"],
            "kind": "example",
            "title": row["title"],
            "reference": row["reference"],
            "example_group": row["example_group"],
            "example_roles": row["example_roles"],
            "program": {
                key: row["program"][key] for key in ("entries", "read_requests")
            },
            "evidence": row["evidence"],
            "applicability": state,
            "summary": row["summary"],
            "facets": row["facets"],
            "condition_sources": row["condition_sources"],
            "dependencies": row["dependencies"],
            "missing_facts": row["missing_facts"],
            "matched_terms": sorted(overlap),
            "unmatched_terms": sorted(set(requested) - overlap),
            "resolve": {
                "text": row["id"],
                "reference_sha256": row["reference"]["sha256"],
            },
        }
        for scenario, _, score, row, state, overlap in ranked[:limit]
        if scenario == ranked[0][0] and score >= 0.8 * ranked[0][2]
    ]


def conflicts(row, text):
    return [
        item["reason"]
        for item in row["_meta"].get("exclude", [])
        if knowledge.match_groups(item["term_groups"], text.casefold())
    ]


def read_object(row, root, pypto_root, reference_sha256):
    if reference_sha256 and reference_sha256 != row["reference"]["sha256"]:
        return {
            "status": "source_changed",
            "reason": "Pinned kernel SHA no longer matches; rediscover the current program.",
        }
    config = host_io.policy(workspace=root, examples_root=root, pypto_root=pypto_root)
    try:
        content = host_io.read_requests([read_request(row["reference"])], config=config)
    except (OSError, ValueError) as error:
        return {"status": "source_unavailable", "reason": str(error)}
    return {"status": "matched", "content": content}
