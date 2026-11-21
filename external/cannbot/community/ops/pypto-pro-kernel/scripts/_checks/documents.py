# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read-only Pro documentation and source-docstring adapters over their owners."""

from __future__ import annotations

import json
from functools import partial
import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from . import feedback, host_io, knowledge, links

SCOPES = {
    "pro-api": ("docs/zh/api/pro_api", "official_pro_api"),
    "pro-guide": ("docs/zh/guide/programming_guide/pro", "official_programming_guide"),
    "pro-quickstart": ("docs/zh/guide/quick_start/pro", "official_quickstart"),
}
_SYMBOL = re.compile(r"(?<![\w.])(?:pypto_pro\.language|pl|vf)(?:\.[A-Za-z_]\w*)+")


def canonical(value):
    return knowledge.symbol(value.replace("pypto_pro.language", "pl"))


def roots(pypto_root, references_root, cards_root=None):
    result = {
        **{name: (pypto_root / path).resolve() for name, (path, _) in SCOPES.items()},
        "skill": references_root.resolve(),
    }
    if cards_root is not None:
        try:
            result["perf-cards"] = Path(cards_root).expanduser().resolve()
        except RuntimeError as error:
            raise ValueError(
                "Card source path cannot be resolved: " + str(error)
            ) from error
    return result


def locator(text, selected_roots):
    if text.startswith("docstring:"):
        name = canonical(text.removeprefix("docstring:"))
        if not name:
            raise ValueError("Docstring IDs require an exact public symbol")
        return "docstring:" + name, None, None
    if not text.startswith("document:"):
        return None
    value, separator, fragment = text.removeprefix("document:").partition("#")
    scope, slash, relative = value.partition("/")
    relative = unquote(relative)
    parts = PurePosixPath(relative)
    if (
        scope not in selected_roots
        or not slash
        or not relative
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in relative
        or "\x00" in relative
        or parts.suffix.lower() != ".md"
        or separator
        and not fragment
    ):
        raise ValueError(
            "Document IDs require a selected scope and a relative Markdown path, with an optional fragment"
        )
    if scope == "skill" and parts.parts[0] == "knowledge-cards":
        raise ValueError("Select the card root and use document:perf-cards/ for Active cards")
    path = (selected_roots[scope] / relative).resolve()
    if not path.is_relative_to(selected_roots[scope]):
        raise ValueError("Document locator escapes its selected source root")
    identity = "document:" + scope + "/" + quote(parts.as_posix(), safe="/")
    return identity, path, unquote(fragment) if separator else None


def collect(
    pypto_root,
    references_root,
    declarations,
    entries,
    api_coverage,
    *,
    exact_target=None,
):
    selected = roots(pypto_root, references_root)
    selection = {"mode": "selected_scopes"}
    if exact_target:
        selection = {
            "mode": "exact_docstring"
            if exact_target[0].startswith("docstring:")
            else "exact_document",
            "id": exact_target[0],
        }
        selected = (
            {}
            if exact_target[1] is None
            else {
                exact_target[0].removeprefix("document:").split("/", 1)[0]: selected[
                    exact_target[0].removeprefix("document:").split("/", 1)[0]
                ]
            }
        )
    records, files, errors, scopes = [], {}, [], {}
    limits = {
        key: host_io.CEILINGS[key]
        for key in ("max_files", "max_scan_bytes", "max_file_bytes")
    }
    scanned, seen = 0, 0
    declared = {row["symbol"] for row in declarations}

    def walk_error(selected_scope, error):
        errors.append(
            {
                "scope": selected_scope,
                "path": str(error.filename),
                "reason": type(error).__name__,
            }
        )

    for scope, root in selected.items():
        role = SCOPES[scope][1] if scope in SCOPES else "skill_reference"
        start_errors = len(errors)
        count = 0
        try:
            if not root.is_dir():
                raise FileNotFoundError(
                    "Selected documentation root is missing or not a directory"
                )

            walker = (
                [(exact_target[1].parent, [], [exact_target[1].name])]
                if exact_target
                else os.walk(
                    root, onerror=partial(walk_error, scope), followlinks=False
                )
            )
            for parent, directories, names in walker:
                if scope == "skill" and Path(parent) == root and "knowledge-cards" in directories:
                    directories.remove("knowledge-cards")  # Only the Active index admits cards.
                directories.sort()
                for name in sorted(names):
                    if not name.lower().endswith(".md"):
                        continue
                    path = Path(parent) / name
                    try:
                        seen += 1
                        if not path.resolve().is_relative_to(root):
                            raise ValueError(
                                "Documentation path escapes its selected root"
                            )
                        if seen > limits["max_files"]:
                            raise ValueError("document_file_scan_limit")
                        data = host_io._bytes(
                            path,
                            min(
                                limits["max_file_bytes"],
                                limits["max_scan_bytes"] - scanned,
                            ),
                        )
                        scanned += len(data)
                        files[str(path.resolve())] = feedback.digest(data)
                        text = data.decode("utf-8")
                        if "\x00" in text:
                            raise ValueError("Documentation contains binary data")
                        reference = {
                            "path": str(path.resolve()),
                            "sha256": feedback.digest(data),
                            "line": 1,
                        }
                        headings = links.heading_locations(text)
                        title = headings[0]["title"] if headings else path.stem
                        primary = (
                            canonical(title.strip("` ")) if scope == "pro-api" else None
                        )
                        relations = (
                            [
                                {
                                    "kind": "describes",
                                    "target": "symbol:" + primary,
                                    "declaration": "available"
                                    if primary in declared
                                    else "not_established",
                                }
                            ]
                            if primary
                            else []
                        )
                        identity = (
                            exact_target[0]
                            if exact_target
                            else "document:"
                            + scope
                            + "/"
                            + quote(path.relative_to(root).as_posix(), safe="/")
                        )
                        records.append(
                            knowledge._object(
                                identity,
                                "document",
                                title,
                                reference,
                                evidence="maintained_reference"
                                if scope == "skill"
                                else "official_documentation",
                                document_role=role,
                                facets={},
                                relations=relations,
                                section_count=len(headings),
                                missing_facts=[
                                    "Documentation and examples are source statements;"
                                    " compare the selected declaration, installation and scoped execution evidence."
                                ],
                                _text=text,
                                _headings=headings,
                                _primary=primary,
                                _mentions={
                                    canonical(match[0])
                                    for match in _SYMBOL.finditer(text)
                                },
                                _contracts={},
                            )
                        )
                        count += 1
                    except (OSError, ValueError) as error:
                        errors.append(
                            {
                                "scope": scope,
                                "path": str(path),
                                "reason": type(error).__name__,
                                "detail": str(error)[:200],
                            }
                        )
                        if (
                            seen >= limits["max_files"]
                            or scanned >= limits["max_scan_bytes"]
                        ):
                            break
                if seen >= limits["max_files"] or scanned >= limits["max_scan_bytes"]:
                    errors.append(
                        {
                            "scope": scope,
                            "path": str(root),
                            "reason": "document_scan_limit",
                        }
                    )
                    break
            if not count and len(errors) == start_errors:
                errors.append(
                    {
                        "scope": scope,
                        "path": str(root),
                        "reason": "no_markdown_documents",
                    }
                )
        except OSError as error:
            errors.append(
                {
                    "scope": scope,
                    "path": str(root),
                    "reason": type(error).__name__,
                    "detail": str(error)[:200],
                }
            )
        scopes[scope] = {
            "root": str(root),
            "role": role,
            "files_read": count,
            "complete": len(errors) == start_errors,
        }
    by_path = {row["reference"]["path"]: row for row in records}
    for meta, path, body in entries:
        owner = str(path.resolve())
        targets = [(owner, None)]
        for link_match in links._LINK.finditer(body):
            target = urlsplit(link_match["target"].strip("<>"))
            if not target.scheme and not target.netloc:
                targets.append(
                    (
                        str((path.parent / unquote(target.path)).resolve()),
                        unquote(target.fragment) or None,
                    )
                )
        for _, line in links.prose_lines(body):
            for source_match in re.finditer(r"`pypto/([^`\n]+)`", line):
                target = urlsplit(source_match[1])
                resolved = (pypto_root / unquote(target.path)).resolve()
                if resolved.is_relative_to(pypto_root.resolve()):
                    targets.append((str(resolved), unquote(target.fragment) or None))
        for target_path, fragment in targets:
            row = by_path.get(target_path)
            if row:
                if fragment is None and target_path == owner:
                    fragment = next(
                        (
                            item["fragment"]
                            for item in row["_headings"]
                            if item["line"] == meta["line"]
                        ),
                        None,
                    )
                row["_contracts"].setdefault("contract:" + meta["id"], fragment)
    for row in declarations:
        if exact_target and exact_target[0] != "docstring:" + row["symbol"]:
            continue
        doc = row.get("_docstring")
        if doc:
            records.append(
                knowledge._object(
                    "docstring:" + row["symbol"],
                    "document",
                    row["symbol"] + " docstring",
                    doc["reference"],
                    evidence="source_docstring",
                    document_role="source_docstring",
                    facets={},
                    relations=[
                        {"kind": "describes", "target": "symbol:" + row["symbol"]}
                    ],
                    missing_facts=[
                        "A source docstring is separate from official usage"
                        " documentation and installed execution behavior."
                    ],
                    _text=doc["text"],
                    _headings=[],
                    _primary=row["symbol"],
                    _mentions={
                        canonical(match[0]) for match in _SYMBOL.finditer(doc["text"])
                    },
                    _contracts={},
                )
            )
    for row in records:
        for meta, _, _ in entries:
            if row["_primary"] in {canonical(name) for name in meta.get("symbols", [])}:
                row["_contracts"].setdefault("contract:" + meta["id"], None)
    manifest = feedback.digest(
        json.dumps(
            {
                "files": files,
                "errors": errors,
                "roots": {key: str(value) for key, value in selected.items()},
                "selection": selection,
            },
            sort_keys=True,
        ).encode()
    )
    coverage = {
        "complete": not errors and api_coverage["complete"],
        "scopes": scopes,
        "files_read": len(files),
        "files_seen": seen,
        "scan_bytes": scanned,
        "limits": limits,
        "manifest_sha256": manifest,
        "error_count": len(errors),
        "errors": errors[:8],
        "omitted_errors": max(0, len(errors) - 8),
        "docstrings": sum(
            row["document_role"] == "source_docstring" for row in records
        ),
        "docstring_source_complete": api_coverage["complete"],
        "selection": selection,
    }
    coverage.update(_files=files, _errors=errors)
    return records, coverage


def match(row, text, exact, routes, declarations, name_hints=()):
    if row["document_role"] == "performance_card":
        if text.casefold() in {value.casefold() for value in row["aliases"]}:
            return "card_id", 950
        if any(relation["target"] in routes for relation in row["relations"]):
            return "related_card", 350
        tokens = knowledge.words(text)
        if (
            not exact
            and not knowledge.directional(text)
            and tokens
            and any(tokens <= knowledge.words(title) for title in row["_card_titles"])
        ):
            return "related_card_section", 250
    if exact and row["_primary"] == exact:
        return (
            ("docstring_symbol", 800)
            if row["document_role"] == "source_docstring"
            else ("document_symbol", 850)
        )
    related = set(row["_contracts"]) & set(routes)
    if related:
        return "contract_document", 400
    if exact:
        kind = declarations.get(exact, {}).get("kind")
        score = (
            250
            if row["document_role"]
            in {"official_programming_guide", "official_quickstart"}
            else 200
            if row["document_role"] == "skill_reference"
            else 150
        )
        return (
            ("document_mention", score)
            if kind not in {"constant", "enum_member", "field"}
            and exact in row["_mentions"]
            else None
        )
    mentioned = {canonical(item[0]) for item in _SYMBOL.finditer(text)}
    mentioned -= {name.rsplit(".", 1)[0] for name in name_hints}
    if row["_primary"] in mentioned:
        return "document_primary_symbol", 325 if row[
            "document_role"
        ] == "source_docstring" else 350
    if mentioned and mentioned & ({row["_primary"]} | row["_mentions"]):
        return "document_mention", 150
    if row["_primary"] in name_hints:
        return "public_name_hint", name_hints[row["_primary"]] + (
            0 if row["document_role"] == "source_docstring" else 1
        )
    if not text or knowledge.directional(text):
        return None
    tokens = knowledge.words(text)
    if tokens and tokens <= knowledge.words(row["title"]):
        return "document_title", 350
    if preferred_fragment(row, text, []):
        return "document_heading", 300
    if tokens and any(
        all(token in paragraph.casefold() for token in tokens)
        for paragraph in re.split(r"\n\s*\n", row["_text"])
    ):
        return "document_text", 100
    return None


def preferred_fragment(row, text, routes):
    for identity in sorted(set(row["_contracts"]) & set(routes)):
        if row["_contracts"][identity]:
            return row["_contracts"][identity]
    tokens = knowledge.words(text)
    return next(
        (
            heading["fragment"]
            for heading in row["_headings"]
            if tokens and all(token in heading["title"].casefold() for token in tokens)
        ),
        None,
    )


def read_object(row, text, selected_roots, pypto_root, reference_sha256):
    located = locator(text, selected_roots)
    fragment = located[2]
    digest = reference_sha256 or row["reference"]["sha256"]
    if digest != row["reference"]["sha256"]:
        return {
            "status": "source_changed",
            "reason": "Pinned source SHA no longer matches; discover the current source before rereading.",
        }
    request = {"path": row["reference"]["path"], "source_sha256": digest}
    if fragment:
        try:
            links.fragment_range(row["_text"], fragment)
        except links.FragmentError as error:
            return {
                "status": error.reason,
                "reason": str(error),
                "sections": row["_headings"],
            }
        request["fragment"] = fragment
    elif row["document_role"] == "source_docstring":
        request.update(
            start_line=row["reference"]["line"],
            max_lines=row["reference"]["end_line"] - row["reference"]["line"] + 1,
        )
    try:
        workspace = (
            selected_roots["perf-cards"]
            if row["document_role"] == "performance_card"
            else selected_roots["skill"]
        )
        config = host_io.policy(workspace=workspace, pypto_root=pypto_root)
        content = host_io.read_requests([request], config=config)
    except (OSError, ValueError) as error:
        return {"status": "source_unavailable", "reason": str(error)}
    return {"status": "matched", "content": content, "sections": row["_headings"]}
