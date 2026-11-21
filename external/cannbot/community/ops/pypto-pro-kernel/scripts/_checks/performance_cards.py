# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional, source-bound card documents selected only by the owner's Active table."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from . import documents, feedback, host_io, knowledge, links

SCHEMA = "pypto-knowledge-links/1"
ROLE = "performance_card"
SCOPE = "perf-cards"
RELATIONS = {"same_mechanism", "overlaps_with", "uses_contract", "alternative_to"}
_ID = re.compile(r"[a-z][a-z0-9-]*-\d{2}")
_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*`?stable`?\s*\|"
)
_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def frontmatter(text):
    """Load bounded YAML data with unique keys and no aliases or object constructors."""
    try:
        import yaml
    except ImportError as error:
        raise ValueError(
            "Card reads need PyYAML; install scripts/requirements-cards.txt"
        ) from error
    lines = text.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError("Card needs YAML frontmatter")
    source = "\n".join(lines[1: lines.index("---", 1)])

    class CardLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ValueError("YAML mapping keys must be unique strings")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    try:
        if any(
            isinstance(token, yaml.tokens.AliasToken) for token in yaml.scan(source)
        ):
            raise ValueError("YAML aliases are not supported in cards")
        loader = CardLoader(source)
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    except (yaml.YAMLError, RecursionError) as error:
        raise ValueError("Invalid card YAML: " + str(error)[:200]) from error
    if not isinstance(data, dict):
        raise ValueError("Card frontmatter must be a mapping")
    return data


def validate_links(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "relations"}
        or value["schema"] != SCHEMA
    ):
        raise ValueError("Expected pypto-knowledge-links/1 with schema and relations")
    rows = value["relations"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError("knowledge_links needs 1..32 relations")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "target",
            "scope",
            "boundary",
        }:
            raise ValueError("A knowledge link needs kind, target, scope and boundary")
        if not isinstance(row["kind"], str) or row["kind"] not in RELATIONS:
            raise ValueError("Unsupported knowledge relation kind")
        target = row["target"]
        if not isinstance(target, dict) or set(target) != {"skill", "id"}:
            raise ValueError("A knowledge target needs skill and id")
        for name, item in [
            ("scope", row["scope"]),
            ("boundary", row["boundary"]),
            ("target.skill", target["skill"]),
            ("target.id", target["id"]),
        ]:
            _string(item, name)
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", target["skill"]):
            raise ValueError("Target skill must be a stable skill name")
        if not (
            re.fullmatch(r"contract:[a-z0-9][a-z0-9_-]*", target["id"])
            or target["id"].startswith("document:skill/")
        ):
            raise ValueError("Target must be an existing contract or skill document ID")
        key = (row["kind"], target["skill"], target["id"])
        if key in seen:
            raise ValueError("Duplicate knowledge relation")
        seen.add(key)
    return rows


def collect(root, *, references_root, entries):
    """Keep valid cards queryable while reporting quarantined or unavailable sources."""
    root, references_root = (
        Path(root).expanduser().absolute(),
        Path(references_root).resolve(),
    )
    files, errors, records = {}, [], []
    scanned, active, seen = 0, 0, 0
    limits = {
        key: host_io.CEILINGS[key]
        for key in ("max_files", "max_file_bytes", "max_scan_bytes")
    }
    contracts = {"contract:" + meta["id"]: (meta, path) for meta, path, _ in entries}

    def read(path, selected_root):
        nonlocal scanned, seen
        seen += 1
        path = path.resolve()
        if not path.is_relative_to(selected_root):
            raise ValueError("Card or relation path escapes its selected root")
        if len(files) >= limits["max_files"] and str(path) not in files:
            raise ValueError("Card file count limit exceeded")
        data = host_io._bytes(
            path, min(limits["max_file_bytes"], limits["max_scan_bytes"] - scanned)
        )
        scanned += len(data)
        digest = feedback.digest(data)
        if str(path) in files and files[str(path)] != digest:
            raise knowledge.QueryError(
                "source_changed",
                "source_changed_during_collection",
                "Card sources changed during collection",
            )
        files[str(path)] = digest
        text = data.decode("utf-8")
        if "\x00" in text:
            raise ValueError("Card source contains binary data")
        return text, {"path": str(path), "sha256": digest, "line": 1}

    def resolve(row):
        target = row["target"]
        if target["skill"] != "pypto-pro-kernel":
            raise ValueError("Target skill is not selected: " + target["skill"])
        identity = target["id"]
        if identity.startswith("contract:"):
            if identity not in contracts:
                raise ValueError(
                    "Knowledge contract target is missing or ambiguous: " + identity
                )
            meta, path = contracts[identity]
            _, reference = read(path, references_root)
            if reference["sha256"] != meta["_reference"]["sha256"]:
                raise ValueError("Knowledge contract changed during collection")
            reference["line"] = meta["line"]
            title = meta["title"]
        else:
            located = documents.locator(identity, {"skill": references_root})
            if located is None:
                raise ValueError("Invalid skill document target")
            text, reference = read(located[1], references_root)
            if located[2]:
                reference["line"] = links.fragment_range(text, located[2])["start_line"]
            identity = located[0] + ("#" + located[2] if located[2] else "")
            heading = next(
                (
                    item
                    for item in links.heading_locations(text)
                    if item["line"] == reference["line"]
                ),
                None,
            )
            title = heading["title"] if heading else located[1].stem
        return {
            "kind": row["kind"],
            "target": identity,
            "target_skill": target["skill"],
            "scope": row["scope"],
            "boundary": row["boundary"],
            "reference": reference,
        }, title

    index = root / "index.md"
    try:
        root = root.resolve()
        index_text, index_reference = read(index, root)
        span = links.fragment_range(index_text, "active-items")
        seen_ids, seen_paths, ambiguous_ids = set(), set(), set()
        for lineno, line in links.prose_lines(index_text):
            if not span["start_line"] < lineno <= span["end_line"]:
                continue
            line = line.lstrip()
            if not line.startswith("|"):
                continue
            if line.split("|", 2)[1].strip().strip(
                "`"
            ) == "item_id" or _SEPARATOR.fullmatch(line):
                continue
            match = _ROW.match(line)
            if not match:
                errors.append(
                    {
                        "path": str(index),
                        "reason": "metadata_invalid",
                        "detail": "Malformed Active index row",
                    }
                )
                continue
            item_id, _, destination = match.groups()
            active += 1
            path = index
            try:
                if not _ID.fullmatch(item_id):
                    raise ValueError("Invalid Active item_id: " + item_id)
                if item_id in seen_ids:
                    ambiguous_ids.add(item_id)
                    raise ValueError("Duplicate Active item_id: " + item_id)
                seen_ids.add(item_id)
                url = urlsplit(destination.strip("<>"))
                relative = unquote(url.path)
                if (
                    url.scheme
                    or url.netloc
                    or url.query
                    or not relative
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or "\\" in relative
                ):
                    raise ValueError(
                        "Active card needs a relative path within its root"
                    )
                relative = PurePosixPath(relative).as_posix()
                path = (root / relative).resolve()
                if (
                    path.suffix != ".md"
                    or not path.name.startswith(item_id + "-")
                    or path in seen_paths
                ):
                    raise ValueError("Card path, item_id or duplicate file is invalid")
                seen_paths.add(path)
                text, reference = read(path, root)
                data = frontmatter(text)
                if (
                    data.get("type") != "PyPTO Performance Optimization Card"
                    or data.get("item_id") != item_id
                    or data.get("status") != "stable"
                ):
                    raise ValueError(
                        "Card type, item_id or status differs from the Active index"
                    )
                for field in (
                    "title",
                    "description",
                    "bound_hint",
                    "applicability",
                    "target_api_gate",
                ):
                    _string(data.get(field), field)
                tags = data.get("tags")
                if (
                    not isinstance(tags, list)
                    or not tags
                    or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
                ):
                    raise ValueError("Card tags must be nonempty strings")
                if url.fragment:
                    links.fragment_range(text, unquote(url.fragment))
                relations, titles, resolved_targets = [], [], set()
                for row in (
                    validate_links(data["knowledge_links"])
                    if "knowledge_links" in data
                    else []
                ):
                    relation, title = resolve(row)
                    key = (
                        relation["kind"],
                        relation["target_skill"],
                        relation["target"],
                    )
                    if key in resolved_targets:
                        raise ValueError(
                            "Duplicate knowledge relation after target normalization"
                        )
                    resolved_targets.add(key)
                    relations.append(relation)
                    titles.append(title)
                headings = links.heading_locations(text)
                metadata = {
                    name: data[name]
                    for name in (
                        "item_id",
                        "status",
                        "description",
                        "tags",
                        "bound_hint",
                        "applicability",
                        "target_api_gate",
                    )
                }
                records.append(
                    knowledge._object(
                        "document:" + SCOPE + "/" + quote(relative, safe="/"),
                        "document",
                        data["title"],
                        reference,
                        evidence="maintained_optimization_card",
                        document_role=ROLE,
                        facets={},
                        relations=relations,
                        card={**metadata, "index": index_reference},
                        aliases=[item_id],
                        section_count=len(headings),
                        conditions=[data["applicability"], data["target_api_gate"]],
                        missing_facts=[
                            "Active is a candidate status, not current applicability,"
                            " runnable code or numerical/performance evidence. Read the"
                            " original card and resolve its conditions."
                        ],
                        _text=text,
                        _headings=headings,
                        _primary=None,
                        _mentions=set(),
                        _contracts={},
                        _card_titles=titles,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(
                    {
                        "path": str(path),
                        "reason": "source_unavailable"
                        if isinstance(error, (OSError, RuntimeError))
                        else "metadata_invalid",
                        "detail": str(error)[:400],
                        "id": item_id,
                    }
                )
        if not active:
            errors.append(
                {
                    "path": str(index),
                    "reason": "metadata_invalid",
                    "detail": "Active table contains no registered cards",
                }
            )
        records = [row for row in records if row["card"]["item_id"] not in ambiguous_ids]
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(
            {
                "path": str(index),
                "reason": "source_unavailable"
                if isinstance(error, (OSError, RuntimeError))
                else "metadata_invalid",
                "detail": str(error)[:400],
            }
        )
    manifest = feedback.digest(
        json.dumps(
            {"root": str(root), "files": files, "errors": errors}, sort_keys=True
        ).encode()
    )
    return records, {
        "complete": not errors,
        "root": str(root),
        "role": ROLE,
        "active_items": active,
        "cards_read": len(records),
        "linked_cards": sum(bool(row["relations"]) for row in records),
        "unlinked_cards": sum(not row["relations"] for row in records),
        "relation_count": sum(len(row["relations"]) for row in records),
        "manifest_sha256": manifest,
        "files_read": len(files),
        "files_seen": seen,
        "scan_bytes": scanned,
        "limits": limits,
        "errors": errors[:8],
        "error_count": len(errors),
        "omitted_errors": max(0, len(errors) - 8),
        "_files": files,
        "_errors": errors,
    }
