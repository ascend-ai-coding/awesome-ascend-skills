# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fresh knowledge objects and deterministic discovery over existing owners."""

from __future__ import annotations

import base64
import json
import re
from urllib.parse import quote

from . import feedback

SCHEMA = "pypto-knowledge-query/1"
SYMBOL_KINDS = ("api", "type", "constant", "enum", "enum_member", "namespace", "field")
KINDS = (*SYMBOL_KINDS, "contract", "document", "example", "diagnostic", "evidence")
FILTERS = (
    "source_memory",
    "destination_memory",
    "dtype",
    "layout",
    "source_dtype",
    "destination_dtype",
)
FILTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        key: {"type": "string", "minLength": 1, "maxLength": 128} for key in FILTERS
    },
}
DEFAULT_LIMIT, MAX_LIMIT = 10, 50
_MEMORY = {
    "ub": "Vec",
    "vec": "Vec",
    "l1": "Mat",
    "mat": "Mat",
    "gm": "GM",
    "l0c": "Acc",
    "acc": "Acc",
    "l0a": "Left",
    "left": "Left",
    "l0b": "Right",
    "right": "Right",
}
_STOP = {
    "a",
    "an",
    "the",
    "to",
    "from",
    "in",
    "on",
    "of",
    "for",
    "and",
    "or",
    "with",
    "how",
    "use",
    "using",
}


class QueryError(ValueError):
    def __init__(self, status, reason, message):
        self.status, self.reason = status, reason
        super().__init__(message)


def normalize_value(key, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(
            "Knowledge filters require nonempty strings up to 128 characters"
        )
    value = value.strip().rsplit(".", 1)[-1]
    if key.endswith("memory"):
        return _MEMORY.get(value.casefold(), value)
    return value.upper().removeprefix("DT_")


def request(
    text, kinds=None, filters=None, limit=None, cursor=None, reference_sha256=None
):
    if text is not None and (
        not isinstance(text, str) or not text.strip() or len(text) > 4096
    ):
        raise ValueError("Query text must contain 1..4096 characters")
    if kinds is not None and (
        not isinstance(kinds, (list, tuple))
        or not kinds
        or any(kind not in KINDS for kind in kinds)
    ):
        raise ValueError("Unsupported knowledge kind")
    if filters is not None and (
        not isinstance(filters, dict) or set(filters) - set(FILTERS)
    ):
        raise ValueError("Unsupported knowledge filter")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= MAX_LIMIT):
        raise ValueError(f"Query limit must be 1..{MAX_LIMIT}")
    if cursor is not None and (
        not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096
    ):
        raise ValueError("Invalid knowledge cursor")
    if text is None and not kinds and not filters:
        raise ValueError("Supply query text, knowledge kinds or filters")
    if reference_sha256 is not None and (
        not isinstance(reference_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference_sha256)
        or not (text or "").strip().startswith(("document:", "docstring:", "example:"))
    ):
        raise ValueError(
            "reference_sha256 requires a document/docstring/example ID and a lowercase SHA256"
        )
    normalized = (text or "").strip()
    defaults = (
        ("document",) if normalized.startswith(("document:", "docstring:")) else KINDS
    )
    return {
        "text": normalized,
        "kinds": sorted(set(kinds or defaults)),
        "filters": {
            key: normalize_value(key, value)
            for key, value in sorted((filters or {}).items())
        },
        "limit": limit or DEFAULT_LIMIT,
        "cursor": cursor,
        "reference_sha256": reference_sha256,
    }


def symbol(text):
    text = text.removeprefix("symbol:")
    if text == "pl.Vf" or text.startswith("pl.Vf."):
        text = "vf" + text[5:]
    return text if re.fullmatch(r"(?:pl|vf)(?:\.[A-Za-z_]\w*)*", text) else None


def match_groups(groups, text):
    cursor = 0
    for group in groups:
        ends = []
        for term in group:
            term = term.casefold()
            pattern = re.escape(term)
            if term.isascii():
                pattern = r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])"
            match = re.search(pattern, text[cursor:])
            if match:
                ends.append(cursor + match.end())
        if not ends:
            return False
        cursor = min(ends)
    return True


def words(text):
    return {
        word
        for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold())
        if word not in _STOP
    }


def directional(text):
    entity = r"(?<![a-z0-9_])(?:fp\d+|bf16|ub|vec|vector|l1|mat|gm|l0[abc]|acc|left|right)(?![a-z0-9_])"
    return bool(
        re.search(entity + r".*?(?:\bto\b|→|->|到|向).*?" + entity, text.casefold())
    )


def public_name_hints(text, declarations):
    """Use public API names explicitly mentioned in a question as discovery hints."""
    if (
        symbol(text)
        or directional(text)
        or text.startswith(
            ("symbol:", "contract:", "document:", "docstring:", "example:")
        )
    ):
        return {}
    namespaces = {
        name.casefold().replace("pl.vf", "vf")
        for name in re.findall(
            r"(?<![a-z0-9_.])(?:pl(?:\.[a-z_][a-z0-9_]*)?|vf)(?![a-z0-9_.])",
            text,
            re.IGNORECASE,
        )
    }
    names = set(
        re.findall(r"(?<![a-z0-9_.])[a-z_][a-z0-9_]*(?![a-z0-9_.])", text.casefold())
    )
    qualified = {
        symbol(name)
        for name in re.findall(
            r"(?<![A-Za-z0-9_.])(?:pl|vf)(?:\.[A-Za-z_][A-Za-z_0-9]*)+(?![A-Za-z_0-9.])",
            text,
        )
    }
    tokens = words(text)
    return {
        row["symbol"]: (350 if row["symbol"] in qualified else 250)
        + 10 * min(8, len(tokens & words(row.get("summary", ""))))
        for row in declarations
        if row["kind"] == "api"
        and (
            row["symbol"] in qualified
            or (
                row["symbol"].rsplit(".", 1)[0].casefold() in namespaces
                and row["symbol"].rsplit(".", 1)[-1].casefold() in names
            )
        )
    }


def _symbol_intent(meta, text):
    """Discover an owner's phrase and qualified symbol in the same clause."""
    if directional(text):
        return False
    for clause in re.split(r"[;；。!?！？\n]|\.(?=\s)", text):
        mentioned = {
            symbol(name)
            for name in re.findall(
                r"(?<![A-Za-z0-9_.])(?:pl|vf)(?:\.[A-Za-z_][A-Za-z_0-9]*)+(?![A-Za-z_0-9.])",
                clause,
            )
        }
        if not mentioned.intersection(meta.get("symbols", [])):
            continue
        clause = clause.casefold()
        for alias in meta.get("aliases", []):
            if alias == meta["id"] or symbol(alias) or len(words(alias)) < 2:
                continue
            phrase = r"\s+".join(map(re.escape, alias.casefold().split()))
            if alias.isascii():
                phrase = r"(?<![a-z0-9_])" + phrase + r"(?![a-z0-9_])"
            for match in re.finditer(phrase, clause):
                before, after = clause[: match.start()], clause[match.end():]
                if re.search(
                    r"(?:\b(?:not|no|without|exclude|excluding|ignore|ignoring)"
                    r"|不是|并非|不涉及|没有|排除|忽略)\s*(?:(?:the|any)\s+)?$",
                    before,
                ):
                    continue
                if re.match(
                    r"\s+(?:is|are)\s+not\s+(?:the\s+)?(?:issue|problem|relevant|present|involved)\b",
                    after,
                ):
                    continue
                return True
    return False


def contract_match(meta, text, exact=None):
    normalized = text.casefold().strip()
    if normalized in {meta["id"].casefold(), "contract:" + meta["id"].casefold()}:
        return "id", 1000
    if normalized in {item.casefold() for item in meta.get("aliases", [])}:
        return "alias", 900
    if meta.get("term_groups") and match_groups(meta["term_groups"], normalized):
        return "scenario", 800
    if exact and exact in meta.get("symbols", []):
        return "symbol", 700
    if not exact:
        if _symbol_intent(meta, text):
            return "symbol_intent", 650
        groups = meta.get("discovery_terms", [])
        if normalized in {
            item.casefold() for item in meta.get("discovery_aliases", [])
        } or (groups and all(match_groups([group], normalized) for group in groups)):
            return "discovery", 600
        # Directional recipes require their declared ordered terms or aliases.
        tokens = words(normalized)
        haystack = words(
            " ".join(
                [
                    meta["id"],
                    meta["title"],
                    meta.get("when", ""),
                    *meta.get("aliases", []),
                ]
            )
        )
        if tokens and not directional(normalized) and tokens <= haystack:
            return "discovery", 100
    return None


def facets(meta):
    declared = meta.get("facets", {})
    if not isinstance(declared, dict) or set(declared) - set(FILTERS):
        raise ValueError(f"Invalid query facets: {meta['id']}")
    result = {}
    for key, values in declared.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Query facets need nonempty string lists: {meta['id']}")
        result[key] = sorted({normalize_value(key, value) for value in values})
    return result


def applicability(declared, filters):
    matched, unknown, conflicts = {}, [], {}
    for key, value in filters.items():
        if key not in declared:
            unknown.append(key)
        elif value in declared[key]:
            matched[key] = value
        else:
            conflicts[key] = {"requested": value, "declared": declared[key]}
    return {
        "status": "unknown" if unknown else "conditional",
        "matched_filters": matched,
        "unknown_filters": unknown,
        "conflicting_filters": conflicts,
        "evidence_limit": (
            "Routing conditions only; shape, consumers and"
            " installation/native/numerical/performance support remain unverified."
        ),
    }


def _object(identity, kind, title, reference, *, evidence, **fields):
    if kind not in KINDS or not identity or not reference.get("sha256"):
        raise ValueError("Invalid knowledge object identity, kind or source")
    return {
        "id": identity,
        "kind": kind,
        "title": title,
        "reference": reference,
        "evidence": {
            "kind": evidence,
            "native": "not_established",
            "numerical": "not_established",
            "performance": "not_established",
        },
        "conditions": [],
        "missing_facts": [],
        "relations": [],
        **fields,
    }


def objects(entries, declarations, examples_root):
    result, missing = [], []
    for row in declarations:
        result.append(
            _object(
                "symbol:" + row["symbol"],
                row["kind"],
                row["symbol"],
                row["reference"],
                evidence="source_declaration",
                symbol=row["symbol"],
                aliases=row["aliases"],
                declaration=row["declaration"],
                summary=row["summary"],
                callable=row["callable"],
                facets={},
                missing_facts=[
                    "Selected source declarations are not installed-runtime or execution acceptance."
                ],
                relations=[{"kind": "member_of", "target": "symbol:" + row["parent"]}]
                if row["parent"]
                else [],
            )
        )
    seen = set()
    for meta, path, body in entries:
        if meta["id"] in seen:
            raise ValueError(f"Duplicate knowledge contract ID: {meta['id']}")
        seen.add(meta["id"])
        relations = [
            {"kind": "symbol", "target": "symbol:" + (symbol(name) or name)}
            for name in meta.get("symbols", [])
        ]
        relations.extend(
            {"kind": "diagnostic", "target": "diagnostic:" + name}
            for name in meta.get("rules", [])
        )
        if meta.get("example"):
            example = (examples_root / meta["example"]).resolve()
            inside = example.is_relative_to(examples_root.resolve())
            availability = (
                "blocked"
                if not inside
                else "available"
                if example.is_file()
                else "missing"
            )
            relation = {
                "kind": "example",
                "target": "example:" + quote(meta["example"], safe="/"),
                "path": str(example),
                "availability": availability,
                "discovery": availability,
            }
            if availability == "available":
                relation["sha256"] = feedback.file_identity(example)["sha256"]
            else:
                missing.append(
                    {
                        "reason": "declared_knowledge_invalid"
                        if not inside
                        else "declared_knowledge_missing",
                        "owner": "contract:" + meta["id"],
                        "relation": relation,
                    }
                )
            relations.append(relation)
        conditions = ([meta["when"]] if meta.get("when") else []) + meta.get(
            "candidate", {}
        ).get("conditions", [])
        result.append(
            _object(
                "contract:" + meta["id"],
                "contract",
                meta["title"],
                {
                    **(meta.get("_reference") or feedback.file_identity(path)),
                    "line": meta["line"],
                },
                evidence="maintained_contract",
                text=body,
                conditions=conditions,
                missing_facts=list(meta.get("missing_facts", []))
                or [
                    "Confirm the contract's conditions against the workload, selected"
                    " installation and execution evidence."
                ],
                relations=relations,
                facets=facets(meta),
                aliases=meta.get("aliases", []),
            )
        )
    by_symbol = {row.get("symbol"): row for row in result if row["kind"] in SYMBOL_KINDS}
    for meta, _, _ in entries:
        for name in meta.get("symbols", []):
            row = by_symbol.get(symbol(name) or name)
            if row:
                row["relations"].append(
                    {"kind": "contract", "target": "contract:" + meta["id"]}
                )
    return result, missing


def assemble(
    entries, declarations, examples_root, document_records=(), example_records=()
):
    """Build the query graph once; ambiguous IDs never choose an arbitrary source."""
    records, missing = objects(entries, declarations, examples_root)
    records.extend(document_records)
    records.extend(example_records)
    grouped = {}
    for row in records:
        grouped.setdefault(row["id"], []).append(row)
    issues = [
        {
            "reason": "duplicate_id",
            "id": identity,
            "references": [row["reference"] for row in rows],
        }
        for identity, rows in sorted(grouped.items())
        if len(rows) > 1
    ]
    records = [row for row in records if len(grouped[row["id"]]) == 1]
    by_id = {row["id"]: row for row in records}
    for row in (row for row in records if row["kind"] == "document"):
        if row.get("document_role") == "performance_card":
            for relation in row["relations"]:
                target = by_id.get(relation["target"].split("#", 1)[0])
                if target:
                    target["relations"].append(
                        {
                            "kind": "knowledge_card",
                            "target": row["id"],
                            "relation": relation["kind"],
                            "scope": relation["scope"],
                            "boundary": relation["boundary"],
                            "target_reference": row["reference"],
                        }
                    )
            continue
        for relation in row["relations"]:
            target = by_id.get(relation["target"])
            if target:
                target["relations"].append(
                    {"kind": "documentation", "target": row["id"]}
                )
        for identity, fragment in row["_contracts"].items():
            row["relations"].append({"kind": "contract", "target": identity})
            if identity in by_id:
                by_id[identity]["relations"].append(
                    {
                        "kind": "documentation",
                        "target": row["id"] + ("#" + fragment if fragment else ""),
                    }
                )
    for row in (row for row in records if row["kind"] == "example"):
        for relation in row["relations"]:
            target = by_id.get(relation["target"].split("#", 1)[0])
            if target and relation["kind"] == "uses":
                target.setdefault(
                    "example_lookup", {"text": target["symbol"], "kinds": ["example"]}
                )
                continue
            if target and not any(
                item["kind"] == "example" and item["target"] == row["id"]
                for item in target["relations"]
            ):
                target["relations"].append({"kind": "example", "target": row["id"]})
    provider_files = {
        row["reference"]["path"]: row["reference"]["sha256"]
        for row in records
        if row["kind"] == "contract"
    }
    return records, missing, provider_files, issues


def discover(
    requested,
    entries,
    declarations,
    api_coverage,
    examples_root,
    *,
    checker_sha256,
    source_identity=None,
    document_records=(),
    document_coverage=None,
    document_locator=None,
    example_records=(),
    example_coverage=None,
    example_locator=None,
    metadata_coverage=None,
    assembled=None,
):
    from . import documents, example_programs

    records, missing, provider_files, issues = assembled or assemble(
        entries, declarations, examples_root, document_records, example_records
    )
    by_id = {row["id"]: row for row in records}
    for row in records:
        for relation in row["relations"]:
            if relation["kind"] == "example" and "discovery" in relation:
                relation["discovery"] = (
                    "available"
                    if relation["target"] in by_id
                    else "not_requested"
                    if example_coverage is None
                    else "source_unavailable"
                )
    identities = {
        "references": provider_files,
        "api": api_coverage["files"],
        "symbol_sources": [row["reference"] for row in declarations],
        "examples": [
            relation
            for row in records
            for relation in row["relations"]
            if relation["kind"] == "example"
        ],
        "documents": document_coverage,
        "example_programs": example_coverage,
        "metadata": metadata_coverage,
        "object_issues": issues,
        "checker": checker_sha256,
        "source": source_identity,
    }
    corpus = feedback.digest(json.dumps(identities, sort_keys=True).encode())
    text, filters = requested["text"], requested["filters"]
    exact = symbol(text)
    identity = (
        "symbol:" + exact
        if text.startswith("symbol:") and exact
        else text
        if text.startswith(("symbol:", "contract:"))
        else None
    )
    if document_locator:
        identity = document_locator[0]
    if example_locator:
        identity = example_locator[0]
    meta_by_id = {"contract:" + meta["id"]: meta for meta, _, _ in entries}
    related_routes = {}
    document_routes = []
    declarations_by_name = {row["symbol"]: row for row in declarations}
    name_hints = public_name_hints(text, declarations)
    for meta, _, _ in entries:
        state = applicability(facets(meta), filters)
        if not state["conflicting_filters"] and (
            contract_match(meta, text, exact) if text else state["matched_filters"]
        ):
            document_routes.append("contract:" + meta["id"])
    if filters:
        for meta, _, _ in entries:
            state = applicability(facets(meta), filters)
            if state["matched_filters"] and not state["conflicting_filters"]:
                for name in meta.get("symbols", []):
                    related_routes.setdefault(symbol(name) or name, []).append(
                        "contract:" + meta["id"]
                    )
    matches, rejected = [], []
    tokens = words(text)
    for row in records:
        if row["kind"] not in requested["kinds"]:
            continue
        state = applicability(row["facets"], filters)
        if identity:
            matched = ("id", 1000) if identity == row["id"] else None
        elif row["kind"] == "contract":
            matched = (
                contract_match(meta_by_id[row["id"]], text, exact) if text else None
            )
        elif row["kind"] == "document":
            matched = (
                documents.match(
                    row, text, exact, document_routes, declarations_by_name, name_hints
                )
                if text
                else (
                    ("contract_document", 400)
                    if set(document_routes) & set(row["_contracts"])
                    else None
                )
            )
        elif row["kind"] == "example":
            matched = (
                example_programs.match(row, text, exact, document_routes)
                if text
                else (
                    ("contract_example", 400)
                    if set(document_routes) & set(row["_contracts"])
                    else None
                )
            )
        elif text == row["id"] or exact and exact == row.get("symbol"):
            matched = "symbol", 1000
        elif row["kind"] == "api" and row["symbol"] in name_hints:
            matched = "public_name_hint", name_hints[row["symbol"]] + 2
        elif (
            text
            and not exact
            and row.get("symbol", "").casefold().startswith(text.casefold())
            and text.endswith(".")
        ):
            matched = "namespace_prefix", 500
        elif (
            text
            and not exact
            and not directional(text)
            and tokens
            and tokens <= words(" ".join([row["title"], row.get("summary", "")]))
        ):
            matched = "declaration_terms", 100
        else:
            matched = None
        if text and not matched:
            continue
        routes = related_routes.get(row.get("symbol"), [])
        if row["kind"] == "document":
            routes = sorted(set(document_routes) & set(row["_contracts"]))
        if row["kind"] == "example":
            routes = sorted(set(document_routes) & set(row["_contracts"]))
        if not text and filters and not state["matched_filters"] and not routes:
            continue
        if state["conflicting_filters"]:
            rejected.append(
                {"id": row["id"], "conflicts": state["conflicting_filters"]}
            )
            continue
        if row["kind"] == "example" and (
            conflicts := example_programs.conflicts(row, text)
        ):
            rejected.append(
                {
                    "id": row["id"],
                    "conflicts": {},
                    "source_conditions": conflicts,
                    "reference": row["reference"],
                }
            )
            continue
        reason, score = matched or (
            ("related_contract_filters", 300)
            if routes
            else ("filters" if filters else "browse", 0)
        )
        if not text and filters:
            # Declared conditions precede linked material; relations do not transfer facets.
            score = (
                600
                if state["matched_filters"]
                else 300
                if row["kind"] in SYMBOL_KINDS
                else 200
            )
        score += 10 * len(state["matched_filters"])
        item = {key: value for key, value in row.items() if not key.startswith("_")}
        item.update(
            applicability=state,
            match={
                "reason": reason,
                "score": score,
                "related_contracts": sorted(routes),
            },
        )
        if row["kind"] == "document":
            fragment = (
                document_locator[2]
                if document_locator
                else documents.preferred_fragment(row, text, routes)
            )
            item["resolve"] = {
                "text": row["id"] + ("#" + fragment if fragment else ""),
                "reference_sha256": row["reference"]["sha256"],
            }
        if row["kind"] == "example":
            item["resolve"] = {
                "text": row["id"],
                "reference_sha256": row["reference"]["sha256"],
            }
            if example_locator and example_locator[0] == row["id"]:
                item["program"] = {**item["program"], **row["_details"]}
            else:
                # Discovery offers role/source locators; exact resolution supplies the original statements and full call graph.
                item["conditions"] = []
                item["validation"] = {"status": "not_established"}
                item["negative_controls"] = [
                    {"name": entry["name"], "reference": entry["reference"]}
                    for entry in row["negative_controls"]
                ]
                item["relations"] = [
                    relation
                    for relation in row["relations"]
                    if relation["kind"] != "uses"
                    or exact
                    and relation["target"] == "symbol:" + exact
                ]
        matches.append(item)
    matches.sort(key=lambda row: (-row["match"]["score"], row["id"]))
    signature = feedback.digest(
        json.dumps(
            {
                key: requested[key]
                for key in ("text", "kinds", "filters", "reference_sha256")
            },
            sort_keys=True,
        ).encode()
    )
    start = 0
    if requested["cursor"]:
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(
                    requested["cursor"] + "=" * (-len(requested["cursor"]) % 4)
                )
            )
            if (
                not isinstance(payload, dict)
                or set(payload) != {"schema", "request", "corpus", "after"}
                or payload["schema"] != SCHEMA
            ):
                raise ValueError("Invalid cursor shape")
        except (ValueError, TypeError, KeyError) as error:
            raise QueryError(
                "invalid_request",
                "invalid_cursor",
                "Invalid knowledge cursor; restart this read-only query",
            ) from error
        if payload["request"] != signature or payload["corpus"] != corpus:
            raise QueryError(
                "cursor_invalidated",
                "source_or_query_changed",
                "Invalid or stale knowledge cursor; restart this read-only query",
            )
        try:
            start = [row["id"] for row in matches].index(payload["after"]) + 1
        except (ValueError, TypeError) as error:
            raise QueryError(
                "cursor_invalidated",
                "item_unavailable",
                "Knowledge cursor item is unavailable; restart this read-only query",
            ) from error
    page = matches[start: start + requested["limit"]]
    following = None
    if page and start + len(page) < len(matches):
        value = {
            "schema": SCHEMA,
            "request": signature,
            "corpus": corpus,
            "after": page[-1]["id"],
        }
        following = (
            base64.urlsafe_b64encode(json.dumps(value, sort_keys=True).encode())
            .decode()
            .rstrip("=")
        )
    connected = set(SYMBOL_KINDS) | {"contract", "document", "example"}
    unconnected = sorted(set(requested["kinds"]) - connected)
    relevant_errors = (
        api_coverage["errors"] if set(requested["kinds"]) & set(SYMBOL_KINDS) else []
    )
    if (
        "document" in requested["kinds"]
        and document_coverage
        and not document_coverage["complete"]
    ):
        relevant_errors = [
            *relevant_errors,
            *document_coverage["errors"],
            *api_coverage["errors"],
        ]
    if (
        "example" in requested["kinds"]
        and example_coverage
        and not example_coverage["complete"]
    ):
        relevant_errors = [*relevant_errors, *example_coverage["errors"]]
    relevant_errors = [
        *relevant_errors,
        *issues,
        *(metadata_coverage or {}).get("errors", []),
    ]
    complete = not relevant_errors
    status = (
        "matched"
        if matches
        else "not_integrated"
        if not set(requested["kinds"]) & connected
        else "source_unavailable"
        if relevant_errors
        else "evidence_insufficient"
        if filters and not rejected
        else "retrieval_miss"
    )
    recovery = (
        example_programs.recovery_programs(
            records, text, filters, limit=min(3, requested["limit"])
        )
        if (
            not matches
            and complete
            and status in {"retrieval_miss", "evidence_insufficient"}
            and "example" in requested["kinds"]
        )
        else []
    )
    return {
        "schema": SCHEMA,
        "request": {key: value for key, value in requested.items() if key != "cursor"},
        "lookup": {
            "status": status,
            "knowledge_absence": "not_established",
            "coverage": "complete" if complete else "partial",
        },
        "items": page,
        "page": {
            "total": len(matches),
            "returned": len(page),
            "next_cursor": following,
        },
        "corpus_sha256": corpus,
        "rejected_conditions": rejected,
        "missing_items": missing,
        "coverage": {
            "complete": complete,
            "contracts": "marked_reference_sections",
            "metadata": metadata_coverage or {"complete": True},
            "integrity": {
                "complete": not issues,
                "errors": issues[:8],
                "error_count": len(issues),
                "omitted_errors": max(0, len(issues) - 8),
            },
            "api": {
                **api_coverage,
                "scope": "selected_public_exports_and_public_members; parser internals excluded",
            },
            "documents": document_coverage or {"status": "not_requested"},
            "examples": example_coverage or {"status": "not_requested"},
            "not_integrated_kinds": unconnected,
            "reference_files": provider_files,
        },
        "source_reconciliation": {
            "status": "not_assessed",
            "instruction": (
                "Compare official documentation, selected declarations/docstrings and"
                " source-bound behavior separately; conflicting statements remain"
                " unresolved. No current installation or execution evidence is inferred."
            ),
        },
        **(
            {
                "recovery": {
                    "status": "related_program_sources_only",
                    "items": recovery,
                    "evidence_limit": (
                        "Word overlap only, not a matching result or an applicability"
                        " judgment. Resolve an original program to check its"
                        " conditions, roles and dependencies; unmatched words and"
                        " unstated conditions remain unresolved."
                    ),
                }
            }
            if recovery
            else {}
        ),
        "evidence_limit": (
            "Discovery over declared sources only. Unknown filters and source examples"
            " are not support or execution acceptance; no match is not API absence."
            " Diagnostic and observation adapters are not integrated."
        ),
    }
