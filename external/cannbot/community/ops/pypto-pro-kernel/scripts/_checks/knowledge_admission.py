# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source-bound knowledge admission over the same collectors and matcher as query."""

from __future__ import annotations

from collections import Counter

from . import (
    api_stub,
    catalog,
    coverage,
    documents,
    example_programs,
    feedback,
    knowledge,
    links,
    query,
)
from .paths import EXAMPLES_ROOT, PYPTO_ROOT


def _issue(result, reason, message, *, severity="error", reference=None, target=None):
    location = (
        {"file": reference.get("path"), "line": reference.get("line", 1)}
        if reference
        else None
    )
    result["diagnostics"].append(
        feedback.diagnostic(
            "knowledge-" + reason.replace("_", "-"),
            severity,
            message,
            location=location,
            evidence={"target": target} if target else {},
            next_action="Repair the owning source or selected-source configuration, then repeat read-only admission.",
        )
    )
    if severity == "unverified" and result["status"] == "passed":
        result["status"] = "unverified"


def _provider_errors(result, sources):
    for scope, errors in sources["provider_errors"].items():
        for error in errors:
            invalid = error["reason"] in {
                "metadata_invalid",
                "duplicate_id",
                "duplicate_candidate_id",
                "invalid_example_metadata",
                "SyntaxError",
            }
            _issue(
                result,
                error["reason"],
                scope + ": " + error.get("detail", error["reason"]),
                severity="error" if invalid else "unverified",
                reference=error,
                target=error.get("id"),
            )
    if not sources["api_coverage"]["complete"]:
        _issue(
            result,
            "selected_source_incomplete",
            "The selected public-export provider is incomplete; absence cannot establish a missing API.",
            severity="unverified",
        )


def _relations(result, records, roots, sources):
    by_id = {row["id"]: row for row in records}
    unresolved = Counter()
    unresolved_items = []
    checked = 0
    for row in records:
        for relation in row["relations"]:
            target = relation["target"]
            identity = target.split("#", 1)[0]
            if relation["kind"] == "diagnostic":
                unresolved["diagnostic_adapter_not_integrated"] += 1
                continue
            if (
                relation["kind"] == "member_of"
                and identity in {"symbol:pl", "symbol:vf"}
                and identity not in by_id
            ):
                unresolved["implicit_language_root"] += 1
                continue
            destination = by_id.get(identity)
            if destination is None:
                if relation["kind"] in {"uses", "describes"}:
                    unresolved["inferred_symbol_outside_selected_exports"] += 1
                    references = row.get("_uses", {}).get(
                        identity.removeprefix("symbol:"), [row["reference"]]
                    )
                    unresolved_items.append(
                        {
                            "owner": row["id"],
                            "target": target,
                            "references": references,
                            "status": "not_established",
                            "reason": "inferred_symbol_outside_selected_exports",
                        }
                    )
                    continue
                partial = (
                    identity.startswith("symbol:")
                    and not sources["api_coverage"]["complete"]
                    or identity.startswith(("document:", "docstring:"))
                    and not sources["document_coverage"]["complete"]
                    or identity.startswith("example:")
                    and not sources["example_coverage"]["complete"]
                )
                severity = "unverified" if partial else "error"
                _issue(
                    result,
                    "dangling_relation",
                    "Relation has no unique target in the selected knowledge scope.",
                    severity=severity,
                    reference=row["reference"],
                    target=target,
                )
                continue
            checked += 1
            if identity.startswith(("document:", "docstring:")):
                try:
                    located = documents.locator(target, roots)
                    if located[2]:
                        links.fragment_range(destination["_text"], located[2])
                except ValueError as error:
                    _issue(
                        result,
                        "invalid_fragment",
                        str(error),
                        reference=row["reference"],
                        target=target,
                    )
    for meta, _, _ in sources["entries"]:
        for group in meta.get("review_symbols", []):
            for name in group:
                if "symbol:" + name not in by_id:
                    _issue(
                        result,
                        "unresolved_review_symbol",
                        "The review route's symbol is absent from the selected public-export scope.",
                        severity="error"
                        if sources["api_coverage"]["complete"]
                        else "unverified",
                        reference=meta["_reference"],
                        target="symbol:" + name,
                    )
    return {
        "checked": checked,
        "unresolved": dict(sorted(unresolved.items())),
        "unresolved_items": unresolved_items,
        "scope": (
            "Integrated object relations and Markdown fragments; inferred source"
            " mentions and deferred diagnostic/evidence providers are not API absence."
        ),
    }


def _routes(
    result,
    records,
    sources,
    assembled,
    preview,
    pypto_root,
    examples_root,
    cards_root=None,
):
    by_id = {row["id"]: row for row in records}
    metadata = {"contract:" + meta["id"]: meta for meta, _, _ in sources["entries"]}
    alias_owners, witnesses = {}, []
    for row in records:
        try:
            if row["kind"] == "document":
                located = documents.locator(
                    row["id"],
                    documents.roots(pypto_root, query.REFERENCES_ROOT, cards_root),
                )
                canonical = located[0]
            elif row["kind"] == "example":
                canonical = example_programs.locator(row["id"], examples_root)[0]
            elif row["kind"] in knowledge.SYMBOL_KINDS:
                canonical = "symbol:" + (knowledge.symbol(row["id"]) or "")
            else:
                canonical = row["id"]
            if canonical != row["id"]:
                raise ValueError(
                    "Object ID does not round-trip through the query locator"
                )
        except ValueError as error:
            _issue(
                result,
                "invalid_object_id",
                str(error),
                reference=row["reference"],
                target=row["id"],
            )
        if (
            row["kind"] not in {"contract", "example"}
            and row.get("document_role") != "performance_card"
        ):
            continue
        meta = metadata.get(row["id"], row.get("_meta", {}))
        for alias in [*row.get("aliases", []), *meta.get("discovery_aliases", [])]:
            alias_owners.setdefault(alias.casefold(), []).append(row)
            witnesses.append((row, alias))
            if alias in by_id and alias != row["id"]:
                _issue(
                    result,
                    "shadowed_route",
                    "An alias shadows another exact object ID.",
                    reference=row["reference"],
                    target=alias,
                )
        for key in ("term_groups", "discovery_terms"):
            if meta.get(key):
                witnesses.append((row, " ".join(group[0] for group in meta[key])))
    overlaps = []
    for alias, owners in sorted(alias_owners.items()):
        unique = {row["id"]: row for row in owners}
        if len(unique) > 1:
            overlaps.append(
                {
                    "alias": alias,
                    "owners": sorted(unique),
                    "references": [unique[key]["reference"] for key in sorted(unique)],
                    "status": "conditional_alternatives",
                    "instruction": (
                        "Inspect each owner's conditions; shared routing words do not prove a semantic contradiction."
                    ),
                }
            )
    checked = 0
    for owner, text in witnesses:
        cursor, found, rejected = None, False, False
        while True:
            view = knowledge.discover(
                knowledge.request(
                    text,
                    kinds=[owner["kind"]],
                    limit=knowledge.MAX_LIMIT,
                    cursor=cursor,
                ),
                sources["entries"],
                sources["declarations"],
                sources["api_coverage"],
                examples_root,
                checker_sha256=result["identity"]["checker_sha256"],
                source_identity=result["identity"],
                document_records=sources["document_records"],
                document_coverage=sources["document_coverage"],
                example_records=sources["example_records"],
                example_coverage=sources["example_coverage"],
                metadata_coverage=sources["metadata_coverage"],
                assembled=assembled,
            )
            found = owner["id"] in {row["id"] for row in view["items"]}
            rejected = any(
                row["id"] == owner["id"] for row in view["rejected_conditions"]
            )
            cursor = view["page"]["next_cursor"]
            if found or not cursor:
                break
        checked += 1
        if not found:
            _issue(
                result,
                "excluded_route" if rejected else "unreachable_route",
                "The declared route witness does not retrieve its owner through the shared matcher.",
                severity="advisory" if rejected else "error",
                reference=owner["reference"],
                target=owner["id"],
            )
    return {
        "objects_checked": len(records),
        "declared_routes_checked": checked,
        "routing_overlaps": overlaps,
        "semantic_conflicts": "not_assessed",
        "corpus_sha256": preview["corpus_sha256"],
    }


def _maintenance(result, sources, examples_root):
    root = query.REFERENCES_ROOT
    checks = []

    def findings(name, callback):
        try:
            rows, details = callback()
            for row in rows:
                result["diagnostics"].append(feedback.finding_dict(row))
            checks.append(
                {"name": name, "status": "failed" if rows else "passed", **details}
            )
        except (OSError, SyntaxError, ValueError, SystemExit) as error:
            _issue(result, name + "_unavailable", str(error), severity="unverified")
            checks.append({"name": name, "status": "unverified"})

    def doc_links():
        link_report = links.collect([root])
        return link_report.findings, {
            "documents": len(link_report.documents),
            "local_links": link_report.local_links,
            "checked_fragments": link_report.checked_fragments,
            "external_links_not_fetched": link_report.external_links,
            "non_markdown_fragments_unverified": link_report.unverified_fragments,
        }

    findings("links", doc_links)

    def indexed():
        catalog_report = catalog.collect(
            examples_root, document=root / "tooling/examples.md"
        )
        return catalog_report.findings, {
            "programs": len(catalog_report.examples),
            "indexed": len(catalog_report.indexed),
        }

    findings("catalog", indexed)
    document = root / "tooling/example-coverage.md"
    try:
        assigned = coverage.collect(examples_root, document=document)
        problems = assigned.problems()
        for message in problems:
            _issue(
                result,
                "api_assignment",
                message,
                reference={"path": str(document), "line": 1},
            )
        checks.append(
            {
                "name": "coverage",
                "status": "failed" if problems else "passed",
                "assigned": len(assigned.assigned),
                "written": len(assigned.written),
                "pending": assigned.pending(),
            }
        )
    except (OSError, SyntaxError, ValueError, SystemExit) as error:
        _issue(
            result,
            "coverage_unavailable",
            str(error),
            severity="unverified",
            reference={"path": str(document), "line": 1},
        )
        checks.append({"name": "coverage", "status": "unverified"})
    path = root / "api/pypto_pro_language.pyi"
    if sources["api_coverage"]["complete"]:
        inspected = api_stub.inspect_reference(
            pypto_root=sources["pypto_root"], path=path
        )
        checks.append({"name": "api_freshness", **inspected})
        if inspected["status"] != "passed":
            _issue(
                result,
                "api_" + inspected["reason"],
                inspected["message"],
                severity="unverified"
                if inspected["status"] == "unverified"
                else "error",
                reference={"path": str(path), "line": 1},
            )
        if inspected.get("reference"):
            sources["files"][inspected["reference"]["path"]] = inspected["reference"][
                "sha256"
            ]
    else:
        checks.append(
            {
                "name": "api_freshness",
                "status": "unverified",
                "reason": "selected_source_incomplete",
            }
        )
    return checks


def report(*, pypto_root=PYPTO_ROOT, examples_root=EXAMPLES_ROOT, cards_root=None):
    result = feedback.envelope("knowledge_admission", pypto_root)
    sources = query.collect(
        knowledge.KINDS,
        pypto_root=pypto_root,
        examples_root=examples_root,
        cards_root=cards_root,
    )
    sources["pypto_root"] = pypto_root
    assembled = knowledge.assemble(
        sources["entries"],
        sources["declarations"],
        examples_root,
        sources["document_records"],
        sources["example_records"],
    )
    records, missing, _, identity_errors = assembled
    view = knowledge.discover(
        knowledge.request(None, kinds=knowledge.KINDS, limit=1),
        sources["entries"],
        sources["declarations"],
        sources["api_coverage"],
        examples_root,
        checker_sha256=result["identity"]["checker_sha256"],
        source_identity=result["identity"],
        document_records=sources["document_records"],
        document_coverage=sources["document_coverage"],
        example_records=sources["example_records"],
        example_coverage=sources["example_coverage"],
        metadata_coverage=sources["metadata_coverage"],
        assembled=assembled,
    )
    result["coverage"] = view["coverage"]
    _provider_errors(result, sources)
    for error in identity_errors:
        for reference in error["references"]:
            _issue(
                result,
                "duplicate_id",
                "Ambiguous object ID was quarantined.",
                reference=reference,
                target=error["id"],
            )
    for item in missing:
        owner = next((row for row in records if row["id"] == item["owner"]), None)
        _issue(
            result,
            "missing_companion",
            item["owner"] + ": " + item["reason"],
            reference=owner["reference"] if owner else None,
            target=item["relation"]["target"],
        )
    for row in records:
        if row["kind"] == "example" and row["program"]["missing_roles"]:
            _issue(
                result,
                "missing_program_role",
                ", ".join(row["program"]["missing_roles"]),
                reference=row["reference"],
                target=row["id"],
            )
    relation_report = _relations(
        result,
        records,
        documents.roots(pypto_root, query.REFERENCES_ROOT, cards_root),
        sources,
    )
    routing = _routes(
        result, records, sources, assembled, view, pypto_root, examples_root, cards_root
    )
    gates = _maintenance(result, sources, examples_root)
    query.verify_files(sources["files"])
    result["admission"] = {
        "schema": "pypto-knowledge-admission/1",
        "object_counts": dict(sorted(Counter(row["kind"] for row in records).items())),
        "objects": len(records),
        "relations": relation_report,
        "queryability": routing,
        "maintenance": gates,
        "source_reconciliation": "not_assessed",
        "evidence_limit": (
            "Knowledge admission checks structure, source identity and declared routing"
            " only; it does not establish installed support, numerical correctness, performance or model effectiveness."
        ),
    }
    return feedback.finish(result)
