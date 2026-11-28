# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read maintained scenario sections and live signatures without a second index."""

from __future__ import annotations

import ast
from pathlib import Path

from . import (
    actions,
    api_stub,
    documents,
    example_programs,
    feedback,
    host_io,
    knowledge,
    performance_cards,
    query_metadata,
)
from .paths import REFERENCES_ROOT, EXAMPLES_ROOT, PYPTO_ROOT


def _match_groups(groups, text):
    return knowledge.match_groups(groups, text)


def sections():
    yield from query_metadata.collect(REFERENCES_ROOT)[0]


def collect(
    kinds,
    *,
    pypto_root=PYPTO_ROOT,
    examples_root=EXAMPLES_ROOT,
    document_target=None,
    cards_root=None,
):
    """The same source snapshot feeds discovery and admission; there is no persistent index."""
    available, metadata_coverage = query_metadata.collect(REFERENCES_ROOT)
    files = metadata_coverage.pop("_files")
    errors = {"metadata": metadata_coverage.pop("_errors")}
    declarations, api_coverage = api_stub.public_symbols(pypto_root)
    additional = [
        {
            str((pypto_root / path).resolve()): digest
            for path, digest in api_coverage["files"].items()
        }
    ]
    document_records, document_coverage, example_records, example_coverage = (
        [],
        None,
        [],
        None,
    )
    document_files = {}
    card_target = document_target and document_target[0].startswith(
        "document:perf-cards/"
    )
    if {"document", "example"} & set(kinds) and not card_target:
        document_records, document_coverage = documents.collect(
            pypto_root,
            REFERENCES_ROOT,
            declarations,
            available,
            api_coverage,
            exact_target=document_target,
        )
        document_files = document_coverage.pop("_files")
        additional.append(document_files)
        errors["documents"] = document_coverage.pop("_errors")
    if cards_root is not None:
        card_records, card_coverage = performance_cards.collect(
            cards_root, references_root=REFERENCES_ROOT, entries=available
        )
        card_files = card_coverage.pop("_files")
        additional.append(card_files)
        errors["cards"] = card_coverage.pop("_errors")
        document_records.extend(card_records)
        document_coverage = document_coverage or {"complete": True, "scopes": {}}
        document_errors = [*errors.get("documents", []), *errors["cards"]]
        manifest = feedback.digest(
            (
                document_coverage.get("manifest_sha256", "")
                + card_coverage["manifest_sha256"]
            ).encode()
        )
        document_coverage = {
            **document_coverage,
            "complete": document_coverage["complete"] and card_coverage["complete"],
            "cards": card_coverage,
            "errors": document_errors[:8],
            "error_count": len(document_errors),
            "omitted_errors": max(0, len(document_errors) - 8),
            "manifest_sha256": manifest,
            "files_read": len(document_files.keys() | card_files.keys()),
            "files_seen": document_coverage.get("files_seen", 0)
            + card_coverage["files_seen"],
            "scan_bytes": document_coverage.get("scan_bytes", 0)
            + card_coverage["scan_bytes"],
            "scopes": {
                **document_coverage["scopes"],
                performance_cards.SCOPE: {
                    "root": card_coverage["root"],
                    "role": performance_cards.ROLE,
                    "complete": card_coverage["complete"],
                    "files_read": card_coverage["cards_read"],
                },
            },
        }
    if "example" in kinds:
        example_records, example_coverage = example_programs.collect(
            examples_root, REFERENCES_ROOT, available, document_records
        )
        additional.append(example_coverage.pop("_files"))
        errors["examples"] = example_coverage.pop("_errors")
    for group in additional:
        for path, digest in group.items():
            if path in files and files[path] != digest:
                raise knowledge.QueryError(
                    "source_changed",
                    "source_changed_during_collection",
                    "Knowledge sources changed during discovery; restart this read-only query",
                )
            files[path] = digest
    return {
        "entries": available,
        "declarations": declarations,
        "api_coverage": api_coverage,
        "document_records": document_records,
        "document_coverage": document_coverage,
        "example_records": example_records,
        "example_coverage": example_coverage,
        "metadata_coverage": metadata_coverage,
        "files": files,
        "provider_errors": errors,
    }


def verify_files(files):
    for path, digest in files.items():
        try:
            current = feedback.file_identity(path)["sha256"]
        except OSError:
            current = None
        if current != digest:
            raise knowledge.QueryError(
                "source_changed",
                "source_changed_during_collection",
                "Knowledge sources changed during discovery; restart this read-only query",
            )


def failure(error, *, pypto_root=PYPTO_ROOT):
    result = feedback.envelope("query", pypto_root)
    result["status"] = "unverified"
    status = getattr(
        error,
        "status",
        "source_unavailable" if isinstance(error, OSError) else "invalid_request",
    )
    reason = getattr(
        error,
        "reason",
        "source_unavailable" if isinstance(error, OSError) else "invalid_request",
    )
    result["knowledge"] = {
        "schema": knowledge.SCHEMA,
        "lookup": {
            "status": status,
            "reason": reason,
            "message": str(error),
            "knowledge_absence": "not_established",
            "coverage": "not_assessed",
        },
        "items": [],
        "page": {"total": 0, "returned": 0, "next_cursor": None},
        "coverage": {"complete": False},
        "evidence_limit": "The request could not complete; this is not a complete retrieval miss or API absence.",
    }
    result["diagnostics"].append(
        feedback.diagnostic(
            "knowledge-query-" + reason.replace("_", "-"),
            "unverified",
            str(error),
            next_action="Correct the request or rediscover current sources; restart an invalidated read-only cursor.",
        )
    )
    return result


def context(ids, *, examples_root=EXAMPLES_ROOT, include_example=False, entries=None):
    result = []
    for meta, path, text in sections() if entries is None else entries:
        if meta["id"] not in ids:
            continue
        item = {
            "id": meta["id"],
            "reference": {
                **(meta.get("_reference") or feedback.file_identity(path)),
                "line": meta["line"],
            },
            "text": text,
            "examples": [],
        }
        if meta.get("example"):
            example = (examples_root / meta["example"]).resolve()
            try:
                if not example.is_relative_to(examples_root.resolve()):
                    raise ValueError("Companion example escapes the selected root")
                data = host_io._bytes(example, host_io.CEILINGS["max_file_bytes"])
                ref = {"path": str(example.resolve()), "sha256": feedback.digest(data)}
                if include_example:
                    ref["text"] = data.decode("utf-8")
                item["examples"].append(ref)
            except (OSError, ValueError) as error:
                item["example_unavailable"] = str(example)
                item["example_unavailable_reason"] = str(error)
        result.append(item)
    return result


def ids_for_rules(rules):
    """Reference owners declare which existing diagnostic boundaries they explain."""
    rules = set(rules)
    return [
        meta["id"]
        for meta, _, _ in sections()
        if rules.intersection(meta.get("rules", []))
    ]


def review_routes(sources, *, pypto_root=PYPTO_ROOT):
    """Offer owner-declared reviews for source references, never infer a cause."""
    from . import api_policy

    calls = []
    for source in sources:
        identity = feedback.file_identity(source)
        observed = []
        api_policy.inspect(source, pypto_root=pypto_root, observed_calls=observed)
        if feedback.file_identity(source) != identity:
            continue
        calls.extend(
            {**call, "file": identity["path"], "source_sha256": identity["sha256"]}
            for call in observed
        )
    symbols = {call["symbol"] for call in calls}
    routes = []
    for meta, path, _ in sections():
        groups = meta.get("review_symbols", [])
        if not groups or not all(symbols.intersection(group) for group in groups):
            continue
        relevant = set().union(*map(set, groups))
        sites = [call for call in calls if call["symbol"] in relevant]
        routes.append(
            {
                "id": meta["id"],
                "title": meta["title"],
                "when": meta["when"],
                "query": meta["id"],
                "reference": {**feedback.file_identity(path), "line": meta["line"]},
                "source_matches": sites[:4],
                "omitted_source_matches": max(0, len(sites) - 4),
            }
        )
    return routes[:3], max(0, len(routes) - 3)


def signatures(symbols, pypto_root, *, declarations=None):
    if declarations is None:
        declarations, _ = api_stub.public_symbols(pypto_root)
    names = {knowledge.symbol(name) or name for name in symbols}
    return [
        {"signature": row["declaration"], "reference": row["reference"]}
        for row in declarations
        if row["kind"] == "api" and row["symbol"] in names
    ]


def store_combinations(pypto_root):
    """Read the selected frontend's declared non-quantized Acc-to-GM table."""
    path = pypto_root / "python/pypto_pro/ir/op/block_ops.py"
    result = {
        "status": "unverified",
        "scope": (
            "Selected Python source declarations only; installed runtime, compilation"
            " and performance are not established."
        ),
        "table": "_A5_STORE_COMBOS",
        "rows": [],
    }
    try:
        with path.open("rb") as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Selected frontend source exceeds the read limit.")
        tree = ast.parse(data)
        declarations = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                declarations.setdefault(node.targets[0].id, []).append(node.value)

        def declaration(name):
            nodes = declarations.get(name, [])
            if len(nodes) != 1:
                raise ValueError(f"No unique literal declaration for {name}.")
            return nodes[0]

        def field(node, name):
            values = []
            if isinstance(node, ast.Dict):
                values = [
                    value
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and key.value == name
                ]
            if len(values) != 1:
                raise ValueError(f"No unique literal mapping for {name}.")
            return values[0]

        stores = field(declaration("_LAYOUT_DTYPE_COMBOS"), "store")
        bindings = (
            [
                value
                for key, value in zip(stores.keys, stores.values)
                if (isinstance(key, ast.Constant) and key.value == "a5")
                or (
                    isinstance(key, ast.Attribute)
                    and key.attr == "DAV_3510"
                    and isinstance(key.value, ast.Name)
                    and key.value.id == "NpuArch"
                )
            ]
            if isinstance(stores, ast.Dict)
            else []
        )
        if len(bindings) != 1:
            raise ValueError(
                "No unique literal store mapping for a5 / NpuArch.DAV_3510."
            )
        binding = bindings[0]
        if isinstance(binding, ast.Tuple):
            if (
                len(binding.elts) != 2
                or not isinstance(binding.elts[1], ast.Name)
                or binding.elts[1].id != "_A5_STORE_QUANT_COMBOS"
            ):
                raise ValueError("Unrecognized ordinary/quantized store table pair.")
            binding = binding.elts[0]
        if not isinstance(binding, ast.Name) or binding.id != result["table"]:
            raise ValueError(
                "The declared store binding does not select the ordinary store table."
            )
        table = declaration(result["table"])
        if not isinstance(table, ast.Tuple):
            raise ValueError("The selected table is not a literal tuple.")
        rows, row_lines = [], []
        for row in table.elts:
            if (
                not isinstance(row, ast.Tuple)
                or len(row.elts) != 6
                or not all(
                    isinstance(node, ast.Constant) and isinstance(node.value, str)
                    for node in row.elts[:2]
                )
            ):
                raise ValueError(
                    "Unrecognized table row; no support inference is available."
                )
            if [node.value for node in row.elts[:2]] != ["Acc", "GM"]:
                continue
            values = []
            for node, owner in zip(
                row.elts[2:], ("TensorLayout", "TensorLayout", "DataType", "DataType")
            ):
                if (
                    not isinstance(node, ast.Attribute)
                    or not isinstance(node.value, ast.Name)
                    or node.value.id != owner
                ):
                    raise ValueError("Unrecognized layout/dtype declaration.")
                values.append(node.attr)
            rows.append(values)
            row_lines.append(row.lineno)
        result.update(
            status="observed_source",
            reference={
                "path": str(path.resolve()),
                "sha256": feedback.digest(data),
                "line": row_lines[0] if row_lines else table.lineno,
            },
            declaration_line=table.lineno,
            row_lines=row_lines[:32],
            columns=[
                "source_layout",
                "destination_layout",
                "source_dtype",
                "destination_dtype",
            ],
            rows=rows[:32],
            omitted_rows=max(0, len(rows) - 32),
        )
    except (OSError, SyntaxError, ValueError) as exc:
        result["reason"] = str(exc)
    return result


def report(
    text=None,
    *,
    source=None,
    pypto_root=PYPTO_ROOT,
    examples_root=EXAMPLES_ROOT,
    goal=None,
    kinds=None,
    filters=None,
    limit=None,
    cursor=None,
    reference_sha256=None,
    cards_root=None,
):
    actions.goal(goal)
    requested = knowledge.request(text, kinds, filters, limit, cursor, reference_sha256)
    text = requested["text"]
    object_view = any(
        value is not None for value in (kinds, filters, limit, cursor, reference_sha256)
    ) or text.startswith(
        ("symbol:", "contract:", "document:", "docstring:", "example:")
    )
    result = feedback.envelope("query", pypto_root, [source] if source else [])
    chosen = []
    matched = []
    exact = knowledge.symbol(text)
    symbols = {exact} if exact else set()
    selected_roots = documents.roots(pypto_root, REFERENCES_ROOT, cards_root)
    document_locator = documents.locator(text, selected_roots)
    sources = collect(
        requested["kinds"],
        pypto_root=pypto_root,
        examples_root=examples_root,
        document_target=document_locator if "document" in requested["kinds"] else None,
        cards_root=cards_root,
    )
    available, declarations, api_coverage = (
        sources["entries"],
        sources["declarations"],
        sources["api_coverage"],
    )
    document_records, document_coverage = (
        sources["document_records"],
        sources["document_coverage"],
    )
    example_locator = example_programs.locator(text, examples_root)
    example_records, example_coverage = (
        sources["example_records"],
        sources["example_coverage"],
    )
    assembled = knowledge.assemble(
        available, declarations, examples_root, document_records, example_records
    )
    result["knowledge"] = knowledge.discover(
        requested,
        available,
        declarations,
        api_coverage,
        examples_root,
        checker_sha256=result["identity"]["checker_sha256"],
        source_identity=result["identity"],
        document_records=document_records,
        document_coverage=document_coverage,
        document_locator=document_locator,
        example_records=example_records,
        example_coverage=example_coverage,
        example_locator=example_locator,
        metadata_coverage=sources["metadata_coverage"],
        assembled=assembled,
    )
    ambiguous = {issue["id"] for issue in assembled[3]}
    if cards_root is not None and sources["provider_errors"].get("cards"):
        result["status"] = "unverified"
        result["diagnostics"].append(
            feedback.diagnostic(
                "knowledge-cards-unverified",
                "unverified",
                "The configured Active card source is incomplete; inspect"
                " documents.cards coverage. Invalid cards were quarantined.",
            )
        )
    declarations = [row for row in declarations if "symbol:" + row["symbol"] not in ambiguous]
    if (
        not sources["metadata_coverage"]["complete"]
        or not result["knowledge"]["coverage"]["integrity"]["complete"]
    ):
        result["diagnostics"].append(
            feedback.diagnostic(
                "knowledge-source-partial",
                "advisory",
                "Invalid or ambiguous owners were quarantined; inspect"
                " metadata/integrity coverage before treating a miss as complete.",
            )
        )
    if example_locator and "example" in requested["kinds"]:
        row = next(
            (item for item in assembled[0] if item["id"] == example_locator[0]), None
        )
        if row:
            located = example_programs.read_object(
                row, examples_root, pypto_root, reference_sha256
            )
            if located["status"] == "matched":
                result["knowledge"]["content"] = located["content"]
            else:
                result["knowledge"]["lookup"].update(located)
                result["status"] = "unverified"
        elif not example_locator[1].is_file():
            result["knowledge"]["lookup"].update(
                status="source_unavailable",
                reason="Example kernel is missing or unreadable",
            )
            result["status"] = "unverified"
    if document_locator and "document" in requested["kinds"]:
        row = next(
            (item for item in assembled[0] if item["id"] == document_locator[0]), None
        )
        if row:
            located = documents.read_object(
                row, text, selected_roots, pypto_root, reference_sha256
            )
            if located["status"] == "matched":
                result["knowledge"].update(
                    content=located["content"], sections=located["sections"]
                )
            else:
                if "sections" in located:
                    result["knowledge"]["sections"] = located.pop("sections")
                result["knowledge"]["lookup"].update(located)
                result["status"] = "unverified"
        elif document_locator[1] is not None:
            try:
                document_locator[1].stat()
                if not document_locator[1].is_file():
                    result["knowledge"]["lookup"].update(
                        status="source_unavailable",
                        reason="not_regular_file",
                        path=str(document_locator[1]),
                    )
            except OSError as error:
                result["knowledge"]["lookup"].update(
                    status="source_unavailable",
                    reason=type(error).__name__,
                    path=str(document_locator[1]),
                )
    page_ids = {row["id"] for row in result["knowledge"]["items"]}
    related = []
    for meta, _, _ in sorted(available, key=lambda entry: entry[0]["id"]):
        state = knowledge.applicability(knowledge.facets(meta), requested["filters"])
        if (
            "contract" not in requested["kinds"]
            or state["conflicting_filters"]
            or object_view
            and "contract:" + meta["id"] not in page_ids
        ):
            continue
        match = knowledge.contract_match(meta, text, exact)
        if match and match[0] in {"id", "alias", "scenario"}:
            chosen.append(meta["id"])
            matched.append(meta)
            symbols.update(meta.get("symbols", []))
        elif match:
            related.append(
                {
                    "id": meta["id"],
                    "title": meta["title"],
                    "when": meta.get(
                        "when",
                        "Read when this contract applies to the current operation.",
                    ),
                    "query": meta["id"],
                    "match": "symbol" if match[0] == "symbol" else "discovery",
                    "reference": {**meta["_reference"], "line": meta["line"]},
                }
            )
    result["context"] = (
        context(
            chosen[:2],
            examples_root=examples_root,
            include_example=True,
            entries=available,
        )
        if not object_view
        else []
    )
    if object_view:
        symbols = {
            row["symbol"]
            for row in result["knowledge"]["items"]
            if row["kind"] == "api"
        }
    result["signatures"] = signatures(symbols, pypto_root, declarations=declarations)
    result["related_contracts"] = related[:3]
    result["omitted_related_contracts"] = max(0, len(related) - 3)
    result["missing_facts"] = []
    if "acc-final-output" in chosen[:2]:
        table = store_combinations(pypto_root)
        expected = result["identity"]["selected_source"]["files"].get(
            "python/pypto_pro/ir/op/block_ops.py"
        )
        if (
            table["status"] == "observed_source"
            and table["reference"]["sha256"] != expected
        ):
            table.update(
                status="unverified",
                rows=[],
                row_lines=[],
                reason="Selected source changed during this query.",
            )
        result["selected_frontend_store"] = table
    if chosen:
        result["missing_facts"] = (
            ["Confirm the selected installation against the source signatures."]
            if result["signatures"]
            else []
        )
        if "ub-l1" in chosen:
            result["missing_facts"].append(
                "Bind physical shape, valid extent, offset, producer address map and"
                " consumer synchronization to the concrete workload."
            )
        for meta in matched[:2]:
            result["missing_facts"].extend(meta.get("missing_facts", []))
        if any(meta.get("candidate") for meta in matched[:2]):
            result["missing_facts"].append(
                "Confirm applicability, realized work and generated code; measure the"
                " complete call on the current workload."
            )
    if source:
        from . import ub_l1, group_depth

        findings, facts, coverage = ub_l1.inspect(source, pypto_root=pypto_root)
        matched_sites = [
            fact
            for fact in facts
            if exact in {"pl.move", "pl.insert"}
            and fact["api"] == exact.split(".")[-1]
            and fact["source"].get("space") == "Vec"
            and fact["destination"].get("space") == "Mat"
        ]
        if matched_sites and "ub-l1" not in chosen and not object_view:
            contexts = context(
                ["ub-l1"],
                examples_root=examples_root,
                include_example=True,
                entries=available,
            )
            for owner in contexts:
                owner["matched_sites"] = [
                    {"file": site["file"], "line": site["line"]}
                    for site in matched_sites[:4]
                ]
            result["context"].extend(contexts)
            result["signatures"] = signatures(
                symbols | {"pl.move", "pl.insert"},
                pypto_root,
                declarations=declarations,
            )
            result["related_contracts"] = [route for route in result["related_contracts"] if route["id"] != "ub-l1"]
        depth_findings, depth_coverage = group_depth.inspect(
            source, pypto_root=pypto_root
        )
        findings += depth_findings
        result["transfers"] = facts[:12]
        result["omitted_transfer_sites"] = max(0, len(facts) - 12)
        result["coverage"]["ub_l1"] = coverage
        result["coverage"]["group_depth"] = depth_coverage
        result["diagnostics"].extend(feedback.finding_dict(f) for f in findings)
        if depth_findings and "tile-group-depth" not in chosen and not object_view:
            result["context"].extend(
                context(
                    ["tile-group-depth"], examples_root=examples_root, entries=available
                )
            )
    if result["related_contracts"] and not result["context"]:
        result["missing_facts"].append(
            "Select a route whose producer, consumer and conditions match; discovery"
            " words do not establish an API declaration, native support or a root cause."
        )
    if (
        result["knowledge"]["lookup"]["status"] != "matched"
        and not chosen
        and not result["signatures"]
        and not result["related_contracts"]
    ):
        result["status"] = "unverified"
        result["diagnostics"].append(
            feedback.diagnostic(
                "query-no-match",
                "unverified",
                (
                    (
                        "Document resolution: "
                        + result["knowledge"]["lookup"]["status"]
                        + ". "
                        + result["knowledge"]["lookup"].get("reason", "")
                    )
                    if document_locator
                    else (
                        "No maintained scenario, document, example program or exact declaration matched this query."
                    )
                ),
                next_action=(
                    "Inspect the returned coverage/sections or use an exact public"
                    " symbol; absence here is not API absence."
                ),
            )
        )
    for row in result["knowledge"]["items"]:
        if row["kind"] == "contract" and (
            not object_view or row["match"]["reason"] not in {"id", "alias", "scenario"}
        ):
            row.pop("text", None)
        row.setdefault("resolve", {"text": row["id"]})
    verify_files(sources["files"])
    for row in result["knowledge"]["items"]:
        for relation in row["relations"]:
            if (
                relation["kind"] == "example"
                and relation.get("availability") == "available"
            ):
                if (
                    feedback.file_identity(Path(relation["path"]))["sha256"]
                    != relation["sha256"]
                ):
                    raise ValueError(
                        "Companion example changed during discovery; restart this read-only query"
                    )
    feedback.finish(result)
    actions.for_report(result, goal)
    return result
