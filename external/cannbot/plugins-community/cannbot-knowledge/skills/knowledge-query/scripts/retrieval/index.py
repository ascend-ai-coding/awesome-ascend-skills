# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.index — deterministic index build / load / verify (SPEC §3.4, §7).

Byte-reproducible JSON: doc_id by sorted path, json.dump(sort_keys), fixed float
precision, content fingerprint over stripped cards.
"""
import hashlib
import json
import os
import re

from retrieval.config import (B, EXACT_NAME_BONUS, FIELD_WEIGHTS, INDEX_DIR,
                              INDEX_PATH, INDEX_VERSION, K1, ROOT, STAT_PREC,
                              TAG_EXACT_BONUS)
from retrieval.cards import (_front_block, _headings, _parse_front, _signatures,
                             concept_paths, id_to_path, strip_related)
from retrieval.errors import CliError
from retrieval.facets import derive_facets, kind_of
from retrieval.output import emit_stderr, emit_stdout
from retrieval.root import is_probable_knowledge_root, score_knowledge_root
from retrieval.tokenizer import tokenize


def _build_state():
    return {
        "docs": [], "fingerprints": [], "postings": {},
        "field_len_sum": {field: 0 for field in FIELD_WEIGHTS}, "df": {},
        "tag_df": {}, "bundle_doc_count": {}, "bundle_tag_count": {},
        "aliases": {}, "warnings": [],
    }


def _record_fields(state, doc_id, fields):
    field_lengths, seen_terms = {}, set()
    for field, text in fields.items():
        tokens = tokenize(text)
        field_lengths[field] = len(tokens)
        state["field_len_sum"][field] += len(tokens)
        frequencies = {}
        for term in tokens:
            frequencies[term] = frequencies.get(term, 0) + 1
        for term, count in frequencies.items():
            state["postings"].setdefault(term, []).append((doc_id, field, count))
            seen_terms.add(term)
    for term in seen_terms:
        state["df"].setdefault(term, set()).add(doc_id)
    return field_lengths


def _quality_score(frontmatter):
    raw_score = frontmatter.get("quality_score", "")
    try:
        return float(raw_score) if raw_score != "" else None
    except (TypeError, ValueError):
        return None


def _record_aliases(state, aliases, base):
    group = sorted({alias.strip().lower() for alias in aliases if alias.strip()} | ({base} if base else set()))
    if len(group) <= 1:
        return
    for alias in group:
        state["aliases"].setdefault(alias, set()).update(item for item in group if item != alias)


def _index_card(state, doc_id, path):
    with open(id_to_path(path), encoding="utf-8") as card_file:
        raw = card_file.read()
    frontmatter, body = _parse_front(raw)
    body = strip_related(body)
    type_raw = frontmatter.get("type", "")
    kind = kind_of(type_raw, frontmatter.get("kind", ""))
    if kind == "unknown":
        state["warnings"].append(
            "unknown kind (kind=%r type=%r): %s" % (frontmatter.get("kind", ""), type_raw, path)
        )
    bundle, reldir, section, category, base = derive_facets(path)
    tags = frontmatter.get("tags", []) or []
    fields = {
        "title": frontmatter.get("title", ""), "description": frontmatter.get("description", ""),
        "tags": " ".join(tags), "path": path[:-3].replace("/", " "),
        "headings": _headings(body), "signatures": _signatures(body), "body": body,
    }
    field_lengths = _record_fields(state, doc_id, fields)
    state["bundle_doc_count"][bundle] = state["bundle_doc_count"].get(bundle, 0) + 1
    for tag in set(tags):
        state["tag_df"][tag] = state["tag_df"].get(tag, 0) + 1
        key = (bundle, tag)
        state["bundle_tag_count"][key] = state["bundle_tag_count"].get(key, 0) + 1
    title = frontmatter.get("title", "") or base
    names = {re.sub(r"[^0-9a-z]", "", title.lower()), base} - {""}
    aliases = frontmatter.get("aliases", []) or []
    _record_aliases(state, aliases, base)
    state["docs"].append({
        "path": path, "bundle": bundle, "section": section, "category": category,
        "reldir": reldir, "base": base, "title": title,
        "description": frontmatter.get("description", ""), "tags": tags,
        "type_raw": type_raw, "kind": kind, "resource": frontmatter.get("resource", ""),
        "schema_version": frontmatter.get("schema_version", ""),
        "source_family": frontmatter.get("source_family", ""),
        "created_at": frontmatter.get("created_at", ""), "updated_at": frontmatter.get("updated_at", ""),
        "source_updated_at": frontmatter.get("source_updated_at", ""),
        "field_lengths": field_lengths, "names": sorted(names), "aliases": aliases,
        "paradigms": frontmatter.get("paradigms", []) or [],
        "platforms": frontmatter.get("platforms", []) or [],
        "severity": frontmatter.get("severity", ""), "confidence": frontmatter.get("confidence", ""),
        "status": frontmatter.get("status", ""), "quality_score": _quality_score(frontmatter),
    })
    fingerprint_text = path + "\0" + _front_block(raw) + "\0" + body
    digest = hashlib.sha1(fingerprint_text.encode("utf-8")).hexdigest()
    state["fingerprints"].append((path, digest))


def _assembled_index(state):
    docs = state["docs"]
    count = len(docs)
    postings = {term: sorted(values) for term, values in state["postings"].items()}
    state["fingerprints"].sort()
    fingerprint = hashlib.sha1(
        "\n".join("%s\t%s" % row for row in state["fingerprints"]).encode("utf-8")
    ).hexdigest()
    bundles = sorted({doc["bundle"] for doc in docs})
    global_tags = sorted({
        tag for (bundle, tag), tag_count in state["bundle_tag_count"].items()
        if tag_count == state["bundle_doc_count"].get(bundle, 0)
    })
    index = {
        "meta": {
            "index_version": INDEX_VERSION,
            "card_count": count,
            "bundles": bundles,
            "content_fingerprint": fingerprint,
            "field_weights": FIELD_WEIGHTS,
            "k1": K1, "b": B,
            "exact_name_bonus": EXACT_NAME_BONUS, "tag_exact_bonus": TAG_EXACT_BONUS,
            "merge": "sum",
            "global_tags": global_tags,
            "alias_map": {alias: sorted(values) for alias, values in state["aliases"].items()},
        },
        "docs": docs,
        "postings": postings,
        "df": {term: len(doc_ids) for term, doc_ids in state["df"].items()},
        "tag_df": dict(sorted(state["tag_df"].items())),
        "avg_field_len": {
            field: round(state["field_len_sum"][field] / count, STAT_PREC) if count else 0.0
            for field in FIELD_WEIGHTS
        },
    }
    return index, bundles, postings


def _write_index(index):
    os.makedirs(INDEX_DIR, exist_ok=True)
    tmp_path = "%s.%s.tmp" % (INDEX_PATH, os.getpid())
    with open(tmp_path, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False, sort_keys=True, indent=2)
        index_file.write("\n")
    os.replace(tmp_path, INDEX_PATH)


def cmd_build(quiet=False, allow_empty=False):
    paths = concept_paths()
    if not paths and not allow_empty:
        raise CliError(
            "no concept cards found under %s; refusing to write an empty index. "
            "Check CANNBOT_KNOWLEDGE_ROOT/--knowledge-root, or pass --allow-empty "
            "for an intentional empty knowledge base." % ROOT
        )
    state = _build_state()
    for doc_id, path in enumerate(paths):
        _index_card(state, doc_id, path)
    index, bundles, postings = _assembled_index(state)
    _write_index(index)
    for warning in state["warnings"]:
        emit_stderr("WARN: " + warning)
    msg = "built index: %d docs, %d bundles, %d terms -> %s" % (
        len(state["docs"]), len(bundles), len(postings), os.path.relpath(INDEX_PATH, ROOT)
    )
    if quiet:
        emit_stderr(msg)
    else:
        emit_stdout(msg)


def _read_index():
    with open(INDEX_PATH, encoding="utf-8") as index_file:
        return json.load(index_file)


def _index_stale_reasons(idx):
    reasons = []
    meta = idx.get("meta", {})
    if meta.get("index_version") != INDEX_VERSION:
        reasons.append(
            "version mismatch: index=%r current=%r"
            % (meta.get("index_version"), INDEX_VERSION)
        )
    corpus = set(concept_paths())
    indexed = {d.get("path") for d in idx.get("docs", [])}
    if corpus != indexed:
        reasons.append("doc set mismatch: +%d -%d" % (len(corpus - indexed), len(indexed - corpus)))
    if corpus == indexed:
        expected_fp = _recompute_fingerprint(sorted(corpus))
        if expected_fp != meta.get("content_fingerprint"):
            reasons.append(
                "content fingerprint stale: index=%s current=%s"
                % (str(meta.get("content_fingerprint", ""))[:12], expected_fp[:12])
            )
    return reasons


def _runtime_index(idx):
    """Attach the in-memory posting pivots required by retrieval routes."""
    pt, docs_with = {}, {}
    for term, postings in idx["postings"].items():
        per = {}
        doc_ids = set()
        for doc_id, field, term_frequency in postings:
            per.setdefault(doc_id, {})[field] = term_frequency
            doc_ids.add(doc_id)
        pt[term] = per
        docs_with[term] = doc_ids
    idx["_pt"] = pt
    idx["_docs_with"] = docs_with
    return idx


def load_index(auto_build=True):
    if not os.path.exists(INDEX_PATH):
        score, markers = score_knowledge_root(ROOT)
        if auto_build and is_probable_knowledge_root(ROOT):
            emit_stderr(
                "no index at %s; auto-building for knowledge base root %s (markers: %s)"
                % (INDEX_PATH, ROOT, ", ".join(markers) or "none")
            )
            cmd_build(quiet=True)
        else:
            raise CliError(
                "no index at %s; knowledge base root %s does not look ready "
                "(score=%d, markers=%s). Set CANNBOT_KNOWLEDGE_ROOT or run "
                "`knowledge_query.py --knowledge-root <path> build`."
                % (INDEX_PATH, ROOT, score, ", ".join(markers) or "none")
            )
    idx = _read_index()
    stale = _index_stale_reasons(idx)
    if not stale:
        return _runtime_index(idx)
    score, markers = score_knowledge_root(ROOT)
    if not auto_build or not is_probable_knowledge_root(ROOT):
        raise CliError(
            "stale index at %s and knowledge base root %s is not safe to auto-build "
            "(score=%d, markers=%s): %s"
            % (INDEX_PATH, ROOT, score, ", ".join(markers) or "none", "; ".join(stale))
        )
    emit_stderr(
        "stale index at %s; auto-rebuilding for knowledge base root %s (%s)"
        % (INDEX_PATH, ROOT, "; ".join(stale))
    )
    cmd_build(quiet=True)
    idx = _read_index()
    stale = _index_stale_reasons(idx)
    if stale:
        raise CliError("index remains stale after rebuild: %s" % "; ".join(stale))
    return _runtime_index(idx)


def _recompute_fingerprint(paths):
    rows = []
    for path in paths:
        with open(id_to_path(path), encoding="utf-8") as card_file:
            raw = card_file.read()
        _, body = _parse_front(raw)
        body = strip_related(body)
        fp_text = path + "\0" + _front_block(raw) + "\0" + body
        rows.append((path, hashlib.sha1(fp_text.encode("utf-8")).hexdigest()))
    rows.sort()
    return hashlib.sha1(
        "\n".join("%s\t%s" % (p, h) for p, h in rows).encode("utf-8")
    ).hexdigest()


def _schema_errors(corpus):
    errors = []
    for path in corpus:
        with open(id_to_path(path), encoding="utf-8") as card_file:
            frontmatter, _ = _parse_front(card_file.read())
        for key in ("schema_version", "kind", "type", "source_family", "updated_at"):
            if not frontmatter.get(key):
                errors.append("missing frontmatter %s: %s" % (key, path))
    return errors


def _strict_errors(docs):
    errors = []
    for doc in docs:
        if doc["kind"] == "unknown":
            errors.append(
                "unknown kind: %s (kind=%r type=%r)"
                % (doc["path"], doc.get("kind", ""), doc["type_raw"])
            )
        resource = doc.get("resource", "")
        if resource and not re.match(r"https?://", resource):
            errors.append("bad resource: %s -> %s" % (doc["path"], resource))
    return errors


def _verify_errors(index, level, corpus, indexed):
    errors = []
    meta = index.get("meta", {})
    if meta.get("index_version") != INDEX_VERSION:
        errors.append("index version mismatch: index=%r current=%r" % (meta.get("index_version"), INDEX_VERSION))
    if corpus != indexed:
        errors.append("doc set mismatch: +%d -%d" % (len(corpus - indexed), len(indexed - corpus)))
    expected_fp = _recompute_fingerprint(sorted(corpus))
    if expected_fp != index["meta"]["content_fingerprint"]:
        errors.append(
            "content fingerprint stale: index=%s current=%s"
            % (index["meta"]["content_fingerprint"][:12], expected_fp[:12])
        )
    if level in ("schema", "strict"):
        errors.extend(_schema_errors(corpus))
    if level == "strict":
        errors.extend(_strict_errors(index["docs"]))
    return errors


def _emit_verify_json(level, index, indexed, errors, *, shown, omitted):
    emit_stdout(json.dumps({
        "level": level,
        "ok": not errors,
        "indexed_concepts": len(indexed),
        "bundles": index["meta"]["bundles"],
        "failure_count": len(errors),
        "failures": shown,
        "omitted_failure_count": omitted,
        "index_path": INDEX_PATH,
    }, ensure_ascii=False, indent=2))


def _emit_verify_text(level, index, indexed, errors, *, shown, omitted):
    emit_stdout("=== verify (%s) ===" % level)
    emit_stdout("indexed concepts: %s  bundles: %s" % (len(indexed), index["meta"]["bundles"]))
    for error in shown:
        emit_stdout("  FAIL: %s" % error)
    if not errors:
        emit_stdout("OK")
        return
    if omitted:
        emit_stdout("  ... %d more failures omitted; use --limit %d to show all" % (omitted, len(errors)))
    emit_stdout("FAILURES: %s" % len(errors))


def cmd_verify(level="index", limit=50, emit_json=False):
    if not os.path.exists(INDEX_PATH):
        raise CliError("no index — run build first")
    index = _read_index()
    corpus = set(concept_paths())
    indexed = {doc["path"] for doc in index["docs"]}
    errors = _verify_errors(index, level, corpus, indexed)

    shown = errors if limit is None or limit < 0 else errors[:limit]
    omitted = max(0, len(errors) - len(shown))
    if emit_json:
        _emit_verify_json(level, index, indexed, errors, shown=shown, omitted=omitted)
        if errors:
            raise CliError(code=1)
        return
    _emit_verify_text(level, index, indexed, errors, shown=shown, omitted=omitted)
    if errors:
        raise CliError(code=1)
