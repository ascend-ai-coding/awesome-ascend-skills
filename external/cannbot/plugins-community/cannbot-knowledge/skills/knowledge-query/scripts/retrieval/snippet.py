# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.snippet - short evidence snippets for search/recall hits."""
from __future__ import annotations

import re

from retrieval.cards import _parse_front, id_to_path, strip_related
from retrieval.tokenizer import tokenize


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_heading(text: str) -> bool:
    return text.lstrip().startswith("#")


def _looks_like_tag_line(text: str) -> bool:
    value = text.strip()
    if len(value) > 180:
        return False
    if any(ch in value for ch in ",.，。:：;；()（）[]【】`"):
        return False
    words = value.split()
    if len(words) < 4:
        return False
    tokenish = 0
    for word in words:
        if re.fullmatch(r"[0-9A-Za-z_+\-.]+", word) or "_" in word or "-" in word:
            tokenish += 1
    return tokenish >= max(4, len(words) - 1)


def _line_quality(text: str, source: str) -> float:
    value = text.strip()
    quality = 0.0
    if source == "description":
        quality += 4.0
    if _is_heading(value):
        quality -= 4.0
    if _looks_like_tag_line(value):
        quality -= 5.0
    if len(value) < 20:
        quality -= 1.0
    if any(ch in value for ch in "。；，:：.-"):
        quality += 0.8
    if "`" in value or "(" in value or "（" in value:
        quality += 0.4
    return quality


def _term_score(text, terms):
    line_terms = {term.lower() for term in tokenize(text)}
    lowered = text.lower()
    boundary_hits = sum(
        1
        for term in terms
        if re.search(r"(?<![0-9a-z_])%s(?![0-9a-z_])" % re.escape(term), lowered)
    )
    return len(terms & line_terms) + boundary_hits


def _best_evidence(texts, terms):
    best = None
    for rank, (source, text) in enumerate(texts):
        if not text.strip() or source != "description" and _looks_like_tag_line(text):
            continue
        score = _term_score(text, terms)
        if score <= 0:
            continue
        candidate = (score * 10 + _line_quality(text, source), -rank, text)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else ""


def _fallback_evidence(texts):
    fallback = ""
    for source, text in texts:
        if not text.strip() or source == "tags" and fallback:
            continue
        if not fallback:
            fallback = text
    return fallback


def snippet_for_doc(doc: dict, query_terms=None, max_chars: int = 220) -> str:
    terms = {str(t).lower() for t in (query_terms or []) if str(t).strip()}
    try:
        with open(id_to_path(doc["path"]), encoding="utf-8") as card_file:
            raw = card_file.read()
    except OSError:
        return ""

    fm, body = _parse_front(raw)
    title = fm.get("title", "")
    description = fm.get("description", "")
    body_lines = [line.strip() for line in strip_related(body).splitlines() if line.strip()]
    evidence_texts = [("description", description)]
    evidence_texts.extend(("body", line) for line in body_lines)
    fallback_texts = [
        ("description", description),
        ("body", next((line for line in body_lines if not _is_heading(line)), "")),
        ("title", title),
        ("tags", " ".join(fm.get("tags", []) or [])),
    ]
    if terms:
        best = _best_evidence(evidence_texts + fallback_texts, terms)
        if best:
            return _clip(best, max_chars)
    fallback = _fallback_evidence(evidence_texts + fallback_texts)
    return _clip(fallback, max_chars) if fallback else ""


def attach_snippets(hits: list[dict], queries=None, idx: dict | None = None) -> list[dict]:
    terms = []
    for query in queries or []:
        terms.extend(tokenize(str(query)))
    docs_by_path = {d["path"]: d for d in idx.get("docs", [])} if idx else {}
    for hit in hits:
        doc = docs_by_path.get(hit.get("path"), hit)
        if "snippet" not in hit:
            hit["snippet"] = snippet_for_doc(doc, terms)
    return hits
