# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""recall.dense — embedding / vector recall (opt-in, non-deterministic backend).

Encodes each candidate card + the query with an EmbeddingBackend, ranks by cosine.
Default backend = Embedding API (needs a key -> else NotConfigured, exit 3);
`--backend hashing` = zero-dependency offline backend (deterministic, lexical-ish).

API card vectors are cached (.build/embed_cache/) keyed by backend signature + per-card
content fingerprint, so re-runs don't re-embed the corpus. The hashing backend is cheap,
so it's computed inline. Backends are constructed lazily (per backend name) — Dense()
itself is side-effect-free for REGISTRY instantiation.
"""
import math

from retrieval.cards import _headings, _parse_front, _signatures, id_to_path, strip_related
from retrieval.cache import DiskCache, fingerprint
from retrieval.config import EMBED_CACHE_DIR
from retrieval.embedding import EMBED_TEXT_VERSION, get_embedding_backend
from retrieval.recall.base import Recall


def _card_text(d):
    parts = [
        d.get("title", ""),
        d.get("description", ""),
        " ".join(d.get("tags", [])),
        " ".join(d.get("aliases", []) or []),
        " ".join(d.get("paradigms", []) or []),
        d.get("path", ""),
    ]
    try:
        with open(id_to_path(d["path"]), encoding="utf-8") as card_file:
            raw = card_file.read()
    except OSError:
        raw = ""
    if raw:
        _, body = _parse_front(raw)
        body = strip_related(body)
        parts.extend([_headings(body), _signatures(body)])
        body_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        parts.append("\n".join(body_lines)[:6000])
    return "\n".join(p for p in parts if p)


def _to_dict(vec):
    return vec if isinstance(vec, dict) else {i: x for i, x in enumerate(vec) if x}


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


class Dense(Recall):
    method = "dense"

    def __init__(self):
        self._backends = {}                      # name -> backend (lazy)

    def recall(self, idx, spec):
        queries = spec.get("queries") or []
        if not queries:
            return []
        bk = self._backend(spec.get("embed_backend"), spec.get("embed_model"))
        cands = self._candidates(spec)
        k = spec.get("k", 50)
        docids = list(cands) if cands is not None else list(range(len(idx["docs"])))
        if not docids:
            return []
        cvecs = self._card_vectors(idx, bk, docids)        # NotConfigured propagates here
        qvec = _to_dict(bk.encode([" ".join(queries)])[0])
        scored = {d: _cos(qvec, cvecs[d]) for d in docids}
        return [self._hit(idx["docs"][d], scored[d]) for d in self._topk(scored, idx, k)]

    def _backend(self, name, model):
        key = (name, model)
        if key not in self._backends:
            self._backends[key] = get_embedding_backend(name, model)
        return self._backends[key]

    def _card_vectors(self, idx, bk, docids):
        texts = [_card_text(idx["docs"][d]) for d in docids]
        if bk.name == "hashing":                 # cheap -> inline, no cache
            return {d: _to_dict(v) for d, v in zip(docids, bk.encode(texts))}
        cache = DiskCache(EMBED_CACHE_DIR)
        sig = fingerprint({"backend": bk.name, "model": bk.model, "base_url": bk.base_url,
                           "dim": bk.dim, "text_version": EMBED_TEXT_VERSION})
        blob = cache.get(sig) or {}
        out, miss, miss_t = {}, [], []
        for d, t in zip(docids, texts):
            p = idx["docs"][d]["path"]
            cfp = fingerprint({"text": t})
            entry = blob.get(p)
            if entry and entry.get("fp") == cfp:
                out[d] = _to_dict(entry["v"])
            else:
                miss.append((d, p, cfp))
                miss_t.append(t)
        if miss_t:
            for (d, p, cfp), v in zip(miss, bk.encode(miss_t)):
                blob[p] = {"fp": cfp, "v": v}
                out[d] = _to_dict(v)
            cache.set(sig, blob)
        return out


recall = Dense().recall
