# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.embedding — pluggable embedding backends for dense recall.

Inherit `EmbeddingBackend` to add a backend; `encode(texts) -> list[vector]` where a
vector is either a dense list[float] or a sparse {int: float} dict (dense.recall handles
both via cosine). All heavy client libs are LAZY-imported inside the backend, and
backends are constructed lazily (never at module import) so the recall REGISTRY stays
stdlib-only.

  APIEmbedding     — DEFAULT. openai-compatible embeddings endpoint; no key/lib ->
                     NotConfigured (exit 3); API failure -> ModelRuntimeError (exit 2).
  HashingEmbedding — zero-dependency, deterministic, offline. Feature-hashes tokens +
                     CJK/sub-word char-bigrams into a sparse L2-normalized dict. Lexical-
                     ish (no deep semantics) but works everywhere; good for tests + as the
                     extensibility example.
"""
import hashlib
import math
import os
from abc import ABC, abstractmethod

from retrieval.errors import ModelRuntimeError, NotConfigured
from retrieval.tokenizer import tokenize

EMBED_BACKEND = os.environ.get("CANNBOT_KNOWLEDGE_QUERY_EMBED_BACKEND", "api")
EMBED_MODEL = os.environ.get("CANNBOT_KNOWLEDGE_QUERY_EMBED_MODEL", "text-embedding-3-small")
EMBED_BASE_URL = os.environ.get(
    "CANNBOT_KNOWLEDGE_QUERY_EMBED_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")
)
EMBED_API_KEY_ENV = "OPENAI_API_KEY"
EMBED_DIM = int(os.environ.get("CANNBOT_KNOWLEDGE_QUERY_EMBED_DIM", "1536"))
EMBED_TEXT_VERSION = "2"
HASHING_DIM = 4096


class EmbeddingBackend(ABC):
    name = ""
    model = ""
    base_url = ""
    dim = 0

    @abstractmethod
    def encode(self, texts):
        ...


class APIEmbedding(EmbeddingBackend):
    name = "api"

    def __init__(self, model=None, base_url=None, dim=None):
        self.model = model or EMBED_MODEL
        self.base_url = base_url or EMBED_BASE_URL
        self.dim = dim or EMBED_DIM
        self._client = None

    def encode(self, texts):
        self._ensure()
        try:
            resp = self._client.embeddings.create(model=self.model, input=list(texts))
        except Exception as e:
            raise ModelRuntimeError("dense", "embedding API call failed: %s: %s"
                                    % (type(e).__name__, e)) from e
        return [d.embedding for d in resp.data]

    def _ensure(self):
        if self._client is not None:
            return
        key = os.environ.get(EMBED_API_KEY_ENV)
        if not key:
            raise NotConfigured(
                "dense",
                "embedding API key not set ($%s). Configure it + CANNBOT_KNOWLEDGE_QUERY_EMBED_BASE_URL / "
                "CANNBOT_KNOWLEDGE_QUERY_EMBED_MODEL, or use `--backend hashing` (zero-dep offline), or a "
                "deterministic recall (bm25/tfidf/tagtype/graph)." % EMBED_API_KEY_ENV)
        try:
            import openai
        except ImportError as e:
            raise NotConfigured("dense", "openai client lib not importable: %s" % e) from e
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**kwargs)


class HashingEmbedding(EmbeddingBackend):
    name = "hashing"

    def __init__(self, dim=None):
        self.model = "hashing"
        self.base_url = ""
        self.dim = dim or HASHING_DIM

    def encode(self, texts):
        return [self._vec(t) for t in texts]

    def _vec(self, text):
        feats = []
        for t in tokenize(text):
            feats.append(t)
            for i in range(len(t) - 1):          # char bigrams: CJK / sub-word overlap
                feats.append(t[i:i + 2])
        v = {}
        for f in feats:
            hh = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
            idx = hh % self.dim
            sign = 1.0 if (hh >> 1) & 1 else -1.0
            v[idx] = v.get(idx, 0.0) + sign
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}


def get_embedding_backend(name=None, model=None):
    name = name or EMBED_BACKEND
    if name == "api":
        return APIEmbedding(model=model)
    if name == "hashing":
        return HashingEmbedding()
    raise NotConfigured("dense", "unknown embedding backend %r (use 'api' or 'hashing')" % name)
