# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.cache — deterministic fingerprint disk cache for the opt-in model routes.

Model outputs (llm-judge / reranker verdicts, embedding card vectors) are non-deterministic
to produce but must be *reproducible once produced*: we key each result by a SHA-1
fingerprint of everything that could change it, so a repeat run with the same inputs is a
cache hit (zero LLM / zero API) and the model route becomes "reproducible modulo cache".

Cache files live under .build/ (gitignored). Pure stdlib.
"""
import hashlib
import json
import os


def fingerprint(parts):
    """Stable SHA-1 over a dict of named components (sorted-key JSON canonicalization).
    Pass EVERYTHING that affects the output — model, prompt/schema version, policy,
    material, query, the ordered candidate payload, etc. Order within the dict is
    irrelevant (sort_keys); order *within* list values is significant (so callers that
    care about candidate order must canonicalize before passing).
    """
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """Tiny JSON file cache keyed by fingerprint string. Atomic writes; corrupt/missing
    entries read as a miss (None).
    """

    def __init__(self, dirpath):
        self.dir = dirpath

    def get(self, key):
        p = self._path(key)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None
        return None

    def set(self, key, obj):
        os.makedirs(self.dir, exist_ok=True)
        p = self._path(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, p)

    def _path(self, key):
        return os.path.join(self.dir, key + ".json")
