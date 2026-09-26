# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.llm — Claude Code SDK wrapper for the opt-in LLM rerank routes
(llm-judge auto path + LLM reranker). Shared by both via retrieval.judge.

Design constraints (SPEC-Retrieve §9):
  · __init__ stores CONFIG ONLY — no client/SDK construction (so REGISTRY instantiation
    stays side-effect-free and stdlib-only).
  · claude_agent_sdk is LAZY-imported inside the call; absent -> NotConfigured (exit 3).
  · Tools are truly disabled: tools=[] (not just allowed_tools=[]) + mcp_servers={} +
    strict_mcp_config=True + setting_sources=[]; one turn; no bypassPermissions needed.
  · A whole-request timeout via asyncio.wait_for (the SDK has no single-request timeout).
  · Output must be JSON; parse failure / timeout / API error -> ModelRuntimeError (exit 2),
    and is NOT written to cache. Successful JSON IS cached (reproducible modulo cache).
"""
import asyncio
import json
import re

from retrieval.cache import DiskCache
from retrieval.config import (LLM_CACHE_DIR, LLM_MAX_BUDGET_USD, LLM_MODEL,
                              LLM_TIMEOUT_S, ROOT)
from retrieval.errors import ModelRuntimeError, NotConfigured

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json(text):
    """Best-effort: whole text -> ```json``` fenced block -> first {...} span. None on fail."""
    text = (text or "").strip()
    if not text:
        return None
    for cand in (text, *(m.group(1) for m in _JSON_BLOCK.finditer(text))):
        try:
            return json.loads(cand)
        except ValueError:
            continue
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except ValueError:
            return None
    return None


class LLMClient:
    def __init__(self, model=None, timeout_s=None, budget_usd=None):
        self.model = model or LLM_MODEL
        self.timeout_s = timeout_s if timeout_s is not None else LLM_TIMEOUT_S
        self.budget_usd = budget_usd if budget_usd is not None else LLM_MAX_BUDGET_USD
        self._cache = DiskCache(LLM_CACHE_DIR)

    @staticmethod
    def _sdk():
        try:
            import claude_agent_sdk as sdk
        except ImportError as error:
            raise NotConfigured(
                "llm",
                "Claude Code SDK (claude_agent_sdk) not importable: %s. Install it, or use "
                "deterministic rerank (bm25f/tagidf/quality), or the prepare-judge + "
                "--verdicts flow." % error,
            ) from error
        return sdk

    @staticmethod
    def _text_blocks(sdk, message):
        if not isinstance(message, sdk.AssistantMessage):
            return []
        return [
            block.text
            for block in getattr(message, "content", []) or []
            if isinstance(block, sdk.TextBlock)
        ]

    def complete_text(self, prompt, system=""):
        sdk = self._sdk()                       # NotConfigured if SDK missing
        try:
            return asyncio.run(asyncio.wait_for(self._consume(sdk, prompt, system),
                                                self.timeout_s))
        except asyncio.TimeoutError as e:
            raise ModelRuntimeError("llm", "LLM call timed out after %ss" % self.timeout_s) from e
        except (NotConfigured, ModelRuntimeError):
            raise
        except Exception as e:                  # SDK/transport/runtime failure
            raise ModelRuntimeError("llm", "LLM call failed: %s: %s" % (type(e).__name__, e)) from e

    def complete_json(self, prompt, system="", cache_key=None):
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        text = self.complete_text(prompt, system)
        obj = _extract_json(text)
        if obj is None:                         # do NOT cache a bad response
            raise ModelRuntimeError(
                "llm", "LLM output was not valid JSON (not cached). First 240 chars: %r"
                % (text[:240]))
        if cache_key is not None:
            self._cache.set(cache_key, obj)
        return obj

    async def _consume(self, sdk, prompt, system):
        opts = sdk.ClaudeAgentOptions(
            tools=[], allowed_tools=[], mcp_servers={}, strict_mcp_config=True,
            setting_sources=[], max_turns=1, model=self.model, cwd=ROOT,
            system_prompt=system or None, max_budget_usd=self.budget_usd)
        chunks = []
        async for msg in sdk.query(prompt=prompt, options=opts):
            chunks.extend(self._text_blocks(sdk, msg))
        return "".join(chunks)
