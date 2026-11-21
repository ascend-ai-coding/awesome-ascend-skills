# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""LLM client for OpenAI API"""

import json
import time
from functools import wraps
from typing import Optional, Type

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from src.settings import load_settings

from . import config


def retry(
    max_retries: int = config.MAX_RETRIES,
    base_delay: float = config.RETRY_BASE_DELAY,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        print(f"[WARN] LLM call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"[ERROR] LLM call failed after {max_retries} attempts: {e}")
            raise last_exception

        return wrapper

    return decorator


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        base_url: Optional[str] = None,
        timeout: float = config.TIMEOUT,
    ):
        settings = load_settings()
        self.client = OpenAI(
            api_key=api_key or settings.incremental_llm_key,
            base_url=base_url or settings.incremental_llm_url,
            http_client=httpx.Client(
                verify=str(config.SSL_CERT) if config.SSL_CERT is not None else True,
                timeout=timeout,
            ),
        )
        self.model = model or settings.model

    @retry()
    def chat(
        self,
        messages: list[dict],
        temperature: float = config.TEMPERATURE_CHAT,
        max_tokens: Optional[int] = 16384,
        enable_thinking: bool = False,
    ) -> str:
        extra_body = {}
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )
        return response.choices[0].message.content

    @retry()
    def chat_with_json(
        self,
        messages: list[dict],
        temperature: float = config.TEMPERATURE_JSON,
        max_tokens: Optional[int] = 16384,
        enable_thinking: Optional[bool] = None,
    ) -> dict:
        extra_body = {}
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body if extra_body else None,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return {"_raw": parsed}
            return parsed
        return content if isinstance(content, dict) else {"_raw": content}

    @retry()
    def chat_with_structured_output(
        self,
        messages: list[dict],
        response_model: Type[BaseModel],
        temperature: float = config.TEMPERATURE_JSON,
        enable_thinking: bool = False,
    ) -> BaseModel:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_model,
            extra_body=extra_body,
        )
        return response.choices[0].message.parsed

    def test_connection(self) -> bool:
        """模型连通性测试，不走 @retry，失败直接抛出异常"""
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
