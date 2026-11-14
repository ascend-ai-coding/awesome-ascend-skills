# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.errors — structured errors for the opt-in model routes (dense / reranker /
llm-judge). Callers parse the JSON (not stderr); the CLI prints `as_json()` and exits.

  NotConfigured   — backend absent/unconfigured (no key, no SDK, no model). exit 3.
  ModelRuntimeError — backend configured but the call failed at runtime (timeout, bad
                      JSON, API error). NOT cached. exit 2. Distinct so callers can
                      tell "you never set this up" from "it ran and broke".
"""


class CliError(Exception):
    """Business-level failure that the top-level CLI converts to an exit status."""

    def __init__(self, message=None, *, code=1):
        self.message = message
        self.code = code
        super().__init__(message)


class NotConfigured(Exception):
    def __init__(self, method, how):
        self.method = method
        self.how = how
        super().__init__(how)

    def as_json(self):
        return {"error": "not_configured", "method": self.method, "how": self.how}


class ModelRuntimeError(Exception):
    def __init__(self, method, detail):
        self.method = method
        self.detail = detail
        super().__init__(detail)

    def as_json(self):
        return {"error": "model_runtime_error", "method": self.method, "detail": self.detail}
