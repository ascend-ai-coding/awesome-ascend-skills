# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Prefix-free, logging-backed adapters for the retrieval CLI streams."""

import logging
import sys


class _PlainHandler(logging.Handler):
    """Write a formatted record to the current redirected stdout or stderr."""

    def __init__(self, stream_name):
        super().__init__()
        self._stream_name = stream_name
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        stream = getattr(sys, self._stream_name)
        stream.write(self.format(record))
        stream.flush()


_STDOUT = _PlainHandler("stdout")
_STDERR = _PlainHandler("stderr")


def _emit(handler, message, args, end):
    payload = message % args if args else str(message)
    record = logging.LogRecord(__name__, logging.INFO, "", 0, "%s%s", (payload, end), None)
    handler.handle(record)


def emit_stdout(message="", *args, end="\n"):
    """Write an exact CLI payload to stdout through a prefix-free logger."""
    _emit(_STDOUT, message, args, end)


def emit_stderr(message="", *args, end="\n"):
    """Write an exact CLI diagnostic to stderr through a prefix-free logger."""
    _emit(_STDERR, message, args, end)
