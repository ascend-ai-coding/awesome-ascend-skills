#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------
"""Write command protocols through a message-only logging handler."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def _write_message(text: str, stream: TextIO, channel: str) -> None:
    """Emit exactly one trailing newline without adding log metadata."""
    logger = logging.getLogger(f"cannbot.gitcode_issue_handler.cli.{channel}")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        message = text[:-1] if text.endswith("\n") else text
        logger.info("%s", message)
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def write_stdout(text: str) -> None:
    """Write the public command result protocol to stdout."""
    _write_message(text, sys.stdout, "stdout")


def write_stderr(text: str) -> None:
    """Write a command error protocol to stderr."""
    _write_message(text, sys.stderr, "stderr")
