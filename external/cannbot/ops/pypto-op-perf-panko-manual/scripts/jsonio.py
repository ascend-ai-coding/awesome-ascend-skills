#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The one channel every script in this skill answers on.

Each of these programs is read by another program: the harness returns a
directive the optimizer parses, and `feasibility`, `predicates`, `swimlane` and
`measure_latency` each return one JSON object to whoever invoked them. So stdout
is a protocol, not a log -- exactly one JSON document per run, nothing beside
it.

That makes it worth routing through a named logger rather than `print`. A logger
can be reconfigured by whoever embeds this (silenced, teed, given a file) without
editing the scripts, while the default handler here writes the document and a
newline to stdout and nothing else: the format is `%(message)s`, so the bytes
are what `print(json.dumps(...))` produced.

`propagate` is off on purpose. A root logger configured by an embedding process
would otherwise prefix every directive with a timestamp and a level, and the
optimizer would fail to parse its own protocol.
"""

import json
import logging
import sys

CHANNEL = "pypto.panko.stdout"

_log = logging.getLogger(CHANNEL)
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
_log.propagate = False


def emit(obj, indent=None):
    """Write one JSON document to the protocol channel."""
    _log.info(json.dumps(obj, ensure_ascii=False, indent=indent))
