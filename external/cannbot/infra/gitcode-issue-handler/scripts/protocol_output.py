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
"""Write one JSON protocol record to the current stdout stream."""

import json
import logging
import sys


def write_json(value, **options):
    """Keep CLI output byte-compatible while using the logging stream handler."""
    message = json.dumps(value, **options)
    record = logging.LogRecord(__name__, logging.INFO, __file__, 0, message, (), None)
    logging.StreamHandler(sys.stdout).emit(record)
