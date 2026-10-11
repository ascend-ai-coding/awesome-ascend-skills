#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Write one JSON protocol record to the current stdout stream."""

import json
import logging
import sys


def write_json(value, **options):
    """Keep CLI output byte-compatible while using the logging stream handler."""
    message = json.dumps(value, **options)
    record = logging.LogRecord(__name__, logging.INFO, __file__, 0, message, (), None)
    logging.StreamHandler(sys.stdout).emit(record)
