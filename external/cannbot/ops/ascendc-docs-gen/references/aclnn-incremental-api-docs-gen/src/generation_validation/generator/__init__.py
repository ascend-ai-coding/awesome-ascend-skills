# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Document generator module"""

__all__ = [
    "TextWriter",
    "DocumentGenerator",
    "get_example",
]

from .writer import TextWriter
from .generator import DocumentGenerator
from .get_example import (
    get_example_code,
    get_example_doc_part,
    get_all_example_codes,
    get_example_codes_by_describe,
    get_example_content_with_source,
)
