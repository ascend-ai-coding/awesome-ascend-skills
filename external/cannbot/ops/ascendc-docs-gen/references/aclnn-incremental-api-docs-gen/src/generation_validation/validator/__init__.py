# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Document validator module"""

__all__ = [
    "ConsistencyChecker",
    "ModificationMetric",
    "DocumentValidator",
    "validate_example",
]

from .consistency import ConsistencyChecker
from .metric import ModificationMetric
from .validator import DocumentValidator
from .example_validator import validate_example
