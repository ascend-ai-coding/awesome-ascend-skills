# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""DocPartConfig factory functions for each doc_part.

Each factory function produces the same DocPartConfig as defined inline in gene_rule_registry.py.
These will eventually replace the inline definitions in GeneRuleRegistry.
"""

__all__ = [
    "make_constraint_config",
]

from src.snapshot_construction.anchor_construction.doc_parts.constraint_config import (
    make_constraint_config,
)
