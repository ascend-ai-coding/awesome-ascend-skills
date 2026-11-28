# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Unified configuration constants for anchor_construction module.

Contains:
- Code map whitelist types (node/edge)
- Anchor type constants
- Document parts (imported from PubRefConfig)
"""

from src.snapshot_construction.pub_ref import PubRefConfig

# Code map node types whitelist (lowercase for case-insensitive matching)
CODE_MAP_NODE_TYPES: frozenset[str] = frozenset(
    {
        "function",
        "class",
        "struct",
        "namespace",
        "constant",
        "errorcode",
    }
)

# Code map edge types whitelist (lowercase for case-insensitive matching)
CODE_MAP_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "calls",
        "returns",
    }
)

# Anchor type constants
ANCHOR_TYPE_FILE: str = "file"
ANCHOR_TYPE_FUNCTION: str = "function"
ANCHOR_TYPE_SECTION: str = "section"
ANCHOR_TYPE_NOTE: str = "note"

# Document parts (imported from PubRefConfig)
DOC_PARTS: list[str] = PubRefConfig.DOC_PARTS
