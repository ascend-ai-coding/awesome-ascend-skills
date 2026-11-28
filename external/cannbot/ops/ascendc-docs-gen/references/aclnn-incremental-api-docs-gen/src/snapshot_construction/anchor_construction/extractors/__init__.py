# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Extractors module for anchor construction.

Provides unified extractor classes for each anchor type:
- FileExtractor: Extracts file anchors from operator source files
- FunctionExtractor: Extracts function anchors from C++ files
- NoteExtractor: Extracts note anchors from Doxygen comment blocks
"""

__all__ = ["FileExtractor", "FunctionExtractor", "NoteExtractor"]

from src.snapshot_construction.anchor_construction.extractors.file_extractor import (
    FileExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.function_extractor import (
    FunctionExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.note_extractor import (
    NoteExtractor,
)
