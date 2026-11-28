# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Unified brace-matching utilities for anchor construction.

This module consolidates brace/bracket matching functions from:
- anchor_construction/support/extract_new_0331.py (line-based functions)
- anchor_construction/sub_function/extract/extract_functions.py (content-based)
- anchor_construction/sub_function/extract/extract_section.py (content-based)

Two paradigms are supported:
1. Line-based: Functions that work on list[str] of file lines, return line numbers
2. Content-based: Functions that work on str content with position indices
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Line-based functions (work on list[str] of lines, return line numbers)
# =============================================================================


def find_matching_brace_end(lines: list[str], open_line_idx: int) -> int:
    """
    Find the line index of the matching closing brace.

    Args:
        lines: List of file lines
        open_line_idx: Line index where the opening brace is located

    Returns:
        Line index of the matching closing brace, or last line index if not found
    """
    depth = 0
    found_open = False
    for i in range(open_line_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
        if found_open and depth == 0:
            return i
    return len(lines) - 1


def find_function_start(lines: list[str], brace_open_idx: int) -> int:
    """
    Find the line index where a function signature starts.

    Scans backwards from the opening brace to find the first line of the function
    signature (after empty lines, semicolons, or closing braces).

    Args:
        lines: List of file lines
        brace_open_idx: Line index where the opening brace is located

    Returns:
        Line index of the function signature start
    """
    for i in range(brace_open_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped == "" or stripped.endswith(";") or stripped.endswith("}"):
            return i + 1
    return 0


def find_enclosing_function(lines: list[str], target_idx: int) -> Tuple[int, int]:
    """
    Find the start and end line indices of the function enclosing a target line.

    Args:
        lines: List of file lines
        target_idx: Line index to find the enclosing function for

    Returns:
        Tuple of (signature_start_line, function_end_line)
    """
    depth = 0
    brace_open_idx = None
    for i in range(target_idx, -1, -1):
        scan = lines[i][::-1]
        for ch in scan:
            if ch == "}":
                depth += 1
            elif ch == "{":
                if depth == 0:
                    brace_open_idx = i
                    break
                depth -= 1
        if brace_open_idx is not None:
            break
    if brace_open_idx is None:
        brace_open_idx = 0
    sig_start = find_function_start(lines, brace_open_idx)
    func_end = find_matching_brace_end(lines, brace_open_idx)
    return sig_start, func_end


def find_if_block_start(lines: list[str], match_idx: int) -> int:
    """
    Find the line index where an if/else if block starts.

    Args:
        lines: List of file lines
        match_idx: Line index to start searching backwards from

    Returns:
        Line index of the if/else if statement, or match_idx if not found
    """
    for i in range(match_idx, -1, -1):
        stripped = lines[i].strip()
        if re.match(r"^(if|else\s*if)\s*\(", stripped) or re.match(
            r"^(if|else\s*if)\b", stripped
        ):
            return i
    return match_idx


def find_full_if_else_block(lines: list[str], if_start_idx: int) -> int:
    """
    Find the end line index of a complete if-else-if-else chain.

    Args:
        lines: List of file lines
        if_start_idx: Line index where the if statement starts

    Returns:
        Line index of the end of the complete if-else block
    """
    end_idx = find_matching_brace_end(lines, if_start_idx)
    i = end_idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        if re.match(r"^else\s*if\b", stripped) or re.match(r"^else\b", stripped):
            end_idx = find_matching_brace_end(lines, i)
            i = end_idx + 1
        else:
            break
    return end_idx


def find_endif_line(lines: list[str], if_line_idx: int) -> int:
    """
    Find the line index of the matching #endif for a preprocessor #if.

    Args:
        lines: List of file lines
        if_line_idx: Line index where the #if/#ifdef/#ifndef is located

    Returns:
        Line index of the matching #endif, or last line index if not found
    """
    depth = 0
    for i in range(if_line_idx, len(lines)):
        stripped = lines[i].strip()
        if re.match(r"#\s*if(def|ndef)?\b", stripped):
            depth += 1
        elif re.match(r"#\s*endif\b", stripped):
            depth -= 1
            if depth == 0:
                return i
    return len(lines) - 1


# =============================================================================
# Content-based functions (work on str content with position indices)
# =============================================================================


def find_function_body_start(content: str, start_pos: int) -> int:
    """
    Find the position of the opening brace for a function body.

    Args:
        content: String content to search
        start_pos: Position to start searching from

    Returns:
        Position of the opening brace, or -1 if not found
    """
    pos = content.find("{", start_pos)
    return pos


def find_matching_brace(content: str, start_pos: int) -> int:
    """
    Find the position of the matching closing brace.

    Args:
        content: String content to search
        start_pos: Position of the opening brace

    Returns:
        Position of the matching closing brace, or -1 if not found
    """
    brace_count = 1
    i = start_pos + 1

    while i < len(content) and brace_count > 0:
        if content[i] == "{":
            brace_count += 1
        elif content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return i
        i += 1

    return -1
