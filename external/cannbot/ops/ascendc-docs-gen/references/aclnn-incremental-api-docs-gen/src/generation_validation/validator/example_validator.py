# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Example code validator"""

import re
import difflib


def clean_text(text: str) -> str:
    if not text:
        return ""
    # \s+ 匹配任意空白字符（空格、换行、制表符等），替换为空
    return re.sub(r"\s+", "", text)


def validate_example(
    old_example: dict[str, str], new_example: dict[str, str]
) -> dict[str, str]:
    """
    Validate if old example matches the new example.

    Supports two modes:
    1. Key-Value Perfect Match: Keys match (case-insensitive) AND Values match.
    2. Value Set Match: If keys don't match, check if the set of values is identical
       (order doesn't matter).

    Args:
        old_example: Dict with operator name/describe as key and example code as value
        new_example: Dict with operator name/describe as key and example code as value

    Returns:
        Dict with "results": "PASS"/"FAIL" and "content": new example code (or empty if pass)
    """
    if not new_example:
        first_old_val = next(iter(old_example.values()), "")
        diff_text = _generate_diff(first_old_val, "")
        return {
            "results": "FAIL",
            "content": "",
            "diff_hunks": _parse_diff_hunks(diff_text),
        }

    if not old_example:
        first_new_val = next(iter(new_example.values()), "")
        diff_text = _generate_diff("", first_new_val)
        return {
            "results": "FAIL",
            "content": first_new_val,
            "diff_hunks": _parse_diff_hunks(diff_text),
        }

    # 1. 尝试 Key-Value 完美匹配
    # 遍历 new_example，看是否能找到 old_example 中对应的 key 和 value
    matched_keys = set()
    all_keys_matched = True

    for new_key, new_content in new_example.items():
        found_match = False
        for old_key, old_content in old_example.items():
            if (
                _keys_match(old_key, new_key)
                and old_content.strip() == new_content.strip()
            ):
                found_match = True
                matched_keys.add(old_key)
                break

        if not found_match:
            all_keys_matched = False
            break

    if all_keys_matched:
        # 如果所有 new 的 key 都在 old 中找到了对应的 key 和 value
        # 还需要确保 old 中没有多余的 key (即两个字典大小一致且完全匹配)
        if len(old_example) == len(new_example):
            return {"results": "PASS", "content": ""}

    # 2. 如果 Key 不匹配，尝试 Value 集合匹配
    # 提取所有非空 content 并标准化（strip）
    old_contents = {clean_text(v) for v in old_example.values() if clean_text(v)}
    new_contents = {clean_text(v) for v in new_example.values() if clean_text(v)}
    if old_contents == new_contents:
        return {"results": "PASS", "content": ""}

    # 3. 都不匹配
    first_new_key = next(iter(new_example.keys()))
    first_old_key = next(iter(old_example.keys()), "")
    diff_text = _generate_diff(
        old_example.get(first_old_key, ""),
        new_example.get(first_new_key, ""),
    )
    return {
        "results": "FAIL",
        "content": new_example[first_new_key],
        "diff_hunks": _parse_diff_hunks(diff_text),
    }


def _keys_match(key1: str, key2: str) -> bool:
    """Check if two operator keys match (case-insensitive)."""
    return key1.lower() == key2.lower()


def _generate_diff(old_text: str, new_text: str) -> str:
    """Generate unified diff between two code strings."""
    old_lines = old_text.splitlines() if old_text else []
    new_lines = new_text.splitlines() if new_text else []
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="doc_json",
        tofile="source_file",
    )
    return "\n".join(diff)


def _parse_diff_hunks(diff_text: str) -> list[dict[str, str]]:
    """Parse unified diff text into individual hunks."""
    if not diff_text:
        return []

    hunks = []
    current_location = ""
    current_lines: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            if current_location:
                hunks.append(
                    {
                        "location": current_location,
                        "reason": "\n".join(current_lines),
                    }
                )
            current_location = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_location:
        hunks.append(
            {
                "location": current_location,
                "reason": "\n".join(current_lines),
            }
        )

    return hunks
