# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reduction paradigm design gates for validate_completeness.py.

Checks that DESIGN.md contains actual code blocks (not prose descriptions)
for critical Reduction kernel implementation components.

Exposed API:
    def validate_gates(design_text: str) -> list[str]:
        Returns list of error strings. Empty = all gates passed.
"""

from __future__ import annotations

import re


def _extract_cpp_blocks(text: str) -> list[str]:
    """Extract all ```cpp ... ``` code blocks from markdown text."""
    pattern = re.compile(r"```(?:cpp|c\+\+)\s*\n(.*?)```", re.DOTALL)
    return [m.group(1) for m in pattern.finditer(text)]


def _block_lines(block: str) -> int:
    """Count non-empty lines in a code block."""
    return sum(1 for line in block.splitlines() if line.strip())


def _has_function_definition(
    blocks: list[str], func_name: str, min_lines: int = 0
) -> bool:
    """Check if any cpp block contains a function definition for func_name.

    Matches patterns like:
        void FunctionName(...) {
        __aicore__ inline void FunctionName(...) {
        static inline void FunctionName(...) {
    """
    # 匹配函数定义：可选的修饰符 + 返回类型 + 函数名 + 参数列表 + {
    pattern = re.compile(
        r"\b(?:__aicore__\s+)?(?:inline\s+)?(?:static\s+)?\w+\s+"
        + re.escape(func_name)
        + r"\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    for block in blocks:
        if pattern.search(block) and _block_lines(block) >= min_lines:
            return True
    return False


def _has_ellipsis_placeholder(blocks: list[str]) -> list[str]:
    """Find cpp blocks containing '// ...' as omission placeholder."""
    pattern = re.compile(r"^\s*//\s*\.{3}\s*$", re.MULTILINE)
    results = []
    for i, block in enumerate(blocks):
        matches = pattern.findall(block)
        if matches:
            results.append(
                f"code block #{i + 1} contains {len(matches)} placeholder(s)"
            )
    return results


def validate_gates(design_text: str) -> list[str]:
    """Validate that DESIGN.md contains required code blocks for Reduction paradigm.

    Args:
        design_text: Full text of DESIGN.md

    Returns:
        List of error strings. Empty list means all gates passed.
    """
    errors: list[str] = []
    blocks = _extract_cpp_blocks(design_text)

    if not blocks:
        errors.append("GATE-0: no cpp code blocks found in DESIGN.md")
        return errors

    # 1. Process() complete source block (>= 30 lines)
    if not _has_function_definition(blocks, "Process", min_lines=30):
        errors.append(
            "GATE-1: missing Process() function definition "
            "(need cpp block with Process() definition and >= 30 non-empty lines)"
        )

    # 2. CopyIn source (DoCopyInTile required)
    if not _has_function_definition(blocks, "DoCopyInTile"):
        errors.append("GATE-2: missing DoCopyInTile function definition in cpp blocks")

    # 3. CopyOut three-path pseudocode (>= 15 lines)
    if not _has_function_definition(blocks, "CopyOut", min_lines=15):
        errors.append(
            "GATE-3: missing CopyOut function definition "
            "(need cpp block with CopyOut() definition and >= 15 non-empty lines)"
        )

    # 4. Cache tree (GetCacheID + DoCaching，两者都必须有，见 patterns.md [6.8.2])
    has_cache_tree = _has_function_definition(
        blocks, "GetCacheID"
    ) and _has_function_definition(blocks, "DoCaching")
    if not has_cache_tree:
        errors.append(
            "GATE-4: missing bisection cache tree function definitions "
            "(need both GetCacheID() and DoCaching())"
        )

    # 5. Mandatory kernel functions per patterns [6.8.2]/[6.8.3]
    required_funcs = [
        "ClearChunkExtensionVf",
        "MergeTmpBufVf",
        "FindNearestPower2",
        "CalLog2",
    ]
    missing_funcs = [name for name in required_funcs if name not in design_text]
    if missing_funcs:
        errors.append(
            f"GATE-5: missing mandatory kernel function definitions: {', '.join(missing_funcs)} "
            f"(ClearInnerBurstTailPadVf 按触发条件表决定是否产出，不强制)"
        )

    # 6. No ellipsis placeholders ('// ...') in cpp blocks
    ellipsis_hits = _has_ellipsis_placeholder(blocks)
    if ellipsis_hits:
        errors.append(
            f"GATE-6: ellipsis placeholder '// ...' found in cpp blocks: "
            f"{'; '.join(ellipsis_hits)}"
        )

    # 7. TilingKey must have ASCENDC_TPL_ARGS_DECL and ASCENDC_TPL_SEL
    has_decl = "ASCENDC_TPL_ARGS_DECL" in design_text
    has_sel = "ASCENDC_TPL_SEL" in design_text
    if not has_decl or not has_sel:
        missing = []
        if not has_decl:
            missing.append("ASCENDC_TPL_ARGS_DECL")
        if not has_sel:
            missing.append("ASCENDC_TPL_SEL")
        errors.append(
            f"GATE-7: missing TilingKey macro definitions: {', '.join(missing)}"
        )

    # 8. TilingKey must not contain dtype —— 只检查 TPL 宏参数行，注释/正文中的 dtype 说明不误伤
    for block in blocks:
        if "ASCENDC_TPL_ARGS_DECL" in block or "ASCENDC_TPL_SEL" in block:
            for line in block.splitlines():
                if (
                    "TPL_BOOL_DECL" in line
                    or "TPL_SEL" in line
                    or "TPL_ARGS_DECL" in line
                ):
                    if re.search(r"\b(dtype|DataType|DTYPE)\b", line, re.IGNORECASE):
                        errors.append(
                            "GATE-8: TilingKey contains dtype parameter — "
                            "Reduction paradigm TilingKey should not split by dtype"
                        )
                        break
            break

    # 9. Tiling side: core function definitions
    # ComputeGroupSplit 仅在算子使用 Group 模板时必查（overview §2.1：算子选择 1~3 个模板）
    tiling_functions = [
        "ComputeAUbFactor",
        "ComputeRUbFactor",
        "ExpandAIfRFullyLoaded",
        "SetWorkspaceSize",
        "FillAndLogTilingData",
    ]
    uses_group = (
        "ProcessGroup" in design_text
        or "GroupKernel" in design_text
        or "ASCENDC_TPL_BOOL_SEL(isGroup, 1)" in design_text
    )
    if uses_group:
        tiling_functions.append("ComputeGroupSplit")
    missing_tiling = []
    for func_name in tiling_functions:
        if not _has_function_definition(blocks, func_name):
            missing_tiling.append(func_name)
    if missing_tiling:
        errors.append(
            f"GATE-9: missing tiling function definitions: {', '.join(missing_tiling)}"
        )

    return errors
