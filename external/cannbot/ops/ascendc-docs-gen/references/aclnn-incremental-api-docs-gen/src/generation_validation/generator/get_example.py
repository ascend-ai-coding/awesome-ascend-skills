# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Get example code from examples directory"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from src.logger import debug, info, log_result

from .. import config
from ..dashboard import dashboard_done, dashboard_log, dashboard_step, dashboard_warn

console_arg = Console()


@dataclass
class ExampleItem:
    describe: str
    code: str
    source_file: str = ""


def get_example_code(root_path: str) -> dict[str, str]:
    """
    Extract example code from examples directory based on operator name.
    Compatible with single-file legacy usage: returns {operator_name: code}.

    Args:
        root_path: Root directory path containing the operator

    Returns:
        Dict with operator name as key and example code content as value
    """
    operator_name = _extract_operator_name(root_path)
    if not operator_name:
        log_result("No operator_name found")
        return {}
    log_result(f"[dim]Operator:[/dim] [cyan]{operator_name}[/cyan]")

    examples_dir = Path(root_path) / config.EXAMPLES_DIR
    log_result(f"[dim]Examples dir:[/dim] [dim]{examples_dir}[/dim]")

    if not examples_dir.exists():
        log_result(f"[red]{operator_name}'s example dir not found[/red]")
        return {}

    cpp_files = list(examples_dir.glob("*.cpp"))
    if not cpp_files:
        log_result(f"[yellow]No .cpp files found in {examples_dir}[/yellow]")
        return {}
    log_result(f"[dim]Found {len(cpp_files)} example file(s)[/dim]")

    matched_file = _find_matched_example(operator_name, cpp_files)
    if not matched_file:
        log_result(f"[red]No matching example file found for '{operator_name}'[/red]")
        return {}
    log_result(f"[dim]匹配文件:[/dim] [cyan]{matched_file.name}[/cyan]")
    content = matched_file.read_text(encoding="utf-8")
    cleaned_content = _remove_header_comment(content)
    log_result(f"[dim]代码行数:[/dim] {len(cleaned_content.splitlines())}")
    return {operator_name: cleaned_content}


def get_all_example_codes(root_path: str, operator_name: str) -> dict[str, str]:
    """
    Extract ALL example codes from examples directory, one per .cpp file.

    Derives describe from filename:
      test_aclnn_addcmul.cpp       -> describe="aclnnAddcmul", code=content
      test_aclnn_inplace_addcmul.cpp -> describe="aclnnInplaceAddcmul", code=content

    This is used in full_pipeline mode where no doc_json exists yet.

    Args:
        root_path: Root directory path containing the operator

    Returns:
        Dict mapping describe -> cleaned code content for each .cpp file
    """
    # operator_name = _extract_operator_name(root_path)
    if not operator_name:
        log_result("[red]No operator_name found[/red]")
        return {}
    log_result(f"[dim]Operator:[/dim] [cyan]{operator_name}[/cyan]")

    examples_dir = Path(root_path) / config.EXAMPLES_DIR
    log_result(f"[dim]Examples dir:[/dim] [dim]{examples_dir}[/dim]")

    if not examples_dir.exists():
        log_result(f"[red]{operator_name}'s example dir not found[/red]")
        return {}

    cpp_files = list(examples_dir.glob("*.cpp"))
    if not cpp_files:
        log_result(f"[yellow]No .cpp files found in {examples_dir}[/yellow]")
        return {}
    log_result(f"[dim]Found {len(cpp_files)} example file(s)[/dim]")

    aclnn_name = f"{config.ACLNN_PREFIX}_{operator_name}"
    result = {}

    for cpp_file in sorted(cpp_files):
        filename = cpp_file.stem  # without .cpp
        describe = _derive_describe_from_filename(filename, aclnn_name)
        content = cpp_file.read_text(encoding="utf-8")
        cleaned = _remove_header_comment(content).lstrip()
        result[describe] = cleaned
        log_result(f"  [dim]File [/dim][cyan]{cpp_file.name}[/cyan]" f"  ->  describe: [cyan]{describe}[/cyan]")
        log_result(f"  [dim]代码行数:[/dim] {len(cleaned.splitlines())}")

    return result


def get_example_codes_by_describe(
    root_path: str,
    describe_list: list[str],
) -> dict[str, str]:
    """
    Extract example code from examples directory, matching each describe to a specific .cpp file.

    For each describe in describe_list, finds the best matching .cpp file and reads its content.
    Example: describe "aclnnInplaceAddcmul" matches file "test_aclnn_inplace_addcmul.cpp".

    Args:
        root_path: Root directory path containing the operator
        describe_list: List of describe strings from doc_json content, used as matching hints.
                       E.g. ["aclnnAddcmul", "aclnnInplaceAddcmul"]

    Returns:
        Dict mapping describe -> cleaned code content
    """
    operator_name = _extract_operator_name(root_path)
    if not operator_name:
        log_result("[red]No operator_name found[/red]")
        return {}
    log_result(f"[dim]Operator:[/dim] [cyan]{operator_name}[/cyan]")

    examples_dir = Path(root_path) / config.EXAMPLES_DIR
    log_result(f"[dim]Examples dir:[/dim] [dim]{examples_dir}[/dim]")

    if not examples_dir.exists():
        log_result(f"[red]{operator_name}'s example dir not found[/red]")
        return {}

    cpp_files = list(examples_dir.glob("*.cpp"))
    if not cpp_files:
        log_result(f"[yellow]No .cpp files found in {examples_dir}[/yellow]")
        return {}
    log_result(f"[dim]Found {len(cpp_files)} example file(s)[/dim]")
    dashboard_step("验证文档模块")
    dashboard_log("[dim]\\[## 调用示例][/dim] 步骤 2: 验证文档模块")
    return _match_examples_by_list(operator_name, cpp_files, describe_list)


def get_example_doc_part(root_path: str) -> dict[str, str]:
    """
    Extract example content from a single JSON file in the root_path.
    Expected JSON structure: {"example": {"content": [...]}}
    Returns dict mapping describe -> code for each item in content array.
    """
    log_result(f"[dim]\\[## 调用示例][/dim]Root path: [dim]{root_path}[/dim]")
    dashboard_step("Load Evidence")
    dashboard_log("[dim]\\[## 调用示例][/dim] 步骤 1: 加载代码证据")

    root = Path(root_path)

    if root.is_file() and root.suffix == ".json":
        doc_file = root
        log_result(f"[dim]\\[## 调用示例][/dim]Using JSON file directly: {doc_file.name}")
    else:
        doc_files = sorted(root.glob("*.json"))

        if len(doc_files) == 0:
            log_result(f"[red]No JSON files found in {root_path}[/red]")
            return {}
        log_result(f"[dim]\\[## 调用示例][/dim]Found {len(doc_files)} JSON file(s)")

        if len(doc_files) > 1:
            file_choices = ", ".join(f"[cyan]{f.name}[/cyan]" for f in doc_files)
            log_result(
                f"[dim]\\[## 调用示例][/dim][yellow]Multiple JSON files: {file_choices}. Using first one[/yellow]"
            )

        doc_file = doc_files[0]
        log_result(f"[dim]\\[## 调用示例][/dim]Parsing example part from {doc_file.name}")

    result_dict = {}

    try:
        # 2. 读取并解析 JSON
        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 3. 获取 content 列表
        content_list = data.get("example", {}).get("content", [])

        if not isinstance(content_list, list):
            log_result(f"[yellow]Warning: 'example.content' is not a list in {doc_file}. " f"Skipping.[/yellow]")
            return {}

        # 4. 遍历列表，构建新的字典 {describe: code}
        for idx, item in enumerate(content_list):
            if not isinstance(item, dict):
                log_result(f"[yellow]Warning: Skipping non-dict item at index {idx}[/yellow]")
                continue

            describe_key = item.get("describe", "")
            code_value = item.get("code", "")

            if describe_key:
                result_dict[describe_key] = code_value
                log_result(f"  [dim]Item {idx + 1}:[/dim] [cyan]{describe_key}[/cyan]")

    except json.JSONDecodeError as e:
        log_result(f"[red]Error: Invalid JSON in {doc_file}: {e}[/red]")
    except Exception as e:
        log_result(f"[red]Error reading {doc_file}: {e}[/red]")

    if result_dict:
        msg = f"  > Extracted {len(result_dict)} example(s)"
        dashboard_done()
    else:
        msg = "  > [yellow]No valid examples found[/yellow]"
        dashboard_warn("No valid examples found")
    log_result(msg)
    dashboard_log(msg)

    return result_dict


def get_example_content_with_source(
    root_path: str,
    describe_map: dict[str, str],
) -> list[ExampleItem]:
    """
    Get example code mapped by describe, returning full ExampleItem objects with file source info.

    Args:
        root_path: Root directory path containing the operator
        describe_map: Dict mapping describe string (from doc_json) -> code string (from doc_json)

    Returns:
        List of ExampleItem with describe, code, and source_file
    """
    operator_name = _extract_operator_name(root_path)
    if not operator_name:
        log_result("[red]No operator_name found[/red]")
        return []

    examples_dir = Path(root_path) / config.EXAMPLES_DIR
    if not examples_dir.exists():
        log_result(f"[red]{operator_name}'s example dir not found[/red]")
        return []

    cpp_files = list(examples_dir.glob("*.cpp"))
    if not cpp_files:
        log_result(f"[yellow]No .cpp files found in {examples_dir}[/yellow]")
        return []

    return _match_examples_with_source(operator_name, cpp_files, describe_map)


def _match_examples_by_describe(
    operator_name: str,
    cpp_files: list[Path],
    describe_map: dict[str, str],
) -> dict[str, str]:
    """Match cpp files to describe keys using describe strings as hints."""
    items = _match_examples_with_source(operator_name, cpp_files, describe_map)
    result = {}
    for item in items:
        result[item.describe] = item.code
    return result


def _match_examples_by_list(
    operator_name: str,
    cpp_files: list[Path],
    describe_list: list[str],
) -> dict[str, str]:
    """
    Match cpp files to a list of describe strings.
    Each describe is treated as a hint to find the corresponding file.

    Logic: For each describe, find the best matching cpp file.
    A describe like "aclnnInplaceAddcmul" should match
    "test_aclnn_inplace_addcmul.cpp".
    """
    aclnn_name = f"{config.ACLNN_PREFIX}_{operator_name}"
    items = []

    for describe in describe_list:
        matched_file = _find_example_for_describe(describe, operator_name, aclnn_name, cpp_files)
        if matched_file:
            content = matched_file.read_text(encoding="utf-8")
            cleaned = _remove_header_comment(content).lstrip()
            items.append((describe, cleaned))
            n_lines = len(cleaned.splitlines())
            msg = (
                f"  > [dim]匹配文件 '{describe}':[/dim] [cyan]{matched_file.name}[/cyan] "
                f"[dim]@代码行数:[/dim] {n_lines}"
            )
        else:
            msg = f"  > [red]No matching file found for describe '{describe}'[/red]"
        log_result(msg)
        dashboard_log(msg)

    return {describe: code for describe, code in items}


def _match_examples_with_source(
    operator_name: str,
    cpp_files: list[Path],
    describe_map: dict[str, str],
) -> list[ExampleItem]:
    """
    Match cpp files to describe keys from doc_json, returning ExampleItem objects.

    For each describe in describe_map, find the most appropriate .cpp file
    based on string matching between the describe and the filename.
    """
    aclnn_name = f"{config.ACLNN_PREFIX}_{operator_name}"
    items = []
    used_files = set()
    leftover_describes = {}

    # Pass 1: Try direct matching using the describe string
    for describe in describe_map:
        matched_file = _find_example_for_describe(describe, operator_name, aclnn_name, cpp_files)
        if matched_file:
            content = matched_file.read_text(encoding="utf-8")
            cleaned = _remove_header_comment(content).lstrip()
            items.append(
                ExampleItem(
                    describe=describe,
                    code=cleaned,
                    source_file=matched_file.name,
                )
            )
            used_files.add(matched_file)
            log_result(f"[dim]匹配文件 '{describe}':[/dim] [cyan]{matched_file.name}[/cyan]")
            log_result(f"[dim]代码行数:[/dim] {len(cleaned.splitlines())}")
        else:
            leftover_describes[describe] = describe_map[describe]

    # Pass 2: Assign leftover files (in case more files than describes, though unlikely)
    unused_files = [f for f in cpp_files if f not in used_files]
    if unused_files:
        log_result(f"[yellow]{len(unused_files)} extra example file(s) not matched to any describe[/yellow]")

    return items


def _find_example_for_describe(
    describe: str,
    operator_name: str,
    aclnn_name: str,
    cpp_files: list[Path],
) -> Path | None:
    """
    Find the best matching cpp file for a given describe string.

    The describe may contain arbitrary text (e.g., intro paragraphs).
    We extract the aclnn function name from it to use as a matching hint.

    Matching priority:
    1. Files containing the full aclnn_name (e.g., aclnnAddcmul)
    2. Files matching suffix keywords from describe (e.g., inplace)
    3. Original single-file matching logic (backwards compatible)
    """
    # Convert describe to lowercase (filenames are all lowercase)
    describe.lower()

    # Extract aclnn function names from describe text
    # Handles both camelCase (aclnnAddcmul) and snake_case (aclnn_addcmul)
    camel_match = re.search(r"(aclnn[^\s\n\r,;:()\[\]{}]+)", describe, re.IGNORECASE)
    snake_match = re.search(r"(aclnn_[a-z0-9_]+)", describe, re.IGNORECASE)

    describe_keywords = set()
    describe_func_snake = set()  # snake_case versions for matching against filenames
    describe_func_camel = set()  # camelCase versions

    if camel_match:
        func = camel_match.group(1)
        describe_func_camel.add(func.lower())
        # snake_case version: aclnnAddcmul -> aclnn_addcmul
        func_snake_text = re.sub(r"([A-Z])", r"_\1", func)
        if func_snake_text.startswith("_"):
            func_snake_text = func_snake_text[1:]
        describe_func_snake.add(func_snake_text.lower())
        # Extract suffix parts
        func_without_aclnn = func.lower().replace("aclnn", "")
        for part in re.split(r"[_\-]", func_without_aclnn):
            if part and part != operator_name.lower():
                describe_keywords.add(part)

    if snake_match:
        snake_func = snake_match.group(1)
        describe_func_snake.add(snake_func)
        snake_parts = snake_func.replace("aclnn_", "").split("_")
        camel_func = snake_parts[0] + "".join(p.capitalize() for p in snake_parts[1:])
        describe_func_camel.add(f"aclnn{camel_func}")
        for part in snake_parts:
            if part and part != operator_name:
                describe_keywords.add(part)

    best_match = None
    best_score = 0

    for cpp_file in cpp_files:
        filename_lower = cpp_file.name.lower()
        score = 0

        # Priority 1: Contains the canonical aclnn_name (case-insensitive)
        if aclnn_name.lower() in filename_lower:
            score += 100

        # Priority 2: any snake_case version of describe function matches filename
        for func_snake in describe_func_snake:
            if func_snake in filename_lower:
                score += 90

        # Priority 3: Each suffix keyword from describe matches in filename
        for kw in describe_keywords:
            if kw in filename_lower:
                score += 15

        # Bonus: inplace keyword
        if "inplace" in describe_keywords and "inplace" in filename_lower:
            score += 10

        # Original matching logic
        if _calculate_similarity(operator_name, filename_lower) > 0:
            score += 30

        if score > best_score:
            best_score = score
            best_match = cpp_file

    if best_score > 30:
        return best_match

    return None


def _derive_describe_from_filename(filename: str, aclnn_name: str) -> str:
    """
    Derive describe string from a cpp filename.

    Examples:
        "test_aclnn_addcmul" -> "aclnnAddcmul"
        "test_aclnn_inplace_addcmul" -> "aclnnInplaceAddcmul"
        "test_abs_1" -> "aclnnAbs_1"

    The describe mirrors the C++ function name pattern: aclnn + CamelCase(rest).
    """
    work = filename
    if work.lower().startswith("test_"):
        work = work[5:]

    # Check if filename starts with aclnn_
    aclnn_prefix = f"{config.ACLNN_PREFIX}_"
    if work.lower().startswith(aclnn_prefix):
        prefix_len = len(aclnn_prefix)
        rest = work[prefix_len:]
        parts = rest.split("_")
        # Convert all parts to CamelCase: inplace_addcmul -> InplaceAddcmul
        camel_parts = []
        for part in parts:
            if part:
                camel_parts.append(part[0].upper() + part[1:])
        return f"{config.ACLNN_PREFIX}{''.join(camel_parts)}"
    else:
        # Fallback: filename
        return filename


def _extract_operator_name(root_path: str) -> str:
    """
    Extract operator name from root path.
    Logic: Skip trailing folders that look like versions (v1, v2, 1.0) or snapshots,
    and return the first meaningful folder name.
    """
    if not root_path:
        return ""

    p = Path(root_path)
    parts = list(p.parts)

    # 定义需要跳过的"版本/快照"后缀模式
    version_pattern = re.compile(
        r"^(v\d+|v\d+\.\d+|\d+(\.\d+)+|base|latest|dev|beta|alpha)$",
        re.IGNORECASE,
    )

    for part in reversed(parts):
        if not version_pattern.match(part):
            return part

    return parts[-1] if parts else ""


def _find_matched_example(operator_name: str, cpp_files: list[Path]) -> Path | None:
    """
    Find the most matching example file based on operator name.

    Priority:
    1. Exact match: test_{aclnn_算子名}_{其它信息}.cpp
    2. Similarity match: find most similar file based on operator name
    """
    aclnn_name = f"{config.ACLNN_PREFIX}_{operator_name}"

    for cpp_file in cpp_files:
        filename = cpp_file.name
        if filename.startswith(config.TEST_PREFIX) and aclnn_name in filename:
            return cpp_file

    best_match = None
    best_score = 0

    for cpp_file in cpp_files:
        filename = cpp_file.name.lower()
        score = _calculate_similarity(operator_name, filename)
        if score > best_score:
            best_score = score
            best_match = cpp_file

    if best_score > 0:
        return best_match

    return None


def _calculate_similarity(operator_name: str, filename: str) -> float:
    """Calculate similarity score between operator name and filename."""
    name_lower = operator_name.lower()
    filename_lower = filename.lower()

    if name_lower in filename_lower:
        return 1.0

    name_parts = re.split(r"[_\-]", name_lower)
    matched_parts = sum(1 for part in name_parts if part in filename_lower)

    if matched_parts > 0:
        return matched_parts / len(name_parts)

    return 0.0


def _remove_header_comment(content: str) -> str:
    """Remove C-style header comment block (/** ... */ or /*! ... */) from content."""
    pattern = r"/\*[\!\*]?(?:[^*]|\*(?!\*/))*?\*/\s*"
    return re.sub(pattern, "", content, flags=re.DOTALL).lstrip()
