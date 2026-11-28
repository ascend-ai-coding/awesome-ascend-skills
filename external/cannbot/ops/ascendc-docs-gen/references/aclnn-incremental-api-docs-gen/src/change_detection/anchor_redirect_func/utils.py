# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Utility functions for the anchor redirect mapper."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMClient
from src.logger import debug, error, info, warning
from src.snapshot_construction.pub_ref import PubRefConfig


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_json_by_ops_name(anchor_path: str, ops_name: str) -> Optional[str]:
    """Find the anchor JSON file for a given ops name."""
    anchor_path_obj = Path(*Path(anchor_path).parts[:-1])
    # Try exact match first
    exact_match = anchor_path_obj / f"{ops_name}.json"
    if exact_match.exists():
        return str(exact_match)

    # Try with _v2 suffix pattern
    v2_match = anchor_path_obj / f"{ops_name}_v2.json"
    if v2_match.exists():
        return str(v2_match)

    # Search for any file containing the ops_name
    for json_file in anchor_path_obj.glob("*.json"):
        if ops_name in json_file.stem:
            return str(json_file)

    return anchor_path_obj


def extract_functions_from_file(file_path: str) -> List[str]:
    """Extract function names from a C++ source file."""
    functions = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Simple function extraction - look for function declarations
        import re

        # Match function declarations: return_type function_name(...)
        pattern = r"(?:virtual\s+)?(?:\w+(?:<[^>]+>)?(?:\s*\*)?(?:\s*&)?\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{"
        matches = re.findall(pattern, content)
        functions = list(set(matches))
    except Exception as e:
        error(f"Error extracting functions from {file_path}: {e}")

    return functions


def jaccard_similarity(set1: List, set2: List) -> float:
    """Calculate Jaccard similarity between two lists."""
    if not set1 and not set2:
        return 1.0

    s1 = set(set1)
    s2 = set(set2)

    if not s1 or not s2:
        return 0.0

    intersection = len(s1 & s2)
    union = len(s1 | s2)

    return intersection / union if union > 0 else 0.0


def extract_call_info_from_code_map(code_map_data: Dict) -> Dict[str, Dict]:
    """Extract call/callee/return info from code_map.

    Returns a dict mapping function names to their call/callee/return info.
    """
    result = {}
    nodes = code_map_data.get("nodes", [])
    edges = code_map_data.get("edges", [])

    # Build a map of function names from nodes
    function_names = set()
    for node in nodes:
        if node.get("type") == "Function":
            func_id = node.get("id", "")
            function_names.add(func_id)

    # Extract edges info
    calls_from = {}  # source -> list of targets it calls
    calls_to = {}  # target -> list of sources that call it

    for edge in edges:
        if edge.get("type") == "calls":
            source = edge.get("source", "")
            target = edge.get("target", "")

            if source not in calls_from:
                calls_from[source] = []
            calls_from[source].append(target)

            if target not in calls_to:
                calls_to[target] = []
            calls_to[target].append(source)

    # For each function, build its call/callee info
    for func_name in function_names:
        result[func_name] = {
            "call": calls_from.get(func_name, []),  # functions this func calls
            "callee": calls_to.get(func_name, []),  # functions that call this func
            "return": [],  # Return info not directly available in code_map
        }

    return result


def normalize_path(path: str) -> str:
    """Normalize file path for comparison."""
    return path.replace("\\", "/").strip()


def find_matching_file(new_file_name: str, search_dir: str) -> Optional[str]:
    """Find a file with matching name in the search directory."""
    search_path = Path(search_dir)  # .parent
    if not search_path.exists():
        return None

    # Exact match
    exact = search_path.rglob(new_file_name)
    for f in exact:
        return str(f)

    # Case-insensitive match
    new_lower = new_file_name.lower()
    for f in search_path.rglob("*"):
        if f.is_file() and f.name.lower() == new_lower:
            return str(f)

    return None


def detect_component(path_parts: List[str]) -> str:
    """Detect the component directory from file path parts.

    Common components: op_api, op_host, op_kernel, op_graph
    """
    components = ["op_api", "op_host", "op_kernel", "op_graph", "tests", "examples"]

    for part in path_parts:
        if part in components:
            return part

    # Return default component
    return "op_api"


def get_directory_tree(dir_path: str, max_depth: int = 3) -> str:
    """Get directory tree structure as string.

    Args:
        dir_path: Root directory path
        max_depth: Maximum depth to traverse

    Returns:
        Tree structure string
    """
    from pathlib import Path

    def _tree(path: Path, prefix: str = "", depth: int = 0) -> str:
        if depth > max_depth:
            return ""

        result = ""
        path = path.parent
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                result += prefix + current_prefix + item.name + "\n"

                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    result += _tree(item, prefix + extension, depth + 1)
        except PermissionError:
            pass

        return result

    root = Path(dir_path)
    if not root.exists():
        return ""

    tree = root.name + "\n"
    tree += _tree(root)
    return tree


def extract_file_functions(file_path: str) -> List[str]:
    """Extract function names from a file.

    Args:
        file_path: Path to the source file

    Returns:
        List of function names
    """
    import re

    functions = []
    debug(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Match function definitions: return_type function_name(...)
        # Handle various patterns: void func(), int func(int a), void (*func)(), etc.
        patterns = [
            # Basic function definition
            r"(?:inline\s+)?(?:static\s+)?(?:virtual\s+)?"
            r"(?:\w+(?:<[^>]+>)?(?:\s*\*)?(?:\s*&)?\s*)+"
            r"(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?\{",
            # Constructor/Destructor
            r"~?(\w+)\s*\([^)]*\)\s*\{",
        ]
        debug(content)
        for pattern in patterns:
            matches = re.findall(pattern, content)
            functions.extend(matches)

        # Filter out keywords and common non-functions
        keywords = {
            "if",
            "else",
            "for",
            "while",
            "switch",
            "catch",
            "return",
            "sizeof",
            "new",
            "delete",
        }
        functions = [f for f in functions if f not in keywords and len(f) > 1]

        # Remove duplicates while preserving order
        seen = set()
        unique_functions = []
        for f in functions:
            if f not in seen:
                seen.add(f)
                unique_functions.append(f)

        return unique_functions

    except Exception as e:
        error(f"Error extracting functions from {file_path}: {e}")
        return []


def call_llm_for_file_match(
    new_file_name: str,
    new_functions: List[str],
    candidate_files: List[Dict],
    directory_tree: str,
) -> Optional[str]:
    """Use LLM to find the most similar old file.

    Args:
        new_file_name: Name of the new file
        new_functions: List of function names in the new file
        candidate_files: List of dicts with 'path' and 'functions' for each candidate
        directory_tree: Directory tree structure of old operator

    Returns:
        Path to the most similar old file, or None if no match found
    """
    # Build prompt
    prompt = f"""你是一个代码匹配专家。请根据以下信息，找出最可能对应的旧版本文件。

## 新版本文件信息
- 文件名: {new_file_name}
- 函数列表: {", ".join(new_functions) if new_functions else "无"}

## 旧版本目录结构:
```
{directory_tree}
```

## 候选文件详细信息:
"""

    for i, candidate in enumerate(candidate_files):
        func_list = candidate.get("functions", [])
        prompt += f"\n{i + 1}. {candidate['path']}\n"
        prompt += f"   函数: {', '.join(func_list) if func_list else '无'}\n"

    prompt += """
## 任务
请从候选文件中找出与新文件最相似的一个。相似度判断依据:
1. 文件名相似度（如 aclnn_abs_v2.cpp 对应 aclnn_abs.cpp）
2. 函数列表相似度（包含相同或相似的函数名）

请只返回最匹配的文件路径，不要包含其他内容。
如果所有候选文件都不匹配，请返回 "NONE"。
"""

    try:
        client = LLMClient(model=PubRefConfig.MODEL)
        result = client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1000).strip()

        if not result or result.upper() == "NONE":
            return None

        for candidate in candidate_files:
            if candidate["path"].name in result or Path(candidate["path"]).name in result:
                return candidate["path"]

        return None

    except Exception as e:
        warning(f"Warning: LLM call failed: {e}")
        return None
