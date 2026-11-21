# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Include 依赖分析器（snapshot_construction 内部模块）。

从 aclnn_patch_jump/analyze_includes.py 内化的核心函数，提供 C++ #include 依赖的
提取、三级分类（① 自身目录 / ② 仓内其他 / ③ 外部）与传递闭包展开能力。
针对单个 ops-* 仓库，供 postprocess/expand_file_deps.py 复用。
"""

import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional

# 排除的非算子分类目录
NON_OP_DIRS = {
    "common",
    "cmake",
    "docs",
    "examples",
    "scripts",
    "tests",
    "experimental",
    "torch_extension",
}

# 源文件后缀集合（预扫描与 anchor 构建共用）
SOURCE_EXTENSIONS = {".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx"}


def build_file_index(ops_root: Path, op_dir: Optional[Path] = None) -> dict[str, Any]:
    """枚举 ops_root 下所有子文件，构建路径索引与文件名索引。

    返回 {"rel": {相对路径: 绝对路径}, "name": {文件名: [绝对路径, ...]}}
    供 classify_include / display_to_abs 做路径搜索替代路径解析。

    rel_index 始终全仓扫描（step 1/2 精确路径匹配需要解析跨算子 ../ include）；
    name_index 当提供 op_dir 时扫描 op_dir + 仓内所有非算子目录（NON_OP_DIRS，
    如 common、tests 等），排除兄弟算子目录以防止跨算子同名文件误匹配；
    未提供 op_dir 时回退到全仓扫描（向后兼容）。
    """
    rel_index: dict[str, Path] = {}
    name_index: dict[str, list[Path]] = {}
    ops_root = Path(ops_root)

    for f in ops_root.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(ops_root))
            rel_index[rel] = f

    if op_dir is None:
        for f in ops_root.rglob("*"):
            if f.is_file():
                name_index.setdefault(f.name, []).append(f)
    else:
        op_dir = Path(op_dir)
        for f in op_dir.rglob("*"):
            if f.is_file():
                name_index.setdefault(f.name, []).append(f)
        for top in ops_root.iterdir():
            if top.is_dir() and top.name in NON_OP_DIRS:
                for f in top.rglob("*"):
                    if f.is_file():
                        name_index.setdefault(f.name, []).append(f)

    return {"rel": rel_index, "name": name_index}


def get_repo_root(abs_path: Path, ops_root: Path) -> Optional[Path]:
    """返回 abs_path 所属的 ops repo root（单仓：就是 ops_root 或 None）。"""
    try:
        abs_path.relative_to(ops_root)
        return ops_root
    except ValueError:
        return None


def display_to_abs(display_path: str, ops_root: Path, file_index: Optional[dict[str, Any]] = None) -> Optional[Path]:
    """ops-math/.../foo.h → /.../ops-math/.../foo.h（display → 绝对路径）。

    有 file_index 时：通过索引路径搜索查找。
    无 file_index 时：回退到原路径拼接方式。
    """
    prefix = f"{ops_root.name}/"
    if not display_path.startswith(prefix):
        return None
    start = len(prefix)
    rel_path = display_path[start:]
    if file_index is not None:
        return file_index["rel"].get(rel_path)
    return ops_root / rel_path


def find_operator_dirs(ops_root: Path, op_name: str) -> list[tuple[Path, Path]]:
    """在本仓中搜索算子目录。返回 [(repo_root, op_dir)]。"""
    results = []
    for cat_dir in sorted(ops_root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in NON_OP_DIRS:
            continue
        candidate = cat_dir / op_name
        if candidate.is_dir():
            results.append((ops_root, candidate))
    return results


def extract_includes(file_path: Path) -> list[tuple[str, str]]:
    """返回 [(include_text, 'angle'|'quote')]。"""
    includes = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return includes
    for line in content.split("\n"):
        code_part = line.split("//")[0]
        m = re.match(r"\s*#include\s+<(.+?)>", code_part)
        if m:
            includes.append((m.group(1), "angle"))
            continue
        m = re.match(r'\s*#include\s+"(.+?)"', code_part)
        if m:
            includes.append((m.group(1), "quote"))
    return includes


def _search_candidates(
    include_path: str,
    file_dir: Path,
    ops_root: Path,
    file_index: Optional[dict[str, Any]] = None,
) -> list[Path]:
    """构建候选文件路径列表。

    有 file_index 时：通过索引搜索目标文件夹子文件（路径搜索）。
    无 file_index 时：回退到原路径解析方式（file_dir/include_path, ops_root/include_path）。
    """
    if file_index is None:
        return [file_dir / include_path, ops_root / include_path]

    candidates: list[Path] = []
    seen: set[str] = set()
    include_normalized = include_path.replace("\\", "/")

    def _add(path: Path):
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    # (1) include_path 作为 ops_root 相对路径精确匹配
    if include_normalized in file_index["rel"]:
        _add(file_index["rel"][include_normalized])

    # (2) 相对当前文件目录拼接后查索引
    try:
        file_rel = file_dir.relative_to(ops_root)
        combined = os.path.normpath(str(file_rel / include_normalized)).replace("\\", "/")
        if combined in file_index["rel"]:
            _add(file_index["rel"][combined])
    except ValueError:
        pass

    # (3) 按文件名模糊匹配（兜底：处理 ../ 或路径不匹配的情况）
    #     按路径相似度排序：优先匹配 include 路径目录结构，避免同名文件误匹配
    filename = Path(include_path).name
    fuzzy_matches = file_index["name"].get(filename, [])
    if fuzzy_matches:
        include_parts = tuple(p for p in Path(include_normalized).parts if p != "..")

        def _suffix_match_len(candidate: Path) -> int:
            try:
                cand_parts = candidate.relative_to(ops_root).parts
            except ValueError:
                return 0
            n = min(len(include_parts), len(cand_parts))
            score = 0
            for i in range(1, n + 1):
                if include_parts[-i] == cand_parts[-i]:
                    score += 1
                else:
                    break
            return score

        for match in sorted(fuzzy_matches, key=_suffix_match_len, reverse=True):
            _add(match)

    return candidates


def classify_include(
    include_path: str,
    file_dir: Path,
    ops_root: Path,
    op_dir: Path,
    file_index: Optional[dict[str, Any]] = None,
):
    """
    多级搜索:
      (1) 相对当前文件目录
      (2) 从 ops 仓库根目录拼接
    返回 (category, display_path_or_None):
      category: 'self' | 'repo_other' | 'external'

    若提供 file_index（build_file_index 构建），则通过索引路径搜索查找文件，
    替代直接路径解析。
    """
    candidates = _search_candidates(include_path, file_dir, ops_root, file_index)

    for resolved in candidates:
        resolved = resolved.resolve()
        # 检查是否在本仓内
        repo_root = get_repo_root(resolved, ops_root)
        if repo_root is None:
            continue
        if not resolved.is_file():
            continue

        display = f"{repo_root.name}/{resolved.relative_to(repo_root)}"

        # 判断是否在算子自身目录下
        try:
            resolved.relative_to(op_dir)
            return ("self", display)
        except ValueError:
            return ("repo_other", display)

    return ("external", None)


def _iter_dir_cpp_companions(abs_path: Path) -> list[Path]:
    """返回 abs_path 同目录下的 .cpp/.cc/.cxx 实现文件（排除自身）。

    CANN 算子的实现单元按目录组织：头文件与同名/同角色的 .cpp 实现同目录
    （如 op_kernel/arch35/ 下 kernel 入口 .cpp + qbmm_*.h；op_host/op_tiling/
    arch35/ 下各 tiling 模板 .cpp/.h），而 .cpp 不被任何文件 #include，
    靠编译链接与模板自注册组织。因此当某文件进入闭包时，其同目录 .cpp
    作为"目录伴生"一并纳入闭包，其 #include 亦被展开。
    """
    cpp_suffixes = {".cpp", ".cc", ".cxx"}
    try:
        entries = sorted(abs_path.parent.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if entry.suffix.lower() in cpp_suffixes and entry != abs_path]


def build_transitive(
    direct_self: list[str],
    direct_repo_other: list[dict[str, str]],
    direct_external: list[str],
    ops_root: Path,
    op_dir: Path,
    depth_limit: Optional[int] = None,
    file_cache: Optional[dict[str, dict[str, Any]]] = None,
    file_index: Optional[dict[str, Any]] = None,
    expand_dir_cpp: bool = False,
    max_companions: int = 200,
):
    """从直接依赖的 ① ② 出发 BFS 展开传递闭包直到 ③。返回传递闭包统计。

    Args:
        expand_dir_cpp: 目录伴生展开。对进入闭包的每个文件，其同目录下的
            .cpp/.cc/.cxx 实现文件（不被 #include 引用、靠编译链接/自注册组织）
            一并纳入闭包并展开其 include。默认 False（向后兼容）。
        max_companions: 目录伴生文件总数上限（防爆兜底），按目录内 sorted 顺序截断。
    """
    if file_cache is None:
        file_cache = {}

    def parse_and_classify(abs_path: Path) -> dict[str, list[Any]]:
        key = str(abs_path)
        if key in file_cache:
            return file_cache[key]
        result = {"self": [], "repo_other": [], "external": []}
        file_dir = abs_path.parent
        for include_path, qtype in extract_includes(abs_path):
            if qtype == "angle":
                continue
            cat, display = classify_include(include_path, file_dir, ops_root, op_dir, file_index=file_index)
            if cat == "self":
                result["self"].append(display)
            elif cat == "repo_other":
                result["repo_other"].append({"原始引用": include_path, "解析路径": display})
            else:
                result["external"].append(include_path)
        result["self"] = list(dict.fromkeys(result["self"]))
        result["external"] = list(dict.fromkeys(result["external"]))
        file_cache[key] = result
        return result

    visited: set[str] = set()
    queue: deque[tuple[Path, int]] = deque()

    def enqueue(display_path: str, depth: int):
        abs_path = display_to_abs(display_path, ops_root, file_index=file_index)
        if abs_path is None:
            return
        key = str(abs_path)
        if key not in visited:
            visited.add(key)
            queue.append((abs_path, depth))

    for p in direct_self:
        enqueue(p, depth=1)
    for item in direct_repo_other:
        enqueue(item["解析路径"], depth=1)

    # 种子计入结果
    all_self: set[str] = set(direct_self)
    all_repo_other: dict[str, str] = {}
    for item in direct_repo_other:
        all_repo_other[item["原始引用"]] = item["解析路径"]
    all_external: set[str] = set(direct_external)

    max_depth = 0
    files_opened = 0
    companion_dir_visited: set[str] = set()
    companions_collected = 0

    def _classify_companion(cpp: Path) -> tuple[str, str]:
        """display 前缀格式须与 classify_include 一致（display_to_abs 依赖此前缀解析）。"""
        try:
            cpp.relative_to(op_dir)
            cat = "self"
        except ValueError:
            cat = "repo_other"
        display = f"{ops_root.name}/{cpp.relative_to(ops_root)}"
        return cat, display

    def _collect_dir_companions(abs_path: Path, depth: int) -> None:
        """abs_path 所在目录（每目录一次）的 .cpp 伴生纳入闭包，入队消耗 depth+1。"""
        nonlocal companions_collected
        dir_key = str(abs_path.parent)
        if dir_key in companion_dir_visited:
            return
        companion_dir_visited.add(dir_key)
        for cpp in _iter_dir_cpp_companions(abs_path):
            if companions_collected >= max_companions:
                return
            cat, display = _classify_companion(cpp)
            companions_collected += 1
            if cat == "self":
                all_self.add(display)
            else:
                all_repo_other[f"目录伴生({cpp.name})"] = display
            enqueue(display, depth + 1)

    while queue:
        abs_path, depth = queue.popleft()
        if depth_limit is not None and depth > depth_limit:
            continue
        max_depth = max(max_depth, depth)
        files_opened += 1

        if expand_dir_cpp:
            _collect_dir_companions(abs_path, depth)

        cached = parse_and_classify(abs_path)
        for s in cached["self"]:
            all_self.add(s)
            enqueue(s, depth + 1)
        for item in cached["repo_other"]:
            all_repo_other[item["原始引用"]] = item["解析路径"]
            enqueue(item["解析路径"], depth + 1)
        for ext in cached["external"]:
            all_external.add(ext)

    # ② 仅传递新增
    direct_repo_keys = {item["解析路径"] for item in direct_repo_other}
    new_repo_other = {raw: res for raw, res in all_repo_other.items() if res not in direct_repo_keys}

    total_quoted = len(all_self) + len(all_repo_other) + len(all_external)
    t_internal = len(all_self) + len(all_repo_other)

    return {
        "展开文件数": files_opened,
        "最大深度": max_depth,
        "引号include总数": total_quoted,
        "①_自身目录": sorted(all_self),
        "②_仓内其他": [{"原始引用": raw, "解析路径": res} for raw, res in sorted(all_repo_other.items())],
        "③_外部": sorted(all_external),
        "②_仅传递新增": [{"原始引用": raw, "解析路径": res} for raw, res in sorted(new_repo_other.items())],
        "自包含率": round(len(all_self) / t_internal, 4) if t_internal else 0.0,
        "仓内率": round(len(all_repo_other) / t_internal, 4) if t_internal else 0.0,
    }
