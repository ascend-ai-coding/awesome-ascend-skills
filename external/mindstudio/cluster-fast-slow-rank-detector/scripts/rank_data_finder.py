"""
Discover slow/fast rank profiler data paths under a cluster root.
Supports text (op_statistic.csv / api_statistic.csv) and db (ascend_pytorch_profiler_{rank}.db).
"""
from __future__ import annotations

import json
import os
from typing import Tuple, Optional, Literal

DataType = Literal["text", "db"]


# 限制从 cluster_path 起向下遍历的最大深度，避免误入深层无关目录
DEFAULT_MAX_WALK_DEPTH = 6


def find_profiler_info_dirs(
    cluster_path: str,
    max_depth: int = DEFAULT_MAX_WALK_DEPTH,
) -> dict[int, str]:
    """
    Find all profiler_info_{rank_id}.json under cluster_path and return
    rank_id -> directory (parent of the json file).
    Traversal is limited to max_depth levels below cluster_path.
    """
    result = {}
    cluster_path = os.path.abspath(cluster_path)
    if not os.path.isdir(cluster_path):
        return result
    cluster_path = os.path.normpath(cluster_path)
    for root, dirs, files in os.walk(cluster_path):
        try:
            rel = os.path.relpath(root, cluster_path)
        except ValueError:
            continue
        depth = len(rel.split(os.sep)) if rel != "." else 0
        if depth > max_depth:
            dirs.clear()
            continue
        for f in files:
            if f.startswith("profiler_info_") and f.endswith(".json"):
                try:
                    rank_str = f.replace("profiler_info_", "").replace(".json", "")
                    rank_id = int(rank_str)
                    result[rank_id] = root
                except ValueError:
                    continue
    return result


def find_rank_csv_path(
    cluster_path: str, rank_id: int, csv_name: str
) -> Optional[str]:
    """
    Find path to csv_name (op_statistic.csv or api_statistic.csv) for the given rank.
    csv_name must be 'op_statistic.csv' or 'api_statistic.csv'.
    """
    dirs = find_profiler_info_dirs(cluster_path)
    rank_dir = dirs.get(rank_id)
    if not rank_dir:
        return None
    candidate = os.path.join(rank_dir, "ASCEND_PROFILER_OUTPUT", csv_name)
    return candidate if os.path.isfile(candidate) else None


def find_rank_db_path(cluster_path: str, rank_id: int) -> Optional[str]:
    """Find path to ascend_pytorch_profiler_{rank_id}.db for the given rank."""
    dirs = find_profiler_info_dirs(cluster_path)
    rank_dir = dirs.get(rank_id)
    if not rank_dir:
        return None
    candidate = os.path.join(
        rank_dir, "ASCEND_PROFILER_OUTPUT", f"ascend_pytorch_profiler_{rank_id}.db"
    )
    return candidate if os.path.isfile(candidate) else None


def _data_type_from_path(path: str) -> DataType:
    return "db" if path.lower().endswith(".db") else "text"


def resolve_rank_paths(
    cluster_path: str,
    slow_rank: int,
    fast_rank: int,
    mode: Literal["op", "api"],
    slow_path: Optional[str] = None,
    fast_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], DataType, str]:
    """
    Resolve paths for slow and fast rank data.

    mode: 'op' -> op_statistic.csv or db for op; 'api' -> api_statistic.csv or db for api.

    Returns:
        (slow_path, fast_path, data_type, error_message)
        If error_message is non-empty, path resolution failed.
    """
    csv_name = "op_statistic.csv" if mode == "op" else "api_statistic.csv"

    if slow_path and fast_path:
        if not os.path.isfile(slow_path):
            return None, None, "text", f"慢卡路径不存在或不是文件: {slow_path}"
        if not os.path.isfile(fast_path):
            return None, None, "text", f"快卡路径不存在或不是文件: {fast_path}"
        dt = _data_type_from_path(slow_path)
        return slow_path, fast_path, dt, ""

    cluster_path = os.path.abspath(cluster_path)
    if not os.path.isdir(cluster_path):
        return None, None, "text", f"集群数据路径不存在或不是目录: {cluster_path}"

    # Resolve slow/fast paths (explicit or discovered)
    def resolve_one(rank_id: int, explicit: Optional[str], get_csv=None, get_db=None):
        if explicit and os.path.isfile(explicit):
            if explicit.lower().endswith(".csv"):
                return explicit, None
            if explicit.lower().endswith(".db"):
                return None, explicit
        return (get_csv() if get_csv else None), (get_db() if get_db else None)

    slow_csv, slow_db = resolve_one(
        slow_rank, slow_path,
        get_csv=lambda: find_rank_csv_path(cluster_path, slow_rank, csv_name),
        get_db=lambda: find_rank_db_path(cluster_path, slow_rank),
    )
    fast_csv, fast_db = resolve_one(
        fast_rank, fast_path,
        get_csv=lambda: find_rank_csv_path(cluster_path, fast_rank, csv_name),
        get_db=lambda: find_rank_db_path(cluster_path, fast_rank),
    )

    if slow_csv and fast_csv:
        return slow_csv, fast_csv, "text", ""
    if slow_db and fast_db:
        return slow_db, fast_db, "db", ""

    if slow_db and fast_db:
        return slow_db, fast_db, "db", ""

    missing = []
    if not slow_csv and not slow_db:
        missing.append(f"Rank {slow_rank}")
    if not fast_csv and not fast_db:
        missing.append(f"Rank {fast_rank}")
    return (
        None,
        None,
        "text",
        f"未找到以下卡的数据（未找到 {csv_name} 或 ascend_pytorch_profiler_{{rank}}.db）：{', '.join(missing)}。请确认集群路径或手动指定 --slow-path / --fast-path。",
    )
