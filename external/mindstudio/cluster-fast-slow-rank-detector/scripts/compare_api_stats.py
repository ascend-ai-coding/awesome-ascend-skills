#!/usr/bin/env python3
"""
对比慢卡与快卡的 CANN API 统计（api_statistic），找出慢卡上耗时异常或调用异常的 API，作为 Host 下发瓶颈等证据。

支持：
- text：从 ASCEND_PROFILER_OUTPUT/api_statistic.csv 读取（通过 profiler_info_{rank}.json 定位目录）
- db：从 ascend_pytorch_profiler_{rank}.db 的 CANN_API 表聚合（startNs/endNs -> duration）
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from rank_data_finder import resolve_rank_paths

# api_statistic.csv 列名多种写法（与 op 类似，名称列多为 api name）
API_FIELD_MAPPINGS = {
    "api_name": ["api name", "API Name", "api_name", "ApiName", "Name", "name"],
    "total_time": ["Time(us)", "Total Time(us)", "Total Time (us)", "total_time", "TotalTime", "Self Time(us)", "Self Time (us)"],
    "count": ["Count", "count", "Call Count"],
    "avg_time": ["Avg(us)", "Avg Time(us)", "Avg Time (us)", "avg_time", "AvgTime"],
}


def _find_col(df, field_key: str):
    import pandas as pd
    for name in API_FIELD_MAPPINGS.get(field_key, []):
        if name in df.columns:
            return df[name]
    return None


def load_api_stats_csv(path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path)
    name_col = _find_col(df, "api_name")
    time_col = _find_col(df, "total_time")
    count_col = _find_col(df, "count")
    if name_col is None or time_col is None:
        raise ValueError(f"CSV 缺少 api 名称列或 total_time 列: {path}")
    df = df.copy()
    df["_total_time_us"] = pd.to_numeric(time_col, errors="coerce").fillna(0)
    df["_count"] = pd.to_numeric(count_col, errors="coerce").fillna(1) if count_col is not None else 1
    name_header = name_col.name if hasattr(name_col, "name") else name_col
    agg = df.groupby(name_header).agg(total_time_us=("_total_time_us", "sum"), count=("_count", "sum")).reset_index()
    agg = agg.rename(columns={name_header: "api_name"})
    return agg.to_dict("records")


def load_api_stats_db(db_path: str) -> List[Dict[str, Any]]:
    """从 ascend_pytorch_profiler_{rank}.db 的 CANN_API 表聚合：duration_ns = endNs - startNs，按 name 聚合。"""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('CANN_API','STRING_IDS')"
        )
        tables = {r[0] for r in cur.fetchall()}
        if "CANN_API" not in tables:
            raise ValueError(f"DB 中缺少 CANN_API 表: {db_path}")

        sql = """
        SELECT
            COALESCE(s.value, CAST(a.name AS TEXT)) AS api_name,
            SUM(a.endNs - a.startNs) / 1000.0 AS total_time_us,
            COUNT(*) AS count
        FROM CANN_API a
        LEFT JOIN STRING_IDS s ON s.id = a.name
        GROUP BY COALESCE(s.value, CAST(a.name AS TEXT))
        """
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return [
            {"api_name": r[0], "total_time_us": r[1], "count": int(r[2])}
            for r in rows
        ]
    finally:
        conn.close()


def compare_api_stats(
    slow_data: List[Dict[str, Any]],
    fast_data: List[Dict[str, Any]],
    top_n: int = 20,
) -> Dict[str, Any]:
    """按 api_name 对齐，计算耗时差、比值，按差异从大到小排序，取 top_n。"""
    fast_by_name = {(r["api_name"]): r for r in fast_data}
    results = []
    for s in slow_data:
        name = s["api_name"]
        slow_time = float(s.get("total_time_us", 0))
        slow_count = int(s.get("count", 0))
        f = fast_by_name.get(name)
        if not f:
            diff_time = slow_time
            ratio = float("inf") if slow_time else 1.0
            fast_time = 0.0
            fast_count = 0
            slow_avg = (slow_time / slow_count) if slow_count else 0.0
            fast_avg = 0.0
        else:
            fast_time = float(f.get("total_time_us", 0))
            fast_count = int(f.get("count", 0))
            diff_time = slow_time - fast_time
            ratio = (slow_time / fast_time) if fast_time else (float("inf") if slow_time else 1.0)
        slow_avg = (slow_time / slow_count) if slow_count else 0.0
        fast_avg = (fast_time / fast_count) if fast_count else 0.0
        avg_time_diff = slow_avg - fast_avg
        results.append({
            "api_name": str(name),
            "slow_total_time_us": slow_time,
            "slow_count": slow_count,
            "fast_total_time_us": fast_time,
            "fast_count": fast_count,
            "diff_total_time_us": diff_time,
            "total_time_ratio_slow_vs_fast": ratio,
            "slow_avg_time_us": slow_avg,
            "fast_avg_time_us": fast_avg,
            "avg_time_diff_us": avg_time_diff,
        })
    results.sort(key=lambda x: abs(x["diff_total_time_us"]), reverse=True)
    top = results[:top_n]
    return {
        "summary": {
            "total_apis_slow": len(slow_data),
            "total_apis_fast": len(fast_data),
            "top_n_by_abs_diff": top_n,
        },
        "top_differences": top,
        "conclusion": "以下为慢卡相对快卡差异最大的 API（按 |diff_total_time_us| 排序），可用于定位 Host 下发或 CANN 调用瓶颈。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比慢卡与快卡的 API 统计，输出差异最大的 API 列表。",
    )
    parser.add_argument("cluster_path", type=str, help="集群性能数据根目录")
    parser.add_argument("slow_rank", type=int, help="慢卡 Rank 号")
    parser.add_argument("fast_rank", type=int, help="快卡 Rank 号（作为基准）")
    parser.add_argument("--slow-path", type=str, default=None, help="慢卡 api_statistic.csv 或 .db 路径（可选）")
    parser.add_argument("--fast-path", type=str, default=None, help="快卡 api_statistic.csv 或 .db 路径（可选）")
    parser.add_argument("--top", type=int, default=20, help="输出差异最大的前 N 条 API，默认 20")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    slow_path, fast_path, data_type, err = resolve_rank_paths(
        args.cluster_path, args.slow_rank, args.fast_rank, "api",
        slow_path=args.slow_path, fast_path=args.fast_path,
    )
    if err:
        print(err, file=sys.stderr)
        return 1

    try:
        if data_type == "text":
            slow_data = load_api_stats_csv(slow_path)
            fast_data = load_api_stats_csv(fast_path)
        else:
            slow_data = load_api_stats_db(slow_path)
            fast_data = load_api_stats_db(fast_path)
    except Exception as e:
        print(f"加载数据失败: {e}", file=sys.stderr)
        return 2

    result = compare_api_stats(slow_data, fast_data, top_n=args.top)
    result["meta"] = {
        "slow_rank": args.slow_rank,
        "fast_rank": args.fast_rank,
        "slow_path": slow_path,
        "fast_path": fast_path,
        "data_type": data_type,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== 慢卡 vs 快卡 API 统计对比 ===")
        print(f"慢卡 Rank {args.slow_rank}，快卡 Rank {args.fast_rank}，数据来源: {data_type}")
        print(f"差异最大的前 {args.top} 个 API（按 |diff_total_time_us| 排序）：\n")
        for i, row in enumerate(result["top_differences"], 1):
            print(f"{i}. {row['api_name']}")
            print(f"   慢卡: total_time_us={row['slow_total_time_us']:.0f}, count={row['slow_count']}")
            print(f"   快卡: total_time_us={row['fast_total_time_us']:.0f}, count={row['fast_count']}")
            print(f"   diff_total_time_us={row['diff_total_time_us']:.0f}, ratio(slow/fast)={row['total_time_ratio_slow_vs_fast']}, avg_time_diff_us={row['avg_time_diff_us']:.2f}")
            print()
        print(result["conclusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
