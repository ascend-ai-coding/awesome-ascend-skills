#!/usr/bin/env python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Merge operator-theoretical-perf CSV columns into raw_ops_details.json.

The external `operator-theoretical-perf` skill owns theoretical estimation and
emits `operator_analysis.csv` by adding columns to the original
`kernel_details.csv`. This script only copies those added columns onto the
current run's selected-step `raw_ops_details.json.operators[]` so render.py can
use one per-kernel detail carrier.

Canonical alignment:
  raw_ops_details.operators[].org_index == 0-based data row in the original
  kernel_details.csv. operator_analysis.csv must preserve the original CSV row
  order. If it also contains an explicit org_index-like column, that is used.
"""

import argparse
import csv
import json
import logging
import os
import sys
import tempfile


logger = logging.getLogger(__name__)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\t", "")
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_present(row, candidates):
    for name in candidates:
        if name in row:
            value = row.get(name)
            if value is not None and str(value).strip() != "":
                return value
    return None


def _bool_from_text(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "supported", "ok"}:
        return True
    if text in {"false", "0", "no", "n", "n/a", "unsupported"}:
        return False
    return None


THEORY_COLUMN_MAP = {
    "theoretical_operator_time_us": [
        "theoretical_operator_time_us",
        "Theoretical Operator Time(us)",
        "theoretical_total_time_us",
    ],
    "theoretical_compute_time_us": [
        "theoretical_compute_time_us",
        "Theoretical Compute Time(us)",
    ],
    "theoretical_memory_time_us": [
        "theoretical_memory_time_us",
        "Theoretical Memory Time(us)",
    ],
    "fixed_overhead_us": [
        "fixed_overhead_us",
        "头尾开销",
        "Fixed Overhead(us)",
    ],
    "duration_over_theoretical": [
        "duration / theoretical",
        "duration_over_theoretical",
        "Gap Ratio",
    ],
}


def _normalized_theory(row):
    out = {}
    for key, candidates in THEORY_COLUMN_MAP.items():
        out[key] = _to_float(_first_present(row, candidates))
    out["bound_type"] = _first_present(row, ["bound_type", "Bound Type"])
    out["duration_analysis"] = _first_present(row, [
        "duration analysis", "duration_analysis",
    ])
    supported = _bool_from_text(_first_present(row, [
        "supported", "Supported", "support",
    ]))
    out["theory_supported"] = (
        supported if supported is not None
        else out.get("theoretical_operator_time_us") is not None
    )
    return out


def _explicit_org_index(row):
    value = _first_present(row, [
        "org_index",
        "Org Index",
        "source_org_index",
        "kernel_details_org_index",
        "csv_org_index",
    ])
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_operator_analysis_csv(path):
    rows_by_org = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("operator_analysis.csv 为空或缺 header")
        for row_no, row in enumerate(reader):
            org_idx = _explicit_org_index(row)
            if org_idx is None:
                org_idx = row_no
            rows_by_org[org_idx] = row
    return rows_by_org


def merge(raw_details_path, operator_analysis_csv, output_path):
    with open(raw_details_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows_by_org = load_operator_analysis_csv(operator_analysis_csv)

    matched = 0
    supported = 0
    missing = []
    for op in payload.get("operators", []):
        org_idx = op.get("org_index")
        row = rows_by_org.get(org_idx)
        if row is None:
            missing.append(org_idx)
            continue
        fields = _normalized_theory(row)
        op.update(fields)
        matched += 1
        if fields.get("theory_supported"):
            supported += 1

    payload["theoretical_perf_source"] = {
        "kind": "operator_analysis_csv",
        "path": os.path.abspath(operator_analysis_csv),
        "matched_operator_count": matched,
        "supported_operator_count": supported,
        "missing_operator_count": len(missing),
    }
    if missing:
        payload["theoretical_perf_source"]["missing_org_index_sample"] = missing[:20]

    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    if os.path.abspath(raw_details_path) == out_abs:
        fd, tmp = tempfile.mkstemp(
            prefix=".raw_ops_details.", suffix=".json",
            dir=os.path.dirname(out_abs) or ".",
        )
        os.close(fd)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, out_abs)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    else:
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload["theoretical_perf_source"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-ops-details", required=True,
                   help="raw_ops_details.json from Step 1")
    p.add_argument("--operator-analysis-csv", required=True,
                   help="operator_analysis.csv from operator-theoretical-perf")
    p.add_argument("-o", "--output", required=True,
                   help="output raw_ops_details.json; may equal --raw-ops-details")
    return p.parse_args()


def main():
    args = parse_args()
    source = merge(
        args.raw_ops_details,
        args.operator_analysis_csv,
        args.output,
    )
    logger.info(
        "merged theoretical columns: %s matched, %s supported, %s missing",
        source["matched_operator_count"],
        source["supported_operator_count"],
        source["missing_operator_count"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        main()
    except Exception as exc:  # 顶层入口兜底，转非零退出码
        logger.error("merge_theoretical_columns failed: %s", exc)
        sys.exit(1)
