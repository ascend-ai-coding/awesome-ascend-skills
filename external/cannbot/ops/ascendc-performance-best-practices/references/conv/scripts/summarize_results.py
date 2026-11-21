#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Merge TTK kernel results with case shapes/attrs into a summary table."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def _parse_literal(value: str):
    if value is None or value == "":
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return json.loads(value)
        except Exception:
            return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.cases, newline="", encoding="utf-8") as f:
        cases = {row["testcase_name"]: row for row in csv.DictReader(f)}

    with open(args.results, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    rows = []
    for r in results:
        name = r.get("testcase_name") or r.get("case_name") or ""
        case = cases.get(name, {})
        tiling_key = r.get("dyn_tiling_key") or r.get("bin_tiling_key") or ""
        tiling_data = r.get("dyn_tiling_data") or r.get("bin_tiling_data") or ""
        perf = r.get("dyn_perf_us") or r.get("bin_perf_us") or r.get("cst_perf_us") or ""
        block_dim = r.get("dyn_block_dim") or r.get("bin_block_dim") or r.get("cst_block_dim") or ""
        shape = case.get("input_ori_shapes") or case.get("input_shapes") or ""
        out_shape = case.get("output_ori_shapes") or case.get("output_shapes") or ""
        attrs = case.get("attributes") or ""
        dtypes = case.get("input_dtypes") or ""
        rows.append(
            {
                "case_id": name.replace("conv2dv2_case_", "") if name.startswith("conv2dv2_case_") else name,
                "testcase_name": name,
                "shape": shape,
                "output_shape": out_shape,
                "dtype": dtypes,
                "attr": attrs,
                "tiling_key": tiling_key,
                "tiling_data": tiling_data,
                "block_dim": block_dim,
                "perf_us": perf,
                "precision_status": r.get("precision_status", ""),
                "perf_status": r.get("perf_status", ""),
                "kernel_name": r.get("dyn_kernel_name") or r.get("bin_kernel_name") or "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "testcase_name",
        "shape",
        "output_shape",
        "dtype",
        "attr",
        "tiling_key",
        "tiling_data",
        "block_dim",
        "perf_us",
        "precision_status",
        "perf_status",
        "kernel_name",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| case_id | shape | attr | tiling_key | tiling_data | perf_us |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in rows:
            tiling_data = str(row["tiling_data"]).replace("|", "\\|").replace("\n", " ")
            if len(tiling_data) > 120:
                tiling_data = tiling_data[:117] + "..."
            f.write(
                f"| {row['case_id']} | `{row['shape']}` | `{row['attr']}` | `{row['tiling_key']}` | "
                f"`{tiling_data}` | {row['perf_us']} |\n"
            )

    print(f"Wrote {out} ({len(rows)} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
