#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# extract_perf.py —— MC2 多卡性能后处理
# 用法: python3 extract_perf.py <output_dir> <main_kernel_name> [last_n]

import csv
import glob
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: extract_perf.py <output_dir> <main_kernel_name> [last_n]")

    output_dir = sys.argv[1]
    main_kernel = sys.argv[2]
    last_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    prof_dirs = sorted(glob.glob(f"{output_dir}/PROF_*"))
    if not prof_dirs:
        raise FileNotFoundError(f"no PROF_* under {output_dir}")

    rank_avgs = []
    for rank_id, prof_dir in enumerate(prof_dirs):
        csv_path = glob.glob(f"{prof_dir}/mindstudio_profiler_output/op_summary_*.csv")
        if not csv_path:
            raise FileNotFoundError(f"no op_summary_*.csv under {prof_dir}")
        with open(csv_path[0], newline="") as f:
            rows = list(csv.DictReader(f))
        main_rows = [r for r in rows if r.get("Op Name", "") == main_kernel]
        if len(main_rows) < last_n:
            raise ValueError(f"rank {rank_id}: only {len(main_rows)} main kernel records")
        last_n_rows = main_rows[-last_n:]
        durations = [float(r["Task Duration (us)"]) for r in last_n_rows]
        avg = sum(durations) / last_n
        rank_avgs.append(avg)
        logger.info(f"Rank {rank_id}: last {last_n} = {durations} us, avg = {avg:.2f} us")

    overall = max(rank_avgs)
    logger.info(f"\nOverall (max of {len(rank_avgs)} ranks): {overall:.2f} us")


if __name__ == "__main__":
    main()
