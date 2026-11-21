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

"""extract_perf.py —— MC2 多卡性能后处理。

用法: python3 extract_perf.py <output_dir> <main_kernel_name> [last_n]
"""
import csv
import glob
import os
import sys

output_dir = sys.argv[1]                # e.g. docs/perf/round_001
main_kernel = sys.argv[2]               # e.g. AllToAllQuantMatmulKernelE4M3E4M3
LAST_N = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# 找所有 PROF_xxx 子目录（每卡一个，由 fork 子进程分别产生）
prof_dirs = sorted(glob.glob(f"{output_dir}/PROF_*"))
assert prof_dirs, f"no PROF_* under {output_dir}"

rank_avgs = []
for rank_id, prof_dir in enumerate(prof_dirs):
    csv_path = glob.glob(f"{prof_dir}/mindstudio_profiler_output/op_summary_*.csv")
    assert csv_path, f"no op_summary_*.csv under {prof_dir}"
    rows = list(csv.DictReader(open(csv_path[0])))
    # 过滤主 kernel 记录，保持采集顺序
    main_rows = [r for r in rows if r.get("Op Name", "") == main_kernel]
    assert len(main_rows) >= LAST_N, f"rank {rank_id}: only {len(main_rows)} main kernel records"
    last_n_rows = main_rows[-LAST_N:]
    durations = [float(r["Task Duration (us)"]) for r in last_n_rows]
    avg = sum(durations) / LAST_N
    rank_avgs.append(avg)
    print(f"Rank {rank_id}: last {LAST_N} = {durations} us, avg = {avg:.2f} us")

overall = max(rank_avgs)
print(f"\nOverall (max of {len(rank_avgs)} ranks): {overall:.2f} us")
