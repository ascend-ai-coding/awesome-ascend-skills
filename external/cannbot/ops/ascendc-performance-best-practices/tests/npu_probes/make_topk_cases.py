# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import sys

from common import *

spec = importlib.util.spec_from_file_location(
    "topk_tiling_model",
    R / "ops/ascendc-perf-optimize/references/tiling/sort/script/topk_tiling.py",
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
p = json.loads((P / "platform.json").read_text())
platform = m.PlatformSpec(p["aiv"], p["ub"], system_workspace_bytes=p["system_workspace"])
dtype_ids = {
    "int8": 0,
    "uint8": 1,
    "int32": 2,
    "int64": 3,
    "fp16": 4,
    "fp32": 5,
    "bf16": 6,
}
requests = []
for dtype in dtype_ids:
    for d in [33, 257, 1024, 4097]:
        requests.append((dtype, 3, d, 1, min(7, d)))
for dtype in ["fp16", "bf16"]:
    for d in [7, 17]:
        requests.append((dtype, 2, d, 35, d))
for dtype in ["fp32", "int32", "fp16", "bf16"]:
    requests.extend([(dtype, 2, 33, 17, 7), (dtype, 1, 4097, 9, 3)])
tail_mode = "--int64-tail" in sys.argv
if tail_mode:
    requests = [
        ("int64", 3, d, 1, k)
        for d in [1, 2, 3, 4, 5, 7, 15, 31, 32, 33, 34, 35, 36, 129, 4096, 4097, 8001]
        for k in sorted({1, min(7, d)})
    ]
records = []
code = [
    '#pragma once\n#include "top_k_launch.h"\n'
    "struct ProbeCase { int dtype; int64_t outer,d,inner,k; bool largest; TopKLaunchParams p; };\n"
    "inline std::vector<ProbeCase> cases(){std::vector<ProbeCase> out;"
]
for dtype, outer, d, n, k in requests:
    for largest in [True, False]:
        shape = (outer, d) if n == 1 else (outer, d, n)
        query_size = 4 if dtype == "bf16" else m.DTYPE_BYTES[dtype]
        merge = p["concat"].get(f"{m.align_up(d, 32)}-{query_size}")
        request = m.TopKRequest(shape, k, 1, dtype, largest)
        plan = m.compute_topk_tiling(request, platform, m.SdkOverrides(merge_tmp_bytes=merge))
        rec = {
            "dtype": dtype,
            "outer": outer,
            "d": d,
            "inner": n,
            "k": k,
            "largest": largest,
            **plan.to_dict(),
        }
        records.append(rec)
        special = plan.tiling_key in [900001, 900002]
        if plan.sdk_queries and not special:
            raise RuntimeError(f"Unresolved SDK query: {dtype}, {d}, {n}, {plan.sdk_queries}")
        if plan.tiling_key == 900003:
            raise RuntimeError("Transpose fallback requires a nested plan")
        code.append(
            "{ ProbeCase c{};"
            + f"c.dtype={dtype_ids[dtype]};c.outer={outer};c.d={d};c.inner={n};c.k={k};"
            + f"c.largest={str(largest).lower()};c.p.tilingKey={plan.tiling_key};"
            + f"c.p.usedCores={plan.block_dim};c.p.localMemoryBytes={plan.local_memory_bytes};"
            + f"c.p.workspaceBytes={plan.workspace_bytes};"
        )
        for key, value in plan.tiling_data.items():
            code.append(f"c.p.tiling.{key}={value};")
        code.append("out.push_back(c);}")
code.append("return out;}")
(P / "src" / ("topk_tail_cases.h" if tail_mode else "topk_cases.h")).write_text("\n".join(code) + "\n")
(P / ("topk-tail-plans.json" if tail_mode else "topk-plans.json")).write_text(json.dumps(records, indent=2) + "\n")
print("Generated", len(records), "TopK plan records")
