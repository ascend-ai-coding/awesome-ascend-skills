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

"""Freshly compile and check the selected templates on dav-3510, logical device 0."""

import sys

from common import *


def script(name):
    subprocess.run([sys.executable, str(P / name)], cwd=P, check=True)


def device(name):
    print(run(name + "-run", [P / "build" / name], timeout=1200)[-1200:])


script("prepare.py")
script("query_platform.py")
script("baseline.py")
for name in ("chain", "sklansky"):
    script("build_" + name + ".py")
    device(name)
script("expand_softmax.py")
for variant in (
    "ar_small_r",
    "ar_full_load",
    "ar_recompute",
    "ara_full_load",
    "ara_recompute",
    "ara_online",
):
    device("softmax_" + variant)
script("build_topk.py")
script("make_topk_cases.py")
script("link_topk.py")
device("topk")
script("make_topk_tail_cases.py")
script("link_topk_tail.py")
device("topk-tail")
for name in ("gather", "gather_types", "conversion"):
    script("build_" + name + ".py")
    device(name.replace("_", "-"))
# Host tilers: valid resource combinations and rejection of infeasible plans.
for name, source, includes, objects in (
    ("softmax-host", "softmax_tiling_host_check.cpp", [P / "include/reduce"], []),
    (
        "conv-host",
        "conv3ddw_streamk_host_check.cpp",
        [P / "include/conv"],
        [P / "build/conv-tiling.o"],
    ),
):
    run(
        name + "-build",
        [
            "g++",
            "-std=c++17",
            "-O2",
            *INC,
            *["-I" + str(x) for x in includes],
            S / "tests" / source,
            *objects,
            *LIB,
            "-o",
            P / "build" / name,
        ],
    )
    print(run(name + "-run", [P / "build" / name]))
print("All selected probes passed; see manifest.json, results.jsonl and logs/.")
