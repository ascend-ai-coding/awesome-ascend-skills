# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from common import *

run(
    "softmax-scan-fresh",
    ["python3", S / "tests/run_template_npu_checks.py", "--work-dir", P / "baseline"],
    timeout=900,
)
print(build("paired", S / "tests/paired_segment_rotate_npu_check.asc"))
print(run("paired-run", [P / "build/paired"]))
c = P / "include/conv"
d = c / "dav310"
run(
    "conv-kernel-build",
    [
        "bisheng",
        "-xasc",
        "--npu-arch=dav-3510",
        "-std=c++17",
        "-O2",
        "-fPIC",
        *INC,
        f"-I{c}",
        f"-I{d}",
        "-c",
        d / "conv3ddw_stream_k_kernel_entry.h",
        "-o",
        P / "build/conv-kernel.o",
    ],
    timeout=600,
)
run(
    "conv-tiling-build",
    [
        "g++",
        "-x",
        "c++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        *INC,
        f"-I{c}",
        "-c",
        c / "conv3ddw_stream_k_tiling_impl.h",
        "-o",
        P / "build/conv-tiling.o",
    ],
)
run(
    "conv-link",
    [
        "g++",
        "-std=c++17",
        "-O2",
        *INC,
        f"-I{c}",
        f"-I{d}",
        S / "tests/conv3ddw_streamk_npu_check.cpp",
        P / "build/conv-kernel.o",
        P / "build/conv-tiling.o",
        *LIB,
        f"-L{DRV}",
        "-lascendc_runtime",
        "-lascendcl",
        "-lruntime",
        "-lregister",
        "-lerror_manager",
        "-lprofapi",
        "-lge_common_base",
        "-lgraph_base",
        "-lunified_dlog",
        "-lmmpa",
        "-ldl",
        "-lascend_dump",
        "-lc_sec",
        "-o",
        P / "build/conv",
    ],
)
print(run("conv-run", [P / "build/conv"], timeout=600)[-300:])
