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

root = P / "include/sort/arch35/dav3510/op_kernel/arch35"
run(
    "topk-tail-link",
    [
        "g++",
        '-DPROBE_CASE_HEADER="topk_tail_cases.h"',
        "-std=c++17",
        "-O2",
        *INC,
        f"-I{root}",
        f"-I{P}/src",
        P / "src/topk_check.cpp",
        P / "build/topk-main.o",
        P / "build/topk-fallback.o",
        P / "build/topk-transpose.o",
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
        P / "build/topk-tail",
    ],
)
