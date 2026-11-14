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
for name, src in [
    ("topk-main", root / "top_k_kernel.cpp"),
    ("topk-fallback", root / "fallback/top_k_fallback.cpp"),
    ("topk-transpose", root / "transpose/top_k_transpose_kernel.cpp"),
]:
    run(
        name + "-build",
        [
            "bisheng",
            "-xasc",
            "--npu-arch=dav-3510",
            "-std=c++17",
            "-O2",
            "-fPIC",
            *(
                [
                    "-DTopKLaunchParams=TopKFallbackParams",
                    "-Dcalc_topk_params=calc_topk_fallback_params",
                    "-Dlaunch_topk=launch_topk_fallback",
                ]
                if name == "topk-fallback"
                else []
            ),
            *INC,
            f"-I{SDK}/asc",
            f"-I{root}",
            f"-I{root}/fallback",
            "-c",
            src,
            "-o",
            P / "build" / f"{name}.o",
        ],
        timeout=900,
    )
