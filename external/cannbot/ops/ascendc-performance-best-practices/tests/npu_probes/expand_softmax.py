# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import concurrent.futures

from common import *

root = P / "include/reduce"
variants = [
    (
        "ar_small_r",
        "SoftmaxV2ArSmallR",
        "SoftmaxV2ArSmallRTilingData",
        "TilingArSmallR",
    ),
    ("ar_full_load", "SoftmaxV2AR", "SoftmaxV2ARTilingData", "TilingArFullLoad"),
    (
        "ar_recompute",
        "SoftmaxV2ArRecompute",
        "SoftmaxV2ArRecomputeTilingData",
        "TilingArRecompute",
    ),
    ("ara_full_load", "SoftmaxV2ARA", "SoftmaxV2ARATilingData", "TilingAraFullLoad"),
    (
        "ara_recompute",
        "SoftmaxV2ARARecompute",
        "SoftmaxV2ARARecomputeTilingData",
        "TilingAraRecompute",
    ),
    (
        "ara_online",
        "SoftmaxV2ARAOnline",
        "SoftmaxV2ARAOnlineTilingData",
        "TilingAraOnline",
    ),
]
BODY = r"""
#include "kernel_operator.h"
#include "softmax_v2_tiling_data.h"
#ifdef __NPU_DEVICE__
#include "softmax_v2_VARIANT.h"
#endif
using namespace AscendC;
template<class T>__global__ __aicore__ __vector__ void SoftmaxProbe(GM_ADDR x,GM_ADDR y,DATA data){
#ifdef __NPU_DEVICE__
TPipe pipe;INIT op.Process();
#endif
}
#ifdef __NPU_DEVICE__
template void SoftmaxProbe<float>(GM_ADDR,GM_ADDR,DATA);
template void SoftmaxProbe<half>(GM_ADDR,GM_ADDR,DATA);
template void SoftmaxProbe<bfloat16_t>(GM_ADDR,GM_ADDR,DATA);
#endif
#ifndef __NPU_DEVICE__
#include "softmax_v2_tiling.h"
#include <acl/acl.h>
#include <platform/platform_ascendc.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "host_types.h"
#define ACL(x) do{auto e=(x);if(e){std::cerr<<#x<<":"<<e<<"\n";return 1;}}while(0)
int main() {
    ACL(aclInit(nullptr));
    ACL(aclrtSetDevice(0));
    aclrtStream stream;
    ACL(aclrtCreateStream(&stream));
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return 1;
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    softmax_tiling::PlatformParam p{};
    p.ubSize = ub;
    p.numBlocks = platform->GetCoreNumAiv();
    int cases = 0, rejected = 0;
    double maxerr[3] = {};
    for (int dtype = 0; dtype < 3; dtype++)
    for (int64_t r : RSIZES)
    for (int64_t a0 : A0SIZES)
    for (int64_t a1 : A1SIZES)
    for (int pattern = 0; pattern < 3; pattern++) {
        softmax_tiling::CaseShape s{a1, r, a0, dtype};
        DATA data{};
        auto blocks = softmax_tiling::TILER(s, p, data);
        if (blocks <= 0) {
            uint64_t minBytes = (((dtype == 0 ? 4 : 2) * 2 + 4 * 2 + 4) * r + 8) * (dtype == 0 ? p.vlFp32 : p.vlFp16);
            if (ALLOW_RESOURCE_REJECTION && minBytes > uint64_t(p.ubSize)) {
                rejected++;
                continue;
            }
            std::cerr << "tiling rejected " << dtype << "," << a1 << "," << r << "," << a0 << "\n";
            return 2;
        }
        size_t count = a1 * r * a0, bytes = count * (dtype ? 2 : 4);
        std::vector<float> x(count);
        std::vector<uint8_t> in(bytes), out(bytes + 64);
        for (size_t i = 0; i < count; i++) {
            float f = pattern == 0 ? (int((i * 17) % 257) - 128) / 8.f
                                   : pattern == 1 ? 3.5f : (int((i * 31) % 641) - 320) / 4.f;
            if (dtype) {
                uint16_t u = encode(f, dtype == 2);
                memcpy(in.data() + i * 2, &u, 2);
                x[i] = decode(u, dtype == 2);
            } else {
                memcpy(in.data() + i * 4, &f, 4);
                x[i] = f;
            }
        }
        void *dx{}, *dy{};
        ACL(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMalloc(&dy, out.size(), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMemcpy(dx, bytes, in.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL(aclrtMemset(dy, out.size(), 0xa5, out.size()));
        if (dtype == 0)
            SoftmaxProbe<float><<<blocks, nullptr, stream>>>((uint8_t*)dx, (uint8_t*)dy, data);
        else if (dtype == 1)
            SoftmaxProbe<half><<<blocks, nullptr, stream>>>((uint8_t*)dx, (uint8_t*)dy, data);
        else
            SoftmaxProbe<bfloat16_t><<<blocks, nullptr, stream>>>((uint8_t*)dx, (uint8_t*)dy, data);
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 20000));
        ACL(aclrtMemcpy(out.data(), out.size(), dy, out.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t row = 0; row < a1; row++)
        for (int64_t col = 0; col < a0; col++) {
            double maxv = -INFINITY;
            for (int64_t rr = 0; rr < r; rr++)
                maxv = std::max(maxv, double(x[(row * r + rr) * a0 + col]));
            double sum = 0;
            for (int64_t rr = 0; rr < r; rr++)
                sum += std::exp(double(x[(row * r + rr) * a0 + col]) - maxv);
            double actualsum = 0;
            for (int64_t rr = 0; rr < r; rr++) {
                size_t i = (row * r + rr) * a0 + col;
                double expected = std::exp(double(x[i]) - maxv) / sum;
                float got;
                if (dtype) {
                    uint16_t u;
                    memcpy(&u, out.data() + i * 2, 2);
                    got = decode(u, dtype == 2);
                    expected = decode(encode(float(expected), dtype == 2), dtype == 2);
                } else {
                    memcpy(&got, out.data() + i * 4, 4);
                }
                double err = std::abs(double(got) - expected);
                maxerr[dtype] = std::max(maxerr[dtype], err);
                double tolerance = 1e-7 + (dtype == 0 ? 3e-4 : dtype == 1 ? 0.002 : 0.016) * std::abs(expected);
                if (!std::isfinite(got) || err > tolerance) {
                    std::cerr << "FAIL VARIANT dtype=" << dtype << " shape=" << a1 << "," << r << "," << a0
                              << " pattern=" << pattern << " at=" << i << " got=" << got
                              << " expected=" << expected << " tol=" << tolerance << "\n";
                    return 3;
                }
                actualsum += got;
            }
            if (std::abs(actualsum - 1) > 0.02) {
                std::cerr << "normalization failed " << actualsum << "\n";
                return 3;
            }
        }
        for (size_t i = bytes; i < out.size(); i++)
            if (out[i] != 0xa5) {
                std::cerr << "output overrun " << i << "\n";
                return 4;
            }
        ACL(aclrtFree(dx));
        ACL(aclrtFree(dy));
        cases++;
    }
    ACL(aclrtDestroyStream(stream));
    ACL(aclrtResetDevice(0));
    ACL(aclFinalize());
    std::cout << "PASS VARIANT " << cases << " FP32/FP16/BF16 cases, rejected=" << rejected
              << " resource-infeasible configurations; stable CPU reference and output canaries; max abs error="
              << maxerr[0] << "," << maxerr[1] << "," << maxerr[2] << "\n";
}
#endif
"""
for variant, cls, data, tiler in variants:
    init = (
        f"SoftmaxV2Ops::{cls}<T,T> op(&data);op.Init(x,y,&pipe);"
        if variant.startswith("ara")
        else f"SoftmaxV2Ops::{cls}<T,T> op(&pipe);op.Init(x,y,&data);"
    )
    r = (
        "{int64_t(1),int64_t(5),int64_t(16)}"
        if variant == "ar_small_r"
        else "{int64_t(3),int64_t(33),int64_t(129)}"
        if variant == "ara_full_load"
        else "{int64_t(33),int64_t(129),int64_t(1025)}"
        if variant == "ar_full_load"
        else "{int64_t(33),int64_t(1025),int64_t(65537)}"
        if variant == "ar_recompute"
        else "{int64_t(33),int64_t(1025),int64_t(16385)}"
    )
    a0 = (
        "{int64_t(3),int64_t(17),int64_t(65)}"
        if variant == "ara_full_load"
        else "{int64_t(3),int64_t(17)}"
        if variant.startswith("ara")
        else "{int64_t(1)}"
    )
    a1 = (
        "{int64_t(1),int64_t(3),int64_t(67)}"
        if variant == "ar_small_r"
        else "{int64_t(1),int64_t(3)}"
        if variant.startswith("ar_")
        else "{int64_t(3)}"
    )
    src = (
        BODY.replace("VARIANT", variant)
        .replace("DATA", data)
        .replace("INIT op.Process();", init + " op.Process();")
        .replace("TILER", tiler)
        .replace("RSIZES", r)
        .replace("A0SIZES", a0)
        .replace("A1SIZES", a1)
        .replace(
            "ALLOW_RESOURCE_REJECTION",
            "true" if variant == "ara_full_load" else "false",
        )
    )
    (P / "src" / f"softmax_{variant}.asc").write_text(src)


def compile_variant(v):
    name = "softmax_" + v[0]
    return build(
        name,
        P / "src" / f"{name}.asc",
        [f"-I{root}", f"-I{root}/dav310", f"-I{P}/src", *LIB, "-lm"],
    )


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
    for result in e.map(compile_variant, variants):
        pass
