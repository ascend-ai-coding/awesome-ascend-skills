/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Materialize templates/softmax_v2_tiling{,_data}.template as .h before compiling.
#include "softmax_v2_tiling.h"
#include <iostream>

template <typename Data, typename Tiler>
bool Check(Tiler tiler, bool smallR, bool ar)
{
    for (int dtype = 0; dtype < 3; ++dtype) {
        softmax_tiling::CaseShape shape{3, smallR ? 8 : 9, ar ? 1 : 17, dtype};
        for (int64_t ub : {32 * 1024, 64 * 1024, 248 * 1024}) {
            for (int64_t cores : {1, 3, 17}) {
                softmax_tiling::PlatformParam platform{};
                platform.ubSize = ub;
                platform.numBlocks = cores;
                Data data{};
                const auto blocks = tiler(shape, platform, data);
                if (blocks < 1 || blocks > cores)
                    return false;
                platform.ubSize = 1; // Too small to hold any supported tile.
                if (tiler(shape, platform, data) != 0)
                    return false;
                platform.ubSize = ub;
                platform.numBlocks = 0;
                if (tiler(shape, platform, data) != 0)
                    return false;
                platform.numBlocks = cores;
                platform.ubSize = 0;
                if (tiler(shape, platform, data) != 0)
                    return false;
            }
        }
    }
    return true;
}

int main()
{
    using namespace softmax_tiling;
    const bool passed = Check<SoftmaxV2ArSmallRTilingData>(TilingArSmallR, true, true) &&
                        Check<SoftmaxV2ARTilingData>(TilingArFullLoad, false, true) &&
                        Check<SoftmaxV2ArRecomputeTilingData>(TilingArRecompute, false, true) &&
                        Check<SoftmaxV2ARATilingData>(TilingAraFullLoad, false, false) &&
                        Check<SoftmaxV2ARARecomputeTilingData>(TilingAraRecompute, false, false) &&
                        Check<SoftmaxV2ARAOnlineTilingData>(TilingAraOnline, false, false);
    // ARA full-load cannot fit even one vector column for this reduction size.
    PlatformParam platform{};
    platform.ubSize = 248 * 1024;
    platform.numBlocks = 17;
    SoftmaxV2ARATilingData data{};
    if (!passed || TilingAraFullLoad({3, 1025, 17, 0}, platform, data) != 0) {
        std::cerr << "FAIL Softmax tiling resource guards\n";
        return 1;
    }
    std::cout << "PASS 6 tilers: 162 valid cases, 486 resource rejection cases, oversized full-load\n";
}
