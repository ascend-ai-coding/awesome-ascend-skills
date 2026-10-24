/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host-only checks against the materialized five-file template.
#include "conv3ddw_stream_k_tiling.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

int main()
{
    Conv3ddwSkTilingIn in{};
    in.n = 2;
    in.cin = 3;
    in.cout = 5;
    in.di = 4;
    in.hi = 6;
    in.wi = 7;
    in.kd = 2;
    in.kh = 3;
    in.kw = 3;
    in.aicoreNum = 8;
    in.ubBytes = 248 * 1024;
    in.padUp = in.padDown = in.padLeft = in.padRight = 1;
    Conv3ddwSkTilingOut out{};
    in.streamkMode = ConvDwSkMode::BATCH_DOUT;
    assert(Conv3ddwSkDoTiling(in, out));
    assert(out.caseData.streamkType == 1 && out.caseData.streamkCores == 3);
    assert(out.caseData.workspaceBytes == 4194304);
    auto missingUb = in;
    missingUb.ubBytes = 0;
    assert(!Conv3ddwSkDoTiling(missingUb, out));
    auto bad = out.caseData;
    bad.workspaceBytes--;
    assert(!ValidateCase(bad));
    bad = out.caseData;
    bad.singleCoreK = 0;
    assert(!ValidateCase(bad));
    bad = out.caseData;
    bad.baseM = 512;
    assert(!ValidateCase(bad));
    bad = out.caseData;
    bad.streamkCores++;
    assert(!ValidateCase(bad));
    for (unsigned field = 0; field < 8; ++field) {
        auto b = in;
        switch (field) {
            case 0:
                b.groups = 2;
                break;
            case 1:
                b.strideH = 0;
                break;
            case 2:
                b.n = 0;
                break;
            case 3:
                b.aicoreNum = 65;
                break;
            case 4:
                b.wi = 0x7fffffff;
                break;
            case 5:
                b.hf32 = 1;
                break;
            case 6:
                b.kh = 256;
                break;
            case 7:
                b.c0Db = 3;
                break;
        }
        assert(!Conv3ddwSkDoTiling(b, out));
    }
    std::mt19937 rng(20260916);
    unsigned count = 0;
    unsigned modes[3] = {};
    for (unsigned trial = 0; trial < 2000; ++trial) {
        auto a = in;
        a.n = 1 + rng() % 4;
        a.cin = 1 + rng() % 40;
        a.cout = 1 + rng() % 350;
        a.di = 1 + rng() % 12;
        a.kd = 1 + rng() % a.di;
        a.hi = 3 + rng() % 12;
        a.wi = 3 + rng() % 12;
        a.kh = 1 + rng() % 3;
        a.kw = 1 + rng() % 3;
        a.aicoreNum = 1 + rng() % 16;
        a.c0Db = 1 + rng() % 2;
        a.dtype = ConvDwSkDType(rng() % 3);
        a.streamkMode = ConvDwSkMode(rng() % 4);
        assert(Conv3ddwSkDoTiling(a, out));
        const auto c = ToCaseView(a, out);
        assert(ValidateCase(c));
        ++modes[c.streamkType];
        // Independent coverage: one complete output tile, or disjoint BD/Ho intervals.
        const uint64_t axis = c.streamkType == 1 ? uint64_t(c.batch) * c.dout : c.ho;
        const uint64_t chunk = c.streamkType == 1 ? c.singleCoreBatchDout : c.singleCoreHo;
        std::vector<unsigned> tiles(c.totalTiles, 0);
        for (uint64_t round = 0; round < c.mainRounds; ++round)
            for (uint64_t core = 0; core < c.usedCoreNum; ++core) {
                auto tile = round * c.usedCoreNum + core;
                if (tile < c.totalTiles)
                    ++tiles[tile];
            }
        if (c.streamkType) {
            std::vector<unsigned> points(c.tailTiles * axis, 0);
            for (uint64_t core = 0; core < c.usedCoreNum; ++core) {
                auto tile = core / c.streamkCores;
                auto part = core % c.streamkCores;
                if (tile >= c.tailTiles)
                    continue;
                for (uint64_t i = part * chunk; i < std::min((part + 1) * chunk, axis); ++i)
                    ++points[tile * axis + i];
            }
            assert(std::all_of(points.begin(), points.end(), [](unsigned n) { return n == 1; }));
            for (uint64_t i = c.mainRounds * c.usedCoreNum; i < c.totalTiles; ++i)
                ++tiles[i];
        }
        assert(std::all_of(tiles.begin(), tiles.end(), [](unsigned n) { return n == 1; }));
        ++count;
    }
    assert(modes[0] && modes[1] && modes[2]);
    std::cout << "PASS " << count << " host cases: MN/BD/Ho coverage, capacity, validation and rejected inputs\n";
}
