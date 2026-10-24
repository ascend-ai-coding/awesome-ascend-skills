// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// ----------------------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

std::tuple<int64_t, int64_t, uint32_t> CalcCompiledWarmupTiling();

extern "C" void LaunchCompiledWarmupHalf(
    GM_ADDR x,
    GM_ADDR out,
    int64_t totalLength,
    int64_t blockLength,
    uint32_t tileBytes,
    uint32_t blockDim,
    void *stream);
