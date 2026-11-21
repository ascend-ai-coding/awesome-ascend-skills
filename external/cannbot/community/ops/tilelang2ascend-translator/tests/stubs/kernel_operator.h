// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// CPU contract model ONLY. Does not emulate NPU instructions or pipelines.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define __global__
#define __gm__
#define GM_ADDR uint8_t*
constexpr int PIPE_V = 0;
namespace AscendC {
enum class HardEvent {
    MTE2_V,
    V_MTE3,
    MTE3_S
};
enum class TPosition {
    VECCALC
};
template <typename T>
struct LocalTensor {
    T* data = nullptr;
    uint32_t size = 0;
    uint32_t offset = 0;
    LocalTensor operator[](uint32_t index) const
    {
        assert(index <= size);
        return {data + index, size - index, offset + index};
    }
    void Require(uint32_t count) const
    {
        assert(data && count <= size);
    }
};
template <typename T>
struct GlobalTensor : LocalTensor<T> {
    void SetGlobalBuffer(T* data, uint32_t count)
    {
        this->data = data;
        this->size = count;
    }
};
template <TPosition>
struct TBuf {
    std::vector<uint32_t> storage;
    template <typename T>
    LocalTensor<T> Get()
    {
        return {reinterpret_cast<T*>(storage.data()), static_cast<uint32_t>(storage.size() * 4 / sizeof(T)), 0};
    }
};
struct TPipe {
    template <TPosition P>
    void InitBuffer(TBuf<P>& buffer, uint32_t bytes)
    {
        assert(bytes > 0 && bytes % 32 == 0);
        buffer.storage.resize(bytes / 4);
    }
    int32_t FetchEventID(HardEvent)
    {
        return 0;
    }
};
template <HardEvent>
void SetFlag(int32_t)
{}
template <HardEvent>
void WaitFlag(int32_t)
{}
template <int>
void PipeBarrier()
{}
template <typename T>
void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
{
    dst.Require(count);
    src.Require(count);
    assert(count * sizeof(T) % 32 == 0);
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void Concat(LocalTensor<T>& concat, const LocalTensor<T>& src, const LocalTensor<T>&, int32_t repeats)
{
    src.Require(repeats * 16);
    concat = src; // Official A2 implementation aliases the source.
}
template <typename T, bool Full>
void Sort(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& indices,
          LocalTensor<T>& tmp, int32_t repeats)
{
    static_assert(sizeof(T) == 4);
    assert(repeats > 0 && repeats <= 255);
    const uint32_t count = repeats * 32;
    dst.Require(2 * count);
    src.Require(count);
    indices.Require(count);
    tmp.Require(2 * count);
    assert(dst.data != src.data && dst.data != tmp.data);
    std::vector<uint32_t> order(count);
    for (uint32_t i = 0; i < count; ++i)
        order[i] = i;
    const uint32_t batch = Full ? count : 32;
    for (uint32_t start = 0; start < count; start += batch)
        std::stable_sort(order.begin() + start, order.begin() + start + batch,
                         [&](auto a, auto b) { return src.data[a] > src.data[b]; });
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[2 * i] = src.data[order[i]];
        std::memcpy(dst.data + 2 * i + 1, indices.data + order[i], sizeof(uint32_t));
    }
}
template <typename T>
void Extract(const LocalTensor<T>& values, const LocalTensor<uint32_t>& indices, const LocalTensor<T>& sorted,
             int32_t repeats)
{
    const uint32_t count = repeats * 32;
    values.Require(count);
    indices.Require(count);
    sorted.Require(2 * count);
    for (uint32_t i = 0; i < count; ++i) {
        values.data[i] = sorted.data[2 * i];
        std::memcpy(indices.data + i, sorted.data + 2 * i + 1, sizeof(uint32_t));
    }
}
struct GatherMaskParams {
    uint16_t src0BlockStride;
    uint16_t repeatTimes;
    uint16_t src0RepeatStride;
    uint16_t src1RepeatStride;
};
template <typename T>
void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint8_t pattern, bool reduceMode, uint32_t mask,
                const GatherMaskParams& params, uint64_t& retained)
{
    assert(!reduceMode && mask == 0 && (pattern == 1 || pattern == 2));
    assert(params.src0BlockStride == 1 && params.src0RepeatStride == 8);
    assert(params.repeatTimes > 0 && params.repeatTimes <= 255);
    assert(src.offset * sizeof(T) % 32 == 0 && dst.offset * sizeof(T) % 32 == 0);
    const uint32_t count = params.repeatTimes * 256 / sizeof(T);
    src.Require(count);
    dst.Require(count / 2);
    retained = 0;
    for (uint32_t i = pattern - 1; i < count; i += 2)
        dst.data[retained++] = src.data[i];
}
} // namespace AscendC
