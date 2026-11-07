// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "scalar_vector_ops.h"
#include <numeric>
#include <random>

extern "C" void topk64(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t);
extern "C" void split_even_odd64(GM_ADDR, GM_ADDR);

int main()
{
    std::mt19937 random(655);
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<float> values(64);
        std::vector<float> output(10, -9999);
        std::vector<uint32_t> indices(64);
        std::vector<uint32_t> outIndices(10, 9999);
        std::vector<uint32_t> order(64);
        std::iota(indices.begin(), indices.end(), 0);
        std::iota(order.begin(), order.end(), 0);
        for (int i = 0; i < 64; ++i)
            values[i] = mode == 0 ? i : mode == 1 ? -i : mode == 2 ? i % 5 : int(random() % 1000) - 500;
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return values[a] > values[b]; });
        topk64(reinterpret_cast<uint8_t*>(values.data()), reinterpret_cast<uint8_t*>(indices.data()),
               reinterpret_cast<uint8_t*>(output.data() + 1), reinterpret_cast<uint8_t*>(outIndices.data() + 1), 512,
               0); // CPU model buffer size, not a replacement for the host tiling API.
        for (int k = 0; k < 8; ++k) {
            assert(output[k + 1] == values[order[k]]);
            assert(outIndices[k + 1] == order[k]);
        }
        assert(output.front() == -9999 && output.back() == -9999);
        assert(outIndices.front() == 9999 && outIndices.back() == 9999);
    }
    for (uint32_t count : {64U, 128U, 192U, 255U * 64U}) {
        for (uint32_t base : {0U, 8U, 16U}) {
            std::vector<float> input(base + count);
            std::vector<float> result(count + 16, -9999);
            std::iota(input.begin(), input.end(), 0.0f);
            ScalarVectorExamples::SplitEvenOdd({result.data() + 8, count, 8}, {input.data(), base + count, 0}, base,
                                               count);
            for (uint32_t i = 0; i < count / 2; ++i) {
                assert(result[8 + i] == input[base + 2 * i]);
                assert(result[8 + count / 2 + i] == input[base + 2 * i + 1]);
            }
            for (uint32_t i = 0; i < 8; ++i)
                assert(result[i] == -9999 && result[count + 8 + i] == -9999);
        }
    }
    std::vector<float> input(64);
    std::vector<float> output(64);
    std::iota(input.begin(), input.end(), 0.0f);
    split_even_odd64(reinterpret_cast<uint8_t*>(input.data()), reinterpret_cast<uint8_t*>(output.data()));
    for (int i = 0; i < 32; ++i) {
        assert(output[i] == 2 * i);
        assert(output[32 + i] == 2 * i + 1);
    }
}
