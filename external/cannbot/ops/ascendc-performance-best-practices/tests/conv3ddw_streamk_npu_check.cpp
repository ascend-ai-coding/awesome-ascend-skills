/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Optional device validation for the five-file template; not required by the operator.
#include "conv3ddw_stream_k_tiling.h"
#include "conv3ddw_stream_k_launch.h"
#include <acl/acl.h>
#include <platform/platform_ascendc.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#define ACL(x) \
    do { \
        auto e = (x); \
        if (e) { \
            std::cerr << #x << " failed " << e << "\n"; \
            return 1; \
        } \
    } while (0)
int RunCase(const Conv3ddwSkTilingIn& in, aclrtStream stream)
{
    Conv3ddwSkTilingOut out{};
    if (!Conv3ddwSkDoTiling(in, out)) {
        std::cerr << "tiling rejected\n";
        return 1;
    }
    auto c = ToCaseView(in, out);
    std::cerr << "dtype=" << int(c.dtype) << " db=" << c.cl0Pbuffer << " mode=" << c.streamkType
              << " R=" << c.streamkCores << " tiles=" << c.totalTiles << " base=" << c.baseM << "," << c.baseN << ","
              << c.baseK << " steps=" << c.stepKa << "," << c.stepKb << "\n";
    size_t nx = size_t(c.batch) * c.cin * c.di * c.hi * c.wi;
    size_t ny = size_t(c.batch) * c.cout * c.dout * c.ho * c.wo;
    size_t nw = size_t(c.cout) * c.cin * c.dk * c.hk * c.wk;
    std::vector<float> x(nx);
    std::vector<float> y(ny);
    std::vector<float> z(nw);
    std::vector<float> first(nw);
    std::vector<double> ref(nw, 0);
    uint32_t seed = 12345;
    auto value = [&]() {
        seed = 1664525U * seed + 1013904223U;
        return std::ldexp(float(int(seed % 31) - 15), -int((seed >> 16) % 11));
    };
    // Exactly representable inputs, varied exponents; accumulation is not generally exact.
    for (auto& v : x) {
        v = value();
    }
    for (auto& v : y) {
        v = value();
    }
    for (unsigned co = 0; co < c.cout; ++co)
        for (unsigned ci = 0; ci < c.cin; ++ci)
            for (unsigned kd = 0; kd < c.dk; ++kd)
                for (unsigned kh = 0; kh < c.hk; ++kh)
                    for (unsigned kw = 0; kw < c.wk; ++kw) {
                        size_t o = ((((size_t(co) * c.cin + ci) * c.dk + kd) * c.hk + kh) * c.wk + kw);
                        for (unsigned b = 0; b < c.batch; ++b)
                            for (unsigned od = 0; od < c.dout; ++od)
                                for (unsigned oh = 0; oh < c.ho; ++oh)
                                    for (unsigned ow = 0; ow < c.wo; ++ow) {
                                        int id = od * c.strideD + kd * c.dilationD - c.padFront;
                                        int ih = oh * c.strideH + kh * c.dilationH - c.padUp;
                                        int iw = ow * c.strideW + kw * c.dilationW - c.padLeft;
                                        if (id < 0 || ih < 0 || iw < 0 || id >= int(c.di) || ih >= int(c.hi) ||
                                            iw >= int(c.wi))
                                            continue;
                                        ref[o] +=
                                            double(
                                                x[((((size_t(b) * c.cin + ci) * c.di + id) * c.hi + ih) * c.wi + iw)]) *
                                            y[((((size_t(b) * c.cout + co) * c.dout + od) * c.ho + oh) * c.wo + ow)];
                                    }
                    }
    std::vector<uint16_t> xh(nx);
    std::vector<uint16_t> yh(ny);
    auto encode = [&](float v) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        if (c.dtype == ConvDwSkDType::BF16)
            return uint16_t(bits >> 16);
        if (v == 0)
            return uint16_t(0);
        return uint16_t(((bits >> 16) & 0x8000) | ((((bits >> 23) & 255) - 112) << 10) | ((bits >> 13) & 1023));
    };
    std::transform(x.begin(), x.end(), xh.begin(), encode);
    std::transform(y.begin(), y.end(), yh.begin(), encode);
    size_t elem = DwSkBytes(c.dtype);
    void* hx = elem == 4 ? static_cast<void*>(x.data()) : xh.data();
    void* hy = elem == 4 ? static_cast<void*>(y.data()) : yh.data();

    void *dx{};
    void *dy{};
    void *dw{};
    void *ws{};
    ACL(aclrtMalloc(&dx, nx * elem, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&dy, ny * elem, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&dw, nw * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    if (c.workspaceBytes)
        ACL(aclrtMalloc(&ws, c.workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMemcpy(dx, nx * elem, hx, nx * elem, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(dy, ny * elem, hy, ny * elem, ACL_MEMCPY_HOST_TO_DEVICE));
    for (int run = 0; run < 3; ++run) {
        ACL(aclrtMemset(dw, nw * 4, 0xA5, nw * 4));
        if (ws)
            ACL(aclrtMemset(ws, c.workspaceBytes, 0x3F, c.workspaceBytes));
        ACL(launch_conv3ddw_sk(dx, dy, dw, ws, c, stream));
        ACL(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        ACL(aclrtMemcpy(z.data(), nw * 4, dw, nw * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        double err = 0;
        size_t bad = 0;
        for (size_t i = 0; i < nw; ++i) {
            if (!std::isfinite(z[i]) || std::abs(z[i] - ref[i]) > 1e-3 + 2e-4 * std::abs(ref[i])) {
                if (bad < 8)
                    std::cerr << i << ": " << z[i] << " vs " << ref[i] << "\n";
                ++bad;
            }
            err = std::max(err, std::abs(z[i] - ref[i]));
        }
        if (bad) {
            std::cerr << "FAIL " << bad << " / " << nw << " maxerr=" << err << "\n";
            return 2;
        }
        if (run && memcmp(z.data(), first.data(), nw * 4)) {
            std::cerr << "nondeterministic\n";
            return 3;
        }
        first = z;
    }
    std::cout << "PASS numerical CPU reference and 3 bitwise equal runs\n";
    ACL(aclrtFree(dx));
    ACL(aclrtFree(dy));
    ACL(aclrtFree(dw));
    if (ws)
        ACL(aclrtFree(ws));
    return 0;
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    ACL(aclInit(nullptr));
    ACL(aclrtSetDevice(device));
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950") != 0) {
        std::cerr << "Requires Ascend950, got " << (soc ? soc : "null") << "\n";
        return 1;
    }
    aclrtStream stream{};
    ACL(aclrtCreateStream(&stream));
    Conv3ddwSkTilingIn base{};
    base.n = 2;
    base.cin = 3;
    base.cout = 5;
    base.di = 4;
    base.hi = 6;
    base.wi = 7;
    base.kd = 2;
    base.kh = 3;
    base.kw = 3;
    base.padUp = base.padDown = base.padLeft = base.padRight = 1;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) {
        std::cerr << "platform query failed\n";
        return 1;
    }
    base.aicoreNum = std::min<uint32_t>(8, platform->GetCoreNumAic());
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, base.ubBytes);
    if (!base.aicoreNum || !base.ubBytes) {
        std::cerr << "invalid target resources\n";
        return 1;
    }
    std::vector<Conv3ddwSkTilingIn> shapes;
    shapes.push_back(base); // pure StreamK, R=3, idle cores
    auto c = base;
    c.di = 12;
    c.kd = 10;
    shapes.push_back(c); // MN full round + tail
    c = base;
    c.di = 10;
    c.kd = 8;
    shapes.push_back(c); // exact full round
    c = base;
    c.di = 7;
    c.kd = 5;
    shapes.push_back(c); // > half-core tail, no StreamK
    c = base;
    c.cin = 19;
    c.cout = 273;
    c.di = 3;
    c.kd = 1;
    c.hi = 4;
    c.wi = 4;
    shapes.push_back(c); // M/N/Cin tails
    c = base;
    c.cin = 1;
    c.cout = 1;
    c.kd = c.kh = c.kw = 1;
    shapes.push_back(c); // zero-sized reduction pieces
    c = base;
    c.di = 2;
    c.kd = 3;
    c.padFront = c.padBack = 5;
    shapes.push_back(c); // entire depth-padding contributors
    c = base;
    c.hi = 9;
    c.wi = 10;
    c.strideH = 2;
    c.strideW = 2;
    c.dilationH = 2;
    c.dilationW = 2;
    c.padLeft = 0;
    c.padRight = 2;
    shapes.push_back(c);
    int count = 0;
    for (size_t shape = 0; shape < shapes.size(); ++shape)
        for (unsigned dtype = 0; dtype < 3; ++dtype)
            for (unsigned db = 1; db <= 2; ++db)
                for (unsigned mode = 1; mode <= 3; ++mode) {
                    c = shapes[shape];
                    c.dtype = ConvDwSkDType(dtype);
                    c.c0Db = db;
                    c.streamkMode = ConvDwSkMode(mode);
                    std::cerr << "shape=" << shape << " ";
                    int rc = RunCase(c, stream);
                    if (rc)
                        return rc;
                    ++count;
                }
    c = base;
    c.dtype = ConvDwSkDType::FP32;
    c.hf32 = 1;
    if (int rc = RunCase(c, stream)) {
        return rc;
    }
    ++count;
    // A segment exceeding the upstream 65536 reduction threshold.
    c = base;
    c.n = 1;
    c.cin = 1;
    c.cout = 1;
    c.di = c.kd = c.kh = c.kw = 1;
    c.hi = 300;
    c.wi = 257;
    c.padUp = c.padDown = c.padLeft = c.padRight = 0;
    c.streamkMode = ConvDwSkMode::DISABLED;
    if (int rc = RunCase(c, stream)) {
        return rc;
    }
    ++count;
    std::cout << "PASS " << count << " device cases, each CPU golden + 3 repeated bitwise checks\n";
    ACL(aclrtDestroyStream(stream));
    ACL(aclrtResetDevice(device));
    ACL(aclFinalize());
}
