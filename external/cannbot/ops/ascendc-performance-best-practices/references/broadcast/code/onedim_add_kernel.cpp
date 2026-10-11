/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file onedim_add_kernel.cpp
 * \brief 端到端样例（Kernel 侧）：z = x + y，y 为标量 → ④ OneDim 快路径。
 *        演示「OneDimCalcCore 运行期推切分 → 标量首块 Duplicate(单块复用)、满输入逐块搬 → 一维写回」。
 *        与 broadcast_add_kernel.cpp 同形式；范式级代码：GET_TILING_DATA / DTYPE_X 等按工程宏占位。
 */
#include "broadcast_common.h"

namespace BrcDemo {

template <typename T>
class OneDimAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const OneDimTilingData& t) {
        ot_ = t;
        xScalar_ = ot_.scalarFlag & (1 << 0);
        yScalar_ = ot_.scalarFlag & (1 << 1);
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        zGm_.SetGlobalBuffer((__gm__ T*)z);
        // 满输入走双缓冲队列；标量输入走单块 TBuf（首块 Duplicate，整轮复用，不进 ping-pong）
        if (xScalar_) pipe_.InitBuffer(xBuf_, ot_.tileNum * sizeof(T));
        else          pipe_.InitBuffer(qx_, 2, ot_.tileNum * sizeof(T));
        if (yScalar_) pipe_.InitBuffer(yBuf_, ot_.tileNum * sizeof(T));
        else          pipe_.InitBuffer(qy_, 2, ot_.tileNum * sizeof(T));
        pipe_.InitBuffer(qz_, 2, ot_.tileNum * sizeof(T));
    }

    __aicore__ inline void Process() {
        // 运行期从 dimLen/tileNum/blockNum 推本核循环参数（对标 Advance）
        OneDimCoreParam cp = OneDimCalcCore(ot_.dimLen, ot_.tileNum, ot_.blockNum);
        int64_t off = cp.baseOffset;
        for (int64_t loop = 0; loop < cp.loops; loop++) {
            int64_t len  = (loop == cp.loops - 1) ? cp.tailLen : ot_.tileNum;   // 最后一块可非对齐
            bool    first = (loop == 0);

            LocalTensor<T> xv = LoadInput(qx_, xBuf_, xGm_, xScalar_, off, len, first);
            LocalTensor<T> yv = LoadInput(qy_, yBuf_, yGm_, yScalar_, off, len, first);

            LocalTensor<T> zv = qz_.template AllocTensor<T>();
            AscendC::Add(zv, xv, yv, len);
            qz_.EnQue(zv);
            if (!xScalar_) qx_.FreeTensor(xv);
            if (!yScalar_) qy_.FreeTensor(yv);

            LocalTensor<T> zo = qz_.template DeQue<T>();
            AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[off], zo, ext);                           // 一维连续写回
            qz_.FreeTensor(zo);
            off += ot_.tileNum;
        }
    }

private:
    // 标量：单块 TBuf，首块 Duplicate 一次后复用（不 EnQue）；满输入：队列 Alloc + 连续搬入 + EnQue→DeQue。
    __aicore__ inline LocalTensor<T> LoadInput(TQue<TPosition::VECIN, 2>& q, TBuf<TPosition::VECCALC>& buf,
            const GlobalTensor<T>& gm, bool isScalar, int64_t off, int64_t len, bool firstTile) {
        if (isScalar) {
            LocalTensor<T> s = buf.template Get<T>();
            OneDimLoadInput(s, gm, off, len, /*isScalar=*/true, firstTile);     // 仅首块 Duplicate 铺满
            return s;
        }
        LocalTensor<T> in = q.template AllocTensor<T>();
        OneDimLoadInput(in, gm, off, len, /*isScalar=*/false, firstTile);       // 满输入：连续搬入
        q.EnQue(in);
        return q.template DeQue<T>();
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> qx_, qy_;
    TQue<TPosition::VECOUT, 2> qz_;
    TBuf<TPosition::VECCALC> xBuf_, yBuf_;     // 标量输入专用单块（满输入不用）
    GlobalTensor<T> xGm_, yGm_, zGm_;
    OneDimTilingData ot_;
    bool xScalar_{false}, yScalar_{false};
};

} // namespace BrcDemo

// 算子入口
extern "C" __global__ __aicore__ void onedim_add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(ot, tiling);                 // 解析 OneDimTilingData
    BrcDemo::OneDimAdd<DTYPE_X> op;
    op.Init(x, y, z, ot);
    op.Process();
}
