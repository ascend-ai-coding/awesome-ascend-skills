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
 * \file broadcast_add_kernel.cpp
 * \brief 端到端样例（Kernel 侧）：z = x + y，演示「§7 helper 还原下标 → 逐输入按 brcMode 走三类写法」。
 *        范式级代码：GET_TILING_DATA / DTYPE_X 等按工程宏占位。
 */
#include "broadcast_common.h"

namespace BrcDemo {

template <typename T>
class BroadcastAdd {
public:
    static constexpr int IN_NUM = 2;
    using TD = BroadcastTilingData<IN_NUM>;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const TD& t) {
        td_ = t;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        zGm_.SetGlobalBuffer((__gm__ T*)z);
        pipe_.InitBuffer(qx_, 2, td_.elemNum * sizeof(T));
        pipe_.InitBuffer(qy_, 2, td_.elemNum * sizeof(T));
        pipe_.InitBuffer(qz_, 2, td_.elemNum * sizeof(T));
        pipe_.InitBuffer(bufSrc_, td_.elemNum * sizeof(T));   // ③ 用：广播前紧凑源
    }

    __aicore__ inline void Process() {
        int64_t loops = (GetBlockIdx() == td_.blockNum - 1) ? td_.blockTail : td_.blockFormer;
        int64_t axes[BROADCAST_MAX_DIMS] = {0};
        GetAxesIndices(axes, td_.blockFormer * GetBlockIdx(), td_.outputDims,
                       td_.ubSplitAxis, td_.dimProductBeforeUbInner);
        for (int64_t loop = 0; loop < loops; loop++) {
            if (loop != 0) UpdateAxesIndices(axes, td_.outputDims, td_.ubSplitAxis, td_.ubOuter);
            int64_t rows   = (axes[td_.ubSplitAxis] == td_.ubOuter - 1) ? td_.ubTail : td_.ubFormer;
            int64_t tile   = rows * td_.outputStrides[td_.ubSplitAxis];   // 本 tile 元素数

            // --- x：普通搬入 ---
            LocalTensor<T> xl = qx_.template AllocTensor<T>();
            CopyInPlain(xGm_, xl, 0, axes, rows);
            qx_.EnQue(xl);
            // --- y：按 brcMode 走三类之一 ---
            LocalTensor<T> yl = qy_.template AllocTensor<T>();
            LoadBroadcastInput(yl, 1, axes, rows, tile);
            qy_.EnQue(yl);
            // --- 计算 + 输出 ---
            LocalTensor<T> xv = qx_.template DeQue<T>();
            LocalTensor<T> yv = qy_.template DeQue<T>();
            LocalTensor<T> zv = qz_.template AllocTensor<T>();
            AscendC::Add(zv, xv, yv, tile);
            qz_.EnQue(zv);
            qx_.FreeTensor(xv); qy_.FreeTensor(yv);
            LocalTensor<T> zo = qz_.template DeQue<T>();
            int64_t zoff = GetGmOffset(axes, td_.outputStrides, td_.ubSplitAxis, td_.ubFormer);
            AscendC::DataCopyExtParams ext{1, (uint32_t)(tile * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[zoff], zo, ext);
            qz_.FreeTensor(zo);
        }
    }

private:
    __aicore__ inline void CopyInPlain(const GlobalTensor<T>& gm, const LocalTensor<T>& ub,
                                       int i, const int64_t (&axes)[BROADCAST_MAX_DIMS], int64_t rows) {
        int64_t off = GetGmOffset(axes, td_.inputStrides[i], td_.ubSplitAxis, td_.ubFormer);
        int64_t len = rows * td_.inputStrides[i][td_.ubSplitAxis];
        DataCopyPadCompact(gm[off], ub, len);
    }

    // 三类广播分发
    __aicore__ inline void LoadBroadcastInput(const LocalTensor<T>& dst, int i,
            const int64_t (&axes)[BROADCAST_MAX_DIMS], int64_t rows, int64_t tile) {
        int64_t off = GetGmOffset(axes, td_.inputStrides[i], td_.ubSplitAxis, td_.ubFormer);
        if (td_.brcMode[i] == BRC_NONE) {                   // 非广播输入：普通紧凑搬入
            int64_t len = rows * td_.inputStrides[i][td_.ubSplitAxis];
            DataCopyPadCompact(yGm_[off], dst, len);
        } else if (td_.brcMode[i] == BRC_NDDMA) {           // ①
            BroadcastNddma(yGm_[off], dst, td_.outputDims, td_.outputStrides,
                           td_.inputStrides[i], td_.shapeLen, td_.ubSplitAxis, rows);
        } else if (td_.brcMode[i] == BRC_DATACOPYPAD) {     // ②：广播轴全在外层
            // 广播由 GetGmOffset 的 stride=0 寻址实现：外层广播轴推进时 off 不变 → 取到同一段 GM。
            // 每轮搬运（正确且安全）；可选复用见 datacopypad_design.md（本样例未启用）。
            int64_t len = rows * td_.inputStrides[i][td_.ubSplitAxis];   // 切分轴非广播，len=tile
            DataCopyPadCompact(yGm_[off], dst, len);
        } else {                                            // ③ BRC_UB
            LocalTensor<T> src = bufSrc_.template Get<T>();
            int64_t srcLen = UbSrcLen(td_.inputStrides[i], td_.inputDims[i], td_.ubSplitAxis, td_.shapeLen, rows);
            DataCopyPadCompact(yGm_[off], src, srcLen);
            UbBroadcast(dst, src, td_.outputDims, td_.inputDims[i], td_.ubSplitAxis, td_.shapeLen, rows);
        }
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> qx_, qy_;
    TQue<TPosition::VECOUT, 2> qz_;
    TBuf<TPosition::VECCALC> bufSrc_;
    GlobalTensor<T> xGm_, yGm_, zGm_;
    TD td_;
};

} // namespace BrcDemo

// 算子入口
extern "C" __global__ __aicore__ void broadcast_add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);                 // 解析 BroadcastTilingData<2>
    BrcDemo::BroadcastAdd<DTYPE_X> op;
    op.Init(x, y, z, td);
    op.Process();
}
