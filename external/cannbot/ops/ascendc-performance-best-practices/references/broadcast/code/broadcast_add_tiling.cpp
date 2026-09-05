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
 * \file broadcast_add_tiling.cpp
 * \brief 端到端样例（Host 侧）：z = x + y，演示「识别广播轴 → 两层切分 → 逐输入选 brcMode」。
 *        范式级代码：AlignDown/CeilDiv/平台查询等按工程惯例占位，落地需接 op proto / tiling 注册。
 *        输出 [M, N]；x 满 shape；y = [1,N]（外层广播）或 [M,1]（尾轴广播）。
 */
#include <set>
#include <algorithm>
#include "broadcast_common.h"

namespace BrcDemo {

constexpr int IN_NUM = 2;   // x, y
using TD = BroadcastTilingData<IN_NUM>;

// 广播轴扫描结果：把 PickBroadcastMode 里「扫描所有轴得到的布尔标志」聚合成小结构体，
// 便于主函数据此选型，同时降低主函数圈复杂度（选型语义不变）。
struct BrcAxisFlags {
    bool anyBrc;          // 存在任意广播轴
    bool brcInTile;       // 广播轴落在 UB tile 内（含尾轴）
    bool nonLastBrc;      // 存在非尾轴广播
    bool lastAxisBrc;     // 尾轴本身是广播轴
    bool inTileExpandable; // tile 内存在真正可展开维（dst>src）
};

// 扫描所有轴，判定各广播标志（与原 PickBroadcastMode 内联逻辑逐条等价）。
static BrcAxisFlags AnalyzeBroadcastAxes(const TD& td, int i) {
    BrcAxisFlags f{false, false, false, false, false};
    for (int64_t j = 0; j < td.shapeLen; j++) {
        if (td.inputStrides[i][j] == 0 && td.outputStrides[j] != 0) {
            f.anyBrc = true;
            if (j >= td.ubSplitAxis)  f.brcInTile = true;       // 广播轴落在 UB tile 内（含尾轴）
            if (j < td.shapeLen - 1)  f.nonLastBrc = true;      // 存在非尾轴广播
            else                      f.lastAxisBrc = true;     // 尾轴本身是广播轴
        }
    }
    // 「在 tile 内」不等于「UB 能展开」：切分轴是广播轴且 ubFormer==1 时 tile 内仅 1 行（dShape==sShape）。
    for (int64_t j = td.ubSplitAxis + 1; j < td.shapeLen; j++)                 // 切分轴以下某轴广播
        if (td.inputStrides[i][j] == 0 && td.outputDims[j] > 1) f.inTileExpandable = true;
    if (td.inputStrides[i][td.ubSplitAxis] == 0 && td.outputDims[td.ubSplitAxis] > 1 && td.ubFormer > 1)
        f.inTileExpandable = true;                                             // 切分轴广播且 tile 内多行
    return f;
}

// dtype 是否属于 UB Broadcast 集合（B8/B16）。
static bool IsUbBroadcastDtype(ge::DataType dtype) {
    static const std::set<ge::DataType> kUb =
        {ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_UINT16};
    return kUb.count(dtype) != 0;
}

// 三类选型（见 broadcast_design.md §1.3）。须扫描所有轴，区分「tile 内是否有广播轴」与
// 「是否所有广播轴都在切分轴之上」，不能只看第一个广播轴。dcache 阈值用平台查询，源码注释约 4096B。
static int PickBroadcastMode(const TD& td, int i, int dtSize, ge::DataType dtype, int64_t nddmaDcache) {
    BrcAxisFlags f = AnalyzeBroadcastAxes(td, i);
    if (!f.anyBrc) return BRC_NONE;

    // ★ 广播轴全在外层（tile 外，严格 < ubSplitAxis）→ ② DataCopyPad（外层广播靠 GetGmOffset 的
    //   stride=0 寻址；此时切分轴非广播，len=rows*stride 成立）。必须先判，否则 ②的 len 公式会错。
    if (!f.brcInTile) return BRC_DATACOPYPAD;

    // 广播轴在 tile 内（≥ubSplitAxis）。但「在 tile 内」不等于「UB 能展开」：
    //   切分轴是广播轴且 ubFormer==1 时，tile 内该轴只有 1 行（dShape==sShape），UB Broadcast 退化成
    //   空广播会出 nan。只有存在真正可展开维（dst>src）时才可选 ③，否则走 ① NDDMA（ubFormer==1 下
    //   NDDMA 搬 inner 连续数据 + 外层靠 offset 广播，仍正确）。
    int64_t lastDim  = td.outputDims[td.shapeLen - 1];
    int64_t dtBytes  = (dtSize > 0) ? dtSize : 1;   // 钳制除数，保证静态可证 >=1（G.EXP.22-CPP）
    int64_t alignEle = BLOCK_LENGTH / dtBytes;

    if (f.inTileExpandable) {                                                  // 仅有可展开维才考虑 ③
        if (lastDim % alignEle == 0 && IsUbBroadcastDtype(dtype))            return BRC_UB; // ①尾轴对齐+B8/B16
        if (f.nonLastBrc && !f.lastAxisBrc && lastDim * dtBytes >= nddmaDcache / 2) return BRC_UB; // ②BigNLast
    }
    // rank>5：without-loop NDDMA 装配不下（>NDDMA_DIM 维），本样例未实现 with-loop → 一律退 UB
    //（含 !inTileExpandable：UB 的退化守卫会在无可展开维时拷贝兜底，不会空广播）。
    if (td.shapeLen - td.ubSplitAxis > NDDMA_DIM)                     return BRC_UB;
    return BRC_NDDMA;                                                                 // 其余 tile 内广播 → ① NDDMA
}

// 简化的切分入口（省略 context 取值/校验）
ge::graphStatus AddBrcTiling(TD& td, int64_t M, int64_t N, const int64_t yShape[2],
                             int dtSize, ge::DataType dtype, int64_t coreNum, int64_t ubSize,
                             int64_t nddmaDcache) {
    // ---- 1. shape / stride（输出连续；广播轴 stride=0）----
    td.shapeLen = 2;
    td.outputDims[0] = M; td.outputDims[1] = N;
    td.outputStrides[0] = N; td.outputStrides[1] = 1;
    // x：满 shape
    td.inputDims[0][0] = M; td.inputDims[0][1] = N;
    td.inputStrides[0][0] = N; td.inputStrides[0][1] = 1;
    // y 紧凑存储：[1,N] → dims{1,N} stride{0,1}（外层广播）；[M,1] → dims{M,1} stride{1,0}（尾轴广播）
    bool yBrcOuter = (yShape[0] == 1);
    td.inputDims[1][0] = yBrcOuter ? 1 : M;
    td.inputDims[1][1] = yBrcOuter ? N : 1;
    td.inputStrides[1][0] = yBrcOuter ? 0 : 1;   // [M,1] 行 stride=1（每行 1 个元素），不是 N
    td.inputStrides[1][1] = yBrcOuter ? 1 : 0;

    // ---- 2. 通用两层切分：自动选 ubSplitAxis，支持尾轴超 UB / 大 shape（见 advanced_tiling.md）----
    // buffer 槽位：x/y/z 各双缓冲(3×2=6) + ③紧凑源单块(1) = 7；这里 4×2=8 取保守上界
    int64_t aliveBuf = 4;
    ComputeTiling(td, dtSize, coreNum, ubSize, aliveBuf);   // 算 ubSplitAxis/ubFormer/ubOuter/ubTail/
                                                           // dimProductBeforeUbInner/blockNum 等（见 broadcast_common.h）

    // ---- 3. 逐输入选 brcMode（依赖上一步算出的 ubSplitAxis）----
    for (int i = 0; i < IN_NUM; i++)
        td.brcMode[i] = PickBroadcastMode(td, i, dtSize, dtype, nddmaDcache);
    // x 满 shape → BRC_NONE。y 的归属由 PickBroadcastMode 按优先级 + 实际 ubSplitAxis 决定：
    //   尾轴对齐且 B8/B16 → BRC_UB；否则看广播轴相对 ubSplitAxis 的位置：
    //   在 tile 内(含切分轴/尾轴) → BRC_NDDMA；全在外层 → BRC_DATACOPYPAD。
    //   注意 ubSplitAxis 现由 ComputeTiling 动态选定（大 N 会切到尾轴），不再恒为 0。

    // context_->SetBlockDim(td.blockNum);  // 落地时设置
    return ge::GRAPH_SUCCESS;
}

} // namespace BrcDemo
