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
 * \file broadcast_common.h
 * \brief AscendC 广播算子通用件：常量 / TilingData / 切分辅助 helper / 四类广播原语。
 *        纯 AscendC API，不依赖任何模板调度框架，可直接拷入自定义算子工程裁剪使用。
 *        平台：DAV_3510（A5 / Ascend950，C310）。
 *
 * 四类广播实现：
 *   ④ OneDim(前置)        —— 合轴塌一维时整算子走此路：标量首块 Duplicate 铺满、满输入直搬，tiling 退化 1D
 *   ① BroadcastNddma      —— GM→UB 多维 DMA，广播轴 srcStride=0 由引擎复制（不占 Vector）
 *   ② DataCopyPadCompact  —— 每轮紧凑搬入，广播由 GetGmOffset 的 stride=0 寻址实现（外层广播轴用）；
 *                            offset 相同时复用为可选优化（本样例未启用）
 *   ③ UbBroadcast         —— DataCopyPad 搬紧凑源后用 Broadcast 矢量指令在 UB 内展开
 */
#ifndef BROADCAST_COMMON_H
#define BROADCAST_COMMON_H

#include "kernel_operator.h"

namespace BrcDemo {
using namespace AscendC;

constexpr int64_t BROADCAST_MAX_DIMS = 8;   // 支持的最大维数
constexpr int64_t NDDMA_DIM          = 5;   // 单次多维 DMA 维数上限
constexpr int64_t BLOCK_LENGTH       = 32;  // 对齐字节

// brcMode：每个输入由 Host tiling 选定的广播实现
enum BrcMode { BRC_NONE = 0, BRC_NDDMA = 1, BRC_DATACOPYPAD = 2, BRC_UB = 3 };

// 自定义算子可直接照搬这组字段（与框架无关）。IN_NUM 由具体算子定。
template <int IN_NUM>
struct BroadcastTilingData {
    int32_t brcMode[IN_NUM];          // 每个输入的广播实现（见 BrcMode）
    int32_t shapeLen;                 // 输出维数
    int32_t ubSplitAxis;              // UB 切分轴
    int32_t ubFormer, ubTail;         // UB 主块/尾块（沿 ubSplitAxis）
    int64_t ubOuter;                  // UB 外循环次数
    int64_t blockNum;                 // 核数
    int64_t blockFormer, blockTail;   // 每核外循环单元数（主/尾核）
    int64_t dimProductBeforeUbInner;  // UB tile 总数 = ∏outputDims[0..ubSplitAxis-1] × ubOuter
    int64_t elemNum;                  // 单 buffer 可容纳元素数

    int64_t outputDims[BROADCAST_MAX_DIMS];
    int64_t outputStrides[BROADCAST_MAX_DIMS];
    int64_t inputDims[IN_NUM][BROADCAST_MAX_DIMS];     // 广播轴长度=1
    int64_t inputStrides[IN_NUM][BROADCAST_MAX_DIMS];  // 广播轴 stride=0 ★关键
};

// ============================ Host 侧通用切分 ============================
// 除数用「(x > 0 ? x : 1)」内联钳制，保证静态分析可证除数恒 >=1（G.EXP.22-CPP）；
// 正常调用 b/al 恒 >=1，钳制恒取原值，行为不变。
static inline int64_t BrcCeilDiv(int64_t a, int64_t b)   { int64_t d = (b > 0) ? b : 1; return (a + d - 1) / d; }
static inline int64_t BrcAlignDown(int64_t a, int64_t al) { int64_t d = (al > 0) ? al : 1; return (a / d) * d; }

// 通用两层切分：自动选 ubSplitAxis 并算 ubFormer/ubOuter/ubTail/dimProductBeforeUbInner/多核切分。
// 适用任意 rank、任意输出 shape，含「尾轴超 UB 容量」「大 shape 无广播」等场景（替代写死 ubSplitAxis=0）。
// 前置：td.shapeLen / td.outputDims / td.outputStrides（连续布局）已填。aliveBuf=存活 buffer 数。
template <int IN_NUM>
inline void ComputeTiling(BroadcastTilingData<IN_NUM>& td, int dtSize, int64_t coreNum,
                          int64_t ubSize, int64_t aliveBuf) {
    int rank = td.shapeLen;
    int64_t dtBytes = (dtSize > 0) ? dtSize : 1;   // 钳制除数，保证静态可证 >=1（G.EXP.22-CPP）
    int64_t alignEle = BLOCK_LENGTH / dtBytes;
    td.elemNum = BrcAlignDown(ubSize / (aliveBuf * dtBytes * 2 /*DoubleBuffer*/), alignEle);
    if (td.elemNum < alignEle) td.elemNum = alignEle;

    // 1) 从尾轴向外选切分轴：尽量把内层整轴塞进 UB
    int splitAxis = 0;
    int64_t innerBelow = 1;                              // ∏ outputDims[splitAxis+1 .. rank-1]
    for (int ax = rank - 1; ax >= 0; ax--) {
        if (innerBelow * td.outputDims[ax] > td.elemNum) { splitAxis = ax; break; }
        innerBelow *= td.outputDims[ax];
        splitAxis = ax;
    }
    // 2) 切分轴每行(inner)元素数 = innerBelow（= outputStrides[splitAxis]）
    int64_t ubFormer;
    if (innerBelow > td.elemNum) {                       // 连一行都放不下 → 退到最后一轴按 elemNum 分块
        td.ubSplitAxis = rank - 1;
        innerBelow = 1;
        ubFormer = td.elemNum;                           // 已对齐
    } else {
        td.ubSplitAxis = splitAxis;
        ubFormer = td.elemNum / innerBelow;              // 一次能放几行
    }
    int64_t splitDim = td.outputDims[td.ubSplitAxis];
    if (ubFormer > splitDim) ubFormer = splitDim;
    if (ubFormer < 1) ubFormer = 1;
    td.ubFormer = ubFormer;
    td.ubOuter  = BrcCeilDiv(splitDim, ubFormer);
    td.ubTail   = splitDim - (td.ubOuter - 1) * ubFormer;   // 可能非 alignEle 倍数，DataCopyPad 自动处理

    // 3) dimProductBeforeUbInner = ∏outputDims[0..ubSplitAxis-1] × ubOuter（UB tile 总数，通用公式）
    int64_t outerProd = 1;
    for (int j = 0; j < td.ubSplitAxis; j++) outerProd *= td.outputDims[j];
    td.dimProductBeforeUbInner = outerProd * td.ubOuter;

    // 4) 多核：先按核数算每核 tile 数(blockFormer)，再回收用不上的核(blockNum)，避免 blockTail 为负。
    //    （错误写法：先 blockNum=min(core,total) 再 blockFormer=ceil(total/blockNum) → 末核 blockTail 可能<0）
    int64_t totalTiles = td.dimProductBeforeUbInner;
    int64_t cn = (coreNum < 1) ? 1 : coreNum;
    td.blockFormer = BrcCeilDiv(totalTiles, cn);          // 每核最多几个 tile
    if (td.blockFormer < 1) td.blockFormer = 1;
    td.blockNum    = BrcCeilDiv(totalTiles, td.blockFormer); // 回收无效核
    td.blockTail   = totalTiles - (td.blockNum - 1) * td.blockFormer; // ∈[1, blockFormer]
}

// ============================ 切分辅助 helper（纯标量） ============================

// ① 线性 tile 序号 → 多维下标。axes[ubSplitAxis] ∈ [0, ubOuter)，以 ubFormer 行为单位。
//    tileLinear = blockFormer*GetBlockIdx() + loop；tileTotal = dimProductBeforeUbInner
__aicore__ inline void GetAxesIndices(int64_t (&axes)[BROADCAST_MAX_DIMS], int64_t tileLinear,
        const int64_t (&outputDims)[BROADCAST_MAX_DIMS], int64_t ubSplitAxis, int64_t tileTotal) {
    int64_t rem = tileTotal;
    for (int64_t idx = 0; idx < ubSplitAxis; idx++) {
        rem /= outputDims[idx];
        axes[idx]   = tileLinear / rem;
        tileLinear -= axes[idx] * rem;
    }
    axes[ubSplitAxis] = tileLinear;
}

// ② 外循环每轮 +1（带进位）。切分轴上界 ubOuter，其余轴 outputDims。ubSplitAxis=0 时无外层轴可进位。
__aicore__ inline void UpdateAxesIndices(int64_t (&axes)[BROADCAST_MAX_DIMS],
        const int64_t (&outputDims)[BROADCAST_MAX_DIMS], int64_t ubSplitAxis, int64_t ubOuter) {
    axes[ubSplitAxis]++;
    if (axes[ubSplitAxis] == ubOuter) {
        axes[ubSplitAxis] = 0;
        if (ubSplitAxis > 0) axes[ubSplitAxis - 1]++;     // 防 ubSplitAxis=0 时 axes[-1] 越界
    }
    for (int64_t idx = ubSplitAxis - 1; idx >= 1; idx--) {
        if (axes[idx] == outputDims[idx]) { axes[idx] = 0; axes[idx - 1]++; }
    }
}

// ③ 多维下标 → GM 线性偏移。广播轴 stride=0 → 该轴自动原地复用，无需特殊分支
__aicore__ inline int64_t GetGmOffset(const int64_t (&axes)[BROADCAST_MAX_DIMS],
        const int64_t (&strides)[BROADCAST_MAX_DIMS], int64_t ubSplitAxis, int64_t ubFormer) {
    int64_t off = 0;
    for (int64_t idx = 0; idx < ubSplitAxis; idx++) off += axes[idx] * strides[idx];
    off += axes[ubSplitAxis] * strides[ubSplitAxis] * ubFormer;   // 切分轴以 ubFormer 行为步长
    return off;
}

// ④ 填 UB 内 Broadcast 的 dst/src shape（首轴随主块/尾块变化），供 ③ 使用
__aicore__ inline void FillUbShape(uint32_t (&dstShape)[BROADCAST_MAX_DIMS],
        uint32_t (&srcShape)[BROADCAST_MAX_DIMS], const int64_t (&outputDims)[BROADCAST_MAX_DIMS],
        const int64_t (&inDims)[BROADCAST_MAX_DIMS], int64_t ubSplitAxis, int64_t shapeLen,
        int64_t curRows) {
    dstShape[0] = (uint32_t)curRows;
    srcShape[0] = (inDims[ubSplitAxis] == 1) ? 1u : (uint32_t)curRows;  // 切分轴广播则源首轴=1
    for (int64_t i = ubSplitAxis + 1, j = 1; i < shapeLen; i++, j++) {
        dstShape[j] = (uint32_t)outputDims[i];
        srcShape[j] = (uint32_t)inDims[i];                             // 广播轴在源上=1
    }
}

// ③ 用：当前 tile 紧凑源的元素数 = 源行数 × 切分轴以下各源轴乘积（广播轴在源上为 1）。
// 注意：不能用输出 tile 长度（那是展开后的），否则 [M,1] 这类尾轴广播会多搬。
__aicore__ inline int64_t UbSrcLen(const int64_t (&inStrides)[BROADCAST_MAX_DIMS],
        const int64_t (&inDims)[BROADCAST_MAX_DIMS], int64_t ubSplitAxis, int64_t shapeLen, int64_t rows) {
    int64_t srcRows = (inStrides[ubSplitAxis] != 0) ? rows : 1;        // 切分轴广播则源只 1 行
    int64_t inner = 1;
    for (int64_t j = ubSplitAxis + 1; j < shapeLen; j++) inner *= inDims[j];
    return srcRows * inner;
}

// ============================ 实现①：NDDMA 搬入即广播 ============================

// 装配多维 DMA 参数：广播轴 loopSrcStride 天然=0 → 引擎复制（逻辑同 atvoss WithoutLoop）
template <typename T>
__aicore__ inline AscendC::MultiCopyParams<T, NDDMA_DIM> MakeNddmaParams(
        const int64_t (&outputDims)[BROADCAST_MAX_DIMS], const int64_t (&outputStrides)[BROADCAST_MAX_DIMS],
        const int64_t (&inputStrides)[BROADCAST_MAX_DIMS], int64_t shapeLen, int64_t ubSplitAxis,
        int64_t ubSplitSize) {
    AscendC::MultiCopyLoopInfo<NDDMA_DIM> loop;
    int64_t axisInsideUb = NDDMA_DIM - (shapeLen - ubSplitAxis);  // 高位需补几个 size=1 的轴
    // 防御：rank=shapeLen-ubSplitAxis>NDDMA_DIM 时 axisInsideUb<0，下面 NDDMA_DIM-1-axisInsideUb 会越界。
    // Host 选型已对 rank>5 回退 BRC_UB（见 broadcast_add_tiling.cpp / PickBroadcastMode），此处再夹一道。
    if (axisInsideUb < 0) axisInsideUb = 0;                       // 仅防越界；rank>5 应由 Host 拦截

    for (int64_t i = 0; i < axisInsideUb; i++) {
        loop.loopSize[NDDMA_DIM - 1 - i]      = 1;
        loop.loopSrcStride[NDDMA_DIM - 1 - i] = inputStrides[ubSplitAxis];
        loop.loopDstStride[NDDMA_DIM - 1 - i] = outputStrides[ubSplitAxis];
    }
    loop.loopSize[NDDMA_DIM - 1 - axisInsideUb]      = ubSplitSize;
    loop.loopSrcStride[NDDMA_DIM - 1 - axisInsideUb] = inputStrides[ubSplitAxis];
    loop.loopDstStride[NDDMA_DIM - 1 - axisInsideUb] = outputStrides[ubSplitAxis];
    for (int64_t i = axisInsideUb + 1; i < NDDMA_DIM; i++) {
        int64_t axis = ubSplitAxis + i - axisInsideUb;
        loop.loopSize[NDDMA_DIM - 1 - i]      = outputDims[axis];
        loop.loopSrcStride[NDDMA_DIM - 1 - i] = inputStrides[axis];   // 广播轴=0
        loop.loopDstStride[NDDMA_DIM - 1 - i] = outputStrides[axis];
    }
    return AscendC::MultiCopyParams<T, NDDMA_DIM>{loop, (T)0};
}

// gm 已按当前 tile 偏移；UB tile 内存在广播（split 轴源/目的跨度不一致）→ 多维 DMA 复制，
// 否则退化为连续搬运（即②）。前置条件：rank = shapeLen - ubSplitAxis <= NDDMA_DIM(=5)，
// rank>5 需 with-loop 版本（本头未实现）；Host 选型须保证（见 broadcast_add_tiling.cpp 的 rank 兜底）。
template <typename T>
__aicore__ inline void BroadcastNddma(const GlobalTensor<T>& gm, const LocalTensor<T>& ub,
        const int64_t (&outputDims)[BROADCAST_MAX_DIMS], const int64_t (&outputStrides)[BROADCAST_MAX_DIMS],
        const int64_t (&inputStrides)[BROADCAST_MAX_DIMS], int64_t shapeLen, int64_t ubSplitAxis,
        int64_t ubSplitSize) {
    if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {
        auto params = MakeNddmaParams<T>(outputDims, outputStrides, inputStrides, shapeLen, ubSplitAxis, ubSplitSize);
        static constexpr AscendC::MultiCopyConfig cfg = {false, 0, 0, false};  // 必须 static：模板参为 const&
        AscendC::DataCopy<T, NDDMA_DIM, cfg>(ub, gm, params);
    } else {
        AscendC::DataCopyExtParams ext{1, (uint32_t)(ubSplitSize * inputStrides[ubSplitAxis] * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(ub, gm, ext, pad);
    }
}

// ============================ 实现②：DataCopyPad 紧凑搬入 ============================

// 紧凑连续搬一段（不复制，lenEle 可非 32B 对齐，DataCopyPad 自动 pad）。② 广播效果不在这里，
// 而在 GM offset：外层广播轴 stride=0，GetGmOffset 会让这些轴推进时仍指向同一段 GM（见 GetGmOffset）。
template <typename T>
__aicore__ inline void DataCopyPadCompact(const GlobalTensor<T>& gm, const LocalTensor<T>& ub, int64_t lenEle) {
    AscendC::DataCopyExtParams ext{1, (uint32_t)(lenEle * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(ub, gm, ext, pad);
}

// 可选优化（本样例未启用，正确性见 datacopypad_design.md）：当某广播输入本轮 GM offset 与上轮相同
// （仅外层广播轴推进、切分轴未变）时可跳过搬运、复用上轮 UB。DoubleBuffer 下需按 buffer 槽位分别记录
// 已填 offset，比"切分轴 stride"判据更通用——后者在 ② 路径恒为真，会让复用永不生效（死路径）。

// ============================ 实现③：UB 内 Broadcast 指令 ============================

// src 已是 UB 内紧凑源（GM 输入先 DataCopyPadCompact，或上游计算结果）；dst 为展开目标
template <typename T>
__aicore__ inline void UbBroadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src,
        const int64_t (&outputDims)[BROADCAST_MAX_DIMS], const int64_t (&inDims)[BROADCAST_MAX_DIMS],
        int64_t ubSplitAxis, int64_t shapeLen, int64_t curRows) {
    uint32_t dShape[BROADCAST_MAX_DIMS], sShape[BROADCAST_MAX_DIMS];
    FillUbShape(dShape, sShape, outputDims, inDims, ubSplitAxis, shapeLen, curRows);
    int64_t rank = shapeLen - ubSplitAxis;
    // 退化守卫：无任何可展开维（所有 dShape==sShape，如切分轴广播但 tile 内仅 1 行）时，
    // Broadcast 会变成 src==dst 的空广播并产生 nan。此时直接拷贝 src→dst（src 已是该 tile 正确数据）。
    bool anyExpand = false; int64_t cnt = 1;
    for (int64_t k = 0; k < rank; k++) { if (dShape[k] > sShape[k]) anyExpand = true; cnt *= dShape[k]; }
    if (!anyExpand) {                              // 无可展开维：直接 UB→UB 拷贝（dtype 无关，避免空广播 nan）
        int32_t alignEle = BLOCK_LENGTH / sizeof(T);
        int32_t cntAlign = ((int32_t)cnt + alignEle - 1) / alignEle * alignEle;  // 向上对齐；buffer=elemNum 不越界
        AscendC::DataCopy(dst, src, cntAlign);    // CopyOut 只写真实 tile，多拷的对齐尾部无害
        return;
    }
    AscendC::BroadcastTiling bt;
    AscendC::GetBroadcastTilingInfo<T>(rank, dShape, sShape, false, bt);
    AscendC::Broadcast<T>(dst, src, dShape, sShape, &bt);
}

// ============================ 实现④：OneDim 合轴塌一维快路径（标量广播） ============================
// 适用：合轴后塌成一维——每个输入要么满 shape（连续）、要么纯标量，全连续。是 scalar 广播（lr/scale/
// alpha 等）与同 shape elementwise 的最快实现：标量首块 Duplicate 铺满 UB 后复用，不走逐块 DMA；多维
// tiling 退化成 1D，固定成本最低（对标 atvoss SCH_MODE_ONE_DIM_ADVANCE，schMode 202）。
// 注意：[M,1]→[M,N] 这类"部分轴广播"合轴后仍是多维，不属本路径，走 ①②③。

// 极简 TilingData（对标 Advance）：只携带 1D 总长 + 块大小 + 核数 + 标量位，切分由 kernel 运行期推导。
struct OneDimTilingData {
    int64_t dimLen;     // 输出总元素数（合轴后一维长度）
    int32_t tileNum;    // 单块 UB 元素数（已对齐）
    int32_t blockNum;   // 实际启用核数
    int32_t scalarFlag; // bit i = 1 → 第 i 个输入是标量（首块 Duplicate）
};

// Host 选型：判断能否走 OneDim，并求 dimLen / scalarFlag。
// 规则：每个输入要么纯标量（∏inputDims==1）要么满 shape（∏inputDims==dimLen），否则塌不成一维返回 false。
template <int IN_NUM>
inline bool TryOneDim(const BroadcastTilingData<IN_NUM>& td, int64_t& dimLen, int32_t& scalarFlag) {
    dimLen = 1;
    for (int j = 0; j < td.shapeLen; j++) dimLen *= td.outputDims[j];
    scalarFlag = 0;
    for (int i = 0; i < IN_NUM; i++) {
        int64_t cnt = 1;
        for (int j = 0; j < td.shapeLen; j++) cnt *= td.inputDims[i][j];
        if (cnt == 1)            scalarFlag |= (1 << i);   // 纯标量
        else if (cnt != dimLen)  return false;            // 既非标量又非满 shape → 不能塌一维
    }
    return true;
}

// Host tiling（对标 Advance）：只算 dimLen / tileNum / blockNum，其余切分留给 kernel 运行期推。
// aliveBuf=存活 buffer 数（满输入双缓冲计 1，标量首块单块也计 1，输出双缓冲计 1；保守取上界即可）。
inline void ComputeOneDimTiling(OneDimTilingData& t, int64_t dimLen, int32_t scalarFlag,
                                int dtSize, int64_t coreNum, int64_t ubSize, int64_t aliveBuf) {
    t.dimLen     = dimLen;
    t.scalarFlag = scalarFlag;
    int64_t dtBytes = (dtSize > 0) ? dtSize : 1;   // 钳制除数，保证静态可证 >=1（G.EXP.22-CPP）
    int64_t alignEle = BLOCK_LENGTH / dtBytes;
    int64_t tileNum  = BrcAlignDown(ubSize / (aliveBuf * dtBytes * 2 /*DoubleBuffer*/), alignEle);
    if (tileNum < alignEle) tileNum = alignEle;
    int64_t ubOuter = BrcCeilDiv(dimLen, tileNum);        // 一维共几块
    int64_t cn = (coreNum < 1) ? 1 : coreNum;
    if (ubOuter < cn) cn = (ubOuter < 1) ? 1 : ubOuter;   // 块数不足核数 → 减核（小 shape 自适应）
    t.tileNum  = (int32_t)tileNum;
    t.blockNum = (int32_t)cn;
}

// Kernel 端运行期推导当前核的循环参数（对标 Advance：tiling 只给 dimLen/tileNum/blockNum）。
//   baseOffset = 本核起始元素偏移；loops = 本核块数；tailLen = 本核最后一块的元素数。
struct OneDimCoreParam { int64_t baseOffset; int64_t loops; int64_t tailLen; };
__aicore__ inline OneDimCoreParam OneDimCalcCore(int64_t dimLen, int64_t tileNum, int64_t blockNum) {
    int64_t tn = (tileNum  > 0) ? tileNum  : 1;   // 钳制除数，保证静态可证 >=1（G.EXP.22-CPP）
    int64_t bn = (blockNum > 0) ? blockNum : 1;
    int64_t ubOuter     = (dimLen + tn - 1) / tn;
    int64_t ubTail      = dimLen - (ubOuter - 1) * tn;                 // 最后一块元素数（可非对齐）
    int64_t blockFormer = (ubOuter + bn - 1) / bn;                     // 每核块数（先定块数）
    int64_t blockTail   = ubOuter - (bn - 1) * blockFormer;            // 尾核块数 ∈[1, blockFormer]
    int64_t bid = AscendC::GetBlockIdx();
    OneDimCoreParam p;
    p.baseOffset = blockFormer * tn * bid;
    bool lastCore = (bid == bn - 1);
    p.loops   = lastCore ? blockTail : blockFormer;
    p.tailLen = lastCore ? ubTail : tn;                                // 非尾核最后一块仍是整块
    return p;
}

// OneDim 单输入搬入：标量仅首块 Duplicate 铺满（后续块复用同一 UB，调用方勿重 Alloc）；满 shape 逐块连续搬。
template <typename T>
__aicore__ inline void OneDimLoadInput(const LocalTensor<T>& dst, const GlobalTensor<T>& gm,
        int64_t offset, int64_t len, bool isScalar, bool firstTile) {
    if (isScalar) {
        if (firstTile) AscendC::Duplicate(dst, gm.GetValue(0), len);   // 标量值铺满；之后复用
    } else {
        DataCopyPadCompact(gm[offset], dst, len);                      // 满输入：连续搬，非对齐 Pad 兜底
    }
}

} // namespace BrcDemo
#endif // BROADCAST_COMMON_H
