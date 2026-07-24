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
 * \file onedim_add_tiling.cpp
 * \brief 端到端样例（Host 侧）：z = x + y，y 为标量 → 合轴塌一维，走 ④ OneDim 快路径。
 *        演示「构造 shape/stride → TryOneDim 命中 → ComputeOneDimTiling 极简切分」。
 *        与 broadcast_add_tiling.cpp 同形式；范式级代码：AlignDown/平台查询/ge 类型等按工程惯例占位，
 *        落地需接 op proto / tiling 注册。
 */
#include "broadcast_common.h"

namespace BrcDemo {

constexpr int IN_NUM = 2;   // x(满 shape), y(标量)
using TD = BroadcastTilingData<IN_NUM>;

// 简化的切分入口（省略 context 取值/校验）。输出 [M,N]；x 满 shape；y 为标量（单元素）。
// 产物是 OneDimTilingData（极简：dimLen/tileNum/blockNum/scalarFlag），写入 GE tiling 上下文。
ge::graphStatus AddScalarTiling(OneDimTilingData& ot, int64_t M, int64_t N,
                                int dtSize, int64_t coreNum, int64_t ubSize) {
    TD td;
    // ---- 1. shape / stride（输出连续；标量输入各轴 dim=1 / stride=0）----
    td.shapeLen = 2;
    td.outputDims[0] = M; td.outputDims[1] = N;
    td.outputStrides[0] = N; td.outputStrides[1] = 1;
    td.inputDims[0][0] = M; td.inputDims[0][1] = N;        // x 满 shape
    td.inputStrides[0][0] = N; td.inputStrides[0][1] = 1;
    td.inputDims[1][0] = 1; td.inputDims[1][1] = 1;        // y 标量（紧凑存 1 个元素）
    td.inputStrides[1][0] = 0; td.inputStrides[1][1] = 0;

    // ---- 2. 前置：能否塌成一维（每输入要么满 shape 要么纯标量）----
    int64_t dimLen; int32_t scalarFlag;
    if (!TryOneDim(td, dimLen, scalarFlag)) {
        // 塌不成一维（部分轴广播，如 [M,1]/[1,N]）→ 改走 ①②③（见 broadcast_add_tiling.cpp）。
        // 本样例 x 满 + y 标量必命中；返回失败仅示意"未命中则换路"。
        return ge::GRAPH_FAILED;
    }
    // scalarFlag：bit0=0(x 满 shape)、bit1=1(y 标量)；kernel 据此对各输入分发

    // ---- 3. OneDim 极简切分：只算 dimLen/tileNum/blockNum，其余切分留给 kernel 运行期 ----
    // buffer 槽位：x 双缓冲(1) + 标量 y 单块(1) + z 双缓冲(1) = 3，取 IN_NUM+1 为保守上界
    int64_t aliveBuf = IN_NUM + 1;
    ComputeOneDimTiling(ot, dimLen, scalarFlag, dtSize, coreNum, ubSize, aliveBuf);

    // context_->SetBlockDim(ot.blockNum);  // 落地时设置
    return ge::GRAPH_SUCCESS;
}

} // namespace BrcDemo
