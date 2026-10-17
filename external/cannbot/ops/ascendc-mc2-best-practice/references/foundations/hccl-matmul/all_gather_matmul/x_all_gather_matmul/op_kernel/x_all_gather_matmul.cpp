/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file x_all_gather_matmul.cpp
 * \brief Kernel entry for the non-quant A2 (Ascend910B) x_all_gather_matmul.
 *
 * Migrated from ops-transformer/mc2/x_all_gather_matmul/op_kernel/x_all_gather_matmul.cpp.
 * Adaptation to the x_ops kernel convention: the source used a templated
 * __global__ entry with the template-tiling-key mechanism (ASCENDC_TPL_* +
 * GET_TPL_TILING_KEY). x_ops kernels are extern "C" __global__ entries that
 * dispatch at runtime with TILING_KEY_IS(<numeric>) (see x_mega_moe /
 * x_rms_norm_*). The four (fullmesh x nd2nz x bias) combinations are preserved
 * verbatim; only the dispatch mechanism changed. The numeric tiling keys are
 * defined in x_all_gather_matmul_tiling_key.h and match the value computed by
 * the host-side UpdateTilingKey.
 *
 * The kernel-side code does not support A5 (the source guards the body with
 * __CCE_AICORE__ != 310); that guard is retained.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "x_all_gather_matmul_tiling.h"
#include "x_all_gather_matmul_tiling_key.h"
#include "x_all_gather_matmul_full_mesh.h"

using namespace AscendC;
using namespace x_all_gather_matmul_tiling_key;

#define INVOKE_ALL_GATHER_MATMUL_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                               \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, true>;                               \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                     \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                  \
        op.Init(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, contextGM, &tilingData,     \
                &pipe);                                                                                                \
        op.Process();                                                                                                  \
    } while (0)

template <class T>
struct BiasType {
    using type = float;
};
template <>
struct BiasType<half> {
    using type = half;
};

// x_ops kernel entry convention: extern "C" __global__ __aicore__ void x_<op>(...).
extern "C" __global__ __aicore__ void x_all_gather_matmul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                          GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(Mc2Tiling::XAllGatherMatmulTilingData); 
// allgathermatmul kernel-side code does not support A5
#if __CCE_AICORE__ != 310

    GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::XAllGatherMatmulTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    // Runtime tiling-key dispatch (bit0=fullmesh(1) | bit1=nd2nz(2) | bit2=bias(4)).
    if (TILING_KEY_IS(X_AGM_TK_FULLMESH_ND2NZ)) {
        // full mesh + nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(XAllGatherMatmulFullMesh, true, false);
    } else if (TILING_KEY_IS(X_AGM_TK_FULLMESH)) {
        // full mesh + no nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(XAllGatherMatmulFullMesh, false, false);
    } else if (TILING_KEY_IS(X_AGM_TK_FULLMESH_ND2NZ_BIAS)) {
        // full mesh + nd2nz + bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(XAllGatherMatmulFullMesh, true, true);
    } else if (TILING_KEY_IS(X_AGM_TK_FULLMESH_BIAS)) {
        // full mesh + no nd2nz + bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_ALL_GATHER_MATMUL_OP_IMPL(XAllGatherMatmulFullMesh, false, true);
    }
#endif
}
