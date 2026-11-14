/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file x_all_gather_matmul_tiling_key.h
 * \brief Tiling-key constants for x_all_gather_matmul (non-quant, A2/Ascend910B).
 *
 * Migrated from the source's template-tiling-key header (ASCENDC_TPL_BOOL_DECL /
 * GET_TPL_TILING_KEY). x_ops kernels use the extern "C" + TILING_KEY_IS(<numeric>)
 * dispatch convention (see x_mega_moe / x_rms_norm_*), so the template-tiling-key
 * machinery is replaced by plain numeric constants with the SAME bit encoding the
 * source used:
 *   bit0 = full mesh (always set for this op)  -> 1
 *   bit1 = ND2NZ optimization                  -> 2
 *   bit2 = bias cast to float                  -> 4
 * The host-side UpdateTilingKey computes the identical value (1 | (nd2nz?2:0) |
 * (bias?4:0)) and sets it via context->SetTilingKey(...); the kernel selects the
 * branch with TILING_KEY_IS(...).
 */

#ifndef __OP_KERNEL_ALL_GATHER_MATMUL_TILING_KEY_H__
#define __OP_KERNEL_ALL_GATHER_MATMUL_TILING_KEY_H__

// Tiling-key values are #define macros (not constexpr) so the TBE precompiler's textual scan can
// resolve TILING_KEY_IS(X_AGM_TK_*): constexpr variables can't be evaluated in the precompilation
// phase ("can not be processed as numeric variables"), which is exactly the error the build logged.
// Matches x_rms_norm_*'s #define convention. Macros escape the namespace (preprocessor is
// namespace-agnostic), so the block is kept only to leave `using namespace x_all_gather_matmul_tiling_key;`
// (kernel/host) a harmless no-op.
namespace x_all_gather_matmul_tiling_key {
// full mesh only (no nd2nz, no bias cast)
#define X_AGM_TK_FULLMESH 1
// full mesh + nd2nz, no bias cast   (A2 default: isND2NZ is always 1)
#define X_AGM_TK_FULLMESH_ND2NZ 3
// full mesh, no nd2nz, bias cast
#define X_AGM_TK_FULLMESH_BIAS 5
// full mesh + nd2nz + bias cast      (A2 with bias)
#define X_AGM_TK_FULLMESH_ND2NZ_BIAS 7
}  // namespace x_all_gather_matmul_tiling_key
#endif  // __OP_KERNEL_ALL_GATHER_MATMUL_TILING_KEY_H__
