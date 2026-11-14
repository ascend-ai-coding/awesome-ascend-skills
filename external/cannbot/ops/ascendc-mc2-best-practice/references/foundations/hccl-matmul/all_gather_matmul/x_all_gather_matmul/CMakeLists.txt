# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# x_all_gather_matmul: non-quant A2 (Ascend910B) all_gather + matmul fusion op.
# Migrated from ops-transformer/mc2/x_all_gather_matmul. Self-contained: the must-depend mc2/common
# framework it needs is vendored under mc2_common/ (trimmed of the mc2/3rd mat_mul_v3 / quant_batch_matmul_v3
# chain), so this operator depends only on the CANN/AscendC toolchain + the x_ops build infra — it does not
# pull in other x_ops operators, x_ops/src/utils, or the existing x_ops/src/mc2/common shared tree.

set(_OP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ---- kernel (bisheng/ccec) include roots + per-op -I/-D -----------------------------------------------
# AscendC kernel include root (processor-specific), same resolution as x_mega_moe.
set(_X_AGM_INC_OPTS "")
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _X_AGM_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _X_AGM_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _X_AGM_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif ()

# Host-side CANN "op_common" include root: provides log/log.h (plus base/err_msg.h, securec.h,
# error_manager) consumed by the vendored mc2 logging layer (mc2_log_compat.h -> log/log.h).
# intf_pub only adds $ROOT/include (+/external, /experiment/*), which lacks log/log.h — it lives
# under <arch>-linux/include/op_common. Same arch variants as the ascendc root above.
# Mirrors ops-transformer/common/CMakeLists.txt, which adds ${OPBASE_SOURCE_PATH}/include/op_common
# for the same reason; the bare #include "log/log.h" is kept verbatim from the source repo (do NOT
# switch to "op_common/log/log.h" — that needs the <arch>-linux/include parent root, which x_ops's
# own cmake does not add and ops-transformer adds explicitly itself, so it would not resolve here).
set(_X_AGM_OP_COMMON_INC "")
foreach (_x_agm_arch IN ITEMS aarch64-linux arm64-linux ${CMAKE_SYSTEM_PROCESSOR}-linux)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/op_common/log/log.h)
        set(_X_AGM_OP_COMMON_INC ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/op_common)
        message(STATUS "x_all_gather_matmul: op_common log root = ${_X_AGM_OP_COMMON_INC}")
        break()
    endif ()
endforeach ()

# Host-side CANN "pkg_inc" include root: provides the HCCL headers (hccl/hcom.h and the hcom/hccl
# cluster) consumed by the vendored mc2_hcom_topo_info.{h,cpp}. intf_pub only adds $ROOT/include
# (+/external, /experiment/*), which lacks hccl/; it lives under <arch>-linux/pkg_inc/hccl.
# Mirrors ops-transformer/common/CMakeLists.txt:73 (${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc).
set(_X_AGM_PKG_INC_INC "")
foreach (_x_agm_arch IN ITEMS aarch64-linux arm64-linux ${CMAKE_SYSTEM_PROCESSOR}-linux)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/pkg_inc/hccl/hcom.h)
        set(_X_AGM_PKG_INC_INC ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/pkg_inc)
        message(STATUS "x_all_gather_matmul: hccl pkg_inc root = ${_X_AGM_PKG_INC_INC}")
        break()
    endif ()
endforeach ()

# Host-side CANN arch "external" include root: provides ge_common/ (ge_api_types.h — Status,
# ge::GRAPH_SUCCESS) needed by the vendored ops_legacy/op_tiling/hcom_topo_info.h. The generic
# $ROOT/include/external (intf_pub) lacks ge_common/; it lives under <arch>-linux/include/external.
# Targeted subdir root (not the <arch>-linux/include parent) to avoid shadowing intf_pub's
# register/graph with arch versions — the vendored mc2 register/graph headers already resolve
# from the generic $ROOT/include, and the parent root would be searched first and could swap them.
set(_X_AGM_ARCH_EXTERNAL_INC "")
foreach (_x_agm_arch IN ITEMS aarch64-linux arm64-linux ${CMAKE_SYSTEM_PROCESSOR}-linux)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/external/ge_common/ge_api_types.h)
        set(_X_AGM_ARCH_EXTERNAL_INC ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/external)
        message(STATUS "x_all_gather_matmul: arch external root = ${_X_AGM_ARCH_EXTERNAL_INC}")
        break()
    endif ()
endforeach ()

# Host-side CANN arch "asc/include/tiling" root: provides platform/platform_ascendc.h for the
# vendored mc2_tiling_utils.h:31. It lives under <arch>-linux/asc/include/tiling/platform/ (NOT
# under <arch>-linux/include/platform/, which only has soc_spec.h). Narrow subdir root.
set(_X_AGM_ASC_TILING_INC "")
foreach (_x_agm_arch IN ITEMS aarch64-linux arm64-linux ${CMAKE_SYSTEM_PROCESSOR}-linux)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/asc/include/tiling/platform/platform_ascendc.h)
        set(_X_AGM_ASC_TILING_INC ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/asc/include/tiling)
        message(STATUS "x_all_gather_matmul: asc tiling root = ${_X_AGM_ASC_TILING_INC}")
        break()
    endif ()
endforeach ()

# Host-side CANN arch "include/platform" root: provides soc_spec.h (SocVersion) for the vendored
# mc2_tiling_utils.h:34 (changed to bare #include "soc_spec.h"). Only <arch>-linux/include/platform
# has soc_spec.h (the generic $ROOT/include/platform lacks it). Narrow subdir root — NOT the parent
# <arch>-linux/include, which also has register/graph that would shadow intf_pub's $ROOT/include.
set(_X_AGM_ARCH_PLATFORM_INC "")
foreach (_x_agm_arch IN ITEMS aarch64-linux arm64-linux ${CMAKE_SYSTEM_PROCESSOR}-linux)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/platform/soc_spec.h)
        set(_X_AGM_ARCH_PLATFORM_INC ${ASCEND_CANN_PACKAGE_PATH}/${_x_agm_arch}/include/platform)
        message(STATUS "x_all_gather_matmul: arch platform root = ${_X_AGM_ARCH_PLATFORM_INC}")
        break()
    endif ()
endforeach ()

# operator's own kernel dir + vendored mc2_common dirs (bare includes resolve through these -I).
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/op_kernel)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common/op_kernel)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common/op_host)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common/op_host/op_tiling)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common/utils)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common)
list(APPEND _X_AGM_INC_OPTS -I${_OP_DIR}/mc2_common/op_api)

# The kernel reads dtype via the DTYPE_X1 / DTYPE_Y macros (defined in the vendored mc2_nd_to_nz.h as
# A_DTYPE / B_DTYPE / C_DTYPE / BIAS_DTYPE). These are NOT passed as -D flags here: the opc build
# (ascendc_bin_param_build.py) reads the op def's DataType({DT_FLOAT16, DT_BF16}) list and auto-bakes
# DTYPE_<input/output-name> per per-dtype binary (x1 -> DTYPE_X1, y -> DTYPE_Y) — same as x_mega_moe
# (DTYPE_A/DTYPE_W1/DTYPE_OUT, no -D). Hardcoding -DDTYPE_X1=half here would OVERRIDE that per-dtype
# baking and force every binary to fp16 (breaking bf16). So: no -DDTYPE_* — single build yields both
# fp16 + bf16 binaries, runtime selects by tensor dtype (multi_kernel).
add_ops_compile_options(OP_NAME XAllGatherMatmul
    OPTIONS --cce-auto-sync=off
            -DHCCL_COMM
            -Wno-deprecated-declarations -Wno-error=option-ignored
            -mllvm -cce-aicore-hoist-movemask=false
            ${_X_AGM_INC_OPTS})

# ---- host include dirs (shared by the host targets below) -------------------------------------------
set(_X_AGM_HOST_INC
    ${_OP_DIR}/op_host
    ${_OP_DIR}/op_host/op_tiling
    ${_OP_DIR}/op_host/op_tiling/arch22
    ${_OP_DIR}/op_kernel
    ${_OP_DIR}/mc2_common
    ${_OP_DIR}/mc2_common/op_kernel
    ${_OP_DIR}/mc2_common/op_host
    ${_OP_DIR}/mc2_common/op_host/op_tiling
    ${_OP_DIR}/mc2_common/utils
    ${_OP_DIR}/mc2_common/op_api
)

# Append the CANN op_common root (provides log/log.h for the vendored mc2 logging layer) when found.
if (_X_AGM_OP_COMMON_INC)
    list(APPEND _X_AGM_HOST_INC ${_X_AGM_OP_COMMON_INC})
endif ()

# Append the CANN pkg_inc root (provides hccl/hcom.h + hcom/hccl cluster for the vendored
# mc2_hcom_topo_info layer) when found.
if (_X_AGM_PKG_INC_INC)
    list(APPEND _X_AGM_HOST_INC ${_X_AGM_PKG_INC_INC})
endif ()

# Append the CANN arch external root (provides ge_common/ge_api_types.h for the vendored
# ops_legacy/op_tiling/hcom_topo_info.h) when found.
if (_X_AGM_ARCH_EXTERNAL_INC)
    list(APPEND _X_AGM_HOST_INC ${_X_AGM_ARCH_EXTERNAL_INC})
endif ()

# Append the CANN asc tiling root (platform/platform_ascendc.h) when found.
if (_X_AGM_ASC_TILING_INC)
    list(APPEND _X_AGM_HOST_INC ${_X_AGM_ASC_TILING_INC})
endif ()

# Append the CANN arch platform root (soc_spec.h, bare-include) when found.
if (_X_AGM_ARCH_PLATFORM_INC)
    list(APPEND _X_AGM_HOST_INC ${_X_AGM_ARCH_PLATFORM_INC})
endif ()

# ---- MC2 op def -> op_host_aclnnInner (aclnnInner is auto-generated from _def.cpp) ------------------
target_sources(op_host_aclnnInner PRIVATE op_host/x_all_gather_matmul_def.cpp)
target_include_directories(op_host_aclnnInner PRIVATE ${_X_AGM_HOST_INC})

# ---- tiling -> optiling (operator's own tiling + vendored mc2 tiling framework) ---------------------
target_sources(optiling PRIVATE
    op_host/op_tiling/x_all_gather_matmul_tiling_base.cpp
    op_host/op_tiling/x_all_gather_formulaic_tiling.cpp
    op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp
    op_host/op_tiling/arch22/x_all_gather_formulaic_tiling_a2a3.cpp
    mc2_common/op_host/op_tiling/hccl_formulaic_tiling.cpp
    mc2_common/op_host/op_tiling/matmul_formulaic_tiling.cpp
    mc2_common/op_host/op_tiling/hccl_performance.cpp
    mc2_common/op_host/op_tiling/matmul_performance.cpp
    mc2_common/op_host/op_tiling/mc2_tiling_utils.cpp
    mc2_common/utils/mc2_hcom_topo_info.cpp
)
target_include_directories(optiling PRIVATE ${_X_AGM_HOST_INC})

# ---- infershape -> opsproto --------------------------------------------------------------------------
# mc2_common_infershape.cpp defines AllGatherMatmulCommonInferShape(), which is called by
# x_all_gather_matmul_proto.cpp (this target). It MUST live in opsproto (the libcust_opsproto_rt
# .so) so the symbol is resolved where it is referenced — otherwise dlopen of the op_proto lib
# fails with "undefined symbol: ops::AllGatherMatmulCommonInferShape". No tiling source references
# any infershape symbol, so it is not needed in optiling. (ENABLE_BUILT_IN is undefined in x_ops,
# so the Mc2Hcom rank-size branch in infershape.cpp is compiled out — no dep on mc2_hcom_topo_info
# here.) optiling and opsproto share the same ${_X_AGM_HOST_INC} include dirs, so this compiles
# identically in either target.
target_sources(opsproto PRIVATE
    op_host/x_all_gather_matmul_proto.cpp
    mc2_common/op_host/mc2_common_infershape.cpp
)
target_include_directories(opsproto PRIVATE ${_X_AGM_HOST_INC})

# ---- public aclnn wrapper -> opapi ------------------------------------------------------------------
if (TARGET opapi)
    target_sources(opapi PRIVATE ${_OP_DIR}/op_host/op_api/aclnn_x_all_gather_matmul.cpp)
    target_include_directories(opapi PRIVATE ${_OP_DIR}/op_host/op_api ${_X_AGM_HOST_INC})
endif ()
