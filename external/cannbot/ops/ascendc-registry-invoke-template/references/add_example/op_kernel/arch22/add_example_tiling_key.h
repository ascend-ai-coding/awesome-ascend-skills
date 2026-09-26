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
 * \file add_example_tiling_key.h
 * \brief Tiling 模板参数定义示例
 * 
 * 模板参数类型 (DECL定义 / SEL选择)：
 *   - DATATYPE: 原生数据类型 (C_DT_FLOAT, C_DT_FLOAT16...)
 *   - DTYPE:    自定义DataType
 *   - FORMAT:   数据格式 (C_FORMAT_ND, C_FORMAT_NCHW...)
 *   - UINT:     无符号整数 (RANGE/LIST/MIX)
 *   - BOOL:     布尔值 (0/1)
 *   - KERNEL_TYPE: 核类型 (AIV_ONLY, AIC_ONLY...)
 *   - DETERMINISTIC: 确定性计算 (仅SEL)
 * 
 * 参考：ascendc/host_api/tiling/template_argument.h
 */

#ifndef __ADD_EXAMPLE_TILING_KEY_H__
#define __ADD_EXAMPLE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(AddExample,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_INT32, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT32),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);

#endif
