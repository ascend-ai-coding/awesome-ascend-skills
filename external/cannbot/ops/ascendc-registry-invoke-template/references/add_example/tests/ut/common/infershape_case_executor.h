/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H

#include "infershape_context_faker.h"

void ExecuteTestCase(gert::InfershapeContextPara&             infershapeContextPara, 
                     ge::graphStatus                          expectResult = ge::GRAPH_FAILED,
                     const std::vector<std::vector<int64_t>>& expectOutputShape = {});

#endif // OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H