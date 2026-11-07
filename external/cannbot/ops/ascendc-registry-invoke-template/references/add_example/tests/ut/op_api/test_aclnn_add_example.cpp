/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "opdev/platform.h"
#include "aclnn_add_example.h"
#include "test_utils.h"

using namespace std;
using namespace op;
using namespace op_api_test;

class AclnnAddExampleTest : public testing::Test {
protected:
    void TearDown() override {}
};

TEST_F(AclnnAddExampleTest, NullptrX1Failure) {
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto out = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnAddExampleGetWorkspaceSize(nullptr, x2, out, &workspaceSize, &executor);
    
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    
    TestTensorFactory::DestroyTensor(x2);
    TestTensorFactory::DestroyTensor(out);
}

TEST_F(AclnnAddExampleTest, NullptrX2Failure) {
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto out = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(out, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnAddExampleGetWorkspaceSize(x1, nullptr, out, &workspaceSize, &executor);
    
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    
    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(out);
}

TEST_F(AclnnAddExampleTest, NullptrOutFailure) {
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnAddExampleGetWorkspaceSize(x1, x2, nullptr, &workspaceSize, &executor);
    
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    
    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(x2);
}

TEST_F(AclnnAddExampleTest, DtypeMismatchFailure) {
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_INT32);
    auto out = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnAddExampleGetWorkspaceSize(x1, x2, out, &workspaceSize, &executor);
    
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    
    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(x2);
    TestTensorFactory::DestroyTensor(out);
}

TEST_F(AclnnAddExampleTest, OutputDtypeMismatchFailure) {
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto out = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_INT32);
    
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    auto ret = aclnnAddExampleGetWorkspaceSize(x1, x2, out, &workspaceSize, &executor);
    
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    
    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(x2);
    TestTensorFactory::DestroyTensor(out);
}
