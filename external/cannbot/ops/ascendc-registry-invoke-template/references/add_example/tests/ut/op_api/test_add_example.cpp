/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "add_example.h"
#include "test_utils.h"

using namespace std;
using namespace op;
using namespace op_api_test;

class AddExampleL0Test : public testing::Test {
protected:
    void TearDown() override {}
};

#if GTEST_HAS_DEATH_TEST
TEST_F(AddExampleL0Test, NullptrX1Failure) {
    auto executor = UniqueExecutor(__func__);
    ASSERT_NE(executor.get(), nullptr);
    
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    ASSERT_NE(x2, nullptr);
    
    EXPECT_DEATH(l0op::AddExample(nullptr, x2, executor.get()), ".*");
    
    TestTensorFactory::DestroyTensor(x2);
}

TEST_F(AddExampleL0Test, NullptrX2Failure) {
    auto executor = UniqueExecutor(__func__);
    ASSERT_NE(executor.get(), nullptr);
    
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    ASSERT_NE(x1, nullptr);
    
    EXPECT_DEATH(l0op::AddExample(x1, nullptr, executor.get()), ".*");
    
    TestTensorFactory::DestroyTensor(x1);
}

TEST_F(AddExampleL0Test, NullptrExecutorFailure) {
    auto x1 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    auto x2 = TestTensorFactory::CreateTensor({32, 4, 4, 4}, DataType::DT_FLOAT);
    
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    
    EXPECT_DEATH(l0op::AddExample(x1, x2, nullptr), ".*");
    
    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(x2);
}
#else
TEST_F(AddExampleL0Test, NullptrTestsSkipped) {
    GTEST_SKIP() << "Death tests not supported in this environment";
}
#endif
