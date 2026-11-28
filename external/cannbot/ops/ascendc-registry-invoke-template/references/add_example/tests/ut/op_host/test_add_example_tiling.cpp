/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "add_example_tiling_data.h"

namespace AddExampleUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "AddExample";
struct AddExampleTestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static AddExampleTestParam testCases[] = {
    {"add_example_0", {32, 4, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {32, 4, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {32, 4, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 256UL, "2048 32 10912 ", {0}, 64, 262144, 4096},
    {"add_example_1", {32, 4, 4, 4}, ge::DT_INT32, ge::FORMAT_ND, {32, 4, 4, 4}, ge::DT_INT32, ge::FORMAT_ND, {32, 4, 4, 4}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 259UL, "2048 32 10912 ", {0}, 64, 262144, 4096},
};

class AddExampleTilingTest : public testing::TestWithParam<AddExampleTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddExampleTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddExampleTilingTest TearDown." << std::endl;
    }
};

struct AddExampleCompileInfo {
} compileInfo;

static void TestOneParamCase(const AddExampleTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
         {x2Shape, param.x2Dtype, param.x2Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(AddExampleTilingTest, tiling_test)
{
    const AddExampleTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    AddExampleTilingTests,
    AddExampleTilingTest,
    testing::ValuesIn(testCases));

}
