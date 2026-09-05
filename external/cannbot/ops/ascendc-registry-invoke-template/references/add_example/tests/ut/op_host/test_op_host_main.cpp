/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

// 参考 ops-math/tests/ut/op_host/test_op_host_main.cpp
class OpHostUtEnvironment : public testing::Environment {
public:
    OpHostUtEnvironment(char** argv) : argv_(argv)
    {}
    
    virtual void SetUp() {
        cout << "Global Environment SetUp." << endl;

        // 获取可执行文件路径
        std::filesystem::path currDir = argv_[0];
        if (currDir.is_relative()) {
            currDir = std::filesystem::weakly_canonical(std::filesystem::current_path() / currDir);
        } else {
            currDir = std::filesystem::canonical(currDir);
        }

        // 加载 libadd_example_op_host_ut_lib.so（包含算子注册代码）
        string opHostSoPath = currDir.parent_path().string() + string("/libadd_example_op_host_ut_lib.so");
        cout << "Loading op_host .so from: " << opHostSoPath << endl;
        
        // 创建 OppSoDesc 并加载 .so
        gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");
        
        // 创建 SpaceRegistry 并添加 .so 中的注册信息
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cerr << "Failed to add .so to registry." << endl;
            return;
        }

        // 设置到全局 DefaultOpImplSpaceRegistryV2
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
        cout << "OpImplSpaceRegistryV2 initialized successfully" << endl;
    }

    virtual void TearDown() {
        cout << "Global Environment TearDown" << endl;
    }

private:
    char** argv_;
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpHostUtEnvironment(argv));
    return RUN_ALL_TESTS();
}
