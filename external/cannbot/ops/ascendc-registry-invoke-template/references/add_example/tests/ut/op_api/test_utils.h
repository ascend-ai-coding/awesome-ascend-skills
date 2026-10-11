/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_utils.h
 * @brief op_api 测试辅助工具
 * 
 * @example
 *   // 创建普通 tensor
 *   auto tensor = TestTensorFactory::CreateTensor({32, 4}, op::DataType::DT_FLOAT);
 *   
 *   // 创建空 tensor
 *   auto emptyTensor = TestTensorFactory::CreateEmptyTensor({0, 4}, op::DataType::DT_FLOAT);
 *   
 *   // RAII 自动管理
 *   TestTensorGuard x1(TestTensorFactory::CreateTensor({32, 4}, DataType::DT_FLOAT));
 *   
 * @note
 *   - ABI 兼容性：使用 _GLIBCXX_USE_CXX11_ABI=0
 *   - 平台依赖：DT_FLOAT16 仅在 Ascend910B 支持
 *   - 执行测试：aclnnAddExample() 需真实 NPU，属 ST 测试范围
 */

#ifndef OP_API_TEST_UTILS_H_
#define OP_API_TEST_UTILS_H_

#include <vector>
#include <cstring>
#include <map>
#include <mutex>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"

namespace op_api_test {

/**
 * @brief Helper class to create test aclTensor objects
 * 
 * This class provides utilities for creating aclTensor objects in unit tests.
 * Uses CANN's public API (aclCreateTensor) for tensor creation.
 * 
 * Note: This class manages tensor data memory internally. The memory will be
 * freed when DestroyTensor() is called.
 */
class TestTensorFactory {
public:
    /**
     * @brief Create a test tensor with given shape and dtype
     * @param shape Tensor shape
     * @param dtype Data type
     * @param tensorData Optional tensor data (if nullptr, will allocate memory)
     * @return aclTensor* Created tensor, or nullptr on failure
     */
    static aclTensor* CreateTensor(const std::vector<int64_t>& shape,
                                   op::DataType dtype,
                                   void* tensorData = nullptr) {
        // Calculate strides (row-major order)
        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        // Allocate memory if not provided
        void* data = tensorData;
        bool ownsMemory = false;
        if (data == nullptr) {
            size_t totalElements = 1;
            for (auto dim : shape) {
                totalElements *= dim;
            }
            size_t dataSize = totalElements * GetDataTypeSize(dtype);
            data = new uint8_t[dataSize]();  // Allocate with value-initialization (zeros)
            ownsMemory = true;
        }
        
        // Convert op::DataType to aclDataType
        aclDataType aclDtype = ConvertDataType(dtype);
        
        // Create tensor using CANN public API
        aclTensor* tensor = aclCreateTensor(
            shape.data(),           // viewDims
            shape.size(),           // viewDimsNum
            aclDtype,               // dataType
            strides.data(),         // stride
            0,                      // offset
            ACL_FORMAT_ND,          // format
            nullptr,                // storageDims (nullptr for ND format)
            0,                      // storageDimsNum
            data                    // tensorData
        );
        
        // Handle memory ownership
        if (ownsMemory) {
            if (tensor != nullptr) {
                std::lock_guard<std::mutex> lock(g_tensorDataMutex);
                g_tensorDataMap[tensor] = data;
            } else {
                // aclCreateTensor failed, free allocated memory to avoid leak
                delete[] static_cast<uint8_t*>(data);
            }
        }
        
        return tensor;
    }
    
    /**
     * @brief Create an empty tensor (shape with 0)
     * @param shape Tensor shape (should contain at least one 0)
     * @param dtype Data type
     * @return aclTensor* Created empty tensor, or nullptr on failure
     */
    static aclTensor* CreateEmptyTensor(const std::vector<int64_t>& shape,
                                       op::DataType dtype) {
        // Empty tensors have nullptr data
        std::vector<int64_t> strides(shape.size(), 0);
        
        aclDataType aclDtype = ConvertDataType(dtype);
        
        aclTensor* tensor = aclCreateTensor(
            shape.data(),
            shape.size(),
            aclDtype,
            strides.data(),
            0,
            ACL_FORMAT_ND,
            nullptr,
            0,
            nullptr  // Empty tensor has no data
        );
        
        return tensor;
    }
    
    /**
     * @brief Destroy a test tensor
     * @param tensor Tensor to destroy
     */
    static void DestroyTensor(aclTensor* tensor) {
        if (tensor != nullptr) {
            // Free associated memory if we own it
            {
                std::lock_guard<std::mutex> lock(g_tensorDataMutex);
                auto it = g_tensorDataMap.find(tensor);
                if (it != g_tensorDataMap.end()) {
                    delete[] static_cast<uint8_t*>(it->second);
                    g_tensorDataMap.erase(it);
                }
            }
            aclDestroyTensor(tensor);
        }
    }

private:
    // Global map to track tensor data memory ownership
    static std::map<aclTensor*, void*> g_tensorDataMap;
    static std::mutex g_tensorDataMutex;

    /**
     * @brief Convert op::DataType to aclDataType
     */
    static aclDataType ConvertDataType(op::DataType dtype) {
        switch (dtype) {
            case op::DataType::DT_FLOAT:
                return ACL_FLOAT;
            case op::DataType::DT_FLOAT16:
                return ACL_FLOAT16;
            case op::DataType::DT_INT32:
                return ACL_INT32;
            case op::DataType::DT_INT64:
                return ACL_INT64;
            case op::DataType::DT_INT8:
                return ACL_INT8;
            case op::DataType::DT_UINT8:
                return ACL_UINT8;
            case op::DataType::DT_BOOL:
                return ACL_BOOL;
            default:
                return ACL_DT_UNDEFINED;
        }
    }
    
    /**
     * @brief Get size in bytes for a data type
     */
    static size_t GetDataTypeSize(op::DataType dtype) {
        switch (dtype) {
            case op::DataType::DT_FLOAT:
                return sizeof(float);
            case op::DataType::DT_FLOAT16:
                return sizeof(int16_t);
            case op::DataType::DT_INT32:
                return sizeof(int32_t);
            case op::DataType::DT_INT64:
                return sizeof(int64_t);
            case op::DataType::DT_INT8:
                return sizeof(int8_t);
            case op::DataType::DT_UINT8:
                return sizeof(uint8_t);
            case op::DataType::DT_BOOL:
                return sizeof(bool);
            default:
                return 0;
        }
    }
};

// Static member definitions
inline std::map<aclTensor*, void*> TestTensorFactory::g_tensorDataMap;
inline std::mutex TestTensorFactory::g_tensorDataMutex;

/**
 * @brief RAII wrapper for test tensors
 */
class TestTensorGuard {
public:
    explicit TestTensorGuard(aclTensor* tensor) : tensor_(tensor) {}
    ~TestTensorGuard() {
        TestTensorFactory::DestroyTensor(tensor_);
    }
    
    aclTensor* Get() const { return tensor_; }
    aclTensor* operator->() const { return tensor_; }
    
private:
    aclTensor* tensor_;
    TestTensorGuard(const TestTensorGuard&) = delete;
    TestTensorGuard& operator=(const TestTensorGuard&) = delete;
};

} // namespace op_api_test

#endif // OP_API_TEST_UTILS_H_
