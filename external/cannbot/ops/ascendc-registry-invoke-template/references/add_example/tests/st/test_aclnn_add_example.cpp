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
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_add_example.h"
#endif

// ============================================================================
// 宏定义
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

// ============================================================================
// CPU Golden 计算函数
// ============================================================================

template<typename T>
void ComputeGolden(const T* x1, const T* x2, T* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = x1[i] + x2[i];
    }
}

// ============================================================================
// 精度比对函数（CANN社区标准 MERE/MARE）
// ============================================================================
// dtype       Threshold            说明
// FLOAT32     2^-13  ≈ 1.22e-4
// FLOAT16     2^-10  ≈ 9.77e-4    (算子未支持，预留)
// BFLOAT16    2^-7   ≈ 7.81e-3    (算子未支持，预留)
//
// 通过条件: MERE < Threshold AND MARE < 10 * Threshold
// MERE = avg(|actual - golden| / (|golden| + 1e-7))
// MARE = max(|actual - golden| / (|golden| + 1e-7))
// ============================================================================

template<typename T>
bool CompareResults(const T* golden, const T* actual, size_t size,
                    double threshold = 1.220703125e-4) {
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }
    if (size == 0) return true;

    double mere = 0.0;
    double mare = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
        mere += rel_err;
        if (rel_err > mare) mare = rel_err;
    }
    mere /= static_cast<double>(size);
    double mare_threshold = 10.0 * threshold;

    bool pass = (mere < threshold) && (mare < mare_threshold);

    if (pass) {
        LOG_PRINT("  [PASS] MERE=%.2e, MARE=%.2e (threshold=%.2e, %zu elems)",
                  mere, mare, threshold, size);
    } else {
        LOG_PRINT("  [FAIL] MERE=%.2e (>= %.2e), MARE=%.2e (>= %.2e)",
                  mere, threshold, mare, mare_threshold);
        int shown = 0;
        for (size_t i = 0; i < size && shown < 5; ++i) {
            double g = static_cast<double>(golden[i]);
            double a = static_cast<double>(actual[i]);
            double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
            if (rel_err > threshold) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%.6f, 实际=%.6f, rel_err=%.2e",
                          i, g, a, rel_err);
                shown++;
            }
        }
    }
    return pass;
}

template<>
bool CompareResults<int32_t>(const int32_t* golden, const int32_t* actual, size_t size, double) {
    int mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (golden[i] != actual[i]) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%d, 实际=%d", i, golden[i], actual[i]);
            }
        }
    }
    
    if (mismatch == 0) {
        LOG_PRINT("  [PASS] 所有 %zu 个元素一致", size);
        return true;
    } else {
        LOG_PRINT("  [FAIL] 发现 %d 个不匹配", mismatch);
        return false;
    }
}

// ============================================================================
// CPU Golden 自测
// ============================================================================

void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测");
    LOG_PRINT("========================================");
    
    {
        float x1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float x2[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float output[5];
        float expected[] = {6.0f, 6.0f, 6.0f, 6.0f, 6.0f};
        
        ComputeGolden(x1, x2, output, 5);
        LOG_PRINT("\n测试 1: 普通浮点数加法");
        bool match = CompareResults(expected, output, 5);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }
    
    {
        float x1[] = {-1.0f, 2.5f, -3.7f, 0.0f, 100.5f};
        float x2[] = {1.0f, -2.5f, 3.7f, 0.0f, -100.5f};
        float output[5];
        float expected[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        ComputeGolden(x1, x2, output, 5);
        CompareResults(expected, output, 5);
    }
    
    {
        int32_t x1[] = {100, 200, -300, 0, 500};
        int32_t x2[] = {50, -100, 300, 0, -500};
        int32_t output[5];
        int32_t expected[] = {150, 100, 0, 0, 0};
        
        ComputeGolden(x1, x2, output, 5);
        LOG_PRINT("\n测试 3: INT32 整数加法");
        bool match = CompareResults(expected, output, 5);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }
    
    LOG_PRINT("\n========================================");
}

// ============================================================================
// Real 模式辅助函数
// ============================================================================

#ifndef USE_MOCK_ACLNN

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                   const std::vector<int64_t>& shape,
                   void** deviceAddr,
                   aclDataType dataType,
                   aclTensor** tensor) {
    size_t size = GetShapeSize(shape) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

#endif

// ============================================================================
// 统一测试执行器
// ============================================================================

#ifdef USE_MOCK_ACLNN

template<typename T>
bool RunTest(const std::vector<T>& x1, const std::vector<T>& x2, 
             const std::vector<int64_t>& shape, const char* dtype) {
    LOG_PRINT("\n[Mock] 测试 - dtype=%s, size=%zu", dtype, x1.size());
    
    if (x1.size() != x2.size()) {
        LOG_PRINT("  [FAIL] 输入大小不匹配: x1=%zu, x2=%zu", x1.size(), x2.size());
        return false;
    }
    
    size_t size = x1.size();
    std::vector<T> golden(size);
    std::vector<T> output(size);
    
    ComputeGolden(x1.data(), x2.data(), golden.data(), size);
    output = golden;
    
    return CompareResults(golden.data(), output.data(), size);
}

#else

template<typename T>
bool RunTest(const std::vector<T>& x1_data, const std::vector<T>& x2_data,
             const std::vector<int64_t>& shape, aclDataType input_dtype, aclrtStream stream) {
    LOG_PRINT("\n[Real] 测试 - size=%zu", x1_data.size());

    if (x1_data.size() != x2_data.size()) {
        LOG_PRINT("  [FAIL] 输入大小不匹配");
        return false;
    }

    size_t size = GetShapeSize(shape);

    void* x1_dev = nullptr;
    void* x2_dev = nullptr;
    void* output_dev = nullptr;
    aclTensor* x1_tensor = nullptr;
    aclTensor* x2_tensor = nullptr;
    aclTensor* output_tensor = nullptr;

    if (CreateAclTensor(x1_data, shape, &x1_dev, input_dtype, &x1_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  创建x1 tensor失败");
        return false;
    }

    if (CreateAclTensor(x2_data, shape, &x2_dev, input_dtype, &x2_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  创建x2 tensor失败");
        aclDestroyTensor(x1_tensor);
        aclrtFree(x1_dev);
        return false;
    }

    std::vector<T> output_host(size, 0);
    if (CreateAclTensor(output_host, shape, &output_dev, input_dtype, &output_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  创建输出tensor失败");
        aclDestroyTensor(x1_tensor);
        aclDestroyTensor(x2_tensor);
        aclrtFree(x1_dev);
        aclrtFree(x2_dev);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnAddExampleGetWorkspaceSize(x1_tensor, x2_tensor, output_tensor,
                                               &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize失败: %d", ret);
        aclDestroyTensor(x1_tensor);
        aclDestroyTensor(x2_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(x1_dev);
        aclrtFree(x2_dev);
        aclrtFree(output_dev);
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  workspace分配失败: %d (size=%lu)", ret, workspaceSize);
            aclDestroyTensor(x1_tensor);
            aclDestroyTensor(x2_tensor);
            aclDestroyTensor(output_tensor);
            aclrtFree(x1_dev);
            aclrtFree(x2_dev);
            aclrtFree(output_dev);
            return false;
        }
    }

    ret = aclnnAddExample(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddExample失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(x1_tensor);
        aclDestroyTensor(x2_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(x1_dev);
        aclrtFree(x2_dev);
        aclrtFree(output_dev);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  流同步失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(x1_tensor);
        aclDestroyTensor(x2_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(x1_dev);
        aclrtFree(x2_dev);
        aclrtFree(output_dev);
        return false;
    }

    std::vector<T> npu_output(size);
    ret = aclrtMemcpy(npu_output.data(), size * sizeof(T), output_dev, size * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  D2H数据拷贝失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(x1_tensor);
        aclDestroyTensor(x2_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(x1_dev);
        aclrtFree(x2_dev);
        aclrtFree(output_dev);
        return false;
    }

    std::vector<T> golden(size);
    ComputeGolden(x1_data.data(), x2_data.data(), golden.data(), size);

    bool passed = CompareResults(golden.data(), npu_output.data(), size);

    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(x1_tensor);
    aclDestroyTensor(x2_tensor);
    aclDestroyTensor(output_tensor);
    aclrtFree(x1_dev);
    aclrtFree(x2_dev);
    aclrtFree(output_dev);

    return passed;
}

#endif

// ============================================================================
// 测试用例定义
// ============================================================================

struct TestCase {
    const char* name;
    std::vector<int64_t> shape;
    std::function<std::pair<std::vector<float>, std::vector<float>>()> data_float;
    std::function<std::pair<std::vector<int32_t>, std::vector<int32_t>>()> data_int32;
    bool is_float;
};

std::vector<TestCase> GetTestCases() {
    return {
        {"FP32 基础加法", {2, 3},
         []() { return std::make_pair(
             std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
             std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}); },
         nullptr, true},
        
        {"FP32 正负数混合", {6},
         []() { return std::make_pair(
             std::vector<float>{-1.5f, 2.5f, -3.7f, 0.0f, 100.5f, -200.8f},
             std::vector<float>{1.5f, -2.5f, 3.7f, 0.0f, -100.5f, 200.8f}); },
         nullptr, true},
        
        {"FP32 大 shape", {32},
         []() { return std::make_pair(
             std::vector<float>(32, 1.5f),
             std::vector<float>(32, 2.5f)); },
         nullptr, true},
        
        {"FP32 多维张量", {2, 3, 4},
         []() { return std::make_pair(
             std::vector<float>(24, 1.0f),
             std::vector<float>(24, 2.0f)); },
         nullptr, true},
        
        {"INT32 基础加法", {2, 3},
         nullptr,
         []() { return std::make_pair(
             std::vector<int32_t>{10, 20, 30, 40, 50, 60},
             std::vector<int32_t>{5, 10, 15, 20, 25, 30}); },
         false},
        
        {"INT32 正负数混合", {6},
         nullptr,
         []() { return std::make_pair(
             std::vector<int32_t>{-100, 200, -300, 0, 500, -600},
             std::vector<int32_t>{100, -200, 300, 0, -500, 600}); },
         false},
        
        {"INT32 大 shape", {64},
         nullptr,
         []() { return std::make_pair(
             std::vector<int32_t>(64, 100),
             std::vector<int32_t>(64, 200)); },
         false},
        
        {"边界条件 - 单个元素", {1},
         []() { return std::make_pair(
             std::vector<float>{123.456f},
             std::vector<float>{876.544f}); },
         nullptr, true},
        
        {"FP32 极小值和极大值", {6},
         []() { return std::make_pair(
             std::vector<float>{std::numeric_limits<float>::min(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::epsilon(),
                               0.0f, 0.0f, 0.0f},
             std::vector<float>(6, 0.0f)); },
         nullptr, true},
        
        {"FP32 零值测试", {4},
         []() { return std::make_pair(
             std::vector<float>{0.0f, -0.0f, 1.5f, -2.5f},
             std::vector<float>(4, 0.0f)); },
         nullptr, true},
    };
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("add_example 算子 ST 测试");
    LOG_PRINT("========================================");
    
#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden)");
#else
    LOG_PRINT("模式: Real (NPU)");
#endif
    
    TestGoldenCorrectness();
    
    int passed = 0, failed = 0;
    
#ifndef USE_MOCK_ACLNN
    int32_t deviceId = 0;
    aclrtStream stream;

    auto initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit失败: %d", initRet);
        return 1;
    }
    initRet = aclrtSetDevice(deviceId);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice(%d)失败: %d", deviceId, initRet);
        aclFinalize();
        return 1;
    }
    initRet = aclrtCreateStream(&stream);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream失败: %d", initRet);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
#endif
    
    LOG_PRINT("\n执行内置测试用例...");
    
    auto test_cases = GetTestCases();
    
    for (const auto& tc : test_cases) {
        LOG_PRINT("\n测试: %s", tc.name);
        
        if (tc.is_float && tc.data_float) {
            auto [x1, x2] = tc.data_float();
#ifdef USE_MOCK_ACLNN
            if (RunTest(x1, x2, tc.shape, "float32")) passed++; else failed++;
#else
            if (RunTest(x1, x2, tc.shape, ACL_FLOAT, stream)) passed++; else failed++;
#endif
        } else if (!tc.is_float && tc.data_int32) {
            auto [x1, x2] = tc.data_int32();
#ifdef USE_MOCK_ACLNN
            if (RunTest(x1, x2, tc.shape, "int32")) passed++; else failed++;
#else
            if (RunTest(x1, x2, tc.shape, ACL_INT32, stream)) passed++; else failed++;
#endif
        }
    }
    
#ifndef USE_MOCK_ACLNN
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif
    
    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("========================================\n");
    
    return failed == 0 ? 0 : 1;
}
