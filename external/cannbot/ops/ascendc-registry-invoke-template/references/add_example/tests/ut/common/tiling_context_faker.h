/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H

#include "op_tiling_context_builder.h"
#include "any_value.h"

namespace gert {

class TilingContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape& shape, 
                          ge::DataType dtype, 
                          ge::Format format, 
                          bool isConst = false, 
                          void* constValue = nullptr) :
            shape_(shape), dtype_(dtype), format_(format), isConst_(isConst), constValue_(constValue) {}
    public:
        gert::StorageShape shape_;
        ge::DataType dtype_ = ge::DT_FLOAT;
        ge::Format format_ = ge::FORMAT_ND;
        bool isConst_ = false;
        void* constValue_ = nullptr;
    };

    class OpAttr {
    public:
        OpAttr(const std::string& attrName, const Ops::Math::AnyValue& attr) : attrName_(attrName), attr_(attr) {}
    public:
        std::string attrName_;
        Ops::Math::AnyValue attr_;
    };
public:
    TilingContextPara(const std::string& opName,
                       const std::vector<TensorDescription>& inputTensorDesc,
                       const std::vector<TensorDescription>& outputTensorDesc,
                       const std::vector<OpAttr>& attrs,
                       void* compileInfo = nullptr,
                       uint64_t coreNum = 64,
                       uint64_t ubSize = 262144,
                       uint64_t tilingDataSize = 4096) : 
                       opName_(opName),
                       inputInstanceNum_(),
                       outputInstanceNum_(),
                       inputTensorDesc_(inputTensorDesc),
                       outputTensorDesc_(outputTensorDesc),
                       attrs_(attrs),
                       coreNum_(coreNum),
                       ubSize_(ubSize),
                       tilingDataSize_(tilingDataSize),
                       compileInfo_(compileInfo) {}

    TilingContextPara(const std::string& opName,
                       const std::vector<TensorDescription>& inputTensorDesc,
                       const std::vector<TensorDescription>& outputTensorDesc,
                       void* compileInfo = nullptr,
                       uint64_t coreNum = 64,
                       uint64_t ubSize = 262144,
                       uint64_t tilingDataSize = 4096) : 
                       opName_(opName),
                       inputInstanceNum_(),
                       outputInstanceNum_(),
                       inputTensorDesc_(inputTensorDesc),
                       outputTensorDesc_(outputTensorDesc),
                       attrs_(),
                       coreNum_(coreNum),
                       ubSize_(ubSize),
                       tilingDataSize_(tilingDataSize),
                       compileInfo_(compileInfo) {}

    TilingContextPara(const std::string& opName,
                       const std::vector<TensorDescription>& inputTensorDesc,
                       const std::vector<TensorDescription>& outputTensorDesc,
                       const std::vector<OpAttr>& attrs,
                       const std::vector<uint32_t>& inputInstanceNum,
                       const std::vector<uint32_t>& outputInstanceNum,
                       void* compileInfo = nullptr,
                       uint64_t coreNum = 64,
                       uint64_t ubSize = 262144,
                       uint64_t tilingDataSize = 4096) : 
                       opName_(opName),
                       inputInstanceNum_(inputInstanceNum),
                       outputInstanceNum_(outputInstanceNum),
                       inputTensorDesc_(inputTensorDesc),
                       outputTensorDesc_(outputTensorDesc),
                       attrs_(attrs),
                       coreNum_(coreNum),
                       ubSize_(ubSize),
                       tilingDataSize_(tilingDataSize),
                       compileInfo_(compileInfo) {}

    TilingContextPara(const std::string& opName,
                       const std::vector<TensorDescription>& inputTensorDesc,
                       const std::vector<TensorDescription>& outputTensorDesc,
                       const std::vector<uint32_t>& inputInstanceNum,
                       const std::vector<uint32_t>& outputInstanceNum,
                       void* compileInfo = nullptr,
                       uint64_t coreNum = 64,
                       uint64_t ubSize = 262144,
                       uint64_t tilingDataSize = 4096) : 
                       opName_(opName),
                       inputInstanceNum_(inputInstanceNum),
                       outputInstanceNum_(outputInstanceNum),
                       inputTensorDesc_(inputTensorDesc),
                       outputTensorDesc_(outputTensorDesc),
                       attrs_(),
                       coreNum_(coreNum),
                       ubSize_(ubSize),
                       tilingDataSize_(tilingDataSize),
                       compileInfo_(compileInfo) {}
public:
    std::string opName_;
    std::vector<uint32_t> inputInstanceNum_;
    std::vector<uint32_t> outputInstanceNum_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    std::vector<OpAttr> attrs_;
    uint64_t coreNum_        = 64;
    uint64_t ubSize_         = 262144;
    uint64_t tilingDataSize_ = 4096;
    void* compileInfo_ = nullptr;
};

class TilingContextFaker : public OpTilingContextBuilder {
public:
    TilingContextFaker& SetOpType(const std::string opType);

    /* only one can be choosed from IrInstanceNum */
    TilingContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    /* can be used for dynamic inputs/outputs
     * only one can be choosed from NodeIoNum */
    TilingContextFaker& IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                      const std::vector<uint32_t>& outputInstanceNum);



    TilingContextFaker& Attr(const std::string& attrName, bool attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, int64_t attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, float attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const ge::AscendString& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<bool>& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<int64_t>& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<float>& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<ge::AscendString>& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attr) {
        OpTilingContextBuilder::AppendAttr(attr);
        return *this;
    }

    TilingContextFaker& InputTensors(const std::vector<Tensor *>& inputTensors);

    TilingContextFaker& OutputTensors(const std::vector<Tensor *>& outputTensors);

    TilingContextFaker& CompileInfo(const void* compileInfo);

    TilingContextFaker& PlatformInfo(const void* platformInfo);

    TilingContextFaker& DeterministicInfo(int32_t* deterministicInfo);

    TilingContextFaker& TilingData(const void* tilingData);

    TilingContextFaker& Workspace(const ContinuousVector* workspace);

    ContextHolder<TilingContext> Build();
};

} // namespace gert
#endif // OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
