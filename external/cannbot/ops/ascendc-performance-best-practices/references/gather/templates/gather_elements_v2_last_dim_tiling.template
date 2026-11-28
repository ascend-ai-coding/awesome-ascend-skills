/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_v2_last_dim_tiling.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_LAST_DIM_TILING_H
#define GATHER_ELEMENTS_V2_LAST_DIM_TILING_H
#include "register/tilingdata_base.h"
#include "gather_elements_v2_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t RESERVED_UB = 10 * 1024;
constexpr int64_t NUM_TEN = 10;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t INT8_DSIZE = 1;
constexpr int64_t INT6_DSIZE = 2;
constexpr int64_t INT32_DSIZE = 4;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t MAX_X_SIZE = 1024;
constexpr int64_t BLOCK_SIZE_T = 32;
constexpr int64_t TILING_KEY_2 = 2;
constexpr int64_t MAX_SLICE_NUM = 5;
constexpr int64_t MAX_SIZE_RATIO = 256;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t INDEX_INDEX = 1;
constexpr uint32_t DOUBLE_TIME = 2;
const int32_t DIVISION_BY_ZERO_LOG = -100000;
constexpr float MASK_SIZE_RATE = 1.0f / 4;

class GatherElementsV2LastDimTiling {
public:
    explicit GatherElementsV2LastDimTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus Init();
    ge::graphStatus MergeAxis(const gert::StorageShape* xShape, const gert::StorageShape* indexShape);
    void SetShape();
    void GetDSize();
    void IfEnableBatch();
    void DoUBSlice();
    void DoNeedUseCore();
    void SetTilingData();
    void DoScalarMode();
    void PrintTilingData() const;
    ge::graphStatus DoOpTiling();

private:
    GatherElementsV2TilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    gert::Shape xShape_;
    gert::Shape indexShape_;

    int64_t totalCoreNum_ = 0;
    int64_t ubSizePlatForm_ = 0;

    int64_t dimNum_ = 0;
    int64_t nonCollectingAxisSzie_ = 1;
    int64_t xDSize_ = 0;
    int64_t xRealDsize_ = 0;
    int64_t indexDSize_ = 0;
    int64_t indexRealDsize_ = 0;
    int64_t tilingKey_ = 0;
    int64_t xAxisSize_ = 0;
    int64_t indexAxisSize_ = 0;
    int64_t eachCalculationLines_ = 0;
    int64_t xSliceNum_ = 0;
    int64_t indexSliceNum_ = 0;
    int64_t xAlignSize_ = 0;
    int64_t indexAlignSize_ = 0;
    int64_t yAlignSize_ = 0;
    int64_t reservedXSize_ = 0;
    int64_t reservedIndexSize_ = 0;
    int64_t needUsedCore_ = 0;
    int64_t formerCoreRowNum_ = 0;
    int64_t formerCoreNum_ = 0;
    int64_t xBufferSize_ = 0;
    int64_t indexBufferSize_ = 0;
    int64_t yBufferSize_ = 0;
    int64_t maskBufferSize_ = 0;
    int64_t specialDataMove_ = 0;
    int64_t xDsizeRatio_ = 1;
    int64_t scalarModeLength_ = 0;
    bool batchProcess_ = false;
};

ge::graphStatus GatherElementsV2LastDimTiling::Init()
{
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* indexShape = context_->GetInputShape(INDEX_INDEX);

    auto compileInfo = static_cast<const GatherElementsV2CompileInfo*>(context_->GetCompileInfo());
    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSizePlatForm_ = static_cast<int64_t>(compileInfo->ubSizePlatForm - RESERVED_UB);

    MergeAxis(xShape, indexShape);
    DoOpTiling();

    size_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsV2LastDimTiling::MergeAxis(const gert::StorageShape* xShape,
                                                         const gert::StorageShape* indexShape)
{
    gert::Shape xOriginShape = xShape->GetStorageShape();
    gert::Shape indexOriginShape = indexShape->GetStorageShape();
    auto xDimNum = static_cast<int>(xOriginShape.GetDimNum());
    auto indexDimNum = static_cast<int>(indexOriginShape.GetDimNum());

    OP_CHECK_IF(xDimNum != indexDimNum,
                OP_LOGE(context_, "The dim %d of x and the dim %d of index are not equal.", xDimNum, indexDimNum),
                return ge::GRAPH_FAILED);

    int i = 0;
    // 合轴，将相邻且相同大小的轴和为一个轴。
    while (i < xDimNum) {
        if (xOriginShape[i] != indexOriginShape[i] || i == xDimNum - 1) {
            xShape_.AppendDim(xOriginShape[i]);
            indexShape_.AppendDim(indexOriginShape[i]);
        } else {
            int j = i;
            while (j < xDimNum && xOriginShape[j] == indexOriginShape[j] && j != xDimNum - 1) {
                j++;
            }
            if (j - i > 1) {
                int64_t val = 1;
                for (int k = i; k < j; k++) {
                    val *= xOriginShape[k];
                }
                xShape_.AppendDim(val);
                indexShape_.AppendDim(val);
            } else {
                xShape_.AppendDim(xOriginShape[i]);
                indexShape_.AppendDim(indexOriginShape[i]);
            }
            i = j - 1;
        }
        i++;
    }
    dimNum_ = static_cast<int64_t>(xShape_.GetDimNum());
    tilingData_.lastDimTiling.set_dimNum(dimNum_);
    SetShape();
    return ge::GRAPH_SUCCESS;
}

void GatherElementsV2LastDimTiling::SetShape()
{
    int64_t xShapeArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexShapeArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStrideArray[TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t indexStrideArray[TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < dimNum_; i++) {
        xShapeArray[i] = xShape_.GetDim(i);
        indexShapeArray[i] = indexShape_.GetDim(i);
        if (i < dimNum_ - 1) {
            nonCollectingAxisSzie_ *= xShapeArray[i];
            if (i != 0 && xShapeArray[i] != indexShapeArray[i]) {
                specialDataMove_ = 1;
            }
        } else {
            xAxisSize_ = xShapeArray[i];
            indexAxisSize_ = indexShapeArray[i];
        }
    }
    for (int i = dimNum_ - 1; i > 0; i--) {
        xStrideArray[i - 1] = xStrideArray[i] * xShapeArray[i];
        indexStrideArray[i - 1] = indexStrideArray[i] * indexShapeArray[i];
    }
    tilingData_.lastDimTiling.set_xShape(xShapeArray);
    tilingData_.lastDimTiling.set_indexShape(indexShapeArray);
    tilingData_.lastDimTiling.set_xStrideArray(xStrideArray);
    tilingData_.lastDimTiling.set_indexStrideArray(indexStrideArray);
}

void GatherElementsV2LastDimTiling::GetDSize()
{
    ge::DataType xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType indexDataType = context_->GetInputDesc(INDEX_INDEX)->GetDataType();
    xDSize_ = ge::GetSizeByDataType(xDataType);
    xRealDsize_ = xDSize_;
    indexDSize_ = ge::GetSizeByDataType(indexDataType);
    indexRealDsize_ = indexDSize_;
    context_->SetTilingKey(TILING_KEY_2);
    if (xDSize_ == INT8_DSIZE) {
        xDSize_ = INT6_DSIZE;
        xDsizeRatio_ = DOUBLE_TIME;
    }
    if (indexDSize_ == INT32_DSIZE) {
        indexDSize_ = INT64_DSIZE;
    }
}

void GatherElementsV2LastDimTiling::IfEnableBatch()
{
    xAlignSize_ = Ops::Base::CeilAlign(xAxisSize_ * xDSize_, BLOCK_SIZE_T * xDsizeRatio_);
    yAlignSize_ = Ops::Base::CeilAlign(indexAxisSize_ * xDSize_, BLOCK_SIZE_T * xDsizeRatio_);
    indexAlignSize_ = Ops::Base::CeilAlign(indexAxisSize_ * indexDSize_, BLOCK_SIZE_T * DOUBLE_TIME);
    batchProcess_ = xAlignSize_ + yAlignSize_ + indexAlignSize_ <= ubSizePlatForm_ / DOUBLE_TIME;
}

void GatherElementsV2LastDimTiling::DoUBSlice()
{
    if (indexShape_[dimNum_ - 1] == 1 && batchProcess_ && specialDataMove_ == 0) {
        tilingData_.lastDimTiling.set_indexAxisSizeEqualOne(NUM_ONE);
        int64_t perGroupSize = xAxisSize_ * xDSize_ + indexAxisSize_ * indexDSize_ + indexAxisSize_ * indexDSize_;
        eachCalculationLines_ = Ops::Base::FloorDiv(ubSizePlatForm_, perGroupSize);
        xBufferSize_ = Ops::Base::CeilAlign(xAxisSize_ * xDSize_ * eachCalculationLines_, BLOCK_SIZE_T * xDsizeRatio_);
        indexBufferSize_ = Ops::Base::CeilAlign(indexAxisSize_ * indexDSize_ * eachCalculationLines_,
                                                BLOCK_SIZE_T * DOUBLE_TIME);
        yBufferSize_ = Ops::Base::CeilAlign(indexAxisSize_ * indexDSize_ * eachCalculationLines_,
                                            BLOCK_SIZE_T * xDsizeRatio_);
    } else if (batchProcess_) {
        int64_t perGroupSize = xAlignSize_ + indexAlignSize_ + yAlignSize_;
        eachCalculationLines_ = Ops::Base::FloorDiv(ubSizePlatForm_, perGroupSize);
        xBufferSize_ = xAlignSize_ * eachCalculationLines_;
        yBufferSize_ = yAlignSize_ * eachCalculationLines_;
        indexBufferSize_ = indexAlignSize_ * eachCalculationLines_;
        if (Ops::Base::CeilDiv(indexAxisSize_ * indexDSize_, BLOCK_SIZE_T) % DOUBLE_TIME == 1 &&
            indexRealDsize_ == INT64_DSIZE) {
            tilingData_.lastDimTiling.set_dataMoveUBStride(NUM_ONE);
        }
    } else if (xAlignSize_ <= ubSizePlatForm_ / DOUBLE_TIME) {
        xSliceNum_ = NUM_ONE;
        eachCalculationLines_ = NUM_ONE;
        xBufferSize_ = xAlignSize_;
        indexBufferSize_ = (ubSizePlatForm_ - xBufferSize_) / (indexDSize_ + xDSize_) * indexDSize_;
        indexBufferSize_ = Ops::Base::CeilAlign(indexBufferSize_, BLOCK_SIZE_T * DOUBLE_TIME);
        yBufferSize_ = indexBufferSize_ / indexDSize_ * xDSize_;
        yBufferSize_ = Ops::Base::CeilAlign(yBufferSize_, BLOCK_SIZE_T);
        indexSliceNum_ = Ops::Base::CeilDiv(indexAlignSize_, indexBufferSize_);
        reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
    } else {
        eachCalculationLines_ = NUM_ONE;
        xBufferSize_ = ubSizePlatForm_ / DOUBLE_TIME;
        xSliceNum_ = Ops::Base::CeilDiv(xAlignSize_, xBufferSize_);
        reservedXSize_ = xAxisSize_ - (xSliceNum_ - 1) * xBufferSize_ / xDSize_;
        indexBufferSize_ = static_cast<float>(xBufferSize_) / (indexDSize_ + xDSize_ * DOUBLE_TIME + MASK_SIZE_RATE) *
                           indexDSize_;
        indexBufferSize_ = Ops::Base::CeilAlign(indexBufferSize_, BLOCK_SIZE_T * INT64_DSIZE);
        indexSliceNum_ = Ops::Base::CeilDiv(indexAlignSize_, indexBufferSize_);
        yBufferSize_ = indexBufferSize_ / indexDSize_ * xDSize_;
        yBufferSize_ = Ops::Base::CeilAlign(yBufferSize_, BLOCK_SIZE_T);
        reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
        maskBufferSize_ = indexBufferSize_ / indexDSize_ / INT64_DSIZE;
        maskBufferSize_ = Ops::Base::CeilAlign(maskBufferSize_, BLOCK_SIZE_T);
    }

    DoScalarMode();
}

void GatherElementsV2LastDimTiling::DoScalarMode()
{
    if (xSliceNum_ > MAX_SLICE_NUM || xAxisSize_ / indexAxisSize_ > MAX_SIZE_RATIO) {
        tilingData_.lastDimTiling.set_scalarMode(NUM_ONE);
        int64_t idxBLockSize = Ops::Base::CeilDiv(indexAlignSize_, BLOCK_SIZE_T * DOUBLE_TIME) * DOUBLE_TIME;
        int64_t yBlockSize = Ops::Base::CeilDiv(indexAxisSize_ * xRealDsize_, BLOCK_SIZE_T);
        eachCalculationLines_ = ubSizePlatForm_ / BLOCK_SIZE_T / (idxBLockSize + yBlockSize);
        bool multRowProcessFlag = eachCalculationLines_ > 1 && (nonCollectingAxisSzie_ > totalCoreNum_ ||
                                                                (nonCollectingAxisSzie_ <= totalCoreNum_ &&
                                                                 indexAxisSize_ < totalCoreNum_ * BLOCK_SIZE_T));
        if (multRowProcessFlag) {
            indexBufferSize_ = eachCalculationLines_ * idxBLockSize * BLOCK_SIZE_T;
            yBufferSize_ = eachCalculationLines_ * yBlockSize * BLOCK_SIZE_T;
        } else {
            indexSliceNum_ = Ops::Base::CeilDiv(indexAlignSize_, indexBufferSize_);
            scalarModeLength_ = indexSliceNum_ * nonCollectingAxisSzie_;
            if (scalarModeLength_ < totalCoreNum_) {
                indexSliceNum_ = totalCoreNum_ / nonCollectingAxisSzie_;
                scalarModeLength_ = indexSliceNum_ * nonCollectingAxisSzie_;
            }
            indexBufferSize_ = Ops::Base::CeilAlign(indexAlignSize_ / indexSliceNum_ + INT64_DSIZE,
                                                    BLOCK_SIZE_T * DOUBLE_TIME);
            yBufferSize_ = indexBufferSize_ / indexDSize_ * xRealDsize_;
            yBufferSize_ = Ops::Base::CeilAlign(yBufferSize_, BLOCK_SIZE_T);
            reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
        }
    }
}

void GatherElementsV2LastDimTiling::DoNeedUseCore()
{
    if (scalarModeLength_ == 0) {
        needUsedCore_ = nonCollectingAxisSzie_ > totalCoreNum_ ? totalCoreNum_ : nonCollectingAxisSzie_;
        formerCoreRowNum_ = nonCollectingAxisSzie_ / needUsedCore_;
        formerCoreNum_ = nonCollectingAxisSzie_ % needUsedCore_;
        tilingData_.lastDimTiling.set_formerCoreRowNum(formerCoreRowNum_);
        tilingData_.lastDimTiling.set_formerCoreNum(formerCoreNum_);
    } else {
        needUsedCore_ = scalarModeLength_ > totalCoreNum_ ? totalCoreNum_ : scalarModeLength_;
    }

    context_->SetBlockDim(needUsedCore_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void GatherElementsV2LastDimTiling::SetTilingData()
{
    tilingData_.lastDimTiling.set_xBufferSize(xBufferSize_);
    tilingData_.lastDimTiling.set_indexBufferSize(indexBufferSize_);
    tilingData_.lastDimTiling.set_yBufferSize(yBufferSize_);
    tilingData_.lastDimTiling.set_maskBufferSize(maskBufferSize_);
    tilingData_.lastDimTiling.set_eachCalculationLines(eachCalculationLines_);
    tilingData_.lastDimTiling.set_xSliceNum(xSliceNum_);
    tilingData_.lastDimTiling.set_indexSliceNum(indexSliceNum_);
    tilingData_.lastDimTiling.set_reservedXSize(reservedXSize_);
    tilingData_.lastDimTiling.set_reservedIndexSize(reservedIndexSize_);
    tilingData_.lastDimTiling.set_scalarModeLength(scalarModeLength_);
    tilingData_.lastDimTiling.set_specialDataMove(specialDataMove_);
}

void GatherElementsV2LastDimTiling::PrintTilingData() const
{
    OP_LOGD(context_, "dimNum_:                    %ld", dimNum_);
    OP_LOGD(context_, "nonCollectingAxisSzie_:     %ld", nonCollectingAxisSzie_);
    OP_LOGD(context_, "xDSize_:                    %ld", xDSize_);
    OP_LOGD(context_, "indexDSize_:                %ld", indexDSize_);
    OP_LOGD(context_, "eachCalculationLines_:      %ld", eachCalculationLines_);
    OP_LOGD(context_, "xAxisSize_:                 %ld", xAxisSize_);
    OP_LOGD(context_, "indexAxisSize_:             %ld", indexAxisSize_);
    OP_LOGD(context_, "tilingKey:                  %ld", tilingKey_);
    OP_LOGD(context_, "xSliceNum_:                 %ld", xSliceNum_);
    OP_LOGD(context_, "indexSliceNum_:             %ld", indexSliceNum_);
    OP_LOGD(context_, "xAlignSize_:                %ld", xAlignSize_);
    OP_LOGD(context_, "indexAlignSize_:            %ld", indexAlignSize_);
    OP_LOGD(context_, "yAlignSize_:                %ld", yAlignSize_);
    OP_LOGD(context_, "reservedXSize_:             %ld", reservedXSize_);
    OP_LOGD(context_, "reservedIndexSize_:         %ld", reservedIndexSize_);
    OP_LOGD(context_, "needUsedCore_:              %ld", needUsedCore_);
    OP_LOGD(context_, "formerCoreRowNum_:          %ld", formerCoreRowNum_);
    OP_LOGD(context_, "indexBufferSize_:           %ld", indexBufferSize_);
    OP_LOGD(context_, "yBufferSize_:               %ld", yBufferSize_);
    OP_LOGD(context_, "xBufferSize_:               %ld", xBufferSize_);
    OP_LOGD(context_, "maskBufferSize_:            %ld", maskBufferSize_);
    OP_LOGD(context_, "specialDataMove_:           %ld", specialDataMove_);
    OP_LOGD(context_, "xDsizeRatio_:               %ld", xDsizeRatio_);
    OP_LOGD(context_, "scalarModeLength_:          %ld", scalarModeLength_);
    OP_LOGD(context_, "batchProcess_:               %d", batchProcess_);
}

ge::graphStatus GatherElementsV2LastDimTiling::DoOpTiling()
{
    GetDSize();
    IfEnableBatch();
    DoUBSlice();
    SetTilingData();
    DoNeedUseCore();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
#endif // GATHER_ELEMENTS_V2_LAST_DIM_TILING_H