// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================
// 索引类算子 op_kernel 参考代码（单行更新 / index_select 风格）
// 适用: index_select, index_add 等 1D 共享索引算子
// 示例: 以 torch.index_select 为例
//       3D 布局: [outerSize, srcDimSize, innerSize]
//       index 为 1D 共享，每个 index 值选择一整行 innerSize 个元素
//       动态核分配: taskId 交错分配 outerIdx + indexRange
//       UB 级 Tiling: 小尾轴(≤4096)多行批量 / 大尾轴分块
// ============================================================

#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t UB_CAPACITY = 192 * 1024;
constexpr int64_t MAX_CORES = 40;
constexpr int64_t MAX_INDEX_BATCH = 512;
constexpr int64_t SMALL_INNER_THRESHOLD = 4096;
constexpr int64_t BUFFER_NUM = 2;

template <typename T>
class <OpName>Kernel {
public:
    __aicore__ inline <OpName>Kernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                int64_t outerSize, int64_t innerSize,
                                int64_t indexSize, int64_t srcDimSize,
                                int64_t totalLength, int64_t coresPerOuter);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessIndexRange(int64_t outerIdx,
                                              int64_t indexStart, int64_t indexEnd);
    __aicore__ inline void ProcessSmallInner(int64_t outerIdx,
                                              int64_t indexStart, int64_t indexEnd);
    __aicore__ inline void ProcessLargeInner(int64_t outerIdx,
                                              int64_t indexStart, int64_t indexEnd);
    __aicore__ inline void LoadIndexBatch(int64_t batchStart, int64_t batchSize);
    __aicore__ inline int32_t GetIndexValue(int64_t localIdx);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> indexBuf;                // 共享索引缓冲
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;      // 双缓冲输出

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> outputGm;

    // 3D 维度
    int64_t outerSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexSize_ = 0;
    int64_t srcDimSize_ = 0;
    int64_t coresPerOuter_ = 0;
    int64_t totalLength_ = 0;

    // UB 级 Tiling 派生参数
    int64_t alignedInnerSize_ = 0;
    int64_t indexBatchSize_ = 0;
    int64_t tileSize_ = 0;
    int64_t alignedTileSize_ = 0;
    int64_t tileCount_ = 0;
    int64_t batchRows_ = 1;
    bool isSmallInner_ = false;
};

// =========================== Init start ============================

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::Init(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                int64_t outerSize, int64_t innerSize,
                                                int64_t indexSize, int64_t srcDimSize,
                                                int64_t totalLength, int64_t coresPerOuter) {
    outerSize_ = outerSize;
    innerSize_ = innerSize;
    indexSize_ = indexSize;
    srcDimSize_ = srcDimSize;
    totalLength_ = totalLength;
    coresPerOuter_ = coresPerOuter;

    inputGm.SetGlobalBuffer((__gm__ T*)input, totalLength);
    indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    // 对齐 innerSize（32 字节对齐）
    constexpr int64_t alignNum = 32 / sizeof(T);  // float32:8, float16/bf16:16
    alignedInnerSize_ = (innerSize_ + alignNum - 1) / alignNum * alignNum;
    if (alignedInnerSize_ == 0) alignedInnerSize_ = alignNum;

    // 索引批次
    indexBatchSize_ = indexSize_ < MAX_INDEX_BATCH ? indexSize_ : MAX_INDEX_BATCH;

    // 索引缓冲区大小（32 字节对齐）
    int64_t indexAlignedSize = (indexBatchSize_ * sizeof(int32_t) + 31) / 32 * 32;
    pipe.InitBuffer(indexBuf, indexAlignedSize);

    // 可用 UB 空间（减去 indexBuf）
    int64_t availableUB = UB_CAPACITY - indexAlignedSize;
    int64_t maxElementsInUB = availableUB / sizeof(T) / BUFFER_NUM;

    // 小尾轴 / 大尾轴判断
    isSmallInner_ = (innerSize_ <= SMALL_INNER_THRESHOLD);

    if (isSmallInner_) {
        // 小尾轴: 多行批量处理
        int64_t maxRows = maxElementsInUB / alignedInnerSize_;
        batchRows_ = maxRows < indexBatchSize_ ? maxRows : indexBatchSize_;
        if (batchRows_ < 1) batchRows_ = 1;

        tileSize_ = innerSize_;
        alignedTileSize_ = alignedInnerSize_;
        tileCount_ = 1;
    } else {
        // 大尾轴: 分块处理
        if (alignedInnerSize_ > maxElementsInUB) {
            tileSize_ = maxElementsInUB / alignNum * alignNum;
            alignedTileSize_ = tileSize_;
            tileCount_ = (innerSize_ + tileSize_ - 1) / tileSize_;
        } else {
            tileSize_ = innerSize_;
            alignedTileSize_ = alignedInnerSize_;
            tileCount_ = 1;
        }
        batchRows_ = 1;
    }

    pipe.InitBuffer(outQueue, BUFFER_NUM, alignedTileSize_ * sizeof(T) * batchRows_);
}

// =========================== Init end ============================


// =========================== 索引加载 start ============================

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::LoadIndexBatch(int64_t batchStart, int64_t batchSize) {
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(batchSize * sizeof(int32_t));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPadExtParams<int32_t> padParams;
    padParams.isPad = false;

    DataCopyPad(indexLocal, indexGm[batchStart], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline int32_t <OpName>Kernel<T>::GetIndexValue(int64_t localIdx) {
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    return indexLocal.GetValue(localIdx);
}

// =========================== 索引加载 end ============================


// =========================== 小尾轴处理 start ============================
// innerSize ≤ 4096: 多行批量，indexBuf 分批加载，每批 batchRows_ 行

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessSmallInner(int64_t outerIdx,
                                                              int64_t indexStart,
                                                              int64_t indexEnd) {
    int64_t idxProcessed = indexStart;
    while (idxProcessed < indexEnd) {
        int64_t currentBatch = indexEnd - idxProcessed;
        if (currentBatch > batchRows_) currentBatch = batchRows_;

        // 加载一批索引
        LoadIndexBatch(idxProcessed, currentBatch);

        // 逐行处理: 每个 index 拷贝 innerSize 个连续元素
        for (int64_t i = 0; i < currentBatch; i++) {
            int32_t indexValue = GetIndexValue(i);
            int64_t inputOffset = outerIdx * srcDimSize_ * innerSize_
                                  + indexValue * innerSize_;
            int64_t outputOffset = outerIdx * indexSize_ * innerSize_
                                   + (idxProcessed + i) * innerSize_;

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<T> padParams;
            padParams.isPad = false;

            DataCopyPad(outputLocal, inputGm[inputOffset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            outQueue.EnQue<T>(outputLocal);

            LocalTensor<T> outLocal = outQueue.DeQue<T>();
            DataCopyPad(outputGm[outputOffset], outLocal, copyParams);
            outQueue.FreeTensor(outLocal);
        }

        idxProcessed += currentBatch;
    }
}

// =========================== 小尾轴处理 end ============================


// =========================== 大尾轴处理 start ============================
// innerSize > 4096: 分块处理，沿 innerSize 切分为 tileSize_ 大小的块

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessLargeInner(int64_t outerIdx,
                                                              int64_t indexStart,
                                                              int64_t indexEnd) {
    // 加载本批索引
    int64_t currentBatch = indexEnd - indexStart;
    if (currentBatch > indexBatchSize_) currentBatch = indexBatchSize_;
    LoadIndexBatch(indexStart, currentBatch);

    for (int64_t i = 0; i < currentBatch; i++) {
        int32_t indexValue = GetIndexValue(i);
        int64_t inputBase = outerIdx * srcDimSize_ * innerSize_
                            + indexValue * innerSize_;
        int64_t outputBase = outerIdx * indexSize_ * innerSize_
                             + (indexStart + i) * innerSize_;

        // 沿 innerSize 分块
        for (int64_t t = 0; t < tileCount_; t++) {
            int64_t tileStart = t * tileSize_;
            int64_t curTileSize = tileSize_;
            if (tileStart + curTileSize > innerSize_) {
                curTileSize = innerSize_ - tileStart;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curTileSize * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<T> padParams;
            padParams.isPad = false;

            DataCopyPad(outputLocal, inputGm[inputBase + tileStart], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            outQueue.EnQue<T>(outputLocal);

            LocalTensor<T> outLocal = outQueue.DeQue<T>();
            DataCopyPad(outputGm[outputBase + tileStart], outLocal, copyParams);
            outQueue.FreeTensor(outLocal);
        }
    }
}

// =========================== 大尾轴处理 end ============================


// =========================== ProcessIndexRange start ============================

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessIndexRange(int64_t outerIdx,
                                                              int64_t indexStart,
                                                              int64_t indexEnd) {
    if (isSmallInner_) {
        ProcessSmallInner(outerIdx, indexStart, indexEnd);
    } else {
        ProcessLargeInner(outerIdx, indexStart, indexEnd);
    }
}

// =========================== ProcessIndexRange end ============================


// =========================== Process 主入口 start ============================

template <typename T>
__aicore__ inline void <OpName>Kernel<T>::Process() {
    int64_t blockIdx = GetBlockIdx();
    int64_t totalTasks = outerSize_ * coresPerOuter_;
    int64_t usedCores = totalTasks < MAX_CORES ? totalTasks : MAX_CORES;

    if (blockIdx >= usedCores) return;

    int64_t indexPerCore = (indexSize_ + coresPerOuter_ - 1) / coresPerOuter_;

    // 核 0 处理任务 0, usedCores, 2*usedCores...
    // 核 1 处理任务 1, usedCores+1, 2*usedCores+1...
    for (int64_t taskId = blockIdx; taskId < totalTasks; taskId += usedCores) {
        int64_t outerIdx = taskId / coresPerOuter_;
        int64_t subIdx = taskId % coresPerOuter_;

        int64_t indexStart = subIdx * indexPerCore;
        int64_t indexEnd = indexStart + indexPerCore;
        if (indexEnd > indexSize_) indexEnd = indexSize_;

        if (indexStart >= indexSize_) continue;

        ProcessIndexRange(outerIdx, indexStart, indexEnd);
    }
}

// =========================== Process 主入口 end ============================


// =========================== Kernel 入口 start ============================
// 搬运类算子不做数值计算，bf16 与 fp16 均为 2 字节，统一走 half 路径搬运

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                  int64_t outerSize, int64_t innerSize,
                                                  int64_t indexSize, int64_t srcDimSize,
                                                  int64_t totalLength, int64_t coresPerOuter) {
    if (outerSize * srcDimSize * innerSize > 0) {
        <OpName>Kernel<float> op;
        op.Init(input, index, output, outerSize, innerSize,
                indexSize, srcDimSize, totalLength, coresPerOuter);
        op.Process();
    }
}

// =========================== Kernel 入口 end ============================
