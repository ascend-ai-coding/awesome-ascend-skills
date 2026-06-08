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
// 索引类算子 op_kernel 参考代码（逐元素索引模式）
// 适用: gather, scatter, scatter_add 等逐元素索引算子
// 示例: 以 torch.gather 为例，index 与 output 同形，每行有独立索引
// 使用: 本文件演示了三种执行路径(GatherPath/UBPath/ElementPath)、
//       多行批量处理、逐行索引加载、Muls索引转换、Gather指令调用等
// ============================================================

#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t UB_CAPACITY = 192 * 1024;
constexpr int64_t MAX_GATHER_SRC_BYTES = 32 * 1024;
constexpr int64_t MAX_CORES = 40;

template<typename T>
class <OpName>Kernel {
public:
    __aicore__ inline <OpName>Kernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                 int64_t inputRows, int64_t srcDimSize,
                                 int64_t indexSize, int64_t totalRows,
                                 int64_t rowOffset);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessGatherPath(int64_t startRow, int64_t numRows);
    __aicore__ inline void ProcessUBPath(int64_t row);
    __aicore__ inline void ProcessElementPath(int64_t row);

private:
    TPipe pipe;

    // GatherPath: 多行批量缓冲
    TBuf<TPosition::VECIN> inputBuf;
    TBuf<TPosition::VECIN> indexBuf;    // 逐行索引，每行独立
    TBuf<TPosition::VECOUT> outputBuf;

    // UBPath: 单行缓冲
    TBuf<TPosition::VECIN> ubInputBuf;
    TBuf<TPosition::VECIN> ubIndexBuf;
    TBuf<TPosition::VECOUT> ubOutputBuf;

    // ElementPath: 仅 index + output
    TBuf<TPosition::VECIN> indexBufElem;
    TBuf<TPosition::VECOUT> outputBufElem;

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> outputGm;

    int64_t inputRows_ = 0;
    int64_t srcDimSize_ = 0;
    int64_t indexSize_ = 0;
    int64_t totalRows_ = 0;
    int64_t rowOffset_ = 0;
    int64_t inputRowBytes_ = 0;
    int64_t indexRowBytes_ = 0;
    int64_t outputRowBytes_ = 0;
    int64_t maxRowsPerBatch_ = 1;
    int8_t path_ = 0;
};

// =========================== Init: 路径判断与 UB 分配 start ============================

template<typename T>
__aicore__ inline void <OpName>Kernel<T>::Init(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                int64_t inputRows, int64_t srcDimSize,
                                                int64_t indexSize, int64_t totalRows,
                                                int64_t rowOffset) {
    inputRows_ = inputRows;
    srcDimSize_ = srcDimSize;
    indexSize_ = indexSize;
    totalRows_ = totalRows;
    rowOffset_ = rowOffset;

    inputGm.SetGlobalBuffer((__gm__ T*)input, inputRows * srcDimSize);
    indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    // 32 字节对齐的行字节数
    inputRowBytes_  = (srcDimSize * sizeof(T) + 31) / 32 * 32;
    indexRowBytes_  = (indexSize * sizeof(int32_t) + 31) / 32 * 32;
    outputRowBytes_ = (indexSize * sizeof(T) + 31) / 32 * 32;

    // gather/scatter: 索引每行不同，indexBuf 按 maxRowsPerBatch 分配
    int64_t perRowUB = inputRowBytes_ + indexRowBytes_ + outputRowBytes_;

    if (perRowUB <= UB_CAPACITY && srcDimSize * sizeof(T) <= MAX_GATHER_SRC_BYTES
        && srcDimSize * sizeof(T) >= 32) {
        // Path 0: GatherPath — 硬件 Gather 指令
        path_ = 0;
        maxRowsPerBatch_ = UB_CAPACITY / perRowUB;
        if (maxRowsPerBatch_ < 1) maxRowsPerBatch_ = 1;
        pipe.InitBuffer(inputBuf,  maxRowsPerBatch_ * inputRowBytes_);
        pipe.InitBuffer(indexBuf,  maxRowsPerBatch_ * indexRowBytes_);
        pipe.InitBuffer(outputBuf, maxRowsPerBatch_ * outputRowBytes_);
    } else if (perRowUB <= UB_CAPACITY) {
        // Path 1: UBPath — 标量 UB 访问
        path_ = 1;
        pipe.InitBuffer(ubInputBuf,  inputRowBytes_);
        pipe.InitBuffer(ubIndexBuf,  indexRowBytes_);
        pipe.InitBuffer(ubOutputBuf, outputRowBytes_);
    } else {
        // Path 2: ElementPath — 逐元素 GM 直接访问
        path_ = 2;
        pipe.InitBuffer(indexBufElem,  indexRowBytes_);
        pipe.InitBuffer(outputBufElem, outputRowBytes_);
    }
}

// =========================== Init end ============================


// =========================== GatherPath start ============================
// 核心接口: DataCopyPad + Muls + ReinterpretCast + Gather
// **关键点**: Gather 的 index 参数必须是字节偏移(uint32_t)，
//           需先用 Muls 将元素索引乘以 sizeof(T) 转换。
// gather/scatter: 每行有独立索引，需逐行加载 index

template<typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessGatherPath(int64_t startRow, int64_t numRows) {
    LocalTensor<T> inputLocal = inputBuf.Get<T>();
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    LocalTensor<T> outputLocal = outputBuf.Get<T>();

    // 1. 逐行加载 input → UB
    DataCopyExtParams inputCopyParams;
    inputCopyParams.blockCount = 1;
    inputCopyParams.blockLen = static_cast<uint32_t>(srcDimSize_ * sizeof(T));
    inputCopyParams.srcStride = 0;
    inputCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> inputPadParams;
    inputPadParams.isPad = false;
    for (int64_t r = 0; r < numRows; r++) {
        DataCopyPad(inputLocal[r * (inputRowBytes_ / sizeof(T))],
                    inputGm[(startRow + r) * srcDimSize_], inputCopyParams, inputPadParams);
    }
    PipeBarrier<PIPE_ALL>();

    // 2. 逐行加载 index → UB（每行有独立索引）
    DataCopyExtParams indexCopyParams;
    indexCopyParams.blockCount = 1;
    indexCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(int32_t));
    indexCopyParams.srcStride = 0;
    indexCopyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> indexPadParams;
    indexPadParams.isPad = false;
    for (int64_t r = 0; r < numRows; r++) {
        DataCopyPad(indexLocal[r * (indexRowBytes_ / sizeof(int32_t))],
                    indexGm[(startRow + r) * indexSize_], indexCopyParams, indexPadParams);
    }
    PipeBarrier<PIPE_ALL>();

    // 3. Muls: 元素索引 → 字节偏移（关键步骤！）
    for (int64_t r = 0; r < numRows; r++) {
        int64_t off = r * (indexRowBytes_ / sizeof(int32_t));
        Muls(indexLocal[off], indexLocal[off], (int32_t)sizeof(T), indexSize_);
    }
    PipeBarrier<PIPE_ALL>();

    // 4. ReinterpretCast + Gather
    LocalTensor<uint32_t> indexUintLocal = indexLocal.ReinterpretCast<uint32_t>();
    for (int64_t r = 0; r < numRows; r++) {
        int64_t inOff  = r * (inputRowBytes_ / sizeof(T));
        int64_t idxOff = r * (indexRowBytes_ / sizeof(uint32_t));
        int64_t outOff = r * (outputRowBytes_ / sizeof(T));
        Gather(outputLocal[outOff], inputLocal[inOff], indexUintLocal[idxOff],
               (uint32_t)0, static_cast<uint32_t>(indexSize_));
        PipeBarrier<PIPE_ALL>();
    }

    // 5. 逐行写回 output UB → GM
    DataCopyExtParams outputCopyParams;
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(T));
    outputCopyParams.srcStride = 0;
    outputCopyParams.dstStride = 0;
    for (int64_t r = 0; r < numRows; r++) {
        DataCopyPad(outputGm[(startRow + r) * indexSize_],
                    outputLocal[r * (outputRowBytes_ / sizeof(T))], outputCopyParams);
    }
    PipeBarrier<PIPE_ALL>();
}
// =========================== GatherPath end ============================


// =========================== UBPath start ============================
// 当 input 行放得进 UB 但不满足 Gather 约束时，使用标量 GetValue/SetValue

template<typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessUBPath(int64_t row) {
    LocalTensor<T> inputLocal = ubInputBuf.Get<T>();
    LocalTensor<int32_t> indexLocal = ubIndexBuf.Get<int32_t>();
    LocalTensor<T> outputLocal = ubOutputBuf.Get<T>();

    // 加载 input 行
    DataCopyExtParams inputCopyParams;
    inputCopyParams.blockCount = 1;
    inputCopyParams.blockLen = static_cast<uint32_t>(srcDimSize_ * sizeof(T));
    inputCopyParams.srcStride = 0;
    inputCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> inputPadParams;
    inputPadParams.isPad = false;
    DataCopyPad(inputLocal, inputGm[row * srcDimSize_], inputCopyParams, inputPadParams);
    PipeBarrier<PIPE_ALL>();

    // 加载该行的 index（每行独立索引）
    DataCopyExtParams indexCopyParams;
    indexCopyParams.blockCount = 1;
    indexCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(int32_t));
    indexCopyParams.srcStride = 0;
    indexCopyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> indexPadParams;
    indexPadParams.isPad = false;
    DataCopyPad(indexLocal, indexGm[row * indexSize_], indexCopyParams, indexPadParams);
    PipeBarrier<PIPE_ALL>();

    // 标量索引收集
    for (int64_t j = 0; j < indexSize_; j++) {
        int32_t idx = indexLocal.GetValue(j);
        T val = inputLocal.GetValue(idx);
        outputLocal.SetValue(j, val);
    }
    PipeBarrier<PIPE_ALL>();

    // 写回 output
    DataCopyExtParams outputCopyParams;
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(T));
    outputCopyParams.srcStride = 0;
    outputCopyParams.dstStride = 0;
    DataCopyPad(outputGm[row * indexSize_], outputLocal, outputCopyParams);
    PipeBarrier<PIPE_ALL>();
}
// =========================== UBPath end ============================


// =========================== ElementPath start ============================
// 当 perRowUB 超出 UB 容量时，仅加载 index，直接从 GM 读取 input

template<typename T>
__aicore__ inline void <OpName>Kernel<T>::ProcessElementPath(int64_t row) {
    LocalTensor<int32_t> indexLocal = indexBufElem.Get<int32_t>();
    LocalTensor<T> outputLocal = outputBufElem.Get<T>();

    // 加载该行的 index
    DataCopyExtParams indexCopyParams;
    indexCopyParams.blockCount = 1;
    indexCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(int32_t));
    indexCopyParams.srcStride = 0;
    indexCopyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> indexPadParams;
    indexPadParams.isPad = false;
    DataCopyPad(indexLocal, indexGm[row * indexSize_], indexCopyParams, indexPadParams);
    PipeBarrier<PIPE_ALL>();

    // 逐元素从 GM 直接读取
    for (int64_t j = 0; j < indexSize_; j++) {
        int32_t idx = indexLocal.GetValue(j);
        T val = inputGm.GetValue(row * srcDimSize_ + idx);
        outputLocal.SetValue(j, val);
    }
    PipeBarrier<PIPE_ALL>();

    // 写回 output
    DataCopyExtParams outputCopyParams;
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = static_cast<uint32_t>(indexSize_ * sizeof(T));
    outputCopyParams.srcStride = 0;
    outputCopyParams.dstStride = 0;
    DataCopyPad(outputGm[row * indexSize_], outputLocal, outputCopyParams);
    PipeBarrier<PIPE_ALL>();
}
// =========================== ElementPath end ============================


// =========================== Process 主入口 start ============================

template<typename T>
__aicore__ inline void <OpName>Kernel<T>::Process() {
    int64_t blockIdx_ = GetBlockIdx();

    if (path_ == 0) {
        // GatherPath: 连续块分配 + 多行批量处理
        int64_t blockDim = (totalRows_ < MAX_CORES) ? totalRows_ : MAX_CORES;
        int64_t base = totalRows_ / blockDim;
        int64_t remainder = totalRows_ % blockDim;
        int64_t myStart, myRows;
        if (blockIdx_ < remainder) {
            myStart = rowOffset_ + blockIdx_ * (base + 1);
            myRows = base + 1;
        } else {
            myStart = rowOffset_ + remainder * (base + 1) + (blockIdx_ - remainder) * base;
            myRows = base;
        }

        for (int64_t r = myStart; r < myStart + myRows; r += maxRowsPerBatch_) {
            int64_t numRows = myStart + myRows - r;
            if (numRows > maxRowsPerBatch_) numRows = maxRowsPerBatch_;
            ProcessGatherPath(r, numRows);
        }
    } else {
        // UBPath / ElementPath: 交错行分配
        int64_t startRow = rowOffset_ + blockIdx_;
        for (int64_t r = startRow; r < rowOffset_ + totalRows_; r += MAX_CORES) {
            if (r >= inputRows_) break;
            if (path_ == 1) {
                ProcessUBPath(r);
            } else {
                ProcessElementPath(r);
            }
        }
    }
}
// =========================== Process 主入口 end ============================


// =========================== Kernel 入口 start ============================
// 搬运类算子不做数值计算，bf16 与 fp16 均为 2 字节，统一走 half 路径搬运

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                  int64_t inputRows, int64_t srcDimSize,
                                                  int64_t indexSize, int64_t dtypeSize,
                                                  int64_t totalRows, int64_t rowOffset) {
    if (dtypeSize == 4) {
        <OpName>Kernel<float> op;
        op.Init(input, index, output, inputRows, srcDimSize, indexSize, totalRows, rowOffset);
        op.Process();
    } else if (dtypeSize == 2) {
        // fp16 和 bf16 均为 2 字节，搬运类算子可直接用 half 路径
        <OpName>Kernel<half> op;
        op.Init(input, index, output, inputRows, srcDimSize, indexSize, totalRows, rowOffset);
        op.Process();
    }
}
// =========================== Kernel 入口 end ============================


// =========================== DMA同步（可选优化）start ============================
// 正确性验证后，可将 PipeBarrier<PIPE_ALL>() 替换为以下轻量同步：

// CPU 等待 MTE2 (GM→UB) 搬运完成
__aicore__ inline void WaitMte2Done() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evt);
}

// CPU 等待 Vector 计算完成
__aicore__ inline void WaitVDone() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
}

// CPU 等待 MTE3 (UB→GM) 搬运完成
__aicore__ inline void WaitMte3Done() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
}
// =========================== DMA同步（可选优化）end ============================
