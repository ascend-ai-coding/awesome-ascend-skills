// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================
// pool 算子 op_kernel 模板
// 适用: Pooling 算子 (MaxPool, AvgPool)
// 使用: 复制到 csrc/ops/<op_name>/op_kernel/<op_name>.cpp，
//       替换 <op_name>/<OpName> 占位符，修改 Compute 逻辑
// ============================================================

#include "kernel_operator.h"

using namespace AscendC;

// Precise sync helpers
__aicore__ inline void MTE2ToVSync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}
__aicore__ inline void VToMTE3Sync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}
__aicore__ inline void MTE3ToMTE2Sync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

template <typename T>
class KernelPooling {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t inputD, uint32_t inputH, uint32_t inputW,
                                 uint32_t outputD, uint32_t outputH, uint32_t outputW,
                                 uint32_t kernelD, uint32_t kernelH, uint32_t kernelW,
                                 uint32_t strideD, uint32_t strideH, uint32_t strideW,
                                 uint32_t padD, uint32_t padH, uint32_t padW,
                                 uint32_t countIncludePad, uint32_t ceilMode,
                                 uint64_t formerNum, uint64_t formerLength, uint64_t tailLength,
                                 // AvgPool3d额外参数: divisorOverride (MaxPool3d不需要, 可传0)
                                 uint64_t divisorOverride,
                                 uint64_t windowWNum) {
        // 保存用例参数
        this->batchSize = batchSize;  this->channels = channels;
        this->inputD = inputD;   this->inputH = inputH;   this->inputW = inputW;
        this->outputD = outputD; this->outputH = outputH; this->outputW = outputW;
        this->kernelD = kernelD; this->kernelH = kernelH; this->kernelW = kernelW;
        this->strideD = strideD; this->strideH = strideH; this->strideW = strideW;
        this->padD = padD;       this->padH = padH;       this->padW = padW;
        this->countIncludePad = countIncludePad;
        this->ceilMode = ceilMode;
        this->windowWNum = windowWNum;

        // 通道对齐: FP32→8倍元素, FP16/BF16→16倍元素 (32字节对齐)
        int64_t alignElements = 32 / sizeof(T);
        this->alignC = ((channels + alignElements - 1) / alignElements) * alignElements;

        // 获取每个核需要计算的起始点和截至点
        this->outputPointNum = GetBlockIdx() < formerNum ? formerLength : tailLength;
        this->outputPointOffset = GetBlockIdx() < formerNum
            ? formerLength * GetBlockIdx()
            : formerNum * formerLength + tailLength * (GetBlockIdx() - formerNum);

        // 设置全局内存
        inputGm.SetGlobalBuffer((__gm__ T*)input, (uint64_t)batchSize * inputD * inputH * inputW * channels);
        outputGm.SetGlobalBuffer((__gm__ T*)output, (uint64_t)batchSize * outputD * outputH * outputW * channels);

        // UB空间分配
        uint32_t rowElements = windowWNum * strideW + kernelW - 1;
        uint64_t dataSize = (uint64_t)rowElements * alignC;
        uint64_t castSize = (uint64_t)rowElements * alignC;
        uint64_t sumSize  = (uint64_t)windowWNum * alignC;

        // 无索引版本
        pipe.InitBuffer(shareBuf, dataSize * sizeof(T) + castSize * sizeof(float) + sumSize * sizeof(float));
        dataLocal = shareBuf.GetWithOffset<T>(dataSize, 0);
        castLocal = shareBuf.GetWithOffset<float>(castSize, dataSize * sizeof(T));
        sumBufLocal = shareBuf.GetWithOffset<float>(sumSize, dataSize * sizeof(T) + castSize * sizeof(float));

        // 有索引版本(MaxPool3d专用): 额外分配 indicesLocal + indicesUpdLocal + maskBufLocal
        // uint64_t idxSize = windowWNum * alignC * sizeof(int32_t);
        // uint64_t maskU16 = windowWNum * ((C + 63) / 64 * 16); // ceil(C/64) chunks × 16 u16 (32B对齐)
        // pipe.InitBuffer(shareBuf, dataSize*sizeof(T) + castSize*sizeof(float) + sumSize*sizeof(float)
        //                  + idxSize*2 + maskU16*sizeof(uint16_t) + 64);
        // indicesLocal = shareBuf.GetWithOffset<int32_t>(windowWNum*alignC, ...);
        // indicesUpdLocal = shareBuf.GetWithOffset<int32_t>(windowWNum*alignC, ...);
        // maskBufLocal = shareBuf.GetWithOffset<uint16_t>(maskU16, ...); // 32B对齐

        // 预计算高维切分参数和填充参数
        this->rightPadding = static_cast<uint8_t>(alignC - channels);
        constexpr uint32_t MAX_MASK_FP32 = 64;
        this->maskLoopCount = (alignC + MAX_MASK_FP32 - 1) / MAX_MASK_FP32;
        this->dstRepStride = static_cast<uint8_t>(alignC / 8);
        this->src1RepStride = static_cast<uint8_t>(strideW * alignC / 8);

        this->isRepeat = (padW == 0 && !ceilMode); // padW==0且非ceilMode时所有W方向数据连续有效，可走快速高维切分路径跳过边界分段扫描
        this->isSamePoolSize = divisorOverride || ((countIncludePad || padW == 0) && !ceilMode); // AvgPool3d专用。为true时所有窗口poolSize相同直接Muls批量除, false时逐窗口单独计算除数
    }

    __aicore__ inline void Process() {
        int64_t curWindowWNum = windowWNum;
        for (int64_t outputPointIdx = outputPointOffset, count = 0;
                outputPointIdx < outputPointOffset + outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {

            // windowWNum处理越界，跨行时截断，每次最多处理w方向同一行的窗口
            curWindowWNum = (count + windowWNum) < outputPointNum ? windowWNum : outputPointNum - count;
            int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % outputW;
            curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                            ? curWindowWNum - newRowWindowWNum : curWindowWNum;

            // 条件说明: UB空间足够处理至少一个完整窗口(含kW个W位置)时走processOneOrMultiWindow,
            // 否则走processSmallerOneWindow兜底（遇到概率极小），K极大时单窗口逐个处理，C极大放不下时for循环对C切块处理
            if (curWindowWNum >= 1) {
                processOneOrMultiWindow(outputPointIdx, curWindowWNum);
            } else {
                processSmallerOneWindow(outputPointIdx, curWindowWNum);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
        // blockCount: 搬运次数, blockLen: 每次搬运元素个数
        // srcStride=0 表示源地址连续，dstStride=0 表示目的地址连续
        DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0}; // rightPadding: 每行尾部填充元素个数,（但所占字节数不能超过32字节）
        DataCopyPad(dataLocal, inputGm[offset], copyParams, padParams);
        MTE2ToVSync();
    }

    __aicore__ inline void CopyOut(int64_t offset, uint16_t blockCount, uint32_t blockLen) {
        // 注意搬出时长度不要超过GM内容空间，如果超过应该截断
        VToMTE3Sync();
        DataCopyExtParams copyParamsOut{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm[offset], dataLocal, copyParamsOut);
        MTE3ToMTE2Sync();
    }

    __aicore__ inline void castXToFp32(LocalTensor<float> dstTensor, LocalTensor<T> srcTensor, uint32_t len) {
        if constexpr (std::is_same_v<T, float> ) {
            Adds(dstTensor, srcTensor, 0.0f, len);
        } else {
            Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, len);
        }
    }

    __aicore__ inline void castFp32ToX(LocalTensor<T> dstTensor, LocalTensor<float> srcTensor, uint32_t len) {
        if constexpr (std::is_same_v<T, float> ) {
            Adds(dstTensor, srcTensor, 0.0f, len);
        } else if constexpr (std::is_same_v<T, half> ) {
            Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, len);
        } else {
            Cast(dstTensor, srcTensor, RoundMode::CAST_RINT, len);
        }
    }

    // 高维切分辅助函数: 构造BinaryRepeatParams并遍历maskLoop执行向量操作
    // <OP>: AvgPool3d使用Add, MaxPool3d使用Max
    __aicore__ inline void doHighDimOp(LocalTensor<float> dstBase,
                                        LocalTensor<float> src1Base,
                                        uint32_t src1Offset, uint32_t repeatTime) {
        constexpr uint32_t MAX_MASK = 64;
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = dstRepStride;
        params.src0RepStride = dstRepStride;
        params.src1RepStride = src1RepStride;

        for (uint32_t loop = 0; loop < maskLoopCount; loop++) {
            uint32_t cStart = loop * MAX_MASK;
            uint32_t curMask = MAX_MASK;
            if (cStart + curMask > alignC) curMask = alignC - cStart;
            // AvgPool3d: Add, MaxPool3d: Max
            <OP>(dstBase[cStart], dstBase[cStart],
                          src1Base[src1Offset + cStart],
                          curMask, static_cast<uint8_t>(repeatTime), params);
        }
    }

    __aicore__ inline void processOneOrMultiWindow(int64_t outputPointIdx, int64_t windowWNum) {

        // 初始化：AvgPool3d用0.0f累加，MaxPool3d用float最小负值做max比较（-3.4028235e38f）
        Duplicate(sumBufLocal, <INIT_VALUE>, windowWNum * alignC);
        // <INIT_VALUE>: AvgPool3d=0.0f, MaxPool3d=-3.4028235e38f
        PipeBarrier<PIPE_V>();

        // 遍历窗口
        for (uint32_t kd = 0; kd < kernelD; kd++) {
            // 计算d位置索引
            for (uint32_t kh = 0; kh < kernelH; kh++) {
                // 计算h位置索引

                uint32_t rowLen = windowWNum * strideW + kernelW - 1;

                // 计算w位置需要搬入数据的起止索引，搬入GM数据到UB,同时搬运windowWNum个窗口位置w方向的输入数据，每个W位置是C个元素,rightPadding将channels填充到alignC字节对齐
                CopyIn(offset, wEnd - wStart, channels, rightPadding);

                // 参考cast方法fp16、bf16升精度
                castXToFp32(castLocal, dataLocal, rowLen * alignC);
                PipeBarrier<PIPE_V>();

                if (isRepeat) [[likely]] {
                    // 快速路径: 所有W位置连续有效，直接按kw偏移做高维切分
                    for (uint32_t kw = 0; kw < kernelW; kw++) {
                        doHighDimOp(sumBufLocal, castLocal, kw * alignC, windowWNum);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    // 边界分段扫描: 需要判断每个W位置是否在有效范围内，分段累加/最大值计算
                    for (uint32_t kw = 0; kw < kernelW; kw++) {
                        int32_t segStart = -1;
                        for (uint32_t j = 0; j <= windowWNum; j++) {
                            int32_t iw = (j < windowWNum)
                                ? (int32_t)((startOw + j) * strideW + kw) - (int32_t)padW
                                : -1;
                            bool isValid = (j < windowWNum) && (iw >= 0 && (uint32_t)iw < inputW);

                            if (isValid && segStart < 0) {
                                segStart = (int32_t)j;
                            } else if (!isValid && segStart >= 0) {
                                uint32_t segLen = j - (uint32_t)segStart;
                                // segIw: 窗口segStart在kw位置的全局W坐标
                                int32_t segIw = (int32_t)((startOw + segStart) * strideW + kw) - (int32_t)padW;
                                uint32_t src1Start = (uint32_t)(segIw - clipWStart) * alignC;

                                doHighDimOp(sumBufLocal[(uint32_t)segStart * alignC],
                                            castLocal, src1Start, segLen);

                                segStart = -1;
                            }
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
        }

        // ==== AvgPool3d专用: 求均值 ====
        // MaxPool3d请删除以下if块
        // if (isSamePoolSize) { //poolsize窗口一致
        //     float poolSize = divisorOverride ? divisorOverride : (1.0f / static_cast<float>(kernelD * kernelH * kernelW));
        //     Muls(sumLocal, sumLocal, poolSize, windowWNum * alignC);
        // } else {
        //     // 遍历windowWNum，每个位置单独计算窗口大小，同一个W位置下的channel窗口除同一个数
        // }

        // 参考cast方法fp16、bf16恢复原有精度
        castFp32ToX(dataLocal, sumBufLocal, windowWNum * alignC);
        PipeBarrier<PIPE_V>();

        // 搬出UB数据到GM
        CopyOut(offset, windowWNum, channels);
    }

    __aicore__ inline void processSmallerOneWindow(int64_t outputPointIdx, int64_t windowWNum) {

        // 遍历窗口
        for (uint32_t loop = 0; loop < loops; loop++){  //单个窗口单个位置需要loops次处理完C个元素，每次处理len个元素，最后一次特殊处理，loops=1表示单个窗口单个处理C个值（此时len=C）
            // 初始化累加器
            Duplicate(sumBufLocal, <INIT_VALUE>, alignC);
            curProcessLen = (loop == loops - 1) ? (channels - (loops-1) * len) : len;

            for (uint32_t kd = 0; kd < kernelD; kd++) {
                // 计算d位置索引
                for (uint32_t kh = 0; kh < kernelH; kh++) {
                    // 计算h位置索引
                    for (uint32_t kw = 0; kw < kernelW; kw++) {
                        // 计算w位置索引

                        //搬入GM数据到UB
                        CopyIn(offset, 1, curProcessLen, rightPadding);

                        // 参考cast方法fp16、bf16升精度
                        castXToFp32(castLocal, dataLocal, curProcessLen);

                        // AvgPool3d: Add, MaxPool3d: Max
                        <OP>(sumBufLocal, sumBufLocal, castLocal, curProcessLen);

                        // 最大值索引(MaxPool3d专用): 需要分配mask buffer(32B对齐, ceil(C/64)*32B/窗口),
                        // C>64需分块循环Compare+Select, Select需ReinterpretCast<float>绕过vsel限制
                    }
                }
            }
            // 求均值-AvgPool3d专用, MaxPool3d删除以下两行
            // float poolSize = divisorOverride ? divisorOverride : (1.0f / static_cast<float>(kernelD * kernelH * kernelW));
            // Muls(sumBufLocal, sumBufLocal, poolSize, alignC);

            // 参考cast方法fp16、bf16恢复原有精度
            castFp32ToX(dataLocal, sumBufLocal, curProcessLen);

            // 搬出UB数据到GM
            CopyOut(offset, 1, curProcessLen);
        }
    }

private:
    TPipe pipe;

    GlobalTensor<T> inputGm, outputGm;
    TBuf<TPosition::VECCALC> shareBuf;
    LocalTensor<T> dataLocal;
    LocalTensor<float> castLocal;
    LocalTensor<float> sumBufLocal;  // MaxPool3d下为maxBufLocal

    uint32_t batchSize, channels;
    uint32_t inputD, inputH, inputW;
    uint32_t outputD, outputH, outputW;
    uint32_t kernelD, kernelH, kernelW;
    uint32_t strideD, strideH, strideW;
    uint32_t padD, padH, padW;
    uint32_t countIncludePad, ceilMode;
    uint32_t alignC;
    uint32_t maskLoopCount;
    uint8_t dstRepStride, src1RepStride, rightPadding;
    uint64_t windowWNum;
    uint64_t outputPointNum, outputPointOffset;
    bool isRepeat;
    bool isSamePoolSize;
};

  // 多数据类型入口: 通过宏统一参数列表, 仅 name 和 dtype 不同
  #define KERNEL_ENTRY(name, dtype) \
  extern "C" __global__ __aicore__ void name( \
      GM_ADDR input, GM_ADDR output, \
      uint32_t N, uint32_t C, uint32_t iD, uint32_t iH, uint32_t iW, \
      uint32_t oD, uint32_t oH, uint32_t oW, \
      uint32_t kD, uint32_t kH, uint32_t kW, \
      uint32_t sD, uint32_t sH, uint32_t sW, \
      uint32_t pD, uint32_t pH, uint32_t pW, \
      uint32_t dilD, uint32_t dilH, uint32_t dilW, \
      uint32_t countIncludePad, uint32_t ceilMode, \
      uint64_t divisorOverride, \
      uint64_t formerNum, uint64_t formerLength, uint64_t tailLength, \
      uint64_t windowWNum) \
  { \
      KernelPooling<dtype> op; \
      op.Init(input, output, N, C, iD, iH, iW, oD, oH, oW, \
              kD, kH, kW, sD, sH, sW, pD, pH, pW, \
              countIncludePad, ceilMode, \
              divisorOverride, \
              formerNum, formerLength, tailLength, windowWNum); \
      op.Process(); \
  }

  KERNEL_ENTRY(pool3d_fp32, float)
  KERNEL_ENTRY(pool3d_fp16, half)
  KERNEL_ENTRY(pool3d_bf16, bfloat16_t)

  // 有索引版本(MaxPool3d专用): 额外包含 GM_ADDR indices 参数和 returnIndices 标志
  // #define KERNEL_ENTRY_INDICES(name, dtype, ret_indices) \
  //     ... \
  //     op.Init(input, output, indices, ..., ceilMode, (uint32_t)ret_indices, ...); \
  // KERNEL_ENTRY_INDICES(pool3d_indices, float, 1)
  // KERNEL_ENTRY_INDICES(pool3d_fp16_indices, half, 1)
  // KERNEL_ENTRY_INDICES(pool3d_bf16_indices, bfloat16_t, 1)
