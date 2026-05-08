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
                                 uint64_t windowWNum) { //...用例信息和host中tiling切分信息传递

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
        inputGm.SetGlobalBuffer((__gm__ T*)input, ...);
        outputGm.SetGlobalBuffer((__gm__ T*)output, ...);

        // UB空间分配
        uint32_t rowElements = windowWNum * strideW + kernelW - 1;
        uint64_t dataSize = (uint64_t)rowElements * alignC;
        uint64_t castSize = (uint64_t)rowElements * alignC;
        uint64_t sumSize  = (uint64_t)windowWNum * alignC;

        pipe.InitBuffer(shareBuf, dataSize * sizeof(T) + castSize * sizeof(float) + sumSize * sizeof(float));
        dataLocal = shareBuf.Get<T>(dataSize);
        castLocal = shareBuf.GetWithOffset<float>(castSize, dataSize * sizeof(T));
        sumBufLocal = shareBuf.GetWithOffset<float>(sumSize, dataSize * sizeof(T) + castSize * sizeof(float));
        // ...maxpool额外增加需要indice空间,shareBuf.GetWithOffset<T>(元素个数, 起始字节偏移)

        this->isRepeatSum = (padW == 0 && !ceilMode); // isRepeatSum: padW==0且非ceilMode时，所有W方向数据连续有效，跳过边界分段扫描和Duplicate清零
        this->isSamePoolSize = divisorOverride || (countIncludePad || padW == 0) && !ceilMode; // isSamePoolSize: countIncludePad或padW==0且非ceilMode时，所有窗口poolCount相同，直接Muls批量除
    }

    __aicore__ inline void Process() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t totalBlocks = AscendC::GetBlockNum();
        int64_t curWindowWNum = windowWNum;
        for (int64_t outputPointIdx = outputPointOffset, count = 0;
                outputPointIdx < outputPointOffset + outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {

                // windowNum处理越界，跨行时截断，每次最多处理w方向同一行的的窗口
                curWindowWNum = (count + windowWNum) < outputPointNum ? windowWNum : outputPointNum - count;
                int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % outputShape.W;
                curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                                ? curWindowWNum - newRowWindowWNum : curWindowWNum;

                if (){  // 同时处理一个/多个windows个窗口数据，当kw>sw时可复用GM数据
                    processOneOrMultiWindow(outputPointIdx, curWindowWNum);
                } else{// 单次不足以处理一个窗口数据，每个窗口每个位置单独累加/最大值计算处理，兜底方案
                    processSmallerOneWindow(outputPointIdx, curWindowWNum);
                }
            }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
        DataCopyExtParams copyParams{blockCount, blockLen * sizeof(T), 0, 0, 0}; //blockCount是搬运次数，blockLen * sizeof(T)是每次搬运字节长度可32字节不对齐（会自动对齐）
        DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};    //rightPadding是blockCount每次填充元素个数（但所占字节数不能超过32字节），processOneOrMultiWindow下rightPadding=alignC-C
        AscendC::DataCopyPad(dataLocal, inputGm[offset], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint16_t blockCount, uint32_t blockLen) {
        // 注意搬出时长度不要超过GM内容空间，如果超过应该截断
        DataCopyExtParams copyParamsOut{blockCount, blockLen * sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[outputOffset], dataLocal, copyParamsOut);
    }

    __aicore__ inline void castXToFp32(LocalTensor<float> dstTensor, LocalTensor<T> srcTensor, uint32_t len) {
        if constexpr (std::is_same_v<T, float> ) {
            AscendC::Add(dstTensor, srcTensor, 0.0f, len);
        } else {
            AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, len);
        }
    }

    __aicore__ inline void castFp32ToX(LocalTensor<float> dstTensor, LocalTensor<T> srcTensor, uint32_t len) {
        if constexpr (std::is_same_v<T, float> ) {
            AscendC::Add(dstTensor, srcTensor, 0.0f, len);
        } else if constexpr (std::is_same_v<T, half> ) {
            AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, len);
        } else {
            AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_RINT, len);
        }
    }

    __aicore__ inline void processOneOrMultiWindow(int64_t outputPointIdx, int64_t windowNum) {

        // 初始化累加器
        AscendC::Duplicate(sumBufLocal, 0.0f, windowWNum * alignC);
        PipeBarrier<PIPE_V>();

        // 计算右填充: channels → alignC 对齐
        uint8_t rightPadding = static_cast<uint8_t>(alignC - channels);

        // 遍历窗口
        for (uint32_t kd = 0; kd < kernelD; kd++) {
            // 计算d位置索引
            for (uint32_t kh = 0; kh < kernelH; kh++) {
                // 计算h位置索引

                uint32_t rowLen = windowNum * strideW + kernelW - 1;

                // isRepeatSum优化: padW==0且非ceilMode时，整行数据连续有效，无需预先清零
                if (!isRepeatSum) {
                    AscendC::Duplicate(dataLocal, (T)(0), rowLen * alignC);
                    PipeBarrier<PIPE_V>();
                }

                // 计算w位置需要搬入数据的起止索引，搬入GM数据到UB,同时搬运windowWNum个窗口位置w方向的输入数据，每个W位置是C个元素,rightPadding将channels填充到alignC字节对齐
                CopyIn(offset, wEnd - wStart, channels, rightPadding); 

                // 参考cast方法fp16、bf16升精度
                castXToFp32(castLocal, dataLocal, rowLen * alignC);
                PipeBarrier<PIPE_V>();

                // 累加: Add高维切分优化
                {
                    constexpr uint32_t MAX_MASK_FP32 = 64;
                    uint32_t maskLoopCount = (alignC + MAX_MASK_FP32 - 1) / MAX_MASK_FP32;
                    uint32_t dstRepStride = alignC / 8;
                    uint32_t src1RepStride = strideW * alignC / 8;

                    if (isRepeatSum) [[likely]] {
                        // 快速路径: 所有W位置连续有效，直接按kw偏移做Add高维切分
                        for (uint32_t kw = 0; kw < kernelW; kw++) {
                            for (uint32_t loop = 0; loop < maskLoopCount; loop++) {
                                uint32_t cStart = loop * MAX_MASK_FP32;
                                uint32_t curMask = MAX_MASK_FP32;
                                if (cStart + curMask > alignC) curMask = alignC - cStart;

                                AscendC::BinaryRepeatParams params;
                                params.dstBlkStride = 1;
                                params.src0BlkStride = 1;
                                params.src1BlkStride = 1;
                                params.dstRepStride = static_cast<uint8_t>(dstRepStride);
                                params.src0RepStride = static_cast<uint8_t>(dstRepStride);
                                params.src1RepStride = static_cast<uint8_t>(src1RepStride);

                                uint32_t src1Offset = kw * alignC + cStart;
                                AscendC::Add(sumBufLocal[cStart], sumBufLocal[cStart],
                                             castLocal[src1Offset],
                                             curMask, static_cast<uint8_t>(windowNum), params);
                            }
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        // 边界分段扫描: 需要判断每个W位置是否在有效范围内，分段累加
                        for (uint32_t kw = 0; kw < kernelW; kw++) {
                            int32_t segStart = -1;
                            for (uint32_t j = 0; j <= windowNum; j++) {
                                int32_t iw = (j < windowNum)
                                    ? (int32_t)((startOw + j) * strideW + kw) - (int32_t)padW
                                    : -1;
                                bool isValid = (j < windowNum) && (iw >= 0 && (uint32_t)iw < inputW);

                                if (isValid && segStart < 0) {
                                    segStart = (int32_t)j;
                                } else if (!isValid && segStart >= 0) {
                                    uint32_t segLen = j - (uint32_t)segStart;
                                    uint32_t src1Start = ((uint32_t)segStart * strideW + kw) * alignC;

                                    for (uint32_t loop = 0; loop < maskLoopCount; loop++) {
                                        uint32_t cStart = loop * MAX_MASK_FP32;
                                        uint32_t curMask = MAX_MASK_FP32;
                                        if (cStart + curMask > alignC) curMask = alignC - cStart;

                                        AscendC::BinaryRepeatParams params;
                                        params.dstBlkStride = 1;
                                        params.src0BlkStride = 1;
                                        params.src1BlkStride = 1;
                                        params.dstRepStride = static_cast<uint8_t>(dstRepStride);
                                        params.src0RepStride = static_cast<uint8_t>(dstRepStride);
                                        params.src1RepStride = static_cast<uint8_t>(src1RepStride);

                                        AscendC::Add(
                                            sumBufLocal[(uint32_t)segStart * alignC + cStart],
                                            sumBufLocal[(uint32_t)segStart * alignC + cStart],
                                            castLocal[src1Start + cStart],
                                            curMask, static_cast<uint8_t>(segLen), params);
                                    }
                                    segStart = -1;
                                }
                            }
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
        }

        // 求均值
        if (isSamePoolSize) { //poolsize窗口一致
            float poolSize = divisorOverride ? divisorOverride : (1.0f / static_cast<float>(kd * kh * kw));
            AscendC::Muls(sumLocal, sumLocal, poolSize, windowWNum * alignC);
        } else {
            // 遍历windowWNum，每个位置单独计算窗口大小，同一个W位置下的channel窗口除同一个数
        }

        // 参考cast方法fp16、bf16恢复原有精度
        castFp32ToX(dataLocal, sumBufLocal, windowNum * alignC);
        PipeBarrier<PIPE_V>();

        // 搬出UB数据到GM
        CopyOut(offset, windowNum, channels);
    }

    __aicore__ inline void processSmallerOneWindow(int64_t outputPointIdx, int64_t windowNum) {

        // 遍历窗口
        for (uint32_t loop = 0; loop < loops; loop++){  //单个窗口单个位置需要loops次处理完C个元素，每次处理len个元素，最后一次特殊处理，loops=1表示单个窗口单个处理C个值（此时len=C）
            // 初始化累加器
            AscendC::Duplicate(sumBufLocal, 0.0f, alignC);
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

                        // 累加
                        AscendC::Add(sumBufLocal, sumBufLocal, castLocal, curProcessLen);
                    }
                }
            }
            // 平均并写出
            float poolSize = divisorOverride ? divisorOverride : (1.0f / static_cast<float>(kd * kh * kw));
            AscendC::Muls(sumBufLocal, sumBufLocal, poolSize, alignC);

            // 参考cast方法fp16、bf16恢复原有精度
            castFp32ToX(dataLocal, sumBufLocal, curProcessLen);

            // 搬出UB数据到GM
            CopyOut(offset, 1, curProcessLen);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<T> inputGm, outputGm;
    TBuf<TPosition::VECCALC> shareBuf;
    LocalTensor<T> dataLocal;
    LocalTensor<float> castLocal;
    LocalTensor<float> sumBufLocal;

    uint32_t batchSize, channels;
    uint32_t inputD, inputH, inputW;
    uint32_t outputD, outputH, outputW;
    uint32_t kernelD, kernelH, kernelW;
    uint32_t strideD, strideH, strideW;
    uint32_t padD, padH, padW;
    uint32_t countIncludePad, ceilMode;
    uint32_t alignC;
    uint64_t formerNum, formerLength, tailLength, windowWNum;
    uint64_t outputPointNum, outputPointOffset;
    bool isRepeatSum;
    bool isSamePoolSize;
};

extern "C" __global__ __aicore__ void pool3d_fp32(GM_ADDR input, GM_ADDR output,
    uint32_t N, uint32_t C, uint32_t iD, uint32_t iH, uint32_t iW,
    uint32_t oD, uint32_t oH, uint32_t oW,
    uint32_t kD, uint32_t kH, uint32_t kW,
    uint32_t sD, uint32_t sH, uint32_t sW,
    uint32_t pD, uint32_t pH, uint32_t pW,
    uint32_t countIncludePad, uint32_t ceilMode,
    uint64_t formerNum, uint64_t formerLength, uint64_t tailLength,
    uint64_t windowWNum)
{
    KernelPooling<float> op;  //float可替换half、bfloat16_t适配不同数据类型
    op.Init(input, output, N, C, iD, iH, iW, oD, oH, oW,
            kD, kH, kW, sD, sH, sW, pD, pH, pW, countIncludePad, ceilMode,
            formerNum, formerLength, tailLength, windowWNum);
    op.Process();
}
