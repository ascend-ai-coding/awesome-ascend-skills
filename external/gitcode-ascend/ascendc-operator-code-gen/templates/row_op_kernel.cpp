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
// 行处理算子 op_kernel 模板
// 适用: LayerNorm, Softmax, BatchNorm 等按行/维度归约算子
// 使用: 复制到 csrc/ops/<op_name>/op_kernel/<op_name>.cpp，
//       替换 <op_name>/<OpName> 占位符，修改 Compute 逻辑
// ============================================================

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class Kernel<OpName> {
public:
    __aicore__ inline Kernel<OpName>() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t totalRows, int64_t dimLength, int64_t dimLengthAlign,
                                int64_t formerNum, int64_t formerLength, int64_t tailLength
                                /* , 其他算子特有参数 */)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t rowOffset;

        if (blockIdx < formerNum) {
            this->blockRows = formerLength;
            rowOffset = formerLength * blockIdx;
        } else {
            this->blockRows = tailLength;
            int64_t tailIdx = blockIdx - formerNum;
            rowOffset = formerLength * formerNum + tailLength * tailIdx;
        }

        xGm.SetGlobalBuffer((__gm__ half *)x + rowOffset * dimLengthAlign,
                             this->blockRows * dimLengthAlign);
        yGm.SetGlobalBuffer((__gm__ half *)y + rowOffset * dimLengthAlign,
                             this->blockRows * dimLengthAlign);

        this->dimLength = dimLength;
        this->dimLengthAlign = dimLengthAlign;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, dimLengthAlign * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, dimLengthAlign * sizeof(half));
        // FP32 临时缓冲 (升精度计算用)
        pipe.InitBuffer(tmpBufA, dimLengthAlign * sizeof(float));
        pipe.InitBuffer(tmpBufB, dimLengthAlign * sizeof(float));
        // 如需更多临时缓冲:
        // pipe.InitBuffer(tmpBufC, dimLengthAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = 0; row < this->blockRows; ++row) {
            CopyIn(row);
            Compute(row);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t row)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[row * this->dimLengthAlign], this->dimLengthAlign);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t row)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

        AscendC::LocalTensor<float> bufA = tmpBufA.Get<float>();
        AscendC::LocalTensor<float> bufB = tmpBufB.Get<float>();

        int32_t dimLen = static_cast<int32_t>(this->dimLength);
        int32_t dimLenAlign = static_cast<int32_t>(this->dimLengthAlign);

        // ==== 在此填充计算逻辑 ====
        //
        // 升精度:
        //   AscendC::Cast(bufA, xLocal, AscendC::RoundMode::CAST_NONE, dimLenAlign);
        //
        // 归约示例 (ReduceSum):
        //   AscendC::Adds(bufB, bufA, 0.0f, dimLenAlign);  // 备份 (归约可能修改 src)
        //   AscendC::ReduceSum<float, true>(tmpResult, bufB, bufA, dimLen);
        //   float sumVal = tmpResult.GetValue(0);
        //
        // 降精度回原类型:
        //   AscendC::Cast(yLocal, bufA, AscendC::RoundMode::CAST_RINT, dimLenAlign);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t row)
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[row * this->dimLengthAlign], yLocal, this->dimLengthAlign);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufA, tmpBufB;
    AscendC::GlobalTensor<half> xGm, yGm;
    int64_t blockRows;
    int64_t dimLength;
    int64_t dimLengthAlign;
};

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR x, GM_ADDR y,
                                                 int64_t totalRows, int64_t dimLength,
                                                 int64_t dimLengthAlign,
                                                 int64_t formerNum, int64_t formerLength,
                                                 int64_t tailLength
                                                 /* , 其他算子特有参数 */)
{
    Kernel<OpName> op;
    op.Init(x, y, totalRows, dimLength, dimLengthAlign,
            formerNum, formerLength, tailLength
            /* , 其他算子特有参数 */);
    op.Process();
}
