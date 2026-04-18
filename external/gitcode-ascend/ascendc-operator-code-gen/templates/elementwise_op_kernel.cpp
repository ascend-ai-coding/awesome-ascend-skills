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
// Elementwise 算子 op_kernel 模板
// 适用: ReLU, GELU, Add, Mul 等逐元素算子
// 使用: 复制到 csrc/ops/<op_name>/op_kernel/<op_name>.cpp，
//       替换 <op_name>/<OpName> 占位符，修改 Compute 逻辑
// ============================================================

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Kernel<OpName> {
public:
    __aicore__ inline Kernel<OpName>() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t formerNum, int64_t formerLength,
                                int64_t tailLength, int64_t tileLength)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();

        if (blockIdx < formerNum) {
            this->blockLength = formerLength;
            int64_t offset = formerLength * blockIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);
        } else {
            this->blockLength = tailLength;
            int64_t tailIdx = blockIdx - formerNum;
            int64_t offset = formerLength * formerNum + tailLength * tailIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);
        }

        this->tileLength = tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
        // 如需升精度计算 (FP16/BF16), 添加 FP32 临时缓冲:
        // pipe.InitBuffer(tmpBuf, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;

        int64_t alignNum = 32 / static_cast<int64_t>(sizeof(T));
        int64_t alignedTailLen = ((tailTileLength + alignNum - 1) / alignNum) * alignNum;

        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (tileNum > 0) {
            CopyIn(tileNum - 1, alignedTailLen);
            Compute(tileNum - 1, alignedTailLen);
            CopyOut(tileNum - 1, alignedTailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], curTileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // ==== 在此填充计算逻辑 ====
        // 简单示例 (ReLU):
        //   AscendC::Relu(yLocal, xLocal, curTileLength);
        //
        // FP16/BF16 升精度示例:
        //   AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        //   AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, curTileLength);
        //   // ... FP32 计算 ...
        //   AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, curTileLength);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, curTileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;  // FP32 临时缓冲
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    int64_t blockLength;
    int64_t tileLength;
};

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR x, GM_ADDR y,
                                                 int64_t formerNum, int64_t formerLength,
                                                 int64_t tailLength, int64_t tileLength,
                                                 int64_t dtypeSize)
{
    if (dtypeSize == 2) {
        Kernel<OpName><half> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    } else {
        Kernel<OpName><float> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
}
