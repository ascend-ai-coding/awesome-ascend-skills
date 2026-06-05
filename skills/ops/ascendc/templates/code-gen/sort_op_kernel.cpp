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
// sort类算子 op_host 参考代码
// 适用: sort、topk等需要行内排序的算子
// ============================================================
#include "kernel_operator.h"

class KernelMySort {
    using TBuf = AscendC::TBuf<AscendC::TPosition::VECCALC>;

public:
    __aicore__ inline KernelMySort() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                GM_ADDR workspace,
                                ...) {
    }

    __aicore__ inline void Process() {
        int64_t startRow = 0, rows = 0;
        if (blockIdx_ < formerNum_) {
            startRow = ...;
            rows = ...;
        } else{
            startRow = ...;
            rows = ...;
        }
        for (int64_t i = 0; i < rows; ++i){
            ProcessOneRow(startRow + i);
        }
    }
    
    // 添加处理单行的逻辑：分块排序、树型归并、分离等
};


extern "C" __global__ __aicore__ void my_sort(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                               GM_ADDR workspace,
                                               int64_t totalRows, int64_t usedCoreNum,
                                               ...) {
    KernelMySort op;
    op.Init(x, values, indices, workspace, ...);
    op.Process();
}