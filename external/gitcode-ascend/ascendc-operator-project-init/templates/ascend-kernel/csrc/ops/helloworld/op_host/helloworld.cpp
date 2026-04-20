// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_kernel_helper.h"

#include "aclrtlaunch_helloworld.h"

namespace ascend_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y)
{
    /* create a result tensor */
    at::Tensor z = at::empty_like(x);

    /* define the block dim */
    uint32_t blockDim = 8;

    /* memory size */
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }

    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(helloworld, blockDim, x, y, z, totalLength);
    return z;
}

}  // namespace ascend_kernel
