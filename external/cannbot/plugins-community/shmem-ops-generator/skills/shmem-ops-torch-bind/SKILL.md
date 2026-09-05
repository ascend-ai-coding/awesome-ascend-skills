---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-torch-bind
description: 将基于 SHMEM 的算子封装为 PyTorch CustomClass，生成 Python 测试脚本并验证。触发：用户明确需要为 SHMEM
  算子接入 Torch/PyTorch、生成绑定或执行 Torch 验证时使用。关键词：基于SHMEM的Torch接入、torch、PyTorch、CustomClass、torch_binding。
original-name: shmem-ops-torch-bind
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子 Torch 接入与验证

**Skill类型**：集成生成型（生成 Torch 绑定 C++ 代码 + Python 测试，编译验证）

> **中文写作要求**：Python 测试脚本的注释和输出信息使用中文。仅 API 名称、变量名、代码片段保留英文原文。

将已通过正确性验证的 SHMEM 算子封装为 PyTorch `CustomClassHolder`，生成多 PE Python 测试脚本，编译为共享 Torch 扩展并在 PyTorch 环境验证端到端正确性。

**构建模式与产物命名**：

| 模式 | Torch 目录 | 产物 `.so` |
| --- | --- | --- |
| `in_tree_example` | `${SHMEM_REPO}/examples/torch_binding/` | `aclshmem_torch.so` |
| `independent_project`（`custom-ops/`） | **`custom-ops/torch_binding/`（共享外层）** | **`shmem_custom_ops_torch.so`** |

custom-ops 布局详见 [references/GUIDE.md](references/GUIDE.md) 与 [references/custom-ops-torch-layout.md](references/custom-ops-torch-layout.md)。

## 必读资料

| 文件 | 阅读时机 | 用途 |
| --- | --- | --- |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | **读仓内 docs/examples 前** | 定位 `SHMEM_REPO` / `SKILLS_ROOT`；禁止 skill 相对路径链到仓 |
| `${SHMEM_REPO}/examples/torch_binding/src/torch_bindings.cpp` | **始终**（先定位 `SHMEM_REPO`） | Torch CustomClassHolder 完整参考实现（Manager / KVShuffle / AllGather） |
| `${SHMEM_REPO}/examples/torch_binding/include/torch_register.h` | **始终** | `REGISTER_SHMEM_OPS_CLASS` 宏定义和注册模式 |
| `${SHMEM_REPO}/examples/torch_binding/include/shmem_torch_kernel.h` | **始终** | kernel 函数声明模板 |
| `${SHMEM_REPO}/examples/torch_binding/CMakeLists.txt` | **始终** | Torch 扩展 CMake 模板（torch/torch_npu 查找、链接） |
| `${SHMEM_REPO}/examples/python_extension/torch_test/allgather.py` | **始终** | 多 PE Python 测试脚本参考（Manager 生命周期、compute、精度验证） |
| `${SHMEM_REPO}/examples/python_extension/torch_test/kv_shuffle.py` | **始终** | 复杂算子（多输入/block table）Python 测试参考 |
| `${SHMEM_REPO}/examples/python_extension/torch_test/README.md` | **始终** | 编译运行命令、参数说明 |

## 前置条件

- 算子已通过正确性验证（`shmem-ops-correctness-eval` 通过）
- 算子已通过代码走读（`shmem-ops-code-review` `interim` 模式通过）
- `design.md` 中 `meta.torch_required` 为 `true`（Phase 0 用户已确认；为 `false` 时编排器跳过本 skill）
- `design.md` 中的 interface 字段完整（input/output dtype、shape、语义）
- CANN 环境已激活、conda 环境已激活、SHMEM whl 已安装

### 运行时环境（MUST）

Torch 多 PE 测试与 C++ `scripts/run.sh` **共用 tcp store / SHMEM 会话**，须遵守：

- 每轮测试使用独立 `IPPORT` 和 `SHMEM_UID_SESSION_ID`（见 [env-setup.snippet.md](../shmem-ops-compile-debug/references/env-setup.snippet.md) 的 `setup_shmem_dynamic_endpoints`）
- 长时间运行的 `torch_test_*.py` 会占用端口；跑 C++ `scripts/run.sh` 前检查 `pgrep -f torch_test_`，或先结束旧进程
- Worker 内 **MUST** 预加载 `build/lib` 下依赖（`lib<op>_kernel.so`、`aclshmem_torch.so`），可用 `ctypes.CDLL(..., RTLD_GLOBAL)`，避免 `ImportError: libxxx.so not found`
- `torch_test_*.py` 的 `ip_port` 勿与仍在运行的其它 shmem 测试默认值冲突（勿全局写死 `27010` / `8899`）

## 工作流

```
步骤 1  提取算子接口契约
步骤 2  生成 C++ Torch 绑定代码
步骤 3  集成 CMake 构建
步骤 4  生成 Python 多 PE 测试脚本
步骤 5  编译与正确性验证
```

---

## 步骤 1：提取算子接口契约

从 `design.md` 和算子源码中提取：

| 信息 | 来源 | 用途 |
| --- | --- | --- |
| `op_name` | design.md DSL `meta.op_name` | 类名、文件名、注册名 |
| Input tensors | design.md DSL `interface.inputs` | C++ `compute()` 参数列表、Python 数据准备 |
| Output tensors | design.md DSL `interface.outputs` | C++ `compute()` 参数列表、Python golden 构造 |
| dtype 列表 | design.md DSL `interface.dtypes` | C++ dtype dispatch（参考 AllGather 的分支模式） |
| Shape 约束 | design.md DSL `interface.shape` | Python gen_data 形状、TORCH_CHECK 约束 |
| Kernel 入口函数 | `src/<op_name>_kernel.h` | C++ binding 调用的 kernel 函数签名 |
| 同步/状态资源 | design.md DSL `memory.state` | C++ 构造函数中 malloc 的 sync_ptr/state buffer |

---

## 步骤 2：生成 C++ Torch 绑定代码

### 输出文件

**in-tree example**（`${SHMEM_REPO}/examples/<op_name>/`）— 修改 `${SHMEM_REPO}/examples/torch_binding/`：

```text
${SHMEM_REPO}/examples/torch_binding/src/torch_bind_<op_name>.cpp
```

**custom-ops 独立工程** — **禁止**在 `<op_name>/torch_binding/` 建子工程；只向共享层追加文件：

```text
custom-ops/torch_binding/src/torch_bind_<op_name>.cpp
custom-ops/torch_binding/include/shmem_torch_kernel.h   # 追加 kernel 声明
```

Manager 类位于 `custom-ops/torch_binding/src/torch_bindings.cpp`（首次创建，后续算子复用）。

### 代码结构（MUST 遵循）

**`torch_bind_<op_name>.cpp` 生成规则**：

```cpp
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "shmem.h"
#include "shmem_torch_kernel.h"
#include "torch_register.h"

namespace ShmemOps {

class <OpName> : public torch::jit::CustomClassHolder {
public:
    <OpName>() : name_("<OpName>"), count_(0), fftsAddr_(util_get_ffts_config())
    {
        // 1. malloc sync/state buffer（参考 design.md memory.state）
        // 2. aclrtMemset 清零
    }

    ~<OpName>()
    {
        // 对应 release sync/state buffer
    }

    std::string get_name() const { return name_; }

    void compute(/* 按 design.md interface 列出全部 input/output tensor */)
    {
        // 1. TORCH_CHECK 校验 dtype/device/shape
        // 2. contiguous 处理
        // 3. 获取 data_ptr + stream
        // 4. dtype dispatch → 调用 kernel 模板函数
        // 5. count_++
    }

private:
    std::string name_;
    int32_t count_;
    uint64_t fftsAddr_;
    void* sync_ptr_;
};

}  // namespace ShmemOps

// 注册到 Torch CustomClass：torch.classes.ShmemOps.<OpName>
REGISTER_SHMEM_OPS_CLASS(<OpName>, compute, get_name);
```

### 关键约束

1. **类名**：PascalCase（如 `AllGather`、`KVShuffle`），与 `op_name`（snake_case）对应
2. **`Manager` 类必须已存在**：检查 `torch_bindings.cpp` 中是否已有 Manager 类。若无，必须先生成 Manager（含 `attr_init`、`finalize`、`malloc_tensor`、`malloc_like`、`free_tensor`）。Manager 是所有算子的前置依赖，负责 SHMEM 生命周期
3. **dtype dispatch**：参考 AllGather 的分支模式，按 `input_tensor.scalar_type()` 选择模板实例化
4. **`count_`**：每次 `compute()` 调用自增，传递给 kernel 作为 magic/sync epoch，防止 signal 复用导致的假通过
5. **`sync_ptr_`**：若算子内部使用 signal/wait，必须在构造函数中 malloc 并按 kernel 所需的 `block_dim` 和 `n_pes` 计算总大小
6. **`TORCH_CHECK`**：输入校验（dtype、device、shape）必须在 `compute()` 开头完成，失败抛出可读错误信息
7. **`contiguous()`**：输入 tensor 必须先转 contiguous 再获取 `storage().data()`，防止非连续 stride 导致数据错误

### Manager 类生成检查

如果 `${SHMEM_REPO}/examples/torch_binding/src/torch_bindings.cpp` 中不存在 Manager 类，**MUST** 先生成以下内容到同一个 `torch_bind_<op_name>.cpp` 中：

- `Manager` 类（继承 `torch::jit::CustomClassHolder`）
- 方法：`attr_init(pe, n_ranks, local_mem_size, ip_port)`、`finalize()`、`malloc_tensor(size)`、`malloc_like(tensor)`、`free_tensor(tensor)`
- 注册：`REGISTER_SHMEM_OPS_CLASS(Manager, attr_init, finalize, malloc_tensor, malloc_like, free_tensor, get_name);`

---

## 步骤 3：集成 CMake 构建

### in-tree example

修改 `${SHMEM_REPO}/examples/torch_binding/CMakeLists.txt`：追加 kernel target 到 `ACLSHMEM_ALL_KERNEL_TARGETS`，追加 `src/torch_bind_<op_name>.cpp`，产物 **`aclshmem_torch.so`**。

### custom-ops 独立工程（MUST）

修改 **`custom-ops/torch_binding/CMakeLists.txt`**（共享 CMake，见 [custom-ops-torch-layout.md](references/custom-ops-torch-layout.md)）：

```cmake
add_library(shmem_custom_ops_torch SHARED
    src/torch_bindings.cpp
    src/torch_bind_<op_name>.cpp
    # 后续算子仅追加 torch_bind_*.cpp
)

set_target_properties(shmem_custom_ops_torch PROPERTIES PREFIX "" OUTPUT_NAME "shmem_custom_ops_torch")

target_link_libraries(shmem_custom_ops_torch PRIVATE
    # 各 custom-ops/<op>/build/lib/lib<op>_kernel.so 或 imported target
    ${TORCH_LIBRARIES} torch_npu shmem tiling_api
)

# ABI MUST 与当前 python 的 torch 一致，否则 load_library 报 undefined symbol（如 torchInternalAssertFail）
# python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"  → True 则 ABI=1，False 则 ABI=0
target_compile_options(shmem_custom_ops_torch PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)  # 按上式探测结果改写
```

**NEVER** 在 `custom-ops/<op>/torch_binding/` 再生成独立 `add_library(aclshmem_torch ...)`。
**NEVER** 默认写死 `_GLIBCXX_USE_CXX11_ABI=0`；必须以运行时 `torch._C._GLIBCXX_USE_CXX11_ABI` 为准。

### 构建模式判断

| 场景 | 处理 |
| --- | --- |
| 算子位于 `${SHMEM_REPO}/examples/<op_name>/`（in-tree） | 修改 `${SHMEM_REPO}/examples/torch_binding/`，产物 `aclshmem_torch.so` |
| 算子位于 `custom-ops/<op_name>/` | 修改 **`custom-ops/torch_binding/`**，产物 **`shmem_custom_ops_torch.so`**；新增算子只加 bind cpp + CMake 一行 |

---

## 步骤 4：生成 Python 多 PE 测试脚本

### 输出文件

```
<op_name>/
├── scripts/
│   └── torch_test_<op_name>.py
```

### Python 测试脚本模板

```python
import os
from glob import glob
import multiprocessing
import argparse
import torch
import torch_npu

PES = 8
DTYPE = torch.float16
# 按 design.md interface 定义 INPUT_SHAPE / OUTPUT_SHAPE

def load_torch_library(lib_name):
    """搜索并加载 shmem_custom_ops_torch.so（custom-ops）或 aclshmem_torch.so（examples）"""
    lib_env = "LD_LIBRARY_PATH"
    lib_paths = os.environ.get(lib_env, "").split(os.pathsep)
    for path in lib_paths:
        if not os.path.isdir(path):
            continue
        match = glob(os.path.join(path, lib_name))
        if match:
            torch.ops.load_library(match[0])
            return
    raise FileNotFoundError(f"Library {lib_name} not found in {lib_env}")

def gen_data(pe):
    """生成 input 和 golden output，必须固定随机种子"""
    torch.manual_seed(42 + pe)
    # 按 design.md 语义生成 local_input 和 golden_output
    return local_input, golden_output

def worker(pe):
    load_torch_library('shmem_custom_ops_torch.so')  # custom-ops；examples 用 aclshmem_torch.so

    # 1. Manager 初始化
    manager = torch.classes.ShmemOps.Manager()
    torch_npu.npu.set_device(pe)
    local_mem_size = 1024 * 1024 * 1024  # 1GB
    ipports = "tcp://127.0.0.1:8662"
    manager.attr_init(pe, PES, local_mem_size, ipports)

    # 2. 创建算子实例
    op = torch.classes.ShmemOps.<OpName>()

    # 3. 生成数据 → 搬运到 NPU
    local_input, golden_output = gen_data(pe)
    input_npu = local_input.npu(non_blocking=True)
    output_npu = torch.zeros_like(golden_output).npu()

    torch_npu.npu.synchronize()

    # 4. 执行 compute
    op.compute(output_npu, input_npu)
    torch_npu.npu.synchronize()

    # 5. 精度验证
    output_cpu = output_npu.cpu()
    if not torch.equal(output_cpu, golden_output):
        print(f"[PE {pe}] FAIL: output mismatch!")
        print(f"[PE {pe}] golden[:10]: {golden_output.flatten()[:10]}")
        print(f"[PE {pe}] output[:10]: {output_cpu.flatten()[:10]}")
        raise AssertionError(f"[PE {pe}] <OpName> result check failed!")

    print(f"[PE {pe}] PASS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pes", type=int, default=8)
    args = parser.parse_args()
    PES = args.pes

    processes = [multiprocessing.Process(target=worker, args=(pe,))
                 for pe in range(PES)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All processes finished successfully")
```

### 关键约束

1. **固定随机种子**：`torch.manual_seed(42 + pe)`，确保每个 PE 的输入可复现
2. **Golden 构造**：在 CPU 上按算子语义计算期望输出，使用 float32（高精度）中间计算
3. **多 PE 启动**：使用 `torchrun --nproc-per-node <N>` 或 `multiprocessing.Process`
4. **精度验证**：使用 `torch.equal`（bitwise exact）或 `torch.allclose(rtol=..., atol=...)`
5. **设备操作前 `set_device(pe)`**：每个 worker 必须先 `torch_npu.npu.set_device(pe)`
6. **同步**：compute 前后必须 `torch_npu.npu.synchronize()`
7. **`--pes` 参数**：支持命令行控制卡数

---

## 步骤 5：编译与正确性验证

### 编译流程

**MUST 使用封装脚本**，禁止在 skill/交付文档中写裸 `cmake -S ... -B ...` 作为首选命令。

```bash
# 1. 环境（Phase 0 确认的 CANN_SET_ENV）
source "${CANN_SET_ENV}"
source "${SHMEM_REPO}/install/set_env.sh"

# 2. custom-ops kernel（custom-ops-entrypoints §1）
OP=<op_name>
cmake -S "${SHMEM_REPO}/custom-ops/${OP}" \
      -B "${SHMEM_REPO}/custom-ops/${OP}/build" \
      -DSHMEM_REPO="${SHMEM_REPO}"
cmake --build "${SHMEM_REPO}/custom-ops/${OP}/build" -j 8

# 3. Torch 扩展（custom-ops-entrypoints §4）
cmake -S "${SHMEM_REPO}/custom-ops/torch_binding" \
      -B "${SHMEM_REPO}/custom-ops/torch_binding/build"
cmake --build "${SHMEM_REPO}/custom-ops/torch_binding/build" -j 8
# 产物: custom-ops/torch_binding/build/lib/shmem_custom_ops_torch.so

# 4. in-tree example（仅 build_mode=in_tree_example）
# cd "${SHMEM_REPO}" && bash scripts/build.sh -python_example
# 产物: aclshmem_torch.so
```

### 运行验证

```bash
# 默认 8 卡
python scripts/torch_test_<op_name>.py

# 指定卡数
python scripts/torch_test_<op_name>.py --pes 2
```

### MUST检查

- [ ] `torch_bind_<op_name>.cpp` 包含 CustomClassHolder 子类和 `REGISTER_SHMEM_OPS_CLASS` 注册
- [ ] `compute()` 包含 TORCH_CHECK（dtype、device、shape）
- [ ] dtype dispatch 覆盖 design.md 声明的全部 dtype
- [ ] Manager 类已存在（含 `attr_init` / `finalize` / `malloc_tensor` / `malloc_like` / `free_tensor`）
- [ ] CMakeLists.txt 正确链接 kernel target、torch、torch_npu、shmem
- [ ] 产物命名正确：`aclshmem_torch.so`（examples）或 **`shmem_custom_ops_torch.so`（custom-ops）**
- [ ] custom-ops 模式下 bind 代码在 **`custom-ops/torch_binding/`**，不在 `<op>/torch_binding/`
- [ ] `torch_test_<op_name>.py` 包含 fixed seed gen_data + multi-PE worker + golden 精度验证
- [ ] Python 测试 2-PE smoke 通过
- [ ] Python 测试 8-PE 通过

**全部通过 → 进入下一阶段**

---

## 反模式（NEVER DO THESE）

- ❌ 跳过 Manager 类生成，直接使用算子 CustomClass
- ❌ compute() 中不调用 `input.contiguous()` 直接取 `storage().data()`
- ❌ 不同 pe 的 golden 使用相同随机种子（导致所有 PE 输出相同，假通过）
- ❌ 用 `torchrun` 时不在 worker 内调用 `load_torch_library`（torchrun 会 fork 子进程，需要在子进程内加载）
- ❌ 不传 `count_` 给 kernel 的 sync magic（可能导致 signal 复用假通过）
- ❌ `torch_test_*.py` 与 C++ `scripts/run.sh` 共用固定 `IPPORT`/`SHMEM_UID_SESSION_ID` 且并行运行（导致 shmem init 失败、C++ 输出全 0）
- ❌ custom-ops 算子在 `<op>/torch_binding/` 内重复建完整 Torch 工程或产出 `aclshmem_torch.so`
- ❌ custom-ops 与 examples 共用同一 `.so` 文件名（易混淆、难并行加载）
- ❌ 在 Python 侧用 SHMEM 底层 API（`aclshmem_putmem`/`aclshmem_getmem` 等），必须通过 CustomClass 封装调用
- ❌ 编译时不 link `torch_npu`
- ❌ Python 测试只跑单 PE
- ❌ Phase 5.5 只验证部分算子 Torch 测试即标完成（批次内 **全部** `torch_test_*.py` 8PE PASS）
- ❌ 在文档/回复中首选裸 `cmake -S custom-ops/torch_binding ...`（**MUST** 用 [custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §4 Torch 编译）
