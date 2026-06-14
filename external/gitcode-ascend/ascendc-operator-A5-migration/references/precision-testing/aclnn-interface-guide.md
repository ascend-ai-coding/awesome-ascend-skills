# aclnn 接口调用指南

## 概述

aclnn（Ascend Computing Language Neural Network）是 CANN 平台的标准算子调用接口。在 A5 迁移的精度验证阶段，**必须**通过 aclnn 接口调用算子。

**唯一标准方式**：通过 `torch_aclnn_helper.h` 中的 `EXEC_NPU_CMD` 宏构建 PyTorch C++ Extension，Python 层通过 `torch.ops.npu.<op_name>(...)` 调用。**禁止**使用 `import ascend_kernel`、Pybind 直接封装、手动 aclnn C API 调用等其他方式。

**核心认知**：Python 层面**不存在** `torch.ops.aclnn.*` 命名空间。aclnn 接口在 C++ 层面通过 `EXEC_NPU_CMD(aclnnXxx, ...)` 宏封装，Python 层面通过 `torch.ops.npu.<op_name>(...)` 调用。

## 调用链路全景

```
Python 层:  torch.ops.npu.<op_name>(x, ...)
    ↓ (PyTorch C++ Extension 注册)
C++ 层:     EXEC_NPU_CMD(aclnnXxx, x, ..., result)
    ↓ (宏展开)
C 层:       aclnnXxxGetWorkspaceSize(...)  →  计算工作空间大小
            aclnnXxxExecute(...)            →  执行算子计算
```

## C 层面两段式调用（仅供理解）

aclnn 接口在 C 层面采用统一的两段式调用模式：

1. **aclnnXxxGetWorkspaceSize**：计算工作空间大小，确定执行资源
2. **aclnnXxxExecute**：执行算子计算

```c
// C 层面（仅供理解，Python 测试不需要直接调用）
aclnnStatus aclnnXxxGetWorkspaceSize(
    aclTensor *x, ...,           // 输入
    aclTensor *out,              // 输出
    uint64_t *workspaceSize      // 工作空间大小（输出）
);

aclnnStatus aclnnXxxExecute(
    void *workspace,             // 工作空间地址
    uint64_t workspaceSize,      // 工作空间大小
    aclOpExecutor *executor,     // 执行器
    aclTensor *x, ...,           // 输入
    aclTensor *out               // 输出
);
```

## C++ 层面 EXEC_NPU_CMD 封装

C++ 层面通过 `EXEC_NPU_CMD` 宏封装两段式调用，定义在 `torch_aclnn_helper.h`：

```cpp
// 宏定义核心逻辑（简化版）
#define EXEC_NPU_CMD(aclnn_api, ...)                                          \
    do {                                                                      \
        auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize"); \
        auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                    \
        auto converted_params = ConvertTypes(__VA_ARGS__, ...);               \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params); \
        /* ... workspace 分配 ... */                                          \
        auto acl_call = [...]() -> int {                                      \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr); \
            return opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
        };                                                                    \
        at_npu::native::OpCommand cmd;                                        \
        cmd.Name(#aclnn_api);                                                 \
        cmd.SetCustomHandler(acl_call);                                       \
        cmd.Run();                                                            \
    } while (false)
```

**⚠️ 重要：aclnn L2 接口是多路径调度器，不是简单的 1:1 映射**

aclnn L2 接口（如 `aclnnGather`）内部通常包含多路径调度逻辑，根据芯片型号（SOC version）和输入特征（shape、dtype、contiguity 等）动态选择不同的 L0 底层算子。**不能假设 aclnn 接口名与底层 L0 算子名一一对应。**

典型示例（`aclnnGather`）：
```
aclnnGatherGetWorkspaceSize(...)
  → CalGather() 调度逻辑
    ├─ 910B/910_93 + 满足条件 → l0op::GatherElementsV2
    ├─ 950 (RegBase)           → l0op::GatherElements (gather_elements_apt 内核)
    ├─ MOE 场景                → l0op::GatherV2
    └─ 其他                    → l0op::GatherElements (AiCore/AiCPU)
```

**MUST 通过阅读 `op_host/op_api/aclnn_*.cpp` 源码确认 950 上的实际调度路径。**

**实际使用示例**（来自 `csrc/aclnn/avg_pool3d.cpp`）：

```cpp
EXEC_NPU_CMD(aclnnAvgPool3d, self, kernel_size, stride, padding,
              ceil_mode, count_include_pad, divisor_override, result);
```

## Python 层调用方式

### 唯一标准方式：torch.ops.npu 命名空间

算子通过 C++ Extension 注册到 `torch.ops.npu` 命名空间后，Python 层直接调用：

```python
import torch
import torch_npu

# 单输入算子
output = torch.ops.npu.acosh(x)

# 多输入 + 属性参数
output = torch.ops.npu.rms_norm(x, gamma, epsilon=1e-6)

# 多输出算子（返回 tuple）
output1, output2 = torch.ops.npu.some_op(x, y)
```

**注册方式**：算子通过 C++ Extension 中的 `TORCH_LIBRARY_FRAGMENT(npu)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1)` 宏注册：

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("avg_pool3d(Tensor self, ...)", &avg_pool3d);
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("avg_pool3d", TORCH_FN(ascend_kernel::avg_pool3d));
}
```

注册后 Python 层可通过 `torch.ops.npu.<op_name>(...)` 调用。

**当 `torch.ops.npu.<op_name>` 不存在时**：需要构建自定义 PyTorch 绑定（参见 [`pytorch-binding-build-guide.md`](pytorch-binding-build-guide.md)），构建完成后通过 `torch.ops.load_library()` 加载 `.so`，再通过 `torch.ops.npu.<op_name>(...)` 调用。

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import torch
import torch_npu

# 加载自定义算子绑定
torch.ops.load_library("<path_to>/libcustom_ops.so")

# 调用算子
output = torch.ops.npu.<op_name>(x, ...)
```

## 如何确认算子的 Python 调用方式

### 方法 1：查看算子安装后的注册信息

```python
import torch
import torch_npu

# 列出 npu 命名空间下的算子
npu_ops = [op for op in dir(torch.ops.npu) if not op.startswith('_')]
print(f"Available npu ops: {len(npu_ops)}")

# 检查特定算子是否可用
def check_op_available(namespace, op_name):
    try:
        ns = getattr(torch.ops, namespace)
        getattr(ns, op_name)
        return True
    except AttributeError:
        return False

print(check_op_available("npu", "acosh"))
```

### 方法 2：查看算子包的 aclnn 头文件

```bash
# 在 CANN 安装目录中查找 aclnn 头文件
find ${ASCEND_HOME_PATH} -name "aclnn_*.h" | grep <op_name>

# 查看头文件中的接口定义（确认 C 层面的接口名）
cat ${ASCEND_HOME_PATH}/include/aclnn_<op_name>.h
```

### 方法 3：查看算子的 _def.cpp 中的 ACLNNTYPE

```python
# 从算子的 _def.cpp 中提取 ACLNNTYPE
# 例如：ACLNNTYPE aclnn_acosh
# → C++ 层 EXEC_NPU_CMD(aclnnAcosh, ...)
# → Python 层 torch.ops.npu.acosh(x)
```

## NPU 调用方式选择

**所有迁移算子必须使用 EXEC_NPU_CMD 构建的 `torch.ops.npu.<op_name>(...)` 调用，禁止使用 PyTorch 原生同名接口。**

```
算子迁移完成
  └─ 必须使用 EXEC_NPU_CMD 构建自定义 PyTorch 绑定
     ├─ 禁止使用 PyTorch 原生同名接口（如 torch.gather, torch.acosh 等）
     │   原因：torch_npu 内置分发路径不受迁移控制，950 上可能走非迁移路径
     └─ 构建完成后通过 torch.ops.load_library() + torch.ops.npu.<op_name>(...) 调用
```

### 为什么禁止使用 PyTorch 原生同名接口

| 场景 | PyTorch 原生接口（禁止） | EXEC_NPU_CMD 绑定（必须） |
|------|------------------------|--------------------------|
| `torch.gather(x, dim, index)` | 走 `torch_npu` 内置分发，950 上可能走非迁移路径 | 走 `aclnnGather`，950 上走 `gather_elements_apt`（迁移后路径） |
| `torch.acosh(x)` | 走 `torch_npu` 内置分发，无法确认 950 调度路径 | 走 `aclnnAcosh`，确认 950 上走迁移后路径 |
| `torch.sigmoid(x)` | 同上 | 同上 |

**核心原因**：PyTorch 原生接口通过 `torch_npu` 内置分发机制调用 aclnn，950 上的调度路径由 `torch_npu` 版本决定，**不受迁移控制**。使用 EXEC_NPU_CMD 绑定可以确保走迁移后的 950 专用 aclnn 路径。

### 如何确认算子的调用方式

```python
import torch
import torch_npu

# 方法 1：尝试 PyTorch 原生接口
try:
    result = torch.gather(x, dim, index)
    print("torch.gather works on NPU")
except Exception as e:
    print(f"torch.gather failed: {e}")

# 方法 2：检查 torch.ops.npu 命名空间
try:
    getattr(torch.ops.npu, "fast_gelu")
    print("torch.ops.npu.fast_gelu is available")
except AttributeError:
    print("torch.ops.npu.fast_gelu is NOT available")
```

## 数据生成规范

**核心原则**：所有测试数据**必须在 CPU 上生成，再通过 `.to(device)` 搬运到 NPU**。

### 原因

部分 PyTorch 函数在 NPU 上不可用或行为异常：
- `torch.arange`：NPU 上可能触发 `aclnnArange` 失败
- `torch.randint`：NPU 上可能触发 `aclnnRandom` 失败
- `torch.full`：部分 dtype 在 NPU 上不支持

### 正确做法

```python
# ✅ 正确：CPU 生成 → .to(device) 搬运
def _make_random(shape, dtype, device):
    x = torch.rand(shape, dtype=torch.float32) * (HIGH - LOW) + LOW
    return x.to(dtype=dtype, device=device)

def _make_arange(shape, dtype, device):
    x = torch.arange(1, int(np.prod(shape)) + 1, dtype=torch.float32).reshape(shape)
    return x.to(dtype=dtype, device=device)

def _make_constant(shape, value, dtype, device):
    x = torch.full(shape, float(value), dtype=torch.float32)
    return x.to(dtype=dtype, device=device)

def _make_index(shape, max_val, device):
    idx = torch.randint(0, max_val, shape, dtype=torch.int64)
    return idx.to(device=device)
```

### 错误做法

```python
# ❌ 错误：直接在 NPU 上生成
x = torch.arange(1, 129, dtype=torch.float32, device="npu:0")  # 可能失败
x = torch.rand(shape, dtype=torch.float32, device="npu:0")      # 部分场景失败
```

## 精度测试中的调用模板

### 单输入算子

```python
def npu_call(x, **kwargs):
    return torch.ops.npu.<op_name>(x, **kwargs)

def cpu_reference(x, dtype, **kwargs):
    return torch.<pytorch_op>(x.cpu().float(), **kwargs).to(dtype)
```

### 多输入算子

```python
def npu_call(x, y, **kwargs):
    return torch.ops.npu.<op_name>(x, y, **kwargs)

def cpu_reference(x, y, dtype, **kwargs):
    return torch.<pytorch_op>(x.cpu().float(), y.cpu().float(), **kwargs).to(dtype)
```

### 多输出算子

```python
def npu_call(x, **kwargs):
    result = torch.ops.npu.<op_name>(x, **kwargs)
    return result  # tuple of tensors

def cpu_reference(x, dtype, **kwargs):
    result = torch.<pytorch_op>(x.cpu().float(), **kwargs)
    if isinstance(result, tuple):
        return tuple(r.to(dtype) for r in result)
    return result.to(dtype)
```

### 需要构建自定义绑定的算子

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import torch
import torch_npu

# 加载自定义算子绑定
torch.ops.load_library("<path_to>/libcustom_ops.so")

def npu_call(x, **kwargs):
    return torch.ops.npu.<op_name>(x, **kwargs)
```

## 禁止事项

- ❌ **禁止**使用 PyTorch 原生同名接口（如 `torch.gather`、`torch.acosh` 等），**必须**使用 EXEC_NPU_CMD 绑定构建的 `torch.ops.npu.<op_name>(...)`（确保走 950 迁移后路径）
- ❌ **禁止**使用 `import ascend_kernel` 加载算子（仅开发阶段可用，生产环境和精度测试禁止使用）
- ❌ **禁止**使用手动 aclnn C API 调用（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`），**必须**使用 `EXEC_NPU_CMD`
- ❌ **禁止**使用 `torch.ops.aclnn.*` 调用方式（Python 层面不存在此命名空间）
- ❌ **禁止**在精度测试脚本中不设置 `ASCEND_RT_VISIBLE_DEVICES`
