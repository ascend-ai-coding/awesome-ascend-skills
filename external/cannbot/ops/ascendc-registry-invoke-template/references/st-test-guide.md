# ST 测试开发指南

本文档提供 ST（System Test）测试工程开发的完整指南，基于 `add_example` 样例说明开发流程和技术要点。

## 1. 测试方式概述

本项目支持两种 ST 测试方式：

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **C++ 原生测试**（默认） | 直接编译运行 C++ 测试程序 | 常规测试验证，无需 PyTorch 依赖 |
| **PyTorch 接入测试**（可选） | 通过 PyTorch 适配层接入 ACLNN 接口 | 需要与 PyTorch 生态集成的场景 |

## 2. 目录结构

```
${op_name}/tests/st/
├── CMakeLists.txt              # C++ 测试构建配置（默认）
├── run.sh                      # C++ 测试执行脚本
├── test_aclnn_${op_name}.cpp   # C++ 测试主程序
├── torch/                      # PyTorch 接入测试（可选）
│   ├── CMakeLists.txt          # PyTorch 适配层构建配置
│   ├── test.py                 # 测试入口（用例定义 + 调度）
│   ├── golden.py               # CPU golden 计算
│   ├── compare.py              # 精度比对逻辑
│   ├── torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装
└── README.md                   # 说明文档（可选）
```

## 3. C++ 原生测试（默认方式）

### 3.1 架构说明

```
test_aclnn_${op_name}.cpp
├── ComputeGolden()        # CPU golden 计算
├── CompareResults()       # 精度比对
├── TestGoldenCorrectness() # CPU golden 自测
├── RunTest()              # 统一测试执行器
├── GetTestCases()         # 测试用例定义
└── main()                 # 主函数
```

**Mock/Real 模式切换**：

| 模式 | 编译选项 | 适用场景 |
|------|---------|---------|
| Mock | `-DUSE_MOCK=ON` | 算子代码未就绪时，验证测试框架流程 |
| Real | `-DUSE_MOCK=OFF` | 算子代码已就绪，执行真实 NPU 精度验证 |

### 3.2 使用方式

```bash
# Real 模式（默认，需要 NPU）
bash run.sh

# Mock 模式（无需 NPU）
bash run.sh --mock
```

### 3.3 完整样例

参考 `references/add_example/tests/st/` 下的文件：
- `CMakeLists.txt`
- `run.sh`
- `test_aclnn_add_example.cpp`

### 3.4 必须修改的部分

1. **ComputeGolden()** - 根据算子计算逻辑实现 CPU golden
2. **TestGoldenCorrectness()** - 添加算子特定的 golden 自测用例
3. **GetTestCases()** - 根据测试设计文档定义测试用例
4. **RunTest()** - 根据算子 ACLNN 接口调整参数
5. **CMakeLists.txt 中的 find_path/find_library** - 修改算子名称

## 4. PyTorch 接入测试（可选方式）

### 4.1 架构说明

```
PyTorch Tensor (NPU) → torch_adapter.cpp → ACLNN → NPU 结果
       │                          │                        │
       ├── forward_meta()         形状推导（Meta 分发）
       └── forward_npu()          NPU 实现（contiguous + OpCommand 异步入 queue）
```

### 4.2 使用方式

```bash
cd torch/
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
make
python3 ../test.py --lib ./libtorch_adapter.so
```

**注意**: PyTorch 测试仅支持 Real 模式，需要完整的 CANN/NPU 环境。如需无 NPU 验证，请使用 C++ Mock 模式。

### 4.3 完整样例

参考 `references/add_example/tests/st/torch/` 下的文件：
- `CMakeLists.txt`
- `test.py`
- `golden.py`
- `compare.py`
- `torch_adapter.cpp`

### 4.4 必须修改的部分

1. **golden.py** - 根据算子计算逻辑实现 CPU golden
2. **compare.py** - 根据算子精度要求调整比对方法和阈值
3. **test.py** - 根据测试设计文档定义测试用例
4. **torch_adapter.cpp** - 调整算子名称、接口参数和 ACLNN 调用
5. **CMakeLists.txt** - 修改算子名称查找路径

## 5. 核心组件说明

### 5.1 CPU Golden 计算

**C++ 方式**：
```cpp
template<typename T>
void ComputeGolden(const T* x1, const T* x2, T* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = x1[i] + x2[i];  // 示例：加法
    }
}
```

**PyTorch 方式**：
```python
def compute_golden_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """计算逐元素加法的 golden 结果: out = x1 + x2"""
    return x1 + x2
```

### 5.2 精度比对

**C++ 方式**：
```cpp
template<typename T>
bool CompareResults(const T* golden, const T* actual, size_t size, 
                    double rtol = 1e-5, double atol = 1e-8) {
    // 比对逻辑
}
```

**PyTorch 方式**：
```python
def compare_results(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """比对 golden 与实际结果"""
    if golden.dtype.is_floating_point:
        # MERE/MARE Threshold 社区标准
    else:
        # 整数精确匹配
    return passed
```

**精度标准**（CANN 算子精度验收社区标准）：

| 数据类型 | 方法 | Threshold |
|---------|------|-----------|
| FLOAT16 | MERE/MARE | 2^-10 ≈ 0.000977 |
| BFLOAT16 | MERE/MARE | 2^-7 ≈ 0.00781 |
| FLOAT32 | MERE/MARE | 2^-13 ≈ 0.000122 |
| INT32 | 精确匹配 | - |

### 5.3 ACLNN 封装（torch_adapter.cpp 内置）

ACLNN 两段式调用封装在 `torch_adapter.cpp` 中，核心结构：

```cpp
// Workspace 封装：持有 aclTensor 和 executor，析构时自动释放
struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x1_acl = nullptr;
    aclTensor* x2_acl = nullptr;
    aclTensor* output_acl = nullptr;
    ~OpWorkspace();  // 自动销毁 aclTensor
};

// 第一阶段：创建 aclTensor + GetWorkspaceSize
OpWorkspace* OpGetWorkspace(x1_ptr, x2_ptr, output_ptr, shape, dtype);

// 第二阶段：执行算子
aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream);
```

## 6. 开发流程

### 6.1 C++ 测试开发步骤

1. **开发测试代码**
   - 实现 `ComputeGolden()` 函数
   - 实现 `TestGoldenCorrectness()` 自测
   - 实现 `GetTestCases()` 测试用例

2. **编译验证**
   ```bash
   cd ${op_name}/tests/st
   bash run.sh --mock  # Mock 模式验证
   bash run.sh         # Real 模式验证（需要 NPU）
   ```

### 6.2 PyTorch 测试开发步骤

1. **开发 Python 测试代码**
   - 实现 `compute_golden_*()` 函数（golden.py）
   - 实现 `test_golden_correctness()` 自测
   - 实现 `get_test_cases()` 测试用例（test.py）

2. **开发 C++ 适配层**
   - 修改命名空间和算子名称、调整 ACLNN 接口调用（torch_adapter.cpp）
   - 配置 `CMakeLists.txt`（修改算子名称查找路径）

3. **编译验证**
   ```bash
   cd ${op_name}/tests/st/torch
   mkdir -p build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
   make
   python3 ../test.py --lib ./libtorch_adapter.so
   ```

## 7. 常见问题

### Q1: CPU golden 计算如何处理不同数据类型？

**C++ 方式**：使用模板支持不同的数据类型。

**PyTorch 方式**：使用 PyTorch 的类型系统，函数自动适配。

### Q2: 如何处理动态 shape？

在测试用例中定义不同的 shape。

### Q3: 精度阈值如何确定？

默认使用 CANN 算子精度验收社区标准（MERE/MARE Threshold），已内置在比对函数中，按 dtype 自动选取阈值。

### Q4: 如何添加新的测试用例？

**C++ 方式**：在 `GetTestCases()` 函数中添加新的 TestCase。

**PyTorch 方式**：在 `get_test_cases()` 函数中添加新的字典。

### Q5: 两种测试方式如何选择？

- **C++ 原生测试**：轻量、无额外依赖，适合大多数场景
- **PyTorch 接入测试**：适合需要与 PyTorch 生态集成、或使用 PyTorch 工具链的场景

## 8. 依赖项

### C++ 测试
- **CMake**: >= 3.10
- **CANN**: Real 模式需要

### PyTorch 测试
- **Python**: Python 3.8+
- **PyTorch**: torch + torch_npu
- **CMake**: >= 3.10
- **CANN**: 必须

## 9. 参考资源

- **完整示例**：`references/add_example/tests/st/`
- **ACLNN 接口调用文档**：`ops-math/docs/zh/invocation/op_invocation.md`
- **Ascend C 编程指南**：https://www.hiascend.com/document
- **PyTorch 自定义算子扩展**：https://pytorch.org/tutorials/advanced/cpp_extension.html
