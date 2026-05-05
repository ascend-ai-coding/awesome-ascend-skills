---
name: external-gitcode-ascend-ascendc-operator-code-gen
description: '根据设计文档生成 AscendC 算子完整代码实现并完成框架适配。TRIGGER when: 设计文档已完成，需要生成 op_host/op_kernel
  代码、注册到 PyTorch 框架、编译测试。关键词：代码生成、op_host、op_kernel、tiling、kernel、框架适配、算子注册。'
original-name: ascendc-operator-code-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# AscendC 算子代码生成与框架适配

根据设计文档生成 op_host + op_kernel 代码，注册到 PyTorch 框架，调用 `ascendc-operator-compile-debug` skill 完成编译安装和精度测试。

**前置条件**: 设计文档 `ascend-kernel/csrc/ops/<op_name>/design.md` 已就绪，目录结构已创建。

## 工作流程总览

```
读取设计文档 → 加载 reference → 选择模板 → 生成 op_host + op_kernel
    → 框架适配 (ops.h + register.cpp + csrc/CMakeLists.txt)
    → 调用 ascendc-operator-compile-debug skill (编译 + 安装 + 测试)
```

---

## 阶段 1: 加载参考文档

**MANDATORY — READ BEFORE CODING**: 读取 [`references/GUIDE.md`](references/GUIDE.md)，根据设计文档中的算子类型，加载对应的 reference 文件。**绝对不要跳过此步骤。**

## 阶段 2: 读取设计文档

从 `ascend-kernel/csrc/ops/<op_name>/design.md` 提取：

| 提取项 | 设计文档章节 | 用途 |
|--------|------------|------|
| 函数签名 + 支持的数据类型 | 原型设计 | op_host 函数原型、kernel 模板参数 |
| 算子类型 | Tiling 切分 → 步骤1 | 选择模板 (elementwise / row) |
| UB 分配表 | UB 空间分配 | 推导 bufferCoefficient、InitBuffer 大小 |
| 计算伪代码 | Kernel 实现 | Compute 函数逻辑 |

---

## 阶段 3: 选择模板并生成代码

根据算子类型选择对应模板，复制到工程目录后修改。

### 模板选择

| 算子类型 | op_host 模板 | op_kernel 模板 |
|---------|-------------|---------------|
| Elementwise (ReLU, GELU, Add...) | [`templates/elementwise_op_host.cpp`](templates/elementwise_op_host.cpp) | [`templates/elementwise_op_kernel.cpp`](templates/elementwise_op_kernel.cpp) |
| 行处理 (LayerNorm, Softmax...) | [`templates/row_op_host.cpp`](templates/row_op_host.cpp) | [`templates/row_op_kernel.cpp`](templates/row_op_kernel.cpp) |
| Index-PerRow (IndexSelect, 共享1D索引) | [`templates/index_op_host.cpp`](templates/index_op_host.cpp) | [`templates/index_op_kernel.cpp`](templates/index_op_kernel.cpp) |
| Index-PerElem (Gather, Scatter, 逐行索引) | [`templates/index_op_per_elem_host.cpp`](templates/index_op_per_elem_host.cpp) | [`templates/index_op_per_elem_kernel.cpp`](templates/index_op_per_elem_kernel.cpp) |
| Sort (Sort, TopK...) | [`templates/sort_op_host.cpp`](templates/sort_op_host.cpp) | [`templates/sort_op_kernel.cpp`](templates/sort_op_kernel.cpp) |

**MANDATORY**: 读取对应模板文件的完整内容，复制到目标路径后修改。

### 生成步骤

1. 读取选中的 op_host 和 op_kernel 模板的完整内容
2. 将内容写入 `ascend-kernel/csrc/ops/<op_name>/op_host/<op_name>.cpp` 和 `op_kernel/<op_name>.cpp`（覆盖骨架占位文件）
3. 替换所有占位符（`<op_name>`, `<OpName>` 等）
4. 根据设计文档修改：
   - **op_host**: 函数签名、输入校验、`bufferCoefficient`、EXEC_KERNEL_CMD 参数
   - **op_kernel**: Init 的 GM 参数、InitBuffer 大小、Compute 计算逻辑

### FP16/FP32 模板分支

Kernel 模板使用 `template <typename T>` 泛型 + `if constexpr` 分支处理不同数据类型：

```cpp
if constexpr (sizeof(T) == sizeof(float)) {
    // float32 直接计算
} else {
    // fp16/bf16: Cast 升精度 → fp32 计算 → Cast 降精度
}
```

AscendC 编译器支持 C++17 `if constexpr`，可安全用于模板分支。如需兼容旧编译器，也可用 `sizeof(T) == 4` 的普通 if（编译器会优化掉死分支）。

### 模板修改要点

#### 硬件参数获取（模板已内置）

模板使用平台 API 动态获取硬件参数，不再硬编码：

```cpp
#include "tiling/platform/platform_ascendc.h"

auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
uint64_t ubSize;
ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
// 需要 workspace 时:
// uint64_t sysWorkspaceSize = ascendc_platform->GetLibApiWorkSpaceSize();
```

#### bufferCoefficient 推导

从设计文档 UB 分配表推导——所有 Buffer 总大小 = `bufferCoefficient * tileLength`：
- 将 UB 分配表中所有 Buffer 的"总大小"列加和
- 结果形如 `tileLength * N`，则 `bufferCoefficient = N`
- 不同 dtype 系数不同时用 `if (dtypeSize == 2) {...} else {...}` 分支

#### FP16/BF16 升精度

**FP16 和 BF16 都必须升精度到 FP32 计算再转回**。模板中已标注升精度代码的插入位置，取消注释并填充计算逻辑即可。

#### GM ↔ UB 搬运（DataCopyPad）

**生产代码必须使用 DataCopyPad**（不要使用 DataCopy 进行 GM↔UB 搬运）：

```cpp
// CopyIn: GM → UB
AscendC::DataCopyExtParams copyInParams{1,
    static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
AscendC::DataCopyPad(xLocal, xGm[progress * tileLength], copyInParams, padParams);

// CopyOut: UB → GM
AscendC::DataCopyExtParams copyOutParams{1,
    static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(yGm[progress * tileLength], yLocal, copyOutParams);
```

> **注意**: 模板文件中可能使用 DataCopy 以简化示例，生产代码应替换为 DataCopyPad。

#### ReduceSum/ReduceMax 注意

归约操作可能修改源 tensor，必须先备份：

```cpp
AscendC::Adds(backup, src, 0.0f, len);  // 备份
AscendC::ReduceSum<float, true>(result, backup, sharedTmp, dimLen);
```

---

## 阶段 4: 框架适配

### 4.1 更新 `csrc/ops.h`

在 `namespace ascend_kernel` 中添加函数声明：

```cpp
<返回类型> <op_name>(<参数列表>);
```

### 4.2 更新 `csrc/register.cpp`

在 `TORCH_LIBRARY_FRAGMENT(npu, m)` 中添加 schema：

```cpp
m.def("<op_name>(<schema_参数>) -> <返回类型>");
```

在 `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 中添加实现：

```cpp
m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
```

**Schema 语法速查**:

| C++ 类型 | Schema 类型 | 示例 |
|---------|------------|------|
| `const at::Tensor &` | `Tensor` | `Tensor self` |
| `at::IntArrayRef` | `int[]` | `int[] kernel_size` |
| `int64_t` | `int` | `int dim=-1` |
| `double` | `float` | `float eps=1e-5` |
| `bool` | `bool` | `bool flag=False` |
| `c10::optional<at::Tensor>` | `Tensor?` | `Tensor? weight=None` |
| `c10::optional<int64_t>` | `int?` | `int? divisor=None` |

### 4.3 更新 `csrc/CMakeLists.txt`

在 `FILE(GLOB OP_SRCS ...)` 中添加 host 源文件：

```cmake
${PROJECT_OP_SRC_BASE}/ops/<op_name>/op_host/<op_name>.cpp
```

在 `ascendc_library(no_workspace_kernel STATIC ...)` 中添加 kernel 源文件：

```cmake
${PROJECT_OP_SRC_BASE}/ops/<op_name>/op_kernel/<op_name>.cpp
```

---

## 阶段 5: 编译安装与测试

**MANDATORY — 按 `ascendc-operator-compile-debug` skill 流程执行**：

1. 加载环境：source CANN 环境变量，激活conda环境
2. 确保 `build.sh` 有执行权限：`chmod +x build.sh`
3. 编译工程：`bash build.sh`
4. 验证编译成功：`ls output/ascend_kernel*.whl`
5. 安装 whl 包：`pip install output/ascend_kernel*.whl --force-reinstall --no-deps`
6. 生成测试文件 `tests/test_<op_name>.py`
7. 运行功能测试：`python tests/test_<op_name>.py`
8. 运行精度测试：`pytest tests/test_<op_name>.py -v`
9. 编译/测试失败进入排错循环（最多 3 次）
10. **精度测试失败时**：若 allclose 不通过、输出偏差、全零或 NaN，**MUST** 读取并按 `ascendc-operator-precision-debug` skill 流程排查根因（误差分析 → 代码审查 → 实验隔离 → 插桩定位 → 修复验证）

> **实战经验**：每次 shell 命令前都要 source 环境变量，否则会找不到编译工具或 Python 包。

---

## 生成后检查清单

### op_host
- [ ] namespace `ascend_kernel`，include `torch_kernel_helper.h` + `tiling/platform/platform_ascendc.h` + `aclrtlaunch_<op_name>.h`
- [ ] 使用平台 API 获取 coreNum 和 ubSize（不硬编码）
- [ ] bufferCoefficient 与设计文档 UB 分配表一致
- [ ] EXEC_KERNEL_CMD 所有参数均为左值
- [ ] [行处理] padding/去 padding 正确

### op_kernel
- [ ] include `kernel_operator.h`，BUFFER_NUM = 2
- [ ] Init 整核/尾核偏移正确，InitBuffer 大小与 UB 分配表一致
- [ ] Process 尾 tile 对齐（alignedTailLen）
- [ ] AllocTensor/FreeTensor 配对，EnQue/DeQue 配对
- [ ] **FP16/BF16 必须升精度到 FP32 计算**
- [ ] ReduceSum 前备份源数据

### 框架适配
- [ ] ops.h 声明与 op_host 函数签名一致
- [ ] register.cpp schema 参数类型/默认值正确
- [ ] csrc/CMakeLists.txt 添加了 host 和 kernel 源文件

## 反模式清单

- **NEVER** 让 FP16/BF16 直接参与复杂数学计算（Mul/Div/Exp/Tanh 等），必须先 Cast 到 FP32
- **NEVER** 在 EXEC_KERNEL_CMD 中传右值（临时对象、字面量、表达式结果）
- **NEVER** 在 kernel 中使用 bool 参数类型，用 int64_t 替代
- **NEVER** 对 GM↔UB 搬运使用 DataCopy，必须用 DataCopyPad
- **NEVER** 在 ReduceSum/ReduceMax 后直接复用源 tensor（归约可能修改源数据）
- **NEVER** 在 kernel 中使用 `std::min/max/abs/sqrt/exp` 等标准库函数
- **NEVER** 向高维切分 API 传入 repeatTime > 255（uint8_t 会静默截断为 0）
- **NEVER** 修改 `cmake/` 或 `csrc/utils/` 下的文件
- **NEVER** 在 register.cpp 的 schema 中遗漏默认值（如 `int dim` 应写为 `int dim=-1`）
- **NEVER** 硬编码核数或 UB 大小，必须通过平台 API 获取
- **NEVER** 让 ReduceMax/Sum 的 dst 与 tmpBuffer 指向同一块内存
