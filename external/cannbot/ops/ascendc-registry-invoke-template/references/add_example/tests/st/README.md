# add_example 算子 ST 测试

## 概述

系统测试（ST）验证算子正确性和精度，支持两种测试方式。

| 测试方式 | 说明 | 适用场景 |
|----------|------|---------|
| **C++ 原生测试**（默认） | 直接编译运行 C++ 测试程序 | 常规测试验证，无需 PyTorch 依赖 |
| **PyTorch 接入测试**（可选） | 通过 PyTorch 适配层接入 ACLNN 接口 | 需要与 PyTorch 生态集成的场景 |

## 目录结构

```
tests/st/
├── CMakeLists.txt              # C++ 测试构建配置
├── run.sh                      # C++ 测试执行脚本（支持 --torch 选项）
├── test_aclnn_add_example.cpp  # C++ 测试主程序
├── torch/                      # PyTorch 接入测试（可选）
│   ├── CMakeLists.txt          # PyTorch 适配层构建配置
│   ├── test.py                 # 测试入口（用例定义 + 调度）
│   ├── golden.py               # CPU golden 计算
│   ├── compare.py              # 精度比对逻辑
│   ├── torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装
└── README.md                   # 本文件
```

## 快速开始

### C++ 原生测试

```bash
# Real 模式（默认，需要 NPU）
bash run.sh

# Mock 模式（无需 NPU）
bash run.sh --mock

# 执行指定用例
bash run.sh --case 0
```

### PyTorch 接入测试

```bash
# 通过 run.sh 统一入口（需要 NPU）
bash run.sh --torch

# 或手动执行
cd torch
mkdir -p build && cd build
cmake .. && make
python3 ../test.py --lib ./libtorch_adapter.so
```

**注意**: PyTorch 测试仅支持 Real 模式，需要完整的 CANN/NPU 环境。

## 测试用例

| 测试 | 描述 | 覆盖场景 |
|------|------|---------|
| 1-3  | FP32 基础/混合/大shape | 核心 dtype, 负数, 零值 |
| 4    | FP32 多维 (2x3x4) | 3D shape |
| 5-7  | INT32 基础/混合/大shape | INT32 dtype |
| 8    | 单元素 (1元素) | 边界条件 |
| 9-10 | FP32 极值/零值 | 浮点极值, -0.0 |

## 依赖项

### C++ 测试
- CMake >= 3.10
- g++ (支持 C++17)
- CANN + NPU 设备（Real 模式）

### PyTorch 测试
- Python 3.8+
- PyTorch + torch_npu
- CMake >= 3.10
- CANN + NPU 设备（Real 模式）

## 常见问题

### 1. 编译失败：找不到 AscendCL

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. PyTorch 测试：找不到 torch_npu

确保 torch_npu 已正确安装：
```bash
python3 -c "import torch_npu; print(torch_npu.__version__)"
```

### 3. NPU 设备不可用

PyTorch 测试需要 NPU 环境。如需无 NPU 验证，请使用 C++ Mock 模式：
```bash
bash run.sh --mock
```
