# 01 - 环境搭建与验证

## 概述

本文档详细介绍 Triton-Ascend 的环境搭建流程，涵盖硬件选型、软件依赖安装、CANN 版本兼容性矩阵以及环境验证步骤。Triton-Ascend 是基于 OpenAI Triton 的华为昇腾 NPU 适配版本，其编译路径为 `Triton IR → Linalg IR → AscendNPU IR → 设备二进制`，因此环境搭建需要同时满足 Triton 编译框架和昇腾 CANN 软件栈的要求。

**关键词**：Triton-Ascend、环境搭建、CANN、Ascend NPU、torch_npu、pip安装、源码安装、Docker安装

---

## 关键概念

| 概念 | 说明 |
|------|------|
| CANN | Compute Architecture for Neural Networks，昇腾异构计算架构，是 Triton-Ascend 运行的核心软件依赖 |
| torch_npu | PyTorch 的昇腾 NPU 适配插件，提供 `device='npu'` 支持 |
| BiSheng Compiler | 毕昇编译器，负责将 Linalg IR 编译为 Ascend NPU 可执行二进制 |
| ASCEND_HOME_PATH | CANN 安装路径的环境变量，编译和运行时均需正确设置 |
| TRITON_ASCEND_ARCH | 指定目标 NPU 架构的环境变量，影响编译选项和代码生成策略 |

---

## 硬件要求

### 支持的 Ascend NPU 产品

Triton-Ascend 支持华为昇腾 AI 产品，具体型号如下：

| 产品系列 | 产品型号 | 芯片架构标识 | 典型整机 | 别称 |
|----------|----------|-------------|---------|------|
| Atlas A3 训练系列 | Atlas 800T A3 超节点服务器 | Ascend910_95* | Atlas 900 A3 SuperPoD | 910C |
| Atlas A3 推理系列 | Atlas 800I A3 超节点服务器 | Ascend910_95* | - | - |
| Atlas A2 训练系列 | Atlas 800T A2 训练服务器 | Ascend910B* | Atlas800T A2 | A2 |
| Atlas A2 推理系列 | Atlas 800I A2 推理服务器 | Ascend910B* | Atlas 300I A2 推理卡 | - |

### 支持的芯片架构详细列表

通过环境变量 `TRITON_ASCEND_ARCH` 可指定目标架构，合法值包括：

| 架构系列 | 具体型号 |
|----------|---------|
| Ascend910B 系列 | Ascend910B1, Ascend910B2, Ascend910B3, Ascend910B4 |
| Ascend910_93 系列 | Ascend910_9362, Ascend910_9372, Ascend910_9381, Ascend910_9382, Ascend910_9391, Ascend910_9392 |
| Ascend910_95 系列 | Ascend910_9579, Ascend910_9581, Ascend910_9589, Ascend910_9599 |
| Ascend310B 系列 | Ascend310B1, Ascend310B2, Ascend310B3, Ascend310B4 |

> **注意**：Ascend910_95 系列和 Ascend950 系列使用独立的编译路径（`linalg_to_bin_enable_npu_compile_910_95`），UB 大小为 256KB（其他系列为 192KB），且不支持 FFTS 特性。

### 最低硬件配置

- 操作系统：Linux (aarch64 / x86_64)
- 显存：单卡 32GB（推荐）
- AI Core 数量：通过 `npu-smi info` 命令查看

---

## 软件要求

### Python 版本

Triton-Ascend 要求 Python 版本为 **3.9 ~ 3.11**。

### CANN 版本

CANN 是昇腾针对 AI 场景推出的异构计算架构，向上支持 PyTorch、MindSpore 等框架，向下服务 AI 处理器编程。安装指引参见 [昇腾社区 CANN 下载页](https://www.hiascend.com/cann/download)。

### CANN 版本兼容矩阵

**商用版**：

| Triton-Ascend 版本 | CANN 商用版本 | CANN 发布日期 |
|---------------------|--------------|--------------|
| 3.2.0 | CANN 8.5.0 | 2026/01/16 |
| 3.2.0rc4 | CANN 8.3.RC2 / CANN 8.3.RC1 | 2025/11/20 / 2025/10/30 |

**社区版**：

| Triton-Ascend 版本 | CANN 社区版本 | CANN 发布日期 |
|---------------------|--------------|--------------|
| 3.2.0 | CANN 8.5.0 | 2026/01/16 |
| 3.2.0rc4 | CANN 8.3.RC2 / CANN 8.5.0.alpha001 / CANN 8.3.RC1 | 2025/11/20 / 2025/11/12 / 2025/10/30 |

> **建议**：优先安装 CANN 8.5.0 版本。

### PyTorch / torch_npu 版本

当前配套的 torch_npu 版本为 **2.7.1**：

```bash
pip install torch_npu==2.7.1
```

如果出现 `ERROR: No matching distribution found for torch==2.7.1+cpu` 报错，需先手动安装 PyTorch：

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1
```

### 系统推荐配置

| PyTorch 版本 | 推荐 GCC 版本 | 推荐 GLIBC 版本 |
|-------------|-------------|----------------|
| PyTorch 2.6.0 | (aarch64) 11.2.1 / (x86) 9.3.1 | (aarch64) >= 2.28 / (x86) >= 2.17 |
| PyTorch 2.7.1 | 11.2.1 | 2.28 |
| PyTorch 2.8.0 | 13.3.1 | 2.28 |
| PyTorch 2.9.1 | 13.3.1 | 2.28 |
| PyTorch 2.10 | 13.3.1 | 2.28 |

---

## 安装方式

### 方式一：pip 安装（推荐快速上手）

#### 最新稳定版本

```bash
pip install triton-ascend
```

> **重要**：社区 Triton 和 Triton-Ascend 不能同时存在。如果安装其他依赖 Triton 的软件时自动安装了社区版 Triton，会覆盖 Triton-Ascend。此时需先卸载再重装：

```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

#### Nightly Build 版本

```bash
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```

如遇 SSL 报错，可追加信任选项：

```bash
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir \
  --trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org
```

> **注意**：nightly 版本是每日构建包，未经稳定测试，可能存在功能 bug。

历史 nightly 包列表参见 [test.pypi.org](https://test.pypi.org/project/triton-ascend/#history)。选择 nightly 包时请注意匹配服务器的 Python 版本和架构 (aarch64 / x86_64)。

### 方式二：源码安装

#### 安装系统库依赖

```bash
# Ubuntu 系统为例
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache  # 可选，加速构建

# CentOS / RHEL (yum)
sudo yum install -y zlib-devel
```

- 推荐 clang >= 15
- 推荐 lld >= 15

#### 安装 Python 构建依赖

```bash
pip install ninja cmake wheel pybind11
```

#### 快速安装（推荐）

```bash
git clone https://gitcode.com/Ascend/triton-ascend/tree/main
cd triton-ascend
git checkout main

# 可选：若本地有编译好的 LLVM，可直接指定
LLVM_SYSPATH=/path/to/LLVM pip install -e python
```

#### 手动安装（基于 LLVM 构建）

Triton 使用 LLVM 20 生成代码，昇腾毕昇编译器也依赖 LLVM。需先编译 LLVM 源码：

**步骤 1：检出指定版本 LLVM**

```bash
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
```

**步骤 2：构建安装 LLVM**

```bash
export LLVM_INSTALL_PREFIX=/path/to/llvm-install
cd llvm-project
mkdir build && cd build
cmake ../llvm \
  -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
  -DCMAKE_LINKER=/usr/bin/lld-15 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
ninja install
```

**步骤 3：构建 Triton-Ascend**

```bash
git clone https://gitcode.com/Ascend/triton-ascend/tree/main && cd triton-ascend/python

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

> **GCC < 9.4.0 的注意事项**：可能出现 `ld.lld: error: unable to find library -lstdc++fs` 报错。需取消 `triton-ascend/CMakeLists.txt` 中以下代码的注释：
>
> ```cmake
> if (NOT WIN32 AND NOT APPLE)
>   link_libraries(stdc++fs)
> endif()
> ```

### 方式三：Docker 安装

Triton-Ascend 提供了 Dockerfile，自动从 CANN 官网下载安装 Toolkit 和 Kernel 包。

**构建参数**：

| 参数名称 | 默认值 | 可选值 |
|---------|--------|-------|
| CHIP_TYPE | A3 | A3, 910b |
| CANN_VERSION | 8.5.0 | 8.5.0, 8.3.RC1, 8.3.RC2, 8.2.RC1, 8.2.RC2 |

**CHIP_TYPE 对应关系**：

| CHIP_TYPE | 对应产品系列 | 典型整机 | 别称 |
|-----------|------------|---------|------|
| A3 | Atlas A3 训练系列 | Atlas 900 A3 SuperPoD | 910C |
| 910b | Atlas A2 训练系列 | Atlas800T A2 | A2 |

**构建镜像**：

```bash
git clone https://gitcode.com/Ascend/triton-ascend/tree/main && cd triton-ascend
docker build \
  --build-arg CHIP_TYPE=A3 \
  --build-arg CANN_VERSION=8.5.0 \
  -t triton-ascend-image:latest -f ./docker/Dockerfile .
```

**启动容器**：

```bash
docker run -u 0 -dit --shm-size=512g \
  --name=triton-ascend_container --net=host --privileged \
  --security-opt seccomp=unconfined \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home:/home \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  triton-ascend-image:latest \
  /bin/bash
```

**进入容器**：

```bash
docker exec -u root -it triton-ascend_container /bin/bash
```

---

## 环境验证步骤

### 步骤 1：检测 NPU 设备

```bash
npu-smi info
```

预期输出类似：

```
+-----------------------------------------------------------------------------------------+
| NPU-SMI V1.0.0                   Client API Version: 1.0.0                            |
+----------------------+-----------------+------------------------------------------------------+
| NPU   Name           | Health          | Bus-Id        | AICore  | Memory-Usage | ... |
| 0     Ascend910B3    | OK              | 0000:01:00.0  | 30      | 0/65536 MB   | ... |
+----------------------+-----------------+------------------------------------------------------+
```

### 步骤 2：设置 CANN 环境变量

```bash
# root 用户默认安装路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 非 root 用户默认安装路径
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

> **建议**：将 `source` 命令写入 `.bashrc` 文件，避免每次手动执行。

### 步骤 3：验证 Python 环境

```python
import torch
import torch_npu

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")

# 测试 NPU 张量创建
x = torch.randn(3, 3, device='npu')
print(f"NPU tensor: {x}")
```

### 步骤 4：验证 Triton-Ascend 安装

```python
import triton
print(f"Triton version: {triton.__version__}")

# 确认是 Triton-Ascend 而非社区版 Triton
import triton.backends.ascend
print("Triton-Ascend backend loaded successfully")
```

### 步骤 5：运行向量加法 Kernel

```bash
# 拉取示例代码（非源码安装时需要）
git clone https://gitcode.com/Ascend/triton-ascend/tree/main

# 运行示例
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

预期输出：

```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

### 步骤 6：查询 NPU 核心数（可选）

```python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
print(f"AI Core 数量: {properties['num_aicore']}")
print(f"Vector Core 数量: {properties['num_vectorcore']}")
```

---

## NPU 适配要点

1. **CANN 环境变量是必须的**：编译和运行时均依赖 `ASCEND_HOME_PATH` 环境变量。如果未设置，编译器会报错 `ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first`。

2. **Triton-Ascend 与社区 Triton 互斥**：两者不能同时安装。安装其他依赖 Triton 的包时可能自动安装社区版 Triton，覆盖 Triton-Ascend。

3. **架构自动检测**：Triton-Ascend 会通过 PCI 设备信息和 `npu-smi` 命令自动检测是否为 Ascend910_95 系列，并选择对应的编译路径。也可通过 `TRITON_ASCEND_ARCH` 环境变量手动指定。

4. **UB 大小差异**：Ascend910_95 / Ascend950 系列 UB 为 256KB，其他 A2 系列 UB 为 192KB。这直接影响 kernel 中 BLOCK_SIZE 的最大取值。

5. **FFTS 特性**：Ascend910_95 系列不支持 FFTS（Fast Fourier Transform Scheduling），编译器会自动禁用。可通过 `TRITON_DISABLE_FFTS=true` 环境变量在其他架构上手动禁用。

---

## 常见问题

### Q1: 安装 torch_npu 时报错 `No matching distribution found for torch==2.7.1+cpu`

**A**: 先手动安装 PyTorch CPU 版本，再安装 torch_npu：

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1
```

### Q2: 编译时报错 `ASCEND_HOME_PATH is not set`

**A**: 需要先 source CANN 环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q3: 安装其他包后 Triton-Ascend 失效

**A**: 社区版 Triton 会覆盖 Triton-Ascend。解决方法：

```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

### Q4: GCC 版本过低导致链接错误 `unable to find library -lstdc++fs`

**A**: 取消 `CMakeLists.txt` 中 `link_libraries(stdc++fs)` 的注释，然后重新构建。

### Q5: 如何确认当前 NPU 型号？

**A**: 使用 `npu-smi info` 命令查看，或在 Python 中：

```python
import torch_npu
import triton.runtime.driver as driver
device = torch.npu.current_device()
arch = driver.active.utils.get_arch()
print(f"当前架构: {arch}")
```

### Q6: nightly 包安装时 SSL 报错

**A**: 追加信任选项：

```bash
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre \
  --trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org
```

### Q7: 如何选择 CHIP_TYPE 参数（Docker 安装）

**A**: 通过 `npu-smi info` 查看 NPU 型号。Atlas A3 系列选择 `A3`，Atlas A2 系列选择 `910b`。

---

## 相关文档

- [02 - 第一个 Triton-Ascend Kernel](./02-first-kernel.md)：编写并运行第一个向量加法 kernel
- [03 - 编译流程全景](./03-compilation-flow.md)：了解 Triton-Ascend 的完整编译流水线
- [源码参考 - installation_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/installation_guide.md)
- [源码参考 - quick_start.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/quick_start.md)
- [源码参考 - get_ascend_devices.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/tools/get_ascend_devices.py)
- [源码参考 - utils.py (架构检测)](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/utils.py)
