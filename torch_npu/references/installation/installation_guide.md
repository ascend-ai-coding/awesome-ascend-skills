# torch_npu 安装指南

本文档提供 torch_npu 的详细安装步骤，覆盖 x86_64 和 aarch64 两种架构。

---

## 前置条件

### 1. 安装 CANN

在安装 torch_npu 之前，必须先安装 CANN 软件。

- **下载地址**：[CANN 安装指南](https://www.hiascend.com/zh/software/cann/community)
- **版本选择**：参考 [version_compatibility.md](./version_compatibility.md) 选择与 PyTorch 版本配套的 CANN 版本

### 2. 检查 Python 版本

```bash
python3 --version
```

确保 Python 版本与 PyTorch 版本配套（参考 [version_compatibility.md](./version_compatibility.md)）。

### 3. 检查系统架构

```bash
uname -m
# 输出：x86_64 或 aarch64
```

---

## 安装步骤（二进制文件）

### 步骤 1：安装 PyTorch

#### aarch64 架构

```bash
pip3 install torch==2.1.0
```

#### x86_64 架构

```bash
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**版本替换**：将 `2.1.0` 替换为你需要的 PyTorch 版本（如 2.7.1, 2.8.0 等）。

### 步骤 2：安装依赖

```bash
pip3 install pyyaml
pip3 install setuptools
```

### 步骤 3：安装 torch_npu

```bash
pip3 install torch-npu==2.1.0.post17
```

**版本替换**：将 `2.1.0.post17` 替换为你需要的 torch-npu 版本（参考 [version_compatibility.md](./version_compatibility.md)）。

**保存安装日志**：

```bash
pip3 install torch-npu==2.1.0.post17 --log /path/to/install.log
```

---

## 安装步骤（源代码编译）

### 步骤 1：克隆仓库

```bash
git clone https://gitcode.com/ascend/pytorch.git -b v2.1.0-7.2.0 --depth 1
```

**分支选择**：根据 CANN 和 PyTorch 版本选择对应分支（参考 [version_compatibility.md](./version_compatibility.md)）。

### 步骤 2：构建 Docker 镜像（推荐）

```bash
cd pytorch/ci/docker/{arch}  # {arch} 为 X86 或 ARM
docker build -t manylinux-builder:v1 .
```

### 步骤 3：进入 Docker 容器

```bash
docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
```

### 步骤 4：编译 torch_npu

```bash
cd /home/pytorch
bash ci/build.sh --python=3.8
```

**Python 版本**：将 `3.8` 替换为你的 Python 版本（3.9, 3.10, 3.11 等）。

**编译产物**：生成的 `.whl` 文件位于 `./dist/` 目录。

---

## 编译选项

### 使用新的 C++ ABI

如果需要与社区 torch 包 ABI 一致，可以启用新的 C++ ABI：

```bash
export _GLIBCXX_USE_CXX11_ABI=1
bash ci/build.sh --python=3.8
```

**要求**：glibc 2.28, gcc 11.2.1

### 配置 -fabi-version

```bash
export _ABI_VERSION=16
bash ci/build.sh --python=3.8
```

---

## 安装后验证

### 1. 初始化 CANN 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**注意**：如果 CANN 安装路径不同，请修改为实际路径。

### 2. 快速验证脚本

```python
import torch
# torch_npu 2.5.1 及以后版本可以不用手动导包
# import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")
```

**预期输出**：

```
tensor([[...]], device='npu:0')
NPU available: True
NPU count: <NPU 数量>
```

### 3. 详细验证

```python
import torch
import torch_npu

# 检查版本
print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")

# 检查 NPU 可用性
if torch.npu.is_available():
    print(f"NPU is available")
    print(f"Device count: {torch.npu.device_count()}")
    print(f"Current device: {torch.npu.current_device()}")
    print(f"Device name: {torch.npu.get_device_name(0)}")
    print(f"Device capability: {torch.npu.get_device_capability(0)}")
else:
    print("NPU is NOT available")
    print("Please check:")
    print("1. CANN is installed")
    print("2. source /usr/local/Ascend/ascend-toolkit/set_env.sh")
    print("3. NPU device is visible")
```

---

## 常见问题

### 问题 1：pip 安装失败

**原因**：网络问题或 wheel 包不存在。

**解决**：

1. 使用国内镜像源：
   ```bash
   pip3 install torch-npu==2.1.0.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. 手动下载 wheel 包：
   - x86_64：[PyTorch 官方网站](https://pytorch.org/)
   - aarch64：[PyTorch 官方网站](https://pytorch.org/)

### 问题 2：版本不匹配

**错误信息**：
```
ERROR: torch-npu==2.1.0 has requirement torch==2.1.0, but you have torch 2.2.0
```

**解决**：确保 PyTorch、torch_npu、CANN 三者版本配套（参考 [version_compatibility.md](./version_compatibility.md)）。

### 问题 3：CANN 环境变量未设置

**错误信息**：
```
ImportError: libascendcl.so: cannot open shared object file: No such file or directory
```

**解决**：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

或在 `~/.bashrc` 中添加：
```bash
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

### 问题 4：NPU 设备不可见

**检查命令**：
```bash
npu-smi info
```

**解决**：
1. 检查 NPU 驱动是否安装：`cat /usr/local/Ascend/driver/version.info`
2. 检查设备权限：`ls -l /dev/davinci*`
3. 重启 NPU 服务：`systemctl restart ascend-docker-proxy`

### 问题 5：编译时 gcc 版本不匹配

**错误信息**：
```
error: incompatible gcc version
```

**解决**：
- ARM 架构：使用 gcc 10.2
- x86_64 架构：使用 gcc 9.3.1

---

## 卸载

### 卸载 torch_npu

```bash
pip3 uninstall torch_npu
```

**保存卸载日志**：
```bash
pip3 uninstall torch_npu --log /path/to/uninstall.log
```

### 卸载 PyTorch

```bash
pip3 uninstall torch
```

### 完整卸载指南

参考 [昇腾官方文档](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。

---

## 硬件支持

### 训练设备

| 产品系列 | 产品型号 |
|---------|---------|
| Atlas 训练系列产品 | Atlas 800 训练服务器（型号：9000） |
| | Atlas 800 训练服务器（型号：9010） |
| | Atlas 900 PoD（型号：9000） |
| | Atlas 300T 训练卡（型号：9000） |
| | Atlas 300T Pro 训练卡（型号：9000） |
| Atlas A2 训练系列产品 | Atlas 800T A2 训练服务器 |
| | Atlas 900 A2 PoD 集群基础单元 |
| | Atlas 200T A2 Box16 异构子框 |
| Atlas A3 训练系列产品 | Atlas 800T A3 训练服务器 |
| | Atlas 900 A3 SuperPoD 超节点 |

### 推理设备

| 产品系列 | 产品型号 |
|---------|---------|
| Atlas 800I A2 推理产品 | Atlas 800I A2 推理服务器 |

---

## 参考链接

- [torch_npu 官方仓库](https://gitcode.com/Ascend/pytorch)
- [PyTorch 官方网站](https://pytorch.org/)
- [CANN 安装指南](https://www.hiascend.com/zh/software/cann/community)
- [昇腾社区 PyTorch 文档](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)
- [软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)
