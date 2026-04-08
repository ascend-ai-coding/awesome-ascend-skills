# PyTorch 版本兼容性指南

本文档提供了 torch_npu 支持的 PyTorch 版本与 CANN、Python 的配套关系，以及不同架构的安装命令。

---

## 版本配套表

| PyTorch 版本 | CANN 版本 | Python 版本 | GitCode 分支 | 架构支持 | 文档链接 |
|-------------|-----------|-------------|--------------|----------|----------|
| **2.9.0** | 7.3.0 | 3.8-3.11 | v7.3.0-pytorch2.9.0 | x86_64, aarch64 | [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html) |
| **2.8.0** | 7.3.0 | 3.8-3.11 | v7.3.0-pytorch2.8.0 | x86_64, aarch64 | [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html) |
| **2.7.1** | 7.3.0 | 3.8-3.11 | v7.3.0-pytorch2.7.1 | x86_64, aarch64 | [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html) |
| **2.6.0** | 7.3.0 | 3.8-3.11 | v7.3.0-pytorch2.6.0 | x86_64, aarch64 | [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html) |
| **2.1.0** | 7.2.0 | 3.7-3.11 | v2.1.0-7.2.0 | x86_64, aarch64 | [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html) |

---

## 最低要求

- **CANN 版本**：>= 7.2.0
- **Python 版本**：3.7 - 3.11
- **系统架构**：x86_64 或 aarch64
- **PyTorch 版本**：必须与 torch_npu 版本配套（不能混用）

---

## 架构差异说明

### x86_64（Intel/AMD）
- **适用平台**：Intel、AMD 处理器
- **编译要求**：gcc 9.3.1
- **glibc 版本**：>= 2.17
- **安装命令**：使用 `torch==2.1.0+cpu` 包

### aarch64（ARM）
- **适用平台**：华为昇腾 Atlas 系列设备
- **编译要求**：gcc 10.2
- **glibc 版本**：>= 2.28
- **安装命令**：使用 `torch==2.1.0` 包

---

## 版本详细说明

### PyTorch 2.9.0（最新）

**配套信息**：
- **CANN 版本**：7.3.0
- **Python 版本**：3.8 - 3.11
- **GitCode 分支**：`v7.3.0-pytorch2.9.0`
- **架构**：x86_64, aarch64

**安装命令**：

```bash
# x86_64 架构
pip3 install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip3 install torch==2.9.0

# 安装 torch_npu
pip3 install torch-npu==2.9.0.post17

# 使用国内镜像
pip3 install torch-npu==2.9.0.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**参考文档**：
- [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html)
- [安装指南](./installation_guide.md)

---

### PyTorch 2.8.0

**配套信息**：
- **CANN 版本**：7.3.0
- **Python 版本**：3.8 - 3.11
- **GitCode 分支**：`v7.3.0-pytorch2.8.0`
- **架构**：x86_64, aarch64

**安装命令**：

```bash
# x86_64 架构
pip3 install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip3 install torch==2.8.0

# 安装 torch_npu
pip3 install torch-npu==2.8.0.post17

# 使用国内镜像
pip3 install torch-npu==2.8.0.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**参考文档**：
- [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html)
- [安装指南](./installation_guide.md)

---

### PyTorch 2.7.1

**配套信息**：
- **CANN 版本**：7.3.0
- **Python 版本**：3.8 - 3.11
- **GitCode 分支**：`v7.3.0-pytorch2.7.1`
- **架构**：x86_64, aarch64

**安装命令**：

```bash
# x86_64 架构
pip3 install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip3 install torch==2.7.1

# 安装 torch_npu
pip3 install torch-npu==2.7.1.post17

# 使用国内镜像
pip3 install torch-npu==2.7.1.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**参考文档**：
- [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html)
- [安装指南](./installation_guide.md)

---

### PyTorch 2.6.0

**配套信息**：
- **CANN 版本**：7.3.0
- **Python 版本**：3.8 - 3.11
- **GitCode 分支**：`v7.3.0-pytorch2.6.0`
- **架构**：x86_64, aarch64

**安装命令**：

```bash
# x86_64 架构
pip3 install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip3 install torch==2.6.0

# 安装 torch_npu
pip3 install torch-npu==2.6.0.post17

# 使用国内镜像
pip3 install torch-npu==2.6.0.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**参考文档**：
- [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html)
- [安装指南](./installation_guide.md)

---

### PyTorch 2.1.0

**配套信息**：
- **CANN 版本**：7.2.0
- **Python 版本**：3.7 - 3.11
- **GitCode 分支**：`v2.1.0-7.2.0`
- **架构**：x86_64, aarch64

**安装命令**：

```bash
# x86_64 架构
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# aarch64 架构
pip3 install torch==2.1.0

# 安装 torch_npu
pip3 install torch-npu==2.1.0.post17

# 使用国内镜像
pip3 install torch-npu==2.1.0.post17 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**参考文档**：
- [API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/api_support/api_support_0001.html)
- [安装指南](./installation_guide.md)

---

## 版本选择建议

### 开发环境
- **推荐**：PyTorch 2.9.0 + CANN 7.3.0（最新版本，支持最新特性）
- **稳定**：PyTorch 2.7.1 + CANN 7.3.0（测试充分，兼容性好）

### 生产环境
- **保守**：PyTorch 2.1.0 + CANN 7.2.0（长期支持，稳定可靠）

### 特殊场景
- **ARM 架构**：必须使用 aarch64 版本的 PyTorch 和 torch_npu
- **CPU 测试**：使用 x86_64 + cpu 版本 PyTorch，避免 NPU 依赖

---

## 注意事项

1. **版本一致性**：PyTorch、torch_npu、CANN 三者版本必须严格配套，不能混用
2. **Python 版本**：选择与你系统 Python 版本一致的 PyTorch 版本
3. **架构匹配**：确保安装的 PyTorch 架构与系统架构匹配
4. **国内镜像**：建议使用国内镜像源加速安装

---

## 故障排查

### 版本不匹配错误
**错误信息**：
```
ERROR: torch-npu==2.1.0 has requirement torch==2.1.0, but you have torch 2.2.0
```

**解决方法**：确保 PyTorch 版本与 torch_npu 版本配套（参考本文档的版本配套表）

### NPU 不可用
**检查步骤**：
1. 确认 CANN 版本与 PyTorch 版本配套
2. 检查 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 是否执行
3. 运行 `npu-smi info` 检查 NPU 状态

---

## 相关链接

- [torch_npu 官方仓库](https://gitcode.com/Ascend/pytorch)
- [PyTorch 官方网站](https://pytorch.org/)
- [CANN 安装指南](https://www.hiascend.com/zh/software/cann/community)
- [昇腾社区 PyTorch 文档](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)
- [安装指南](./installation_guide.md)
