---
name: migration-ascend-torchnpu-skills-torch-npu-reference
description: Provides torch\_npu API compatibility reference and interface mapping knowledge. Invoke when checking NPU interface support, finding equivalent replacements, or resolving torch\_npu compatibility issues.
---

# Skill: torch\_npu接口兼容性参考

你是一位精通torch\_npu接口适配的工程师。本Skill提供torch\_npu的接口兼容性查询和扩展接口参考。

> 官方参考文档（以下链接含版本号路径，MUST实时从昇腾官网查阅最新版本文档，链接仅供参考）：
>
> - 训练迁移指导：昇腾官网 → 文档 → PyTorch框架适配 → 训练迁移指导
> - PyTorch原生接口支持度：昇腾官网 → 文档 → PyTorch框架适配 → 接口参考 → PyTorch原生接口
> - torch\_npu扩展接口：昇腾官网 → 文档 → PyTorch框架适配 → 接口参考 → 扩展接口
> - torch\_npu官方仓库：<https://gitcode.com/Ascend/pytorch>

<constraints>
- MUST 查询最新版本的torch_npu支持列表，NEVER凭记忆判断接口支持状态
- NEVER 在未验证的情况下假设某个接口在NPU上行为与CUDA完全一致
- 查询接口支持度时，MUST根据实际使用的PyTorch版本查阅对应版本的接口文档
- MUST 优先使用镜像站或ModelScope获取模型/数据集资源，NEVER默认使用HuggingFace官网；仅在用户明确确认网络可达HuggingFace时方可使用HuggingFace官网
- MUST 使用pip安装时优先指定第三方镜像源（如阿里源、清华源），仅在镜像源失败时回退至pip默认源
</constraints>

## 一、torch\_npu概述

### 1.1 torch\_npu定位

torch\_npu是PyTorch的昇腾NPU适配扩展，通过PyTorch 2.1版本提供的插件机制，动态添加昇腾后端适配，包含NPU设备、HCCL等一系列能力的支持。

```
PyTorch 原生支持：
  torch.cpu  ← CPU后端
  torch.cuda ← NVIDIA GPU后端

扩展支持：
  torch.npu  ← 昇腾NPU后端（通过torch_npu包提供）
```

### 1.2 torch\_npu安装与导入

> MUST实时查阅昇腾官网最新版本文档，以下链接仅供参考：
> 昇腾官网 → 文档 → PyTorch框架适配 → 软件安装指南 → 安装 → 使用二进制包安装
> 参考链接：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md>
>
> 根据目标PyTorch版本、架构(aarch64/x86\_64)、Python版本，从上述文档查找对应的下载链接。

```bash
# 方式1：wget下载whl包安装（推荐，优先使用）
# 示例：PyTorch 2.9.0 + aarch64 + Python 3.11 + CANN 9.0.0
pip install pyyaml setuptools -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl
pip install torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
wget https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.9.0/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl
pip install torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl

# 方式2：pip直接安装（备选，网络通畅时使用）
# pip install torch-npu=={version} -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  # version必须与PyTorch版本对应
# 如阿里源失败，尝试清华源：pip install torch-npu=={version} -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如镜像源均失败，回退至默认源：pip install torch-npu=={version}
```

```python
# 导入（必须在import torch之后）
import torch
import torch_npu  # 注册NPU后端

# 验证
print(torch.npu.is_available())  # True
```

### 1.3 torch\_npu源码仓库

- 仓库地址：<https://gitcode.com/Ascend/pytorch>
- Release下载：<https://gitcode.com/Ascend/pytorch/releases>

### 1.4 从源码获取解决方案

当无法从本Skill或官方文档解决接口适配问题时，可拉取torch\_npu代码仓，根据实际代码分析接口实现和适配方案：

```bash
# 拉取代码仓（--recursive确保第三方依赖也被拉取）
git clone https://gitcode.com/Ascend/pytorch.git --recursive

# 切换到与当前PyTorch版本对应的分支
cd pytorch
git checkout v{PyTorch版本号}

# 如果克隆时未加--recursive，可后续补充拉取子模块
git submodule update --init --recursive
```

## 二、PyTorch接口NPU支持状态

### 2.1 接口支持度查询

昇腾官方提供了按PyTorch版本区分的接口支持度清单，MUST根据实际使用的PyTorch版本查阅对应文档：

| PyTorch版本 | 接口文档路径                                                                   |
| --------- | ------------------------------------------------------------------------ |
| 2.10.0    | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-10-0/overview.md` |
| 2.9.0     | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/overview.md`  |
| 2.8.0     | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-8-0/overview.md`  |
| 2.7.1     | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/overview.md`  |
| 2.6.0     | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-6-0/overview.md`  |
| 2.1.0     | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-1-0/overview.md`  |

> 以上路径均基于文档根目录 `https://www.hiascend.com/document/detail/zh/Pytorch/2600/` 下。版本迭代较快，MUST以官网最新版本文档为准。

**查询方法**：

1. 确定当前项目使用的PyTorch版本
2. 打开对应版本的接口支持度文档
3. 搜索目标接口名称，查看是否在支持列表中
4. 如在支持列表中，确认接口的行为差异说明
5. 如不在支持列表中，查阅torch\_npu扩展接口文档是否有替代

### 2.2 torch\_npu扩展接口

torch\_npu提供了昇腾NPU专用的扩展接口，MUST查阅官方扩展接口文档确认接口名称、参数和使用方式：

- 扩展接口文档：昇腾官网 → 文档 → PyTorch框架适配 → 接口参考 → 扩展接口

### 2.3 常见不支持或行为差异的接口

| 接口                       | 状态                 | 替代方案                                            |
| ------------------------ | ------------------ | ----------------------------------------------- |
| `torch.cuda.*`           | 需替换为 `torch.npu.*` | 参见 [迁移Skill](../migration-ascend-torchnpu-skills-migration-execution/SKILL.md) 第二章设备映射替换规则 |
| `torch.backends.cudnn.*` | 不支持                | 删除或条件跳过                                         |
| 自定义CUDA扩展(.cu)           | 不支持                | 使用torch\_npu扩展接口或PyTorch原生实现                    |

<verification>
完成接口查询后逐项检查：
- [ ] 所有torch接口已查询NPU支持状态
- [ ] 不支持的接口已有替代方案
- [ ] 替代方案已验证数学一致性
- [ ] 接口兼容性查询结果已纳入迁移报告
</verification>
