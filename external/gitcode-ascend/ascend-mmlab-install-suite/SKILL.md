---
name: external-gitcode-ascend-ascend-mmlab-install-suite
description: 昇腾NPU环境安装OpenMMLab系列库套件（mmcv/mmdet/mmdet3d/detectron2），支持本地+远程混合开发模式
original-name: ascend-mmlab-install-suite
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ascend-mmlab-install-suite

## 功能

在昇腾NPU容器环境中安装 OpenMMLab 系列库，支持：
- mmcv（计算机视觉基础库）
- mmdet（目标检测库）
- mmdet3d（3D目标检测库）
- detectron2（Facebook的检测分割库）

## 运行环境

- **目标环境**：昇腾NPU容器（Linux）
- **客户端**：Windows/macOS/Linux
- **网络**：需要访问 GitHub/Gitee

## 模块

- **mmcv**：编译安装 mmcv-full（含NPU算子）→ `mmcv/SKILL.md`
- **mmdet**：安装 mmdetection → `mmdet/SKILL.md`
- **mmdet3d**：安装 mmdetection3d（含mmsegmentation依赖）→ `mmdet3d/SKILL.md`
- **detectron2**：从源码安装 detectron2（NPU兼容）→ `detectron2/SKILL.md`

## 配置

### 必要信息（需向用户确认）

1. **目标服务器信息**：
   - 服务器地址/SSH连接方式
   - 容器名称或ID
   - 工作目录挂载路径（如 /home/zqq）

2. **conda环境**：
   - 环境名称（如 torch2.1.0_py38）
   - Python版本
   - PyTorch版本

3. **网络情况**：
   - 是否有代理
   - GitHub连接是否超时

### 可选信息

4. **patch文件位置**：DrivingSDK中的配置文件路径
5. **安装顺序**：根据用户需求调整

### 前置三方库依赖

在安装 OpenMMLab 库之前，需要安装前置依赖包。根据目标 PyTorch 版本选择对应的 requirements 文件：

| PyTorch版本 | requirements文件 |
|-------------|------------------|
| torch 2.1.x | requirements.txt |
| torch 2.6.0 | requirements_pytorch2.6.0.txt |
| torch 2.7.1 | requirements_pytorch2.7.1.txt |
| torch 2.7.1 (Ascend NPU) | requirements_pytorch2.7.1_a5.txt |

文件位置参考：`DrivingSDK/model_examples/BEVFormer/requirements*.txt`

安装方式：
```bash
pip install -r /path/to/requirements_xxx.txt
```

## 关键原则

### ⚠️ 重要：禁止直接pip安装OpenMMLab库

**严禁使用 `pip install mmcv-full` 等方式直接安装**，因为：
1. DrivingSDK提供的patch文件包含NPU适配修改，直接安装会导致patch无法应用
2. 必须使用源码安装 + 应用patch的方式

**正确流程**：
1. 尝试直接git clone仓库, 如果超时, 则进行下一步
2. 本地git clone仓库（使用镜像或直接clone）
3. scp传输到服务器
4. 应用DrivingSDK中的patch文件
5. 编译安装

### NPU环境检查

安装完成后必须验证：
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
```
如果有 CUDA 相关路径输出，说明安装的是 CUDA 版本，需要重新安装。

### 网络策略

| 情况 | 策略 |
|------|------|
| 服务器直连GitHub正常 | 在服务器上直接 git clone |
| 服务器网络超时 | 本地克隆 → scp传输到服务器 |
| 有代理 | 配置代理后操作 |

### 依赖关系

```
mmcv (基础)
  ├── mmdet
  ├── mmdet3d → 依赖 mmsegmentation
  └── detectron2
```

## 决策指引

**向用户提问确认**：
1. 使用哪个conda环境？
2. 服务器网络情况如何？（能否直接访问GitHub）
3. 是否需要安装所有库？还是特定库？
4. patch文件位置？（如使用DrivingSDK）

**安装前检查**：
1. 确认目标环境是NPU而非CUDA
2. 检查torch_npu是否已安装
3. 确认工作目录可写
