---
name: external-gitcode-ascend-mmdet3d
description: 在昇腾NPU容器中安装mmdetection3d（含mmsegmentation依赖）。适用于3D目标检测模型的开发。
type: capability
original-name: ascend-mmdet3d-install
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ascend-mmdet3d-install

## 功能

在昇腾NPU容器中安装 mmdetection3d（3D目标检测库）

## 前置条件

- mmcv 已安装
- mmsegmentation 已安装（mmdet3d 依赖）
- 容器中已安装 Ascend CANN 和 torch_npu
- 工作目录已挂载

## 使用

### 步骤1：确认前置条件

确保 mmcv 已安装：
```bash
python -c "import mmcv; print(mmcv.__version__)"
```

### 步骤2：安装 mmsegmentation

mmdet3d 依赖 mmsegmentation，根据版本选择：

**推荐版本**（适配 mmcv 1.7.x）：
```bash
pip install mmsegmentation==0.30.0
```

### 步骤3：克隆 mmdetection3d 仓库

根据网络情况选择：

**服务器直连**：
```bash
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
```

**服务器网络超时**：配置代理或本地克隆后scp传输
```bash
# 方案1：配置代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git

# 方案2：本地克隆后传输（仍使用GitHub链接）
# 本地执行
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
scp -r mmdetection3d user@server:/target/path/
```

**重要**：始终使用GitHub官方链接，不使用镜像站点。

### 步骤4：应用patch（如有）

```bash
cd mmdetection3d
cp /path/to/DrivingSDK/model_examples/BEVFormer/mmdet3d_config.patch .
git apply --reject mmdet3d_config.patch
```

### 步骤5：安装

```bash
source /opt/conda/bin/activate <conda-env-name>
pip install -e .
```

### 步骤6：验证安装

```bash
python -c "import mmdet3d; print(mmdet3d.__version__)"
```

**NPU验证**：
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
# 应输出空列表
```

## 依赖关系

```
mmcv
  └── mmsegmentation == 0.30.0
        └── mmdet3d
```

## 常见问题

- **mmsegmentation 版本**：必须使用 0.30.0 版本，与 mmcv 1.7.x 兼容
- **import mmseg 失败**：确保 mmsegmentation 正确安装
- **patch失败**：跳过非关键patch或手动处理

## 决策指引

**向用户提问**：
1. mmcv 是否已安装？
2. 使用哪个版本的 mmsegmentation？
3. 服务器网络能否访问GitHub？
4. 是否需要应用特定patch？
