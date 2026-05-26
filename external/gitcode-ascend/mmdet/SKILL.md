---
name: external-gitcode-ascend-mmdet
description: 在昇腾NPU容器中安装mmdetection。适用于目标检测模型的开发。
type: capability
original-name: ascend-mmdet-install
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ascend-mmdet-install

## 功能

在昇腾NPU容器中安装 mmdetection（2D目标检测库）

## 前置条件

- mmcv 已安装
- 容器中已安装 Ascend CANN 和 torch_npu
- 工作目录已挂载

## 使用

### 步骤1：确认前置条件

确保 mmcv 已正确安装：
```bash
python -c "import mmcv; print(mmcv.__version__)"
```

### 步骤2：克隆mmdetection仓库

根据网络情况选择：

**服务器直连**：
```bash
git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
```

**服务器网络超时**：配置代理或本地克隆后scp传输
```bash
# 方案1：配置代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git

# 方案2：本地克隆后传输（仍使用GitHub链接）
# 本地执行
git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
scp -r mmdetection user@server:/target/path/
```

**重要**：始终使用GitHub官方链接，不使用镜像站点。

### 步骤3：应用patch（如有）

```bash
cd mmdetection
cp /path/to/DrivingSDK/model_examples/BEVFormer/mmdet_config.patch .
git apply --reject mmdet_config.patch
```

### 步骤4：安装

```bash
source /opt/conda/bin/activate <conda-env-name>
pip install -e .
```

### 步骤5：验证安装

```bash
python -c "import mmdet; print(mmdet.__version__)"
```

**NPU验证**：
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
# 应输出空列表
```

## 常见问题

- **依赖缺失**：确保 mmcv 和 mmengine 已安装
- **patch失败**：跳过非关键patch或手动处理

## 决策指引

**向用户提问**：
1. mmcv 是否已安装？
2. 服务器网络能否访问GitHub？
3. 是否需要应用特定patch？
