---
name: external-gitcode-ascend-mmcv
description: 在昇腾NPU容器中编译安装mmcv-full，支持NPU算子。适用于需要mmcv作为依赖的其他OpenMMLab库安装前的前置步骤。
type: capability
original-name: ascend-mmcv-install
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ascend-mmcv-install

## 功能

在昇腾NPU容器中编译安装mmcv-full（含NPU算子支持）

## 前置条件

- 容器中已安装 Ascend CANN 和 torch_npu
- 工作目录已挂载（如 /home/zqq）

## 使用

### 步骤0：安装前置三方库依赖

根据目标 PyTorch 版本安装前置依赖：

```bash
source /opt/conda/bin/activate <conda-env-name>
# 根据版本选择对应的 requirements 文件
pip install -r /path/to/DrivingSDK/model_examples/BEVFormer/requirements.txt
# 或 requirements_pytorch2.6.0.txt / requirements_pytorch2.7.1.txt 等
```

### 步骤1：确认conda环境（向用户确认）

询问用户使用的conda环境名称

### 步骤2：克隆mmcv仓库

根据网络情况选择：

**服务器直连正常**：
```bash
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
```

**服务器网络超时**：配置代理或本地克隆后scp传输
```bash
# 方案1：配置代理
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
git clone -b 1.x https://github.com/open-mmlab/mmcv.git

# 方案2：本地克隆后传输（仍使用GitHub链接）
# 本地执行
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
scp -r mmcv user@server:/target/path/
```

**重要**：始终使用GitHub官方链接，不使用镜像站点。

### 步骤3：应用patch

从DrivingSDK复制patch并应用：
```bash
cd mmcv
cp /path/to/DrivingSDK/model_examples/BEVFormer/mmcv_config.patch .
git apply --reject mmcv_config.patch
```

### 步骤4：安装依赖

```bash
source /opt/conda/bin/activate <conda-env-name>
pip install -r requirements/runtime.txt
pip install ninja
```

### 步骤5：编译安装

```bash
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
```

### 步骤6：验证安装

```bash
python -c "import mmcv; print(mmcv.__version__)"
```

**NPU验证**：确认没有CUDA依赖
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
# 应输出空列表
```

## 关键环境变量

| 变量 | 说明 |
|------|------|
| MMCV_WITH_OPS=1 | 编译NPU算子 |
| FORCE_NPU=1 | 强制使用NPU后端 |
| MAX_JOBS=8 | 并行编译任务数 |

## 常见问题

- **numpy版本冲突**：尝试 `pip install numpy==1.23.5`
- **编译超时**：增加 MAX_JOBS
- **patch失败**：手动处理或跳过非关键patch

## 决策指引

**向用户提问**：
1. conda环境名称？
2. 服务器网络能否访问GitHub？
3. patch文件具体路径？
