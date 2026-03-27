---
name: ai-for-science-ascend-tf-community
description: 昇腾 TensorFlow Community 迁移适配 Skill，适用于将基于 TensorFlow 2.x 的模型原生部署到华为 Ascend NPU，而不经过 TF 到 PyTorch 转换，覆盖 aarch64 源码编译 TF 2.6.5、tfplugin 安装、自动迁移工具使用、手动适配与精度验证。
keywords:
  - ai-for-science
  - tensorflow
  - tf-community
  - npu_device
  - tfplugin
  - ascend
---

# 昇腾 TensorFlow Community 迁移适配 Skill

## 项目概述

本 Skill 记录在昇腾 NPU 上使用 TensorFlow Community 方式（非 TF→PyTorch 转换）
部署 TF 2.x 模型的完整流程。核心思路是：保持原始 TF 代码基本不变，通过
`npu_device` 插件将算子调度到 Ascend NPU 上执行。

**适用场景：**
- 已有 TF 2.x 训练/推理代码，希望直接运行在昇腾 NPU 上
- 不希望或不适合做 TF→PyTorch 框架转换
- 参考文档：https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/migration/tfmigr2/

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| CANN | ≥ 8.0（推荐 8.5） |
| Python | 3.7.x / 3.8.x / 3.9.x |
| TensorFlow | 2.6.5（需源码编译，ABI=0） |
| npu_device | 0.1（来自 Ascend-cann-tfplugin） |
| Bazel | 3.7.2（编译 TF 所需） |

---

## 迁移流程总览

```
1. 环境检查（NPU、CANN、GCC）
→ 2. 源码编译 TF 2.6.5（aarch64 专用流程）
→ 3. 安装 tfplugin（npu_device）
→ 4. 代码适配（自动迁移工具 or 手动）
→ 5. 推理/训练验证
→ 6. 精度对比
```

---

## Step 1：环境检查

```bash
# 检查 NPU 设备
npu-smi info

# 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || \
cat /home/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null

# CANN 环境变量（根据实际安装路径选择）
source /home/Ascend/8.5/cann-8.5.0/set_env.sh
# 或
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查编译器
gcc --version   # 需要 GCC 7.3+
uname -m        # aarch64 需要源码编译 TF
```

---

## Step 2：源码编译 TF 2.6.5（aarch64）

> **关键**：aarch64 平台 pip 不提供 TF 2.6.x wheel，必须从源码编译。
> x86_64 平台可直接 `pip install tensorflow-cpu==2.6.5`。

### 2.1 创建 Conda 环境

```bash
conda create -n tf_npu python=3.8 -y
conda activate tf_npu
conda install -y hdf5
pip install numpy==1.21.6 keras-preprocessing packaging h5py==3.7.0
```

### 2.2 安装 Bazel 3.7.2

```bash
# 从华为镜像下载（GitHub 可能超时）
wget 'https://mirrors.huaweicloud.com/bazel/3.7.2/bazel-3.7.2-linux-arm64' \
     -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel
bazel version  # 确认 Build label: 3.7.2
```

### 2.3 克隆 TF 源码 + nsync 补丁

```bash
git clone --depth 1 --branch v2.6.5 \
    https://github.com/tensorflow/tensorflow.git tf265_src
cd tf265_src
```

**nsync aarch64 补丁**（昇腾文档要求）：

```bash
# 下载 nsync 源码
wget 'https://github.com/google/nsync/archive/1.22.0.tar.gz' \
     -O /tmp/nsync-1.22.0.tar.gz
cd /tmp && tar -zxf nsync-1.22.0.tar.gz
```

编辑 `/tmp/nsync-1.22.0/platform/c++11/atomic.h`，在 `NSYNC_CPP_START_` 后添加：

```c
#define ATM_CB_() __sync_synchronize()
```

并在每个 `atm_cas_*_u32_` 函数中，将 `return (std::atomic_compare_exchange_strong_explicit(...));`
改为：

```c
int result = (std::atomic_compare_exchange_strong_explicit(...));
ATM_CB_();
return result;
```

重新打包并更新 sha256：

```bash
cd /tmp && rm -f nsync-1.22.0.tar.gz
tar -czf nsync-1.22.0.tar.gz nsync-1.22.0/
sha256sum nsync-1.22.0.tar.gz   # 记录新的 hash
```

修改 `tensorflow/workspace2.bzl` 中 nsync 的 `sha256` 和 `urls`（添加 `file:///tmp/nsync-1.22.0.tar.gz`）。

### 2.4 配置编译

```bash
# 设置环境变量
export PYTHON_BIN_PATH=$(which python)
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_ENABLE_XLA=0
export CC_OPT_FLAGS='-march=armv8-a'

yes '' | ./configure
```

**关键：设置 ABI=0**（昇腾文档要求 TF 和 FwkPlugin ABI 一致）：

在 `.tf_configure.bazelrc` 顶部添加：

```
build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
```

### 2.5 编译

```bash
# 如果 GitHub 不通，替换 workspace 文件中的 github.com URL：
sed -i 's|"https://github.com/|"https://ghfast.top/https://github.com/|g' \
    tensorflow/workspace2.bzl

bazel build --config=opt \
    //tensorflow/tools/pip_package:build_pip_package \
    --jobs=$(nproc)

# 打包 wheel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_whl

# 安装
pip install /tmp/tf_whl/tensorflow-2.6.5-*.whl --no-deps
```

### 2.6 安装配套依赖

```bash
pip install 'absl-py~=0.10' 'clang~=5.0' 'flatbuffers~=1.12.0' \
    'keras~=2.6' 'keras-preprocessing~=1.1.1' 'opt-einsum~=3.3.0' \
    'six~=1.15.0' 'termcolor~=1.1.0' 'wrapt~=1.12.1' \
    'grpcio' 'protobuf' 'tensorboard~=2.6' 'tensorflow-estimator~=2.6' \
    'typing-extensions>=3.7,<3.11' 'google-pasta' 'astunparse'

# 验证
python -c "import tensorflow as tf; print(tf.__version__); \
    print(tf.reduce_sum(tf.random.normal([100,100])))"
```

---

## Step 3：安装 tfplugin（npu_device）

### 3.1 下载 Ascend-cann-tfplugin

```bash
# CANN 8.0.RC3 版本（与 TF 2.6 ABI 匹配）
wget 'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-tfplugin_8.0.RC3_linux-aarch64.run' \
     -O /tmp/tfplugin.run
chmod +x /tmp/tfplugin.run
```

### 3.2 提取并安装 npu_device wheel

```bash
# 第一层解包
/tmp/tfplugin.run --noexec --extract=/tmp/tfplugin_ext

# 第二层解包
chmod +x /tmp/tfplugin_ext/run_package/CANN-fwkplugin-*.run
/tmp/tfplugin_ext/run_package/CANN-fwkplugin-*.run --noexec --extract=/tmp/fwk_ext

# 安装 npu_device wheel
pip install /tmp/fwk_ext/fwkplugin/bin/npu_device-0.1-py3-none-any.whl
```

### 3.3 安装 CANN Python 依赖

```bash
pip install decorator sympy scipy attrs psutil
```

### 3.4 验证 NPU 设备

```bash
source /home/Ascend/8.5/cann-8.5.0/set_env.sh  # 根据实际路径

python -c "
import npu_device
npu = npu_device.open().as_default()
print('NPU device opened')
import tensorflow as tf
print('result:', tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy())
"
# 输出 result: 6.0 表示成功
```

---

## Step 4：代码适配

### 4.1 最小侵入式适配（推荐）

对于大多数 TF 2.x 项目，只需在入口脚本顶部添加两行：

```python
import npu_device
npu_device.open().as_default()
```

TF 的标准 Keras API（Dense、Conv、LSTM、BatchNorm 等）会自动在 NPU 上执行。

### 4.2 自动迁移工具 convert_tf2npu（可选）

CANN 提供了 `convert_tf2npu` 工具进行批量代码转换：

```bash
# 工具位置（以 CANN 8.5 为例）
TOOL=/home/Ascend/8.5/cann-8.5.0/tools/ms_fmk_transplt

# 也可在 tensorflow-1.15-ascend 仓库中找到
TOOL=/home/show/tensorflow-1.15-ascend/convert_tf2npu

# 运行（TF 2.x 项目使用 v2 配置）
cd $TOOL
python main.py -i /path/to/project -o /path/to/output -m /path/to/main.py
```

> 注意：该工具主要处理 `tf.Session`、`tf.ConfigProto` 等 TF 1.x 模式的适配。
> 对于纯 TF 2.x Eager 模式项目，通常手动添加 `npu_device.open()` 即可。

### 4.3 常见适配点

| 原始代码 | 适配方式 |
|---------|---------|
| `import tensorflow as tf` | 前面加 `import npu_device; npu_device.open().as_default()` |
| `tf.device('/gpu:0')` | 删除或改为 `tf.device('/cpu:0')`（npu_device 自动接管默认设备） |
| `tensorflow-gpu` 依赖 | 替换为源码编译的 `tensorflow` |
| `tf.config.experimental.set_memory_growth(gpu, True)` | 删除（NPU 不需要） |
| CUDA 相关环境变量 | 不再需要，改用 `ASCEND_RT_VISIBLE_DEVICES` |

---

## Step 5：精度验证

### CPU vs NPU 对比模板

```python
import json, numpy as np

# 加载 CPU 和 NPU 的预测结果
with open('cpu_pred_scores.json') as f:
    cpu = json.load(f)
with open('npu_pred_scores.json') as f:
    npu = json.load(f)

cpu_scores = np.array(cpu['Y_hat'])
npu_scores = np.array(npu['Y_hat'])

max_diff = np.max(np.abs(cpu_scores - npu_scores))
mean_diff = np.mean(np.abs(cpu_scores - npu_scores))
cos_sim = np.dot(cpu_scores.flatten(), npu_scores.flatten()) / (
    np.linalg.norm(cpu_scores) * np.linalg.norm(npu_scores) + 1e-12)

print(f"max_diff={max_diff:.6e}")
print(f"mean_diff={mean_diff:.6e}")
print(f"cosine_sim={cos_sim:.8f}")
```

### 预期精度范围

| 精度模式 | 典型 max_diff | 说明 |
|---------|--------------|------|
| allow_fp32 | < 1e-5 | 纯 FP32，精度最高 |
| allow_hf32（默认） | < 1e-2 | 使用 HF32 混合精度，性能更好 |

如需更高精度：

```python
import npu_device
npu_device.global_options().precision_mode = "allow_fp32"
npu_device.open().as_default()
```

---

## 踩坑记录

### 编译相关

1. **aarch64 无 pip wheel**：TF 2.6.5 在 aarch64 上无 pip 预编译包，必须源码编译
2. **ABI 必须为 0**：TF 和 tfplugin 的 `_GLIBCXX_USE_CXX11_ABI` 必须一致（均为 0）
3. **nsync 补丁**：aarch64 上 nsync 原子操作需要添加 `__sync_synchronize()` 屏障
4. **GitHub 超时**：bazel 下载依赖时 GitHub 可能超时，用 `sed` 替换为镜像（ghfast.top）
5. **Bazel 版本**：TF 2.6.5 严格要求 Bazel 3.7.2，过高或过低都会失败

### 运行时相关

1. **npu_device 缺少 Python 依赖**：`decorator`、`sympy` 等模块缺失会导致 GE 引擎初始化失败
2. **numpy 版本**：TF 2.6 不兼容 numpy 1.24+（`np.object` 已移除），需使用 1.21.x
3. **h5py 版本**：TF 2.6 要求 h5py~=3.1.0，但 3.7.0 也可运行，用 `--no-deps` 安装 TF 绕过
4. **CANN 版本兼容性**：tfplugin 8.0.RC3 的 npu_device 可在 CANN 8.5 环境下正常工作

### 精度相关

1. **HF32 模式**：Ascend910 默认使用 HF32（half-float32），部分算子精度低于纯 FP32
2. **Unrecognized graph engine option 警告**：CANN 版本差异导致，不影响正确性

---

## 已知限制

- 仅支持 TF 2.6.5（与 tfplugin ABI 绑定）
- aarch64 必须源码编译 TF（约 4-5 分钟，取决于核心数）
- 不支持 tf.distribute 多卡策略（需使用 npu_device 专用分布式 API）
- 动态 shape 算子部分可能 fallback 到 CPU 执行

## 配套脚本

- TensorFlow Community 环境检查：`python scripts/check_tf_npu_env.py --open-device`

## 参考资料

- TF Community 编译与安装检查单：[`references/build-checklist.md`](references/build-checklist.md)
