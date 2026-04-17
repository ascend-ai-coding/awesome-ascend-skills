---
name: ai-for-science-oligoformer
description: OligoFormer 昇腾 NPU 迁移 Skill，适用于将基于 PyTorch Transformer 的 siRNA 效能预测模型迁移到华为 Ascend NPU，覆盖环境搭建、RNA-FM 依赖安装、代码适配、推理验证与可选训练流程。
keywords:
  - ai-for-science
  - oligoformer
  - sirna
  - rna-fm
  - pytorch
  - ascend
---

# OligoFormer 昇腾 NPU 迁移 Skill

本 Skill 提供将 OligoFormer（siRNA 效能预测模型）从 CUDA 迁移到昇腾 NPU 的完整步骤。
项目地址：https://github.com/lulab/OligoFormer.git

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | ≥ 8.0（推荐 8.2+ 或 8.3.RC1） |
| Python | 3.10（原项目要求 3.8，因 torch 2.5.1 不支持 3.8，升级至 3.10） |
| PyTorch | 2.5.1 |
| torch_npu | 2.5.1 |
| conda | 已安装 |

## 迁移流程

### 1. 环境初始化与 CANN 配置

```bash
# 设置 CANN 环境（路径根据实际安装位置调整）
source /home/Ascend/ascend-toolkit/set_env.sh

# 验证 NPU 可用
npu-smi info

# 设置华为镜像源
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 2. 克隆仓库

```bash
cd /home/show
git clone https://github.com/lulab/OligoFormer.git
cd OligoFormer
```

### 3. 创建 conda 环境并安装依赖

```bash
# 创建 Python 3.10 环境
conda create -n oligoformer-conda python=3.10 -y
conda activate oligoformer-conda

# 安装 PyTorch + torch_npu（版本需与 CANN 匹配）
pip install torch==2.5.1 torchvision==0.20.1 torch_npu==2.5.1 \
    -i https://repo.huaweicloud.com/repository/pypi/simple/

# 安装基础运行时依赖
pip install numpy==1.26.4 decorator attrs psutil absl-py cloudpickle \
    ml-dtypes scipy tornado \
    -i https://repo.huaweicloud.com/repository/pypi/simple/

# 安装项目依赖（不安装 requirements.txt 中的 torch/torchvision，已单独安装）
pip install bio matplotlib pandas prefetch-generator ptflops \
    pytorch-ignite scikit-learn tqdm yacs \
    -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### 4. 安装 RNA-FM 依赖

OligoFormer 依赖 RNA-FM 生成 RNA 嵌入特征。

```bash
cd /home/show/OligoFormer

# 下载打包好的 RNA-FM（含预训练权重）
wget 'https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1' -O RNA-FM.tar.gz
tar -zxf RNA-FM.tar.gz
```

### 5. 验证 torch_npu 可用

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
python -c "import torch; import torch_npu; a = torch.randn(3,4).npu(); print(a + a)"
```

应输出 `device='npu:0'` 的 Tensor。

### 6. 代码适配

#### 6.1 入口文件注入 transfer_to_npu

在 `scripts/main.py` 最顶部（所有 import 之前）添加：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

`transfer_to_npu` 会自动将 `torch.cuda.*` 映射到 `torch.npu.*`，
包括 `torch.device('cuda:0')` → `npu:0`、`.cuda()` → `.npu()` 等。

命令行操作：

```bash
sed -i '1i import torch_npu\nfrom torch_npu.contrib import transfer_to_npu' scripts/main.py
```

#### 6.2 添加 ASCEND_RT_VISIBLE_DEVICES 环境变量

在以下 4 个文件中，`CUDA_VISIBLE_DEVICES` 设置后追加 `ASCEND_RT_VISIBLE_DEVICES`：
- `scripts/train.py`
- `scripts/test.py`
- `scripts/test_single.py`
- `scripts/train_single.py`

```bash
sed -i 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = str(Args.cuda)/os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)\n\tos.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(Args.cuda)/' \
    scripts/train.py scripts/test.py scripts/test_single.py scripts/train_single.py
```

```bash
# 添加 CANN 环境 source
sed -i '1a source /home/Ascend/ascend-toolkit/set_env.sh' \
    scripts/RNA-FM.sh scripts/RNA-FM-features.sh

# 替换 python 为 conda 环境完整路径
sed -i 's|^python |/root/anaconda3/envs/oligoformer-conda/bin/python |g' \
    scripts/RNA-FM.sh scripts/RNA-FM-features.sh
```

### 7. 适配说明

#### 7.1 迁移特点

- **纯 PyTorch 项目**：无 .cu 文件，无 flash_attn/xformers 等 CUDA 特有第三方库
- **无分布式**：项目不使用 DataParallel 或 DDP
- **无 AMP**：项目不使用混合精度训练
- **CUDA 使用简单**：仅在设备选择处使用 `torch.cuda.is_available()` 和 `torch.device('cuda:0')`
- **transfer_to_npu 可覆盖所有 CUDA API**：无需手动修改 `torch.cuda.*` 调用

#### 7.2 注意事项

- `model.py` 中 `PositionalEncoding` 在 `__init__` 中使用模块级 `device` 变量创建 tensor，
  `transfer_to_npu` 会自动将其映射到 `npu:0`，无需额外处理
- Ascend910 不支持 fp64（double），torch_npu 会自动降级为 fp32，
  可能出现 warning 但不影响结果
- RNA-FM 的 `setup.py` 中依赖版本过旧，需放宽版本约束
- RNA-FM Shell 脚本中的 `python` 命令需要指向 conda 环境中的 Python

### 8. 验证

#### 8.1 推理验证

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd /home/show/OligoFormer

python scripts/main.py --infer 1 -i1 data/example.fa
```

#### 8.2 验证通过标准

- 程序正常退出，无 NPU 相关报错
- `result/` 目录下生成 `RNA*_ranked.txt`、`RNA*_ranked_filtered.txt` 等结果文件
- 输出包含 siRNA 效能预测值（efficacy 列非空）

#### 8.3 训练验证（可选，需生成 RNA-FM 特征）

```bash
# 先生成训练数据的 RNA-FM 特征
bash scripts/RNA-FM-features.sh

# 训练
python scripts/main.py --datasets Hu Mix --cuda 0 --learning_rate 0.0001 \
    --batch_size 16 --epoch 200 --early_stopping 30
```

### 9. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/main.py` | 顶部注入 `torch_npu` + `transfer_to_npu` |
| `scripts/train.py` | 添加 `ASCEND_RT_VISIBLE_DEVICES` |
| `scripts/test.py` | 添加 `ASCEND_RT_VISIBLE_DEVICES` |
| `scripts/test_single.py` | 添加 `ASCEND_RT_VISIBLE_DEVICES` |
| `scripts/train_single.py` | 添加 `ASCEND_RT_VISIBLE_DEVICES` |
| `RNA-FM/redevelop/launch/predict.py` | 注入 `torch_npu` + 添加 `ASCEND_RT_VISIBLE_DEVICES` |
| `RNA-FM/setup.py` | 放宽依赖版本约束 |
| `scripts/RNA-FM.sh` | source CANN 环境 + 使用 conda Python 路径 |
| `scripts/RNA-FM-features.sh` | source CANN 环境 + 使用 conda Python 路径 |

## 配套脚本

- OligoFormer 与 RNA-FM 预检：`python scripts/validate_oligoformer_env.py --rna-fm-path /path/to/RNA-FM`

## 参考资料

- OligoFormer 与 RNA-FM 集成说明：[`references/rna-fm-integration.md`](references/rna-fm-integration.md)
