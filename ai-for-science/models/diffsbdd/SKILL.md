---
name: ai-for-science-diffsbdd
description: DiffSBDD 昇腾 NPU 迁移 Skill，适用于将基于等变扩散模型的结构化药物设计项目从 CUDA 迁移到华为 Ascend NPU，覆盖环境搭建、依赖安装、torch_scatter 源码编译、代码适配以及 de novo 推理验证。
keywords:
  - ai-for-science
  - diffsbdd
  - drug-design
  - diffusion
  - torch-scatter
  - ascend
---

# DiffSBDD 昇腾 NPU 迁移 Skill

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | ≥ 8.2（推荐 8.2.RC1 或 8.3.RC1） |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| torch_npu | 2.5.1 |

## 迁移流程

### 1. 克隆仓库

```bash
git clone https://github.com/arneschneuing/DiffSBDD.git
cd DiffSBDD
```

### 2. 创建 Conda 环境

```bash
conda create -n diffSBDD-kernerlcat python=3.10 -y
conda activate diffSBDD-kernerlcat
```

### 3. 配置 CANN 环境和华为镜像源

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 4. 安装 PyTorch + torch_npu

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install torch_npu==2.5.1
pip install numpy==1.26.4
pip install pyyaml decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado
```

**关键**：numpy 必须为 1.26.4，numpy 2.x 与 CANN TBE 编译器不兼容。

### 5. 安装项目依赖

```bash
pip install pytorch-lightning==1.8.4 wandb biopython==1.79 imageio==2.21.2 \
    pandas==2.2.2 seaborn==0.13.2 torchmetrics==1.4.2 tqdm==4.66.5
conda install -c conda-forge rdkit openbabel -y
# conda 安装后可能覆盖 numpy，需重新固定
pip install numpy==1.26.4 --force-reinstall
# 检查并移除 conda 残留的 numpy-base 2.x
conda remove numpy-base --force -y 2>/dev/null
pip install numpy==1.26.4 --force-reinstall
```

### 6. 源码编译 torch_scatter

**强制要求：必须从源码编译，禁止使用 PyG 预编译 wheel。**

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
export FORCE_CUDA=0
export TORCH_CUDA_ARCH_LIST=''
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
python setup.py bdist_wheel
pip install dist/*.whl
cd ..
```

### 7. 代码适配

#### 7.1 注入 transfer_to_npu（自动迁移）

在以下 6 个入口文件顶部添加：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

需要修改的文件：
- `generate_ligands.py`
- `test.py`
- `inpaint.py`
- `optimize.py`
- `train.py`
- `lightning_modules.py`

`transfer_to_npu` 会自动将所有 `torch.cuda.*` 调用重定向到 NPU，
包括 `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 等模式。

#### 7.2 补充缺失的 `__init__.py`

```bash
touch analysis/__init__.py
touch analysis/SA_Score/__init__.py
```

### 8. 验证

#### 8.1 下载预训练模型

```bash
mkdir -p checkpoints
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```

#### 8.2 De novo 推理验证

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand example/3rfm_B_CFF.sdf \
    --outfile example/3rfm_generated.sdf \
    --n_samples 2 --num_nodes_lig 44
```

验证通过标准：
- 程序正常退出（exit code 0）
- 输出 SDF 文件非空且格式正确

## 特殊适配点

| 适配项 | 说明 |
|--------|------|
| numpy 版本 | 必须 1.26.4，conda 安装 rdkit 后会被覆盖为 2.x，需手动恢复 |
| numpy-base | conda 可能残留 numpy-base 2.x，需 `conda remove numpy-base --force` |
| torch_scatter | 必须源码编译，设置 `FORCE_CUDA=0` |
| wandb | 原始 0.13.1 不兼容 numpy 1.26.4，需升级到最新版 |
| `__init__.py` | `analysis/` 和 `analysis/SA_Score/` 缺少 `__init__.py` |
| fp64 精度 | Ascend910 不支持 double，自动降级为 fp32，扩散模型比较分布指标即可 |
| transfer_to_npu | 自动处理所有 CUDA→NPU 设备映射，无需手动改 device 逻辑 |

## 配套脚本

- 依赖与 NPU 运行时预检：`python scripts/validate_diffsbdd_env.py`

## 参考资料

- DiffSBDD 依赖与验证备注：[`references/dependency-notes.md`](references/dependency-notes.md)
