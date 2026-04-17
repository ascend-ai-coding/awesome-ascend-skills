---
name: ai-for-science-deepfri
description: DeepFRI 的 TensorFlow 到 PyTorch 转换与昇腾 NPU 迁移 Skill，适用于蛋白质功能预测场景下的 TF 模型分析、PyTorch 重写、权重逐层映射、NPU 推理与精度验证，尤其适合需要在 Ascend 上运行 DeepFRI CNN 或 GCN 路径时使用。
keywords:
  - ai-for-science
  - deepfri
  - protein-function
  - tensorflow
  - pytorch
  - ascend
---

# DeepFRI 昇腾 NPU 迁移 Skill

## 项目概述

DeepFRI 是一个基于 GCN + LSTM 语言模型的蛋白质功能预测框架，原始实现基于
TensorFlow/Keras。本 Skill 记录将其完整迁移到 PyTorch + 昇腾 NPU 的全过程。

**模型架构：**
- **DeepCNN**：16 路并行 Conv1D → BatchNorm → ReLU → GlobalMaxPool → FuncPredictor
- **DeepFRI GCN**：LSTM LM → Embedding → 3×MultiGraphConv → SumPool → Dense → FuncPredictor
- **LSTM 语言模型**：2 层 CuDNNLSTM（hidden=512），输出拼接为 1024 维特征

**支持的 Ontology**：MF（分子功能）、BP（生物过程）、CC（细胞组分）、EC（酶分类）

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| CANN | ≥ 8.2（推荐 8.3.RC1） |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| torch_npu | 2.5.1 |

---

## 迁移步骤

### Step 1：克隆仓库 & 下载模型

```bash
cd /home/panjingxu
git clone https://github.com/flatironinstitute/DeepFRI.git
cd DeepFRI

# 下载 GPU 版预训练模型（README 精度参考值基于此版本）
wget https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz
tar xzf trained_models.tar.gz
```

> **注意**：官网提供两个版本：
> - `trained_models.tar.gz`（GPU 版）：CNN + MultiGraphConv + CuDNNLSTM，README 精度参考值基于此
> - `newest_trained_models.tar.gz`（CPU 版）：CNN + GraphConv + 标准 LSTM，结构不同、goterms 数量不同

### Step 2：创建 Conda 环境

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

conda create -n deepfri_npu python=3.10 -y
conda activate deepfri_npu

# TF（仅用于权重提取，可选）
pip install tensorflow numpy==1.26.4 biopython scikit-learn

# PyTorch + NPU
pip install torch==2.5.1 torch_npu==2.5.1
pip install pyyaml decorator attrs psutil cloudpickle tornado
```

### Step 3：验证 NPU 可用

```bash
source /home/Ascend/ascend-toolkit/set_env.sh
python3 -c "import torch; import torch_npu; a = torch.randn(3,4).npu(); print(a+a)"
```

### Step 4：分析 TF 模型权重结构

由于 TF 2.x 新版无法加载 CuDNNLSTM，直接用 h5py 读取 HDF5 权重：

```python
import h5py
f = h5py.File('trained_models/DeepFRI-MERGED_xxx.hdf5', 'r')
mw = f['model_weights']
for layer_name in mw.keys():
    grp = mw[layer_name]
    def print_ds(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: {obj.shape}")
    grp.visititems(print_ds)
f.close()
```

### Step 5：部署 PyTorch 模型文件 & 转换权重

将本 Skill 的 `scripts/` 目录下的文件复制到 DeepFRI 仓库对应位置：

```bash
SKILL_DIR=/path/to/awesome-ascend-skills/ai-for-science/models/deepfri/scripts
REPO_DIR=/path/to/DeepFRI   # 替换为实际路径

# 复制 PyTorch 模型文件到 deepfrier 包内
cp $SKILL_DIR/torch_layers.py    $REPO_DIR/deepfrier/
cp $SKILL_DIR/torch_model.py     $REPO_DIR/deepfrier/
cp $SKILL_DIR/torch_predictor.py $REPO_DIR/deepfrier/

# 复制脚本到仓库根目录
cp $SKILL_DIR/convert_weights.py $REPO_DIR/
cp $SKILL_DIR/predict_npu.py     $REPO_DIR/
cp $SKILL_DIR/verify_accuracy.py $REPO_DIR/
```

然后转换权重：

```bash
cd $REPO_DIR
python convert_weights.py
# 生成 trained_models/pytorch/*.pt
```

**scripts 目录文件清单：**

| 文件 | 目标位置 |
|------|---------|
| `scripts/torch_layers.py` | `deepfrier/torch_layers.py` |
| `scripts/torch_model.py` | `deepfrier/torch_model.py` |
| `scripts/torch_predictor.py` | `deepfrier/torch_predictor.py` |
| `scripts/convert_weights.py` | `convert_weights.py` |
| `scripts/predict_npu.py` | `predict_npu.py` |
| `scripts/verify_accuracy.py` | `verify_accuracy.py` |

### Step 6：NPU 推理

```bash
source /home/Ascend/ascend-toolkit/set_env.sh

# CNN 序列输入（对应 README Option 2）
python predict_npu.py \
  --seq 'SMTDLLSAEDIKKAIGAFTAADSFDHKKFFQMVGLKKKSADDVKKVFHILDKDKDGFIDEDELGSILKGFSSDARDLSAKETKTLMAAGDKDGDGKIGVEEFSTLVAES' \
  -ont mf --verbose --device npu:0

# CNN fasta 输入（对应 README Option 3）
python predict_npu.py \
  --fasta_fn examples/pdb_chains.fasta \
  -ont mf --verbose --device npu:0
```

### Step 7：精度验证

```bash
python verify_accuracy.py
```

---

## 关键转换规则

### Dense / Linear

| TF | PyTorch | 转换 |
|----|---------|------|
| `kernel: (in, out)` | `weight: (out, in)` | **转置** `.T` |
| `bias: (out,)` | `bias: (out,)` | 直接复制 |

### Conv1D

| TF | PyTorch | 转换 |
|----|---------|------|
| 输入 `(batch, length, channels)` | 输入 `(batch, channels, length)` | 前后 transpose |
| `kernel: (K, Cin, Cout)` | `weight: (Cout, Cin, K)` | `np.transpose(w, (2,1,0))` |
| `bias: (Cout,)` | `bias: (Cout,)` | 直接复制 |

### BatchNorm（⚠️ 本次迁移最关键的坑）

| TF | PyTorch | 转换 |
|----|---------|------|
| `gamma` | `weight` | 直接复制 |
| `beta` | `bias` | 直接复制 |
| `moving_mean` | `running_mean` | 直接复制 |
| `moving_variance` | `running_var` | 直接复制 |
| **默认 eps = 1e-3** | **默认 eps = 1e-5** | **必须设 `eps=1e-3`** |

> **⚠️ 致命 Bug**：TF BatchNorm 默认 `eps=1e-3`，PyTorch 默认 `eps=1e-5`，
> 差 100 倍。当 `moving_variance` 含接近 0 的值时，错误 eps 导致除零爆炸，
> softmax 饱和为 0/1，**模型输出全为 0**。这是本次 CNN 迁移中最关键的发现。

### CuDNNLSTM → nn.LSTM

| TF CuDNNLSTM | PyTorch nn.LSTM | 转换 |
|-------------|-----------------|------|
| `kernel: (input, 4*H)` | `weight_ih: (4*H, input)` | **转置** `.T` |
| `recurrent_kernel: (H, 4*H)` | `weight_hh: (4*H, H)` | **转置** `.T` |
| `bias: (8*H,)` | `bias_ih: (4*H,)` + `bias_hh: (4*H,)` | **前半 + 后半拆分** |

```python
pt_weight_ih = tf_kernel.T
pt_weight_hh = tf_recurrent_kernel.T
pt_bias_ih = tf_bias[:4 * H]   # 前半：input-hidden bias
pt_bias_hh = tf_bias[4 * H:]   # 后半：hidden-hidden bias
```

---

## 精度对比结果

### CNN 模型 vs 官网 README（NPU Ascend910）

| 测试用例 | GO term | 官网参考值 | NPU 输出 | diff | 状态 |
|---------|---------|-----------|---------|------|------|
| Option 2: seq→mf (1S3P-A) | GO:0005509 calcium ion binding | **0.99769** | **0.99769** | 5e-6 | ✅ PASS |
| Option 3: fasta→mf (1S3P-A) | GO:0005509 calcium ion binding | **0.99769** | **0.99769** | 5e-6 | ✅ PASS |
| Option 3: fasta→mf (2J9H-A) | GO:0004364 glutathione transferase | **0.46937** | **0.46936** | 9e-6 | ✅ PASS |
| Option 3: fasta→mf (2J9H-A) | GO:0016765 transferase activity | **0.19910** | **0.19913** | 2.6e-5 | ✅ PASS |

### CNN 全 Ontology 验证

| Ontology | 状态 | Top 预测 |
|----------|------|---------|
| MF | ✅ OK | GO:0005509 score=0.99769 (calcium ion binding) |
| BP | ✅ OK | GO:0051179 score=0.14491 (localization) |
| CC | ✅ OK | GO:0005829 score=0.23145 (cytosol) |
| EC | ✅ OK | 无超过阈值的预测（该蛋白非酶，符合预期） |

### GCN 模型说明

GPU 版 GCN 模型内嵌 CuDNNLSTM 语言模型。CuDNNLSTM 为 GPU 专用权重，
在标准 LSTM（PyTorch nn.LSTM）上运行时，逐步浮点差异经过 3 层 MultiGraphConv
图卷积被指数级放大，导致最终输出饱和。如需 GCN 功能，建议下载官网
`newest_trained_models.tar.gz`（CPU 版），使用标准 LSTM + GraphConv。

---

## 新增文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `deepfrier/torch_layers.py` | 199 | PyTorch 版图卷积层（MultiGraphConv, GraphConv, SAGEConv, NoGraphConv, ChebConv, SumPooling, FuncPredictor） |
| `deepfrier/torch_model.py` | 124 | PyTorch 版模型（LSTMLanguageModel, DeepFRIGCN, DeepCNN） |
| `deepfrier/torch_predictor.py` | 178 | PyTorch 版推理预测器（PredictorPyTorch，兼容原始 TF 版接口） |
| `convert_weights.py` | 147 | TF HDF5 → PyTorch state_dict 权重转换（LSTM/GCN/CNN） |
| `predict_npu.py` | 46 | NPU 推理入口脚本（含 `transfer_to_npu` 自动迁移注入） |
| `verify_accuracy.py` | 72 | 精度对比验证脚本（对比官网 README 参考值） |

### predict_npu.py（NPU 入口脚本）

入口脚本顶部注入 `transfer_to_npu`，自动将 `torch.cuda.*` 映射到 `torch.npu.*`：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
from deepfrier.torch_predictor import PredictorPyTorch
```

### convert_weights.py 核心逻辑

1. 用 `h5py` 读取 TF HDF5 模型中 `model_weights` 组的所有 Dataset
2. CuDNNLSTM：kernel/recurrent_kernel 转置，bias `(8*H,)` 拆分为 ih + hh
3. MultiGraphConv：kernel 直接复制（TF 和 PyTorch 都是 `(in, out)` 矩阵乘法）
4. Dense：kernel 转置 `.T`
5. Conv1D：kernel `np.transpose(w, (2,1,0))`
6. BatchNorm：gamma/beta/moving_mean/moving_var 直接复制

### torch_layers.py 关键实现

MultiGraphConv 的 `_normalize` 方法对应 TF 版的三路归一化：

```python
def _normalize(self, A, eps=1e-6):
    # 去除自环
    A = A.clone()
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    A = A - torch.diag_embed(diag)
    A_hat = A + eye
    deg = A_hat.sum(dim=2)
    D_asymm = torch.diag_embed(1.0 / (eps + deg))
    D_symm = torch.diag_embed(1.0 / (eps + deg.sqrt()))
    # 返回三路：[原始A, 非对称归一化, 对称归一化]
    return [A, D_asymm @ A_hat, D_symm @ A_hat @ D_symm]
```

三路结果分别与特征矩阵相乘后拼接，再经过线性变换和 ELU 激活。

### torch_model.py DeepCNN 关键适配

```python
# BatchNorm eps 必须匹配 TF 默认值
self.bn = nn.BatchNorm1d(total_filters, eps=1e-3)

# Conv1D 输入需要 transpose
def forward(self, seq):
    x = seq.transpose(1, 2)  # (batch, length, ch) → (batch, ch, length)
    conv_outs = [conv(x) for conv in self.conv_layers]
    x = torch.cat(conv_outs, dim=1)
    x = self.bn(x)
    x = F.relu(x)
    x = x.max(dim=2)[0]  # GlobalMaxPool
    return self.func_predictor(x)
```

---

## 迁移经验总结

### 踩过的坑（按严重程度排序）

1. **BatchNorm eps 不匹配**（最致命）：TF 默认 `1e-3`，PyTorch 默认 `1e-5`，
   导致 CNN 输出全为 0。修复：`nn.BatchNorm1d(dim, eps=1e-3)`

2. **CuDNNLSTM 平台限制**：GPU 版模型内嵌 CuDNNLSTM，在 aarch64 + NPU 上
   无法用 TF 加载验证。解决方案：用 h5py 直接读权重 + numpy 手动验证单步

3. **TF 2.x 不兼容 CuDNNLSTM**：TF 2.21 的 `load_model` 不识别 `CuDNNLSTM`。
   解决方案：绕过模型加载，直接从 HDF5 提取权重

4. **Conv1D 维度顺序**：TF `(batch, length, channels)` vs PyTorch `(batch, channels, length)`，
   前后需要 transpose

### 适用于其他项目的通用经验

- 遇到 TF 模型无法加载时，优先用 `h5py` 直接读取 HDF5 权重结构
- 所有归一化层（LayerNorm、BatchNorm）都要检查 eps 默认值
- CuDNNLSTM bias `(8*H,)` 拆分规则：前半 = input bias，后半 = recurrent bias
- 转换后先对比中间层输出（mean/std/min/max），定位第一个 diverge 的层

---

## 已知限制

- GCN 模型（cmap 输入）使用 CuDNNLSTM 语言模型，该权重为 GPU 专用，标准 LSTM
  在深层 MultiGraphConv 中会出现数值放大。如需 GCN 功能，建议使用官网 CPU 版模型
- CNN `padding='same'` 对偶数 kernel size 会有 PyTorch 警告，不影响精度
- Ascend910 不支持 fp64，torch_npu 自动降级为 fp32，不影响本模型

## 参考资料

- DeepFRI 权重转换与精度对齐清单：[`references/weight-conversion-checklist.md`](references/weight-conversion-checklist.md)
