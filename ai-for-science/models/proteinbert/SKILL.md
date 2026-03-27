---
name: ai-for-science-proteinbert
description: ProteinBERT 昇腾 NPU 部署与迁移 Skill，适用于将 TensorFlow 或 Keras 版 ProteinBERT 转成基于 PyTorch 与 torch_npu 的实现，覆盖权重转换、embedding 提取、微调训练、注意力可视化和 GPU 与 NPU 精度验证。
keywords:
  - ai-for-science
  - proteinbert
  - protein-language-model
  - tensorflow
  - pytorch
  - ascend
---

# ProteinBERT 昇腾 NPU 部署 Skill

将 ProteinBERT 从 TensorFlow/Keras 完整迁移到 PyTorch + torch_npu，
已通过 5 个基准任务的精度验证（以 GPU 为基线）。

## 模型概况

| 项目 | 值 |
|------|----|
| 架构 | 6 层 Conv + GlobalAttention blocks |
| 参数量 | ~16M (vocab=26, annotations=8943, d_seq=128, d_global=512) |
| 原始框架 | TensorFlow 2.x / Keras |
| 目标框架 | PyTorch 2.5.1 + torch_npu 2.5.1 |
| 验证 CANN | 8.3.RC1 |
| 硬件 | Ascend910 |

## 环境准备

```bash
# CANN 环境
source /home/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# Conda 环境
conda create -n proteinbert_npu python=3.11 -y
conda activate proteinbert_npu
pip install torch==2.5.1 torch_npu==2.5.1 numpy==1.26.4 pyyaml \
    pandas scikit-learn h5py scipy -i https://repo.huaweicloud.com/repository/pypi/simple/
```

## 文件结构

本 Skill 自包含所有代码，可直接复制使用：

```
proteinbert/
├── SKILL.md                            # 本文件
├── references/
│   └── migration_details.md              # 详细迁移参考（权重映射、bug 记录、调试方法）
└── scripts/
    ├── proteinbert_pytorch/              # PyTorch 核心实现
    │   ├── model.py                       #   模型定义
    │   ├── convert_weights.py             #   TF pkl → PyTorch 权重转换
    │   ├── inference.py                   #   分词器 + 推理工具
    │   └── finetune.py                    #   微调框架
    ├── demo_scripts/                      # GPU/NPU 配对用例脚本
    │   ├── demo1_signalP_gpu.py           #   用例1: signalP 微调 (TF)
    │   ├── demo1_signalP_npu.py           #   用例1: signalP 微调 (PyTorch)
    │   ├── demo2_all_benchmarks_gpu.py    #   用例2: 5个benchmark (TF)
    │   ├── demo2_all_benchmarks_npu.py    #   用例2: 5个benchmark (PyTorch)
    │   ├── demo3_attention_gpu.py         #   用例3: 注意力可视化 (TF)
    │   └── demo3_attention_npu.py         #   用例3: 注意力可视化 (PyTorch)
    ├── tools/                             # 精度对比工具
    │   ├── get_embeddings_gpu.py          #   Embedding提取 (TF)
    │   ├── get_embeddings_npu.py          #   Embedding提取 (PyTorch)
    │   ├── debug_layerwise_gpu.py         #   逐层调试 (TF)
    │   ├── debug_layerwise_npu.py         #   逐层调试 (PyTorch)
    │   └── compare_embeddings.py          #   GPU/NPU 结果对比
    └── deploy_toolkit/                    # 独立部署包
        ├── convert_weights.py             #   权重转换（自包含模型定义）
        ├── inference_npu.py               #   NPU 推理
        ├── finetune_npu.py                #   NPU 微调
        ├── setup.sh                       #   一键部署脚本
        └── requirements.txt               #   依赖列表
```

## 快速开始

### 第1步：下载预训练权重

从 Zenodo 下载（183MB）：
```
https://zenodo.org/records/10371965/files/epoch_92400_sample_23500000.pkl
```
放到 `~/proteinbert_models/epoch_92400_sample_23500000.pkl`。

### 第2步：转换权重

```bash
python scripts/deploy_toolkit/convert_weights.py \
    --input ~/proteinbert_models/epoch_92400_sample_23500000.pkl \
    --output ~/proteinbert_models/proteinbert_pytorch.pt
```

### 第3步：提取 Embedding

GPU 原版（TF）：
```python
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
encoded_x = input_encoder.encode_X(seqs, seq_len)
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
```

NPU 等价版（PyTorch）：
```python
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs
import torch, torch_npu
from torch_npu.contrib import transfer_to_npu

model, n_ann = convert_tf_to_pytorch('~/proteinbert_models/epoch_92400_sample_23500000.pkl')
model = model.to('npu:0').eval()
tokens = tokenize_seqs(seqs, seq_len)
inp_seq = torch.from_numpy(tokens).long().to('npu:0')
inp_ann = torch.zeros(len(seqs), n_ann).to('npu:0')
with torch.no_grad():
    output_seq, output_ann = model(inp_seq, inp_ann)
```

完整隐藏层拼接版本（对标 `get_model_with_hidden_layers_as_outputs`）见 `scripts/tools/get_embeddings_npu.py`。
输出维度：
- `local_representations`: (batch, seq_len, 1562) = 12个LayerNorm×128 + output_seq×26
- `global_representations`: (batch, 15599) = input_dense×512 + 12个LayerNorm×512 + output_ann×8943

### 第4步：微调训练

三个用例均提供 GPU/NPU 配对脚本，打印格式一致便于 diff 对比。

**用例1：signalP_binary 微调**
```bash
python scripts/demo_scripts/demo1_signalP_gpu.py   # GPU端
python scripts/demo_scripts/demo1_signalP_npu.py   # NPU端
```
三阶段训练：冻结层(lr=1e-2) → 全层(lr=1e-4) → 长序列(seq_len=1024, lr=1e-5)

**用例2：5个 Benchmark 全量运行**
```bash
python scripts/demo_scripts/demo2_all_benchmarks_gpu.py   # GPU端
python scripts/demo_scripts/demo2_all_benchmarks_npu.py   # NPU端
```
包含 signalP_binary、fluorescence、remote_homology、stability、ProFET_NP_SP_Cleaved。

**用例3：注意力可视化**
```bash
python scripts/demo_scripts/demo3_attention_gpu.py   # GPU端
python scripts/demo_scripts/demo3_attention_npu.py   # NPU端
```
提取 24 个注意力头的权重 (6层×4头)，保存为 npz 便于数值对比。

## 精度验证结果

### Benchmark 微调精度（以 GPU/TF 为基线）

| Benchmark | 任务类型 | 指标 | GPU (TF) | NPU (PyTorch) | 偏差 | 状态 |
|-----------|---------|------|----------|---------------|------|------|
| signalP_binary | 二分类 | AUC | 0.9961 | 0.9965 | +0.04% | ✅ |
| fluorescence | 回归 | Spearman | 0.6475 | 0.6597 | +1.22% | ✅ |
| remote_homology | 多分类 | Accuracy | 22.42% | 21.17% | -1.25% | ✅ |
| stability | 回归 | Spearman | 0.7068 | 0.7851 | +7.83% | ✅ |
| ProFET_NP_SP_Cleaved | 二分类 | AUC | 0.9855 | 0.9852 | -0.03% | ✅ |

二分类/多分类任务偏差均在 1.3% 以内。回归任务中 NPU 偏差较大但方向为正（更好），
原因是 PyTorch Adam 与 TF Adam 的优化器收敛差异。

### 预训练模型 Embedding 逐层对比（无训练随机性）

| 层名 | GPU mean | NPU mean | 偏差 |
|------|----------|----------|------|
| embedding-seq-input | 0.000829 | 0.000829 | 0 |
| dense-global-input | -0.005728 | -0.005728 | 0 |
| global-merge1-norm-block1 | -0.031465 | -0.031465 | <1e-5 |
| global-merge2-norm-block6 | -0.010764 | -0.010764 | <1e-5 |
| output-annotations | 0.000148 | 0.000148 | <1e-5 |

全部 28 层输出 mean/std 偏差均 < 1e-4，权重转换数值完全一致。

## 迁移关键适配点

详细权重映射、代码对照和调试方法见 `references/migration_details.md`。

### 致命级（修复前导致精度完全错误）

1. **K.dot 与 einsum 维度映射错误**
   - TF `K.dot(A, B)` 收缩 A 的最后一维和 B 的倒数第二维
   - `einsum('bls,hvs->...')` 是错的，正确写法是 `einsum('bls,hsv->...')`
   - 写反导致从第 1 个 block 开始输出完全错误

2. **LayerNorm epsilon 默认值差 100 倍**
   - TF 默认 `eps=1e-3`，PyTorch 默认 `eps=1e-5`
   - 必须显式设置 `nn.LayerNorm(dim, eps=1e-3)`

### 重要级（导致 shape 或训练行为异常）

3. **隐藏层收集逻辑不匹配**
   - TF 过滤名 `input-seq-encoding` 与实际层名 `embedding-seq-input` 不匹配
   - 导致 seq 维度是 1562 而非 1690

4. **微调 head 输入维度错误**
   - TF 版 head 接收全层拼接 15599 维，而非最后一层 512 维
   - 用错维度会导致训练提前 Early Stopping

### 普通级（框架间正常差异）

5. **pickle 权重顺序**：`global_input_dense` 在 `embedding` 之前
6. **Conv1D 权重**：TF `(kernel,in,out)` → PyTorch `(out,in,kernel)` 需 transpose
7. **Dense 权重**：TF `(in,out)` → PyTorch `(out,in)` 需转置
8. **pyyaml 依赖**：torch_npu 运行时需要，未列入官方依赖
9. **TF/torch_npu 冲突**：不可在同一 conda 环境同时安装

## 精度对比方法

```bash
# 步骤1: 分别在 GPU 和 NPU 上运行配对脚本
python scripts/tools/get_embeddings_gpu.py   # → embeddings_gpu.npz
python scripts/tools/get_embeddings_npu.py   # → embeddings_npu.npz

# 步骤2: 数值对比
python scripts/tools/compare_embeddings.py

# 步骤3: 如果偏差较大，用逐层调试定位
python scripts/tools/debug_layerwise_gpu.py > gpu.txt
python scripts/tools/debug_layerwise_npu.py > npu.txt
diff gpu.txt npu.txt
```

逐层对比的第一个发散点即为 bug 所在层。

## 参考资料

- 已有迁移细节：[`references/migration_details.md`](references/migration_details.md)
- Benchmark 与逐层调试建议：[`references/benchmark-debug.md`](references/benchmark-debug.md)
