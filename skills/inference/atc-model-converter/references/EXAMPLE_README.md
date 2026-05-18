# FunASR Paraformer 语音识别 — Ascend NPU 端到端推理指南

本文档记录了将阿里达摩院 FunASR Paraformer-large 中文语音识别模型部署到华为昇腾 NPU 的完整流程。
照此文档从头到尾执行，即可在 Ascend 910B 上完成模型转换并运行端到端语音识别推理。

## 环境信息

| 项目 | 值 |
|------|-----|
| SoC | Ascend910B3 |
| CANN | 8.1.RC1 / 8.3.RC2 |
| Python | 3.10 (conda env: `atc_py310`) |
| PyTorch | 2.10.0+cpu |
| FunASR | 1.3.1 |
| NumPy | 1.26.4 |
| 模型 | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| 词表大小 | 8404 tokens |
| 输入特征 | 560-dim (80-dim Fbank × 7 LFR frames) |

## 模型简介

Paraformer (Parallel Transformer) 是达摩院开源的非自回归端到端语音识别模型，在中文 ASR 任务上达到业界领先水平。

**架构：**
```
Audio (16kHz WAV)
  → Fbank (80-dim) → LFR (7帧拼接, 步长6) → CMVN 归一化 → [560-dim features]
  → Encoder (50层 SANM) → [hidden states + CIF alphas]
  → CIF 积分 (CPU) → [acoustic embeddings]
  → Decoder (16层 SANM) → logits → argmax → 文本
```

**为什么要拆分模型：** Paraformer 的 CIF (Continuous Integrate-and-Fire) 模块使用了 `Loop`、`NonZero`、`ScatterND` 等动态算子，ATC 均不支持。因此将模型拆分为 Encoder OM + Decoder OM，CIF 积分用 numpy 在 CPU 上完成（<2ms）。

## Quick Start（快速开始）

假设已完成环境准备和 OM 转换，3 条命令即可运行推理：

```bash
cd /home/z00879328/01-projects/02-internal/01-pae/07-ATC-SKILLS-TEST/funasr_paraformer
conda activate atc_py310
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 运行推理
python3 infer_om.py --wav test.wav
```

预期输出：
```
============================================================
Input: test.wav
============================================================

[1/5] Preprocessing audio...
  Waveform: 88786 samples (5.55s)
  Fbank frames: 553
  LFR frames: 93 (dim=560)
  -> speech: (1, 500, 560), actual_frames: 93, speech_lengths: [500]

[2/5] Running encoder on NPU...
  -> enc: (1, 500, 512), alphas: (1, 500)
  -> token_num (predicted): 17.0
  -> alphas sum (actual region): 17.0000
  -> Encoder time: 6.700s

[3/5] CIF integration (CPU)...
  -> acoustic_embeds: (1, 16, 512), tokens: 16
  -> CIF time: 0.001s

[4/5] Running decoder on NPU...
  -> logits: (1, 200, 8404)
  -> Decoder time: 2.000s

[5/5] Postprocessing...
  -> Token IDs (16): [...]

============================================================
Recognized text: 欢迎大家来体验达摩院推出的语音识别模型
============================================================
```

---

## 详细步骤

### Step 1: 环境准备

```bash
# SSH 到服务器
ssh root@175.99.1.3
cd /home/z00879328/01-projects/02-internal/01-pae/07-ATC-SKILLS-TEST
mkdir -p funasr_paraformer && cd funasr_paraformer

# 激活 conda 环境 (Python 3.10，必须 ≤ 3.10)
conda activate atc_py310

# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装依赖
pip install funasr==1.3.1 modelscope torchaudio onnx==1.16.1 onnxruntime==1.18.1
pip install "numpy<2.0"
```

> **注意：** Python 必须 ≤ 3.10，否则 CANN TBE 编译不兼容。NumPy 必须 < 2.0。

### Step 2: 下载模型并导出 ONNX

创建 `export_onnx.py`：

```python
#!/usr/bin/env python3
"""Export Paraformer model to split ONNX (encoder + decoder)."""
import os
import torch

# === PyTorch 2.10 兼容性 Monkey-patch ===
# PyTorch 2.10 的 torch.onnx.export 默认使用 dynamo 导出器，
# 与 FunASR 的 dynamic_axes 参数不兼容，需要强制使用 legacy API
_original_export = torch.onnx.export
def patched_export(*args, **kwargs):
    kwargs['dynamo'] = False
    return _original_export(*args, **kwargs)
torch.onnx.export = patched_export

from funasr import AutoModel
from funasr.register import tables
from funasr.utils.torch_function import sequence_mask

# 下载并加载模型（首次运行会自动下载 ~1GB）
print("Loading Paraformer model...")
model = AutoModel(
    model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    device='cpu'
)
m = model.model
m.eval()

# 获取 Export 版本的子模块
encoder_class = tables.encoder_classes.get('SANMEncoderExport')
predictor_class = tables.predictor_classes.get('CifPredictorV2Export')
decoder_class = tables.decoder_classes.get('ParaformerSANMDecoderExport')

encoder_export = encoder_class(m.encoder, onnx=True)
predictor_export = predictor_class(m.predictor, onnx=True)
decoder_export = decoder_class(m.decoder, onnx=True)
make_pad_mask = sequence_mask(500, flip=False)

# ---- Encoder + CIF CNN 封装 ----
class EncoderPredCNN(torch.nn.Module):
    """Encoder + CIF 卷积层（只输出 alphas，不做 CIF 积分循环）"""
    def __init__(self, encoder, predictor, make_pad_mask):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.make_pad_mask = make_pad_mask

    def forward(self, speech, speech_lengths):
        batch = {'speech': speech, 'speech_lengths': speech_lengths}
        enc, enc_len = self.encoder(**batch)
        mask = self.make_pad_mask(enc_len)[:, None, :]
        alphas, token_num = self.predictor.forward_cnn(enc, mask)
        return enc, enc_len, alphas, token_num

# ---- Decoder 封装 ----
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, enc, enc_len, acoustic_embeds, token_length):
        decoder_out, _ = self.decoder(enc, enc_len, acoustic_embeds, token_length)
        return torch.log_softmax(decoder_out, dim=-1)

os.makedirs('split_onnx', exist_ok=True)

# 导出 Encoder ONNX
print("Exporting encoder ONNX...")
enc_model = EncoderPredCNN(encoder_export, predictor_export, make_pad_mask)
enc_model.eval()
torch.onnx.export(
    enc_model,
    (torch.randn(1, 500, 560), torch.tensor([500], dtype=torch.int32)),
    'split_onnx/encoder.onnx',
    opset_version=14,
    input_names=['speech', 'speech_lengths'],
    output_names=['enc', 'enc_len', 'alphas', 'token_num'],
    dynamic_axes={
        'speech': {0: 'batch', 1: 'feats_len'},
        'speech_lengths': {0: 'batch'},
        'enc': {0: 'batch', 1: 'enc_len_dim'},
        'enc_len': {0: 'batch'},
        'alphas': {0: 'batch', 1: 'alphas_len'},
        'token_num': {0: 'batch'},
    }
)

# 导出 Decoder ONNX
print("Exporting decoder ONNX...")
dec_model = DecoderWrapper(decoder_export)
dec_model.eval()
torch.onnx.export(
    dec_model,
    (torch.randn(1, 250, 512), torch.tensor([250], dtype=torch.int32),
     torch.randn(1, 100, 512), torch.tensor([100], dtype=torch.int32)),
    'split_onnx/decoder.onnx',
    opset_version=14,
    input_names=['enc', 'enc_len', 'acoustic_embeds', 'token_length'],
    output_names=['logits'],
    dynamic_axes={
        'enc': {0: 'batch', 1: 'enc_len_dim'},
        'enc_len': {0: 'batch'},
        'acoustic_embeds': {0: 'batch', 1: 'token_len'},
        'token_length': {0: 'batch'},
        'logits': {0: 'batch', 1: 'logits_len'},
    }
)

print("Done! Files saved to split_onnx/")
print("  split_onnx/encoder.onnx")
print("  split_onnx/decoder.onnx")
```

运行：
```bash
python3 export_onnx.py
```

同时复制模型附带的配置文件到工作目录：
```bash
# am.mvn (CMVN 统计量) 和 tokens.json (词表) 从下载的模型目录中复制
# 模型默认缓存在 ~/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
MODEL_CACHE=~/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
cp $MODEL_CACHE/am.mvn .
cp $MODEL_CACHE/tokens.json .
```

### Step 3: ONNX → OM 转换 (ATC)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TE_PARALLEL_COMPILER=16
mkdir -p split_om

# Encoder OM (输入: 1×500×560 的音频特征)
atc --model=split_onnx/encoder.onnx \
    --framework=5 \
    --output=split_om/encoder \
    --soc_version=Ascend910B3 \
    --input_shape="speech:1,500,560;speech_lengths:1" \
    --log=info

# Decoder OM (输入: encoder输出 + acoustic embeddings)
atc --model=split_onnx/decoder.onnx \
    --framework=5 \
    --output=split_om/decoder \
    --soc_version=Ascend910B3 \
    --input_shape="enc:1,500,512;enc_len:1;acoustic_embeds:1,200,512;token_length:1" \
    --log=info
```

转换完成后检查输出：
```bash
ls -lh split_om/
# encoder_linux_aarch64.om  (~417MB)
# decoder_linux_aarch64.om  (~199MB)
```

> **注意：** `--soc_version` 必须与设备完全匹配。运行 `npu-smi info | grep Name` 获取准确型号。

### Step 4: 准备推理脚本

将 `preprocess.py` 和 `infer_om.py` 放到工作目录下。这两个脚本实现了完整的推理流程：

- **`preprocess.py`** — 音频预处理：WAV → Fbank(80-dim) → LFR(560-dim) → CMVN → 零填充
- **`infer_om.py`** — 端到端推理：预处理 → Encoder(NPU) → CIF积分(CPU) → Decoder(NPU) → 文本

### Step 5: 运行推理

```bash
# 准备测试音频 (16kHz 单声道 WAV)
# 可以用 FunASR 提供的测试音频，或自己录制

# 运行推理
python3 infer_om.py --wav test.wav

# 指定其他音频
python3 infer_om.py --wav /path/to/your/audio.wav
```

#### 推理脚本完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wav` | `test.wav` | 输入音频文件 (16kHz 单声道 WAV) |
| `--encoder-om` | `split_om/encoder_linux_aarch64.om` | Encoder OM 路径 |
| `--decoder-om` | `split_om/decoder_linux_aarch64.om` | Decoder OM 路径 |
| `--mvn` | `am.mvn` | CMVN 统计量文件 |
| `--tokens` | `tokens.json` | 词表文件 |
| `--max-feats-len` | `500` | 最大输入帧数 (须与 ATC 转换一致) |
| `--max-token-len` | `200` | 最大输出 token 数 (须与 ATC 转换一致) |

---

## 关键挑战与解决方案

### 1. PyTorch 2.10 ONNX 导出不兼容

**问题：** PyTorch 2.10 的 `torch.onnx.export` 默认使用 dynamo 导出器，与 FunASR 的 `dynamic_axes` 参数冲突，报错：
```
TypeError: export() got an unexpected keyword argument 'dynamic_axes'
```

**解决：** Monkey-patch `torch.onnx.export`，强制使用 legacy API：
```python
_original_export = torch.onnx.export
def patched_export(*args, **kwargs):
    kwargs['dynamo'] = False
    return _original_export(*args, **kwargs)
torch.onnx.export = patched_export
```

### 2. ModelScope 预导出 ONNX 含量化算子

**问题：** ModelScope 上的预导出 ONNX 模型 (`model_quant.onnx`) 使用 `MatMulInteger` 和 `DynamicQuantizeLinear` 算子，ATC 不支持。

**解决：** 从 PyTorch checkpoint 重新导出 FP32 ONNX。不要使用预导出的量化版本。

### 3. CIF 模块含 ATC 不支持的动态算子

**问题：** Paraformer 的 CIF 模块使用：
- `Loop` 算子（CIF 积分循环）
- `NonZero` / `ScatterND`（布尔索引）
- 动态形状 `Concat`

即使 FunASR 内置的向量化导出版本 (`cif_v1_export`) 也使用了 ATC 无法处理的高级索引操作。

**解决：** 将模型拆分为 Encoder（含 CIF 卷积层）和 Decoder 两个 OM。CIF 积分循环在 CPU 上用 numpy 实现，计算量极小（<2ms）。

### 4. 静态形状下 speech_lengths 的 broadcast 错误

**问题：** ATC 需要固定输入形状，但 Encoder 内部的 attention mask 依赖 `speech_lengths` 的实际值。当 `speech_lengths < max_feats_len` 时触发形状不匹配错误。

**解决：** 始终将 `speech_lengths` 设为 `max_feats_len`（500），让 Encoder 处理包含零填充的完整输入。副作用是 CIF 可能多预测 1 个 token，通过后处理中的启发式规则修正（检测 alphas 求和的小数部分）。

---

## 预处理流程详解

```
16kHz WAV
  ↓ load_audio()          读取为 float32，转单声道
  ↓ ×32768                缩放到 int16 范围 (Paraformer 要求)
  ↓ pre-emphasis(0.97)    预加重
  ↓ framing               帧长25ms(400样本), 帧移10ms(160样本), Hamming窗
  ↓ 512-point FFT         取功率谱
  ↓ 80 Mel filters        20Hz ~ 8000Hz
  ↓ log                   取对数
  ↓ LFR(m=7, n=6)         每6帧取7帧拼接 → 560维
  ↓ CMVN(am.mvn)          均值方差归一化
  ↓ zero-pad to 500       填充到固定长度
  ↓
(1, 500, 560) float32    → 送入 Encoder OM
```

## 验证结果

| 测试音频 | 时长 | FunASR PyTorch 结果 | OM 推理结果 | 匹配 |
|----------|------|-------------------|-----------|------|
| test.wav | 5.55s | 欢迎大家来体验达摩院推出的语音识别模型 | 欢迎大家来体验达摩院推出的语音识别模型 | 精确匹配 |
| test2.wav | 7.92s | 国务院发展研究中心市场经济研究所副所长邓玉松认为 | 国务院发展研究中心市场经济研究所副所长邓玉松认为 | 精确匹配 |

## 性能参考

| 阶段 | 执行位置 | 时间 |
|------|---------|------|
| 预处理 | CPU | ~0.01s |
| Encoder | NPU | ~6.7s |
| CIF 积分 | CPU | ~0.001s |
| Decoder | NPU | ~2.0s |
| **总计** | | **~8.7s** |

> 注：首次推理包含模型加载时间，后续推理会显著更快。

## 文件结构

```
funasr_paraformer/
├── README.md                     # 本文档
├── preprocess.py                 # 音频预处理 (Fbank + LFR + CMVN)
├── infer_om.py                   # 端到端 OM 推理脚本
├── export_onnx.py                # ONNX 导出脚本
├── am.mvn                        # CMVN 统计量 (随模型下载)
├── tokens.json                   # 词表 8404 tokens (随模型下载)
├── test.wav                      # 测试音频 1
├── test2.wav                     # 测试音频 2
├── split_onnx/
│   ├── encoder.onnx              # Encoder ONNX (~636MB)
│   └── decoder.onnx              # Decoder ONNX (~229MB)
└── split_om/
    ├── encoder_linux_aarch64.om  # Encoder OM (~417MB)
    └── decoder_linux_aarch64.om  # Decoder OM (~199MB)
```

## 已知限制

1. **最大音频时长 ~30 秒** — 静态形状 `max_feats_len=500` 限制。更长音频需重新 ATC 转换（增大 `--input_shape` 中的帧数）或对长音频分段处理。
2. **仅 batch_size=1** — 当前 ATC 转换仅支持单条推理。可通过 `--dynamic_batch_size` 支持批量推理。
3. **CIF 在 CPU 执行** — CIF 积分未在 NPU 上运行，但计算量极小（<2ms），不影响整体性能。
4. **零填充影响 CIF** — 由于 `speech_lengths` 必须等于填充长度，CIF 可能多预测 1 个 token，通过启发式规则修正。
5. **仅标准 Paraformer** — 未使用 SeACo 热词增强版本（其 Loop 算子更多）。
6. **Fbank 实现差异** — `preprocess.py` 使用纯 numpy 实现的 Kaldi 风格 Fbank，与 FunASR 原生实现在边界帧上可能有微小差异，但不影响识别精度。
