# Encoder 拆分流水线 + CANNPAK 打包

端侧分段加载时，把 encoder 拆成多段独立 omc，再连同 decoder / punc / embedding 打包成单个 bin。本文给出拆分流水线与打包格式。

---

## 为什么拆分

- 端侧按段加载、节省峰值内存；
- 便于单段替换 / 调试（某层有问题只重转该段）。

典型拆法：`encoder_emd`（embedding/前处理）+ `encoder_layer_0..N`（各 conformer 层）+ `encoder_ctc`（ctc 投影），共若干段。

---

## 拆分子图量化流水线

**前置**：先跑完整网量化（最好是 pwlinear 压缩版），拿到整网的 `quant_param_file` 与已落网格的权重 onnx。拆分**复用整网量化参数，不对每段重新 PTQ**。

每个 conformer 层子图（原始浮点）流水线：

```
encoder_layer_N.onnx (原始浮点)
  └ s0_pwlinear_convert.py   pointwise Conv1d → MatMul
  └ s3_rewrite_split.py      改名 encoder.encoders.N.* + MatMul 3D→2D unfold
  └ s5_grid_weights.py       【9030 关键】把整网 dopt 网格权重(UINT4 [0,15]) 按节点名搬进本段
  └ s4_buwei_split.py        激活 Pad / Slice 补维（不动权重）
  └ omg (容器)               --compress_conf=整网 quant_param_file --platform=kirin9030
```

特殊段：

- `encoder_emd`：先 `modify_encoder_emd.py`（offset int32→float、Cast 设 int32）→ rewrite → buwei → omg。
- `encoder_ctc`：`prepare_ctc_fixed.py` → rewrite → buwei → omg。

**`s5_grid_weights.py` 的必要性**（9030 专属）：整网参数按「量化到网格后的权重」标定，拆分子图是原始浮点（含负值），套同一份参数会量化出 -1 → UINT4 越界。把整网网格权重按 MatMul 节点名搬进每段即解决。`emd` / `ctc` 是 8bit（aigc 非对称），容忍原始权重，不需替换。原理详见 `platform-9030.md`。

---

## CANNPAK 打包成单 bin

把所有 omc + embedding 二进制按 CANNPAK 格式合成单文件，端侧用 `asr_init_from_packed()` 类接口加载。

**格式**：`CANNPAK\0` 头 + 索引表（每项 name / offset / size）+ 数据区。

**典型组成**（按实际模型数量调整）：

| 类别 | 来源 |
| --- | --- |
| encoder 各段 | `encoder_{emd, layer_0..N, ctc}/*.omc` |
| decoder | decoder 量化产物 `*.omc` |
| punc | 标点量化产物 `*.omc`（端侧匹配的内置 embed 单模型版） |
| embed_quantized | decoder 的 token embedding 量化二进制（如 vocab×dim 的 int8） |

**打包脚本（`pack_models.py` 思路）**：

```bash
python pack_models.py            # 打包 -> packed_models/xxx_packed.bin
python pack_models.py --info     # 查看各项 offset / size
python pack_models.py --extract [out_dir]   # 解包校验
```

各项来源路径建议用环境变量可覆盖（如 `DECODER_OMC=/path python pack_models.py`），便于换件重打。

---

## 打包前自检清单

- [ ] 每段 omc 都在容器内转成功（铁律 1）
- [ ] 4bit 件无 `CompressInt4ToInt8 ... out of range` 报错（铁律 2 / 网格权重）
- [ ] encoder 各段已带 int32 offset 修复（`model-strategies.md`）
- [ ] embedding 二进制 dim / dtype 与端侧约定一致
- [ ] `--info` 列出的项数与端侧加载预期一致
