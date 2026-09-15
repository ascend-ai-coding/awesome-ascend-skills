# encoder / decoder / punc 量化策略与专属坑

三类 ASR 模型走同构的五阶段流程（见 `quantization-pipeline.md`），但混合精度策略与图改写细节不同。本文给出各自的策略与必踩的坑。

---

## Encoder

**结构**：Conformer（self-attn + ffn + conv_module）。

**策略**：

- 4bit（`Quant_act_weight_eco` + UINT4）：各 self_attn / ffn 的 Linear。
- 8bit（`Quant_aigc_ptq`）：`ctc_lo`、`embed.conv`。
- float：`depthwise_conv`（非 1×1，无法折叠为 MatMul，占用很小）、LayerNorm、位置编码。

**可选压缩——pwlinear**：把 conv_module 的 1×1 pointwise Conv1d（`pointwise_conv1/2`）折叠为数学等价的 `nn.Linear`，使其也能走 W4 MatMul → 4bit。加载时把 conv 权重 `squeeze` 成 Linear 权重。效果：fp32 pointwise（约 25MB）→ 4bit，omc 显著变小。`depthwise_conv` 仍保持 Conv1d。

### Encoder 精度修复（核心，所有 encoder 版本必带）

offset / att_mask 若按 float 导出会破坏其整数索引语义，端侧出现识别异常 / 插字。修复：**用 int32 导出 offset / att_mask，在图改写阶段统一转 float**。

落地三处：

1. `s3_export.py`：`offset` / `att_mask` 设 `torch.int32`。
2. 模型定义（position_encoding）：`arange` / `max` 等常量的 dtype 跟随 `offset.dtype`，否则 int32 导出报 `Inputs of Max must have same dtype. Got Long and Int`（int64 常量 vs int32 输入）。
3. `s3_rewrite.py`：
   - 加 `new_offset` / `new_att_mask` 的 float 输入；
   - int 常量转 float、删冗余 `/Cast`、删 att_mask 上的 Cast；
   - **保留 Gather 索引为 int32**（删掉那个会把索引转成 int64 的 Cast），否则 omg 报 `GatherV2D Input[1] DataType INT64 is wrong`。

---

## Decoder

**结构**：left_decoder（自回归）+ right_decoder + output_layer + embed。

**策略**：

- 4bit（`Quant_act_weight_eco` + UINT4）：仅 `left_decoder` 分支的非输出 Linear。
- 8bit：`output_layer`（大词表投影）。
- float：`right_decoder`、`embed`。

**图改写专属**：NPU 不支持 `LogSoftmax`，在 `s3_rewrite.py` 拆成 `Softmax`(axis=2) + `Log`。

### Decoder 关键坑（9030）

**不能**用导入固定浮点 onnx 的旧路（9020 习惯）：9030 上 UINT4 会越界、INT4 又不支持，两条死路。

**正解**：走 `s3_export.py`——加载 dopt 的 `fake_quant_weight.pth` 后用 torch 导出，权重落 4bit 网格，UINT4 打包才成功（即铁律 2，见 `platform-9030.md`）。

---

## 标点 punc

**结构**：标点 Transformer（token_embed → encoder/decoder → 标点分类）。

**策略**：

- 4bit（`Quant_act_weight_eco` + UINT4）：Transformer 各 Linear。
- 8bit：`decoder_punct`（输出节点）。
- float：`mhsa.linear_pos`（位置编码常量输入，量化融合不支持）、`Embedding`。

**专属处理**：

1. **pwlinear**：同 encoder，把 conv_module 1×1 pointwise Conv1d 折叠为 Linear 进 W4A16。
2. **两遍标定**：第一遍标定后自动检测激活 scale 退化（低至 ~1e-44）的 FFN 层，强制 float（规避 NPU fp16 溢出），第二遍再标定。
3. **单输入切图**：端侧引擎只喂 `token_ids[1,30]`，把 embedding 查表（Gather）与 src_mask 生成都折进图，图输入只保留 `token_ids`。omg `--input_shape` 形如 `token_embed:1,30,256;src_mask:1,1,30`（按切图方案调整）。

### punc 注意

标点的校准数据分布要贴近**口语**（部署真实输入），用正式书面文本标定会导致 scale 偏移、端侧标点不准。标点 omc 历史上有端侧精度问题，转换后仍需端侧验证。

---

## 配置 json 片段示例

4bit（`Quant_act_weight_eco`，权重 4bit / group128，激活 16bit）：

```json
"encoder.encoders.2.self_attn.linear_v": {
    "type": "<class 'torch.nn.modules.linear.Linear'>",
    "quant_strategy": "Quant_act_weight_eco",
    "weight": { "bit": 4, "group_size": 128 },
    "input":  { "bit": 16 },
    "output": { "bit": 16, "per_channel": true, "input_algo": "min_max", "unsigned_quant": false }
}
```

8bit（`Quant_aigc_ptq`，Conv / 大投影）：

```json
"encoder.embed.conv.0": {
    "type": "<class 'torch.nn.modules.conv.Conv2d'>",
    "quant_strategy": "Quant_aigc_ptq",
    "weight": { "bit": 8 },
    "input":  { "bit": 8, "input_algo": "min_max", "unsigned_quant": false }
}
```

> 注意：Conv 算子暂不支持 4bit，建议用 8-8bit。生成的空配置 json 需手动改成上述混合精度策略后再次执行标定。
