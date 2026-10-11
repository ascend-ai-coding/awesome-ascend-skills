# 五阶段量化流程详解（PTQ → 导参 → 导 ONNX → 图改写 → 补维 + omg）

> 目标：把浮点 PyTorch 模型量化为 W4A16 的 `.omc`。两条主线（量化数值线 ①②、图结构线 ③④⑤）由 omg 在最后一步汇合。
> 核心认知：**导出的 ONNX 始终是 fp32 权重，量化由 omg 在转换时通过 `--compress_conf` 外挂施加**——单看 ONNX 看不到量化信息。

各阶段产物一览：

```
quant_pth/quant.pth                 # ① 标定状态
quant_onnx/quant_param_file         # ② omg 量化参数（--compress_conf）
quant_onnx/fake_quant_weight.pth    # ② 伪量化权重（供 ③）
quant_onnx/model_xxx.onnx           # ③ fp32 ONNX
quant_onnx/model_xxx_opt1.onnx      # ④ 重命名/拆算子/降维后
quant_onnx/model_xxx_opt_0528.onnx  # ⑤a 补维后（omg 实际输入）
quant_onnx/xxx.omc                  # ⑤b 端侧产物
```

---

## 阶段 ① PTQ 标定 — `s1_ptq.py`

**目的**：后训练量化。用真实数据跑一遍浮点模型，统计每个待量化层的激活 min/max，据此算量化 scale / zero-point；同时对权重做分组量化标定。

**步骤**：

1. 构建浮点模型，加载原始 checkpoint。
2. `generate_config_file(model, config.json)`：为每层生成默认（float）策略模板。
3. **改写为混合精度策略**（精度与可转换性的核心）：

   | 层类型 | 量化策略 | 位宽 | 原因 |
   | --- | --- | --- | --- |
   | 普通 Linear（ffn / mhsa / 投影） | `Quant_act_weight_eco` | W4（weight 4bit, group_size=128）+ A16（激活 16bit） | 主体计算，4bit 收益最大 |
   | 输出层（`output_layer` / `ctc_lo`，[512, 大词表]） | `Quant_aigc_ptq` | 8bit weight + 8bit 激活 | 大投影 4bit 不支持，退 8bit |
   | Conv1d / Conv2d | float 或 8bit | — | dopt 不支持 Conv 4bit；非 1×1 无法折叠为 MatMul |
   | Embedding | float / 8bit（端侧外置） | — | 端侧通常外置查表 |
   | LayerNorm / 激活 / 位置编码 | float | — | 不量化 |

4. `optimize_model(model, config.json)`：按配置把可量化层包成 QLinear 等量化模块。
5. 标定：`set_calibrate_state(True)` 后，用约 256 条真实校准数据前向，统计激活 min/max。
6. 保存 `quant.pth`（含标定好的 scale / zp 等量化状态 buffer）。

**为什么必须真实数据**：量化 scale = f(激活 min/max)，校准分布必须贴近部署时真实输入，否则 scale 偏移、量化误差放大、端侧精度崩。

---

## 阶段 ② 导量化参数 — `s2_genparams.py`

**目的**：把 `quant.pth` 里 PyTorch 形式的量化状态导成 omg 能直接读取的二进制压缩参数文件。

**步骤**：

1. 重新 `build` 模型 + `optimize_model`（套量化结构）。
2. `load_state_dict(quant.pth)`，恢复 ① 标定的 scale。
3. `set_quant_state(weight_state=True, input_state=True)` + `set_calibrate_state(False)`：进入量化推理态。
4. `generate_quant_params(model, out_dir, quant_param_2=True, embedding_separate=True)`：
   - **`quant_param_2=True`**：9030 必须按 UINT4 打包权重（平台相关开关，9020 才用 False/INT4）。
   - `embedding_separate=True`：embedding 参数单独处理（端侧外置）。

**产物**：

- `quant_param_file`：omg 转换时经 `--compress_conf` 喂入，按**节点名**把量化参数贴到对应 MatMul。
- `fake_quant_weight.pth`：伪量化（量化-反量化后）权重，供阶段 ③ 导 ONNX 使用——保证权重落在 4bit 网格上。

**为什么单独一步**：omg 不读 PyTorch 量化状态，量化信息必须落成独立文件，转换时与 ONNX 图按节点名对齐合并。

---

## 阶段 ③ 导出 ONNX — `s3_export.py`

**目的**：把模型导成标准 ONNX 图（omg 输入格式）。

**步骤**：

1. 构建模型 + `optimize_model`，`load(fake_quant_weight.pth)`。
2. **关闭量化态**：`set_quant_state(weight_state=False, input_state=False)`，导纯 fp32 图。
3. `torch.onnx.export(..., opset=13, do_constant_folding=True)`，指定 `input_names` / `output_names`。

**关键点**：导出的是 fp32 图，但权重已是 `fake_quant_weight` 的量化-反量化值，因此落在 4bit 网格上。**9030 上必须走这条 torch 导出路径**，不能导入预先固定的浮点 onnx——否则权重含任意浮点值，UINT4 打包越界。

---

## 阶段 ④ 图改写 — `s3_rewrite.py`

用 `auto_optimizer`（onnx surgeon）对 ONNX 做手术，使其符合 NPU 量化通路的硬性要求：

1. **带权重 MatMul 重命名**：把 `/.../MatMul` 改成点分名（如 `encoder.encoders.N.*`、`decoder.left_decoder.*`），与 `quant_param_file` 里的 key 对齐——omg 靠节点名匹配量化参数，名字对不上就贴不上量化信息。
2. **拆不支持算子**：如 decoder 的 `LogSoftmax` → `Softmax`(axis=2) + `Log`。
3. **量化 MatMul 降维（unfold，3D→2D）**：在每个带权重 MatMul 前插 `Squeeze`(axes=[0])、下游后插 `Unsqueeze`(axes=[0])，把激活从 `[1,T,C]` 压成 `[T,C]`——NPU 量化 MatMul kernel 走 2D。
4. **（encoder 专属）int32 语义保护**：offset / att_mask 用 int32 导出后，在此阶段加 float 输入、int 常量转 float、删冗余 `/Cast`，并保留 Gather 索引为 int32（删掉会把索引转 int64 的 Cast）。详见 `model-strategies.md`。

---

## 阶段 ⑤ 补维 + omg 转 omc

### 5a. 补维（buwei）— `s4_buwei.py`

**目的**：NPU cube 单元有 M 维对齐约束，量化 MatMul 的激活行数需凑到特定倍数，否则 kernel 不支持或精度异常。

**做法**：对每个带权重 MatMul（输出层除外）：

1. `infer_shape()` 拿激活 M 维。
2. `mydim = 2 if M==30 else 32`（补的行数，按实际模型调整）。
3. 前插 `Pad`：M 维头部补 `mydim` 行 0（`pads=[mydim,0,0,0]`，值 0.0）。
4. 后插 `Slice`：从第 `mydim` 行切到末尾，切回原始有效长度。

`Pad → MatMul → Slice` 数值等价（补 0 行算完切掉），同时满足硬件对齐。

### 5b. omg 转换（容器内执行）

```bash
# 容器内先激活 AscendC 工具链（9030 必须，否则 W4 MatMul 退回预编译 kernel）
source <ddk>/tools/tools_ascendc/install.sh
source <ddk>/tools/tools_ascendc/set_ascendc_env.sh
which ccec   # 必须非空

omg --framework=5 \
    --model=model_xxx_opt_0528.onnx \
    --output=final_quant_model_Quant \
    --compress_conf=quant_param_file \
    --input_shape="<按模型>" \
    --platform=kirin9030 \
    --target=omc
```

- `--framework=5`：ONNX
- `--compress_conf`：阶段 ② 的量化参数文件（**图与量化数值在此合并**）
- `--platform=kirin9030 --target=omc`：目标平台与产物类型

**omg 内部**：解析 ONNX → IR 图（pre-check 算子支持性）→ 按节点名匹配 `quant_param_file`，在量化 MatMul 前后插 `QuantizeV2 / DequantizeV2`，权重转 UINT4 → 算子融合、内存分配、生成 NPU kernel → 序列化为 omc。

> 容器 / glibc / 工具链细节见 `platform-9030.md`。
