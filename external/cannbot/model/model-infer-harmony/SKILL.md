---
name: external-cannbot-model-model-infer-harmony
description: 麒麟 NPU 端侧（Kirin9030 / HarmonyOS）ASR 模型 4bit W4A16 量化、omg 离线模型转换与 CANNPAK
  打包全流程技能。覆盖 dopt PTQ 标定 → 导量化参数 → 导 ONNX → 图改写 → MatMul 补维 → 容器内 omg 转 omc → 多 omc
  打包成端侧单 bin，并含量化精度修复与转换报错调试。触发：当用户在 Kirin9030 NPU 上量化部署 ASR 的 encoder / decoder /
  标点（punc）模型、把浮点 PyTorch 模型转成端侧 omc、遇到 UINT4 越界 / MatMul don't support / GatherV2D
  INT64 / 容器 glibc 等 omg 转换报错或端侧精度问题、或需要把多个 omc 打包成端侧单文件时使用。不覆盖端侧推理代码实现，也不覆盖上游浮点模型训练。
original-name: model-infer-harmony
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# 麒麟端侧 ASR 4bit 量化与模型转换（Kirin9030）

把浮点 PyTorch ASR 模型（encoder / decoder / 标点 punc）量化为 **W4A16**（4bit 权重 + 16bit 激活）的 `.omc` 离线模型，并在 Kirin9030（HarmonyOS 端侧）上部署。流程基于 dopt PTQ 量化库 + auto_optimizer 图改写 + omg 离线转换，最终可把多个 omc 打包成端侧可加载的单个 bin。

官方权威步骤参考华为 [CANNKit 插件量化文档](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-plug-in-quantification)；本 skill 在其基础上沉淀 9030 平台特有约束与排错经验。

---

## 适用范围

- Kirin9030 上 ASR encoder / decoder / 标点（punc）的 4bit W4A16 量化
- dopt PTQ 量化 → 量化参数导出 → ONNX 图改写 → MatMul 补维 → omg 转 omc
- encoder 拆分为多段 omc（端侧分段加载）
- 多个 omc + embedding 二进制打包成 CANNPAK 单 bin
- 常见 omg / 量化报错的归因与修复

**不做**：

- 不实现端侧 C++/ArkTS 推理引擎代码
- 不训练或蒸馏上游浮点模型
- 不覆盖 Kirin9020 链路差异（9020 用 INT4，约束与本 skill 相反，不要混用）

---

## 9030 两条铁律（贯穿全流程，违反必报错）

这是 9030 与其它平台最关键的区别，所有阶段都要遵守：

1. **omg 必须在容器内跑**。宿主 glibc 2.17 跑不了 9030 omg，需 glibc ≥ 2.35 的容器（AscendC 工具链）。容器内必须先激活 AscendC 工具链（`source install.sh` + `source set_ascendc_env.sh`），确认 `which ccec` 非空——否则 W4 MatMul 退回预编译 kernel，报 `type MatMul don't support`。

2. **9030 的 W4 内核只支持 UINT4（无符号 [0,15]），不支持 INT4**。因此：
   - 量化必须 `quant_param_2=True`（按 UINT4 打包权重）；
   - 权重必须落在 dopt 4bit 网格上（用 torch 加载 `fake_quant_weight.pth` 后 `torch.onnx.export` 导出），否则原始浮点权重含负值，UINT4 打包报 `CompressInt4ToInt8 Value -1 out of range [0,15]`。

> 平台铁律的完整背景、容器命令、网格权重处理见 `references/platform-9030.md`。

---

## 五阶段量化流程总览

整个流程分两条主线，最后由 omg 汇合（**导出的 ONNX 始终是 fp32 权重，真正的量化由 omg 在转换时通过 `--compress_conf` 施加**）：

```
                          ┌──────── 量化数值线 ────────┐
浮点模型 ─①PTQ标定─► quant.pth ─②导量化参数─► quant_param_file ──┐
                                                               │
                          ┌──────── 图结构线 ──────────┐        │
浮点模型 ─③导ONNX(fp32)─► model.onnx ─④图改写─► opt1.onnx ─⑤补维─► opt_0528.onnx
                                                               │
                                    ⑤ omg(图 + quant_param_file) ◄─┘  →  xxx.omc
```

| 阶段 | 脚本（示例命名） | 输入 | 产物 |
| --- | --- | --- | --- |
| ① PTQ 标定 | `s1_ptq.py` | 浮点权重 + 真实校准数据（约 256 条） | `quant_pth/quant.pth` |
| ② 导量化参数 | `s2_genparams.py` | `quant.pth`，**`quant_param_2=True`** | `quant_param_file` + `fake_quant_weight.pth` |
| ③ 导出 ONNX | `s3_export.py` | `fake_quant_weight.pth`（关闭量化态导 fp32） | `model_xxx.onnx` |
| ④ 图改写 | `s3_rewrite.py` | `model_xxx.onnx` | `model_xxx_opt1.onnx` |
| ⑤ 补维 + omg | `s4_buwei.py` + `omg`（容器内） | `opt1.onnx` + `quant_param_file` | `opt_0528.onnx` → `xxx.omc` |

> 每阶段的目的、关键参数、为什么这样做，见 `references/quantization-pipeline.md`。

**通用 4bit 配方**：普通 Linear（ffn / mhsa / 投影）走 `Quant_act_weight_eco`（weight 4bit / group_size=128，激活 16bit）+ `quant_param_2=True`；大投影输出层（`output_layer` / `ctc_lo`）退 8bit（`Quant_aigc_ptq`）；Conv / Embedding / LayerNorm / 位置编码保持 float 或 8bit。

---

## 参考文档使用矩阵

| 场景 | 读取 | 用途 |
| --- | --- | --- |
| 走完五阶段量化 | `references/quantization-pipeline.md` | 每阶段目的、参数、产物 |
| 配 9030 环境 / 容器 omg / 拆分网格权重 | `references/platform-9030.md` | 两条铁律的落地命令与原理 |
| 量化 encoder / decoder / punc | `references/model-strategies.md` | 三类模型的策略差异与专属坑 |
| encoder 拆分 + 打包单 bin | `references/split-and-packing.md` | 拆分流水线 + CANNPAK 格式 |
| 任何报错 | `references/troubleshooting.md` | 现象 → 阶段 → 根因 → 对策 |

---

## 三类模型差异速查

| 模型 | 4bit 范围 | 8bit / float | 关键专属点 |
| --- | --- | --- | --- |
| encoder | conformer 各 Linear；可选把 1×1 pointwise Conv1d 折叠为 Linear（pwlinear）进一步压缩 | ctc_lo / embed.conv 8bit；depthwise_conv float | offset / att_mask 用 **int32 导出**，在图改写阶段统一转 float（精度修复，必带） |
| decoder | left_decoder 各 Linear | output_layer 8bit；right_decoder / embed float | 必须走 `s3_export`（torch 导出落网格），**不能**导入固定浮点 onnx |
| 标点 punc | Transformer 各 Linear | decoder_punct 8bit；mhsa.linear_pos / Embedding float | 单输入切图（端侧只喂 token_ids）；校准分布要贴近口语 |

> 详细策略、配置 json 片段、各自踩过的坑见 `references/model-strategies.md`。

---

## 排错路由（先按现象定位，再读 troubleshooting）

| 现象 | 大概率根因 | 先看 |
| --- | --- | --- |
| `CompressInt4ToInt8 Value -1 out of range [0,15]` | 权重非网格 + UINT4 | 铁律 2 / `platform-9030.md` 网格权重 |
| `type MatMul don't support!` | INT4（应 UINT4）或容器工具链未激活 | 铁律 1+2 |
| `GatherV2D Input[1] INT64 is wrong` | Gather 索引被转成 int64 | `model-strategies.md` encoder int32 修复 |
| `Inputs of Max must have same dtype. Got Long and Int` | int32 offset 导出时常量 dtype 不一致 | `model-strategies.md` encoder |
| omg 宿主直接崩 / glibc 报错 | 在宿主而非容器跑 omg | 铁律 1 |

完整速查表（含端侧识别异常、scale 退化等）见 `references/troubleshooting.md`。

---

## 端到端最短路径

```
① encoder：量化（必要时 pwlinear 压缩 / 拆分 10 段）→ 容器 omg → encoder omc(若干段)
② decoder：量化（torch 导出落网格）→ 容器 omg → decoder omc
③ punc：量化（单输入切图）→ 合规机器 / 容器 omg → punc omc
④ embedding：导出 token embedding 量化二进制
⑤ 打包：把 ①~④ 的产物按 CANNPAK 格式合成单 bin，端侧加载
```

---

## 参考文档索引

| 主题 | 路径 |
| --- | --- |
| 五阶段量化流程详解（PTQ / 导参 / 导 ONNX / 图改写 / 补维 / omg） | `references/quantization-pipeline.md` |
| 9030 平台铁律：容器 omg、UINT4、拆分网格权重 | `references/platform-9030.md` |
| encoder / decoder / punc 量化策略与专属坑 | `references/model-strategies.md` |
| encoder 拆分流水线 + CANNPAK 打包格式 | `references/split-and-packing.md` |
| 报错现象 → 根因 → 对策速查表 | `references/troubleshooting.md` |
