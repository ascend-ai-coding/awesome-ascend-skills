---
name: external-cannbot-model-model-infer-migrator
description: 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。从 HF 链接或本地模型代码出发，按 cann-recipes-infer
  仓库规范适配到 ModelRunner 推理框架，输出可运行的标准模型目录和性能基线数据。触发场景：新模型适配到昇腾 NPU 推理框架、已有模型的部署基线采集、模型迁移和初始跑通验证。
original-name: model-infer-migrator
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# 模型框架适配与部署基线技能

从 HF 模型链接或本地代码出发，按 cann-recipes-infer 仓库规范完成框架适配和部署基线建立。产出符合仓库标准的模型目录结构（Runner + infer.py + infer.sh + YAML 配置），并采集标准化的性能基线数据。

本技能聚焦基础适配（eager 模式、PyTorch 标准算子），不涉及融合算子替换、KVCache 优化、图模式适配等性能优化。

---

## 覆盖场景

| 场景 | 输入状态 | 处理 |
|------|---------|------|
| A | 只有 HF 链接，无代码 | 下载 + 适配框架 + 生成标准文件 + 基线采集 |
| B | 有本地代码但跑不通 | 诊断修复（最多 5 轮）+ 基线采集 |
| C | 代码可运行但无标准化基线 | 直接基线采集 |

---

## 执行流程

### Step 1: 环境检测 + 场景判断

若 dispatch prompt 未提供环境信息，先确认 NPU 可用（`npu-smi info`）、torch 和 torch_npu 版本匹配。

根据工作目录状态判断场景：
- 无 infer.sh → 场景 A → Step 2
- 有 infer.sh → 执行一次：成功则场景 C → Step 4，失败则场景 B → Step 2
- NPU 环境不可用 → 报告环境问题，结束

> 无论 baseline/baseline_metadata.json 是否已存在，都按标准流程重新采集并覆盖。已有基线可能不符合标准采集方式（缺少 warmup、未使用 collect_baseline.py、字段不完整等）。

### Step 2: 代码准备（场景 A/B）

**场景 A**：按框架标准生成完整文件结构，详见「代码准备规范」。

**场景 B**：诊断 infer.sh 执行失败的原因，修复后重试，最多 5 轮。NPU 运行时错误参考 model-infer-runtime-debug skill。

### Step 3: 试运行

执行 `bash infer.sh`，根据结果分流：

- **跑通** → Step 4
- **OOM** → 估算显存需求，报告需多卡，结束
- **其他错误** → 回到 Step 2

### Step 4: 基线采集

使用 `dataset: "default"`（读 `dataset/default_prompt.json`）。

infer.py 内 `model_generate(warm_up=True)` 做预热，`model_generate()` 正式推理，框架 ModelRunner 自动输出 Prefill/Decode 耗时日志。其中 Prefill 耗时为首次 forward 的耗时，Decode 耗时为后续每步 forward 的平均值，warmup 和编译开销不计入。

> **注意**：基线采集前必须确认 infer.py 包含 warmup 调用（`model_generate(warm_up=True)`）。无 warmup 时首次推理包含编译开销，导致基线数据偏高。

推理完成后执行 `collect_baseline.py` 从日志解析耗时 + 采集环境信息，输出 `baseline/baseline_metadata.json`。

框架 `launch` 函数将日志输出到 `${RES_PATH}/log_0.log`（RES_PATH 由框架自动创建，格式为 `res/{date}/{model_name}/`）。

```bash
python3 scripts/collect_baseline.py \
    --log-file {RES_PATH}/log_0.log \
    --output {output_dir}/{model_name}/baseline/baseline_metadata.json \
    --yaml-file {yaml_path}
```

**多卡场景**：多卡推理生成多个 rank 日志（`log_0.log` ~ `log_N.log`），使用 rank 0 的日志。

**验证要求**：输出吐字正常（可读、不重复、长度合理），异常则回到 Step 2 诊断。baseline_metadata.json 中的 output_text 需记录完整输出。

---

## 代码准备规范（场景 A）

### 产出文件结构

输出目录默认 `cann-recipes-infer/models/{model_name}/`，可通过 dispatch prompt 配置其他路径。

```
{output_dir}/{model_name}/
├── config/
│   └── {model_name}_{config}.yaml
├── cann-recipes-infer/models/
│   ├── configuration_{model_name}.py
│   ├── model_setting.py
│   └── modeling_{model_name}.py
├── infer.py
├── infer.sh
├── runner_{model_name}.py
├── requirements.txt
├── README.md
└── baseline/
    └── baseline_metadata.json
```

### 通用适配流程

所有模型都走以下固定步骤：

1. **复制 HF 实现到 cann-recipes-infer/models/ 目录**：从 HF transformers 复制 modeling/configuration 相关文件，最小改动适配框架。**禁止 `from transformers` 导入模型类：modeling 和 configuration 代码必须 vendoring 到仓库内，不依赖 transformers 提供模型实现**：
   - modeling 构造函数对齐：`__init__(self, config, runner_settings=None, **kwargs)`（框架通过 `from_pretrained` 将 `runner_settings` 作为 kwarg 传入，modeling class 需要能接收）
   - 确保 `import torch_npu` 在模型或 runner 代码中执行，否则 NPU 设备不可用，推理会回退到 CPU
   - 删除训练专用代码（gradient_checkpointing、labels/loss 等）
   - 保留 PyTorch 标准算子，不替换 NPU 融合算子

2. **创建 Runner**（继承 ModelRunner）：通过路由表选择架构相近的参考模型，学习其框架接口用法（方法签名、input_dict 构造方式、调用链）。modeling 代码采用最小改动 vendoring 策略——直接复制 HF 的 modeling 实现到仓库，仅做框架接口适配（forward 签名、input_dict 对接），不重写推理逻辑、不从仓库参考模型复制 modeling 代码
3. **创建 infer.py、infer.sh、YAML 配置、model_setting.py**（参考 `references/templates.md`）
4. **创建 requirements.txt、README.md**

### Runner 接口要点

继承 `executor.model_runner.ModelRunner`，需要实现的方法：

| 方法 | 职责 |
|------|------|
| `__init__(self, runner_settings)` | 调用 `super().__init__(runner_settings)`，初始化模型特有参数 |
| `init_model()` | 加载模型和 config，调用 `super().init_model(ModelClass, ConfigClass)` |
| `model_input_prepare(self, input_dict)` | 从 input_dict 提取字段，构造模型 forward 所需的入参字典 |
| `model_output_process(self, model_inputs, outputs, input_dict)` | 从 outputs 提取 logits，更新 input_dict 状态（next token、kv_len、is_prefill 等） |
| `model_generate(self, prompts, warm_up=False)` | 顶层推理方法（见下方调用链说明） |

**调用链**：Runner 子类的 `model_generate` 是顶层入口，接收原始 prompts 字符串。内部需要：
1. tokenize prompts → 构造 input_dict（包含 input_ids、generate_ids、attention_mask、is_prefill 等）
2. 调用基类 `super().model_generate(input_dict, input_lens, warm_up)`
3. 基类内部循环调用 `model_input_prepare` → `model_inference` → `model_output_process`

**model_setting.py**：包含 `check_vars`（校验并行参数）和 `update_vars`（计算派生配置），参考仓库模型的同名文件。

> **重要**：实现 Runner 前必须完整阅读参考模型的 runner 代码（特别是 model_generate 和 model_input_prepare 的完整实现），理解 input_dict 的构造方式和字段含义。不同模型的 input_dict 字段差异较大，不能仅依据方法签名编写。

### 参考模型路由表

| 模型架构 | 参考模型 | 参考路径 |
|---------|---------|---------|
| 标准 LLM（MHA/GQA） | gpt-oss | `cann-recipes-infer/models/gpt_oss/` |
| MoE 架构 | qwen3-moe | `cann-recipes-infer/models/qwen3_moe/` |
| MLA 架构（DeepSeek 系列） | deepseek-r1 | `cann-recipes-infer/models/deepseek_r1/` |
| 长序列 | longcat-flash | `cann-recipes-infer/models/longcat-flash/` |
| 多模态/视频生成 | hunyuan-video | `cann-recipes-infer/models/hunyuan-video/` |

---

## 权重管理

通过 YAML 的 `model_path` 指定权重路径，支持任意本地路径。

```
权重已在本地？
  ├─ 是 → 填入 YAML model_path
  └─ 否 → 权重大小？
           ├─ < 5GB → 下载到本地（如 huggingface-cli、snapshot_download、modelscope 等），路径填入 YAML
           └─ >= 5GB → 提示用户自行下载并提供路径
```

> **并行部署场景**：基线采集前确认权重已正确配置（在线切分 `enable_online_split_weight: True`，或离线预切分目录结构完整）。权重处理由 parallel-impl skill 负责，migrator 仅做运行前检查。

---

## 基线采集

`scripts/collect_baseline.py` 从框架 ModelRunner 的推理日志中解析性能数据，采集环境信息，输出 `baseline/baseline_metadata.json`。

### baseline_metadata.json 格式

```json
{
  "timestamp": "2026-04-06T15:30:00",
  "environment": {
    "npu_model": "Ascend 910B4",
    "num_cards": 1,
    "cann_version": "8.5.0",
    "pytorch_version": "2.6.0",
    "torch_npu_version": "2.6.0.post1",
    "exe_mode": "eager"
  },
  "model_config": {
    "model_name": "Qwen3-0.6B",
    "model_source": "https://huggingface.co/Qwen/Qwen3-0.6B"
  },
  "performance": {
    "prefill_ms": 70.15,
    "decode_avg_ms": 71.5,
    "output_text": "..."
  }
}
```

> 格式从框架日志可解析的字段出发，不包含框架不输出的数据。

---

## 结束条件

全部条件满足后结束，不做额外优化或探索。

### 场景 A/B：单卡适配完成

1. 标准文件结构完整（modeling + config + runner + infer.py + infer.sh + YAML + model_setting）
2. `bash infer.sh` 在 NPU 上跑通
3. 输出吐字正常（可读、不重复、长度合理）
4. baseline_metadata.json 已生成

### 场景 A/B：单卡显存不足

1. 完整的单卡框架适配已完成：modeling（HF 实现本地化 + 框架接口适配）、Runner（继承 ModelRunner）、infer.py、infer.sh、YAML（world_size=1）、model_setting.py、configuration、requirements.txt
2. 代码可作为并行化改造的基础（parallel-impl 在此基础上替换并行层、加通信组、生成多卡 YAML）
3. 报告显存需求和预估卡数

### 场景 C：多卡基线采集

1. `bash infer.sh` 在多卡 NPU 上跑通
2. baseline_metadata.json 已生成
3. 输出吐字正常（可读、不重复、长度合理）

---

## 参考文档索引

| 文档 | 路径 |
|------|------|
| 框架适配模板（infer.py / infer.sh / YAML） | `references/templates.md` |
| 基线采集脚本 | `scripts/collect_baseline.py` |
| ModelRunner 基类 | `cann-recipes-infer/executor/model_runner.py` |
