# README 共享模板

migrator 阶段产物的 README 模板，**框架部署与独立部署共用**。两种模式的差异点在 "Agent 优化说明" 的部署模式行二选一，按当前部署模式填写即可。

---

## 模板正文

````markdown
<!--
本模板是 README 起始骨架，agent 写入模型 README.md 时按当前模型实际情况调整：
- `{...}` 占位符必须替换为实际值
- 通用流程段（如"环境准备"）默认通用，如有模型特定差异请扩展
- 标 "_由 optimize skill 在优化总结阶段补全。_" 的占位段，migrator 阶段保留占位符不替换
- 描述模型 / 优化点等内容用完整中文句子，禁止堆砌配置参数（如 `EP=8`、`attn_tp=1`）或接口名缩写；详细规范见 optimization_report_template.md
- 避免环境痕迹（IP / 卡号 / 端口 / 临时目录 / shell 变量）、版本控制痕迹（commit hash / 分支名）、过程性标记（emoji 徽章 / "阶段 N" / "已完成 / 待补全"）
- 术语沿用既有命名：优先使用仓内代码函数名、仓库设计文档和注册模型既有命名；描述性词汇优先用中文，行业缩写和技术名保留英文（如 MoE / MLA / Decode / KVP）；禁止自造同义新词
-->

# {ModelName}模型在NPU{上推理 / 实现低时延推理 / 实现高性能推理}

## 概述

{1-2 句模型用途与架构亮点，例如："基于 MLA + Sparse MoE 架构的大语言模型，参数量 80B"}

- HuggingFace: {hf_repo_link}
- 架构: {Dense / GQA / MoE / MLA / Diffusion / 多模态}
- 参数量: {N B}

## 支持的产品型号

{按模型实际支持列出，删除不支持的项}

<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## Agent 优化说明

本样例由 NPU 推理优化 Agent Skills 完成迁移与优化。

- **部署模式**：{框架部署，接入 cann-recipes-infer/executor/core/ / 独立部署，自管 Runner}
- **优化点参考**：本样例已落地的主要优化点（详细方案见 [agentic/optimization_report.md](agentic/optimization_report.md)）：
  - {优化点 1：1 行自然语言描述，例如 "混合并行部署：MoE 模块按 Expert Parallel (EP) 切分到 8 卡，Embedding 与 LM Head 采用 Tensor Parallel (TP) 切分到 8 卡"}
  - {优化点 2：例如 "KV Cache 采用 Paged Attention 方案管理，提升长序列与高并发场景下的显存利用率"}
  - {根据实际落地优化补充更多条目}

agent 各优化阶段过程归档在 [agentic/](agentic/) 目录：

- [agentic/optimization_report.md](agentic/optimization_report.md) — 优化报告
- [agentic/progress.md](agentic/progress.md) / [agentic/progress_history.md](agentic/progress_history.md) — agent 各优化阶段过程归档
- [agentic/baseline/baseline_metadata.json](agentic/baseline/baseline_metadata.json) — migrator 阶段采集的性能基线

## 环境准备

<!-- 默认 CANN + torch_npu + pip 流程；如模型需要 vendoring 第三方库 / 编译自定义算子，在末尾追加步骤 -->

| 项 | 版本 |
|----|----|
| CANN | 9.0.0 |
| torch_npu | 2.8.0 |
| Python | >= 3.10 |

1. 安装 CANN 软件包：参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/)。
2. 安装 Ascend Extension for PyTorch（torch_npu）：参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/)。
3. 安装项目依赖：

   ```bash
   cd cann-recipes-infer
   pip3 install -r ./models/{model_name}/requirements.txt
   ```

4. 配置 `cann-recipes-infer/executor/scripts/set_env.sh` 的 `cann_path`；多节点部署时按 rank 顺序填 `IPs`，单节点忽略。
5. {若有 HCCL / 多机额外环境变量，在此说明}

## 权重准备

从 HuggingFace 下载 {ModelName} 原始权重到本地路径（例如 `/data/models/{ModelName}`），并将 yaml 内 `model_config.model_path` 改为该路径：

- HuggingFace 仓库：{hf_repo_link}

### 权重转换（仅量化 / 格式转换模型需要）

<!-- 模型权重无需转换时整段删除 -->

{说明转换场景，例如 "FP8 原权重需转换为 BF16" 或 "需进行 W8A8 量化"}：

```bash
bash utils/weight_convert.sh \
    --input_fp8_hf_path /data/models/{ModelName} \
    --output_hf_path /data/models/{ModelName}-{quant_mode} \
    --quant_mode {w8a8 / w4a16 / w8a8c8}
```

转换后将 yaml 内 `model_config.model_path` 改为转换后的路径。

## 推理执行

1. 配置 yaml：`model_config.model_path` 改为本地权重路径（必改）；其他参数按需调整，通用字段含义见 [YAML 参数描述](../../docs/common/inference_config_guide.md)。

   <!-- 本模型若在 model_config.custom_params 下有特有参数（如多流、控核、benchmark 等开关），按以下表格列出；无特有参数时整段删除 -->

   除框架统一配置外，本模型在 `model_config.custom_params` 下额外支持以下特有参数：

   | 参数名 | 类型 | 默认值 | 含义 |
   | --- | --- | --- | --- |
   | `{param_name}` | `int / bool / str` | `{default}` | {一句话说明用途和典型取值} |

   配置文件清单：
   - `config/{model_name}_1tp.yaml` — 单卡 eager
   - `config/{model_name}_{N}tp.yaml` — N 卡 TP
   - {按模型实际配置补充，例如 N 卡 EP、量化模式 yaml}

2. 准备推理 prompt：
   - **内置 prompt**：使用 `cann-recipes-infer/dataset/default_prompt.json`，无需额外准备
   - **自定义 / 长序列**：替换 `cann-recipes-infer/dataset/default_prompt.json` 内容，或在 yaml 内将 `data_config.dataset` 改为对应 benchmark（如 `"LongBench"` / `"InfiniteBench"`）

   > 若长序列场景出现 OOM，参考 [AGENTS.md 注意事项](../../AGENTS.md) 的 OOM 缓解顺序调整 batch_size / kvp_size / 量化模式。

3. 执行推理：

   <!-- 框架部署：使用上游统一脚本；独立部署：删除以下统一脚本调用，改为本模型自管的推理入口（如 `python infer.py --yaml ...`） -->

   ```bash
   bash cann-recipes-infer/executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml
   ```

   切换其他 yaml 替换 `--yaml` 参数；PD 分离 online 模式参考 `bash cann-recipes-infer/executor/scripts/infer.sh -h`。

## 性能基线

migrator 阶段采集的基线指标（详见 `agentic/baseline/baseline_metadata.json`）。

| 量化模式 | Global Batch Size | Seq Length (in/out) | 卡数 | TPOT (ms) | 吞吐 (tokens/s) |
|------------|-------------------|---------------------------|-------|-----------|------------------------|
| {BF16 / W8A8 / ...} | {N} | {例如 4096/1024} | {例如 8×A3} | {N} | {N} |

## 附录（可选）

<!-- 仅在确有必要的复用知识时添加（如该模型特有的 OOM 处理、典型错误排查、特定 yaml 参数说明）。无必要内容时整段删除，不要为了完整性而填充泛用流程。 -->
````
