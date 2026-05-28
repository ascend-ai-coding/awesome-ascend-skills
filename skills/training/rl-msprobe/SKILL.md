---
name: rl-msprobe
description: 自动化 verl msprobe 精度数据采集；开始前检查/预装 msprobe（pip install mindstudio-probe）。自动识别三种模式：(1) 训练采集——global_profiler + precision_debugger stages；(2) 推理采集——vLLM/SGLang rollout dump；(3) 训推一致性——engine patch + PROMPTS_ONLY。触发词：verl dump、msprobe、mindstudio-probe、训练采集、推理采集、训推一致性、PrecisionDebugger。
keywords:
    - verl
    - msprobe
    - dump
    - precision_debugger
    - 训推一致性
    - 训练采集
    - 推理采集
    - vllm
    - sglang
    - fsdp
    - megatron
    - enforce_eager
    - global_profiler
    - mindstudio-probe
---

# verl msprobe 精度数据采集

自动识别并执行三类 msprobe 采集。**最先检查 msprobe 环境**，再判定采集模式。

## 前置：msprobe 环境（必做）

三种模式（training / inference / consistency）均依赖 **msprobe**。在改脚本、patch 或启动训练**之前**执行：

```bash
bash scripts/ensure-msprobe-env.sh
```

| 项 | 说明 |
|----|------|
| PyPI 包名 | `mindstudio-probe` |
| Python import | `from msprobe.pytorch import PrecisionDebugger` |
| 环境 | 与 **verl 训练/推理同一 Python**（容器内请在容器里执行） |
| 自动安装 | 脚本默认执行 `pip install mindstudio-probe` |
| 仅检测 | `bash scripts/ensure-msprobe-env.sh --no-install` |

**安装失败时**：停止后续 dump 配置，提示用户自行安装并校验：

```bash
pip install mindstudio-probe
python3 -c "from msprobe.pytorch import PrecisionDebugger; print('OK')"
```

未通过环境检查前，不要进入路径 A/B/C。

默认 verl 源码：`VERL_ROOT=/home/zhouyang/skill/verl`（**以用户实际训练/容器内使用的 verl 树为准**，可能与默认路径不同）

| 模式 | 标识 | 改源码 | 后端差异 |
|------|------|--------|----------|
| **training** 训练采集 | `global_profiler` + `stages` 三阶段 | 否 | FSDP/Megatron 共用 |
| **inference** 推理 rollout | `dump_config_path` / `msprobe_dump_config` | SGLang<0.5.11 需改 SGLang | **vLLM / SGLang** |
| **consistency** 训推一致性 | 训练 engine patch + 推理 dump + PROMPTS_ONLY | 是（verl engine + 可选 vLLM dispatch） | 训练 FSDP/Megatron + 推理 vLLM/SGLang |

**共用强约束（推理相关）**：`actor_rollout_ref.rollout.enforce_eager=True`（不开图模式）。

**共用 config.json 规范**：所有 msprobe `config.json` **优先使用 `"level": "mix"`**（L0 模块级 + L1 API 级混合采集）。用户未指定 level 时默认 `mix`；仅在明确只要 L0/L1 单粒度时才改。

---

## 第 0 步：识别采集模式

```bash
bash scripts/detect-dump-mode.sh "$TRAINING_SCRIPT" "$USER_HINT"
# 输出: training | inference | consistency | unknown
```

| 信号 | 模式 |
|------|------|
| 「推理采集」「rollout dump」「generate dump」 | inference |
| 「训推一致」「prefill 对齐」 | consistency |
| 「训练采集」「global_profiler」「precision_debugger」 | training |
| 提及 `actor_compute_log_prob` / `ref_compute_log_prob` / `actor_update` 任一 stage | training |
| 脚本含 `global_profiler.tool=precision_debugger` 或 `precision_debugger.stages` | training |
| 脚本含 `dump_config_path` / `msprobe_dump_config` | inference |
| 脚本含 `DUMP_ON` + `PROMPTS_ONLY` | consistency |

无法判定时询问用户选 **A 训练 / B 推理 / C 训推一致**。

训练采集（路径 A）按以下 **三个 stage** 分别获取 dump 数据（通过 `global_profiler.global_tool_config.precision_debugger.stages` 选择子集或全部）：

| stage | 角色 | 采集内容 |
|-------|------|----------|
| `actor_compute_log_prob` | actor | actor 前向 log prob |
| `ref_compute_log_prob` | ref | ref 前向 log prob |
| `actor_update` | actor | actor 策略更新（前向+反向） |

默认三阶段全采；用户可只指定其中一部分，但须同步开启对应 role 的 `profiler.enable`。

---

## 路径 A：训练采集（training）

> 详细说明：[references/training-profiler-dump.md](references/training-profiler-dump.md)

训练侧数据按 **stage** 粒度采集，核心三阶段为 `actor_compute_log_prob`、`ref_compute_log_prob`、`actor_update`。每个 stage 在 `{save_path}/step_{global_step}/{stage}/step0/rank0/dump.json` 独立落盘。

### 配置要点

```bash
global_profiler.tool=precision_debugger \
global_profiler.steps='[1,2]' \
global_profiler.save_path=outputs/profile \
+global_profiler.global_tool_config.precision_debugger.config_path=/path/to/config.json \
+global_profiler.global_tool_config.precision_debugger.stages='[actor_compute_log_prob,ref_compute_log_prob,actor_update]' \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.ref.profiler.enable=True
```

- `stages` 决定采哪几个阶段；省略时 verl 默认不限制（建议显式列出）
- 只采 `actor_compute_log_prob` + `actor_update` 时，ref 的 `profiler.enable` 可关
- 不改 verl 源码；**禁止**在 config.json 设 `dump_path`
- `config.json` 中 **`"level": "mix"`**（默认推荐，见上方共用规范）
- 校验：`python3 scripts/check-training-profiler.py run_profiler_*.sh`

---

## 路径 B：推理采集（inference）

> vLLM：[references/inference-vllm-dump.md](references/inference-vllm-dump.md)  
> SGLang：[references/inference-sglang-dump.md](references/inference-sglang-dump.md)

### B0. 识别 rollout 后端

```bash
bash scripts/detect-rollout-backend.sh "$TRAINING_SCRIPT"
# vllm | sglang | unknown
```

### B1. 公共前置

```bash
export TORCHDYNAMO_DISABLE=1   # 推荐
actor_rollout_ref.rollout.enforce_eager=True   # 必须
```

### B2. vLLM（两步即可，不改 verl 源码）

**① config.json**（须含 `dump_path`）：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/generate",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [], "list": [], "tensor_list": [],
    "data_mode": ["all"], "summary_mode": "statistics"
  }
}
```

模板：`assets/config_generate.json.example`

**② 脚本参数**：

```bash
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.enforce_eager=True \
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path="/path/to/generate_config.json"
```

### B3. SGLang（按 msprobe 官方方式）

| SGLang 版本 | 做法 |
|-------------|------|
| **>= 0.5.11** | verl 传参：`++actor_rollout_ref.rollout.engine_kwargs.sglang.msprobe_dump_config=/path/to/config.json` |
| **< 0.5.11** | 按 [sglang_eager_dump_instruct.md](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md) patch SGLang `ModelRunner` |

verl 侧仍须：

```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.enforce_eager=True
```

PD 分离：`export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`

### B4. 校验与产物

```bash
python3 scripts/check-inference-dump.py run_infer_dump_*.sh --backend vllm   # 或 sglang
```

期望：`{dump_path}/step_N/rank_M/dump.json`（vLLM 可能含 `{pid}/` 子目录）

---

## 路径 C：训推一致性（consistency）

= **路径 B 推理侧** + **训练 engine worker patch** + 对齐约束

> **主文档**：[references/consistency-engine-worker.md](references/consistency-engine-worker.md)（legacy megatron → engine worker 迁移）  
> **辅助 patch**：[references/consistency-auxiliary-patches.md](references/consistency-auxiliary-patches.md)  
> **对齐约束**：[references/shared-prerequisites.md](references/shared-prerequisites.md)

### C1. 推理侧

按 B2/B3 配置 rollout dump（`enforce_eager=True` 必须）。

**并行对齐（必检）**：
- **FSDP**：`actor_rollout_ref.rollout.tensor_model_parallel_size` = rollout 占用卡数（如 4 卡 → `gen_tp=4`）；consistency 阶段建议 `data_parallel_size=1`
- **Megatron**：actor/ref 的 TP/PP/EP 须与 rollout 的 `tensor_model_parallel_size` / `data_parallel_size` / `expert_parallel_size` 逐项一致

详见 [references/shared-prerequisites.md](references/shared-prerequisites.md#训推并行切分对齐consistency-必检)。

vLLM 训推一致推荐 vLLM-Ascend `dispatch_logger` patch（`dispatch_log.jsonl` 关联 prefill step），见 consistency-engine-worker 文档。

### C2. 训练侧

按 FSDP/Megatron patch engine worker + 辅助文件：

- [references/fsdp-engine-patch.md](references/fsdp-engine-patch.md) / [references/megatron-engine-patch.md](references/megatron-engine-patch.md)
- [references/consistency-auxiliary-patches.md](references/consistency-auxiliary-patches.md)

```bash
export DUMP_ON=1 PROMPTS_ONLY=1 TORCHDYNAMO_DISABLE=1 DUMP_PHASE=log_prob
export MSPROBE_ACTOR_CONFIG=/path/to/config_actor.json
bash scripts/detect-train-backend.sh "$TRAINING_SCRIPT"
# engine worker 钩子（transformer_impl.py）
python3 scripts/check-engine-patch.py --verl-root "$VERL_ROOT" --backend "$TRAIN_BACKEND"
# 训推一致全量（PROMPTS_ONLY、request_id、可选 vLLM dispatch）
python3 scripts/check-consistency-patch.py \
  --verl-root "$VERL_ROOT" --backend "$TRAIN_BACKEND" \
  --vllm-ascend-root /path/to/vllm-ascend
```

**注意**：patch 须打在**实际运行**的 verl 源码树（可能与 `VERL_ROOT` 不同，如 `/verl`）。

### C3. 对齐约束与产物

- **Batch**：`train/mini/micro=1`，`n=DP`（`DP=world_size/SP`）；**优先** `actor_rollout_ref.actor.ulysses_sequence_parallel_size=1` 与 `actor_rollout_ref.ref.ulysses_sequence_parallel_size=1`（即使原脚本 `sp_size>1` 也须在采集脚本覆盖）
- **并行**：FSDP 推理 `tensor_model_parallel_size` = rollout 卡数；Megatron 训练 TP/PP/EP 与 rollout 逐项一致

见 [references/shared-prerequisites.md](references/shared-prerequisites.md) 与 [references/dump-output-correlation.md](references/dump-output-correlation.md)。

---

## 三模式速查

| 项目 | training | inference | consistency |
|------|----------|-----------|-------------|
| config `dump_path` | 禁止 | **必须** | **必须**（generate） |
| `enforce_eager` | — | **必须 True** | **必须 True** |
| vLLM 入口 | — | `dump_config_path` | 同 inference |
| SGLang 入口 | — | `msprobe_dump_config` 或 patch | 同 inference |
| 改 verl 源码 | 否 | 否（SGLang 旧版改 SGLang） | 是（engine） |
| 训推并行对齐 | — | — | **FSDP：推理 TP=卡数；Megatron：训练/推理切分一致** |
| step / stage 控制 | `global_profiler.steps` + `stages` 三阶段 | config `step` / msprobe step | 两侧分别控制 |
| 训练 stage | `actor_compute_log_prob` / `ref_compute_log_prob` / `actor_update` | — | engine micro_batch |

## 禁止事项

- 禁止在未确认 `msprobe` 可 import 时继续配置或启动 dump
- 禁止 `enforce_eager=False` 下做推理 dump
- 禁止未判模式就 patch engine
- 禁止直接改用户原始训练脚本
- vLLM 与 SGLang 采集方式**不可混用**
- consistency 禁止 **FSDP 下 `gen_tp < rollout 卡数`**（推理 dump rank 与训练 rank 无法按卡对齐）
- consistency 禁止 **Megatron 训练切分与 rollout 切分不一致**（TP/PP/EP 须逐项相同）

## 参考

- 普通训练：[references/training-profiler-dump.md](references/training-profiler-dump.md)（三 stage 采集）
- vLLM 推理：[references/inference-vllm-dump.md](references/inference-vllm-dump.md)
- SGLang 推理：[references/inference-sglang-dump.md](references/inference-sglang-dump.md)
- 训推 engine：[references/consistency-engine-worker.md](references/consistency-engine-worker.md) / [references/fsdp-engine-patch.md](references/fsdp-engine-patch.md) / [references/megatron-engine-patch.md](references/megatron-engine-patch.md)
- msprobe SGLang：https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md
- msprobe 异步训推：`msprobe/docs/zh/dump/verl_async_consistency_preprocess_dump.md`
