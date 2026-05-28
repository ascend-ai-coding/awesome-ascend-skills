# 训推一致性：Legacy Megatron → Engine Worker 适配

本文档说明如何将 msprobe **legacy 分支**（`megatron_workers.py` + `megatron_actor.py`）的训推一致采集方案，迁移到 verl **v0.7+ engine worker + 异步 rollout** 架构。

## 架构差异

| 维度 | Legacy（v0.6.x / SPMD） | Engine Worker + 异步（v0.7+） |
|------|-------------------------|-------------------------------|
| 训练 dump 挂载点 | `megatron_workers.py` 的 `compute_log_prob` / `generate_sequences` | `FSDPEngine` / `MegatronEngine` 的 `forward_backward_batch` |
| PROMPTS_ONLY 裁剪 | `megatron_actor.py` 内多处（forward_step / loss_func） | `ray_trainer.py` 在 `old_log_prob` 前裁剪 batch；FSDP engine 用 `use_remove_padding` + rolled labels 自然对齐 |
| 推理 dump | worker 内直接 `debugger.start(infer_model)` | vLLM-Ascend `dump_config_path` + `enforce_eager=True` |
| request_id 链路 | 不适用 | `agent_loop.py` 或 `llm_server.py` 注入 `extra_fields["request_id"]` |
| 调度日志 | 不适用 | vLLM-Ascend `dispatch_logger.py`（可选，用于 prefill step 关联） |

**关键结论**：legacy 里对 `megatron_actor.py` 的大段 forward/loss 改动，在 FSDP engine 路径下**不必逐行移植**；engine worker 只需在 `forward_backward_batch` 挂 debugger，PROMPTS_ONLY 在 trainer 侧裁剪即可。

## 必改文件清单（verl）

| 文件 | 作用 | 说明 |
|------|------|------|
| `verl/workers/engine/{fsdp,megatron}/transformer_impl.py` | 训练侧 msprobe dump | `_ensure_debugger` + `forward_backward_batch` hook |
| `verl/trainer/ppo/ray_trainer.py` | PROMPTS_ONLY | `old_log_prob` 前裁剪 response token |
| `verl/utils/debug/metrics.py` | 辅助 | PROMPTS_ONLY 时跳过 rollout/actor diff 指标 |
| `verl/trainer/ppo/rollout_corr_helper.py` | 辅助 | PROMPTS_ONLY 时跳过 rollout correction |
| `verl/experimental/agent_loop/agent_loop.py` **或** `verl/workers/rollout/llm_server.py` | request_id | 将 vLLM `request_id` 写入 `TokenOutput.extra_fields` |

> **request_id 注入位置**：较新 verl 用 `llm_server.py`；部分部署（如 `/verl` 安装树）仅有 `agent_loop.py` 中的 `LLMServerClient.generate()`，需 patch 该处。校验脚本会检查二者至少其一。

## 可选：vLLM-Ascend dispatch_logger

| 文件 | 作用 |
|------|------|
| `vllm_ascend/worker/dispatch_logger.py` | 新增，写 `dispatch_log.jsonl` |
| `vllm_ascend/worker/model_runner_v1.py` | 在 `execute_model` 的 debugger stop 前调用 `DispatchLogger.log_step` |

推理侧 tensor dump 不依赖 dispatch_logger；但要做 **request_id 精确关联**（prefill step ↔ 训练 micro_batch）时需要它。

## 环境变量

```bash
export DUMP_ON=1
export PROMPTS_ONLY=1
export TORCHDYNAMO_DISABLE=1          # FSDP 推荐
export DUMP_PHASE=log_prob            # 默认仅采 old_log_prob 前向（prefill 对齐）
export MSPROBE_ACTOR_CONFIG=/path/to/config_actor.json
```

`DUMP_PHASE` 取值：

- `log_prob`（默认）：仅在 `forward_only=True`（`compute_log_prob` / `infer_batch`）时 dump
- `update_actor`：仅在 actor 更新 backward 时 dump
- `all`：两阶段都 dump

## Batch / Hydra 约束

### 通用

1. `train_batch_size = ppo_mini_batch_size = ppo_micro_batch_size_per_gpu = 1`（**推荐**）
2. `rollout.n = DP`，其中 `DP = world_size / SP`；**优先 `SP=1`**（见下）
3. `use_remove_padding=True`、`use_dynamic_bsz=False`、`balance_batch=False`、`val_before_train=False`
4. 关闭 rollout skip，必须真实 rollout
5. 不考虑 gac 时，仍须满足 `real_train_batch_size = train_batch_size * n` 可被 `n_gpus` 整除

### 优先关闭 SP（Ulysses）

原训练脚本若 `sp_size>1`，采集脚本仍应 **强制**：

```bash
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
```

可从原脚本 `grep '^sp_size='` 读取并在日志中提示 `sp_size X -> 1`。SP 关闭后 `DP = world_size`。

### FSDP + 4 GPU 参考（SP 关闭 + 并行对齐，已验证）

> **并行对齐**：推理 `tensor_model_parallel_size` = rollout 占用卡数（4 卡 → `gen_tp=4`）。详见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)。

```bash
data.train_batch_size=1
actor_rollout_ref.actor.ppo_mini_batch_size=1
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.n=4                    # DP = world_size，SP=1
actor_rollout_ref.rollout.agent.num_workers=1
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.data_parallel_size=1
```

### Megatron 最小采集（Qwen2.5-3B，切分对齐）

**原则**：train/rollout **TP·PP·EP 一致**。当前 verl **vLLM 不支持 rollout PP>1**，故采集将 train/ref/rollout **PP 均覆写为 1**（即使生产 `train_pp=2`）。

```bash
NPUS_PER_NODE=1 bash run_consistency_dump_megatron.sh
# train_pp 原=2 → 采集 PP=1；actor rank0 ↔ generate rank0
```

框架支持 vLLM PP 后，可 `CONSIST_PP=2` 且 `NPUS_PER_NODE=2`，比对 `actor rank1` ↔ `generate rank1`。

### Megatron 并行对齐（必检）

> **Megatron 训推一致**：`train_tp==gen_tp`，`train_pp==rollout.pipeline_model_parallel_size`，`train_ep==gen_ep`。详见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)。

```bash
train_tp=1; train_pp=2; train_ep=1
gen_tp=1;   gen_pp=2;   gen_dp=1; gen_ep=1   # gen_pp 必须等于 train_pp（采集脚本覆盖）
trainer.balance_batch=False
```

### 推理侧

```bash
actor_rollout_ref.rollout.enforce_eager=True
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path="/path/to/config_generate.json"
```

## msprobe 配置

训练侧 `config_actor.json`（须含 `dump_path`，`level: mix`）：

- 模板：`assets/config_actor.json.example`

推理侧 `config_generate.json`（须含 `dump_path`，建议 `step: [0]`）：

- 模板：`assets/config_generate.json.example`

## 采集脚本模板

基于原训练脚本复制为 `run_consistency_dump_<date>.sh`，**不要改原脚本**。

- FSDP 示例：`assets/run_consistency_dump.sh.example`
- Megatron 示例：`assets/run_consistency_dump_megatron.sh.example`；可运行副本见 `verl/examples/test_skill_dump/run_consistency_dump_megatron.sh`（基于 `run_qwen2.5-3b-megatron.sh`）

```bash
export DUMP_ON=1 PROMPTS_ONLY=1 TORCHDYNAMO_DISABLE=1 DUMP_PHASE=log_prob
export MSPROBE_ACTOR_CONFIG="${SCRIPT_DIR}/config_actor.json"

CONSIST_GPUS="${GPUS_PER_NODE:-4}"
CONSIST_WORLD_SIZE=$((CONSIST_GPUS * NNODES))
CONSIST_SP=1
ORIG_SP=$(grep -E '^sp_size=' "${ORIG_SCRIPT}" | tail -1 | cut -d= -f2 | tr -d ' ')
# 若 ORIG_SP>1，日志提示并强制 SP=1

CONSIST_DP=$((CONSIST_WORLD_SIZE / CONSIST_SP))
CONSIST_N_RESP=${CONSIST_DP}

TMP_SCRIPT="$(mktemp)"
sed -e 's/actor_rollout_ref\.rollout\.skip\.enable=True/actor_rollout_ref.rollout.skip.enable=False/' \
  "${ORIG_SCRIPT}" > "${TMP_SCRIPT}"

bash "${TMP_SCRIPT}" \
  data.train_batch_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=${CONSIST_N_RESP} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${CONSIST_GPUS} \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${CONSIST_SP} \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=${CONSIST_SP} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  trainer.val_before_train=False \
  trainer.balance_batch=False \
  trainer.total_training_steps=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path="${GENERATE_CONFIG}"
```

## 校验

```bash
# patch 完整性（verl + 可选 vLLM-Ascend）
python3 scripts/check-consistency-patch.py \
  --verl-root "$VERL_ROOT" \
  --backend fsdp \
  --vllm-ascend-root /path/to/vllm-ascend

# 仅 engine worker
python3 scripts/check-engine-patch.py --verl-root "$VERL_ROOT" --backend fsdp
```

**重要**：patch 必须打在**实际运行训练**的 verl 树上。若 `python3 -m verl.trainer.main_ppo` 使用的是 `/verl` 而非 skill 里的 `VERL_ROOT`，需在 `/verl` 同步 patch。

## 期望产物

| 侧 | 路径 | 说明 |
|----|------|------|
| 推理 | `{generate_dump_path}/step0/rank_*/dump.json` | vLLM prefill step 0；**rank 数 = rollout TP**（FSDP 须 TP=卡数；Megatron 须与训练 TP 一致） |
| 推理 | `{generate_dump_path}/{pid}/dispatch_log.jsonl` | 需 dispatch_logger patch |
| 训练 | `{actor_dump_path}/step_*/rank_*/dump.json` | engine worker old_log_prob 前向 |
| 训练 | `{actor_dump_path}/{pid}/update_actor_log.jsonl` | request_id 元数据 |

关联方法见 [dump-output-correlation.md](dump-output-correlation.md)。

## 参考

- msprobe legacy megatron：`msprobe/docs/zh/dump/verl_megatron_consistency_preprocess_dump.md`
- msprobe 异步 engine worker：`msprobe/docs/zh/dump/verl_async_consistency_preprocess_dump.md`
