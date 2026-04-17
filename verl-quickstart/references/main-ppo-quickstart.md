# main_ppo 训练骨架（参考）

本仓提供两种训练后端示例：

- **FSDP（默认）**：参考 `examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh`
- **Megatron**：参考 `examples/grpo_trainer/run_qwen2_5-32b_grpo_megatron_vllm_npu.sh`

## 1) FSDP 后端示例（参考 7B 脚本）

特点：直接 `python3 -m verl.trainer.main_ppo ...`，通过 `actor_rollout_ref.actor.fsdp_config.*` / `actor_rollout_ref.ref.fsdp_config.*` 控制 FSDP/offload。

```bash
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export LOG="${LOG:-logs/run_qwen2_5_7b_grpo_fsdp_vllm_npu.log}"
mkdir -p "$(dirname "$LOG")"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=1024 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=5e-8 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger=console \
  trainer.project_name='verl_grpo_example_gsm8k' \
  trainer.experiment_name='qwen2_5_7b_fsdp_vllm_npu' \
  trainer.n_gpus_per_node="${NGPUS:-16}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=5 \
  "$@" | tee "$LOG"
```

## 2) Megatron 后端示例（参考 32B 脚本）

特点：`main_ppo` 通过 `--config-name='ppo_megatron_trainer.yaml'` 指定 Megatron trainer，并使用 `actor_rollout_ref.*.megatron.*` 进行 TP/PP/EP 等并行配置；通常还会配置 `dist_checkpointing_path` 等。

### 常见报错（MindSpeed / flash-attn mask）

如果遇到类似：

`RuntimeError: Please set micro_batch_size or set use_flash_attn=True in config.`

通常需要显式打开 flash-attn（MindSpeed 侧依赖该开关生成 attention mask），可以在命令行追加：

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
```

```bash
mkdir -p logs

export NNODES="${NNODES:-1}"
export NPUS_PER_NODE="${NPUS_PER_NODE:-16}"

export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export TRAIN_FILE="${TRAIN_FILE:-$RAY_DATA_HOME/dataset/gsm8k/train.parquet}"
export TEST_FILE="${TEST_FILE:-$RAY_DATA_HOME/dataset/gsm8k/test.parquet}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-32B}"
export MCORE_MODEL_PATH="${MCORE_MODEL_PATH:-Qwen/Qwen2.5-32B-dist}"

python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name='ppo_megatron_trainer.yaml' \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=False \
  data.truncation='left' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="$MCORE_MODEL_PATH" \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
  actor_rollout_ref.ref.megatron.context_parallel_size=1 \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
  actor_rollout_ref.ref.megatron.dist_checkpointing_path="$MCORE_MODEL_PATH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  trainer.project_name='GRPO-Qwen2.5-32B-BASE-MATH' \
  trainer.experiment_name='GRPO-Qwen2.5-32B-BASE-Megatron-vLLM' \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$NPUS_PER_NODE" \
  trainer.device='npu' \
  trainer.total_epochs=15 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  "$@" | tee logs/run_qwen2_5_32b_grpo_megatron_vllm_npu.log
```

## OOM/不稳时通用建议

- 先降：`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`、`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`
- 再降：rollout 侧 `gpu_memory_utilization`
- 再降：`data.max_prompt_length` / `data.max_response_length`
