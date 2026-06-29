#!/bin/bash
# run_one_step_off_policy.sh - Verl One-Step-Off-Policy DAPO 训练启动脚本
#
# 特性：
# - Trainer 和 Rollout 资源隔离
# - 支持 Megatron/FSDP2 框架
# - 自动包含所有必需的 reward_model 配置
#
# 用法：
#   TRAIN_STEPS=100 bash run_one_step_off_policy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/common.sh"

cd /verl

# ==========================================
# 配置参数
# ==========================================
FRAMEWORK="${FRAMEWORK:-megatron}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_8B}"
TRAIN_FILE="${TRAIN_FILE:-$DEFAULT_TRAIN_DATA}"
VAL_FILE="${VAL_FILE:-$DEFAULT_VAL_DATA}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
EXP_NAME="${EXP_NAME:-DAPO-Qwen3-8b-one-step-off-${FRAMEWORK}-$(date +%m%d_%H%M)}"
PROJECT_NAME="${PROJECT_NAME:-DAPO}"
CKPTS_DIR="${CKPTS_DIR:-/mnt2/ckpt/${EXP_NAME}}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TRAINER_GPUS="${TRAINER_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# 长度参数
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# Batch 参数
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"

# 并行参数
GEN_TP="${GEN_TP:-4}"
TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-1}"

# 特性开关
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-False}"

# ==========================================
# 路径验证
# ==========================================
info "验证路径..."
validate_all_paths "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"
if [ $? -gt 0 ]; then
    fail "路径验证失败，请检查后重试"
    exit 1
fi

# ==========================================
# 环境准备
# ==========================================
cleanup_ray
load_proxy_config
setup_proxy

export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200

# ==========================================
# 启动训练
# ==========================================
info "=========================================="
info "启动 Verl One-Step-Off-Policy DAPO 训练"
info "=========================================="
info "框架: $FRAMEWORK"
info "项目: $PROJECT_NAME / 实验: $EXP_NAME"
info "步数: $TRAIN_STEPS"
info "资源: Trainer=$TRAINER_GPUS GPUs, Rollout=$ROLLOUT_GPUS GPUs"
info "模型: $MODEL_PATH"
info "=========================================="

if [ "$FRAMEWORK" = "megatron" ]; then
    python3 -m recipe.one_step_off_policy.main_ppo \
        --config-path=config \
        --config-name='one_step_off_ppo_megatron_trainer.yaml' \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.strategy=megatron \
        critic.strategy=megatron \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
        actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING} \
        actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
        actor_rollout_ref.actor.megatron.param_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.actor.megatron.optimizer_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.actor.megatron.grad_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
        actor_rollout_ref.ref.megatron.param_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.nccl_timeout=7200 \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${MAX_RESPONSE_LENGTH} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
        trainer.logger='["console"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.val_before_train=False \
        trainer.test_freq=20 \
        trainer.save_freq=50 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=${TRAIN_STEPS} \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${TRAINER_GPUS} \
        rollout.nnodes=1 \
        rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
        "$@"
else
    # FSDP2 框架
    python3 -m recipe.one_step_off_policy.main_ppo \
        --config-path=config \
        --config-name='one_step_off_ppo_trainer.yaml' \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.strategy=fsdp2 \
        critic.strategy=fsdp2 \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING} \
        actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ENABLE_OFFLOAD} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${MAX_RESPONSE_LENGTH} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
        trainer.logger='["console"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.val_before_train=False \
        trainer.test_freq=20 \
        trainer.save_freq=50 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=${TRAIN_STEPS} \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${TRAINER_GPUS} \
        rollout.nnodes=1 \
        rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
        "$@"
fi

echo "Training completed successfully!"