#!/bin/bash
# ============================================================================
# run_training.sh — 下层训练脚本模板
# 职责：训练超参、特性配置、Hydra python 训练命令
#
# 由 generate_training.sh 通过 sed 替换占位符生成最终脚本
# 被 start_verl.sh 调用
#
# 特性处理：
#   Type A (bit0-2): 直接 sed 替换 True/False 占位符
#   Type B (bit3-6): 默认注释，按 bit 取消注释或删除
# ============================================================================

#!/usr/bin/env bash
set -xeuo pipefail

# ==========================================
# 1. 项目标识
# ==========================================
PROJECT_NAME="${PROJECT_NAME:?PROJECT_NAME 未设置}"
project_name="$PROJECT_NAME"
exp_name="DAPO-${project_name}-megatron-$(date +%m%d_%H%M)"

# ==========================================
# 2. 集群拓扑（由上层 start_verl.sh export 继承）
# ==========================================
NNODES="${NNODES:-1}"
NPU_PER_NODE="${NPU_PER_NODE:-8}"

# ==========================================
# 3. 路径配置
# ==========================================
MODEL_PATH="MODEL_PATH_PLACEHOLDER"
CKPTS_DIR="CKPTS_DIR_PLACEHOLDER"
TRAIN_FILE="TRAIN_FILE_PLACEHOLDER"
TEST_FILE="TEST_FILE_PLACEHOLDER"
LOG_DIR="LOG_DIR_PLACEHOLDER"

# ==========================================
# 4. 并行策略配置
# ==========================================
gen_tp="GEN_TP_PLACEHOLDER"
train_tp="TRAIN_TP_PLACEHOLDER"
train_pp="TRAIN_PP_PLACEHOLDER"

# ==========================================
# 5. 算法参数
# ==========================================
adv_estimator="grpo"
use_kl_in_reward="False"
kl_coef="0.0"
use_kl_loss="False"
kl_loss_coef="0.0"
clip_ratio_low="0.2"
clip_ratio_high="0.28"
loss_agg_mode="token-mean"

# 序列长度
max_prompt_length="MAX_PROMPT_LENGTH_PLACEHOLDER"
max_response_length="MAX_RESPONSE_LENGTH_PLACEHOLDER"
enable_overlong_buffer="True"
overlong_buffer_len="OVERLONG_BUFFER_LEN_PLACEHOLDER"
overlong_penalty_factor="1.0"

# Batch
train_prompt_bsz="TRAIN_PROMPT_BSZ_PLACEHOLDER"
n_resp_per_prompt="N_RESP_PER_PROMPT_PLACEHOLDER"
train_prompt_mini_bsz="TRAIN_PROMPT_MINI_BSZ_PLACEHOLDER"

# ==========================================
# 6. 采样参数
# ==========================================
temperature="1.0"
top_p="1.0"
top_k="-1"
val_top_p="0.7"

# ==========================================
# 7. 性能参数
# ==========================================
use_dynamic_bsz="False"
offload="True"
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# 优化器
lr="LR_PLACEHOLDER"
lr_warmup_steps="LR_WARMUP_STEPS_PLACEHOLDER"
weight_decay="0.1"
clip_grad="1.0"
entropy_coeff="0"

# micro batch
ppo_micro_batch_size_per_gpu="PPO_MICRO_BATCH_PLACEHOLDER"
ref_log_prob_micro_batch_size_per_gpu="REF_MICRO_BATCH_PLACEHOLDER"
rollout_log_prob_micro_batch_size_per_gpu="ROLLOUT_MICRO_BATCH_PLACEHOLDER"

# rollout
gpu_memory_utilization="0.80"
enable_chunked_prefill="True"

# ==========================================
# 8. 训练器参数
# ==========================================
total_training_steps="TOTAL_TRAINING_STEPS_PLACEHOLDER"
total_epochs="2"
val_before_train="False"
test_freq="TEST_FREQ_PLACEHOLDER"
save_freq="SAVE_FREQ_PLACEHOLDER"
resume_mode="auto"
balance_batch="False"

# ==========================================
# 配置摘要
# ==========================================
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "  项目名:          $project_name"
echo "  实验名:          $exp_name"
echo "  模型:            $MODEL_PATH"
echo "  训练数据:        $TRAIN_FILE"
echo "  验证数据:        $TEST_FILE"
echo "  Checkpoint:      $CKPTS_DIR"
echo "  节点数:          $NNODES"
echo "  每节点NPU:       $NPU_PER_NODE"
echo "  gen_tp:          $gen_tp"
echo "  train_tp:        $train_tp"
echo "  train_pp:        $train_pp"
echo "  训练步数:        $total_training_steps"
echo "=========================================="

# ==========================================
# 9. 启动训练
# ==========================================
mkdir -p "$LOG_DIR"

python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=PARAM_OFFLOAD_PLACEHOLDER \
    actor_rollout_ref.actor.megatron.optimizer_offload=OPTIMIZER_OFFLOAD_PLACEHOLDER \
    actor_rollout_ref.actor.megatron.grad_offload=GRAD_OFFLOAD_PLACEHOLDER \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.optim.clip_grad=${clip_grad} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.use_remove_padding=USE_REMOVE_PADDING_PLACEHOLDER \
    actor_rollout_ref.actor.use_dynamic_bsz=USE_DYNAMIC_BSZ_PLACEHOLDER \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=USE_DYNAMIC_BSZ_PLACEHOLDER \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=USE_DYNAMIC_BSZ_PLACEHOLDER \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.param_offload=PARAM_OFFLOAD_PLACEHOLDER \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NPU_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.balance_batch=${balance_batch} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permutation_async_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_ring_attention_update=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True \
    # actor_rollout_ref.rollout.enable_prefix_caching=True \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8 \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    # actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \
    trainer.device=npu 2>&1 | tee "${LOG_DIR}/verl_qwen3_8b_I_megatron_$(date +%Y%m%d_%H%M).log"
