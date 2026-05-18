#!/bin/bash
# quick_start.sh - 快速启动 DAPO 训练
#
# 用法:
#   CONTAINER=jins TRAIN_STEPS=100 bash quick_start.sh
#
# 环境变量:
#   CONTAINER      - 容器名称 (默认: jins)
#   TRAIN_STEPS    - 训练步数 (默认: 100)
#   FEATURES       - 特性列表 (默认: 空则使用默认配置)
#   SWANLAB_HOST   - SwanLab 地址
#   SWANLAB_API_KEY - SwanLab API 密钥

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ==========================================
# 配置参数
# ==========================================
CONTAINER="${CONTAINER:-jins}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
FEATURES="${FEATURES:-}"

# SwanLab
SWANLAB_HOST="${SWANLAB_HOST:-http://10.143.2.129:8000}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

# 路径
MODEL_PATH="/mnt/public/models/huggface_models/Qwen/Qwen3-8B"
TRAIN_FILE="/mnt2/wql/data/dapo-math-17k.parquet"
VAL_FILE="/mnt2/wql/data/aime-2024.parquet"

# 训练参数
PROJECT_NAME="DAPO-Qwen3-8B-async"
EXP_NAME="DAPO-Qwen3-8B-async-$(date +%m%d_%H%M)"
TRAINER_GPUS=4
ROLLOUT_GPUS=4
TOTAL_GPUS=$((TRAINER_GPUS + ROLLOUT_GPUS))

# ==========================================
# 交互式询问（非交互模式跳过）
# ==========================================

if [ -t 0 ]; then
    # 终端交互模式
    echo "=========================================="
    echo "Verl 异步 DAPO 训练启动"
    echo "=========================================="
    
    # 询问特性
    if [ -z "$FEATURES" ]; then
        echo ""
        echo "是否有自定义特性需求？(直接回车使用默认配置)"
        echo "  性能特性: flash_attn, dynamic_batch, remove_padding, gradient_checkpointing (默认全开)"
        echo "  显存特性: offload, recompute (默认关闭)"
        echo "  可选特性: prefix_cache, chunked_prefill"
        echo ""
        read -p "输入特性(逗号分隔): " user_features
        FEATURES="$user_features"
    fi
fi

# ==========================================
# 启动训练
# ==========================================

info "在容器 $CONTAINER 中启动训练..."

docker exec "$CONTAINER" bash -c "
cd /verl

# 环境配置
export SWANLAB_HOST=$SWANLAB_HOST
export SWANLAB_API_KEY=$SWANLAB_API_KEY
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200

# 清理旧 Ray
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray
sleep 2

# 启动 Ray (显式声明 GPU)
ray start --head --num-gpus=$TOTAL_GPUS --dashboard-port=8265 --port=6379 2>&1 | tail -5
sleep 3

# 验证 Ray
ray status 2>&1 | head -10

# 提交训练任务
ray job submit --no-wait --address='http://localhost:8265' \\
    -- python3 -m recipe.dapo.main_dapo \\
    data.train_files='$TRAIN_FILE' \\
    data.val_files='$VAL_FILE' \\
    data.prompt_key=prompt \\
    data.truncation=left \\
    data.max_prompt_length=2048 \\
    data.max_response_length=8192 \\
    data.gen_batch_size=24 \\
    data.train_batch_size=8 \\
    actor_rollout_ref.rollout.n=4 \\
    algorithm.adv_estimator=grpo \\
    algorithm.use_kl_in_reward=False \\
    algorithm.kl_ctrl.kl_coef=0.0 \\
    actor_rollout_ref.model.path='$MODEL_PATH' \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \\
    actor_rollout_ref.actor.optim.weight_decay=0.1 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.actor.grad_clip=1.0 \\
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=True \\
    actor_rollout_ref.rollout.temperature=1.0 \\
    actor_rollout_ref.rollout.top_p=1.0 \\
    actor_rollout_ref.rollout.top_k=-1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \\
    reward_model.reward_manager=dapo \\
    trainer.logger='[\"console\",\"swanlab\"]' \\
    trainer.project_name='$PROJECT_NAME' \\
    trainer.experiment_name='$EXP_NAME' \\
    trainer.n_gpus_per_node=$TOTAL_GPUS \\
    trainer.nnodes=1 \\
    trainer.val_before_train=False \\
    trainer.test_freq=20 \\
    trainer.save_freq=50 \\
    trainer.total_epochs=1 \\
    trainer.total_training_steps=$TRAIN_STEPS \\
    2>&1
"

info "=========================================="
info "训练任务已提交！"
info "=========================================="
info "SwanLab: $SWANLAB_HOST/@TrainingMaster/$PROJECT_NAME"
info "Ray Dashboard: http://<container-ip>:8265"
info "=========================================="