#!/bin/bash
# feature_manager.sh - 智能特性管理
#
# 特性策略：
# 1. 用户显式指定特性 → 优先级最高
# 2. 用户未指定 → 使用默认性能特性（全开）
# 3. OOM 时 → 自动追加显存特性重试
# 4. 显存特性默认关闭，仅在 OOM 时开启

set -euo pipefail

# ==========================================
# 特性定义
# ==========================================

# 性能特性 - 提升吞吐，默认开启
PERF_FEATURES_DEFAULT="flash_attn dynamic_batch remove_padding gradient_checkpointing"

# 显存特性 - 节省显存，默认关闭，OOM 时开启
MEM_FEATURES="offload recompute"

# ==========================================
# 配置变量
# ==========================================

# 用户显式指定 (最高优先级)
USER_FEATURES="${USER_FEATURES:-}"

# OOM 重试计数
OOM_RETRY_COUNT="${OOM_RETRY_COUNT:-0}"

# ==========================================
# 核心逻辑：决定最终特性
# ==========================================

determine_features() {
    local result=""
    local mem_features_to_add=""
    
    # ==========================================
    # Step 1: 基础特性来源
    # ==========================================
    
    if [ -n "$USER_FEATURES" ]; then
        # 用户显式指定：使用用户特性作为基础
        info "用户显式指定特性: $USER_FEATURES"
        result=$(echo "$USER_FEATURES" | tr ',' ' ')
    else
        # 用户未指定：使用默认性能特性
        info "使用默认性能特性: $PERF_FEATURES_DEFAULT"
        result="$PERF_FEATURES_DEFAULT"
    fi
    
    # ==========================================
    # Step 2: OOM 时追加显存特性
    # ==========================================
    
    if [ "$OOM_RETRY_COUNT" -ge 1 ]; then
        info "OOM 重试 #$OOM_RETRY_COUNT：追加显存优化特性"

        # 根据 OOM 重试次数，逐步开启显存特性
        [ "$OOM_RETRY_COUNT" -ge 1 ] && mem_features_to_add="$mem_features_to_add offload"
        [ "$OOM_RETRY_COUNT" -ge 2 ] && mem_features_to_add="$mem_features_to_add recompute"
        
        # 检查是否已包含显存特性，避免重复
        for feat in $mem_features_to_add; do
            if ! echo "$result" | grep -qw "$feat"; then
                result="$result $feat"
            fi
        done
    fi
    
    # ==========================================
    # Step 3: 输出最终特性
    # ==========================================
    
    info "最终特性配置: $result"
    echo "$result"
}

# ==========================================
# OOM 检测
# ==========================================

check_oom() {
    local log_file="${1:-/verl/train.log}"
    
    # 检查常见的 OOM 错误模式
    if grep -qiE "out of memory|OOM|CUDA out of memory|NPU out of memory|memory allocation failed" "$log_file" 2>/dev/null; then
        warn "检测到 OOM 错误！"
        return 0  # OOM detected
    fi
    return 1  # No OOM
}

# ==========================================
# 特性转 Hydra 参数
# ==========================================

features_to_hydra() {
    local features="$1"
    local hydra_args=""
    
    for feat in $features; do
        case "$feat" in
            flash_attn)
                hydra_args="$hydra_args ++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True"
                hydra_args="$hydra_args ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True"
                hydra_args="$hydra_args ++critic.megatron.override_transformer_config.use_flash_attn=True"
                ;;
            dynamic_batch|dynamic_bsz)
                hydra_args="$hydra_args actor_rollout_ref.actor.use_dynamic_bsz=True"
                hydra_args="$hydra_args actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
                hydra_args="$hydra_args actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
                ;;
            remove_padding)
                hydra_args="$hydra_args actor_rollout_ref.model.use_remove_padding=True"
                ;;
            gradient_checkpointing|recompute)
                hydra_args="$hydra_args actor_rollout_ref.model.enable_gradient_checkpointing=True"
                ;;
            offload)
                hydra_args="$hydra_args actor_rollout_ref.actor.megatron.param_offload=True"
                hydra_args="$hydra_args actor_rollout_ref.actor.megatron.optimizer_offload=True"
                hydra_args="$hydra_args actor_rollout_ref.actor.megatron.grad_offload=True"
                ;;
            prefix_cache|prefix_caching)
                hydra_args="$hydra_args actor_rollout_ref.rollout.enable_prefix_caching=True"
                ;;
            chunked_prefill)
                hydra_args="$hydra_args actor_rollout_ref.rollout.enable_chunked_prefill=True"
                ;;
            vpp2|vpp)
                warn "VPP2 特性需要额外配置 pipeline_model_parallel_size，暂不支持自动配置"
                ;;
            *)
                warn "未知特性: $feat (已忽略)"
                ;;
        esac
    done
    
    echo "$hydra_args"
}

# ==========================================
# 打印特性说明
# ==========================================

print_feature_help() {
    cat << 'EOF'
特性管理说明：

【性能特性】默认开启，提升吞吐
  - flash_attn          Flash Attention 加速
  - dynamic_batch       动态 Batch Size
  - remove_padding      Remove Padding 优化
  - gradient_checkpointing  梯度检查点

【显存特性】默认关闭，OOM 时自动开启
  - offload             参数/优化器/梯度卸载到 CPU
  - recompute           重计算节省显存

【可选特性】默认关闭
  - prefix_cache        Prefix Cache (vLLM 推理加速)
  - chunked_prefill     Chunked Prefill
  - vpp2                Virtual Pipeline Parallel

使用方式：
  export USER_FEATURES="offload,recompute,prefix_cache"
  bash scripts/run_dapo.sh
EOF
}

# ==========================================
# 主函数
# ==========================================

main() {
    local action="${1:-determine}"
    
    case "$action" in
        determine)
            determine_features
            ;;
        check_oom)
            check_oom "${2:-/verl/train.log}"
            ;;
        to_hydra)
            features_to_hydra "${2:-}"
            ;;
        help|--help|-h)
            print_feature_help
            ;;
        *)
            echo "用法: $0 {determine|check_oom|to_hydra|help}"
            exit 1
            ;;
    esac
}

# 如果直接执行
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi