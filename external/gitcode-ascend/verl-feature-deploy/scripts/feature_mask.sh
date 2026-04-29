#!/bin/bash
# feature_mask.sh - 7 位二进制特性掩码管理
#
# 7 个特性对应 7 位二进制掩码 (FEATURE_MASK):
#   Bit 6 (MSB): remove_padding     - Remove Padding 优化 (吞吐 1.2-2.0×)
#   Bit 5:        dynamic_batch     - 动态 Batch Size (内存优化+负载均衡)
#   Bit 4:        offload           - 参数/优化器/梯度 Offload (显存省 30-50%)
#   Bit 3:        prefix_cache      - Prefix Cache (相同前缀生成加速)
#   Bit 2:        recompute         - 重计算/Gradient Checkpointing (显存省 30-40%)
#   Bit 1:        swap_optimizer    - Swap Optimizer (内存优化)
#   Bit 0 (LSB): vpp               - Virtual Pipeline Parallelism (吞吐提升)
#
# 使用方式:
#   FEATURE_MASK=1100000  → remove_padding + dynamic_batch
#   FEATURE_MASK=1111111  → 全部开启
#   FEATURE_MASK=0000000  → 全部关闭 (baseline)
#
# 也可通过单独环境变量覆盖 (优先级高于 FEATURE_MASK):
#   ENABLE_REMOVE_PADDING, ENABLE_DYNAMIC_BATCH, ENABLE_OFFLOAD,
#   ENABLE_PREFIX_CACHE, ENABLE_RECOMPUTE, ENABLE_SWAP_OPTIMIZER, ENABLE_VPP

set -euo pipefail

# ==========================================
# 特性名称定义 (从 MSB 到 LSB)
# ==========================================
readonly _FM_NAMES=("remove_padding" "dynamic_batch" "offload" "prefix_cache" "recompute" "swap_optimizer" "vpp")

# ==========================================
# 0/1/True/False → True/False
# ==========================================
_fm_bit_to_bool() {
    case "$1" in
        1|True|true|ON|on)   echo "True" ;;
        0|False|false|OFF|off) echo "False" ;;
        *) echo "False" ;;
    esac
}

# ==========================================
# 解析 FEATURE_MASK → 各 ENABLE_* 变量
# ==========================================
parse_feature_mask() {
    local mask="${FEATURE_MASK:-1100000}"

    # 校验: 7 位二进制
    if ! echo "$mask" | grep -qE '^[01]{7}$'; then
        echo "[ERROR] FEATURE_MASK 必须是 7 位二进制 (如 1100000), 当前: $mask" >&2
        exit 1
    fi

    # 逐位解析，仅当对应 ENABLE_* 未设置时才从 mask 读取
    ENABLE_REMOVE_PADDING="${ENABLE_REMOVE_PADDING:-$(echo "$mask" | cut -c1)}"
    ENABLE_DYNAMIC_BATCH="${ENABLE_DYNAMIC_BATCH:-$(echo "$mask" | cut -c2)}"
    ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-$(echo "$mask" | cut -c3)}"
    ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-$(echo "$mask" | cut -c4)}"
    ENABLE_RECOMPUTE="${ENABLE_RECOMPUTE:-$(echo "$mask" | cut -c5)}"
    ENABLE_SWAP_OPTIMIZER="${ENABLE_SWAP_OPTIMIZER:-$(echo "$mask" | cut -c6)}"
    ENABLE_VPP="${ENABLE_VPP:-$(echo "$mask" | cut -c7)}"

    # 统一转为 True/False
    ENABLE_REMOVE_PADDING=$(_fm_bit_to_bool "$ENABLE_REMOVE_PADDING")
    ENABLE_DYNAMIC_BATCH=$(_fm_bit_to_bool "$ENABLE_DYNAMIC_BATCH")
    ENABLE_OFFLOAD=$(_fm_bit_to_bool "$ENABLE_OFFLOAD")
    ENABLE_PREFIX_CACHE=$(_fm_bit_to_bool "$ENABLE_PREFIX_CACHE")
    ENABLE_RECOMPUTE=$(_fm_bit_to_bool "$ENABLE_RECOMPUTE")
    ENABLE_SWAP_OPTIMIZER=$(_fm_bit_to_bool "$ENABLE_SWAP_OPTIMIZER")
    ENABLE_VPP=$(_fm_bit_to_bool "$ENABLE_VPP")

    export ENABLE_REMOVE_PADDING ENABLE_DYNAMIC_BATCH ENABLE_OFFLOAD
    export ENABLE_PREFIX_CACHE ENABLE_RECOMPUTE ENABLE_SWAP_OPTIMIZER ENABLE_VPP
}

# ==========================================
# 从当前 ENABLE_* 反向生成 FEATURE_MASK
# ==========================================
build_feature_mask() {
    local mask=""
    mask="${mask}$([ "$ENABLE_REMOVE_PADDING" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_DYNAMIC_BATCH" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_OFFLOAD" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_PREFIX_CACHE" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_RECOMPUTE" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_SWAP_OPTIMIZER" = "True" ] && echo 1 || echo 0)"
    mask="${mask}$([ "$ENABLE_VPP" = "True" ] && echo 1 || echo 0)"
    echo "$mask"
}

# ==========================================
# OOM 自动恢复: 逐步开启显存特性
# 优先级: offload → recompute → swap_optimizer
# ==========================================
oom_upgrade_features() {
    local oom_retry="${OOM_RETRY_COUNT:-0}"

    if [ "$oom_retry" -ge 1 ] && [ "$ENABLE_OFFLOAD" != "True" ]; then
        ENABLE_OFFLOAD="True"
        echo "[OOM] 追加 offload 特性" >&2
    fi
    if [ "$oom_retry" -ge 2 ] && [ "$ENABLE_RECOMPUTE" != "True" ]; then
        ENABLE_RECOMPUTE="True"
        echo "[OOM] 追加 recompute 特性" >&2
    fi
    if [ "$oom_retry" -ge 3 ] && [ "$ENABLE_SWAP_OPTIMIZER" != "True" ]; then
        ENABLE_SWAP_OPTIMIZER="True"
        echo "[OOM] 追加 swap_optimizer 特性" >&2
    fi

    export ENABLE_OFFLOAD ENABLE_RECOMPUTE ENABLE_SWAP_OPTIMIZER
}

# ==========================================
# 生成 sed 替换命令 (用于模板占位符替换)
# ==========================================
feature_sed_commands() {
    local sed_cmds=""

    # Remove Padding
    sed_cmds="${sed_cmds}s|USE_REMOVE_PADDING_PLACEHOLDER|${ENABLE_REMOVE_PADDING}|g;"
    # Dynamic Batch
    sed_cmds="${sed_cmds}s|USE_DYNAMIC_BSZ_PLACEHOLDER|${ENABLE_DYNAMIC_BATCH}|g;"
    # Offload
    sed_cmds="${sed_cmds}s|OFFLOAD_PLACEHOLDER|${ENABLE_OFFLOAD}|g;"
    # Prefix Cache
    sed_cmds="${sed_cmds}s|ENABLE_PREFIX_CACHING_PLACEHOLDER|${ENABLE_PREFIX_CACHE}|g;"
    # Recompute / Gradient Checkpointing
    sed_cmds="${sed_cmds}s|ENABLE_GRADIENT_CHECKPOINTING_PLACEHOLDER|${ENABLE_RECOMPUTE}|g;"
    # Swap Optimizer
    sed_cmds="${sed_cmds}s|SWAP_OPTIMIZER_PLACEHOLDER|${ENABLE_SWAP_OPTIMIZER}|g;"
    # VPP size (True→2, False→1 即不启用)
    local vpp_val="1"
    [ "$ENABLE_VPP" = "True" ] && vpp_val="2"
    sed_cmds="${sed_cmds}s|VPP_SIZE_PLACEHOLDER|${vpp_val}|g;"

    echo "$sed_cmds"
}

# ==========================================
# 打印特性状态表
# ==========================================
print_feature_status() {
    local mask="${FEATURE_MASK:-$(build_feature_mask)}"
    echo "特性掩码:  ${mask}"
    echo "  [6] remove_padding:   $ENABLE_REMOVE_PADDING"
    echo "  [5] dynamic_batch:    $ENABLE_DYNAMIC_BATCH"
    echo "  [4] offload:          $ENABLE_OFFLOAD"
    echo "  [3] prefix_cache:     $ENABLE_PREFIX_CACHE"
    echo "  [2] recompute:        $ENABLE_RECOMPUTE"
    echo "  [1] swap_optimizer:   $ENABLE_SWAP_OPTIMIZER"
    echo "  [0] vpp:              $ENABLE_VPP"
}

# ==========================================
# OOM 检测
# ==========================================
check_oom() {
    local log_file="${1:-train.log}"
    if grep -qiE "out of memory|OOM|CUDA out of memory|NPU out of memory|memory allocation failed" "$log_file" 2>/dev/null; then
        echo "[WARN] 检测到 OOM 错误！" >&2
        return 0  # OOM detected
    fi
    return 1  # No OOM
}

# ==========================================
# Help
# ==========================================
print_feature_help() {
    cat << 'EOF'
7 位二进制特性掩码 (FEATURE_MASK):
  Bit 6 (MSB): remove_padding     - Remove Padding 优化 (吞吐 1.2-2.0×)
  Bit 5:        dynamic_batch     - 动态 Batch Size (内存优化+负载均衡)
  Bit 4:        offload           - 参数/优化器/梯度 Offload (显存省 30-50%)
  Bit 3:        prefix_cache      - Prefix Cache (相同前缀生成加速)
  Bit 2:        recompute         - 重计算/Gradient Checkpointing (显存省 30-40%)
  Bit 1:        swap_optimizer    - Swap Optimizer (内存优化)
  Bit 0 (LSB): vpp               - Virtual Pipeline Parallelism (吞吐提升)

默认掩码: 1100000 (remove_padding + dynamic_batch)
OOM 自动恢复: offload → recompute → swap_optimizer

使用方式:
  FEATURE_MASK=1100000  bash start_verl.sh   # 默认特性
  FEATURE_MASK=1111111  bash start_verl.sh   # 全开
  FEATURE_MASK=0000000  bash start_verl.sh   # baseline

  # 单独覆盖 (优先级高于 FEATURE_MASK)
  FEATURE_MASK=1100000 ENABLE_OFFLOAD=True bash start_verl.sh
EOF
}

# ==========================================
# CLI 入口
# ==========================================
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    action="${1:-help}"
    case "$action" in
        parse)
            parse_feature_mask
            print_feature_status
            ;;
        mask)
            # 从当前环境变量生成 mask
            ENABLE_REMOVE_PADDING="${ENABLE_REMOVE_PADDING:-False}"
            ENABLE_DYNAMIC_BATCH="${ENABLE_DYNAMIC_BATCH:-False}"
            ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-False}"
            ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-False}"
            ENABLE_RECOMPUTE="${ENABLE_RECOMPUTE:-False}"
            ENABLE_SWAP_OPTIMIZER="${ENABLE_SWAP_OPTIMIZER:-False}"
            ENABLE_VPP="${ENABLE_VPP:-False}"
            build_feature_mask
            ;;
        sed)
            parse_feature_mask
            feature_sed_commands
            ;;
        help|--help|-h)
            print_feature_help
            ;;
        *)
            echo "用法: $0 {parse|mask|sed|help}"
            exit 1
            ;;
    esac
fi
