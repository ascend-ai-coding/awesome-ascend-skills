#!/bin/bash
# ============================================================================
# generate_training.sh — 训练脚本生成器
# 根据用户参数和 FEATURE_MASK 生成 start_verl.sh + run_training.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==========================================
# 参数解析
# ==========================================
MASK=""
OUTPUT_DIR=""
MODEL_PATH=""
TRAIN_FILE=""
TEST_FILE=""
CKPTS_DIR=""
NPU_DEVICES=""
MASTER_ADDR=""
SOCKET_IFNAME=""
CURRENT_IP=""
TRAIN_STEPS=""
GEN_TP=""
TRAIN_TP=""
TRAIN_PP=""
LR=""
LR_WARMUP_STEPS=""
TRAIN_PROMPT_BSZ=""
N_RESP_PER_PROMPT=""
TRAIN_PROMPT_MINI_BSZ=""
PPO_MICRO_BATCH=""
MAX_PROMPT_LENGTH=""
MAX_RESPONSE_LENGTH=""
SWANLAB="no"
SWANLAB_HOST=""
SWANLAB_API_KEY=""
SWANLAB_MODE=""
SWANLAB_WORKSPACE=""
SWANLAB_LOG_DIR=""
PROJECT_NAME=""
LOG_DIR=""
TEST_FREQ=""
SAVE_FREQ=""

while [ $# -gt 0 ]; do
    case "$1" in
        --mask)          MASK="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --model-path)    MODEL_PATH="$2"; shift 2 ;;
        --train-file)    TRAIN_FILE="$2"; shift 2 ;;
        --test-file)     TEST_FILE="$2"; shift 2 ;;
        --ckpts-dir)     CKPTS_DIR="$2"; shift 2 ;;
        --npu-devices)   NPU_DEVICES="$2"; shift 2 ;;
        --master-addr)   MASTER_ADDR="$2"; shift 2 ;;
        --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
        --current-ip)    CURRENT_IP="$2"; shift 2 ;;
        --train-steps)   TRAIN_STEPS="$2"; shift 2 ;;
        --gen-tp)        GEN_TP="$2"; shift 2 ;;
        --train-tp)      TRAIN_TP="$2"; shift 2 ;;
        --train-pp)      TRAIN_PP="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --lr-warmup)     LR_WARMUP_STEPS="$2"; shift 2 ;;
        --train-bsz)     TRAIN_PROMPT_BSZ="$2"; shift 2 ;;
        --n-resp)        N_RESP_PER_PROMPT="$2"; shift 2 ;;
        --train-mini-bsz) TRAIN_PROMPT_MINI_BSZ="$2"; shift 2 ;;
        --ppo-micro-bsz) PPO_MICRO_BATCH="$2"; shift 2 ;;
        --max-prompt)    MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max-response)  MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --swanlab)       SWANLAB="$2"; shift 2 ;;
        --swanlab-host)  SWANLAB_HOST="$2"; shift 2 ;;
        --swanlab-api-key) SWANLAB_API_KEY="$2"; shift 2 ;;
        --swanlab-mode)  SWANLAB_MODE="$2"; shift 2 ;;
        --swanlab-workspace) SWANLAB_WORKSPACE="$2"; shift 2 ;;
        --swanlab-log-dir) SWANLAB_LOG_DIR="$2"; shift 2 ;;
        --project-name)  PROJECT_NAME="$2"; shift 2 ;;
        --log-dir)       LOG_DIR="$2"; shift 2 ;;
        --test-freq)     TEST_FREQ="$2"; shift 2 ;;
        --save-freq)     SAVE_FREQ="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 校验必需参数
[ -z "$MASK" ] && echo "[ERROR] --mask required" && exit 1
[ -z "$OUTPUT_DIR" ] && echo "[ERROR] --output-dir required" && exit 1
[ -z "$MODEL_PATH" ] && echo "[ERROR] --model-path required" && exit 1

# 校验 MASK 格式
if ! echo "$MASK" | grep -qE '^[01]{7}$'; then
    echo "[ERROR] MASK must be 7-bit binary (e.g. 1100000), got: $MASK"
    exit 1
fi

# ==========================================
# 解析 FEATURE_MASK 各位
# ==========================================
bit_rmvPad="${MASK:0:1}"
bit_dynBsz="${MASK:1:1}"
bit_offload="${MASK:2:1}"
bit_prefixCache="${MASK:3:1}"
bit_recompute="${MASK:4:1}"
bit_swapOpt="${MASK:5:1}"
bit_vpp="${MASK:6:1}"

echo "特性掩码: $MASK"
echo "  [0] Remove Padding:  $bit_rmvPad"
echo "  [1] Dynamic BSZ:     $bit_dynBsz"
echo "  [2] Offload:         $bit_offload"
echo "  [3] Prefix Cache:    $bit_prefixCache"
echo "  [4] Recompute:       $bit_recompute"
echo "  [5] Swap Optimizer:  $bit_swapOpt"
echo "  [6] VPP:             $bit_vpp"

# ==========================================
# 计算特性变量值
# ==========================================
# Type A: bit0-2 → True/False
[ "$bit_rmvPad" = "1" ] && VAL_REMOVE_PADDING="True" || VAL_REMOVE_PADDING="False"
[ "$bit_dynBsz" = "1" ] && VAL_DYNAMIC_BSZ="True" || VAL_DYNAMIC_BSZ="False"
[ "$bit_offload" = "1" ] && VAL_PARAM_OFFLOAD="True" || VAL_PARAM_OFFLOAD="False"

# Offload 细化：optimizer_offload 默认 False（避免与 SwapOpt 冲突）
[ "$bit_offload" = "1" ] && VAL_OPTIMIZER_OFFLOAD="False" || VAL_OPTIMIZER_OFFLOAD="False"
[ "$bit_offload" = "1" ] && VAL_GRAD_OFFLOAD="True" || VAL_GRAD_OFFLOAD="False"

# 互斥处理：SwapOpt 开启时强制 optimizer_offload=False
if [ "$bit_swapOpt" = "1" ]; then
    VAL_OPTIMIZER_OFFLOAD="False"
    echo "  [互斥] SwapOpt 开启, optimizer_offload 强制 False"
fi

# ==========================================
# 准备输出目录
# ==========================================
mkdir -p "$OUTPUT_DIR"

# ==========================================
# 生成 start_verl.sh（上层脚本）
# ==========================================
START_TEMPLATE="$SCRIPT_DIR/../assets/start_template.sh"
START_OUT="$OUTPUT_DIR/start_verl.sh"

if [ ! -f "$START_TEMPLATE" ]; then
    echo "[ERROR] 上层模板不存在: $START_TEMPLATE"
    exit 1
fi

cp "$START_TEMPLATE" "$START_OUT"

# --- 替换上层模板占位符 ---
sed -i "s|NPU_DEVICES_PLACEHOLDER|${NPU_DEVICES:-0,1,2,3,4,5,6,7}|g" "$START_OUT"
sed -i "s|SOCKET_IFNAME_PLACEHOLDER|${SOCKET_IFNAME:-eth0}|g" "$START_OUT"
sed -i "s|CURRENT_IP_PLACEHOLDER|${CURRENT_IP:-127.0.0.1}|g" "$START_OUT"
sed -i "s|MASTER_ADDR_PLACEHOLDER|${MASTER_ADDR:-$CURRENT_IP}|g" "$START_OUT"
sed -i "s|FEATURE_MASK_PLACEHOLDER|${MASK}|g" "$START_OUT"

# SwanLab 配置（上层模板中标记区域）
if [ "$SWANLAB" = "yes" ]; then
    # 不将 API Key 写入文件，改为运行时通过环境变量注入
    # 模板中已改为 export SWANLAB_HOST="${SWANLAB_HOST:?...}" 形式
    # 此处只需确保运行时 export 了这些变量即可
    :
else
    # SwanLab 未开启：删除标记区域内的所有 export 行
    sed -i '/=== SWANLAB_CONFIG_START ===/,/=== SWANLAB_CONFIG_END ===/d' "$START_OUT"
fi

chmod +x "$START_OUT"
echo "[OK] 生成上层脚本: $START_OUT"

# ==========================================
# 生成 run_training.sh（下层训练脚本）
# ==========================================
TRAIN_TEMPLATE="$SCRIPT_DIR/../assets/training_template.sh"
TRAIN_OUT="$OUTPUT_DIR/run_training.sh"

if [ ! -f "$TRAIN_TEMPLATE" ]; then
    echo "[ERROR] 下层模板不存在: $TRAIN_TEMPLATE"
    exit 1
fi

cp "$TRAIN_TEMPLATE" "$TRAIN_OUT"

# --- Type A：替换简单变量占位符 ---
sed -i "s|USE_REMOVE_PADDING_PLACEHOLDER|${VAL_REMOVE_PADDING}|g" "$TRAIN_OUT"
sed -i "s|USE_DYNAMIC_BSZ_PLACEHOLDER|${VAL_DYNAMIC_BSZ}|g" "$TRAIN_OUT"
sed -i "s|PARAM_OFFLOAD_PLACEHOLDER|${VAL_PARAM_OFFLOAD}|g" "$TRAIN_OUT"
sed -i "s|OPTIMIZER_OFFLOAD_PLACEHOLDER|${VAL_OPTIMIZER_OFFLOAD}|g" "$TRAIN_OUT"
sed -i "s|GRAD_OFFLOAD_PLACEHOLDER|${VAL_GRAD_OFFLOAD}|g" "$TRAIN_OUT"

# --- 基础参数替换 ---
sed -i "s|MODEL_PATH_PLACEHOLDER|${MODEL_PATH}|g" "$TRAIN_OUT"
sed -i "s|TRAIN_FILE_PLACEHOLDER|${TRAIN_FILE}|g" "$TRAIN_OUT"
sed -i "s|TEST_FILE_PLACEHOLDER|${TEST_FILE}|g" "$TRAIN_OUT"
sed -i "s|CKPTS_DIR_PLACEHOLDER|${CKPTS_DIR}|g" "$TRAIN_OUT"
sed -i "s|LOG_DIR_PLACEHOLDER|${LOG_DIR:-/mnt/project/verl/logs}|g" "$TRAIN_OUT"
sed -i "s|GEN_TP_PLACEHOLDER|${GEN_TP:-4}|g" "$TRAIN_OUT"
sed -i "s|TRAIN_TP_PLACEHOLDER|${TRAIN_TP:-4}|g" "$TRAIN_OUT"
sed -i "s|TRAIN_PP_PLACEHOLDER|${TRAIN_PP:-2}|g" "$TRAIN_OUT"
sed -i "s|TOTAL_TRAINING_STEPS_PLACEHOLDER|${TRAIN_STEPS:-4}|g" "$TRAIN_OUT"
sed -i "s|LR_PLACEHOLDER|${LR:-1e-6}|g" "$TRAIN_OUT"
sed -i "s|LR_WARMUP_STEPS_PLACEHOLDER|${LR_WARMUP_STEPS:-2}|g" "$TRAIN_OUT"
sed -i "s|TRAIN_PROMPT_BSZ_PLACEHOLDER|${TRAIN_PROMPT_BSZ:-32}|g" "$TRAIN_OUT"
sed -i "s|N_RESP_PER_PROMPT_PLACEHOLDER|${N_RESP_PER_PROMPT:-8}|g" "$TRAIN_OUT"
sed -i "s|TRAIN_PROMPT_MINI_BSZ_PLACEHOLDER|${TRAIN_PROMPT_MINI_BSZ:-32}|g" "$TRAIN_OUT"
sed -i "s|PPO_MICRO_BATCH_PLACEHOLDER|${PPO_MICRO_BATCH:-2}|g" "$TRAIN_OUT"
sed -i "s|REF_MICRO_BATCH_PLACEHOLDER|4|g" "$TRAIN_OUT"
sed -i "s|ROLLOUT_MICRO_BATCH_PLACEHOLDER|4|g" "$TRAIN_OUT"
sed -i "s|MAX_PROMPT_LENGTH_PLACEHOLDER|${MAX_PROMPT_LENGTH:-1024}|g" "$TRAIN_OUT"
sed -i "s|MAX_RESPONSE_LENGTH_PLACEHOLDER|${MAX_RESPONSE_LENGTH:-4096}|g" "$TRAIN_OUT"
sed -i "s|OVERLONG_BUFFER_LEN_PLACEHOLDER|2048|g" "$TRAIN_OUT"
sed -i "s|TEST_FREQ_PLACEHOLDER|${TEST_FREQ:-2}|g" "$TRAIN_OUT"
sed -i "s|SAVE_FREQ_PLACEHOLDER|${SAVE_FREQ:-2}|g" "$TRAIN_OUT"

# --- Type B：注释控制 ---
# bit3: Prefix Cache
if [ "$bit_prefixCache" = "1" ]; then
    sed -i 's|^# \(actor_rollout_ref.rollout.enable_prefix_caching=True\)|\1|' "$TRAIN_OUT"
fi

# bit4: Recompute (3 行)
if [ "$bit_recompute" = "1" ]; then
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full\)|\1|' "$TRAIN_OUT"
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block\)|\1|' "$TRAIN_OUT"
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8\)|\1|' "$TRAIN_OUT"
fi

# bit5: Swap Optimizer
if [ "$bit_swapOpt" = "1" ]; then
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True\)|\1|' "$TRAIN_OUT"
fi

# bit6: VPP
if [ "$bit_vpp" = "1" ]; then
    sed -i 's|^# \(actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2\)|\1|' "$TRAIN_OUT"
fi

# 清理未启用的 Type B 注释行
sed -i '/^#.*\(+\?actor_rollout_ref\|enable_prefix_caching\)/d' "$TRAIN_OUT"

# --- SwanLab logger 处理 ---
if [ "$SWANLAB" = "no" ]; then
    sed -i "s|trainer.logger='\[\"console\",\"swanlab\"\]'|trainer.logger='[\"console\"]'|g" "$TRAIN_OUT"
fi

chmod +x "$TRAIN_OUT"
echo "[OK] 生成下层脚本: $TRAIN_OUT"

# ==========================================
# 输出摘要
# ==========================================
echo ""
echo "=========================================="
echo "脚本生成完成"
echo "=========================================="
echo "  上层脚本: $START_OUT"
echo "  下层脚本: $TRAIN_OUT"
echo "  特性掩码: $MASK"
echo "  SwanLab:  $SWANLAB"
if [ "$SWANLAB" = "yes" ]; then
    echo "  注意: SwanLab 凭证通过环境变量注入，未写入脚本文件"
    echo "  运行时需 export: SWANLAB_HOST, SWANLAB_API_KEY, SWANLAB_MODE, SWANLAB_WORKSPACE, SWANLAB_LOG_DIR, PROJECT_NAME"
fi
echo ""
echo "下一步："
echo "  1. docker cp $START_OUT <container>:/verl/start_verl.sh"
echo "  2. docker cp $TRAIN_OUT <container>:/verl/run_training.sh"
echo "  3. docker exec -it <container> bash"
echo "  4. cd /verl && bash start_verl.sh"
echo "=========================================="
