#!/bin/bash
# AISBench Quick Accuracy Test Script
# Usage: ./run_accuracy_test.sh <model> <dataset> [options]

set -e

usage() {
    echo "Usage: $0 <model> <dataset> [options]"
    echo ""
    echo "Arguments:"
    echo "  model    Model task name (e.g., vllm_api_general_chat)"
    echo "  dataset  Dataset task name (e.g., gsm8k_gen_4_shot_cot_chat_prompt)"
    echo ""
    echo "Options:"
    echo "  --host-ip IP         Inference service IP (default: localhost)"
    echo "  --host-port PORT     Inference service port (default: 8080)"
    echo "  --batch-size N       Concurrent requests (default: 1)"
    echo "  --max-out-len N      Max output tokens (default: 512)"
    echo "  --num-prompts N      Number of prompts to test (default: all)"
    echo "  --parallel N         Number of parallel tasks (default: 1)"
    echo "  --debug              Enable debug output"
    echo "  --dry-run            Show command without executing"
    echo ""
    echo "Examples:"
    echo "  $0 vllm_api_general_chat demo_gsm8k_gen_4_shot_cot_chat_prompt"
    echo "  $0 vllm_api_general_chat gsm8k_gen --host-ip 192.168.1.100 --host-port 8000"
    echo "  $0 vllm_api_general_chat mmlu_gen --parallel 4 --batch-size 8"
    exit 1
}

# Default values
HOST_IP="localhost"
HOST_PORT="8080"
BATCH_SIZE=1
MAX_OUT_LEN=512
NUM_PROMPTS=""
PARALLEL=1
DEBUG=""
DRY_RUN=false

# Parse arguments
if [[ $# -lt 2 ]]; then
    usage
fi

MODEL="$1"
DATASET="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case $1 in
        --host-ip)
            HOST_IP="$2"
            shift 2
            ;;
        --host-port)
            HOST_PORT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-out-len)
            MAX_OUT_LEN="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="--num-prompts $2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Find model config file
MODEL_CONFIG=$(ais_bench --models "$MODEL" --datasets dummy --search 2>/dev/null | grep -A1 "models" | grep -v "models" | awk '{print $2}' || echo "")

if [[ -z "$MODEL_CONFIG" ]]; then
    echo "Error: Could not find config for model '$MODEL'"
    echo "Run: ais_bench --models $MODEL --datasets dummy --search"
    exit 1
fi

# Update model config
echo "Updating model config: $MODEL_CONFIG"
echo "  host_ip: $HOST_IP"
echo "  host_port: $HOST_PORT"
echo "  batch_size: $BATCH_SIZE"
echo "  max_out_len: $MAX_OUT_LEN"

# Create sed commands for config update
sed -i.bak \
    -e "s/host_ip=\"[^\"]*\"/host_ip=\"$HOST_IP\"/" \
    -e "s/host_port=[0-9]*/host_port=$HOST_PORT/" \
    -e "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/" \
    -e "s/max_out_len=[0-9]*/max_out_len=$MAX_OUT_LEN/" \
    "$MODEL_CONFIG"

# Build command
CMD="ais_bench --models $MODEL --datasets $DATASET $NUM_PROMPTS --max-num-workers $PARALLEL $DEBUG"

echo ""
echo "Command: $CMD"
echo ""

if $DRY_RUN; then
    echo "[Dry run - not executing]"
    exit 0
fi

# Execute
echo "Starting accuracy evaluation..."
echo "========================================="
exec $CMD
