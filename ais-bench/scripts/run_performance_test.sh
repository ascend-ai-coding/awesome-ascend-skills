#!/bin/bash
# AISBench Quick Performance Test Script
# Usage: ./run_performance_test.sh <model> <dataset> [options]

set -e

usage() {
    echo "Usage: $0 <model> <dataset> [options]"
    echo ""
    echo "Arguments:"
    echo "  model    Model task name (e.g., vllm_api_general_chat)"
    echo "  dataset  Dataset task name or 'custom' for custom dataset"
    echo ""
    echo "Options:"
    echo "  --host-ip IP         Inference service IP (default: localhost)"
    echo "  --host-port PORT     Inference service port (default: 8080)"
    echo "  --concurrency N      Concurrent requests (default: 1)"
    echo "  --num-prompts N      Number of requests (default: all)"
    echo "  --custom-dataset PATH  Path to custom dataset file"
    echo "  --output-dir DIR     Output directory (default: outputs/perf)"
    echo "  --debug              Enable debug output"
    echo "  --dry-run            Show command without executing"
    echo ""
    echo "Examples:"
    echo "  $0 vllm_api_general_chat sharegpt --concurrency 100"
    echo "  $0 vllm_api_general_chat custom --custom-dataset ./my_data.jsonl"
    echo "  $0 mindie_api_general mtbench --host-ip 192.168.1.100 --num-prompts 1000"
    exit 1
}

# Default values
HOST_IP="localhost"
HOST_PORT="8080"
CONCURRENCY=1
NUM_PROMPTS=""
CUSTOM_DATASET=""
OUTPUT_DIR="outputs/perf"
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
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="--num-prompts $2"
            shift 2
            ;;
        --custom-dataset)
            CUSTOM_DATASET="--custom-dataset-path $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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

# Find and update model config
MODEL_CONFIG=$(ais_bench --models "$MODEL" --datasets dummy --search 2>/dev/null | grep -A1 "models" | grep -v "models" | awk '{print $2}' || echo "")

if [[ -z "$MODEL_CONFIG" ]]; then
    echo "Error: Could not find config for model '$MODEL'"
    exit 1
fi

echo "Updating model config: $MODEL_CONFIG"
sed -i.bak \
    -e "s/host_ip=\"[^\"]*\"/host_ip=\"$HOST_IP\"/" \
    -e "s/host_port=[0-9]*/host_port=$HOST_PORT/" \
    -e "s/batch_size=[0-9]*/batch_size=$CONCURRENCY/" \
    "$MODEL_CONFIG"

# Build command
CMD="ais_bench --models $MODEL --datasets $DATASET $CUSTOM_DATASET $NUM_PROMPTS --mode perf --work-dir $OUTPUT_DIR $DEBUG"

echo ""
echo "========================================="
echo "AISBench Performance Test"
echo "========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Concurrency: $CONCURRENCY"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Command: $CMD"
echo ""

if $DRY_RUN; then
    echo "[Dry run - not executing]"
    exit 0
fi

# Execute
echo "Starting performance evaluation..."
exec $CMD
