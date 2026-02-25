#!/bin/bash
# msModelSlim One-Click Quantization Script
# Usage: ./quantize_model.sh --model_path /path/to/model --save_path /path/to/output [options]

set -e

# Default values
DEVICE="npu"
MODEL_TYPE=""
QUANT_TYPE="w8a8"
CONFIG_PATH=""
TRUST_REMOTE_CODE="True"
EXTRA_ARGS=""

# Help message
show_help() {
    echo "Usage: $0 --model_path PATH --save_path PATH [options]"
    echo
    echo "Required:"
    echo "  --model_path PATH       Path to original model"
    echo "  --save_path PATH        Path to save quantized model"
    echo
    echo "Options:"
    echo "  --device DEVICE         Device type (default: npu)"
    echo "                          Options: npu, npu:0,1,2,3, cpu"
    echo "  --model_type TYPE       Model type name (e.g., Qwen2.5-7B-Instruct)"
    echo "  --quant_type TYPE       Quantization type (default: w8a8)"
    echo "                          Options: w8a8, w4a8, w8a8c8, w8a8s"
    echo "  --config_path PATH      Custom YAML config file"
    echo "  --trust_remote_code     Trust remote code (default: True)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  # Basic W8A8 quantization"
    echo "  $0 --model_path /models/Qwen2.5-7B --save_path /output/Qwen2.5-7B-w8a8 \\"
    echo "      --model_type Qwen2.5-7B-Instruct --quant_type w8a8"
    echo
    echo "  # Multi-device quantization"
    echo "  $0 --model_path /models/Qwen2.5-72B --save_path /output/Qwen2.5-72B-w8a8 \\"
    echo "      --device npu:0,1,2,3 --model_type Qwen2.5-72B-Instruct --quant_type w8a8"
    echo
    echo "  # Using custom config"
    echo "  $0 --model_path /models/Qwen2.5-7B --save_path /output/Qwen2.5-7B-custom \\"
    echo "      --config_path assets/quant_config_w8a8.yaml --model_type Qwen2.5-7B-Instruct"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --quant_type)
            QUANT_TYPE="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --trust_remote_code)
            TRUST_REMOTE_CODE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    show_help
    exit 1
fi

if [ -z "$SAVE_PATH" ]; then
    echo "Error: --save_path is required"
    show_help
    exit 1
fi

if [ -z "$MODEL_TYPE" ]; then
    echo "Error: --model_type is required"
    show_help
    exit 1
fi

# Set environment for multi-device
if [[ "$DEVICE" == *"npu:"* ]]; then
    IFS=',' read -ra DEVICES <<< "${DEVICE#npu:}"
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICES[*]}"
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
    echo "Multi-device mode: ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
fi

# Build command
CMD="msmodelslim quant \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --device $DEVICE \
    --model_type $MODEL_TYPE \
    --trust_remote_code $TRUST_REMOTE_CODE"

# Add quant_type or config_path
if [ -n "$CONFIG_PATH" ]; then
    CMD="$CMD --config_path $CONFIG_PATH"
else
    CMD="$CMD --quant_type $QUANT_TYPE"
fi

# Add extra args
CMD="$CMD $EXTRA_ARGS"

echo "=========================================="
echo "  msModelSlim Quantization"
echo "=========================================="
echo "Model Path:    $MODEL_PATH"
echo "Save Path:     $SAVE_PATH"
echo "Device:        $DEVICE"
echo "Model Type:    $MODEL_TYPE"
[ -n "$CONFIG_PATH" ] && echo "Config Path:   $CONFIG_PATH" || echo "Quant Type:    $QUANT_TYPE"
echo "Command:       $CMD"
echo "=========================================="
echo

# Execute
echo "Starting quantization..."
$CMD

echo
echo "=========================================="
echo "  Quantization Complete!"
echo "=========================================="
echo "Output saved to: $SAVE_PATH"
echo
echo "To deploy with vLLM-Ascend:"
echo "  vllm serve $SAVE_PATH --quantization ascend --max-model-len 4096"
echo
