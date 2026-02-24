#!/bin/bash
# Batch ONNX to OM conversion helper
# Usage: ./convert_onnx.sh model.onnx [soc_version]

set -e

MODEL_PATH="$1"
SOC_VERSION="${2:-Ascend310P3}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model.onnx> [soc_version]"
    echo "Example: $0 resnet50.onnx Ascend310P3"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Get model info
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_PATH" .onnx)
OUTPUT_PATH="${MODEL_DIR}/${MODEL_NAME}_om"

echo "=== ONNX to OM Conversion ==="
echo "Input:  $MODEL_PATH"
echo "Output: $OUTPUT_PATH.om"
echo "Soc:    $SOC_VERSION"
echo

# Get input info using Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Analyzing model inputs..."
python3 "${SCRIPT_DIR}/get_onnx_info.py" "$MODEL_PATH"
echo

# Ask for confirmation
read -p "Proceed with conversion? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run ATC conversion
echo
echo "Running ATC conversion..."
atc \
    --model="$MODEL_PATH" \
    --framework=5 \
    --output="$OUTPUT_PATH" \
    --soc_version="$SOC_VERSION" \
    --log=info

if [ $? -eq 0 ]; then
    echo
    echo "✓ Conversion successful!"
    echo "Output: $OUTPUT_PATH.om"
    ls -lh "$OUTPUT_PATH.om"
else
    echo
    echo "✗ Conversion failed!"
    exit 1
fi
