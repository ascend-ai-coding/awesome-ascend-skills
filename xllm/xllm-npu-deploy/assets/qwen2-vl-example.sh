#!/bin/bash
# Example: Deploy Qwen2.5-VL model

MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH="/path/to/test_image.jpg"

# Detect model type
python ../bin/detect_model.py "$MODEL_PATH" --recommend

# Deploy VLM
python ../bin/deploy_vlm.py \
    --model "$MODEL_PATH" \
    --devices npu:0 \
    --image "$IMAGE_PATH" \
    --prompt "描述这张图片" \
    --disable_prefix_cache \
    --disable_chunked_prefill \
    --enable_shm
