# Online Serving Templates

## Single-Card Server

```bash
#!/bin/bash

# Environment
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export ASCEND_RT_VISIBLE_DEVICES=0

# Configuration
MODEL_PATH="/home/data1/Qwen3-8B-mxfp8"
MODEL_NAME="Qwen3-8B"
PORT=8000

# Launch
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name ${MODEL_NAME} \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

## Multi-Card TP2 Server

```bash
#!/bin/bash

# Environment
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600

# Configuration
MODEL_PATH="/home/data1/Qwen3-30B-mxfp8"
MODEL_NAME="Qwen3-30B-TP2"
PORT=8000

# Launch (background)
nohup vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name ${MODEL_NAME} \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --no-enforce-eager \
    --additional-config '{"enable_cpu_binding":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > vllm.log 2>&1 &

echo "Server started. Logs: vllm.log"
```

## Multi-Card TP4 Server

```bash
#!/bin/bash

# Environment
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# Configuration
MODEL_PATH="/home/data1/Qwen3-72B-mxfp8"
MODEL_NAME="Qwen3-72B-TP4"
PORT=8000

# Launch
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name ${MODEL_NAME} \
    --trust-remote-code \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

## Non-Quantized Model

```bash
#!/bin/bash

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /home/data1/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --async-scheduling
    # NO --quantization param for non-quantized models!
```
