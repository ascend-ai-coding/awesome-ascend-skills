# Deployment Workflow Guide

## Overview

This guide provides detailed instructions for each phase of vLLM deployment on Ascend NPU.

## Phase 0: Platform Setup

### Step 0.1: Remote vs Local

**Ask user:**

```
Where is the target server?
1. Local - This machine
2. Remote - Via SSH
```

**If Remote:**

1. Invoke **remote-server-guide** skill
2. Collect SSH credentials
3. Select SSH tool (ssh/sshpass/paramiko/fabric/ssh with tmux for interactive)
4. Test connection
5. Proceed to Step 0.2

### Step 0.2: Execution Platform

**Ask user:**

```
Where will vLLM be deployed?
1. Bare metal (裸机)
2. Existing container (已有容器)
3. Docker image (镜像)
```

**Platform Actions:**

| Platform | Action |
|----------|--------|
| Bare metal | Proceed to Phase 1 |
| Existing container | `docker exec -it <name> bash`, then Phase 1 |
| Docker image | Use npu-docker-launcher, mount model to `/model` |

**Docker Image Configuration:**

Default mounts:

```
-v <model-path>:/model
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver
```

Network options:

- `--network host` (default, no port mapping)
- `--network bridge -p <port>:8000` (isolated, port mapping)

## Phase 1: Environment Checks

### 1.1 NPU Availability

```bash
npu-smi info
ls -l /dev/davinci*
```

**Expected output:** Device list with status "OK"

**If failed:** Check driver installation, hardware connection

### 1.2 vLLM Installation

```bash
pip show vllm | grep Version
pip show vllm-ascend | grep Version
```

**If not installed:**

```bash
pip install vllm vllm-ascend
```

### 1.3 Memory Status

```bash
npu-smi info | grep -A 5 "Memory-Usage"
```

**Verify:** Sufficient free memory for model

## Phase 1.5: NPU Availability Check

After user selects NPU cards, verify they are not occupied.

### Check Commands

```bash
# Overall status
npu-smi info

# Check if specific card is in use
fuser -v /dev/davinci0 2>/dev/null && echo "In use" || echo "Available"

# Memory usage per card
npu-smi info -t board

# Find processes using NPU
ps aux | grep -E "python|vllm" | grep -v grep
```

### Status Detection

| Indicator | Status |
|-----------|--------|
| Memory > 1GB | Likely in use |
| `fuser -v` shows PID | In use |
| High temperature | Possibly in use |

### Handling Conflicts

**If selected cards are occupied:**

```
Card 0: ❌ In use (PID: 12345)
Card 1: ✅ Available

Options:
1. Select different cards
2. Kill process 12345 (with confirmation)
3. Wait and retry
```

**Kill process workflow:**

```bash
# Show process info
ps aux | grep <PID>

# Confirm with user
"Kill process <PID>? [yes/no]"

# Kill if confirmed
kill <PID>
```

## Phase 2: Model Discovery

### 2.1 Search Paths

```bash
SEARCH_PATHS=(
    "/home/weights"
    "/home/weight"
    "/home/data*"
    "/data*"
)

find "${SEARCH_PATHS[@]}" -name "config.json" -type f 2>/dev/null
```

### 2.2 Model Identification

For each `config.json` found:

1. Read `architectures` field
2. Estimate size from `hidden_size`
3. Check for `quant_model_description.json`

**Display to user:**

```
# | Model Name        | Path                    | Type
--|-------------------|-------------------------|--------
1 | Qwen3-8B         | /home/weights/Qwen3-8B  | Dense
2 | Qwen3-8B-mxfp8   | /home/data1/Qwen3-8B    | Quantized
```

### 2.3 Quantization Detection

```bash
[ -f "<model>/quant_model_description.json" ] && echo "quantized" || echo "non-quantized"
```

See [quantization.md](quantization.md) for details.

## Phase 3: Configuration Gathering

### Required Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Mode | online | online/offline |
| NPU cards | 0 | 0,1,2,3 for multi-card |
| Port | 8000 | Container: ask about mapping |
| TP size | Auto | Based on model size |
| Max length | 32768 | Or from model config |
| Max sequences | 256 | Reduce for memory |

### Quantization Parameter

| Model Type | Parameter |
|------------|-----------|
| Quantized | `--quantization ascend` (required) |
| Non-quantized | None |

### Graph Mode Selection

See [graph-mode.md](graph-mode.md) for guidance.

| Scenario | Mode |
|----------|------|
| Production | Graph mode (AclGraph) |
| Development | Eager mode |
| Debugging | Eager mode |

**Quick config:**

- Graph: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
- Eager: `--enforce-eager`

### Container Network (if Docker)

**Ask user:**

```
Network mode:
1. host (recommended) - Direct network access
2. bridge - Port mapping required

If bridge selected:
Host port for mapping: [8000]
```

## Phase 4: Configuration Generation

### 4.1 Environment Variables

**Single card:**

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

**Multi-card (TP > 1):**

```bash
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### 4.2 vLLM Command Template

```bash
vllm serve <model-path> \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name <name> \
  --trust-remote-code \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size <tp> \
  [QUANTIZATION_PARAM] \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**QUANTIZATION_PARAM:**

- Quantized: `--quantization ascend`
- Non-quantized: (empty)

## Phase 5: Execution

### 5.1 Display Configuration

```
## Generated Configuration

Model: <name>
Path: <path>
Mode: online/offline
Quantized: yes/no
TP Size: <n>
Port: 8000

### Environment Variables:
<env_vars>

### Launch Command:
<command>

Proceed? [yes/no/edit]
```

### 5.2 Execute by Platform

**Persistent session (tmux):** If you connected via tmux and are already inside the target environment (remote host / container / both), execute commands directly — same as bare metal.

**Stateless (SSH key / sshpass / paramiko / fabric):**

| Platform | Method |
|----------|--------|
| Bare metal | Execute directly in shell |
| Existing container (local) | `docker exec -it <container> bash`, then run command |
| Existing container (remote) | SSH → `docker exec -d` for background |
| Docker (create new) | Use npu-docker-launcher skill |

**Existing container (remote, stateless example):**

```bash
# Background launch with logging
ssh user@host "docker exec -d <container> bash -c 'cd /workspace && \
  export TASK_QUEUE_ENABLE=1 && \
  export ASCEND_RT_VISIBLE_DEVICES=0,1 && \
  vllm serve /workspace/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    ... 2>&1 | tee /tmp/vllm.log'"
```

### 5.3 Verify Startup

**Inside container (bare metal / tmux already in container):**

```bash
ps aux | grep vllm
tail -100 /tmp/vllm.log
tail -f /tmp/vllm.log
```

**From host or remote (stateless, each command needs `docker exec`):**

```bash
docker exec <container> ps aux | grep vllm
docker exec <container> tail -100 /tmp/vllm.log
docker exec <container> tail -f /tmp/vllm.log
```

**Success indicators in logs:**

- `"Starting vLLM API server"`
- `"Application startup complete"`
- `"Available routes are:"`

### 5.4 Common Execution Patterns

**Pattern 1: Remote SSH → Existing Container → Background vLLM**

```bash
# Step 1: Create container (if needed)
ssh user@host "docker run -it -d --name vllm-server --privileged --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home/weights/Qwen3-8B:/workspace/model \
  -w /workspace \
  quay.io/ascend/vllm-ascend:v0.13.0 /bin/bash"

# Step 2: Start vLLM in container background
ssh user@host "docker exec -d vllm-server bash -c ' \
  export TASK_QUEUE_ENABLE=1 && \
  export VLLM_USE_V1=1 && \
  export ASCEND_RT_VISIBLE_DEVICES=0,1 && \
  vllm serve /workspace/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --async-scheduling \
    --compilation-config \"{\\\"cudagraph_mode\\\": \\\"FULL_DECODE_ONLY\\\"}\" \
    2>&1 | tee /tmp/vllm.log'"

# Step 3: Monitor startup
ssh user@host "docker exec vllm-server tail -f /tmp/vllm.log"
```

**Pattern 2: Local Container → Interactive vLLM**

```bash
# Enter container interactively
docker exec -it vllm-server bash

# Inside container, run with nohup for background
nohup vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  ... > /tmp/vllm.log 2>&1 &

# Exit container, service keeps running
exit
```

## Phase 6: Verification

### Health Check

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}`

### Model List

```bash
curl http://localhost:8000/v1/models
```

### Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

## Error Recovery

| Error Phase | Common Issues | Recovery |
|-------------|---------------|----------|
| Environment | NPU not found | Check driver, devices |
| Model | Path not found | Verify model path |
| Config | OOM during init | Reduce max_num_seqs |
| Launch | Port in use | Kill existing process |
| Verify | Health check fails | Check logs, wait longer |

