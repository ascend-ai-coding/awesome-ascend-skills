---
name: vllm-ascend-server
description: "Deploy vLLM inference services on Ascend NPU servers with automatic model detection and optimized configuration. Supports local and remote deployment across bare metal, containers, and Docker images. Handles model discovery, quantization auto-detection, tensor parallelism configuration, graph/eager mode selection, and service health verification. Use when users need to: (1) Start or deploy vLLM server on NPU, (2) Launch LLM inference service, (3) Configure multi-card tensor parallel deployment, (4) Enable speculative decoding (Eagle) or quantization, (5) Run vllm offline batch inference, (6) Check or test vLLM service status."
---

# vLLM-Ascend Server Launcher

## Overview

This skill deploys vLLM inference services on Ascend NPU servers with automatic model detection, quantization handling, and performance optimization.

**Key Features:**

- Automatic model discovery and detection
- Quantization auto-detection (`quant_model_description.json`)
- Graph mode / Eager mode guidance
- Container deployment support
- Multi-card tensor parallelism

## Workflow Summary

```
Phase 0: Platform (Local/Remote, Bare metal/Container)
    ↓
Phase 1: Environment Check (NPU, vLLM, Memory)
    ↓
Phase 2: Model Discovery (Find models, detect quantization)
    ↓
Phase 3: Gather Requirements (Port, TP size, mode selection)
    ↓
Phase 4: Generate Config (Env vars, vLLM command)
    ↓
Phase 5: Execute (Deploy and start service)
    ↓
Phase 6: Verify (Health check, test inference)
```

**Detailed workflow:** [workflow-guide.md](references/workflow-guide.md)

## Phase 0: Platform Confirmation

### Location

```
1. Local - Deploy on this machine
2. Remote - Deploy via SSH (→ remote-server-guide skill)
```

### Platform

```
1. Bare metal (裸机) - Virtual environment on host
2. Existing container (已有容器) - Connect to running container
3. Docker image (镜像) - Create with npu-docker-launcher
```

**Docker image defaults:**

- Model mount: `-v <model-path>:/model`
- Network: `host` (default) or `bridge` with port mapping
- Port: 8000 (ask about mapping if bridge mode)

## Phase 1: Environment Checks

```bash
# NPU check
npu-smi info

# vLLM check
pip show vllm vllm-ascend

# Memory check
npu-smi info | grep -A 5 "Memory-Usage"
```

## Phase 1.5: NPU Availability Check

**Before deployment, verify selected NPU cards are not occupied:**

```bash
# Check NPU usage status
npu-smi info

# Check for running processes on specific cards
fuser -v /dev/davinci0 2>/dev/null && echo "Card 0 in use" || echo "Card 0 available"
fuser -v /dev/davinci1 2>/dev/null && echo "Card 1 in use" || echo "Card 1 available"

# Alternative: Check memory usage (high usage = occupied)
npu-smi info -t board | grep -E "NPU|Memory-Usage"
```

**If selected cards are occupied:**

```
## NPU Card Status

Card 0: ❌ In use (Memory: 28GB/32GB, PID: 12345)
Card 1: ✅ Available (Memory: 0GB/32GB)
Card 2: ✅ Available (Memory: 0GB/32GB)
Card 3: ❌ In use (Memory: 30GB/32GB, PID: 67890)

Selected cards [0,1] have conflicts:
- Card 0 is occupied by process 12345

Options:
1. Select different cards
2. Kill occupying process (with user confirmation)
3. Wait and retry

How to proceed? [1/2/3]
```

**Kill process (with confirmation):**

```bash
# Show what's using the card
ps aux | grep <PID>

# Confirm before killing
"Kill process <PID> (<process-name>)? [yes/no]"

# Kill if confirmed
kill -9 <PID>
```

**Detailed NPU check:** [workflow-guide.md](references/workflow-guide.md#npu-availability-check)

## Phase 2: Model Discovery

### Search Paths

```bash
/home/weights, /home/weight, /home/data*, /data*
```

Recursive search for `config.json` to find models.

### Quantization Detection

```bash
# Quantized model
[ -f "<model>/quant_model_description.json" ] → --quantization ascend

# Non-quantized model
[ ! -f "<model>/quant_model_description.json" ] → No param
```

**Critical:** Never add `--quantization ascend` for non-quantized models!

See [quantization.md](references/quantization.md) for details.

## Phase 3: Configuration Selection

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Mode | online | online/offline |
| Port | 8000 | Default for vLLM |
| NPU cards | 0 | 0,1 for TP2 |
| TP size | Auto | Based on model |

### Graph Mode Decision

| Scenario | Mode | Config |
|----------|------|--------|
| Production | Graph | `--no-enforce-eager` |
| Development | Eager | `--enforce-eager` |
| Debugging | Eager | `--enforce-eager` |

See [graph-mode.md](references/graph-mode.md) for details.

### TP Size by Model

| Model Size | TP | Cards |
|------------|----|----|
| ≤14B | 1 | 1 |
| 14B-70B | 2-4 | 2-4 |
| >70B | 4-8 | 4-8 |

## Phase 4: Generate Configuration

### Environment Variables

**Single card:**

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

**Multi-card:**

```bash
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
```

### vLLM Command Template

```bash
vllm serve /model/<model-name> \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --tensor-parallel-size <tp> \
  [QUANT_PARAM] \
  --gpu-memory-utilization 0.9 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**QUANT_PARAM:**

- Quantized model: `--quantization ascend`
- Non-quantized: (omit)

## Phase 5: Execute

Display generated config, confirm with user, then execute.

### Execution Methods by Platform

**Persistent session (tmux):** If you connected via tmux and are already inside the target environment (remote host / container / both), execute commands directly — same as bare metal.

**Stateless (SSH key / sshpass / paramiko / fabric):**

| Platform | Method |
|----------|--------|
| Bare metal | Execute directly in shell |
| Existing container | `docker exec` to run command |
| Remote | SSH → run command |
| Remote container | SSH → `docker exec -d` for background |

### Container Background Launch

**For containers, start vLLM in background with logging:**

```bash
# Inside container (bare metal / tmux already in container)
nohup vllm serve /model ... > /tmp/vllm.log 2>&1 &

# From host (stateless, background in container)
docker exec -d <container> bash -c 'cd /workspace && vllm serve /model ... 2>&1 | tee /tmp/vllm.log'

# Remote via SSH (stateless)
ssh user@host "docker exec -d <container> bash -c 'vllm serve /model ... 2>&1 | tee /tmp/vllm.log'"
```

### Check Service Status

**Inside container (bare metal / tmux already in container):**

```bash
ps aux | grep vllm
tail -50 /tmp/vllm.log
tail -f /tmp/vllm.log
```

**From host or remote (stateless, each command needs `docker exec`):**

```bash
docker exec <container> ps aux | grep vllm
docker exec <container> tail -50 /tmp/vllm.log
docker exec <container> tail -f /tmp/vllm.log
```

## Phase 6: Verify

```bash
# Health check
curl http://localhost:8000/health

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Quick Examples

### Quantized Model (Single Card)

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /model/Qwen3-8B-mxfp8 \
  --port 8000 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --async-scheduling
```

### Non-Quantized Model

```bash
vllm serve /model/Qwen3-8B \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --async-scheduling
  # NO --quantization param!
```

### Multi-Card TP2

```bash
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=1024

vllm serve /model/Qwen3-30B-mxfp8 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --gpu-memory-utilization 0.9
```

### Docker (Bridge Network)

```bash
docker run -it -d \
  --name vllm-server \
  --network bridge \
  -p 8000:8000 \
  -v /home/weights/Qwen3-8B:/model \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  vllm-ascend:latest

# Inside container
vllm serve /model --quantization ascend ...
```

## Decision Rules

### Quantization

| Detection | Action |
|-----------|--------|
| `quant_model_description.json` exists | Add `--quantization ascend` |
| File not found | No quantization param |

### Graph Mode

| Use Case | Mode |
|----------|------|
| Production | Graph (AclGraph) |
| First deployment | Eager → test → Graph |
| Errors in graph | Fall back to Eager |
| Debugging | Eager |

### Port

| Network | Port Config |
|---------|-------------|
| host | No mapping needed |
| bridge | Ask user for host port |

## Error Handling

| Error | Solution |
|-------|----------|
| OOM | Reduce `max_num_seqs` or `max_model_len` |
| Graph capture failed | Use `--enforce-eager` |
| Quantization error | Check if model is actually quantized |
| Port in use | Change port or kill process |
| HCCL timeout | Increase `HCCL_CONNECT_TIMEOUT` |
| Connection reset/timeout | Network issue, retry SSH connection |
| Container exits immediately | Check docker logs, verify mounts exist |

## Reference Files

### Configuration Guides

- [workflow-guide.md](references/workflow-guide.md) - Detailed workflow steps
- [quantization.md](references/quantization.md) - Quantization detection and config
- [graph-mode.md](references/graph-mode.md) - Graph vs Eager mode guide

### Technical References

- [models.md](references/models.md) - Model support matrix
- [model_configs/](references/model_configs/) - Per-model configurations
- [environment-variables.md](references/environment-variables.md) - Env var reference
- [parameters.md](references/parameters.md) - vLLM parameters
- [features.md](references/features.md) - Feature guides
- [launch_templates.md](references/launch_templates.md) - Script templates
- [profiling.md](references/profiling.md) - Profiling guide

## Related Skills

- **remote-server-guide** - SSH connection to remote servers
- **npu-docker-launcher** - Docker container creation

## External Documentation

- [Supported Models](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html)
- [Supported Features](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html)
- [Environment Variables](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/env_vars.html)
