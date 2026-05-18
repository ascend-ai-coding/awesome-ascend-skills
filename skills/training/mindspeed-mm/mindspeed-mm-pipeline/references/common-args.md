# MindSpeed-MM Common Training Args Reference

This document covers common parameters for MindSpeed-MM training scripts, sourced from `docs/zh/pytorch/args_readme.md`. Applicable to all model types (VLM, Generative, Omni, Audio).

## GPT_ARGS (Core Training Parameters)

### Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tensor-model-parallel-size` | int | 1 | Tensor parallelism degree (TP). Splits attention heads and FFN across devices. Value must evenly divide the number of attention heads. |
| `--pipeline-model-parallel-size` | int | 1 | Pipeline parallelism degree (PP). Splits model layers across devices. Value must evenly divide the total number of layers. |
| `--context-parallel-size` | int | 1 | Context parallelism degree (CP). Splits long sequences along the sequence dimension across devices. Used for ultra-long sequence training. |
| `--expert-model-parallel-size` | int | 1 | Expert parallelism degree (EP). Distributes different experts across different devices in MoE models. |
| `--sequence-parallel` | flag | false | Enable sequence parallelism. Further splits the sequence dimension on top of TP to reduce activation memory. Requires TP > 1. |

**Parallelism Constraints**:
- TP x PP x DP = total number of devices
- DP (data parallelism) = total devices / (TP x PP), computed automatically
- global_batch_size must be divisible by DP x micro_batch_size

### Batch and Sequence

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--micro-batch-size` | int | required | Number of samples per device per step. Limited by NPU memory. |
| `--global-batch-size` | int | required | Global batch size. Gradient accumulation steps = global / (micro x DP). |
| `--seq-length` | int | required | Maximum training sequence length. Affects memory usage; must match the model's supported context length. |
| `--max-position-embeddings` | int | -- | Maximum position encoding length. Usually equal to or greater than seq-length. Some models (e.g., RoPE) do not require this. |

### Learning Rate and Optimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr` | float | required | Initial learning rate. Pre-training typically 1e-4 ~ 5e-4; fine-tuning typically 1e-5 ~ 5e-5. |
| `--min-lr` | float | 0 | Minimum learning rate. Lower bound for cosine decay. Usually set to 1/10 of lr. |
| `--lr-decay-style` | str | linear | Learning rate decay strategy. Options: `cosine`, `linear`, `constant`. `cosine` is recommended. |
| `--lr-warmup-fraction` | float | 0 | Learning rate warmup as a fraction of total steps. Typically 0.01 ~ 0.1. |
| `--lr-warmup-iters` | int | 0 | Learning rate warmup steps. Use either this or `lr-warmup-fraction`, not both. |
| `--weight-decay` | float | 0.01 | Weight decay coefficient. Pre-training commonly uses 0.1; fine-tuning commonly uses 0.01 ~ 0.1. |
| `--adam-beta1` | float | 0.9 | Adam optimizer beta1. |
| `--adam-beta2` | float | 0.999 | Adam optimizer beta2. 0.95 is recommended for bf16 training. |
| `--clip-grad` | float | 1.0 | Gradient clipping threshold. Prevents gradient explosion. |

### Precision and Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--bf16` | flag | false | Use BF16 mixed precision training. Recommended for Ascend NPU. |
| `--fp16` | flag | false | Use FP16 mixed precision training. May be needed for some models. Mutually exclusive with bf16. |
| `--initial-loss-scale` | int | 2^32 | Initial loss scale for FP16 mixed precision. Not needed for bf16 training (bf16 does not use loss scaling). |
| `--use-flash-attn` | flag | false | Enable FlashAttention. Significantly reduces attention memory usage; recommended. |
| `--use-fused-rmsnorm` | flag | false | Enable fused RMSNorm. Improves computation efficiency. |
| `--use-fused-swiglu` | flag | false | Enable fused SwiGLU. Improves FFN computation efficiency. |
| `--use-fused-rotary-pos-emb` | flag | false | Enable fused RoPE. Improves positional encoding computation efficiency. |

### Recomputation (Memory Optimization)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--recompute-granularity` | str | -- | Recomputation granularity. `full`: recompute entire Transformer layer; `selective`: recompute attention only. |
| `--recompute-method` | str | -- | Recomputation method. `uniform`: distribute evenly; `block`: distribute by blocks. |
| `--recompute-num-layers` | int | -- | Number of layers to recompute. Effective only with the `block` method. |
| `--use-distributed-optimizer` | flag | false | Use ZeRO-1 distributed optimizer. Shards optimizer states across devices to reduce memory. Recommended. |

### Checkpointing and Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save` | str | -- | Checkpoint save path. |
| `--load` | str | -- | Checkpoint load path. Should point to the converted MM-format weights during training. |
| `--save-interval` | int | -- | Save checkpoint every N steps. |
| `--log-interval` | int | 100 | Print training log every N steps. |
| `--eval-interval` | int | 1000 | Run evaluation every N steps. |
| `--no-load-optim` | flag | false | Do not load optimizer state when loading checkpoint. Recommended for fine-tuning. |
| `--no-load-rng` | flag | false | Do not load RNG state when loading checkpoint. Recommended for fine-tuning. |

### Training Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-iters` | int | -- | Total training steps. Use either this or `--train-epochs`. |
| `--train-epochs` | int | -- | Total training epochs. |
| `--finetune` | flag | false | Fine-tuning mode. Loads pre-trained weights but resets the training step counter. |
| `--no-save-optim` | flag | false | Do not save optimizer state in checkpoints. Saves disk space. |

### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data-path` | str | -- | Training data path prefix. In MindSpeed-MM, this is usually specified in the YAML configuration. |
| `--split` | str | 969,30,1 | Train/validation/test data split ratio. |
| `--tokenizer-type` | str | -- | Tokenizer type. Use `PretrainedFromHF` for HuggingFace models. |
| `--tokenizer-name-or-path` | str | -- | Tokenizer path. Points to the HF model directory (containing tokenizer.json). |

## MOE_ARGS (MoE Model Parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-experts` | int | -- | Total number of experts. E.g., DeepSeek-V2 uses 64. |
| `--moe-router-topk` | int | -- | Number of experts selected per token. Usually 2 or 6. |
| `--expert-model-parallel-size` | int | 1 | Expert parallelism degree. Distributes different experts across different devices. |
| `--moe-token-dispatcher-type` | str | -- | Token dispatch method. Options: `alltoall`, `allgather`. |
| `--moe-grouped-gemm` | flag | false | Enable grouped GEMM. Merges matrix multiplications from multiple experts. |
| `--moe-router-load-balancing-type` | str | -- | Load balancing type. Options: `aux_loss`, `none`. |
| `--moe-aux-loss-coeff` | float | 0 | Auxiliary load balancing loss coefficient. Usually 0.01. |

## OUTPUT_ARGS (Output Control Parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--log-interval` | int | 100 | Log printing interval (steps). |
| `--save-interval` | int | -- | Checkpoint save interval (steps). |
| `--eval-interval` | int | 1000 | Evaluation interval (steps). |
| `--eval-iters` | int | 100 | Number of steps per evaluation run. |
| `--tensorboard-dir` | str | -- | TensorBoard log directory. |
| `--tensorboard-log-interval` | int | 1 | TensorBoard logging interval. |
| `--wandb-project` | str | -- | Weights & Biases project name. |
| `--wandb-exp-name` | str | -- | Weights & Biases experiment name. |

## FSDP2 Configuration Parameters

When using the FSDP2 backend, specify a YAML configuration file via `--fsdp2-config-path`, replacing Megatron's TP/PP parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `--fsdp2-config-path` | str | FSDP2 configuration file path. When specified, Megatron parallelism parameters like TP/PP are ignored. |

FSDP2 configuration file example:

```yaml
fsdp:
  sharding_strategy: FULL_SHARD      # Sharding strategy
  mixed_precision:
    param_dtype: bfloat16             # Parameter precision
    reduce_dtype: float32             # Gradient reduction precision
  cpu_offload: false                  # Whether to offload parameters to CPU
```

## Environment Variables

| Variable | Description | Recommended Value |
|----------|-------------|-------------------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | Maximum device connections | `1` (must be set) |
| `HCCL_CONNECT_TIMEOUT` | HCCL communication connection timeout (seconds) | `1800` (recommended for multi-node training) |
| `PYTORCH_NPU_ALLOC_CONF` | NPU memory allocation strategy | `expandable_segments:True` |
| `HCCL_BUFFSIZE` | HCCL communication buffer size | `120` (MB, increase for large models) |
| `HCCL_OP_BASE_FFTS_MODE_ENABLE` | HCCL operation optimization | `TRUE` |
| `COMBINED_ENABLE` | Communication-computation overlap | `1` |
| `ASCEND_LAUNCH_BLOCKING` | Synchronous execution mode (for debugging) | `0` (production) / `1` (debugging) |
| `ASCEND_GLOBAL_LOG_LEVEL` | Ascend log level | `3` (ERROR) / `1` (INFO, for debugging) |
| `TASK_QUEUE_ENABLE` | Asynchronous task queue | `2` (recommended) |
| `MULTI_STREAM_MEMORY_REUSE` | Multi-stream memory reuse | `1` |

### Environment Variable Setup Example

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Performance optimization (optional)
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=1
```

## LoRA Fine-tuning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lora-r` | int | -- | LoRA rank. Common values: 8, 16, 32. Higher values increase fitting capacity but consume more memory. |
| `--lora-alpha` | float | -- | LoRA scaling coefficient. Usually set to 2x lora-r. |
| `--lora-target-modules` | list | -- | Modules to apply LoRA to. Common choices: `linear_qkv linear_proj linear_fc1 linear_fc2`. |
| `--lora-load` | str | -- | Path to load existing LoRA weights. Used for continued training or inference. |

## MindSpeed-MM Specific Configuration

MindSpeed-MM model configuration is primarily specified via YAML files (`model.json` or `model.yaml`), rather than entirely through command-line arguments.

### YAML Configuration Structure

```yaml
model:
  model_name: "qwen2.5vl"
  model_config:
    ...                         # Model architecture configuration
  load:
    ckpt_dir: "ckpt/mm_path/..."  # Converted MM-format weight path
  data:
    dataset_type: "mllm"          # VLM uses mllm; generative models use sora_dataset, etc.
    data_path: "dataset/train.json"
    ...
  training:
    ...                         # Training hyperparameters
```

> Paths and parameters in YAML are merged with command-line arguments. Command-line arguments take higher priority.

## Docker Runtime Considerations

When running MindSpeed-MM training inside Docker containers, the following settings are critical:

### Shared Memory (`--ipc=host` / `--shm-size`)

PyTorch DataLoader workers use shared memory (`/dev/shm`) to pass data between processes. Docker containers default to a 64 MB `/dev/shm`, which is far too small for multimodal training.

| Solution | Docker Flag | Description |
|----------|------------|-------------|
| Host IPC namespace (preferred) | `--ipc=host` | Shares the host's `/dev/shm`; no size limit from Docker |
| Explicit shm size | `--shm-size=16g` | Sets `/dev/shm` to 16 GB inside the container |
| Workaround | `--num-workers 0` in the training script | Disables worker processes; avoids shared memory entirely but slows data loading |

If neither `--ipc=host` nor `--shm-size` is set and `--num-workers > 0`, training will crash with `Bus error (core dumped)`.

### Privileged Mode for NPU Access

Ascend NPU devices require access to `/dev/davinci*` and related kernel interfaces. The simplest approach is:

```bash
docker run --privileged ...
```

Alternatively, map devices explicitly (see [ascend-docker](../../../ascend-docker/SKILL.md) for details).

### `MASTER_PORT` Conflicts

`torchrun` binds to `MASTER_PORT` (default 6000 or as set in the training script). If a previous training run was killed without cleanup, the port may still be occupied.

**Diagnosis**:
```bash
# Check if the port is in use
ss -tlnp | grep <MASTER_PORT>

# Find and kill stale torchrun processes
ps aux | grep torchrun | grep -v grep | awk '{print $2}' | xargs kill -9
```

**Prevention**: Change `MASTER_PORT` in the training script to an unused port (e.g., 6001, 29500).

## Parameter Tuning Recommendations

### Steps to Take When Running Out of Memory

1. Reduce `--micro-batch-size` (most direct)
2. Enable `--recompute-granularity full` (trade compute for memory)
3. Enable `--use-distributed-optimizer` (shard optimizer states)
4. Enable `--sequence-parallel` (requires TP > 1)
5. Increase PP (reduce per-device layer count)
6. Use LoRA instead of full-parameter fine-tuning

### Training Speed Optimization

1. Increase `--micro-batch-size` (improve compute efficiency)
2. Enable fused operators (`--use-flash-attn`, `--use-fused-rmsnorm`, `--use-fused-swiglu`)
3. Adjust TP/PP ratio (typically TP within a single node, PP across nodes)
4. Set performance environment variables (see environment variables section above)
