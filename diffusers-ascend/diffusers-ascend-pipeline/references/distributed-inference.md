# Diffusers 多卡并行推理（Ascend NPU）

本文给出在 Ascend NPU 上使用 Diffusers 多卡并行的实践路径。

## 1. 前置条件

- `torch` + `torch_npu` 可正常识别多卡 NPU。
- 通过 `torchrun` 拉起多进程。
- 使用 `torch.distributed` 初始化通信后端 `hccl`。

## 2. 版本分支

| Diffusers 版本 | 建议策略 |
|---|---|
| `>=0.36.0` | 优先使用 `ContextParallelConfig` / `ParallelConfig` |
| `<=0.35.2` | API 树中无 `api/parallel.md`，使用常规多进程并行（按请求切分）或升级版本 |

## 3. Context Parallel 最小示例（FLUX.1-dev）

```python
import os

import torch
import torch.distributed as dist
import torch_npu

from diffusers import ContextParallelConfig, DiffusionPipeline


def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.npu.set_device(local_rank)
    return local_rank, world_size


local_rank, world_size = setup_dist()
device = f"npu:{local_rank}"

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to(device)

# Attention backend 要先设置，再启用 context parallel
pipe.transformer.set_attention_backend("_native_npu")

cp_config = ContextParallelConfig(ulysses_degree=world_size)
pipe.transformer.enable_parallelism(config=cp_config)

image = pipe(
    prompt="a tiny astronaut hatching from an egg on the moon",
    guidance_scale=3.5,
    num_inference_steps=30,
).images[0]

if dist.get_rank() == 0:
    image.save("flux_cp_output.png")

dist.destroy_process_group()
```

启动命令（2 卡示例）：

```bash
torchrun --nproc_per_node=2 scripts/run_context_parallel.py --parallel-mode context
```

如果是在 CUDA 上，官方文档示例通常使用：

```python
pipe.transformer.set_attention_backend("_native_cudnn")
```

如果是在 Ascend NPU 上，应优先尝试：

```python
pipe.transformer.set_attention_backend("_native_npu")
```

## 4. Ulysses Attention

默认推荐优先尝试 Ulysses Attention：

```python
from diffusers import ContextParallelConfig

pipe.transformer.set_attention_backend("_native_npu")
pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))
```

对于更复杂场景，也可以组合 Ring 和 Ulysses：

```python
pipe.transformer.enable_parallelism(
    config=ContextParallelConfig(ring_degree=1, ulysses_degree=2)
)
```

如果当前运行栈出现 attention 内核限制（例如 `Native attention does not support return_lse=True`），可改用 `data` 并行模式（每卡一个进程）：

```bash
torchrun --nproc_per_node=2 scripts/run_context_parallel.py \
    --model ./fake_flux_dev \
    --prompt "a tiny astronaut hatching from an egg on the moon" \
    --parallel-mode data \
    --device-type npu --backend hccl \
    --steps 20 --output flux_dp_output.png
```

## 4. Docker 运行建议

- 容器需正确映射 Ascend 设备与驱动目录。
- 多卡通信建议使用 `--network=host`，避免容器网络导致通信初始化失败。
- 在容器内先验证：

```bash
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

## 5. 常见问题

### 5.1 进程组初始化失败

- 检查 `torch.distributed.init_process_group(backend="hccl")` 是否执行。
- 检查多进程环境变量（`RANK`, `WORLD_SIZE`, `LOCAL_RANK`）是否由 `torchrun` 正确注入。

### 5.2 版本中找不到并行 API

- 使用版本化 API 索引先确认 `api/parallel.md` 是否存在。
- 若不存在（如 `0.35.2`），改用常规多进程推理或升级到支持版本。

## 6. Data Parallel 退化路径

如果你明确想改用 Ring Attention，则可显式指定：

```python
pipe.transformer.enable_parallelism(config=ContextParallelConfig(ring_degree=2))
```

当你不希望直接改动 transformer 的 context parallel 配置，或者只是想先验证多卡 HCCL 拉起是否正常时，可以先用最简单的多进程 data parallel：

```bash
torchrun --nproc_per_node=2 scripts/run_context_parallel.py \
    --model ./fake_flux_dev \
    --prompt "a tiny astronaut hatching from an egg on the moon" \
    --parallel-mode data \
    --device-type npu --backend hccl \
    --attention-backend _native_npu
```

## 7. 参考

- `https://huggingface.co/docs/diffusers/v0.36.0/en/api/parallel`
- `https://huggingface.co/docs/diffusers/v0.36.0/en/training/distributed_inference`
- `https://huggingface.co/docs/diffusers/v0.36.0/en/optimization/attention_backends`
