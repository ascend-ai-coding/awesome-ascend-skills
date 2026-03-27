---
name: ai-for-science-ai4s-profiling
description: AI for Science 场景下的昇腾 NPU Profiling 采集与性能分析 Skill，用于在华为 Ascend NPU 上使用 torch_npu.profiler 采集 L0、L1、L2 级性能数据，分析训练或推理中的算子耗时、调用栈、内存与瓶颈，并指导后续调优。
keywords:
  - ai-for-science
  - profiling
  - torch_npu
  - performance
  - cann
  - ascend
---

# 昇腾 NPU Profiling 采集与性能分析 Skill

本 Skill 提供在华为昇腾 NPU 上采集性能 Profiling 数据的标准化流程，
支持 L0（最小膨胀）、L1（算子级）、L2（完整调用栈）三个采集级别，
覆盖训练和推理两种场景，以及多种训练框架的接入方式。

## 重要默认行为

1. **默认采集级别**：当用户只说"采集 profiling"而未明确指定采集等级时，
   默认采集 **NPU L0（最小膨胀）** 级别，即第 3.1 节的模板。
   仅当用户明确要求算子分析、调用栈、内存分析等更深层需求时，才升级到 L1 或 L2。

2. **不要修改训练/推理脚本中的 CUDA 代码**：在查看用户的训练或推理脚本时，
   即使代码中存在 `cuda`、`.cuda()`、`torch.device("cuda")` 等字样，
   也**不需要**帮用户改成 `npu` 相关写法。
   因为昇腾环境下通过 `import torch_npu` 配合自动迁移（`transfer_to_npu`），
   这些 CUDA 调用会在运行时自动转换为 NPU 调用，无需手动修改源码。
   Profiling 代码本身使用 `torch_npu.profiler` 是必要的，但训练/推理业务代码保持原样即可。

## 前置条件

执行 Profiling 前确认以下环境就绪：

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| CANN | ≥ 8.0（推荐 8.2+） |
| Python | 3.8 – 3.10 |
| PyTorch | 与 CANN 版本匹配 |
| torch_npu | 与 PyTorch 版本一致 |
| 磁盘空间 | 建议 ≥ 10GB 可用（L2 数据量较大） |

## 流程总览

```
0. 环境准备与校验
→ 1. 确定采集场景（训练 / 推理）
→ 2. 选择采集级别（L0 / L1 / L2）
→ 3. 植入 Profiling 代码
→ 4. 执行采集
→ 5. 数据解析与可视化
→ 6. 性能分析与调优建议
→ 7. GPU 对比采集（可选）
```

---

## 0. 环境准备与校验

### 0.1 CANN 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
npu-smi info
```

### 0.2 torch_npu 可用性验证

```bash
python3 -c "import torch; import torch_npu; print(torch.npu.is_available()); a = torch.randn(3,4).npu(); print(a.device)"
```

输出 `True` 和 `npu:0` 表示环境正常。

### 0.3 性能优化环境变量（采集前必设）

在执行 Profiling 采集命令前，建议设置以下环境变量以获得更准确的性能数据：

```bash
# CPU 绑核：将训练进程绑定到固定 CPU 核心，减少跨核调度带来的性能抖动
export CPU_AFFINITY_CONF=1

# 流水优化：开启 Host 侧任务下发与 Device 侧执行的异步流水，
# 减少 Host 等待 Device 的空闲时间，提升整体吞吐
# 0=关闭, 1=开启, 2=开启增强模式（推荐）
export TASK_QUEUE_ENABLE=2
```

**说明：**
- `CPU_AFFINITY_CONF=1` 避免操作系统在不同 CPU 核心间频繁调度训练线程，
  减少 cache miss 和 NUMA 跨节点访问，使采集到的性能数据更加稳定可复现。
- `TASK_QUEUE_ENABLE=2` 开启增强模式的 Host-Device 异步流水调度，
  可有效隐藏算子下发延迟。在 Profiling 时设置此项，可观测到更接近生产环境的真实性能表现。
- 这两个环境变量应在启动训练/推理脚本**之前**设置，与 `source set_env.sh` 一同执行。


---

## 1. 确定采集场景

### 1.1 训练场景

训练场景下使用 `schedule` 控制采集哪些迭代步，避免全量采集导致数据膨胀：

- `skip_first`: 跳过初始迭代（含编译、数据加载等非稳态开销）
- `wait`: 跳过后等待的步数
- `warmup`: 预热步数（采集但不记录）
- `active`: 实际记录的步数
- `repeat`: 重复采集的轮数

**典型配置：**
```python
schedule = torch_npu.profiler.schedule(
    wait=1, warmup=1, active=1, repeat=1, skip_first=20
)
```

此配置表示：跳过前 20 步 → 等待 1 步 → 预热 1 步 → 采集 1 步，重复 1 轮。

**注意：** 采集结束后会自动解析数据，解析后模型会继续跑剩余迭代。
如不需要继续训练，可在采集完成后及时停止。

### 1.2 推理场景

推理场景下同样需要在推理结束后调用 `prof.step()` 以触发 trace 数据导出：

```python
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./npu-profiling-inference")
) as prof:
    model(input_data)
    prof.step()
```

---

## 2. 采集级别说明

| 级别 | 说明 | 数据量 | 典型用途 |
|------|------|--------|----------|
| L0（默认） | 仅采集 NPU 活动，最小膨胀 | 小 | 快速定位热点、整体耗时分布 |
| L1 | 采集 CPU + NPU + 算子详情 | 中 | 分析 Cube/Vector/MatMul/Conv 算子耗时 |
| L2 | L1 + 调用栈 + 内存 | 大 | 深度分析调用链、内存瓶颈 |

---

## 3. 植入 Profiling 代码

### 3.1 L0 —— 最小膨胀采集（NPU Only）

仅采集 NPU 侧活动，数据膨胀最小，适合初步分析：

```python
import torch
import torch_npu

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        "./npu-profiling-L0"
    )
) as prof:
    for step, batch in enumerate(train_dataloader):
        train_one_step(batch)
        prof.step()
```

### 3.2 L1 —— 算子级采集（CPU + NPU + 算子详情）

采集 CPU 和 NPU 双侧活动，启用 `ProfilerLevel.Level1`，
可分析 Cube、Vector、MatMul、Conv 等算子结构的详细耗时：

```python
import torch
import torch_npu

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=False,
    record_shapes=True,
    profile_memory=False,
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        "./npu-profiling-L1"
    )
) as prof:
    for step, batch in enumerate(train_dataloader):
        train_one_step(batch)
        prof.step()
```

**参数说明：**
- `record_shapes=True`：采集 torch op 的 input shape 和 input type
- `with_stack=False`：不采集函数调用栈（减少空间占用）
- `profile_memory=False`：不采集内存相关数据
- `experimental_config`：NPU 专有参数，设置 profiler 级别

### 3.3 L2 —— 完整调用栈 + 内存采集

在 L1 基础上开启调用栈和内存采集，适合深度性能分析：

```python
import torch
import torch_npu

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        "./npu-profiling-L2"
    )
) as prof:
    for step, batch in enumerate(train_dataloader):
        train_one_step(batch)
        prof.step()
```

**注意：** `with_stack=True` 和 `profile_memory=True` 会显著增大输出数据量，
仅在需要分析调用栈或内存瓶颈时开启。

---

## 4. 框架适配

### 4.1 PyTorch Lightning 训练采集

对于使用 PyTorch Lightning 的项目，无法直接使用 `with` 语句包住训练循环，
需要通过 Callback 机制接入 Profiler：

```python
import torch
import torch_npu
import pytorch_lightning as pl


class NPUProfilingCallback(pl.Callback):
    """昇腾 NPU Profiling 回调，支持 L0/L1/L2 级别切换。"""

    def __init__(self, output_dir="./npu-profiling", level="L0",
                 skip_first=20, wait=1, warmup=1, active=1, repeat=1):
        super().__init__()
        self.output_dir = output_dir
        self.level = level
        self.skip_first = skip_first
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.prof = None

    def _build_profile_kwargs(self):
        kwargs = {
            "activities": [torch_npu.profiler.ProfilerActivity.NPU],
            "schedule": torch_npu.profiler.schedule(
                wait=self.wait, warmup=self.warmup,
                active=self.active, repeat=self.repeat,
                skip_first=self.skip_first,
            ),
            "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(
                self.output_dir
            ),
        }
        if self.level in ("L1", "L2"):
            kwargs["activities"].insert(
                0, torch_npu.profiler.ProfilerActivity.CPU
            )
            kwargs["record_shapes"] = True
            kwargs["experimental_config"] = (
                torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                )
            )
        if self.level == "L2":
            kwargs["with_stack"] = True
            kwargs["profile_memory"] = True
        return kwargs

    def on_train_start(self, trainer, pl_module):
        self.prof = torch_npu.profiler.profile(**self._build_profile_kwargs())
        self.prof.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.prof is not None:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, trainer, pl_module):
        if self.prof is not None:
            self.prof.stop()
            self.prof = None
```

**使用方式：** 将 Callback 加入 Trainer 的 callbacks 列表：

```python
trainer = pl.Trainer(
    callbacks=[
        ...,
        NPUProfilingCallback(
            output_dir="./npu-profiling-L1",
            level="L1",
            skip_first=20,
        ),
    ],
    ...
)
```

### 4.2 HuggingFace Transformers Trainer

对于使用 HuggingFace `Trainer` 的项目，可通过自定义 `TrainerCallback` 接入：

```python
import torch
import torch_npu
from transformers import TrainerCallback


class NPUProfilingTrainerCallback(TrainerCallback):

    def __init__(self, output_dir="./npu-profiling", skip_first=20):
        self.output_dir = output_dir
        self.skip_first = skip_first
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=1, warmup=1, active=1, repeat=1,
                skip_first=self.skip_first,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                self.output_dir
            ),
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.stop()
            self.prof = None
```

### 4.3 DeepSpeed 场景

使用 DeepSpeed 时，Profiling 代码仍然包在训练循环外层，但需注意：

- 确保 `prof.step()` 在 `model_engine.step()` 之后调用
- 多卡场景建议仅在 rank 0 上采集以减少数据量

```python
import torch
import torch_npu

if local_rank == 0:
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=1, repeat=1, skip_first=20
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            "./npu-profiling-deepspeed"
        ),
    )
    prof.start()

for step, batch in enumerate(train_dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
    if local_rank == 0:
        prof.step()

if local_rank == 0:
    prof.stop()
```

---


## 5. GPU 对比采集（可选）

当需要进行 GPU vs NPU 性能对比时，在 GPU 环境上使用 **相同 schedule 配置**
采集数据。GPU 采集使用原生 `torch.profiler`，与 NPU 的区别：

1. 使用 `torch.profiler.profile` 而非 `torch_npu.profiler.profile`
2. 使用 `torch.profiler.ProfilerActivity.CUDA` 而非 NPU
3. **不使用** `experimental_config` 参数（GPU 无此参数）

### 5.1 GPU 最小膨胀采集

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./gpu-profiling-L0"
    )
) as prof:
    for step, batch in enumerate(train_dataloader):
        train_one_step(batch)
        prof.step()
```

### 5.2 GPU 带 CPU 数据采集

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./gpu-profiling-with-cpu"
    )
) as prof:
    for step, batch in enumerate(train_dataloader):
        train_one_step(batch)
        prof.step()
```

---

## 6. 快速参考：采集代码模板选择

根据实际场景选择合适的模板：

| 场景 | 级别 | 模板位置 |
|------|------|----------|
| 训练 + 快速概览 | L0 | 第 3.1 节 |
| 训练 + 算子分析 | L1 | 第 3.2 节 |
| 训练 + 深度分析 | L2 | 第 3.3 节 |
| 推理 | L0 | 第 1.2 节 |
| PyTorch Lightning | L0/L1/L2 | 第 4.1 节 |
| HuggingFace Trainer | L0 | 第 4.2 节 |
| DeepSpeed 多卡 | L0 | 第 4.3 节 |
| GPU 对比 | - | 第 5 节 |

## 配套脚本

- Profiling 预检：`python scripts/validate_profiling_env.py --device npu:0 --output-dir ./profiling_output`

## 参考资料

- Profiling 采集与结论模板：[`references/analysis-checklist.md`](references/analysis-checklist.md)
