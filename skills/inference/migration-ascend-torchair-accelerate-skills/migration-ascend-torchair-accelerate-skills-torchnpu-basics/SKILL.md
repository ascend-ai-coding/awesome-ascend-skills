---
name: migration-ascend-torchair-accelerate-skills-torchnpu-basics
description: Provides foundational knowledge of torch_npu on Ascend NPU, including version compatibility, migration methods, device verification, CUDA-to-NPU API mapping, benchmark standards, and common troubleshooting. Invoke during steps 1-3 of the torchair migration workflow.
---

# Skill: torch_npu 基础知识

你是一名昇腾 NPU 推理工程师。本 Skill 提供在 Ascend NPU 上使用 torch_npu 进行模型迁移与 eager 模式推理的基础知识，服务于迁移步骤3（torch_npu 迁移）。

> 官方仓库：
> - torch_npu（Ascend Extension for PyTorch）：<https://gitcode.com/Ascend/pytorch>
> - 迁移调优指南（FrameworkPTAdapter）：<https://gitcode.com/Ascend/docs/tree/master/FrameworkPTAdapter/26.0.0>
>
> **⚠ 版本持续更新，必须实时查阅官方仓库获取最新文档。**

---

## 一、torch_npu 简介

torch_npu（Ascend Extension for PyTorch）是昇腾为 PyTorch 框架提供的 NPU 适配插件，其 Python 包名为 `torch_npu`，使 PyTorch 模型能够在 Ascend NPU 上运行。

版本分支命名规则：`{PyTorch版本}-{昇腾版本}`，例如 `v2.7.1-26.0.0` 表示匹配 PyTorch 2.7.1 的 torch_npu 26.0.0 版本。

### 1.1 版本配套关系

| torch_npu 版本 | GitCode 分支 | PyTorch 版本 | CANN 版本 | Python 版本 |
|---------------|-------------|-------------|-----------|------------|
| 2.10.0 | v2.10.0-26.0.0 | 2.10.0 | CANN 9.0.0 | 3.10~3.13 |
| 2.9.0.post2 | v2.9.0-26.0.0 | 2.9.0 | CANN 9.0.0 | 3.10~3.13 |
| 2.8.0.post4 | v2.8.0-26.0.0 | 2.8.0 | CANN 9.0.0 | 3.9~3.13 |
| 2.7.1.post4 | v2.7.1-26.0.0 | 2.7.1 | CANN 9.0.0 | 3.9~3.13 |
| 2.9.0 | v2.9.0-7.3.0 | 2.9.0 | CANN 8.5.0 | 3.10~3.13 |
| 2.8.0.post2 | v2.8.0-7.3.0 | 2.8.0 | CANN 8.5.0 | 3.9~3.13 |
| 2.7.1.post2 | v2.7.1-7.3.0 | 2.7.1 | CANN 8.5.0 | 3.9~3.13 |
| 2.6.0.post5 | v2.6.0-7.3.0 | 2.6.0 | CANN 8.5.0 | 3.9~3.11 |
| 2.1.0.post12 | v2.1.0-7.0.0 | 2.1.0 | CANN 8.1.RC1 | 3.8~3.11 |

> 完整版本配套表参见：<https://gitcode.com/Ascend/pytorch>

### 1.2 分支维护策略

| 状态 | 说明 |
|------|------|
| **开发** | 定期发布新版本，合入新特性。常规分支 6~12 个月，长期分支 12 个月 |
| **维护** | 常规分支维护 1 年，长期分支维护 3.5 年。仅修复重大 BUG，不合入新特性 |
| **EOL** | 分支不再接受任何修改 |

---

## 二、NPU 设备验证

### 2.1 基础检查

```bash
# 检查 NPU 驱动状态与设备信息
npu-smi info

# 检查 torch_npu 是否可用
python -c "
import torch
import torch_npu
print('PyTorch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('Device name:', torch.npu.get_device_name(0))
"
```

### 2.2 设备选择

```python
import torch

device = torch.device("npu:0")

# 或根据可用性自动选择
device = torch.device("npu" if torch.npu.is_available() else "cpu")
```

### 2.3 软件安装

torch_npu 的安装取决于 CANN 版本和 PyTorch 版本，具体参见官方安装指南：<https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/installation_guide/menu_installation_guide.md>。

---

## 三、模型迁移至 NPU

### 3.1 迁移方式概览

昇腾支持三种迁移方式（来源：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/mig_methods_comp.md>）：

| 方式 | 原理 | 适用场景 | 推荐度 |
|------|------|---------|--------|
| **自动迁移** | 添加 `from torch_npu.contrib import transfer_to_npu`，运行时自动将 CUDA 接口替换为 NPU 接口 | 模型主要使用标准 PyTorch 接口 | 推荐首选 |
| **工具迁移** | 使用 PyTorch GPU2Ascend 工具自动转换脚本，生成迁移报告后再运行 | 自动迁移与三方库冲突、需使用 `torch.jit.script` | 备选 |
| **手工迁移** | 手动对照 GPU 接口替换为 NPU 接口 | 前两种均不适用 | 兜底 |

> 迁移前置条件：模型脚本须能在 GPU 或 CPU 上正常运行。

### 3.2 自动迁移（推荐）

**PyTorch 2.5.1 及之后版本**（支持设备插件自动加载）：

```python
import torch
from torch_npu.contrib import transfer_to_npu  # 一行完成迁移

model = YourModel().to("npu")
# 后续代码无需修改
```

**PyTorch 2.4.0 及之前版本**（需显式导入 torch_npu）：

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

model = YourModel().to("npu")
```

**使用约束：**
- 自动迁移不支持 `channel_last` 特性，建议使用 `contiguous()` 替代
- 若原脚本中 `backend='nccl'` 自动替换为 `hccl`，后续代码中 `if backend == 'nccl'` 判断需手动改为 `'hccl'`
- 包含 `torch.jit.script` 的脚本与自动迁移冲突，需改用工具迁移
- 自动迁移与部分已适配的套件/三方库可能冲突，冲突时改用工具迁移

### 3.3 手工迁移（自动迁移不可用时）

需手动进行以下对照替换：

```python
# ========== 设备定义 ==========
device = torch.device("cuda")           → device = torch.device("npu")

# ========== Tensor 创建 ==========
x = torch.randn(3, 3).cuda()            → x = torch.randn(3, 3).to("npu")
x = torch.randn(3, 3, device="cuda")    → x = torch.randn(3, 3, device="npu")

# ========== 模型迁移 ==========
model.cuda()                            → model.to("npu")

# ========== CUDA API 对照 ==========
torch.cuda.is_available()               → torch.npu.is_available()
torch.cuda.device_count()               → torch.npu.device_count()
torch.cuda.synchronize()                → torch.npu.synchronize()
torch.cuda.current_device()             → torch.npu.current_device()
torch.cuda.set_device(i)                → torch.npu.set_device(i)
torch.cuda.empty_cache()                → torch.npu.empty_cache()
torch.cuda.Stream()                     → torch.npu.Stream()
torch.cuda.max_memory_allocated()       → torch.npu.max_memory_allocated()
torch.cuda.amp.autocast()               → torch.npu.amp.autocast()

# ========== 多卡通信 ==========
torch.distributed.init_process_group(   → torch.distributed.init_process_group(
    backend="nccl")                         backend="hccl")
```

> 完整手工迁移指南：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/manual_migration.md>

### 3.4 迁移后验证

```python
import torch
from torch_npu.contrib import transfer_to_npu

device = torch.device("npu:0")
model = YourModel().to(device)
model.eval()

# 确认所有参数在 NPU 上
for name, p in model.named_parameters():
    assert p.device.type == 'npu', f"Parameter '{name}' on {p.device}, expected NPU"

for name, b in model.named_buffers():
    assert b.device.type == 'npu', f"Buffer '{name}' on {b.device}, expected NPU"

# 小规模 forward 验证
test_input = torch.randn(1, ...).to(device)
with torch.no_grad():
    output = model(test_input)
print("Model forward on NPU: OK")
```

### 3.5 迁移问题排查

若迁移过程中出现 CUDA 接口报错，可能原因及处理：

| 情况 | 处理方式 |
|------|---------|
| 个别 API 不支持 | 使用 PyTorch Analyse 工具分析脚本，获取不支持 API 列表；参考《PyTorch 框架特性指南》的"自定义算子适配开发"章节适配，或至 [昇腾开源社区](https://gitcode.com/Ascend/pytorch/issues) 提 ISSUE |
| 整体迁移失败 | 查阅昇腾迁移调优文档仓库寻求答案：<https://gitcode.com/Ascend/docs/tree/master/FrameworkPTAdapter/26.0.0> |

---

## 四、混合精度（AMP）

Ascend NPU 支持自动混合精度（Automatic Mixed Precision），可提升推理性能并减少显存占用。

```python
from torch_npu.npu import amp

# 开启 autocast（自动选择 FP16/BF16）
with amp.autocast():
    output = model(input_data)

# 若需 GradScaler（训练场景）
scaler = amp.GradScaler()
```

> 详细 AMP 配置与精度调优参见：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/adaptation_introduction.md>

---

## 五、多卡与分布式推理

### 5.1 单机多卡

```python
import torch
import torch.distributed as dist

# 初始化 HCCL 通信
dist.init_process_group(backend="hccl")

# 指定当前 NPU
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
```

### 5.2 多机多卡

需额外设置 Master 地址和端口：

```python
dist.init_process_group(
    backend="hccl",
    init_method="tcp://{master_ip}:{master_port}",
    world_size=world_size,
    rank=rank
)
```

> 多卡迁移指南：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/multi_gpu.md>

---

## 六、降低 CPU 侧计算

CPU 上的 tensor 操作可能导致后续 torchair 编译时出现 Dynamo graph break、stream capture 失败（`aclrtMemcpy error 107030`）。

### 6.1 检查方法

```python
def analyze_device_placement(model, example_input):
    """分析模型参数的设备位置及热路径上可能的 CPU 操作"""
    import torch

    # 1. 参数和 buffer 设备检查
    cpu_params = []
    for name, p in model.named_parameters():
        if p.device.type != 'npu':
            cpu_params.append(('param', name, str(p.device)))
    for name, b in model.named_buffers():
        if b.device.type != 'npu':
            cpu_params.append(('buffer', name, str(b.device)))
    if cpu_params:
        print(f"WARNING: {len(cpu_params)} items not on NPU:")
        for kind, name, dev in cpu_params[:10]:
            print(f"  [{kind}] {name} on {dev}")

    # 2. 使用 profiler 追踪热路径上的 CPU 操作
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        with torch.no_grad():
            _ = model(example_input)
    cpu_events = [e for e in prof.function_events
                  if hasattr(e, 'device_type') and 'cpu' in str(e.device_type).lower()]
    if cpu_events:
        print(f"WARNING: {len(cpu_events)} CPU operations on hot path:")
        for e in cpu_events[:5]:
            print(f"  - {e.name} ({e.cpu_time_total:.3f}ms)")

    # 3. 检查常见的 CPU tensor 创建模式（代码审查）
    #    搜索 torch.ones/torch.zeros/torch.arange 不带 device=参数 后接 .to("npu")
    #    此类模式说明先在 CPU 创建后拷贝至 NPU，应在创建时直接指定 device='npu'
```

### 6.2 优化优先级

1. **直接创建在 NPU**：`torch.ones(..., device='npu')` 取代 `torch.ones(...).to('npu')`
2. **NPU 原生算子替换**：`torch.arange(..., device='npu')` 替代 CPU arange + .to
3. **避免 NPU→CPU 搬出**：尽量减少热路径上的 `.item()`、`.cpu()`、`.numpy()` 等操作
4. **无法迁移的 CPU 操作**（如 `.item()` 控制流）→ 规划为后续 torchair 编译时的 eager 区域

### 6.3 数据类型注意事项

- Ascend NPU 原生支持 FP16、BF16、FP32 等精度
- 注意模型代码中是否有隐式类型转换（如 `.float()`、`.half()`）
- 迁移时建议统一数据类型，避免频繁的设备间 + 类型间转换

---

## 七、性能测量规范

### 7.1 标准化 Benchmark 模板

所有性能测量必须遵循：**warmup=3, measure=5, 取中位数, 标注标准差**。严格禁止使用单次运行时间或无 warmup 的首次运行时间。

```python
import torch
import time
import numpy as np

WARMUP, MEASURE = 3, 5
device = torch.device("npu:0")

model = YourModel().to(device)
model.eval()
input_data = load_input().to(device)

# Warmup
for _ in range(WARMUP):
    _ = model.full_pipeline(input_data)
    torch.npu.synchronize()

# Measure
e2e_times = []
for _ in range(MEASURE):
    torch.npu.synchronize()
    t0 = time.time()
    _ = model.full_pipeline(input_data)
    torch.npu.synchronize()
    e2e_times.append((time.time() - t0) * 1000)  # ms

e2e_times.sort()
median_ms = e2e_times[len(e2e_times) // 2]
std_ms = np.std(e2e_times)
print(f"端到端: median={median_ms:.2f}ms, std={std_ms:.2f}ms")
```

### 7.2 接口一致性原则

步骤3记录的推理接口签名（如 `model.transcribe(audio_path)`）是后续所有性能对比的唯一基准。步骤4每次验证、最终对比必须使用同一接口。

**禁止行为：**
- 步骤3测 `model.full_pipeline(audio)`，步骤4改测 `model.encoder.forward(x)` 然后声称加速比
- 仅测量子模块 forward 的纯计算时间作为最终性能数据

---

## 八、常见问题排查

### Q1: `RuntimeError: Could not run 'torch_npu::npu_fusion_attention'`

**可能原因：** 输入 shape/dtype 不满足 NPU 融合注意力算子的约束。

**排查步骤：**
1. 检查 attention 输入的 shape 和 dtype
2. 设置环境变量 `TORCH_NPU_FUSED_ATTENTION=0` 禁用融合注意力
3. 切换至 math SDPA：`model.config._attn_implementation = "sdpa"`

### Q2: `no kernel registered` 或 `couldn't find appropriate backend` 错误

**可能原因：** 使用了 NPU 未实现的算子，或包含 CUDA 自定义扩展。

**排查步骤：**
1. 检查代码中是否有 CUDA 自定义扩展（`import _C`、`load_library`）
2. 查看完整错误 log 获取具体失败的函数名
3. 将该函数调用替换为纯 PyTorch 内置实现
4. 查询算子支持清单：<https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/native_apis/menu_pt_native_apis.md>

### Q3: `TBEException: GEGraph executor failed` 或 `EZ9999: Inner Error`

**可能原因：** CANN 版本与 torch_npu 版本不配套。

**排查步骤：**
1. 检查 CANN 版本与 torch_npu 版本是否匹配（参见 §1.1 版本配套表）
2. 尝试更换至配套的 CANN 版本

### Q4: `aclrtMemcpy error 107030` 或 `Event query device id failed`

**可能原因：** CPU→NPU memcpy 或 stream capture 失败，通常由热路径上的 CPU tensor 创建引起。

**排查步骤：**
1. 检查代码中是否有 `torch.*(...).to('npu')` 在热路径上
2. 改用 `torch.*(..., device='npu')` 直接创建在 NPU
3. 后续在 torchair 迁移中使用 split-compile 策略将 CPU 操作隔离在编译区域外

### Q5: `Can't get ascend_hal device count`

**可能原因：** Docker 未配置 `--privileged` 或 NPU 设备映射不完整。

**排查步骤：**
1. 重新启动容器，添加 `--privileged`
2. 添加设备映射：`--device=/dev/davinci0 --device=/dev/davinci_manager`
3. 挂载驱动目录：`-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`

### Q6: 模型在 CPU 上正常但 NPU 上精度差异大

**可能原因：** NPU 上的算子实现与 CPU/GPU 存在数值差异（尤其是 FP16/BF16 场景）。

**排查步骤：**
1. 对比 FP32 精度下的输出，确认是否为浮点精度导致
2. 查阅《PyTorch 训练模型迁移调优指南》的精度调优章节：<https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/overview.md>
3. 使用 `torch.npu.set_dump()` 或 PrecisionTool 进行逐层精度比对
