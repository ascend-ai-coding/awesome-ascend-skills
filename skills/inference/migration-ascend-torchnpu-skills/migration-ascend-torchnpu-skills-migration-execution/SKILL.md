---
name: migration-ascend-torchnpu-skills-migration-execution
description: Provides step-by-step code migration procedures from GPU/CPU to Ascend NPU. Invoke when performing actual code adaptation, interface replacement, or debugging NPU compatibility issues.
---

# Skill: 代码迁移执行

你是一位专注于PyTorch代码迁移至昇腾NPU的工程师。本Skill提供代码迁移的具体操作步骤、替换规则和验证方法。

> 官方参考文档：
> - 训练迁移指导：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/mig_methods_comp.md>
> - 亲和API替换（性能调优）：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/affinity_api_repl.md>
> - PyTorch原生接口支持度：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-10-0/overview.md>
> - torch_npu扩展接口：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/Extensionapi/docs/zh/extension_apis/overview.md>
> - torch_npu官方仓库：<https://gitcode.com/Ascend/pytorch>
> 
> **⚠ 版本持续更新，URL中的版本号（如26.0.0）随PyTorch框架适配版本迭代。MUST将URL中的版本号替换为实际使用的版本号，访问对应版本文档。**

<constraints>
- MUST 逐模块迁移并验证，NEVER一次性全量替换后统一验证
- MUST 对每个接口替换论证数学一致性
- NEVER 修改代码仓核心源码或第三方库，除非无替代方案
- MUST 保留原始代码的注释说明修改原因
- 查询接口支持度时，MUST根据实际使用的PyTorch版本查阅对应版本的接口文档
- MUST 优先使用镜像站或ModelScope获取模型/数据集资源，NEVER默认使用HuggingFace官网；仅在用户明确确认网络可达HuggingFace时方可使用HuggingFace官网
</constraints>

## 一、迁移方式选择

昇腾提供两种迁移方式：

### 1.1 自动迁移（推荐）

通过 `transfer_to_npu` 自动将运行时的 `torch.cuda` 等接口映射为 `torch.npu` 对应接口，无需手动修改代码。

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

**适用场景**：未使用CUDA高阶能力（如自定义算子、直接操作GPU显存等）的简单场景

**限制**：
- 仅支持PyTorch 1.8.1及以上版本
- 自动迁移适合没有使用CUDA高阶能力的简单场景，如果涉及自定义算子、主动申请GPU显存等操作，则需要额外进行手动迁移适配

### 1.2 手动迁移

手动修改代码中的CUDA相关调用为NPU对应接口。适用于自动迁移无法覆盖的场景。

**前提条件**：
- 要迁移的训练任务代码在GPU上多次训练稳定可收敛
- 已完成迁移环境准备

**约束和限制**：
- 安装插件后，大部分能力能够对应在GPU上的使用，但并不是所有行为和GPU上是一一对应的
- 在torch_npu下，当PyTorch版本低于2.1.0时，一个进程只能操作一张昇腾卡，不支持一个进程操作多卡的能力；在PyTorch 2.1.0及以上版本中torch_npu才支持一个进程中使用多张昇腾卡
- 基于PyTorch上的第三方开发库非常多，例如transformers、accelerate、deepspeed以及Megatron-LM等，这些三方库昇腾也做了类似PyTorch Adapter的适配插件库，可在gitcode.com/Ascend官方仓库按需使用。部分三方库例如最新版本deepspeed已原生支持NPU

## 二、手动迁移：设备映射替换

### 2.1 torch_npu初始化

在torch_npu安装后，该部分并没有直接植入到PyTorch中生效，需要用户显式调用。

```python
import torch_npu
```

调用后，前端会通过monkey-patch的方式注入到torch对象中，后端会注册NPU设备以及HCCL的参数面通信能力。

### 2.2 基础设备替换规则

| 原始代码 | 替换为 | 说明 |
|----------|--------|------|
| `torch.device('cuda:0')` | `torch.device('npu:0')` | 设备指定 |
| `torch.cuda.set_device(0)` | `torch.npu.set_device(0)` | 设置当前设备 |
| `torch.cuda.is_available()` | `torch.npu.is_available()` | 设备可用性检查 |
| `torch.cuda.device_count()` | `torch.npu.device_count()` | 设备数量查询 |
| `torch.cuda.current_device()` | `torch.npu.current_device()` | 获取当前设备 |
| `.cuda()` | `.npu()` | Tensor/Model设备转移 |
| `.to('cuda:0')` | `.to('npu:0')` | Tensor设备转移 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 设备同步 |
| `torch.cuda.memory_allocated()` | `torch.npu.memory_allocated()` | 显存查询 |
| `torch.cuda.max_memory_allocated()` | `torch.npu.max_memory_allocated()` | 显存峰值查询 |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` | 显存缓存清理 |

### 2.3 混合精度替换规则

模型从GPU适配到NPU时，需要将代码torch.cuda.amp修改为torch_npu.npu.amp。

| 原始代码 | 替换为 |
|----------|--------|
| `torch.cuda.amp.autocast()` | `torch.npu.amp.autocast()` |
| `torch.cuda.amp.GradScaler()` | `torch.npu.amp.GradScaler()` |

> 昇腾NPU的Cube计算单元仅支持FP16矩阵运算，FP32的矩阵运算无法调用Cube高算力。整网使用FP16数值范围小，容易导致梯度消失，因此混合精度训练（AMP）在NPU上是必要操作，不是可选项。

### 2.4 分布式训练替换规则

昇腾只支持DDP模式，不支持DP模式。

| 原始代码 | 替换为 | 说明 |
|----------|--------|------|
| `backend="nccl"` | `backend="hccl"` | 集合通信后端 |
| `torch.nn.DataParallel` | `torch.nn.parallel.DistributedDataParallel` | NPU不支持DP，MUST改为DDP |
| `torch.cuda.default_generators` | `torch_npu.npu.default_generators` | 随机数生成器 |

分布式初始化方式：

```python
# 方式1：通过接口设置
dist.init_process_group(
    backend='hccl',
    init_method="tcp://127.0.0.1:29688",
    world_size=args.world_size,
    rank=args.rank
)

# 方式2：通过环境变量设置
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29688'
dist.init_process_group(
    backend='hccl',
    world_size=args.world_size,
    rank=args.rank
)
```

### 2.5 NPU不支持的接口

以下接口在NPU上不支持或行为不同，MUST手动处理：

- `torch.nn.DataParallel`：不支持，MUST改为DDP
- `amp_C`模块：不支持，MUST手动删除
- `torch.cuda.get_device_capability`：迁移后返回`None`，如遇报错需手动修改为固定值
- `torch.cuda.get_device_properties`：返回值不包含`minor`和`major`属性，需注释掉相关代码

## 三、接口等价替换

### 3.1 替换流程

1. 从接口分析报告中筛选出NPU不支持的接口
2. 查阅昇腾官方接口支持度文档，确认该接口是否已适配：
   - PyTorch原生接口支持度：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-10-0/overview.md>
   - torch_npu扩展接口：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/Extensionapi/docs/zh/extension_apis/overview.md>
3. 如无官方适配方案，按以下优先级寻找替代：
   - 优先级1：使用torch_npu提供的扩展接口
   - 优先级2：使用等价的PyTorch原生接口组合
   - 优先级3：使用自定义实现（MUST论证数学一致性）

### 3.2 接口支持度查询

昇腾官方提供了按PyTorch版本区分的接口支持度清单，MUST根据实际使用的PyTorch版本查阅对应文档：

| PyTorch版本 | 接口文档路径 |
|------------|------------|
| 2.10.0 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-10-0/overview.md` |
| 2.9.0 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/overview.md` |
| 2.8.0 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-8-0/overview.md` |
| 2.7.1 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/overview.md` |
| 2.6.0 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-6-0/overview.md` |
| 2.1.0 | `apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-1-0/overview.md` |

> 以上路径均基于文档根目录 `https://www.hiascend.com/document/detail/zh/Pytorch/2600/` 下。版本迭代较快，MUST以官网最新版本文档为准。

**查询方法**：
1. 确定当前项目使用的PyTorch版本
2. 打开对应版本的接口支持度文档
3. 搜索目标接口名称，查看是否在支持列表中
4. 如在支持列表中，确认接口的行为差异说明
5. 如不在支持列表中，查阅torch_npu扩展接口文档是否有替代

### 3.3 数学一致性论证模板

对每个等价替换，MUST按以下模板论证：

```markdown
### 接口替换：[原始接口] → [替换接口]

**数学定义**：
- 原始接口：y = f(x)，数学定义式为...
- 替换接口：y = g(x)，数学定义式为...

**等价性证明**：
- 当满足条件...时，f(x) = g(x)
- 证明过程：...

**数值验证**：
- 测试输入：生成N组随机输入
- 最大绝对误差：{value}
- 最大相对误差：{value}
- 结论：误差在可接受范围内 / 需要进一步处理
```

## 四、调试技巧

### 4.1 定位不支持接口

```python
try:
    output = model(input_npu)
except RuntimeError as e:
    if "not implemented for" in str(e) and "NPU" in str(e):
        print(f"不支持的接口: {e}")
```

### 4.2 逐层验证

```python
def verify_layer_by_layer(model, sample_input, device="npu"):
    x = sample_input.to(device)
    for name, layer in model.named_children():
        x = layer(x)
        print(f"Layer {name}: output shape={x.shape}, "
              f"mean={x.mean().item():.6f}, std={x.std().item():.6f}")
```

### 4.3 精度对比工具

```python
def compare_outputs(cpu_output, npu_output, atol=1e-5, rtol=1e-3):
    cpu_np = cpu_output.detach().cpu().numpy()
    npu_np = npu_output.detach().cpu().numpy()
    max_abs_diff = abs(cpu_np - npu_np).max()
    max_rel_diff = (abs(cpu_np - npu_np) / (abs(cpu_np) + 1e-8)).max()
    print(f"最大绝对误差: {max_abs_diff:.2e}")
    print(f"最大相对误差: {max_rel_diff:.2e}")
    assert max_abs_diff < atol or max_rel_diff < rtol, "精度超出允许范围"
```

## 五、NPU亲和性调优

> 完成基础迁移（设备映射、接口替换）后，模型已可在NPU上正常运行。但要充分发挥NPU硬件性能，需要进行亲和性调优。以下方法来源于昇腾官方文档：
> <https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/affinity_api_repl.md>
>
> **⚠ 版本持续更新，亲和API列表和用法可能随版本变化，MUST实时查阅上述官网链接获取最新信息。**

### 5.1 亲和API替换概述

在基础迁移完成后，将代码中的部分原生PyTorch接口替换为昇腾NPU亲和接口，可进一步提升训练效率。亲和接口针对NPU硬件架构进行了深度优化，能显著减少算子调度开销、提升内存访问效率。

### 5.2 融合转置操作：`npu_confusion_transpose`

`torch_npu.npu_confusion_transpose` 用于融合 `transpose` + `view`（或 `reshape`）组合操作，减少中间张量的产生，降低显存占用和算子调度开销。

**使用场景**：当模型代码中频繁出现 `transpose` 后紧跟 `view/reshape` 的模式时（常见于多头注意力等模块），使用此API可提升性能。

```python
# torch原生写法
import torch
data = torch.rand(64, 3, 64, 128).npu()
result = data.transpose(1, 2).view(64, 64, -1)

# torch_npu亲和写法
import torch_npu
data = torch.rand(64, 3, 64, 128).npu()
result = torch_npu.npu_confusion_transpose(data, perm=[0, 2, 1, 3], shape=[64, 64, -1])
```

### 5.3 NPU亲和优化器

使用 `torch_npu` 提供的融合优化器替换原生PyTorch优化器。NPU亲和优化器通过算子融合减少参数更新时的多次kernel launch开销。

**前提条件**：在替换优化器前，需确保已完成NPU亲和优化器的导入。

**常用NPU亲和优化器**（以PyTorch 2.6.0版本为例，新版本文档可能新增更多）：

| 原生PyTorch优化器 | NPU亲和优化器 | 说明 |
|------------------|--------------|------|
| `torch.optim.SGD` | `torch_npu.optim.NpuFusedSGD` | 融合SGD优化器 |
| `torch.optim.AdamW` | `torch_npu.optim.NpuFusedAdamW` | 融合AdamW优化器 |
| `torch.optim.Adam` | `torch_npu.optim.NpuFusedAdam` | 融合Adam优化器 |

```python
# 原生PyTorch写法
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# torch_npu亲和写法
import torch_npu
optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)
```

> **注意事项**：
> - 并非所有优化器都有NPU融合版本，MUST查阅官方文档确认当前版本支持的融合优化器列表
> - 融合优化器的超参数与原生优化器基本一致，但部分特殊参数可能有差异，需对照官方文档确认

### 5.4 NPU亲和梯度裁剪

在替换为NPU亲和梯度裁剪API之前，MUST确保已使用NPU亲和优化器（参见5.3节）。NPU亲和梯度裁剪与NPU亲和优化器配合使用，可进一步减少参数更新的kernel launch次数。

```python
# 原生PyTorch写法
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... loss计算 ...
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# torch_npu亲和写法
import torch_npu
optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)
# ... loss计算 ...
loss.backward()
torch_npu.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

> **注意**：梯度裁剪API的具体接口名可能随版本变化，MUST查阅当前版本官方文档确认。

### 5.5 多流并行优化（补充）

昇腾NPU支持多流（multi-stream）并行执行，可通过显式管理计算流来提升并发度：

```python
# 创建多个stream并行执行不依赖的操作
stream1 = torch.npu.Stream()
stream2 = torch.npu.Stream()

with torch.npu.stream(stream1):
    output1 = model_part1(data1)
with torch.npu.stream(stream2):
    output2 = model_part2(data2)

torch.npu.synchronize()  # 等待所有stream完成
```

> 多流并行适用于模型中的独立计算分支，能有效隐藏数据搬运延迟。使用前需确保各分支之间无数据依赖关系。

### 5.6 性能调优查阅路径

亲和性调优属昇腾性能调优体系的一部分，完整的性能调优方法和最新亲和API列表MUST查阅昇腾官方文档：

- **PyTorch训练模型迁移调优总览**：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/mig_methods_comp.md>
- **亲和API替换（含最新API列表）**：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/affinity_api_repl.md>
- **性能调优方法总览**：<https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/npu_suit_optimize.md>

> **⚠ 以上URL中的版本号（如26.0.0）对应当前PyTorch框架适配版本。版本会持续更新，MUST将URL中的版本号替换为实际使用的版本号，获取匹配的文档内容。**

<verification>
完成迁移后逐项检查：
- [ ] torch_npu已正确初始化（`import torch_npu`）
- [ ] 所有 `cuda` 字符串已替换为 `npu`（含import语句）
- [ ] DP模式已改为DDP模式
- [ ] 分布式后端已从nccl改为hccl
- [ ] 所有不支持接口已有替换方案
- [ ] 每个替换方案已验证数学一致性
- [ ] 混合精度已从torch.cuda.amp改为torch.npu.amp
- [ ] 训练循环可在NPU上完整运行
- [ ] 模型可正常保存和加载
- [ ] 数据加载管道无报错
- [ ] （可选）已评估NPU亲和优化器替换可行性
- [ ] （可选）已评估亲和API替换可行性（如npu_confusion_transpose）
- [ ] （可选）已查阅最新官方文档确认亲和API列表
</verification>
