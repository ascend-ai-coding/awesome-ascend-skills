# NPU API 映射表

本文档列出 PyTorch CUDA API 与 NPU API 的对应关系。

## 使用说明

在使用支持的 CUDA 接口时，需要将 API 名称中的 `cuda` 变换为 NPU 形式：
- `torch.cuda.xxx` → `torch_npu.npu.xxx`
- 或 `torch.cuda.xxx` → `torch.npu.xxx`

两种调用方式功能一致。

> 💡 **进阶技巧**：使用 `from torch_npu.contrib import transfer_to_npu` 可以自动完成以下大部分替换，无需手动修改代码。只有当自动替换失败时才需要手动映射。

---

## 设备参数批量替换规则

`transfer_to_npu` 实现了智能设备参数识别，手动迁移时可参考：

| 参数类型 | 原值 | 替换为 | 示例 |
|----------|------|--------|------|
| 字符串设备 | `"cuda"` | `"npu"` | `device="cuda"` → `device="npu"` |
| 字符串设备+序号 | `"cuda:0"` | `"npu:0"` | `device="cuda:0"` → `device="npu:0"` |
| torch.device | `torch.device("cuda")` | `torch.device("npu")` | - |
| torch.device带序号 | `torch.device("cuda:1")` | `torch.device("npu:1")` | - |
| 纯整数设备 | `0` (作为device参数) | `"npu:0"` | `device=0` → `device="npu:0"` |
| 设备字典 | `{"cuda": "0"}` | `{"npu": "0"}` | `{"cuda:1": "1GB"}` → `{"npu:1": "1GB"}` |

### 涉及的关键参数名
`device`, `device_type`, `map_location`, `device_ids`

---

## 设备管理

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.is_available()` | `torch_npu.npu.is_available()` | ✅ | |
| `torch.cuda.device_count()` | `torch_npu.npu.device_count()` | ✅ | |
| `torch.cuda.current_device()` | `torch_npu.npu.current_device()` | ✅ | |
| `torch.cuda.set_device(x)` | `torch_npu.npu.set_device(x)` | ✅ | |
| `torch.cuda.get_device_name(x)` | `torch_npu.npu.get_device_name(x)` | ✅ | |
| `torch.cuda.get_device_properties(x)` | `torch_npu.npu.get_device_properties(x)` | ⚠️ | 仅支持部分属性 |
| `torch.cuda.get_device_capability()` | - | ❌ | NPU 无此概念 |
| `torch.cuda.init()` | `torch_npu.npu.init()` | ✅ | |

---

## 内存管理

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.memory_allocated()` | `torch_npu.npu.memory_allocated()` | ✅ | |
| `torch.cuda.memory_reserved()` | `torch_npu.npu.memory_reserved()` | ✅ | |
| `torch.cuda.empty_cache()` | `torch_npu.npu.empty_cache()` | ✅ | |
| `torch.cuda.mem_get_info()` | `torch_npu.npu.mem_get_info()` | ✅ | |
| `torch.cuda.max_memory_allocated()` | `torch_npu.npu.max_memory_allocated()` | ✅ | |
| `torch.cuda.reset_peak_memory_stats()` | `torch_npu.npu.reset_peak_memory_stats()` | ✅ | |

---

## Stream 和 Event

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.Stream` | `torch_npu.npu.Stream` | ✅ | |
| `torch.cuda.default_stream` | `torch_npu.npu.default_stream` | ✅ | |
| `torch.cuda.current_stream` | `torch_npu.npu.current_stream` | ✅ | |
| `torch.cuda.synchronize()` | `torch_npu.npu.synchronize()` | ✅ | |
| `torch.cuda.Event` | `torch_npu.npu.Event` | ✅ | |

---

## 随机种子

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.manual_seed(x)` | `torch_npu.npu.manual_seed(x)` | ✅ | |
| `torch.cuda.manual_seed_all(x)` | `torch_npu.npu.manual_seed_all(x)` | ✅ | |
| `torch.cuda.get_rng_state()` | `torch_npu.npu.get_rng_state()` | ✅ | |
| `torch.cuda.set_rng_state(x)` | `torch_npu.npu.set_rng_state(x)` | ✅ | |

---

## 混合精度 (AMP)

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.amp.GradScaler` | `torch_npu.npu.amp.GradScaler` | ✅ | |
| `torch.cuda.amp.autocast` | `torch_npu.npu.amp.autocast` | ✅ | |

**精度配置：**
```python
# NPU 启用 HF32
torch_npu.npu.matmul.allow_hf32 = True
torch_npu.npu.conv.allow_hf32 = True

# CUDA 启用 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 分布式训练

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `backend="nccl"` | `backend="hccl"` | ✅ | 必须替换 |
| `torch.cuda.nccl.*` | - | ❌ | 不支持 |

### 分布式相关自动替换（transfer_to_npu 逻辑）

`transfer_to_npu` 还会自动处理这些分布式场景：

| 原代码 | 替换为 | 说明 |
|--------|--------|------|
| `torch.distributed.init_process_group(backend="nccl", ...)` | `backend="hccl"` | 自动替换 |
| `torch.distributed.is_nccl_available()` | `torch.distributed.is_hccl_available()` | 函数替换 |
| `ProcessGroup._get_backend(device)` | 自动识别 npu 设备 | 方法拦截 |
| `torch.distributed.fsdp.FullyShardedDataParallel(...)` | 自动处理 device 参数 | 需显式传 device |
| `device_mesh.init_device_mesh("cuda", ...)` | `device_mesh.init_device_mesh("npu", ...)` | 参数替换 |

### 双通道分布式通信模式

NPU 分布式训练推荐使用双通道模式（数据并行用 HCCL，元数据用 Gloo）：

```python
# 初始化 HCCL 用于数据通信
dist.init_process_group(backend="hccl", ...)

# 如需 CPU 端元数据通信，可额外初始化 Gloo 后端
# 仅在需要时使用，不影响主流程
```

---

## Graph 模式

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.cuda.CUDAGraph` | `torch.npu.NPUGraph` | ⚠️ | 仅推理支持 |
| `torch.cuda.graph()` | `torch.npu.graph()` | ⚠️ | 仅推理支持 |

**注意：** NPU Graph 目前仅支持推理场景，不支持训练场景。

---

## Profiler

| CUDA API | NPU API | 是否支持 | 说明 |
|----------|---------|----------|------|
| `torch.profiler` | `torch_npu.profiler` | ✅ | |

---

## 不支持的 API

以下 API 在 NPU 上没有对应实现：

| API | 说明 |
|-----|------|
| `torch.cuda.get_device_capability()` | NPU 设备无架构概念 |
| `torch.cuda.memory_usage()` | 无对应实现 |
| `torch.cuda.comm.scatter()` | 不支持 |
| `torch.cuda.comm.gather()` | 不支持 |
| `torch.fft.ihfft2()` | 不支持 |
| `torch.fft.ihfftn()` | 不支持 |

---

## 张量方法替换

| 原方法 | 替换为 | 说明 |
|--------|--------|------|
| `tensor.cuda()` | `tensor.npu()` | 移动到 NPU |
| `tensor.to("cuda")` | `tensor.to("npu")` | 移动到 NPU |
| `tensor.to(torch.device("cuda"))` | `tensor.to(torch.device("npu"))` | 移动到 NPU |
| `tensor.is_cuda` | `tensor.is_npu` | 属性检查 |
| `UninitializedTensorMixin.to(...)` | 自动适配 | 特殊处理 |

### torch.* 全局函数替换（transfer_to_npu 覆盖）

以下函数会通过 `transfer_to_npu` 自动处理设备参数：

```
torch.arange, torch.empty, torch.zeros, torch.ones, torch.full, torch.rand,
torch.randn, torch.randint, torch.tensor, torch.as_tensor, torch.frombuffer,
torch.eye, torch.linspace, torch.logspace, torch.sparse_coo_tensor,
torch.randperm, torch.randn_like, torch.rand_like, torch.zeros_like,
torch.ones_like, torch.full_like, torch.randint_like, torch.empty_like,
torch.autocast, torch.load, torch.set_default_device, ...
```

**注意**：`torch.jit.script` 和 `torch.jit.script_method` 在使用 `transfer_to_npu` 时会被禁用（会打印警告）。

### torch.Tensor.* 方法替换

| 原方法 | 说明 |
|--------|------|
| `tensor.new_tensor(..., device="cuda")` | 参数自动替换 |
| `tensor.new_empty(..., device="cuda")` | 参数自动替换 |
| `tensor.new_zeros(..., device="cuda")` | 参数自动替换 |
| `tensor.new_ones(..., device="cuda")` | 参数自动替换 |
| `tensor.new_full(..., device="cuda")` | 参数自动替换 |
| `tensor.to(..., device="cuda")` | 参数自动替换 |
| `tensor.pin_memory()` | 自动适配 |

### torch.nn.Module.* 方法替换

| 原方法 | 说明 |
|--------|------|
| `model.to(device="cuda")` | 参数自动替换 |
| `model.to_empty(device="cuda")` | 参数自动替换 |

---

## 模型方法替换

| 原方法 | 替换为 | 说明 |
|--------|--------|------|
| `model.cuda()` | `model.npu()` | 移动到 NPU |
| `model.to("cuda")` | `model.to("npu")` | 移动到 NPU |

---

---

## 第三方库自动适配（transfer_to_npu 机制）

`transfer_to_npu` 通过配置文件 `apis_config.json` 实现了对主流 ML 库的自动适配。

### 已支持的库

| 库 | 版本要求 | 适配的内容 |
|----|----------|------------|
| **transformers** | ≥ 4.32.0 | TrainingArguments 设备检测、Trainer 随机状态、CUDA 可用性判断等 |
| **trl** | ≥ 0.7.1 | bitsandbytes 设备检测、kbit_device_map 等 |
| **peft** | ≥ 0.5.0 | infer_device 推断设备 |
| **accelerate** | ≥ 0.22.0 | 环境命令、启动器、模型加载、内存计算、CUDA 可用性检测等 |

### 适配机制说明

对于第三方库，`transfer_to_npu` 不是简单替换 API，而是**包装函数**来改变行为：

```python
# 原理：包装原函数，使其返回符合 NPU 环境的结果
def _wrapper_libraries_func(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        # 临时恢复 torch.cuda.is_available() 的原始行为
        patched_is_available = torch.cuda.is_available
        torch.cuda.is_available = is_available  # 改成 torch.npu.is_available
        result = fn(*args, **kwargs)
        torch.cuda.is_available = patched_is_available  # 恢复
        return result
    return decorated
```

### 未覆盖的第三方库

如果遇到未在上述列表中的库（如 timm、detectron2 等），需要手动处理：
1. 搜索代码中 `cuda` 字符串
2. 手动替换为 `npu` 或添加设备判断逻辑

---

## CUDAGraph 与 NPUGraph

| CUDA API | NPU API | 支持程度 |
|----------|---------|----------|
| `torch.cuda.CUDAGraph` | `torch.npu.NPUGraph` | ⚠️ 仅推理 |
| `torch.cuda.graph()` | `torch.npu.graph()` | ⚠️ 仅推理 |
| `torch.cuda.graphCaptureBegin()` | - | ❌ 不支持 |
| `torch.cuda.graphCaptureEnd()` | - | ❌ 不支持 |

**注意**：NPU Graph 目前仅支持推理场景，不支持训练场景。使用 `transfer_to_npu` 时会自动替换。

---

## Profiler 适配

| CUDA Profiler | NPU Profiler |
|---------------|--------------|
| `torch.profiler.profile` | `torch_npu.profiler.profile` |
| `torch.profiler.schedule` | `torch_npu.profiler.schedule` |
| `torch.profiler.tensorboard_trace_handler` | `torch_npu.profiler.tensorboard_trace_handler` |
| `torch.profiler.ProfilerActivity.CUDA` | `torch_npu.profiler.ProfilerActivity.NPU` |

**使用 `transfer_to_npu` 时的注意事项**：
- 会自动删除 `experimental_config` 参数（CUDA 特有）
- 如需使用 profiler 的 experimental 功能，需手动适配

---

*最后更新: 2026-04-08*
*来源: Ascend PyTorch 官方文档 v2.7.1 - v7.3.0 + transfer_to_npu 源码分析*