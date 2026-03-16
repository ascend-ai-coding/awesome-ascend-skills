# 内存优化指南

Diffusers 模型在昇腾 NPU 上的内存优化策略。大模型（>10B 参数）在 bfloat16 下仍可能占用 20GB+ 显存，需要合理优化。

## 显存估算

估算公式：

| 组件 | 估算方式 |
|------|---------|
| 模型参数 | 参数量 × dtype 字节数（bfloat16 = 2 bytes） |
| 推理激活 | 约 1-3× 模型参数大小 |
| VAE 解码 | 取决于输出分辨率，1024×1024 约 1-2 GB |
| 安全余量 | 预留 2-4 GB |

示例（FLUX.1-dev，~12B 参数）：

```
模型参数:  12B × 2 bytes = ~24 GB
推理激活:  ~8-12 GB（注意力计算）
VAE 解码:  ~1-2 GB
总计:      ~33-38 GB
```

查询 NPU 可用显存：

```python
import torch
import torch_npu

for i in range(torch.npu.device_count()):
    free, total = torch.npu.mem_get_info(i)
    print(f"NPU:{i}  总量: {total/1024**3:.1f}GB  可用: {free/1024**3:.1f}GB")
```

## 策略详解

### 1. 使用 bfloat16（推荐首选）

将模型精度从 float32 降为 bfloat16，显存减半，NPU 对 bfloat16 有硬件加速：

```python
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 显存减半
)
```

### 2. Attention Slicing

将注意力计算切分为多个小块，减少峰值显存，代价是速度略慢：

```python
pipe.enable_attention_slicing()

# 指定切片大小（可选）
pipe.enable_attention_slicing(slice_size="auto")
pipe.enable_attention_slicing(slice_size=1)  # 最省内存但最慢

# 关闭
pipe.disable_attention_slicing()
```

### 3. VAE Slicing

将 VAE 编解码拆分为逐图处理，对批量生成有效：

```python
pipe.enable_vae_slicing()

# 关闭
pipe.disable_vae_slicing()
```

### 4. VAE Tiling

将 VAE 解码按空间 tile 分块执行，显著减少高分辨率图像的显存：

```python
pipe.enable_vae_tiling()

# 关闭
pipe.disable_vae_tiling()
```

适用场景：生成 1024×1024 或更高分辨率图像时。

### 5. Model CPU Offload

将不活跃的子模型卸载到 CPU，仅在需要时加载到 NPU。显存节省显著，但增加 CPU-NPU 传输开销：

```python
# 注意: 使用 cpu_offload 时不要先调用 pipe.to("npu")
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device="npu:0")
```

**NPU 注意事项**：
- `enable_model_cpu_offload()` 需要 `accelerate` 库
- 在 NPU 上使用时需传入 `device="npu:0"` 参数
- 效果取决于 PCIe 带宽，传输延迟可能较高
- 如遇兼容性问题，优先使用 attention slicing + VAE tiling 组合

### 6. Sequential CPU Offload

更激进的 CPU offload，逐层而非逐模型卸载，显存占用最小但速度最慢：

```python
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device="npu:0")
```

**仅在其他方式都无法装入显存时使用**。

## 推荐组合

### 显存充足（≥40GB）

```python
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe = pipe.to("npu:0")
```

### 显存紧张（32-40GB）

```python
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe = pipe.to("npu:0")
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
```

### 显存不足（<32GB）

```python
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device="npu:0")
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
```

### 极端情况（<16GB）

```python
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device="npu:0")
pipe.enable_attention_slicing(slice_size=1)
pipe.enable_vae_tiling()
```

## 显存监控

推理过程中监控显存使用：

```python
import torch
import torch_npu

torch.npu.reset_peak_memory_stats()

# ... 执行推理 ...

peak_memory = torch.npu.max_memory_allocated() / 1024**3
current_memory = torch.npu.memory_allocated() / 1024**3
print(f"峰值显存: {peak_memory:.2f} GB")
print(f"当前显存: {current_memory:.2f} GB")
```

## 其他技巧

### 推理后释放显存

```python
import gc
import torch

del pipe
gc.collect()
torch.npu.empty_cache()
```

### 减少推理步数

步数越少速度越快，但质量下降。在 20-30 步通常可获得不错结果：

```python
image = pipe(prompt=prompt, num_inference_steps=20).images[0]
```

### 降低分辨率

分辨率对显存影响巨大（平方关系）：

| 分辨率 | 显存需求（相对） |
|--------|----------------|
| 512×512 | 1× |
| 768×768 | 2.25× |
| 1024×1024 | 4× |
| 2048×2048 | 16× |
