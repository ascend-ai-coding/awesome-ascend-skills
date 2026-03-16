# 故障排查

Diffusers Pipeline 在昇腾 NPU 上推理的常见问题及解决方案。

## 1. NPU 设备不可用

**报错**：`RuntimeError: No NPU devices available`

**原因**：torch_npu 未正确安装，或 CANN 环境未加载。

**解决**：

```bash
# 1. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source /usr/local/Ascend/cann/set_env.sh

# 2. 验证
python -c "import torch; import torch_npu; print(torch.npu.is_available())"

# 3. 如仍失败，检查驱动
npu-smi info
```

## 2. 显存不足（OOM）

**报错**：`torch.npu.OutOfMemoryError: NPU out of memory`

**原因**：模型参数 + 推理激活超出 NPU 显存。

**解决**：

```python
# 方案 1：启用内存优化
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()

# 方案 2：使用 CPU offload（不要先 .to("npu")）
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device="npu:0")

# 方案 3：降低分辨率
image = pipe(prompt=prompt, height=512, width=512).images[0]

# 方案 4：减少推理步数
image = pipe(prompt=prompt, num_inference_steps=15).images[0]
```

清理已占用显存：

```python
import gc, torch
del pipe
gc.collect()
torch.npu.empty_cache()
```

## 3. Generator 设备不匹配

**报错**：`RuntimeError: Expected a 'npu' device type for generator but found 'cpu'`

**原因**：Generator 在 CPU 上创建，但模型在 NPU 上。

**解决**：

```python
# 错误
generator = torch.Generator().manual_seed(42)

# 正确
generator = torch.Generator("npu:0").manual_seed(42)

# 如果使用 CPU offload，generator 需要在 CPU 上
generator = torch.Generator("cpu").manual_seed(42)
```

## 4. Pipeline 类型不匹配

**报错**：`ValueError: Cannot load ... because ... is not compatible with ...`

**原因**：使用了错误的 Pipeline 类加载模型。

**解决**：

```python
# 方案 1：使用自动检测
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(model_path)

# 方案 2：使用 AutoPipeline
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained(model_path)
```

## 5. bfloat16 不支持某些操作

**报错**：`RuntimeError: "xxx" not implemented for 'BFloat16'`

**原因**：部分算子在 NPU 上尚未支持 bfloat16。

**解决**：

```python
# 尝试 float16
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("npu:0")

# 或 float32（显存翻倍）
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe = pipe.to("npu:0")
```

## 6. Scheduler 不兼容

**报错**：`ValueError: ... is not a valid scheduler for this pipeline`

**原因**：并非所有 Scheduler 都兼容所有 Pipeline。

**解决**：

```python
# 查看 Pipeline 兼容的 Scheduler
print(pipe.scheduler.compatibles)

# 从兼容列表中选择
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

## 7. CPU Offload 在 NPU 上不生效

**报错**：`ValueError: ... device npu is not supported`

**原因**：`accelerate` 版本过旧或不支持 NPU 设备。

**解决**：

```bash
# 升级 accelerate
pip install -U accelerate

# 如仍不工作，跳过 CPU offload，使用其他优化
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe = pipe.to("npu:0")
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
```

## 8. 模型加载缓慢

**原因**：大模型权重文件读取 I/O 慢。

**解决**：

```python
# 使用 safetensors 格式（比 .bin 快 ~2x）
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
```

## 9. 图像输出全黑或全白

**原因**：
- 假权重（随机初始化）产生无意义输出 —— **这是正常的**
- 真权重下出现则可能是 dtype 精度问题

**解决**：

```python
# 1. 确认使用的是真权重（非假权重）
# 2. 尝试切换 dtype
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)

# 3. 检查 guidance_scale 是否合理
image = pipe(prompt=prompt, guidance_scale=7.5).images[0]
```

## 10. 代理 / 网络问题导致模型下载失败

**解决**：

```bash
# 取消 socks 代理（常见冲突源）
unset all_proxy ALL_PROXY http_proxy https_proxy

# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用 Phase #2 的下载脚本
python scripts/download_weights.py hf model_id --proxy https://hf-mirror.com
```

## 诊断信息收集

遇到未知问题时，收集以下信息用于排查：

```python
import torch
import torch_npu
import diffusers
import transformers

print(f"PyTorch:      {torch.__version__}")
print(f"torch_npu:    {torch_npu.__version__}")
print(f"Diffusers:    {diffusers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count:     {torch.npu.device_count()}")
for i in range(torch.npu.device_count()):
    print(f"NPU:{i} - {torch.npu.get_device_name(i)}")
    free, total = torch.npu.mem_get_info(i)
    print(f"  Memory: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
```
