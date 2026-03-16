# LoRA 集成指南

在昇腾 NPU 上使用 Diffusers 的 LoRA（Low-Rank Adaptation）适配器进行模型微调效果推理。

## 基本用法

### 加载单个 LoRA

```python
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe = pipe.to("npu:0")

# 从本地路径加载
pipe.load_lora_weights("./lora_weights/", adapter_name="my_lora")

# 从 HuggingFace Hub 加载
pipe.load_lora_weights("username/lora-repo", adapter_name="hub_lora")

# 推理
image = pipe(
    prompt="a photo in my_lora style",
    num_inference_steps=30,
).images[0]
```

### 调整 LoRA 权重

```python
# 设置 LoRA 强度 (0.0 = 不生效, 1.0 = 完全生效)
pipe.set_adapters(["my_lora"], adapter_weights=[0.8])

# 推理
image = pipe(prompt="...", num_inference_steps=30).images[0]
```

### 卸载 LoRA

```python
pipe.unload_lora_weights()
```

## 多 LoRA 推理

同时加载多个 LoRA，按权重混合效果：

```python
# 加载多个 LoRA
pipe.load_lora_weights("./style_lora/", adapter_name="style")
pipe.load_lora_weights("./character_lora/", adapter_name="character")

# 设置各自权重
pipe.set_adapters(
    ["style", "character"],
    adapter_weights=[0.7, 0.5],
)

# 推理
image = pipe(prompt="...", num_inference_steps=30).images[0]
```

### 切换活跃 LoRA

```python
# 只启用 style
pipe.set_adapters(["style"], adapter_weights=[1.0])
image_style = pipe(prompt="...", num_inference_steps=30).images[0]

# 只启用 character
pipe.set_adapters(["character"], adapter_weights=[1.0])
image_char = pipe(prompt="...", num_inference_steps=30).images[0]

# 禁用所有 LoRA
pipe.disable_lora()

# 重新启用
pipe.enable_lora()
```

### 删除特定 LoRA

```python
pipe.delete_adapters(["character"])
```

## LoRA 权重合并

将 LoRA 权重永久合入基础模型，消除推理时的额外计算开销：

```python
# 加载 LoRA
pipe.load_lora_weights("./lora_weights/", adapter_name="my_lora")

# 合并到基础模型
pipe.fuse_lora(adapter_names=["my_lora"], lora_scale=0.8)

# 合并后推理（不再需要 LoRA 相关设置）
image = pipe(prompt="...", num_inference_steps=30).images[0]

# 保存合并后的模型
pipe.save_pretrained("./merged_model/")
```

### 撤销合并

```python
pipe.unfuse_lora()
```

## LoRA 文件格式

Diffusers 支持的 LoRA 格式：

| 格式 | 文件 | 说明 |
|------|------|------|
| Diffusers | `pytorch_lora_weights.safetensors` | 官方格式 |
| Kohya | `*.safetensors` | 社区训练工具格式 |
| PEFT | `adapter_model.safetensors` | HuggingFace PEFT 格式 |

加载不同格式：

```python
# Diffusers 格式（自动识别）
pipe.load_lora_weights("./lora_dir/")

# 指定权重文件名
pipe.load_lora_weights("./lora_dir/", weight_name="custom_name.safetensors")
```

## NPU 注意事项

1. **LoRA 加载位置**：建议在 `.to("npu:0")` 之后加载 LoRA，LoRA 权重会自动迁移到相同设备
2. **显存占用**：每个 LoRA 增加少量显存（通常 <100MB），多 LoRA 线性增长
3. **兼容性**：加载 LoRA 前确认基础模型支持 LoRA（需要 `peft` 库）
4. **性能**：合并（fuse）后的推理速度等同于无 LoRA 的基础模型

安装 PEFT：

```bash
pip install peft
```

## 示例：完整 LoRA 工作流

```python
import torch
import torch_npu
from diffusers import DiffusionPipeline

# 1. 加载基础模型
pipe = DiffusionPipeline.from_pretrained(
    "./base_model",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("npu:0")

# 2. 加载风格 LoRA
pipe.load_lora_weights(
    "./watercolor_lora/",
    adapter_name="watercolor",
)

# 3. 设置权重
pipe.set_adapters(["watercolor"], adapter_weights=[0.75])

# 4. 生成
generator = torch.Generator("npu:0").manual_seed(42)
image = pipe(
    prompt="a mountain landscape, watercolor painting",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images[0]
image.save("watercolor_landscape.png")

# 5. 清理
pipe.unload_lora_weights()
```
