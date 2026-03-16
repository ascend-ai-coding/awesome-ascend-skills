---
name: diffusers-ascend-weight-prep
description: Diffusers 模型权重准备工具，用于华为昇腾 NPU。支持从 HuggingFace 和 ModelScope 下载模型权重，以及基于 model_index.json 和各组件 config.json 生成假权重用于业务验证。当用户需要下载 Diffusers 模型权重或生成测试权重时使用。
keywords:
    - diffusers
    - huggingface
    - modelscope
    - weights
    - model-download
    - fake-weights
    - npu
---

# Diffusers 模型权重准备

本 Skill 提供两个核心能力：
1. 从 HuggingFace 或 ModelScope 下载模型权重
2. 基于 Diffusers Pipeline 的 config 文件生成假权重（随机初始化），用于业务验证

## 快速开始

### 1. 下载模型权重

**从 HuggingFace 下载：**

```bash
pip install -U huggingface_hub

# 基础用法
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers

# 使用代理（国内访问）
export HF_ENDPOINT=https://hf-mirror.com
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers

# 通过脚本下载（支持 dry-run 测试）
python scripts/download_weights.py hf Wan-AI/Wan2.2-I2V-A14B-Diffusers --proxy https://hf-mirror.com
python scripts/download_weights.py hf Wan-AI/Wan2.2-I2V-A14B-Diffusers --dry-run
```

**从 ModelScope 下载：**

```bash
pip install modelscope

# 基础用法
modelscope download --model Wan-AI/Wan2.2-T2V-A14B

# 通过脚本下载
python scripts/download_weights.py modelscope Wan-AI/Wan2.2-T2V-A14B -o ./models
python scripts/download_weights.py modelscope Wan-AI/Wan2.2-T2V-A14B --dry-run
```

### 2. 生成假权重

自动从 HuggingFace 下载元数据（config.json、model_index.json、tokenizer 等，**不下载权重文件**），然后基于 config 随机初始化每个子模型并保存，生成与原始模型**文件结构完全一致**的假权重。

**从 HuggingFace 直接生成：**

```bash
# 生成 Z-Image-Turbo 假权重
python scripts/generate_fake_weights.py from-hub Tongyi-MAI/Z-Image-Turbo -o ./fake_z_image --dtype bfloat16

# 生成 Qwen-Image-2512 假权重（国内使用镜像）
python scripts/generate_fake_weights.py from-hub Qwen/Qwen-Image-2512 -o ./fake_qwen --proxy https://hf-mirror.com
```

**从已有的本地元数据目录生成：**

```bash
python scripts/generate_fake_weights.py from-local ./model_metadata_dir -o ./fake_weights --dtype bfloat16
```

**验证假权重可正常加载：**

```python
import torch
import diffusers

pipe = diffusers.DiffusionPipeline.from_pretrained(
    "./fake_z_image",
    torch_dtype=torch.bfloat16,
)
print(f"Pipeline: {type(pipe).__name__}")
for name, comp in pipe.components.items():
    if hasattr(comp, "parameters"):
        pc = sum(p.numel() for p in comp.parameters())
        print(f"  {name}: {type(comp).__name__} ({pc:,} params)")
```

## 环境要求

| 组件 | 安装命令 | 说明 |
|------|---------|------|
| huggingface_hub | `pip install -U huggingface_hub` | HuggingFace 下载 & 假权重元数据下载 |
| modelscope | `pip install modelscope` | ModelScope 下载 |
| torch | `pip install torch` | 假权重生成 |
| diffusers | `pip install diffusers` | Pipeline 模型实例化 |
| transformers | `pip install transformers` | text_encoder 实例化 |

## 假权重生成原理

生成流程：

1. 从 HuggingFace 通过 `snapshot_download` 下载元数据文件（排除 `*.safetensors`、`*.bin`）
2. 读取 `model_index.json`，解析 Pipeline 的每个组件（scheduler、text_encoder、tokenizer、transformer、vae）
3. 对于每个模型组件，使用 `from_config()` 从 config.json 实例化模型（随机权重）
4. 通过 `save_pretrained()` 保存，自动生成正确格式的 safetensors 文件和 index.json
5. tokenizer 和 scheduler 直接复制配置文件

生成的目录结构与原始模型完全一致，例如 Z-Image-Turbo：

```
fake_z_image_turbo/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── model.safetensors.index.json
├── tokenizer/
│   ├── merges.txt
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   ├── diffusion_pytorch_model-00002-of-00002.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## 脚本参考

### download_weights.py

| 参数 | 必需 | 说明 |
|------|------|------|
| `source` | 是 | 下载源：`hf` 或 `modelscope` |
| `model_id` | 是 | 模型 ID（如 `Wan-AI/Wan2.2-I2V-A14B-Diffusers`） |
| `-o, --output-dir` | 否 | 输出目录 |
| `--proxy` | 否 | HuggingFace 代理 URL（仅 `hf` 源） |
| `--dry-run` | 否 | 测试模式，不实际下载 |

### generate_fake_weights.py

**子命令 `from-hub`**：从 HuggingFace 下载元数据 + 生成假权重

| 参数 | 必需 | 说明 |
|------|------|------|
| `model_id` | 是 | HuggingFace 模型 ID |
| `-o, --output-dir` | 是 | 输出目录 |
| `--proxy` | 否 | HuggingFace 代理 URL |
| `--dtype` | 否 | 数据类型：`float32`、`float16`、`bfloat16`（默认：`bfloat16`） |

**子命令 `from-local`**：从本地元数据目录生成假权重

| 参数 | 必需 | 说明 |
|------|------|------|
| `metadata_dir` | 是 | 本地元数据目录（需包含 model_index.json） |
| `-o, --output-dir` | 是 | 输出目录 |
| `--dtype` | 否 | 数据类型（默认：`bfloat16`） |

## 已验证模型

| 模型 | Pipeline 类型 | 组件 | 验证状态 |
|------|-------------|------|---------|
| Qwen/Qwen-Image-2512 | QwenImagePipeline | vae(127M) + text_encoder(8.3B) + transformer(20.4B) | ✅ |
| Tongyi-MAI/Z-Image-Turbo | ZImagePipeline | vae(84M) + text_encoder(4.0B) + transformer(6.2B) | ✅ |

## 注意事项

- 假权重为随机初始化，生成的图像无意义，仅用于验证代码流程
- 大模型（>10B 参数）生成假权重可能需要较多内存和时间
- 国内环境建议使用 `--proxy https://hf-mirror.com` 或设置 `HF_ENDPOINT` 环境变量
- 如遇 socks 代理冲突，先 `unset all_proxy ALL_PROXY` 再运行

## 参考资源

- [HuggingFace Hub 文档](https://huggingface.co/docs/huggingface_hub)
- [ModelScope 文档](https://modelscope.cn/docs)
- [HuggingFace 镜像站](https://hf-mirror.com)
