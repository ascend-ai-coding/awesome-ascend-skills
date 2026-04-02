# 本地权重加载参考

本参考文档给出 Ankh 系列模型在 Ascend NPU 上的本地 HuggingFace 权重加载约定。

## 目录映射

推荐目录映射：

```text
ankh_base  -> <weights_root>/Ankh_base
ankh_large -> <weights_root>/Ankh_large
ankh3_large -> <weights_root>/Ankh3_large
ankh3_xl -> <weights_root>/Ankh3_XL
```

## 统一 dtype 解析

```python
import torch


def infer_torch_dtype(device: torch.device, dtype: str | None = None):
    if dtype is None or dtype == "auto":
        if str(device).startswith("npu"):
            return torch.bfloat16
        if str(device).startswith("cuda"):
            return torch.float16
        return torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype.lower()]
```

## Ankh3 XL 本地目录加载

```python
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer


def load_local_ankh3_model(local_dir, task="encoder", device=None, dtype="auto"):
    local_dir = Path(local_dir)
    device = resolve_device(device)
    torch_dtype = infer_torch_dtype(device, dtype)
    tokenizer = T5Tokenizer.from_pretrained(local_dir)
    if task == "encoder":
        model = T5EncoderModel.from_pretrained(local_dir, torch_dtype=torch_dtype)
    else:
        model = T5ForConditionalGeneration.from_pretrained(local_dir, torch_dtype=torch_dtype)
    model = model.eval().to(device)
    return model, tokenizer, device
```

## Ankh3 XL 注意事项

- tokenizer 优先使用 `T5Tokenizer`
- 分片权重使用目录级 `from_pretrained()`，不要手工读取单个 `.bin`
- embedding 抽取优先用 `T5EncoderModel`
- generation 和 likelihood 场景再使用 `T5ForConditionalGeneration`
