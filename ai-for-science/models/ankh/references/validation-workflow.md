# 验证与训练流程参考

本参考文档给出 Ankh 迁移到 Ascend NPU 后的最小验证与训练 smoke test 方案。

## 运行时环境检查

```python
import torch
import ankh

print(torch.__version__)
print("torch_npu installed:", ankh.has_torch_npu())
print("npu available:", ankh.is_npu_available())
print("resolved device:", ankh.resolve_device("auto"))
```

## 最小前向验证

```python
import ankh
import torch

model, tokenizer = ankh.load_model("ankh_base")
device = ankh.resolve_device("auto")
model = model.eval().to(device)

tokens = tokenizer(
    [list("MKTAYIAKQ")],
    add_special_tokens=True,
    padding=True,
    is_split_into_words=True,
    return_tensors="pt",
)
tokens = {k: v.to(device) for k, v in tokens.items()}

with torch.no_grad():
    outputs = model(**tokens)

print(outputs.last_hidden_state.shape)
```

## 推荐验证命令

```bash
python scripts/verify_ankh_base_npu.py --device auto
python scripts/verify_ankh_large_npu.py --device auto
python scripts/verify_ankh3_large_npu.py --device auto
python scripts/verify_ankh3_xl_npu.py --device auto --weights-dir <weights_root>/Ankh3_XL
```

## 训练 smoke test

```bash
python scripts/train_smoke_test.py --model ankh_base --device auto --dtype auto
```

预期结果：

- `resolved_device` 为 `npu:0` 或自动降级设备
- `train_step_ok` 为 `true`
- `loss` 为有限数值
- 梯度参数计数大于 0
