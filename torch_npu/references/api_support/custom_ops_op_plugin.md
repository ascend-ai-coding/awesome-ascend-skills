# 自定义算子扩展

本文档介绍了如何通过 op-plugin 扩展 torch_npu 的算子支持。

## 官方文档

- **op-plugin 仓库**：[PyTorch API 扩展文档](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/menu_Pytorch_API.md)

## 功能说明

op-plugin 提供自定义算子注册、加速算子实现、格式转换、稀疏算子等功能。

## 快速开始

### 1. 安装 op-plugin

```bash
pip install op-plugin
```

### 2. 注册自定义算子

```python
from op_plugin import register_op

@register_op(name="custom_add", input_dtypes=[torch.float32], output_dtype=torch.float32)
def custom_add(a):
    """自定义加法算子"""
    return a + 1
```

### 3. 使用自定义算子

```python
import torch

a = torch.randn(2, 3)
result = custom_add(a)
```


## 注意事项

1. 自定义算子需要实现完整的输入/输出规格
2. 性能测试对比自定义算子与原生实现
3. 精度验证与 CPU 实现对齐

## 相关文档

- [API 索引总览](./api_index.md)
- [二进制黑名单](./binary_blocklist.md)
- [确定性计算 API](./deterministic_computing.md)

