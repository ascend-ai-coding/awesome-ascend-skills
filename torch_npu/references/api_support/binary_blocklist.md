# 二进制黑名单

本文档列出了 torch_npu 中不支持或有限支持的 API，称为"二进制黑名单"。这些 API 在 NPU 上可能无法正常工作，或性能较差。

## 官方文档

- **华为昇腾官方文档**：[二进制黑名单示例](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/example_of_adding_a_binary_blocklist.md)

## 功能说明

黑名单 API 通常有以下问题：

1. **不支持**：NPU 完全不实现该功能
2. **有限支持**：NPU 有基本实现，但功能不完整
3. **性能差**：NPU 实现比 CPU 慢很多

## 常见黑名单 API

以下 API 在 NPU 上可能无法正常使用：

- **复杂 CUDA 特性**：`torch.backends.cudnn.benchmark`、自定义 CUDA kernels
- **特定硬件指令**：某些 SIMD/AVX 指令集相关操作
- **动态内存管理**：复杂的内存分配和释放操作
- **稀疏矩阵格式**：某些稀疏矩阵运算

## 避免黑名单 API

### 1. 检查是否在黑名单中

```python
import torch
import torch_npu

# 某些 API 可能不支持
try:
    torch.backends.cudnn.benchmark = True
except RuntimeError as e:
    print(f"不支持该 API: {e}")
```

### 2. 使用兼容的替代方案

将黑名单 API 替换为 NPU 支持的等价操作，参见[自定义算子扩展](./custom_ops_op_plugin.md)。

### 3. 启用确定性计算

某些黑名单 API 在确定性模式下可能可工作，详见[deterministic_computing.md](./deterministic_computing.md)。

## 注意事项

1. 黑名单列表会随着版本更新而变化
2. 检查当前版本的 API 支持情况，详见[API 索引](./api_index.md)
3. 如需添加新的黑名单 API，参考官方文档中的示例

## 相关文档

- [确定性计算 API](./deterministic_computing.md)
- [自定义算子扩展](./custom_ops_op_plugin.md)
- [API 索引总览](./api_index.md)
