# 确定性计算 API

本文档介绍了 torch_npu 的确定性计算功能，该功能可以保证 NPU 计算结果与 CPU 完全一致，适用于需要可复现性的场景。

## 官方文档

- **华为昇腾官方文档**：[确定性计算 API](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/deterministic_computing_apis.md)

## 功能说明

确定性计算通过以下机制保证结果可复现：

- **随机数种子控制**：固定随机数生成器的种子，确保相同输入产生相同输出
- **算子执行顺序**：保证计算图的执行顺序一致
- **浮点精度控制**：支持 FP32 累加，减少精度损失

## 使用示例

```python
import torch
import torch_npu

# 启用确定性计算模式
torch_npu.set_deterministic(True)

# 随机数生成可复现
torch.manual_seed(42)
a = torch.randn(2, 3)
print(a)  # 每次运行结果相同
```

## 注意事项

1. 启用确定性计算可能会降低性能，仅在需要可复现性时使用
2. 某些算子可能不完全支持确定性模式，会发出警告
3. 检查[黑名单文档](./binary_blocklist.md)了解不支持确定性计算的 API

## 相关文档

- [二进制黑名单](./binary_blocklist.md) - 不支持/有限支持的 API
- [API 索引总览](./api_index.md) - 完整的 API 文档链接
