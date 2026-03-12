# API 索引总览

本文档提供了 torch_npu 不同版本的 API 支持清单，以及确定性计算、黑名单、自定义算子等扩展功能文档。

## 版本化 API 清单

| 版本 | 文档链接 | 主要特性 |
|------|---------|---------|
| **2.9.0** | [PyTorch-2-9-0](https://gitcode.com/Ascend/pytorch/blob/v7.3.0-pytorch2.9.0/docs/zh/native_apis/PyTorch-2-9-0.md) | 最新版本，完整 API 支持 |
| **2.8.0** | [PyTorch-2-8-0](https://gitcode.com/Ascend/pytorch/blob/v7.3.0-pytorch2.8.0/docs/zh/native_apis/PyTorch-2-8-0.md) | 增量更新，性能优化 |
| **2.7.1** | [PyTorch-2-7-1](https://gitcode.com/Ascend/pytorch/blob/v7.3.0-pytorch2.7.1/docs/zh/native_apis/PyTorch-2-7-1.md) | 稳定版本，bug 修复 |
| **2.6.0** | [PyTorch-2-6-0](https://gitcode.com/Ascend/pytorch/blob/v7.3.0-pytorch2.6.0/docs/zh/native_apis/PyTorch-2-6-0.md) | 性能提升，新算子支持 |
| **2.1.0** | [torch_npu_apis](https://gitcode.com/Ascend/pytorch/blob/v2.1.0-7.2.0/docs/api/torch_npu_apis.md) | 基础版本，核心功能 |

## 扩展功能文档

### 确定性计算 API
[**deterministic_computing.md**](./deterministic_computing.md)

启用确定性模式后，NPU 计算结果与 CPU 保持一致，适用于需要可复现性的场景。

### 二进制黑名单
[**binary_blocklist.md**](./binary_blocklist.md)

列出不支持或有限支持的 API，避免在 NPU 上调用导致性能下降或错误。

### 自定义算子扩展
[**custom_ops_op_plugin.md**](./custom_ops_op_plugin.md)

通过 op-plugin 扩展 NPU 算子支持，提供额外的自定义算子。


## 常见问题

**Q: 如何选择 PyTorch 版本？**

A: 根据您的 CANN 版本和 Python 版本选择配套的 torch_npu 版本，详见[版本配套表](../installation/version_compatibility.md)。

**Q: 某个 API 在 NPU 上不可用怎么办？**

A: 检查 API 是否在黑名单中，或者使用 op-plugin 扩展支持。

**Q: 如何启用确定性计算？**

A: 参见 [deterministic_computing.md](./deterministic_computing.md)。

## 相关资源

- [torch-npu 官方仓库](https://gitcode.com/Ascend/pytorch)
- [op-plugin 仓库](https://gitcode.com/Ascend/op-plugin)
- [华为昇腾文档中心](https://www.hiascend.com/document)
