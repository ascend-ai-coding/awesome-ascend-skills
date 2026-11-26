# torch_npu.matmul_checksum

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

提供基于原生torch.matmul和Tensor.matmul接口的aicore错误硬件故障检测接口。内部执行矩阵计算结果校验过程，并对校验误差和实时计算的校验门限进行对比，判断校验误差是否超过门限，若超过则认为发生了aicore错误。

## 函数原型

```python
torch_npu.matmul_checksum(a, b, c) -> Tensor
```

## 参数说明

- **a** (`Tensor`)：必选输入，进行原生matmul计算的输入input。
- **b** (`Tensor`)：必选输入，进行原生matmul计算的输入other。
- **c** (`Tensor`)：必选输入，原生matmul计算的输出out。

## 返回值说明

`Tensor`

返回NPU上的`torch.bool`标量。结果为True时，表示存在aicore错误的硬件故障。

## 约束说明

- 该接口仅支持device为NPU的场景。
- CANN版本为9.2.0及以上，且输入参数`a`、`b`、`c`为2维Tensor时，支持`torch.bfloat16`和`torch.float32`数据类型；其余场景仅支持`torch.bfloat16`数据类型。

## 调用示例

   ```python
   >>> import torch
   >>> import torch_npu
   >>> matrix1 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
   >>> matrix2 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
   >>> product = torch.matmul(matrix1, matrix2)
   >>> checksum = torch_npu.matmul_checksum(matrix1, matrix2, product)
   >>> print(checksum)
   tensor(False, device='npu:0')
   ```
