# （beta）torch_npu.npu_max

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

使用dim对最大结果进行计算。类似于torch.max，优化NPU设备实现。

## 函数原型

```python
torch_npu.npu_max(self, dim, keepdim=False) -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`)：必选参数，输入张量。
- **dim** (`int`)：必选参数，待降低维度。
- **keepdim** (`bool`)：可选参数，输出张量是否保留dim。默认值为False。

## 返回值说明

- **values** (`Tensor`)：输入张量中的最大值。
- **indices** (`Tensor`)：输入张量中最大值的索引。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> print(input)
tensor([[[[-1.8135,  0.2078],
          [-0.6678,  0.7846]],

        [[ 0.6458, -0.0923],
          [-0.2124, -1.9112]]],

        [[[-0.5800, -0.4979], 
         [ 0.2580,  1.1335]],

          [[ 0.6669,  0.1876],
          [ 0.1160, -0.1061]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_max(input, 2)
>>> print(outputs)
tensor([[[-0.6678,  0.7846],
        [ 0.6458, -0.0923]],

        [[ 0.2580,  1.1335],
        [ 0.6669,  0.1876]]], device='npu:0')
>>> print(indices)
tensor([[[1, 1],
        [0, 0]],

        [[1, 1],
        [0, 0]]], device='npu:0', dtype=torch.int32)
```
