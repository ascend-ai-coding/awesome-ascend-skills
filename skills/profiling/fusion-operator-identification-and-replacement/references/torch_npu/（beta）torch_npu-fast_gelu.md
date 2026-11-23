# （beta）torch\_npu.fast\_gelu

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算FastGelu。支持FakeTensor模式。

## 函数原型

```python
torch_npu.fast_gelu(self) -> Tensor
```

## 参数说明

**self** (`Tensor`)：支持的数据类型为`torch.float16`、`torch.float32`。

## 调用示例

示例一：

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2).npu()
>>> print(x)
tensor([0.5991, 0.4094], device='npu:0')
>>> print(torch_npu.fast_gelu(x))
tensor([0.4403, 0.2733], device='npu:0')
```

示例二：

```python
>>> import torch
>>> import torch_npu
# FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2).npu()
...     torch_npu.fast_gelu(x)
>>> FakeTensor(..., device='npu:0', size=(2,))
```
