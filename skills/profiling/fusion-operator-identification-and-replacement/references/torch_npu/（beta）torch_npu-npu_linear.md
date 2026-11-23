# （beta）torch_npu.npu_linear

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

将矩阵“a”乘以矩阵“b”，生成“a\*b”。

## 函数原型

```python
torch_npu.npu_linear(input, weight, bias=None) -> Tensor
```

## 参数说明

- **input**(`Tensor`)：必选参数，2D矩阵张量。数据类型支持`torch.float32`、`torch.float16`、`torch.int32`、`torch.int8`。数据格式支持$ND$、$NHWC$、$FRACTAL\_NZ$。

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型不支持`torch.int32`、`torch.int8`。
  <!-- end id6 -->

- **weight**(`Tensor`)：必选参数，2D矩阵张量。数据类型支持`torch.float32`、`torch.float16`、`torch.int32`、`torch.int8`。数据格式支持$ND$、$NHWC$、$FRACTAL\_NZ$。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型不支持`torch.int32`、`torch.int8`。
  <!-- end id7 -->

- **bias**(`Tensor`)：**可选参数**，1D张量。数据类型支持`torch.float32`、`torch.float16`、`torch.int32`。数据格式支持$ND$、$NHWC$。默认值为None。

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型不支持`torch.int32`。
  <!-- end id8 -->

## 返回值说明

- **out**(`Tensor`)：输出张量，数据类型与计算输入`input`和`weight`的类型一致。数据格式支持$ND$。

## 约束说明

- 该接口仅支持单算子模式。

## 调用示例

单算子模式调用如下：

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2, 16).npu()
>>> w = torch.rand(4, 16).npu()
>>> b = torch.rand(4).npu()
>>> output = torch_npu.npu_linear(x, w, b)
>>> print(output)
tensor([[3.6335, 4.3713, 2.4440, 2.0081],
        [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
```
