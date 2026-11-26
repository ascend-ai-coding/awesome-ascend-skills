# （beta）torch_npu.npu_alloc_float_status

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

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

申请一个专门用于存储浮点运算状态标志的tensor。该tensor用于后续记录计算过程中的溢出状态。

## 函数原型

```python
torch_npu.npu_alloc_float_status(input) -> Tensor
```

## 参数说明

**input** (`Tensor`)：必选参数，任意构建的一个NPU张量（主要用于确定device信息）。

## 返回值说明

`Tensor`

一个包含8个`torch.float32`类型全零值的Tensor。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.randn([1,2,3]).npu()
## 分配状态空间
>>> output = torch_npu.npu_alloc_float_status(input)
>>> print(input)
tensor([[[ 2.2324,  0.2478, -0.1056],
        [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
>>> print(output)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')

## 清除状态
>>> torch_npu.npu_clear_float_status(output)

## 执行可能溢出的计算操作
## ...模型前向/反向传播...

## 获取检测结果
>>> result = torch_npu.npu_get_float_status(output)

```
