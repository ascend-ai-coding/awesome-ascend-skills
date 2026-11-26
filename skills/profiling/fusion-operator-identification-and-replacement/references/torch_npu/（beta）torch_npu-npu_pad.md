# （beta）torch_npu.npu_pad

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.nn.functional.pad。

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

填充张量。

## 函数原型

```python
torch_npu.npu_pad(input, paddings) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：输入张量。
- **paddings** (`List[int]`)：数据类型为`torch.int32`、`torch.int64`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16).to("npu")
>>> paddings = [1, 1, 1, 1]
>>> output = torch_npu.npu_pad(input, paddings)
>>> print(output)
tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 20., 20., 10., 10.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
```
