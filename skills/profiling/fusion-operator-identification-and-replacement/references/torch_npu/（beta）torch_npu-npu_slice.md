# （beta）torch_npu.npu_slice

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.slice。

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

从张量中提取切片。

## 函数原型

```python
torch_npu.npu_slice(self, offsets, size) -> Tensor
```

> [!CAUTION]  
> 该接口不支持反向计算。

## 参数说明

- **self** (`Tensor`)：输入张量。
- **offsets** (`List[int]`)：支持的数据类型为`torch.int32`，`torch.int64`。
- **size** (`List[int]`)：支持的数据类型为`torch.int32`，`torch.int64`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16).to("npu")
>>> offsets = [0, 0]
>>> size = [2, 2]
>>> output = torch_npu.npu_slice(input, offsets, size)
>>> print(output)
tensor([[1., 2.],
        [6., 7.]], device='npu:0', dtype=torch.float16)
```
