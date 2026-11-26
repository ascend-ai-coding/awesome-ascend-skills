# （beta）torch_npu.npu_indexing

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

以begin为起始索引，end为结束索引，strides为步长，对输入张量进行切片。

## 函数原型

```python
torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，输入张量。
- **begin** (`List[int]`)：必选参数，待选择的第一个值的index。
- **end** (`List[int]`)：必选参数，结束索引（不包含该位置），即索引区间为左闭右开`[begin, end)`。
- **strides** (`List[int]`)：必选参数，index增量。
- **begin_mask** (`int`)：可选参数，位掩码（bitmask），其中位“i”为“1”意味着忽略开始值，尽可能使用最大间隔。默认值为0。
- **end_mask** (`int`)：可选参数，类似于“begin_mask”。默认值为0。
- **ellipsis_mask** (`int`)：可选参数，位掩码，其中位“i”为“1”意味着第“i”个位置实际上是省略号。默认值为0。
- **new_axis_mask** (`int`)：可选参数，位掩码，其中位“i”为“1”意味着在第“i”位创建新的1D shape。默认值为0。
- **shrink_axis_mask** (`int`)：可选参数，位掩码，其中位“i”意味着第“i”位应缩小维数。默认值为0。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
>>> print(input)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
>>> output = torch_npu.npu_indexing(input, [0, 0], [2, 2], [1, 1])
>>> print(output)
tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
```
