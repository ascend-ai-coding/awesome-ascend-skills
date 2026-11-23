# （beta）torch_npu.npu_random_choice_with_mask

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

获取非零元素的index，混洗后输出。

## 函数原型

```python
torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，数据类型仅支持`torch.bool`。
- **count** (`int`)：可选参数，输出计数，默认值为256。如果值为0，则输出所有非零元素。
- **seed** (`int`)：可选参数，随机种子值，默认值为0，数据类型支持`torch.int32`，`torch.int64`。
- **seed2** (`int`)：可选参数，随机种子值，默认值为0，数据类型支持`torch.int32`，`torch.int64`。

## 返回值说明

- **y** (`Tensor`)：2D张量，非零元素的index。
- **mask** (`Tensor`)：1D张量，确定对应index是否有效。

## 调用示例

```python
>>> import torch, torch_npu
>>> x = torch.tensor([1, 0, 1, 0], dtype=torch.bool).to("npu")
>>> result, mask = torch_npu.npu_random_choice_with_mask(x, 2, 1, 0)
>>> print(result)
tensor([[0],[2]], device='npu:0', dtype=torch.int32)
>>> print(mask)
tensor([True, True], device='npu:0')
```
