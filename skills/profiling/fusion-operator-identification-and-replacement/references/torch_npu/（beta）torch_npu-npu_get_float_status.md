# （beta）torch_npu.npu_get_float_status

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

获取溢出检测结果。

## 函数原型

```python
torch_npu.npu_get_float_status(self) -> Tensor
```

## 参数说明

**self** (`Tensor`)：数据内存地址张量，数据类型为`torch.float32`。

## 返回值说明

`Tensor`

返回检测结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_get_float_status(x)
tensor([0, 0, 0, 0, 0, 0, 0, 0], device='npu:0', dtype=torch.int32)
```
