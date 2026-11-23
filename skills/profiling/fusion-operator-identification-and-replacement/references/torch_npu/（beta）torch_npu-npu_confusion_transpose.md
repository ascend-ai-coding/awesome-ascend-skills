# （beta）torch_npu.npu_confusion_transpose

> [!NOTICE]  
> 该接口计划废弃，可以使用`Tensor.view()`和`torch.permute`接口进行替换。

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

融合reshape和transpose运算。

## 函数原型

```python
torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：数据类型支持`torch.float16`、`torch.float32`、`torch.int8`、`torch.int16`、`torch.int32`、`torch.int64`、`torch.uint8`、`torch.uint16`、`torch.uint32`、`torch.uint64`。
- **perm**（`List[int]`）：`self`张量的维度排列。
- **shape**（`List[int]`）：reshape操作后的目标shape。
- **transpose_first**（`bool`）：如果值为`True`，首先执行transpose，否则先执行reshape。

## 调用示例

```python
>>> x = torch.rand(2, 3, 4, 6).npu()
>>> print(x.shape)
torch.Size([2, 3, 4, 6])
>>> y = torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
>>> print(y.shape)
torch.Size([2, 4, 18])
>>> y2 = torch_npu.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
>>> print(y2.shape)
torch.Size([2, 6, 12])
```
