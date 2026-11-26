# （beta）torch_npu.npu_nms_v4

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

按分数降序选择标注框的子集。

## 函数原型

```python
torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)
```

## 参数说明

- **boxes** (`Tensor`)：必选参数，shape为[num_boxes, 4]的2维浮点张量。
- **scores** (`Tensor`)：必选参数，shape为[num_boxes]的1维浮点张量，表示每个框对应的一个分数。
- **max_output_size** (`Scalar`)：必选参数，表示non-max suppression下要选择的最大框数的标量。
- **iou_threshold** (`Tensor`)：必选参数，0维浮点张量，表示框间IoU阈值上限。
- **scores_threshold** (`Tensor`)：必选参数，0维浮点张量，表示决定何时删除框的分数阈值。
- **pad_to_max_output_size** (`bool`)：可选参数，如果为True，则输出的selected_indices将填充为max_output_size长度。默认值为False。

## 返回值说明

- **selected_indices** (`Tensor`)：shape为[M]的1维整数张量，表示从boxes张量中选定的index，其中M <= max_output_size。
- **valid_outputs**(`Tensor`)：0维整数张量，表示selected_indices中有效元素的数量，有效元素按顺序排序在前面。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> boxes=torch.randn(100,4).npu()
>>> scores=torch.randn(100).npu()
>>> boxes.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> max_output_size = 20
>>> iou_threshold = torch.tensor(0.5).npu()
>>> scores_threshold = torch.tensor(0.3).npu()
>>> npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
>>> print(npu_output)
(tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
        54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
```
