# （beta）torch_npu.npu_grid_assign_positive

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

执行position-sensitive的候选区域池化梯度计算。

## 函数原型

```python
torch_npu.npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，`torch.float16`或`torch.float32`类型的张量，shape为(n, )。
- **overlaps** (`Tensor`)：必选参数，数据类型与`self`相同，表示gt_bboxes和bboxes之间的IoU，shape为(k,n)。
- **box_responsible_flags** (`Tensor`)：必选参数，支持`torch.uint8`数据类型。表示框是否responsible的标志。
- **max_overlaps** (`Tensor`)：必选参数，数据类型与`self`相同。
- **argmax_overlaps** (`Tensor`)：必选参数，支持`torch.int32`数据类型。
- **gt_max_overlaps** (`Tensor`)：必选参数，数据类型与`self`相同。
- **gt_argmax_overlaps** (`Tensor`)：必选参数，支持`torch.int32`数据类型。
- **num_gts** (`Tensor`)：必选参数，支持`torch.int32`数据类型，表示真实GT数量，shape为(1, )。
- **pos_iou_thr** (`float`)：必选参数，正检测框的IoU阈值。
- **min_pos_iou** (`float`)：必选参数，检测框被视为正检测框的最小IoU。
- **gt_max_assign_all** (`bool`)：必选参数，是否将与某个gt有相同最高重叠的所有检测框分配给该gt。

## 返回值说明

`Tensor`

候选区域池化梯度计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> assigned_gt_inds = torch.rand(4).npu()
>>> overlaps = torch.rand(2,4).npu()
>>> box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).npu()
>>> max_overlaps = torch.rand(4).npu()
>>> argmax_overlaps = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
>>> gt_max_overlaps = torch.rand(2).npu()
>>> gt_argmax_overlaps = torch.tensor([1, 0],dtype=torch.int32).npu()
>>> num_gts = torch.tensor([128], dtype=torch.int32).npu()
>>> output = torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, 0.5, 0., True)
>>> print(output.shape)
torch.Size([4])
```
